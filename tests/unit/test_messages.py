"""Wire protocol (section 3.4, IF-4.1 to IF-4.5, NFR-3.1 to NFR-3.3).

The two properties that matter most are that every message fits one ESP-NOW frame
and that a bad frame is discarded *without effect*. The second is easy to get
subtly wrong: a receiver that raises on a corrupt frame can be silenced by noise
alone, so ``decode`` must return None rather than throw.
"""

from __future__ import annotations

import struct

import pytest

from communication import messages as M
from communication.messages import (
    Announce,
    Beacon,
    Bid,
    Blockage,
    Claim,
    Codec,
    Complete,
    DecodeError,
    Digest,
    DigestEntry,
    Intent,
    MessageType,
    Reserve,
    Telemetry,
    crc16,
)
from core import config


@pytest.fixture(params=[8, 16])
def codec(request) -> Codec:
    return Codec(request.param)


def sample_payloads(codec: Codec) -> list:
    """One of every message type, valid at this codec's id width."""
    return [
        Intent(
            current_node=3, next_nodes=(4, 5, 6), eta_ms=(1000, 2200, 3400),
            priority=200, state=3, battery_pct=87, pheromone=40,
        ),
        Reserve(junction_id=7, window_start_ms=10_000, window_end_ms=11_600, priority=100),
        Announce(
            task_id=42, pickup_node=1, drop_node=9, priority=10,
            created_at_ms=5000, zone_id=3,
        ),
        Bid(task_id=42, bid_value=-1234, is_idle=True),
        Claim(task_id=42, bid_value=-1234),
        Digest(entries=tuple(DigestEntry(i, 2000 + i, i) for i in range(5))),
        Telemetry(
            current_node=3, next_node=4, progress_q8=128, battery_pct=87,
            state=3, task_id=42, priority=10,
        ),
        Complete(task_id=42),
        Blockage(edge_id=4, passable=False),
        Beacon(fleet_time_ms=1_234_567),
    ]


class TestFrameBudget:
    """IF-4.1 / CON-2: one ESP-NOW frame, 250 bytes."""

    def test_every_type_fits_at_both_id_widths(self) -> None:
        for bits in (8, 16):
            for name, size in M.packed_sizes(edge_id_bits=bits).items():
                assert size <= config.MAX_FRAME_BYTES, f"{name} is {size} B at {bits}-bit"

    def test_intent_is_small(self) -> None:
        """INTENT is the only message on the continuous safety path, at 5 Hz per
        robot, so its size sets the airtime floor for the whole fleet.

        24 in section 3.4.1; 26 with the held task (defect 3); 28 with the plan
        sequence and released-step index that execution by precedence reads."""
        assert M.packed_sizes(edge_id_bits=8)["INTENT"] == 28

    def test_digest_slice_is_capped_to_fit(self) -> None:
        """CON-2 / FR-2.5: a rotating slice, not the whole table. A full 540-edge
        table would need several kilobytes."""
        codec = Codec(16)
        full = Digest(
            entries=tuple(DigestEntry(i, 2000, 5) for i in range(M.MAX_DIGEST_ENTRIES))
        )
        assert len(codec.encode(1, 1, 0, full)) <= config.MAX_FRAME_BYTES

    def test_an_oversized_digest_is_truncated_not_rejected(self) -> None:
        """Losing the tail of a gossip slice costs nothing -- DIGEST repeats every
        10 s and merges by sample count -- whereas raising would stop a robot
        sharing anything at all."""
        codec = Codec(8)
        huge = Digest(entries=tuple(DigestEntry(i % 200, 2000, 5) for i in range(500)))
        decoded = codec.decode(codec.encode(1, 1, 0, huge))
        assert decoded is not None
        assert len(decoded.payload.entries) == M.MAX_DIGEST_ENTRIES

    def test_the_frame_limit_is_enforced_at_encode_time(self, monkeypatch) -> None:
        """No real payload can exceed 250 bytes, which is the point -- every type is
        fixed-size or capped. The guard in ``encode`` is a safety net for a future
        message type, so it is verified by lowering the limit rather than by
        inventing a payload that cannot otherwise exist. Silently sending a
        truncated safety message would be far worse than failing loudly.
        """
        monkeypatch.setattr(config, "MAX_FRAME_BYTES", 12)
        with pytest.raises(ValueError, match="IF-4.1"):
            Codec(8).encode(1, 1, 0, Intent(1, (2, 3, 4), (10, 20, 30), 1, 1, 100, 0))


class TestRoundTrip:
    def test_every_payload_survives_a_round_trip(self, codec: Codec) -> None:
        for payload in sample_payloads(codec):
            frame = codec.encode(sender=7, seq=1234, timestamp_ms=98_765, payload=payload)
            decoded = codec.decode(frame)
            assert decoded is not None, f"{type(payload).__name__} failed to decode"
            assert decoded.sender == 7
            assert decoded.seq == 1234
            assert decoded.timestamp_ms == 98_765
            assert decoded.type is payload.TYPE

    def test_intent_fields_survive_exactly(self, codec: Codec) -> None:
        original = Intent(
            current_node=3, next_nodes=(4, 5, 6), eta_ms=(1000, 2200, 3400),
            priority=200, state=3, battery_pct=87, pheromone=40,
        )
        decoded = codec.decode(codec.encode(1, 1, 0, original)).payload
        assert decoded == original

    def test_a_short_intent_route_round_trips(self, codec: Codec) -> None:
        """Near the end of a route there are fewer than three junctions left.
        Padding must not reappear as a phantom node the reservation table then
        treats as a real future position."""
        original = Intent(
            current_node=3, next_nodes=(4,), eta_ms=(1000,),
            priority=1, state=3, battery_pct=100, pheromone=0,
        )
        decoded = codec.decode(codec.encode(1, 1, 0, original)).payload
        assert decoded.next_nodes == (4,)
        assert decoded.eta_ms == (1000,)

    def test_a_negative_bid_survives(self, codec: Codec) -> None:
        """The aging and idle terms subtract without bound, so a long-waiting task
        genuinely prices below zero. Clamping at zero would flatten the ordering
        that discharges FR-4.11's anti-starvation guarantee."""
        decoded = codec.decode(codec.encode(1, 1, 0, Bid(9, -50_000, False))).payload
        assert decoded.bid_value == -50_000

    def test_an_enormous_bid_saturates_rather_than_overflowing(self, codec: Codec) -> None:
        decoded = codec.decode(codec.encode(1, 1, 0, Bid(9, 2**40, False))).payload
        assert decoded.bid_value == 2**31 - 1

    def test_unzoned_tasks_round_trip_as_unzoned(self, codec: Codec) -> None:
        original = Announce(1, 2, 3, 10, 0, zone_id=-1)
        assert codec.decode(codec.encode(1, 1, 0, original)).payload.zone_id == -1

    def test_no_next_node_round_trips_as_absent(self, codec: Codec) -> None:
        original = Telemetry(3, -1, 0, 100, 1, -1, 0)
        decoded = codec.decode(codec.encode(1, 1, 0, original)).payload
        assert decoded.next_node == -1
        assert decoded.task_id == -1

    def test_sequence_numbers_wrap_rather_than_overflow(self, codec: Codec) -> None:
        frame = codec.encode(1, 0x1_0001, 0, Complete(1))
        assert codec.decode(frame).seq == 1


class TestIntegrity:
    """IF-4.3, NFR-3.1: a corrupt frame is discarded without effect."""

    def test_a_flipped_bit_is_caught(self, codec: Codec) -> None:
        frame = bytearray(codec.encode(1, 1, 0, Complete(42)))
        frame[HEADER := 9] ^= 0x01
        assert codec.decode(bytes(frame)) is None

    def test_every_single_byte_corruption_is_caught(self) -> None:
        """Exhaustive over one frame: CRC-16 must catch any single-byte change."""
        codec = Codec(8)
        good = codec.encode(
            3, 77, 12_345,
            Intent(1, (2, 3, 4), (100, 200, 300), 10, 3, 90, 5),
        )
        for index in range(len(good)):
            for mask in (0x01, 0x80, 0xFF):
                corrupted = bytearray(good)
                corrupted[index] ^= mask
                assert codec.decode(bytes(corrupted)) is None, (
                    f"corruption at byte {index} with mask {mask:#x} went undetected"
                )

    def test_decode_returns_none_rather_than_raising(self, codec: Codec) -> None:
        """A receiver that threw on a corrupt frame could be silenced by noise."""
        for rubbish in (b"", b"\x01", b"\xff" * 40, b"nonsense frame bytes here!!"):
            assert codec.decode(rubbish) is None

    def test_decode_strict_explains_itself(self, codec: Codec) -> None:
        with pytest.raises(DecodeError, match="too short"):
            codec.decode_strict(b"\x01\x02")
        with pytest.raises(DecodeError, match="CRC"):
            codec.decode_strict(codec.encode(1, 1, 0, Complete(1))[:-2] + b"\x00\x00")

    def test_an_unknown_message_type_is_discarded(self, codec: Codec) -> None:
        body = struct.pack(M.HEADER, 200, 1, 1, 0)
        frame = body + struct.pack("<H", crc16(body))
        assert codec.decode(frame) is None

    def test_an_oversized_frame_is_rejected(self, codec: Codec) -> None:
        body = b"\x01" * (config.MAX_FRAME_BYTES + 4)
        assert codec.decode(body + struct.pack("<H", crc16(body))) is None

    def test_crc_is_not_trivially_zero(self) -> None:
        assert crc16(b"") != 0
        assert crc16(b"a") != crc16(b"b")


class TestRangeValidation:
    """NFR-3.3: field values outside the valid ranges are discarded."""

    def test_an_impossible_battery_level_is_discarded(self) -> None:
        """Built from struct rather than by patching a byte offset, so the test does
        not silently stop testing anything when the layout changes."""
        codec = Codec(8)
        body = struct.pack(M.HEADER, int(MessageType.INTENT), 1, 1, 0) + struct.pack(
            "<BBBBHHHBBBBH",
            1, 2, 3, 4,        # current_node, next_nodes[3]
            100, 200, 300,     # eta_ms[3]
            10, 3,             # priority, state
            200,               # battery_pct -- impossible
            5,                 # pheromone
            0xFFFF,            # held_task_id: none
        )
        assert codec.decode(body + struct.pack("<H", crc16(body))) is None

    def test_a_valid_battery_level_is_accepted(self) -> None:
        """The companion: the frame above must be good apart from the one field."""
        codec = Codec(8)
        body = struct.pack(M.HEADER, int(MessageType.INTENT), 1, 1, 0) + struct.pack(
            "<BBBBHHHBBBBHBB", 1, 2, 3, 4, 100, 200, 300, 10, 3, 90, 5, 0xFFFF, 0, 0
        )
        assert codec.decode(body + struct.pack("<H", crc16(body))) is not None

    def test_a_backwards_reservation_window_is_discarded(self) -> None:
        """A window ending before it starts would make overlap arithmetic
        nonsensical and could let two robots both believe they were clear."""
        codec = Codec(8)
        body = struct.pack(M.HEADER, int(MessageType.RESERVE), 1, 1, 0) + struct.pack(
            "<BIIB", 5, 9000, 1000, 10
        )
        assert codec.decode(body + struct.pack("<H", crc16(body))) is None

    def test_an_announce_to_nowhere_is_discarded(self) -> None:
        codec = Codec(8)
        body = struct.pack(M.HEADER, int(MessageType.ANNOUNCE), 0, 1, 0) + struct.pack(
            "<HBBBIB", 1, 4, 4, 10, 0, 0xFF
        )
        assert codec.decode(body + struct.pack("<H", crc16(body))) is None

    def test_a_digest_claiming_more_entries_than_it_carries_is_discarded(self) -> None:
        codec = Codec(8)
        body = struct.pack(M.HEADER, int(MessageType.DIGEST), 1, 1, 0) + b"\x10\x01"
        assert codec.decode(body + struct.pack("<H", crc16(body))) is None


class TestCodecIdentity:
    def test_an_invalid_id_width_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="8 or 16"):
            Codec(12)

    def test_the_two_widths_produce_different_layouts(self) -> None:
        """Which is why the width lives on the codec: decoding with the wrong one
        would not raise, it would silently produce plausible garbage."""
        narrow, wide = Codec(8), Codec(16)
        payload = Intent(1, (2, 3, 4), (10, 20, 30), 1, 1, 100, 0)
        assert len(narrow.encode(1, 1, 0, payload)) != len(wide.encode(1, 1, 0, payload))


class TestClassification:
    def test_safety_critical_set_matches_section_3_4_2(self) -> None:
        """Section 3.4.2's five, plus PATH. PATH is the route booking that keeps
        robots apart under route-level reservation, so it inherits every rule the
        section applies to RESERVE: no acknowledgement, no retransmission on demand,
        no connection (IF-4.5). It heals by repetition, as INTENT does."""
        assert M.SAFETY_CRITICAL == {
            MessageType.INTENT, MessageType.RESERVE, MessageType.ANNOUNCE,
            MessageType.BID, MessageType.CLAIM, MessageType.PATH,
        }

    def test_telemetry_and_digest_are_not_safety_critical(self) -> None:
        """Section 3.4.2 marks both No, which is what lets the dashboard be
        entirely absent without affecting the fleet (FR-8.5)."""
        assert MessageType.TELEMETRY not in M.SAFETY_CRITICAL
        assert MessageType.DIGEST not in M.SAFETY_CRITICAL

    def test_message_reports_its_own_criticality(self) -> None:
        codec = Codec(8)
        intent = codec.decode(codec.encode(1, 1, 0, Intent(1, (2,), (5,), 1, 1, 100, 0)))
        telemetry = codec.decode(codec.encode(1, 2, 0, Telemetry(1, 2, 0, 100, 1, 1, 0)))
        assert intent.is_safety_critical
        assert not telemetry.is_safety_critical


class TestPath:
    """The route booking on the wire."""

    def test_round_trips_including_a_past_offset(self) -> None:
        """A mid-edge replan names the region it entered a moment ago, so offsets
        must be signed."""
        for bits in (8, 16):
            codec = Codec(bits)
            path = M.Path(
                plan_seq=3, committed_ms=123_456, priority=100,
                entries=((5, -4000), (6, 0), (7, 12_340), (8, 400_000)), goal_task_id=42,
            )
            decoded = codec.decode(codec.encode(1, 1, 1000, path))
            assert decoded is not None
            assert decoded.payload == path

    def test_offsets_are_quantised_to_the_tick(self) -> None:
        codec = Codec(8)
        path = M.Path(plan_seq=1, committed_ms=0, priority=1, entries=((1, 1234),))
        decoded = codec.decode(codec.encode(1, 1, 0, path)).payload
        assert decoded.entries == ((1, 1220),)  # 1234 // 20 * 20

    def test_far_future_offsets_saturate_rather_than_wrap(self) -> None:
        """A wrapped offset would place a step in the past and free a resource that is
        still booked. Saturating keeps it in the future, merely imprecise."""
        codec = Codec(8)
        path = M.Path(plan_seq=1, committed_ms=0, priority=1, entries=((1, 10_000_000),))
        decoded = codec.decode(codec.encode(1, 1, 0, path)).payload
        assert decoded.entries[0][1] == M.PATH_OFFSET_MAX * M.PATH_TICK_MS

    @pytest.mark.parametrize("bits", [8, 16])
    def test_the_longest_route_on_every_map_fits_one_frame(self, bits: int) -> None:
        """Every plan must go out whole: a plan that needs two frames could have its
        second half lost and leave peers with a route that ends mid-aisle."""
        import itertools

        from core.graph import Graph
        from core.planner_astar import AStarPlanner
        from tests.conftest import ALL_MAP_NAMES, MAPS_DIR

        limit = M.max_path_nodes(bits)
        for name in ALL_MAP_NAMES:
            graph = Graph.load(MAPS_DIR / f"{name}.json")
            if bits == 8 and len(graph.edges) > 255:
                continue  # this map is only ever run at 16 bits
            planner = AStarPlanner(graph)
            stops = sorted(graph.task_endpoints | set(graph.parking_nodes))
            longest = 0
            for a, b in itertools.combinations(stops, 2):
                route = planner.route(a, b)
                if route is not None:
                    longest = max(longest, len(route))
            assert longest <= limit, (
                f"{name}: a {longest}-node route does not fit a {limit}-node PATH at "
                f"{bits}-bit ids"
            )

    def test_an_overlong_path_is_refused_at_encode(self) -> None:
        codec = Codec(16)
        too_many = tuple((i, i * 20) for i in range(M.max_path_nodes(16) + 1))
        with pytest.raises(ValueError):
            codec.encode(1, 1, 0, M.Path(plan_seq=1, committed_ms=0, priority=1, entries=too_many))

    def test_a_truncated_path_is_discarded(self) -> None:
        codec = Codec(8)
        frame = codec.encode(1, 1, 0, M.Path(plan_seq=1, committed_ms=0, priority=1, entries=((1, 0), (2, 20))))
        body = frame[:-2]
        cut = body[:-2]  # drop one entry's worth
        assert codec.decode(cut + struct.pack("<H", crc16(cut))) is None
