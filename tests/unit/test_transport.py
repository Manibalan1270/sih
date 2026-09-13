"""Transport backends (FR-1.3, FR-1.6, FR-1.7, FR-9.4, IF-4.2, IF-4.5).

The in-process bus is the one to scrutinise. It has to give reproducible runs
while still being a genuine message channel: if it ever let a robot observe a peer
directly it would quietly make the simulation centralised, and every behavioural
test would still pass.
"""

from __future__ import annotations

import pytest

from communication.messages import Codec, Complete, Intent, MessageType
from communication.transport import (
    InProcBus,
    ReceiveOnly,
    ReceiveOnlyError,
    SequenceFilter,
)
from core import config


def frame(codec: Codec, sender: int, seq: int = 1) -> bytes:
    return codec.encode(sender, seq, 0, Complete(task_id=seq))


@pytest.fixture
def bus() -> InProcBus:
    positions = {1: (0, 0), 2: (1000, 0), 3: (2000, 0)}
    made = InProcBus(seed=1, range_mm=None, position_of=positions.get)
    for robot_id in (1, 2, 3):
        made.join(robot_id)
    return made


class TestBroadcast:
    def test_a_broadcast_reaches_every_other_member(self, bus: InProcBus) -> None:
        codec = Codec(8)
        bus.broadcast(1, frame(codec, 1))
        bus.deliver()
        assert len(bus.receive(2)) == 1
        assert len(bus.receive(3)) == 1

    def test_a_robot_does_not_hear_its_own_broadcast(self, bus: InProcBus) -> None:
        codec = Codec(8)
        bus.broadcast(1, frame(codec, 1))
        bus.deliver()
        assert bus.receive(1) == []

    def test_frames_are_delivered_on_the_next_tick_not_immediately(
        self, bus: InProcBus
    ) -> None:
        """A robot must not hear an INTENT in the same tick it was formed -- no real
        radio could do that, and allowing it would let tick ordering leak
        information between robots."""
        codec = Codec(8)
        bus.broadcast(1, frame(codec, 1))
        assert bus.receive(2) == [], "delivered without a deliver() call"
        bus.deliver()
        assert len(bus.receive(2)) == 1

    def test_receiving_clears_the_inbox(self, bus: InProcBus) -> None:
        codec = Codec(8)
        bus.broadcast(1, frame(codec, 1))
        bus.deliver()
        assert len(bus.receive(2)) == 1
        assert bus.receive(2) == []

    def test_a_member_who_left_hears_nothing(self, bus: InProcBus) -> None:
        """TC-5: a robot that has lost power cannot announce it, so peers notice
        only through silence."""
        codec = Codec(8)
        bus.leave(2)
        bus.broadcast(1, frame(codec, 1))
        bus.deliver()
        assert bus.receive(2) == []
        assert len(bus.receive(3)) == 1

    def test_a_member_who_left_is_not_heard(self, bus: InProcBus) -> None:
        codec = Codec(8)
        bus.leave(2)
        bus.broadcast(2, frame(codec, 2))
        bus.deliver()
        assert bus.receive(1) == []


class TestRadioRange:
    """FR-9.4: INTENT and RESERVE scope is the physical neighbourhood."""

    def test_a_distant_robot_is_out_of_range(self) -> None:
        positions = {1: (0, 0), 2: (config.RADIO_RANGE_MM * 3, 0)}
        bus = InProcBus(seed=1, position_of=positions.get)
        bus.join(1)
        bus.join(2)
        bus.broadcast(1, frame(Codec(8), 1))
        bus.deliver()
        assert bus.receive(2) == []
        assert bus.stats.dropped_range == 1

    def test_a_nearby_robot_is_in_range(self) -> None:
        positions = {1: (0, 0), 2: (config.RADIO_RANGE_MM // 2, 0)}
        bus = InProcBus(seed=1, position_of=positions.get)
        bus.join(1)
        bus.join(2)
        bus.broadcast(1, frame(Codec(8), 1))
        bus.deliver()
        assert len(bus.receive(2)) == 1

    def test_range_covers_anything_that_could_collide(self) -> None:
        """ASM-7 is the SRS's own "single most critical assumption": any two AMRs
        that can physically collide must be in mutual radio range. If the modelled
        range were ever below the collision distance the safety guarantee would be
        void by construction."""
        assert config.RADIO_RANGE_MM > config.COLLISION_DISTANCE_MM * 10

    def test_wired_members_ignore_range(self) -> None:
        """The gateway is infrastructure, not a robot with a position. Without the
        exemption its ANNOUNCE would be range-filtered against a position it does
        not have, and no robot would ever hear about a task -- which looks like a
        broken auction rather than a broken radio model."""
        positions = {1: (0, 0)}
        bus = InProcBus(seed=1, position_of=positions.get, wired={0})
        bus.join(0)
        bus.join(1)
        bus.broadcast(0, frame(Codec(8), 0))
        bus.deliver()
        assert len(bus.receive(1)) == 1

    def test_range_without_positions_is_rejected(self) -> None:
        """Otherwise the bus would silently flood instead of filtering."""
        with pytest.raises(ValueError, match="FR-9.4"):
            InProcBus(seed=1, range_mm=10_000, position_of=None)

    def test_a_robot_with_no_position_is_unreachable(self) -> None:
        bus = InProcBus(seed=1, position_of=lambda _: None)
        bus.join(1)
        bus.join(2)
        bus.broadcast(1, frame(Codec(8), 1))
        bus.deliver()
        assert bus.receive(2) == []


class TestPacketLoss:
    """ASM-8, TC-28: loss is expected, not exceptional."""

    def test_loss_is_applied(self) -> None:
        bus = InProcBus(seed=7, loss_permille=500, range_mm=None)
        bus.join(1)
        bus.join(2)
        codec = Codec(8)
        for seq in range(1, 201):
            bus.broadcast(1, frame(codec, 1, seq))
        bus.deliver()
        received = len(bus.receive(2))
        assert 60 < received < 140, f"{received}/200 delivered at 50% loss"
        assert bus.stats.dropped_loss == 200 - received

    def test_loss_is_deterministic_for_a_seed(self) -> None:
        """Otherwise two runs of one seed would drop different frames and NFR-4.4
        would fail intermittently -- the worst way for it to fail."""
        counts = []
        for _ in range(3):
            bus = InProcBus(seed=99, loss_permille=300, range_mm=None)
            bus.join(1)
            bus.join(2)
            codec = Codec(8)
            for seq in range(1, 101):
                bus.broadcast(1, frame(codec, 1, seq))
            bus.deliver()
            counts.append(len(bus.receive(2)))
        assert counts[0] == counts[1] == counts[2]

    def test_zero_loss_delivers_everything(self, bus: InProcBus) -> None:
        codec = Codec(8)
        for seq in range(1, 51):
            bus.broadcast(1, frame(codec, 1, seq))
        bus.deliver()
        assert len(bus.receive(2)) == 50
        assert bus.stats.dropped_loss == 0


class TestStats:
    def test_frames_are_counted_by_type(self, bus: InProcBus) -> None:
        codec = Codec(8)
        bus.broadcast(1, codec.encode(1, 1, 0, Complete(1)))
        bus.broadcast(1, codec.encode(1, 2, 0, Intent(1, (2,), (5,), 1, 1, 100, 0)))
        assert bus.stats.sent["COMPLETE"] == 1
        assert bus.stats.sent["INTENT"] == 1
        assert bus.stats.total_sent == 2

    def test_auction_frames_are_countable(self) -> None:
        """Section 4.9 measures auction cost in BID and CLAIM frames, which is how
        the FR-9.6 and NFR-1.12 budgets are checked."""
        bus = InProcBus(seed=1, range_mm=None)
        bus.stats.sent["BID"] = 12
        bus.stats.sent["CLAIM"] = 4
        bus.stats.sent["INTENT"] = 900
        assert bus.stats.auction_frames() == 16

    def test_summary_mentions_losses(self, bus: InProcBus) -> None:
        assert "dropped" in bus.stats.summary()


class TestReceiveOnly:
    """FR-8.4 / IF-1.6 / BR-7: the dashboard cannot command a robot."""

    def test_receiving_works(self, bus: InProcBus) -> None:
        codec = Codec(8)
        bus.broadcast(1, frame(codec, 1))
        bus.deliver()
        assert len(ReceiveOnly(bus).receive(2)) == 1

    def test_broadcasting_is_impossible(self, bus: InProcBus) -> None:
        """Structural, not a UI rule: there is no working send path, so no future
        edit can add a command channel by accident."""
        with pytest.raises(ReceiveOnlyError, match="FR-8.4"):
            ReceiveOnly(bus).broadcast(2, b"\x01")

    def test_the_wrapped_bus_is_untouched_by_a_refused_send(self, bus: InProcBus) -> None:
        with pytest.raises(ReceiveOnlyError):
            ReceiveOnly(bus).broadcast(2, b"\x01")
        assert bus.stats.total_sent == 0


class TestSequenceFilter:
    """FR-1.6, NFR-3.2: reject replayed and out-of-order frames."""

    def test_the_first_frame_from_a_sender_is_accepted(self) -> None:
        assert SequenceFilter().accept(sender=3, seq=500)

    def test_increasing_sequences_are_accepted(self) -> None:
        sequence_filter = SequenceFilter()
        assert all(sequence_filter.accept(3, seq) for seq in (1, 2, 3, 10, 500))

    def test_a_replay_is_rejected(self) -> None:
        sequence_filter = SequenceFilter()
        sequence_filter.accept(3, 100)
        assert not sequence_filter.accept(3, 100)
        assert not sequence_filter.accept(3, 99)
        assert sequence_filter.rejected == 2

    def test_senders_are_tracked_independently(self) -> None:
        sequence_filter = SequenceFilter()
        sequence_filter.accept(1, 900)
        assert sequence_filter.accept(2, 5), "one sender's sequence blocked another's"

    def test_wraparound_is_handled(self) -> None:
        """Sequence numbers are uint16 and a naive seq > last would reject every
        frame for the rest of the run after the first wrap, silently turning a
        working fleet into a set of deaf robots. At 5 Hz per robot the wrap arrives
        in about three and a half hours -- well inside a shift."""
        sequence_filter = SequenceFilter()
        assert sequence_filter.accept(1, 65_534)
        assert sequence_filter.accept(1, 65_535)
        assert sequence_filter.accept(1, 0), "wrap to zero was rejected as a replay"
        assert sequence_filter.accept(1, 1)

    def test_a_replay_across_the_wrap_is_still_rejected(self) -> None:
        sequence_filter = SequenceFilter()
        sequence_filter.accept(1, 3)
        assert not sequence_filter.accept(1, 65_000), (
            "a frame from far behind should read as older, not as a huge jump ahead"
        )

    def test_forgetting_a_peer_lets_a_reboot_back_in(self) -> None:
        """FR-1.5 / FR-6.5: a robot that rebooted starts its sequence again, and
        must not be mistaken for a replay attacker."""
        sequence_filter = SequenceFilter()
        sequence_filter.accept(1, 40_000)
        sequence_filter.forget(1)
        assert sequence_filter.accept(1, 1)

    def test_last_seen_is_reported(self) -> None:
        sequence_filter = SequenceFilter()
        sequence_filter.accept(1, 77)
        assert sequence_filter.last_seen(1) == 77
        assert sequence_filter.last_seen(2) is None
