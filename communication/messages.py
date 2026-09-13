"""The wire protocol (section 3.4).

Framing is binary, not JSON, and that is a requirement rather than a preference.
IF-4.1 and CON-2 cap a coordination message at one ESP-NOW frame of 250 bytes. An
INTENT expressed as JSON runs to roughly 180 bytes of mostly field names, and a
DIGEST carrying a useful slice of learned edge times would not fit at all. Packed
with ``struct`` an INTENT is 24 bytes, which leaves the airtime budget the fleet
actually needs at 5 Hz per robot.

Every frame carries a sequence number and a timestamp (IF-4.4) and a CRC
(IF-4.3), so replayed, reordered and corrupted frames can all be rejected. A
malformed frame is *discarded without effect* (NFR-3.1, NFR-3.3): ``decode``
returns None rather than raising, because a corrupt frame on a shared radio is an
expected event, not an exceptional one, and a receiver that threw on one would be
trivially deniable by noise.

Identifier width is configurable. ASM-2 and CON-10 make node and edge ids 8-bit,
which is what the ESP32 memory and airtime argument in section 2.5 rests on; the
scale100 scenario needs 16-bit ids for a 288-node map, which ASM-2 itself
anticipates ("packet field widths and memory budget must be revised").

Three message types here are not in the section 3.4 table but are required
elsewhere in the SRS, and are noted as a specification gap in README.md:
COMPLETE (Appendix A: AT_DROP broadcasts it), BLOCKAGE (FR-6.1: "broadcast the
blockage so that peers avoid the edge") and BEACON (FR-7.4: "a clock beacon shall
be broadcast every 5 s").
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

from core import config

# ---------------------------------------------------------------------------
# CRC-16/CCITT-FALSE (IF-4.3, NFR-3.1)
# ---------------------------------------------------------------------------

_CRC_POLY = 0x1021


def _build_crc_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ _CRC_POLY) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC_TABLE = _build_crc_table()


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE over ``data``."""
    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc


class MessageType(IntEnum):
    """One byte on the wire. Values are frozen: changing one breaks the fleet."""

    INTENT = 1
    RESERVE = 2
    ANNOUNCE = 3
    BID = 4
    CLAIM = 5
    DIGEST = 6
    TELEMETRY = 7
    COMPLETE = 8
    BLOCKAGE = 9
    BEACON = 10


SAFETY_CRITICAL: frozenset[MessageType] = frozenset(
    {
        MessageType.INTENT,
        MessageType.RESERVE,
        MessageType.ANNOUNCE,
        MessageType.BID,
        MessageType.CLAIM,
    }
)
"""Section 3.4.2's safety-critical column. IF-4.5 forbids these from requiring
acknowledgement, retransmission or an established connection."""


class DecodeError(ValueError):
    """Raised only by ``decode_strict``; ``decode`` returns None instead."""


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Intent:
    """Section 3.4.1. Broadcast every 200 ms by every robot (FR-1.1).

    The only message on the continuous safety-critical path. A single received
    INTENT updates both the reservation table and the traffic model, with no
    second message needed (FR-1.8) -- which is why the pheromone deposit rides
    along in the same frame.
    """

    TYPE: ClassVar[MessageType] = MessageType.INTENT

    current_node: int
    next_nodes: tuple[int, ...]
    """FR-1.2: the next three junctions the sender intends to cross. Padded with
    ``NO_NODE`` when the remaining route is shorter."""

    eta_ms: tuple[int, ...]
    """Arrival at each of those junctions, relative to ``timestamp``.

    Relative rather than absolute: the field is a uint16, so an absolute aligned
    timestamp would overflow after 65 seconds of operation. Offsets from the
    frame's own 32-bit timestamp keep the range useful and make the arithmetic
    robust to a clock correction between send and receive."""

    priority: int
    state: int
    battery_pct: int
    pheromone: int
    """Congestion deposit for the edge just traversed (FE-2)."""

    held_task_id: int = -1
    """The task this robot is currently working, or -1 if none.

    **Not in section 3.4.1, and added deliberately.** FR-4.8 requires that where two
    AMRs claim the same task, the lower robot_id retains it and the other
    relinquishes within one auction cycle. CLAIM is a one-shot message, and IF-4.5
    forbids requiring acknowledgement or retransmission on the safety path, so a
    robot whose peer's CLAIM was lost has no way to discover the collision -- it
    simply keeps the task and does the work twice.

    Measured at 30% packet loss on bench3 before this field existed: 13 completions
    for 9 tasks, and one task held by two robots simultaneously.

    Carrying the held task in INTENT fixes it using the mechanism the SRS already
    relies on everywhere else -- section 6 of the implementation guide puts it
    plainly: "INTENT repeats every 200 ms, so a dropped packet is self-healing".
    Duplicate holding now resolves within one INTENT period instead of never. Costs
    two bytes; INTENT goes from 24 to 26, against a 250-byte budget."""


@dataclass(frozen=True, slots=True)
class Reserve:
    """Section 3.4.2. Broadcast before entering a junction (FR-5.5)."""

    TYPE: ClassVar[MessageType] = MessageType.RESERVE

    junction_id: int
    window_start_ms: int
    window_end_ms: int
    priority: int


@dataclass(frozen=True, slots=True)
class Announce:
    """Section 3.4.2. Gateway to mesh, per task (FR-4.1).

    Carries no robot identity by design: FR-4.1 forbids any component assigning a
    task to a named robot.
    """

    TYPE: ClassVar[MessageType] = MessageType.ANNOUNCE

    task_id: int
    pickup_node: int
    drop_node: int
    priority: int
    created_at_ms: int
    zone_id: int
    """FR-9.2. ``NO_ZONE`` (0xFF) for an unzoned fleet."""


@dataclass(frozen=True, slots=True)
class Bid:
    """Section 3.4.2. A robot's own price for a task (FR-4.2, FR-4.3)."""

    TYPE: ClassVar[MessageType] = MessageType.BID

    task_id: int
    bid_value: int
    """Insertion cost in milliseconds, adjusted by the battery, aging and idle
    terms. Signed on the wire: the aging and idle terms subtract, so a long-waiting
    task genuinely can price below zero, and clamping at zero would destroy the
    ordering that makes FR-4.11's anti-starvation guarantee work."""

    is_idle: bool


@dataclass(frozen=True, slots=True)
class Claim:
    """Section 3.4.2. Asserts the win (FR-4.7)."""

    TYPE: ClassVar[MessageType] = MessageType.CLAIM

    task_id: int
    bid_value: int
    """The winner's own bid, carried so a challenger can apply FR-4.10's stability
    margin without having to have heard the original auction."""


@dataclass(frozen=True, slots=True)
class DigestEntry:
    edge_id: int
    ewma_ms: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class Digest:
    """Section 3.4.2. Gossip of learned edge times every 10 s (FR-2.5).

    A rotating slice rather than the whole table, because CON-2 caps the frame at
    250 bytes and a full 540-edge table would need several kilobytes.
    """

    TYPE: ClassVar[MessageType] = MessageType.DIGEST

    entries: tuple[DigestEntry, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Telemetry:
    """Section 3.4.2. Mesh to dashboard at 1 Hz. Never safety-critical.

    ``progress_q8`` is progress along the current edge as Q8 fixed point, so the
    frontend can animate smoothly between junctions without needing raw simulator
    coordinates (section 8 of the implementation guide).
    """

    TYPE: ClassVar[MessageType] = MessageType.TELEMETRY

    current_node: int
    next_node: int
    progress_q8: int
    battery_pct: int
    state: int
    task_id: int
    priority: int


@dataclass(frozen=True, slots=True)
class Complete:
    """Appendix A: broadcast on reaching the drop."""

    TYPE: ClassVar[MessageType] = MessageType.COMPLETE

    task_id: int


@dataclass(frozen=True, slots=True)
class Blockage:
    """FR-6.1: broadcast so peers avoid the edge."""

    TYPE: ClassVar[MessageType] = MessageType.BLOCKAGE

    edge_id: int
    passable: bool = False
    """False marks the edge impassable; True retracts a blockage that has cleared.
    Without the retraction an aisle blocked once stays avoided for the rest of the
    shift, and a pallet that was moved would permanently cost the fleet a route."""


@dataclass(frozen=True, slots=True)
class Beacon:
    """FR-7.4: clock beacon every 5 s. Any AMR or the gateway may originate it."""

    TYPE: ClassVar[MessageType] = MessageType.BEACON

    fleet_time_ms: int


Payload = (
    Intent | Reserve | Announce | Bid | Claim | Digest | Telemetry | Complete
    | Blockage | Beacon
)


@dataclass(frozen=True, slots=True)
class Message:
    """A decoded frame: envelope plus payload."""

    sender: int
    seq: int
    timestamp_ms: int
    payload: Payload

    @property
    def type(self) -> MessageType:
        return self.payload.TYPE

    @property
    def is_safety_critical(self) -> bool:
        return self.type in SAFETY_CRITICAL

    def __str__(self) -> str:
        return f"{self.type.name} from r{self.sender} seq={self.seq} @{self.timestamp_ms}"


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

NO_NODE_8 = 0xFF
NO_NODE_16 = 0xFFFF
NO_ZONE_WIRE = 0xFF
NO_TASK = 0xFFFF

HEADER = "<BBHI"
"""type, sender, seq, timestamp. 8 bytes."""
HEADER_SIZE = struct.calcsize(HEADER)
CRC_SIZE = 2

MAX_DIGEST_ENTRIES = 24
"""Entries per DIGEST slice. Sized so the frame stays inside CON-2's 250 bytes at
16-bit ids: 24 * 5 + 10 = 130 bytes, leaving room for the ESP-NOW header the
simulation does not model."""


class Codec:
    """Encodes and decodes frames for one identifier width.

    One instance per scenario. Holding the width here rather than passing it to
    every call means a frame cannot accidentally be decoded with a different
    layout than it was encoded with -- which would not raise, it would silently
    produce plausible garbage.
    """

    def __init__(self, id_bits: int = config.EDGE_ID_BITS_DEFAULT) -> None:
        if id_bits not in (8, 16):
            raise ValueError(f"id_bits must be 8 or 16, got {id_bits}")
        self.id_bits = id_bits
        self._id = "B" if id_bits == 8 else "H"
        self.no_node = NO_NODE_8 if id_bits == 8 else NO_NODE_16
        self._formats = self._build_formats()

    def _build_formats(self) -> dict[MessageType, str]:
        node = self._id
        return {
            MessageType.INTENT: f"<{node}{node * 3}{'H' * 3}BBBBH",
            MessageType.RESERVE: f"<{node}IIB",
            MessageType.ANNOUNCE: f"<H{node}{node}BIB",
            MessageType.BID: "<HiB",
            MessageType.CLAIM: "<Hi",
            MessageType.TELEMETRY: f"<{node}{node}HBBHB",
            MessageType.COMPLETE: "<H",
            MessageType.BLOCKAGE: f"<{node}B",
            MessageType.BEACON: "<I",
        }

    # -- sizes ---------------------------------------------------------------

    def frame_size(self, message_type: MessageType, *, digest_entries: int = MAX_DIGEST_ENTRIES) -> int:
        if message_type is MessageType.DIGEST:
            entry = struct.calcsize(f"<{self._id}HB")
            return HEADER_SIZE + 1 + digest_entries * entry + CRC_SIZE
        return HEADER_SIZE + struct.calcsize(self._formats[message_type]) + CRC_SIZE

    # -- encoding ------------------------------------------------------------

    def encode(self, sender: int, seq: int, timestamp_ms: int, payload: Payload) -> bytes:
        body = self._encode_payload(payload)
        header = struct.pack(
            HEADER, int(payload.TYPE), sender & 0xFF, seq & 0xFFFF, timestamp_ms & 0xFFFFFFFF
        )
        frame = header + body
        frame += struct.pack("<H", crc16(frame))
        if len(frame) > config.MAX_FRAME_BYTES:
            raise ValueError(
                f"{payload.TYPE.name} encodes to {len(frame)} bytes, over IF-4.1's "
                f"{config.MAX_FRAME_BYTES}-byte ESP-NOW limit"
            )
        return frame

    def _encode_payload(self, payload: Payload) -> bytes:
        if isinstance(payload, Intent):
            nodes = list(payload.next_nodes[:3]) + [self.no_node] * (3 - len(payload.next_nodes))
            etas = list(payload.eta_ms[:3]) + [0] * (3 - len(payload.eta_ms))
            return struct.pack(
                self._formats[MessageType.INTENT],
                payload.current_node, *nodes, *[min(e, 0xFFFF) for e in etas],
                payload.priority, payload.state, payload.battery_pct, payload.pheromone,
                NO_TASK if payload.held_task_id < 0 else payload.held_task_id,
            )
        if isinstance(payload, Reserve):
            return struct.pack(
                self._formats[MessageType.RESERVE],
                payload.junction_id, payload.window_start_ms, payload.window_end_ms,
                payload.priority,
            )
        if isinstance(payload, Announce):
            return struct.pack(
                self._formats[MessageType.ANNOUNCE],
                payload.task_id, payload.pickup_node, payload.drop_node,
                payload.priority, payload.created_at_ms,
                NO_ZONE_WIRE if payload.zone_id < 0 else payload.zone_id,
            )
        if isinstance(payload, Bid):
            return struct.pack(
                self._formats[MessageType.BID],
                payload.task_id, _clamp_i32(payload.bid_value), int(payload.is_idle)
            )
        if isinstance(payload, Claim):
            return struct.pack(
                self._formats[MessageType.CLAIM],
                payload.task_id, _clamp_i32(payload.bid_value)
            )
        if isinstance(payload, Digest):
            entries = payload.entries[:MAX_DIGEST_ENTRIES]
            out = struct.pack("<B", len(entries))
            entry_format = f"<{self._id}HB"
            for item in entries:
                out += struct.pack(
                    entry_format,
                    item.edge_id, min(item.ewma_ms, 0xFFFF), min(item.sample_count, 0xFF)
                )
            return out
        if isinstance(payload, Telemetry):
            return struct.pack(
                self._formats[MessageType.TELEMETRY],
                payload.current_node,
                self.no_node if payload.next_node < 0 else payload.next_node,
                payload.progress_q8, payload.battery_pct, payload.state,
                NO_TASK if payload.task_id < 0 else payload.task_id, payload.priority,
            )
        if isinstance(payload, Complete):
            return struct.pack(self._formats[MessageType.COMPLETE], payload.task_id)
        if isinstance(payload, Blockage):
            return struct.pack(
                self._formats[MessageType.BLOCKAGE], payload.edge_id, int(payload.passable)
            )
        if isinstance(payload, Beacon):
            return struct.pack(
                self._formats[MessageType.BEACON], payload.fleet_time_ms & 0xFFFFFFFF
            )
        raise TypeError(f"cannot encode {type(payload).__name__}")

    # -- decoding ------------------------------------------------------------

    def decode(self, frame: bytes) -> Message | None:
        """Decode a frame, or return None if it must be discarded.

        NFR-3.1 and NFR-3.3 require corrupt and out-of-range frames to be
        discarded *without effect*. Returning None rather than raising matters:
        a corrupt frame on a shared 2.4 GHz band is an expected event, and a
        receiver that threw on one could be silenced by noise alone.
        """
        try:
            return self.decode_strict(frame)
        except (DecodeError, struct.error, ValueError):
            return None

    def decode_strict(self, frame: bytes) -> Message:
        if len(frame) < HEADER_SIZE + CRC_SIZE:
            raise DecodeError(f"frame of {len(frame)} bytes is too short")
        if len(frame) > config.MAX_FRAME_BYTES:
            raise DecodeError(f"frame of {len(frame)} bytes exceeds IF-4.1")

        body, received_crc = frame[:-CRC_SIZE], struct.unpack("<H", frame[-CRC_SIZE:])[0]
        if crc16(body) != received_crc:
            raise DecodeError("CRC mismatch")

        raw_type, sender, seq, timestamp = struct.unpack(HEADER, body[:HEADER_SIZE])
        try:
            message_type = MessageType(raw_type)
        except ValueError:
            raise DecodeError(f"unknown message type {raw_type}") from None

        payload = self._decode_payload(message_type, body[HEADER_SIZE:])
        return Message(sender=sender, seq=seq, timestamp_ms=timestamp, payload=payload)

    def _decode_payload(self, message_type: MessageType, body: bytes) -> Payload:
        if message_type is MessageType.DIGEST:
            if not body:
                raise DecodeError("DIGEST with no entry count")
            count = body[0]
            if count > MAX_DIGEST_ENTRIES:
                raise DecodeError(f"DIGEST claims {count} entries, over the slice cap")
            entry_format = f"<{self._id}HB"
            size = struct.calcsize(entry_format)
            if len(body) < 1 + count * size:
                raise DecodeError("DIGEST shorter than its declared entry count")
            entries = tuple(
                DigestEntry(*struct.unpack_from(entry_format, body, 1 + index * size))
                for index in range(count)
            )
            return Digest(entries=entries)

        fields = struct.unpack(self._formats[message_type], body)

        if message_type is MessageType.INTENT:
            (
                current, n1, n2, n3, e1, e2, e3,
                priority, state, battery, pheromone, held,
            ) = fields
            if battery > 100:
                raise DecodeError(f"battery {battery}% is out of range")
            nodes = tuple(n for n in (n1, n2, n3) if n != self.no_node)
            return Intent(
                current_node=current,
                next_nodes=nodes,
                eta_ms=(e1, e2, e3)[: len(nodes)],
                priority=priority, state=state,
                battery_pct=battery, pheromone=pheromone,
                held_task_id=-1 if held == NO_TASK else held,
            )
        if message_type is MessageType.RESERVE:
            junction, start, end, priority = fields
            if end < start:
                raise DecodeError(f"reservation window [{start},{end}] ends before it starts")
            return Reserve(junction, start, end, priority)
        if message_type is MessageType.ANNOUNCE:
            task_id, pickup, drop, priority, created, zone = fields
            if pickup == drop:
                raise DecodeError("ANNOUNCE with pickup == drop is not a journey")
            return Announce(
                task_id, pickup, drop, priority, created,
                -1 if zone == NO_ZONE_WIRE else zone,
            )
        if message_type is MessageType.BID:
            return Bid(fields[0], fields[1], bool(fields[2]))
        if message_type is MessageType.CLAIM:
            return Claim(fields[0], fields[1])
        if message_type is MessageType.TELEMETRY:
            current, nxt, progress, battery, state, task_id, priority = fields
            return Telemetry(
                current_node=current,
                next_node=-1 if nxt == self.no_node else nxt,
                progress_q8=progress, battery_pct=battery, state=state,
                task_id=-1 if task_id == NO_TASK else task_id, priority=priority,
            )
        if message_type is MessageType.COMPLETE:
            return Complete(fields[0])
        if message_type is MessageType.BLOCKAGE:
            return Blockage(fields[0], bool(fields[1]))
        if message_type is MessageType.BEACON:
            return Beacon(fields[0])
        raise DecodeError(f"no decoder for {message_type.name}")


def _clamp_i32(value: int) -> int:
    """Clamp to a signed 32-bit range rather than letting struct raise.

    A bid can legitimately be large or negative -- the aging term subtracts
    without bound -- and an overflow at the auction layer would be a far worse
    failure than a saturated bid, which merely loses ordering at the extremes.
    """
    return max(-(2**31), min(2**31 - 1, value))


def packed_sizes(*, edge_id_bits: int = config.EDGE_ID_BITS_DEFAULT) -> dict[str, int]:
    """Encoded size of every message type, for the IF-4.1 architecture test."""
    codec = Codec(edge_id_bits)
    return {
        message_type.name: codec.frame_size(message_type) for message_type in MessageType
    }
