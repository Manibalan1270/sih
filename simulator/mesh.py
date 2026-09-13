"""The mesh: where typed messages become frames and back.

Robots deal in typed payloads; the radio deals in bytes. This is the boundary, and
keeping it out of ``core/`` is what lets the same robot object run over the
in-process bus, real UDP, and an ESP-NOW bridge without knowing which (IF-3.1).

It also owns the per-sender sequence counter. FR-1.6 and NFR-3.2 make sequence
numbers a real protocol element, and they belong to the *link*, not to the
decision logic: a robot has no reason to know that its INTENT is the 4,312th frame
it has sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from communication.messages import Codec, Message
from communication.transport import InProcBus, SequenceFilter
from core import config

GATEWAY_ID = 0
"""Sender id for the order gateway. Robots are numbered from 1, so zero is free.

The gateway is on the mesh as a peer rather than as a privileged node: IF-3.2 and
IF-3.3 let it originate only ANNOUNCE and the clock beacon, and give it no way to
assign work or command a robot."""


@dataclass
class Mesh:
    """Encodes outgoing payloads, decodes and filters incoming frames."""

    bus: InProcBus
    codec: Codec = field(default_factory=Codec)
    filters: dict[int, SequenceFilter] = field(default_factory=dict)
    """One per receiver. FR-1.6 is a receiver policy -- each robot decides what it
    will accept -- and on real hardware the radio has no idea what a sequence
    number means."""

    drop_stale: bool = True
    _seq: dict[int, int] = field(default_factory=dict, repr=False)
    decoded: int = 0
    discarded_malformed: int = 0
    discarded_stale: int = 0

    # -- membership ----------------------------------------------------------

    def join(self, member_id: int) -> None:
        self.bus.join(member_id)
        self.filters.setdefault(member_id, SequenceFilter())

    def leave(self, member_id: int) -> None:
        """Remove a member. A killed robot hears nothing and is heard by nobody."""
        self.bus.leave(member_id)
        for other in self.filters.values():
            other.forget(member_id)

    # -- sending -------------------------------------------------------------

    def send(self, sender: int, payload, now_ms: int) -> bytes:
        """Encode and broadcast one payload.

        The sequence number is per sender and wraps at 16 bits, which
        ``SequenceFilter`` is written to handle. Wrapping is not hypothetical: at
        5 Hz per robot a run reaches 65,536 INTENTs in about three and a half
        hours, well inside a warehouse shift.
        """
        seq = (self._seq.get(sender, 0) + 1) & 0xFFFF
        self._seq[sender] = seq
        frame = self.codec.encode(sender, seq, now_ms, payload)
        self.bus.broadcast(sender, frame)
        return frame

    def send_all(self, sender: int, payloads, now_ms: int) -> int:
        for payload in payloads:
            self.send(sender, payload, now_ms)
        return len(payloads)

    # -- receiving -----------------------------------------------------------

    def deliver(self) -> None:
        """Move everything in flight into inboxes. Once per tick."""
        self.bus.deliver()

    def inbox(self, receiver: int) -> list[Message]:
        """Decoded, integrity-checked, de-duplicated messages for one receiver.

        Three filters apply in order, each discarding *without effect*:
        malformed or CRC-failed frames (IF-4.3, NFR-3.1, NFR-3.3), then frames
        whose sequence number is not greater than the last accepted from that
        sender (FR-1.6, NFR-3.2).
        """
        frames = self.bus.receive(receiver)
        if not frames:
            return []

        sequence_filter = self.filters.setdefault(receiver, SequenceFilter())
        messages: list[Message] = []
        for frame in frames:
            message = self.codec.decode(frame)
            if message is None:
                self.discarded_malformed += 1
                continue
            if self.drop_stale and not sequence_filter.accept(message.sender, message.seq):
                self.discarded_stale += 1
                continue
            self.decoded += 1
            messages.append(message)
        return messages

    # -- reporting -----------------------------------------------------------

    @property
    def stats(self):
        return self.bus.stats

    def summary(self) -> str:
        return (
            f"{self.bus.stats.summary()}; decoded {self.decoded}, "
            f"discarded {self.discarded_malformed} malformed / "
            f"{self.discarded_stale} stale"
        )


def build_mesh(
    *,
    seed: int,
    id_bits: int,
    position_of,
    loss_permille: int = 0,
    range_mm: int | None = config.RADIO_RANGE_MM,
    wired: set[int] | None = None,
) -> Mesh:
    """Standard mesh for a simulated fleet.

    ``wired`` defaults to the gateway alone: it is infrastructure with a fixed
    installation rather than a robot with a position, so range filtering does not
    apply to it (IF-3.3).
    """
    bus = InProcBus(
        seed=seed,
        loss_permille=loss_permille,
        range_mm=range_mm,
        position_of=position_of,
        wired={GATEWAY_ID} if wired is None else wired,
        codec=Codec(id_bits),
    )
    return Mesh(bus=bus, codec=Codec(id_bits))
