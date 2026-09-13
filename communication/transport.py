"""How frames get between robots.

One interface, three backends, and the same ``core/`` code on all of them:

* ``InProcBus`` -- deterministic, seeded, in-process. Used by the benchmark. It is
  message-passing only: robots hold no references to each other and see nothing
  but bytes, so decentralisation is preserved while a run stays reproducible from
  its seed (NFR-4.4). This is the subtle one to get right, because a bus that let
  a robot observe a peer's state directly would quietly make the whole simulation
  centralised while every test still passed.
* ``UdpBroadcast`` -- real sockets between OS processes, one process per robot.
  This is the configuration that demonstrates genuine decentralisation, and it
  mirrors ESP-NOW's connectionless broadcast (IF-4.2, IF-4.5).
* ``ReceiveOnly`` -- a wrapper with no send path at all. The dashboard's read-only
  guarantee (FR-8.4, IF-1.6, BR-7) is structural rather than a UI rule: the
  backend is handed one of these, so there is no ``broadcast`` to call.

Two behaviours are modelled rather than assumed away:

**Radio range** (FR-9.4). INTENT and RESERVE are not zone-scoped; their scope is
the physical neighbourhood, which is already exactly the set of robots capable of
colliding. Implementing that is also what makes delivery O(N x neighbours) instead
of O(N squared), and so what makes 100 robots tractable.

**Packet loss** (ASM-8, TC-28). Loss is expected, not exceptional: INTENT repeats
every 200 ms precisely so a dropped frame is self-healing, and IF-4.5 forbids
acknowledgement or retransmission on the safety path. A transport that never
dropped anything would let a latent dependence on reliable delivery survive all
the way to hardware.
"""

from __future__ import annotations

import random
import socket
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from core import config
from communication.messages import Codec, Message, MessageType

PositionFn = Callable[[int], tuple[int, int] | None]
"""Robot id -> (x_mm, y_mm), or None if that robot is not present in the world."""


@dataclass
class TransportStats:
    """Frame counts, for the FR-9.6 and NFR-1.12 auction traffic budget."""

    sent: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    delivered: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dropped_loss: int = 0
    dropped_range: int = 0
    dropped_absent: int = 0
    """Frames a non-member tried to send. Non-zero means something attempted to
    transmit on behalf of a robot that has left the mesh."""

    @property
    def total_sent(self) -> int:
        return sum(self.sent.values())

    @property
    def total_delivered(self) -> int:
        return sum(self.delivered.values())

    def auction_frames(self) -> int:
        """BID and CLAIM frames sent. Section 4.9 counts auction cost in these."""
        return self.sent.get("BID", 0) + self.sent.get("CLAIM", 0)

    def summary(self) -> str:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(self.sent.items()))
        return (
            f"sent {self.total_sent} ({kinds}); delivered {self.total_delivered}; "
            f"dropped {self.dropped_loss} to loss, {self.dropped_range} out of range, "
            f"{self.dropped_absent} from absent senders"
        )


class Transport(Protocol):
    """The whole interface. Broadcast, and collect what arrived."""

    def broadcast(self, sender: int, frame: bytes) -> None:
        ...

    def receive(self, receiver: int) -> list[bytes]:
        ...


class ReceiveOnlyError(RuntimeError):
    """Something tried to transmit through a receive-only transport."""


@dataclass
class InProcBus:
    """Deterministic in-process broadcast bus.

    Frames broadcast during a tick are delivered at the *start* of the next tick,
    not immediately. That one-tick latency is not an implementation shortcut: it
    stops a robot from hearing a peer's INTENT in the same tick it was formed,
    which no real radio could do, and it is what makes tick ordering unable to
    leak information between robots.
    """

    seed: int = 0
    loss_permille: int = 0
    """Frames dropped per thousand. Deterministic given the seed."""

    range_mm: int | None = config.RADIO_RANGE_MM
    """None means unlimited range, which is only correct for a fleet small enough
    that every robot can collide with every other."""

    position_of: PositionFn | None = None
    """Required when ``range_mm`` is set. Without positions the bus cannot know
    who is in range and would silently fall back to flooding."""

    wired: set[int] = field(default_factory=set)
    """Members exempt from range filtering.

    The order gateway belongs here. IF-3.3 has it bridging between the mesh and the
    dashboard transport, so it is infrastructure with a fixed installation rather
    than a robot with a position. Without this exemption an ANNOUNCE would be
    range-filtered against a position the gateway does not have, and no robot would
    ever hear about a task -- a failure that looks like a broken auction rather than
    a broken radio model."""

    stats: TransportStats = field(default_factory=TransportStats)
    codec: Codec = field(default_factory=Codec)

    _in_flight: list[tuple[int, bytes]] = field(default_factory=list, repr=False)
    _inboxes: dict[int, list[bytes]] = field(default_factory=dict, repr=False)
    _members: list[int] = field(default_factory=list, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.range_mm is not None and self.position_of is None:
            raise ValueError(
                "range_mm is set but position_of is not, so the bus cannot tell "
                "who is in range and would silently flood instead (FR-9.4)"
            )

    # -- membership ----------------------------------------------------------

    def join(self, robot_id: int) -> None:
        if robot_id not in self._inboxes:
            self._inboxes[robot_id] = []
            self._members = sorted(self._inboxes)

    def leave(self, robot_id: int) -> None:
        """Remove a robot from the mesh. A killed robot hears nothing and is
        heard by nobody, which is what makes peers notice it through silence."""
        self._inboxes.pop(robot_id, None)
        self._members = sorted(self._inboxes)

    @property
    def members(self) -> list[int]:
        return list(self._members)

    # -- sending -------------------------------------------------------------

    def broadcast(self, sender: int, frame: bytes) -> None:
        """Put a frame on the air.

        A sender that is not a member transmits nothing. That is the whole point of
        ``leave``: a robot that has lost power cannot send, and TC-5 requires its
        peers to notice through silence alone. Enforcing it here as well as in the
        engine means no future caller can resurrect a dead robot's voice.
        """
        if sender not in self._inboxes:
            self.stats.dropped_absent += 1
            return
        self.stats.sent[self._kind(frame)] += 1
        self._in_flight.append((sender, frame))

    def _kind(self, frame: bytes) -> str:
        if not frame:
            return "EMPTY"
        try:
            return MessageType(frame[0]).name
        except ValueError:
            return "UNKNOWN"

    # -- delivery ------------------------------------------------------------

    def deliver(self) -> None:
        """Move everything in flight into receivers' inboxes.

        Called once per tick by the engine, before robots decide. Iteration order
        over senders and receivers is sorted so the seeded loss draws happen in a
        fixed order -- otherwise two runs of the same seed would drop different
        frames and NFR-4.4 would fail intermittently, which is the worst way for
        it to fail.
        """
        pending, self._in_flight = self._in_flight, []
        for sender, frame in pending:
            kind = self._kind(frame)
            for receiver in self._members:
                if receiver == sender:
                    continue  # a robot does not hear its own broadcast
                if not self._in_range(sender, receiver):
                    self.stats.dropped_range += 1
                    continue
                if self.loss_permille and self._rng.randrange(1000) < self.loss_permille:
                    self.stats.dropped_loss += 1
                    continue
                self._inboxes[receiver].append(frame)
                self.stats.delivered[kind] += 1

    def _in_range(self, sender: int, receiver: int) -> bool:
        if self.range_mm is None:
            return True
        if sender in self.wired or receiver in self.wired:
            return True
        assert self.position_of is not None
        here, there = self.position_of(sender), self.position_of(receiver)
        if here is None or there is None:
            return False  # one of them is not in the world
        gap_x, gap_y = here[0] - there[0], here[1] - there[1]
        return gap_x * gap_x + gap_y * gap_y <= self.range_mm * self.range_mm

    def receive(self, receiver: int) -> list[bytes]:
        """Take and clear one robot's inbox."""
        inbox = self._inboxes.get(receiver)
        if not inbox:
            return []
        self._inboxes[receiver] = []
        return inbox

    def decode_all(self, frames: list[bytes]) -> list[Message]:
        """Decode a batch, discarding anything malformed (NFR-3.1, NFR-3.3)."""
        decoded = []
        for frame in frames:
            message = self.codec.decode(frame)
            if message is not None:
                decoded.append(message)
        return decoded


@dataclass
class ReceiveOnly:
    """A transport with no send path (FR-8.4, IF-1.6, BR-7).

    The dashboard backend is constructed with one of these. Its read-only property
    is then structural: there is no working ``broadcast`` to call, so no future
    edit can add a command path by accident. A UI rule saying "do not send" would
    decay; this cannot.
    """

    inner: Transport

    def receive(self, receiver: int) -> list[bytes]:
        return self.inner.receive(receiver)

    def broadcast(self, sender: int, frame: bytes) -> None:
        raise ReceiveOnlyError(
            "this transport is receive-only: the dashboard must never issue a "
            "command capable of altering AMR behaviour (FR-8.4, IF-1.6, BR-7)"
        )


@dataclass
class UdpBroadcast:
    """Real UDP broadcast between OS processes, one process per robot.

    Used by the multi-process decentralisation demo rather than by the benchmark:
    real sockets are wall-clock bound and lossy in ways no seed controls, so a run
    over this transport cannot satisfy NFR-4.4. What it does demonstrate is that
    the coordination logic needs nothing but datagrams -- no shared memory, no
    broker, no access point (IF-4.2, IF-4.5).
    """

    port: int = 47_808
    group: str = "255.255.255.255"
    bind_host: str = ""
    stats: TransportStats = field(default_factory=TransportStats)
    codec: Codec = field(default_factory=Codec)
    _socket: socket.socket | None = field(default=None, init=False, repr=False)

    def open(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((self.bind_host, self.port))
        sock.setblocking(False)
        self._socket = sock

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "UdpBroadcast":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def broadcast(self, sender: int, frame: bytes) -> None:
        if self._socket is None:
            raise RuntimeError("transport is not open; call open() first")
        del sender  # the frame already carries its sender
        self.stats.sent[self._kind(frame)] += 1
        self._socket.sendto(frame, (self.group, self.port))

    def receive(self, receiver: int) -> list[bytes]:
        """Drain the socket. Non-blocking, so it returns what has arrived.

        Frames this process sent itself are filtered out by sender id, because a
        broadcast datagram loops back to its own sender and a robot must not act
        on its own INTENT.
        """
        if self._socket is None:
            raise RuntimeError("transport is not open; call open() first")
        frames: list[bytes] = []
        while True:
            try:
                data, _ = self._socket.recvfrom(config.MAX_FRAME_BYTES * 2)
            except BlockingIOError:
                break
            except OSError:
                break
            if len(data) > 1 and data[1] == receiver:
                continue  # our own loopback
            frames.append(data)
            self.stats.delivered[self._kind(data)] += 1
        return frames

    def _kind(self, frame: bytes) -> str:
        if not frame:
            return "EMPTY"
        try:
            return MessageType(frame[0]).name
        except ValueError:
            return "UNKNOWN"


@dataclass
class SequenceFilter:
    """Rejects replayed and out-of-order frames (FR-1.6, NFR-3.2).

    FR-1.6: discard a frame whose sequence number is not greater than the last
    accepted one from that sender. Kept separate from the transport because it is
    a *receiver* policy: each robot decides what it will accept, and on real
    hardware the radio has no idea what a sequence number means.

    Sequence numbers are uint16 and wrap. A naive ``seq > last`` comparison would
    reject every frame for the rest of the run after the first wrap, silently
    turning a working fleet into a set of deaf robots. Half-range wrap-around
    comparison handles it.
    """

    window: int = 1 << 15
    _last: dict[int, int] = field(default_factory=dict, repr=False)
    rejected: int = 0

    def accept(self, sender: int, seq: int) -> bool:
        previous = self._last.get(sender)
        if previous is None:
            self._last[sender] = seq
            return True
        # Signed difference modulo 2^16: positive means newer.
        delta = ((seq - previous + (1 << 15)) & 0xFFFF) - (1 << 15)
        if delta <= 0:
            self.rejected += 1
            return False
        self._last[sender] = seq
        return True

    def forget(self, sender: int) -> None:
        """Drop state for a peer declared lost, so a rebooted robot starting its
        sequence again is not treated as a replay attacker (FR-1.5, FR-6.5)."""
        self._last.pop(sender, None)

    def last_seen(self, sender: int) -> int | None:
        return self._last.get(sender)
