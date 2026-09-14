"""Peer liveness and last-known state (FE-1, FR-1.4, FR-1.5, FR-6.5).

A robot learns everything about its neighbours from their INTENT heartbeat, and
learns that one has *gone* from the absence of it. There is no death notification --
a robot that has lost power cannot send one -- so silence is the only signal, and
FR-1.5 fixes the threshold at 1500 ms.

Declaring a peer lost has three consequences under FR-6.5, and this module's job is
to raise the event carrying the facts each one needs: expire its reservations,
treat its last known node as an obstacle, and re-announce the task it was holding.
That last one is why ``held_task_id`` is tracked at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core import config


@dataclass
class Peer:
    """What one robot knows about one neighbour, all of it overheard."""

    robot_id: int
    last_seen_ms: int
    current_node: int = -1
    priority: int = 0
    state: int = 0
    battery_pct: int = 100
    held_task_id: int = -1
    intents_received: int = 0

    plan_seq: int = 0
    plan_index: int = 0
    """Which route plan the peer says is in force, and how many of its steps it has
    released (see communication.messages.Path). Execution by precedence reads these:
    a robot waiting to enter a resource checks that everyone booked ahead of it there
    has a plan_index past that step. Zero plan_seq means no plan."""

    next_nodes: tuple[int, ...] = ()
    eta_ms: tuple[int, ...] = ()
    """The peer's declared route horizon and arrival times, rebased onto the aligned
    clock (section 3.4.1).

    Kept so a robot can *anticipate* traffic ahead of it on an edge instead of
    discovering it with a proximity sensor. Reactive proximity halting is what §1.5
    defines stop-and-wait as, and FR-5.6/NFR-2.4 reserve braking for non-cooperative
    obstacles -- which a peer broadcasting five times a second is not. Everything
    needed to avoid it is already in the frame; it simply was not being retained."""

    def age_ms(self, now_ms: int) -> int:
        return max(0, now_ms - self.last_seen_ms)

    @property
    def heading_to(self) -> int:
        """The next node the peer intends to cross, or -1 if it declared none."""
        return self.next_nodes[0] if self.next_nodes else -1

    def arrival_at(self, node: int) -> int | None:
        """Aligned-clock arrival at ``node``, if the peer declared it."""
        for declared, eta in zip(self.next_nodes, self.eta_ms):
            if declared == node:
                return eta
        return None

    def is_on_edge(self, from_node: int, to_node: int) -> bool:
        """Whether the peer is traversing ``from_node -> to_node`` right now.

        Direction matters: a peer coming the other way along a bidirectional aisle is
        in the other lane and is not traffic to follow.
        """
        return self.current_node == from_node and self.heading_to == to_node

    def is_live(self, now_ms: int, timeout_ms: int = config.PEER_TIMEOUT_MS) -> bool:
        return self.age_ms(now_ms) <= timeout_ms

    def __str__(self) -> str:
        return (
            f"r{self.robot_id} @{self.current_node} p{self.priority} "
            f"bat{self.battery_pct}% task{self.held_task_id}"
        )


@dataclass(frozen=True, slots=True)
class PeerLost:
    """A peer declared lost, with what FR-6.5 needs to respond.

    Carries the facts rather than a reference to the peer, because by the time this
    is acted on the entry is gone -- and the last known node has to survive the
    removal to be usable as an obstacle.
    """

    robot_id: int
    last_seen_ms: int
    last_node: int
    held_task_id: int

    def __str__(self) -> str:
        return (
            f"r{self.robot_id} lost (last seen {self.last_seen_ms} ms at node "
            f"{self.last_node}, holding task {self.held_task_id})"
        )


@dataclass
class PeerTable:
    """FR-1.4: neighbours from which INTENT has arrived within the liveness window."""

    timeout_ms: int = config.PEER_TIMEOUT_MS
    peers: dict[int, Peer] = field(default_factory=dict)
    lost_count: int = 0

    def __len__(self) -> int:
        return len(self.peers)

    def __contains__(self, robot_id: int) -> bool:
        return robot_id in self.peers

    def get(self, robot_id: int) -> Peer | None:
        return self.peers.get(robot_id)

    @property
    def ids(self) -> tuple[int, ...]:
        """Known peers in id order, so iteration is deterministic (FR-5.8)."""
        return tuple(sorted(self.peers))

    def observe(
        self,
        *,
        robot_id: int,
        now_ms: int,
        current_node: int,
        priority: int,
        state: int,
        battery_pct: int,
        held_task_id: int,
        next_nodes: tuple[int, ...] = (),
        eta_ms: tuple[int, ...] = (),
        plan_seq: int = 0,
        plan_index: int = 0,
    ) -> Peer:
        """Record an INTENT from a peer, creating the entry if it is new.

        FR-2.7's counterpart for peers: a robot joining an operating fleet is
        discovered by its first heartbeat, with no registration step and nothing to
        configure. NFR-4.6 requires adding an AMR to need no reconfiguration of the
        existing ones, and this is where that holds.
        """
        peer = self.peers.get(robot_id)
        if peer is None:
            peer = Peer(robot_id=robot_id, last_seen_ms=now_ms)
            self.peers[robot_id] = peer
        peer.last_seen_ms = now_ms
        peer.current_node = current_node
        peer.priority = priority
        peer.state = state
        peer.battery_pct = battery_pct
        peer.held_task_id = held_task_id
        peer.next_nodes = tuple(next_nodes)
        peer.eta_ms = tuple(eta_ms)
        peer.plan_seq = plan_seq
        peer.plan_index = plan_index
        peer.intents_received += 1
        return peer

    def reap(self, now_ms: int) -> list[PeerLost]:
        """Declare silent peers lost (FR-1.5). Returns them in id order.

        Removing the entry is deliberate rather than flagging it. A peer kept in the
        table as "probably dead" is a peer whose stale position keeps being consulted
        -- and FR-6.5's response is to treat the last node as an obstacle *once*, not
        to keep avoiding a robot that may have been carried away hours ago.
        """
        lost: list[PeerLost] = []
        for robot_id in sorted(self.peers):
            peer = self.peers[robot_id]
            if peer.is_live(now_ms, self.timeout_ms):
                continue
            lost.append(
                PeerLost(
                    robot_id=robot_id,
                    last_seen_ms=peer.last_seen_ms,
                    last_node=peer.current_node,
                    held_task_id=peer.held_task_id,
                )
            )
        for entry in lost:
            del self.peers[entry.robot_id]
            self.lost_count += 1
        return lost

    def live(self, now_ms: int) -> tuple[Peer, ...]:
        return tuple(
            self.peers[robot_id]
            for robot_id in self.ids
            if self.peers[robot_id].is_live(now_ms, self.timeout_ms)
        )

    def forget(self, robot_id: int) -> None:
        self.peers.pop(robot_id, None)

    def clear(self) -> None:
        self.peers.clear()

    def summary(self, now_ms: int) -> str:
        if not self.peers:
            return "no peers heard"
        return "; ".join(
            f"{self.peers[i]} ({self.peers[i].age_ms(now_ms)} ms ago)" for i in self.ids
        )
