"""The robot: its state, and its decision loop.

This is the file the whole architecture exists to make possible. Every robot runs
an identical copy of it, holds no shared memory with any other, and decides
everything locally from what it can see and what it has been told.

Three constraints shape the design:

* ``step()`` is a pure function of its arguments and the robot's own state. No
  wall clock, no sockets, no simulator. Time arrives as ``now_ms`` on the aligned
  fleet clock. That is what lets the same object be driven by the headless
  engine, by a Webots controller, or by an ESP32 main loop (IF-3.1, NFR-4.7), and
  what makes a run reproducible from its seed (NFR-4.4).
* The robot decides *where* to go; it never moves itself. It emits a
  ``MotionCommand`` and something else -- the simulator, Webots, real motors --
  carries it out and reports back. This is the division of responsibility the
  implementation guide sets out, and collapsing it would make the decision logic
  untestable without a physics engine.
* Collaborators are injected, not imported. The planner is a protocol, so Phase 3
  can substitute A* for whatever Phase 2 used without this file changing.

At this phase the robot plans and follows routes. Bidding, reservation and
arbitration are wired into the same loop in Phases 5 and 6; the hooks they will
use are marked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from communication.messages import (
    Announce,
    Bid,
    Claim,
    Complete,
    Intent,
    Message,
    MessageType,
    Reserve,
)
from core import config
from core.arbitration import Arbiter, Outcome, crossing_window, outranks
from core.auction import NO_WINNER, Auctioneer
from core.graph import Graph
from core.peers import PeerTable
from core.reservation import ReservationTable, Window
from core.state_machine import Event, State, StateMachine
from core.task import Leg, Task, TaskQueue, TaskState


class RoutePlanner(Protocol):
    """Anything that can produce a route over the graph.

    A protocol rather than a concrete import so that core/robot.py never depends
    on which planner is in use. Phase 3 supplies A*; Phase 9 supplies D* Lite for
    repair. Returning ``None`` means no traversable route exists, which the robot
    turns into an UNREACHABLE declaration (FR-3.7).
    """

    def route(self, start: int, goal: int) -> list[int] | None:
        ...

    def travel_cost(self, start: int, goal: int) -> int:
        """Least cost between two nodes, or ``INFINITE_COST`` if unreachable.

        Separate from ``route`` because the auction prices journeys rather than
        following them, and should not have to build and discard a route to find
        out what one would cost (FR-4.3).
        """
        ...


@dataclass(frozen=True, slots=True)
class MotionCommand:
    """What the robot asks the world to do this tick.

    ``target_node`` is the immediate next junction, not the final goal: the robot
    re-decides every tick, so a yield or a replan takes effect on the next
    command rather than needing the previous one cancelled.
    """

    speed_mm_s: int
    target_node: int | None = None

    @property
    def is_hold(self) -> bool:
        return self.target_node is None or self.speed_mm_s <= 0

    @classmethod
    def hold(cls) -> "MotionCommand":
        return cls(speed_mm_s=0, target_node=None)


@dataclass
class StepResult:
    """Everything one tick of the decision loop produced."""

    command: MotionCommand
    outbox: list[object] = field(default_factory=list)
    """Message payloads to broadcast.

    Typed payload objects, not encoded bytes. Encoding belongs at the mesh
    boundary, so identifier width and CRC never reach the decision logic -- which
    is what lets the same robot object run over the in-process bus, real UDP, and
    an ESP-NOW bridge without noticing (IF-3.1)."""

    notes: list[str] = field(default_factory=list)
    """Human-readable record of what was decided and why. Feeds the dashboard's
    per-robot decision panel and the benchmark event log."""


_STATE_CODES: dict[State, int] = {
    state: index for index, state in enumerate(State)
}
"""State to the uint8 the INTENT ``state`` field carries (section 3.4.1).

Derived from declaration order rather than written out, so a state added to
Appendix A cannot be forgotten here and silently encode as another state."""


@dataclass
class RobotMetrics:
    """Per-robot counters the benchmark reads (FR-10.2, FR-10.3).

    Accumulated here rather than derived afterwards because some of these are not
    recoverable from a position log: time spent stopped and yields granted are
    facts about decisions, not about trajectories.
    """

    distance_mm: int = 0
    moving_ms: int = 0
    stopped_ms: int = 0
    yields_lost: int = 0
    """Arbitration losses -- this robot gave way (FR-10.3)."""

    yields_won: int = 0
    """Arbitration wins -- a peer gave way to this robot."""

    replans: int = 0
    tasks_completed: int = 0
    bids_sent: int = 0
    edges_used: set[int] = field(default_factory=set)
    """Distinct edges traversed, for the route-diversity metric."""


@dataclass
class Robot:
    """One AMR's complete onboard state and decision loop."""

    robot_id: int
    graph: Graph
    planner: RoutePlanner
    home_node: int
    zone_id: int = -1

    # -- logical position ----------------------------------------------------
    current_node: int = -1
    """Section 3.4.1: the junction most recently occupied or departed. While
    traversing an edge this stays at the node behind the robot."""

    next_node: int | None = None
    edge_id: int | None = None
    progress_mm: int = 0
    """Distance travelled along ``edge_id``. Owned by the robot but written by
    whatever is moving it, via ``advance``."""

    # -- onboard state -------------------------------------------------------
    battery_pct: int = 100
    task_priority: int = 0
    """Priority of the held task, used as the arbitration ranking key's first
    component (FR-5.4, BR-1). Zero when idle, so a working robot outranks an idle
    one at a junction."""

    machine: StateMachine = field(default_factory=StateMachine)
    queue: TaskQueue = field(default_factory=lambda: TaskQueue(config.QUEUE_CAP))
    metrics: RobotMetrics = field(default_factory=RobotMetrics)
    auctioneer: Auctioneer | None = None
    """This robot's own participation in task allocation (FE-4). None disables
    bidding, which is how Configuration A's baseline runs the same robot object
    with no auction at all (FR-10.5)."""

    zone_eligible: Callable[[int], bool] | None = None
    """Whether a task in a given zone may be bid on (FR-4.14, FR-9.3). Injected so
    the robot never has to know how the warehouse is partitioned."""

    reservations: ReservationTable = field(default_factory=ReservationTable)
    """This robot's own view of who holds which junction (FR-5.1)."""

    peers: PeerTable = field(default_factory=PeerTable)
    """Neighbours heard from within the liveness window (FR-1.4)."""

    arbiter: Arbiter | None = None
    """Junction arbitration (FE-5). None disables it, which is how Configuration A
    runs the same robot object with no coordination at all (FR-10.5)."""

    forward_clearance_mm: int = 1 << 30
    """Distance to the nearest obstacle directly ahead, from the forward sensor
    (IF-2.4). Written every tick by whatever is driving the robot -- the simulator
    here, a real sensor on hardware. Defaults to "clear"."""

    position_confident: bool = True
    """FR-5.14 / NFR-2.3. Phase 10 drives this from core/localization.py; until then
    a robot always knows where it is."""

    yielding_until_ms: int = 0
    """When the conflict this robot yielded to is expected to clear. YIELD is left
    on CONFLICT_CLEARED once this passes (Appendix A: an excursion from MOVING that
    returns to it directly)."""

    route: list[int] = field(default_factory=list)
    route_index: int = 0
    """Index in ``route`` of ``current_node``. The remaining route is
    ``route[route_index:]``."""

    completed_tasks: list[Task] = field(default_factory=list)
    """Finished tasks awaiting collection by whatever is measuring the run.

    The robot keeps them rather than discarding them because FR-10.2 needs the
    per-task completion time, and that is only knowable from the task object --
    a position log cannot tell you when an order was satisfied."""

    released_tasks: list[Task] = field(default_factory=list)
    """Tasks this robot has handed back to the mesh (FR-6.2, FR-3.7).

    A robot that cannot reach a task must *let go* of it, not merely report it.
    Leaving an UNREACHABLE task in the queue makes the robot cycle
    IDLE -> PLANNING -> IDLE forever, re-failing the same plan every tick. The
    auction layer drains this list and re-announces, so the task gets another
    chance from another robot -- or from this one once the blockage clears."""

    _completion_echo: dict[int, int] = field(default_factory=dict)
    """task_id -> INTENT periods left to keep re-announcing its completion.

    See ``_emit_intent``: COMPLETE is one-shot, and a peer that missed it cannot tell
    "done" from "still someone else's"."""

    COMPLETION_ECHOES: int = 4
    """Periods to repeat COMPLETE. Four covers 800 ms, which at 30% independent loss
    leaves under a 1% chance of a peer missing every copy."""

    _charge_accumulator_ms: int = 0
    """Sub-percent charging time carried between ticks, mirroring the drain side."""

    _avoid_edges: set[int] = field(default_factory=set)
    """Edges this robot is temporarily avoiding after losing a junction contest
    (Appendix B's AVOID, second branch).

    Local to the robot, and deliberately not a graph blockage: the aisle is
    perfectly passable, it is merely contended, and marking the graph would make one
    robot's yield everyone's problem. Cleared once the yield resolves."""

    _last_intent_ms: int = -10_000
    """When INTENT was last broadcast. Negative so the first tick emits one."""

    _drain_accumulator_mm: int = 0
    """Sub-percent battery drain carried between ticks. See ``drain_battery``."""

    def __post_init__(self) -> None:
        if self.current_node < 0:
            self.current_node = self.home_node
        self.graph.node(self.current_node)  # fail now if the spawn node is bogus

    # -- identity and reporting ---------------------------------------------

    @property
    def state(self) -> State:
        return self.machine.state

    @property
    def task(self) -> Task | None:
        return self.queue.current

    @property
    def is_idle(self) -> bool:
        """FR-4.3 / BR-5: idleness earns a small bid preference."""
        return self.state is State.IDLE and not self.queue

    @property
    def remaining_route(self) -> list[int]:
        return self.route[self.route_index :]

    @property
    def progress_fraction_q8(self) -> int:
        """Progress along the current edge as Q8 fixed point (0..256).

        Sent to the dashboard so the frontend can animate smoothly between
        junctions without needing raw simulator coordinates.
        """
        if self.edge_id is None:
            return 0
        length = max(1, self.graph.length_mm(self.edge_id))
        return min(config.Q8_ONE, (self.progress_mm * config.Q8_ONE) // length)

    def footprint_mm(self) -> tuple[int, int]:
        """Position including lane offset, for the geometric collision check.

        Distinct from ``position_mm``, which returns the aisle centre line and is
        what the dashboard animates. On a bidirectional aisle robots keep to their
        own side (ASM-5), so opposing traffic is laterally separated and does not
        collide; on a single-lane aisle there is only the centre line, so a head-on
        is a real overlap. Collapsing the two would either invent collisions that
        cannot happen or hide ones that can.
        """
        x, y = self.position_mm()
        if self.edge_id is None or self.next_node is None:
            return x, y
        edge = self.graph.edge(self.edge_id)
        if edge.single_lane:
            return x, y

        here = self.graph.node(self.current_node)
        there = self.graph.node(self.next_node)
        span_x, span_y = there.x_mm - here.x_mm, there.y_mm - here.y_mm
        length = max(1, self.graph.length_mm(self.edge_id))
        # Right-hand perpendicular, scaled to the lane offset. Integer throughout.
        offset_x = -span_y * config.AISLE_LANE_OFFSET_MM // length
        offset_y = span_x * config.AISLE_LANE_OFFSET_MM // length
        return x + offset_x, y + offset_y

    def position_mm(self) -> tuple[int, int]:
        """Interpolated position on the aisle centre line, for the dashboard."""
        node = self.graph.node(self.current_node)
        if self.edge_id is None or self.next_node is None:
            return node.x_mm, node.y_mm
        target = self.graph.node(self.next_node)
        length = max(1, self.graph.length_mm(self.edge_id))
        travelled = min(self.progress_mm, length)
        x = node.x_mm + (target.x_mm - node.x_mm) * travelled // length
        y = node.y_mm + (target.y_mm - node.y_mm) * travelled // length
        return x, y

    # -- work intake ---------------------------------------------------------

    def accept_task(
        self,
        task: Task,
        now_ms: int,
        *,
        bid: int = 0,
        at_index: int | None = None,
    ) -> None:
        """Take on a task this robot has won.

        Called after the auction concludes in Phase 5. It exists separately from
        the auction so Phase 2 can exercise the movement path directly, and so a
        queued second task can be added without re-entering BIDDING.

        ``bid`` is retained on the task because FR-4.10 needs the holder's own bid
        to judge whether a challenger has beaten it by more than delta.
        """
        own = task.copy()  # never share a Task object between robots
        own.claim(self.robot_id, bid, now_ms)
        if at_index is None:
            self.queue.add(own)
        else:
            self.queue.insert(at_index, own)
        if self.state is State.IDLE:
            self.machine.fire(Event.QUEUE_HAS_WORK, now_ms)

    # -- the decision loop ---------------------------------------------------

    def step(self, now_ms: int, inbox: list[Message] | None = None) -> StepResult:
        """One tick of the robot's own decision making.

        Order within the tick matters. Messages are processed first, because what
        peers said changes what this robot should decide; then auctions are
        advanced; then the state machine acts. Deciding before reading the inbox
        would make the robot act on a world one tick stale.

        The state dispatch is deliberately flat rather than a chain of conditions.
        Appendix A defines behaviour per state, and matching that shape means a
        reviewer can check this against the SRS table line by line.
        """
        result = StepResult(command=MotionCommand.hold())

        # FR-5.11: claims lapse on their own, with no release message. Done before
        # anything reads the table, so a decision is never taken against a window
        # that has already passed.
        self.reservations.expire(now_ms)
        self._reap_peers(now_ms, result)

        if inbox:
            self._handle_inbox(inbox, now_ms, result)
        if self.auctioneer is not None:
            self._advance_auctions(now_ms, result)
            self._emit_intent(now_ms, result)

        # Invariant guard. A robot in a working state with no task cannot make
        # progress and will never report itself finished, so a defect that produces
        # one turns into a hung run rather than a failing assertion -- the worst
        # failure mode for a benchmark. Recover instead of hanging.
        if self.machine.holds_task and self.queue.current is None:
            self.machine.fire_if_possible(Event.TASK_WITHDRAWN, now_ms)
            result.notes.append(
                "held no task while in a working state; returned to IDLE"
            )
            self.route, self.route_index = [], 0
            self.next_node, self.edge_id, self.progress_mm = None, None, 0

        if self.state is State.IDLE:
            self._step_idle(now_ms, result)
        elif self.state is State.PLANNING:
            self._step_planning(now_ms, result)
        elif self.state in (State.MOVING, State.YIELD):
            self._step_moving(now_ms, result)
        elif self.state is State.REPLAN:
            self._step_replan(now_ms, result)
        elif self.state is State.AT_DROP:
            self._step_at_drop(now_ms, result)
        elif self.state is State.CHARGING:
            self._step_charging(now_ms, result)
        # BIDDING and FAULT hold position. BIDDING is a 300 ms window during which
        # the robot keeps its committed plan but starts nothing new; the auction
        # itself is driven by messages, not by this loop.

        return result

    # -- messaging -----------------------------------------------------------

    def _handle_inbox(self, inbox: list[Message], now_ms: int, result: StepResult) -> None:
        """Act on what peers said this tick.

        Unknown message types are ignored rather than rejected. A fleet is not
        necessarily uniform in firmware version during a rolling update, and a
        robot that faulted on a message type it did not recognise would turn a
        compatible addition into a fleet-wide outage.
        """
        auctioneer = self.auctioneer
        for message in inbox:
            payload = message.payload
            if auctioneer is None:
                continue
            if message.type is MessageType.INTENT:
                self._on_peer_intent(message.sender, payload, message.timestamp_ms, now_ms, result)
            elif message.type is MessageType.RESERVE:
                self._on_peer_reserve(message.sender, payload, now_ms)
            elif message.type is MessageType.COMPLETE:
                self._on_peer_complete(message.sender, payload.task_id, now_ms, result)
            elif message.type is MessageType.ANNOUNCE:
                self._on_announce(payload, message.timestamp_ms, result)
            elif message.type is MessageType.BID:
                auctioneer.on_bid(
                    payload.task_id, message.sender, payload.bid_value, payload.is_idle
                )
            elif message.type is MessageType.CLAIM:
                note = auctioneer.on_claim(
                    payload.task_id, message.sender, payload.bid_value, now_ms
                )
                result.notes.append(note)
                self._relinquish_if_lost(payload.task_id, now_ms, result)

    def _emit_intent(self, now_ms: int, result: StepResult) -> None:
        """Broadcast INTENT every 200 ms (FR-1.1, tolerance +/-20 ms).

        The continuous heartbeat of the whole protocol. Phase 6 consumes it to build
        reservation tables; it exists already because FR-4.8's loss tolerance
        depends on it -- see ``Intent.held_task_id``.

        A robot that stops sending INTENT is indistinguishable from a dead one after
        PEER_TIMEOUT_MS (FR-1.5), so this is emitted in *every* state including
        IDLE, CHARGING and FAULT. Appendix A's broadcasting column says the same.
        """
        if now_ms - self._last_intent_ms < config.INTENT_PERIOD_MS:
            return
        self._last_intent_ms = now_ms

        # Repeat COMPLETE for a few periods after finishing a task.
        #
        # COMPLETE is one-shot, and a peer that missed it has no way to learn the work
        # is done: INTENT's held_task_id goes to -1 on completion, so silence about a
        # task is indistinguishable from "still mine". A duplicate holder then delivers
        # the same goods again -- measured at 30% loss as 10 completions for 9 tasks.
        #
        # This is opportunistic repetition, not the acknowledgement IF-4.5 forbids:
        # nothing waits for a reply and correctness does not depend on any single frame
        # arriving. It is the same discipline that makes RESERVE and INTENT loss
        # tolerant.
        for task_id in list(self._completion_echo):
            remaining = self._completion_echo[task_id] - 1
            result.outbox.append(Complete(task_id=task_id))
            if remaining <= 0:
                del self._completion_echo[task_id]
            else:
                self._completion_echo[task_id] = remaining

        horizon = self.remaining_route[1 : 1 + config.INTENT_HORIZON]
        etas: list[int] = []
        running = self._remaining_on_current_edge_ms()
        previous = self.next_node if self.next_node is not None else self.current_node
        for node in horizon:
            if node != previous:
                edge = self.graph.edge_between(previous, node)
                running += self.graph.nominal_cost_ms(edge) if edge is not None else 0
            etas.append(running)
            previous = node

        held = self.queue.current
        result.outbox.append(
            Intent(
                current_node=self.current_node,
                next_nodes=tuple(horizon),
                eta_ms=tuple(etas),
                priority=self.task_priority,
                state=_STATE_CODES[self.state],
                battery_pct=self.battery_pct,
                pheromone=0,  # Phase 8 fills this from the traffic model
                held_task_id=held.task_id if held is not None else -1,
            )
        )

    def _remaining_on_current_edge_ms(self) -> int:
        """Time still to run on the edge being traversed, for the first ETA."""
        if self.edge_id is None:
            return 0
        length = max(1, self.graph.length_mm(self.edge_id))
        left = max(0, length - self.progress_mm)
        return self.graph.nominal_cost_ms(self.edge_id) * left // length

    def _on_peer_intent(
        self,
        sender: int,
        payload: Intent,
        sent_at_ms: int,
        now_ms: int,
        result: StepResult,
    ) -> None:
        """React to a peer's heartbeat.

        FR-1.8: one received INTENT updates the reservation table *and* the peer
        table from the same frame, with no second message required. Phase 8 adds the
        traffic-model update from the same call, which is the third consumer.
        """
        self.peers.observe(
            robot_id=sender,
            now_ms=now_ms,
            current_node=payload.current_node,
            priority=payload.priority,
            state=payload.state,
            battery_pct=payload.battery_pct,
            held_task_id=payload.held_task_id,
        )
        if self.arbiter is not None:
            self.reservations.record_intent(
                robot_id=sender,
                priority=payload.priority,
                junctions=payload.next_nodes,
                eta_ms=payload.eta_ms,
                sent_at_ms=sent_at_ms,
                now_ms=now_ms,
                is_junction=lambda node: self.graph.node(node).is_junction,
                current_node=payload.current_node,
                corridor_of=self._single_lane_edge_between,
            )

        held = payload.held_task_id
        if held < 0 or not self.queue.holds(held):
            return
        if sender >= self.robot_id:
            return  # we outrank them; they will let go when they hear our INTENT
        if self.auctioneer is not None:
            self.auctioneer.settled[held] = sender
        # Hand it back to the mesh. The peer is doing it, so an announcement is
        # redundant -- but the gateway suppresses announcements for any task it hears
        # an INTENT holding, so redundancy costs one frame at worst. Dropping it
        # silently would lose the task outright if this heal turned out to be wrong,
        # and a lost task is far worse than a wasted ANNOUNCE.
        self._drop_task(held, now_ms, hand_back=True)
        result.notes.append(
            f"r{sender} also holds task {held} and outranks me; relinquishing "
            f"(FR-4.8, healed via INTENT)"
        )

    def _on_peer_complete(
        self, sender: int, task_id: int, now_ms: int, result: StepResult
    ) -> None:
        """A peer finished a task. Drop it if we somehow hold it too.

        This closes the last hole in FR-4.8's loss tolerance. INTENT healing covers a
        duplicate holding while both robots are still working, but a peer that
        *finished* the job before we ever heard an INTENT mentioning it would leave
        us doing completed work -- which showed up as 9 completions for 8 distinct
        tasks at 30% packet loss. The work is already done; carrying on would
        deliver the same goods twice.
        """
        if task_id < 0 or not self.queue.holds(task_id):
            return
        if self.auctioneer is not None:
            self.auctioneer.settled[task_id] = sender
        # Not handed back: the work is done, so announcing it again would have a
        # third robot deliver the same goods.
        self._drop_task(task_id, now_ms, hand_back=False)
        result.notes.append(
            f"r{sender} already completed task {task_id}; dropping my copy"
        )

    def _single_lane_edge_between(self, u: int, v: int) -> int | None:
        """Edge id if ``u -> v`` is a single-lane corridor, else None."""
        try:
            edge_id = self.graph.edge_between(u, v)
        except KeyError:
            return None
        if edge_id is None:
            return None
        return edge_id if self.graph.edge(edge_id).single_lane else None

    def _on_peer_reserve(self, sender: int, payload: Reserve, now_ms: int) -> None:
        """Record a peer's explicit junction claim (FR-5.1, FR-5.5)."""
        if self.arbiter is None:
            return
        self.reservations.record_reserve(
            robot_id=sender,
            priority=payload.priority,
            junction=payload.junction_id,
            window_start_ms=payload.window_start_ms,
            window_end_ms=payload.window_end_ms,
            now_ms=now_ms,
        )

    def _reap_peers(self, now_ms: int, result: StepResult) -> None:
        """Declare silent peers lost and act on it (FR-1.5, FR-6.5).

        Three consequences, all local: expire the peer's reservations so the fleet
        stops avoiding junctions nobody is coming to, forget its sequence state so a
        rebooted robot is not mistaken for a replay, and re-announce the task it was
        holding so the work is not stranded on a dead robot.
        """
        for lost in self.peers.reap(now_ms):
            removed = self.reservations.drop_robot(lost.robot_id)
            result.notes.append(
                f"{lost}; expired {removed} of its reservations (FR-6.5)"
            )
            if lost.held_task_id >= 0 and self.auctioneer is not None:
                # Forget the auction outcome so the task can be won again. Without
                # this the robot would remember it as settled against a robot that no
                # longer exists and never bid on it.
                self.auctioneer.forget(lost.held_task_id)

    def _drop_task(self, task_id: int, now_ms: int, *, hand_back: bool) -> None:
        """Let go of a task.

        ``hand_back`` decides whether it returns to the mesh. Hand back when another
        robot merely *holds* it -- if that belief is wrong the task must not vanish.
        Do not hand back when a peer has *completed* it, or a third robot would win
        it and repeat finished work.
        """
        current = self.queue.current
        if current is not None and current.task_id == task_id:
            dropped = self.queue.pop_current()
            if hand_back and dropped is not None:
                dropped.release(now_ms)
                self.released_tasks.append(dropped)
            self.task_priority = 0
            self.route, self.route_index = [], 0
            self.next_node, self.edge_id, self.progress_mm = None, None, 0
            self.machine.fire_if_possible(Event.TASK_WITHDRAWN, now_ms)
        else:
            removed = self.queue.remove(task_id)
            if hand_back and removed is not None:
                removed.release(now_ms)
                self.released_tasks.append(removed)

    def _on_announce(self, payload: Announce, announced_at_ms: int, result: StepResult) -> None:
        assert self.auctioneer is not None
        task = Task(
            task_id=payload.task_id,
            pickup=payload.pickup_node,
            drop=payload.drop_node,
            priority=payload.priority,
            created_at_ms=payload.created_at_ms,
            zone_id=payload.zone_id,
        )
        if self.auctioneer.on_announce(task, announced_at_ms) is not None:
            result.notes.append(f"heard ANNOUNCE for task {task.task_id}")

    def _relinquish_if_lost(self, task_id: int, now_ms: int, result: StepResult) -> None:
        """FR-4.8: let go of a task a lower robot_id also claimed.

        Reachable in ordinary operation, not only under attack: one lost BID is
        enough to make two robots compute different winners (TC-9).
        """
        assert self.auctioneer is not None
        if not self.queue.holds(task_id):
            return
        if not self.auctioneer.must_relinquish(task_id):
            return
        holding_current = self.queue.current is not None and self.queue.current.task_id == task_id
        if holding_current:
            self._release_current(now_ms, unreachable=False)
            # The state machine must be told, or the robot stays in MOVING holding
            # nothing and never reports itself finished. That is how this presented:
            # every task complete, one robot stuck mid-route forever.
            self.machine.fire_if_possible(Event.TASK_WITHDRAWN, now_ms)
        else:
            removed = self.queue.remove(task_id)
            if removed is not None:
                removed.release(now_ms)
                self.released_tasks.append(removed)
        self.auctioneer.relinquished += 1
        result.notes.append(f"relinquished task {task_id} to a lower robot_id (FR-4.8)")

    def _advance_auctions(self, now_ms: int, result: StepResult) -> None:
        """Bid, and decide auctions whose window has closed.

        There is no auctioneer in the protocol (FR-4.6): this robot sorts the bids
        it happened to hear and reaches its own conclusion. Every robot does the
        same, and they agree whenever nothing was lost.
        """
        auctioneer = self.auctioneer
        assert auctioneer is not None

        for auction in sorted(
            auctioneer.open_auctions.values(), key=lambda a: a.task.task_id
        ):
            if auction.own_bid is not None or auction.settled:
                continue
            if auction.is_window_closed(now_ms):
                continue  # too late to bid; the window is measured from ANNOUNCE

            allowed, reason = auctioneer.may_bid(
                queue_length=len(self.queue),
                battery_pct=self.battery_pct,
                zone_eligible=(
                    True if self.zone_eligible is None
                    else self.zone_eligible(auction.task.zone_id)
                ),
                faulted=self.state is State.FAULT,
            )
            if not allowed:
                result.notes.append(f"not bidding on task {auction.task.task_id}: {reason}")
                auction.settled = True  # nothing more to do with it here
                continue

            bid = auctioneer.price(
                auction.task,
                start_node=self.current_node,
                queue=self.queue,
                battery_pct=self.battery_pct,
                travel_cost=self.planner.travel_cost,
                now_ms=now_ms,
                is_idle=self.is_idle,
            )
            if bid is None:
                result.notes.append(
                    f"not bidding on task {auction.task.task_id}: unroutable for me"
                )
                auction.settled = True
                continue

            auction.own_bid = bid
            auction.record_own(bid)
            auctioneer.bids_placed += 1
            self.metrics.bids_sent += 1
            result.outbox.append(
                Bid(task_id=bid.task_id, bid_value=bid.value, is_idle=bid.is_idle)
            )
            result.notes.append(str(bid))

        for auction in auctioneer.closing(now_ms):
            winner = auction.winner()
            if winner == NO_WINNER:
                # FR-4.13: nobody bid. Hand it back so it is re-announced with its
                # aging term still accruing.
                auctioneer.conclude(auction.task.task_id, NO_WINNER)
                continue
            if winner != self.robot_id:
                auctioneer.conclude(auction.task.task_id, winner)
                result.notes.append(
                    f"task {auction.task.task_id} lost to r{winner}"
                )
                continue
            if self.queue.is_full:
                # Won, but filled up since bidding. Decline rather than exceed the
                # cap: BR-3 is a hard limit, and the insertion cost this robot
                # quoted no longer describes the plan it holds.
                auctioneer.conclude(auction.task.task_id, NO_WINNER)
                result.notes.append(
                    f"won task {auction.task.task_id} but queue filled; declining"
                )
                continue

            bid = auction.own_bid
            assert bid is not None
            result.outbox.append(Claim(task_id=bid.task_id, bid_value=bid.value))
            # Record our own claim before concluding. A robot does not hear its own
            # broadcast, so without this its auctioneer holds no claim at all, and
            # the first peer CLAIM to arrive looks like the winner -- making even the
            # lowest-id robot relinquish work it had correctly won.
            auctioneer.on_claim(bid.task_id, self.robot_id, bid.value, now_ms)
            self.accept_task(
                auction.task, now_ms, bid=bid.value, at_index=bid.insert_at
            )
            auctioneer.conclude(auction.task.task_id, self.robot_id)
            result.notes.append(f"WON task {bid.task_id} at {bid.value}; claiming")

        for auction in auctioneer.overdue_claims(now_ms):
            # FR-4.12: the winner never claimed. The runner-up re-announces, so
            # responsibility is fixed rather than left to whoever notices first --
            # which would produce a burst of duplicate announcements.
            if auction.runner_up() != self.robot_id:
                continue
            auctioneer.conclude(auction.task.task_id, NO_WINNER)
            result.outbox.append(
                Announce(
                    task_id=auction.task.task_id,
                    pickup_node=auction.task.pickup,
                    drop_node=auction.task.drop,
                    priority=auction.task.priority,
                    created_at_ms=auction.task.created_at_ms,
                    zone_id=auction.task.zone_id,
                )
            )
            result.notes.append(
                f"winner of task {auction.task.task_id} never claimed; "
                f"re-announcing as runner-up (FR-4.12)"
            )

    def _step_idle(self, now_ms: int, result: StepResult) -> None:
        if self.queue:
            self.machine.fire(Event.QUEUE_HAS_WORK, now_ms)
            result.notes.append(f"queued work: {self.queue.current}")
            return

        # FR-6.7 / BR-4 / Appendix A: below reserve, with held work complete, go and
        # charge. Checked here rather than mid-task because BR-4 is explicit that an
        # AMR below reserve finishes work it already holds.
        if self.battery_pct < config.BATTERY_RESERVE_PCT:
            if self.machine.fire_if_possible(Event.BATTERY_LOW, now_ms):
                result.notes.append(
                    f"battery {self.battery_pct}% below reserve; routing to a charger"
                )
            return

        self._vacate_junction(now_ms, result)

    def _step_charging(self, now_ms: int, result: StepResult) -> None:
        """Drive to a charger, then charge until fit to work again.

        Appendix A has CHARGING entered when charge falls below reserve and held work
        is complete, and left when charge is restored above threshold. Nothing
        implemented the middle of that, so the state was unreachable and a flat robot
        simply stopped for good: on a 24-task run every robot reached 0%, refused new
        work, and the run stalled with tasks unallocated. It looked like a coordination
        deadlock and was not one.
        """
        if self.battery_pct >= config.BATTERY_RESUME_PCT:
            self.machine.fire(Event.CHARGED, now_ms)
            result.notes.append(f"charged to {self.battery_pct}%; returning to service")
            return

        chargers = self.graph.chargers
        if not chargers:
            return  # nowhere to go; hold and hope for an operator (FR-6.9 RESTRICTED)

        if self.current_node in chargers and self.next_node is None:
            self._charge_accumulator_ms += config.MOTION_TICK_MS
            gained, self._charge_accumulator_ms = divmod(
                self._charge_accumulator_ms, config.BATTERY_CHARGE_MS_PER_PERCENT
            )
            if gained:
                self.battery_pct = min(100, self.battery_pct + gained)
            return

        if self.next_node is None:
            target = min(
                chargers,
                key=lambda node: (self.planner.travel_cost(self.current_node, node), node),
            )
            route = self.planner.route(self.current_node, target)
            if route is None or len(route) < 2:
                return
            self._adopt_route(route)
            result.notes.append(f"heading to charger {target}")

        if self.next_node is not None:
            speed = self._limit_for_clearance(config.NOMINAL_SPEED_MM_S, result)
            result.command = MotionCommand(speed_mm_s=speed, target_node=self.next_node)

    def _vacate_junction(self, now_ms: int, result: StepResult) -> None:
        """Move off a junction when idle, and keep moving until clear.

        **Not in the SRS, and it has to be.** Appendix A has IDLE broadcasting INTENT
        and leaving only on winning an auction; nothing says where a robot idles. An
        idle robot standing on a junction is a permanent obstacle: peers approaching
        it stop at their following distance (IF-2.4) and wait for a robot that has no
        reason to move, forever.

        That is a genuine deadlock and Appendix C does not cover it. The proof is
        about a set of AMRs "in mutual conflict" over junctions and shows exactly one
        does not yield -- but a robot with no task is not a contender at all, so the
        total order never ranks it and it never yields to anyone. Observed directly:
        two robots held 1200 mm short of L_MID while a third sat on it, idle, and the
        run never finished.

        Real warehouses solve this with staging areas, and the map already has the
        nodes for it -- depots and chargers are marked non-junction precisely because
        nothing crosses there. So an idle robot withdraws to the nearest one.
        """
        if not self.graph.node(self.current_node).is_junction:
            return  # already parked somewhere harmless

        # The test is "nothing left to follow", not "no route". A completed route
        # leaves a stale single-node remainder behind, which is non-empty and would
        # make an idle robot decide it was already on its way somewhere. It then sat
        # on the junction indefinitely with two robots queued behind it.
        if self.next_node is None:
            parking = self._nearest_parking_node()
            if parking is None or parking == self.current_node:
                return
            route = self.planner.route(self.current_node, parking)
            if route is None or len(route) < 2:
                return
            self._adopt_route(route)
            result.notes.append(
                f"idle on junction {self.current_node}; withdrawing to {parking}"
            )

        if self.next_node is None:
            return
        speed = self._limit_for_clearance(config.NOMINAL_SPEED_MM_S, result)
        result.command = MotionCommand(speed_mm_s=speed, target_node=self.next_node)

    def _nearest_parking_node(self) -> int | None:
        """Closest free bay where standing still obstructs nobody.

        Occupancy comes from the peer table, so a robot avoids a bay another robot is
        already in or heading for. Simply taking the nearest bay concentrates the
        whole idle fleet on one node, and a bay is a dead-end spur that holds one --
        the queue behind it then blocks the aisle, including for a robot whose task is
        at that very node. Observed as a jam at the depot with two robots stacked on
        the approach.

        Falls back to the nearest bay when all are taken: standing in a queue for a
        bay is still better than standing on a junction.
        """
        bays = list(self.graph.parking_nodes)
        if not bays:
            return None
        taken = {peer.current_node for peer in self.peers.peers.values()}
        taken |= {
            reservation.junction
            for junction_claims in self.reservations.entries.values()
            for reservation in junction_claims.values()
        }
        free = [bay for bay in bays if bay not in taken]
        pool = free or bays
        return min(
            pool,
            key=lambda node: (self.planner.travel_cost(self.current_node, node), node),
        )

    def _step_planning(self, now_ms: int, result: StepResult) -> None:
        task = self.task
        if task is None:
            # Nothing to plan for. Can happen if a task was withdrawn during the
            # same tick the robot entered PLANNING.
            self.machine.fire(Event.ROUTE_UNREACHABLE, now_ms)
            result.notes.append("nothing to plan for; returning to IDLE")
            return

        goal = task.target_node
        route = self.planner.route(self.current_node, goal)
        if route is None:
            # FR-3.7: declare UNREACHABLE and hand the task back to the mesh
            # (FR-6.2) rather than retrying forever on a map that has changed.
            self._release_current(now_ms, unreachable=True)
            self.machine.fire(Event.ROUTE_UNREACHABLE, now_ms)
            result.notes.append(f"UNREACHABLE: no route {self.current_node}->{goal}")
            return

        self._adopt_route(route)
        self.task_priority = task.priority
        task.begin()
        self.machine.fire(Event.ROUTE_READY, now_ms)
        result.notes.append(
            f"route to {goal} ({task.leg.value}): {'->'.join(map(str, route))}"
        )

    def _step_moving(self, now_ms: int, result: StepResult) -> None:
        if self.next_node is None:
            # At the end of the route. Either a leg is finished or the drop is
            # reached; _arrive handles the distinction, so reaching here with no
            # next node means the route was empty.
            self._finish_leg(now_ms, result)
            return

        speed = config.NOMINAL_SPEED_MM_S
        if self.arbiter is not None:
            corridor_speed = self._arbitrate_corridor(now_ms, result)
            if corridor_speed is not None:
                result.command = MotionCommand(
                    speed_mm_s=corridor_speed, target_node=self.next_node
                )
                return
            speed = self._arbitrate_ahead(now_ms, result)
        elif self.state is State.YIELD:
            speed = config.YIELD_SPEED_MM_S

        speed = self._limit_for_clearance(speed, result)
        result.command = MotionCommand(speed_mm_s=speed, target_node=self.next_node)

    def _limit_for_clearance(self, speed: int, result: StepResult) -> int:
        """Cap speed for what the forward sensor sees (IF-2.4, FR-6.6).

        Reactive and unconditional: it applies whatever arbitration decided, because
        a robot that has won a junction still must not drive into the back of one
        that has not moved. This is the one place emergency braking *is* the right
        answer -- FR-6.6 reserves it for obstacles, and something physically in the
        way is an obstacle whether or not it is a fleet member.
        """
        if self.forward_clearance_mm >= config.FOLLOWING_DISTANCE_MM * 2:
            return speed
        if self.forward_clearance_mm <= config.FOLLOWING_DISTANCE_MM:
            if speed > 0:
                result.notes.append(
                    f"holding: {self.forward_clearance_mm} mm clearance ahead"
                )
            return 0
        return min(speed, config.YIELD_SPEED_MM_S)

    def _arbitrate_corridor(self, now_ms: int, result: StepResult) -> int | None:
        """FR-5.10: decide at the last passing point before a single-lane corridor.

        Returns a speed to command if the corridor governs this tick, or None to fall
        through to ordinary junction arbitration.

        This is checked *only at the entry node*, with the robot not yet committed.
        Once inside there is no alternative edge and no room to pass, so a decision
        taken later has no outcome available but a head-on standoff -- which is
        exactly what FR-5.10 exists to forbid and TC-2 tests.

        Junction arbitration cannot substitute. Two robots entering from opposite
        ends each want the *far* junction, so they never contend for the same one and
        both proceed. With junction arbitration alone every remaining collision on
        the benchmark map was a head-on in the choke corridor.
        """
        assert self.arbiter is not None
        corridor = self._corridor_ahead()
        if corridor is None:
            return None

        traverse_ms = self.graph.nominal_cost_ms(corridor)
        entry_ms = 0 if corridor == self.edge_id else self._remaining_on_current_edge_ms()
        window = Window(
            now_ms + entry_ms,
            now_ms + entry_ms + traverse_ms + config.JUNCTION_OCCUPANCY_MS,
        )
        # The conflict set spans the corridor *and* both its endpoint junctions.
        #
        # This is what keeps Appendix C's proof applicable. The proof is about one set
        # of AMRs in mutual conflict under one total order, and shows it has a unique
        # maximum. Rank the corridor and its junctions separately and that premise
        # fails: observed directly, r1 yielded junction C_WEST to r3 while r3 yielded
        # the corridor to r1 -- a cyclic wait between two robots, each correctly
        # applying the total order to a different resource. Neither ever moved.
        #
        # Treating the corridor and the junctions it joins as a single resource
        # restores a single conflict set, and with it the unique maximum.
        edge = self.graph.edge(corridor)
        # Which end we enter by. Already on the corridor: the node behind us.
        # Approaching it: the node we are heading for, which is its mouth.
        entry_node = self.current_node if corridor == self.edge_id else self.next_node
        if entry_node not in (edge.u, edge.v):
            entry_node = -1  # unknown; fall back to treating every claim as opposing
        conflicts = self.reservations.corridor_conflicts(
            corridor, window, exclude_robot=self.robot_id, from_node=entry_node
        )
        route = self.remaining_route
        exit_node = edge.other_end(entry_node) if entry_node >= 0 else edge.v
        beyond = route[2] if len(route) >= 3 else -1

        # Which junctions still lie ahead. Once on the corridor the entry junction is
        # behind us and must be left out: asking about a node already passed compares
        # our position against robots still approaching it, and they legitimately
        # outrank us for a junction we no longer want. That put a robot inside the
        # corridor into a permanent yield to a robot queued behind it.
        endpoints: list[tuple[int, int, int]] = []
        if corridor != self.edge_id and entry_node >= 0:
            endpoints.append((entry_node, self.current_node, exit_node))
            beyond = route[3] if len(route) >= 4 else -1
        endpoints.append((exit_node, entry_node, beyond))

        for endpoint, entering_from, leaving_to in endpoints:
            conflicts += self.reservations.conflicts(
                endpoint,
                window,
                exclude_robot=self.robot_id,
                from_node=entering_from,
                to_node=leaving_to,
            )
        # NOTE (open, Phase 7): a *ring* of distinct single-lane corridors can still
        # deadlock, and Appendix C's proof does not cover it. The proof concerns one
        # conflict set under one total order and shows it has a unique maximum; a cycle
        # of separate corridors defeats that, because each robot is the rightful winner
        # of the segment it wants while being blocked by a different robot holding the
        # next. The loop map provokes it and does not clear.
        #
        # The fix is sequence-level reservation -- hold the whole contended chain before
        # entering any of it, per [R8] -- not a local test. A "do not enter unless your
        # exit is clear" rule was tried and is too blunt: it also refuses entry behind a
        # peer that is about to leave, which cost bench3 three of six runs.
        if not conflicts:
            if self.state is State.YIELD:
                self.machine.fire_if_possible(Event.CONFLICT_CLEARED, now_ms)
                result.notes.append(f"corridor e{corridor} clear; entering")
            return self._limit_for_clearance(config.NOMINAL_SPEED_MM_S, result)

        rival = max(conflicts, key=lambda held: held.ranking_key())
        if outranks(self.task_priority, self.robot_id, rival.priority, rival.robot_id):
            if self.state is State.YIELD:
                self.machine.fire_if_possible(Event.CONFLICT_CLEARED, now_ms)
            self.metrics.yields_won += 1
            result.outbox.append(
                Reserve(
                    junction_id=self.next_node,
                    window_start_ms=window.start_ms,
                    window_end_ms=window.end_ms,
                    priority=self.task_priority,
                )
            )
            # Returning a speed rather than None deliberately: this decision governs
            # the tick. Falling through to junction arbitration would rank the same
            # contenders again over a different resource, which is precisely how the
            # cyclic wait above arises.
            return self._limit_for_clearance(config.NOMINAL_SPEED_MM_S, result)

        # Lost. Hold short of the corridor -- the last passing point. Not emergency
        # braking: the robot has not entered, and stopping short of a corridor is
        # what a human driver does at a passing place.
        if self.state is not State.YIELD:
            if self.machine.fire_if_possible(Event.CONFLICT_LOST, now_ms):
                self.metrics.yields_lost += 1
                result.notes.append(
                    f"corridor e{corridor} held by r{rival.robot_id} "
                    f"(p{rival.priority}); waiting at the last passing point (FR-5.10)"
                )
        return 0

    def _corridor_ahead(self) -> int | None:
        """The single-lane corridor this robot is about to enter, if any.

        Looks one step further than the current edge, because FR-5.10 puts the
        decision at the last passing point *preceding* the corridor -- the node
        before it, where an alternative edge still exists. Deciding once already
        inside leaves no outcome available but a head-on standoff.

        Returns None once committed: past ENTRY_COMMIT_MM there is nothing to decide,
        and continuing to arbitrate would have a robot stop dead in a corridor it has
        already entered, blocking it for everyone.
        """
        if self.edge_id is None:
            return None
        if self.graph.edge(self.edge_id).single_lane:
            return None if self.progress_mm > config.ENTRY_COMMIT_MM else self.edge_id
        remaining = self.remaining_route
        if len(remaining) >= 3:
            return self._single_lane_edge_between(remaining[1], remaining[2])
        return None

    def _arbitrate_ahead(self, now_ms: int, result: StepResult) -> int:
        """Run Appendix B for the junction ahead, and return the speed to command.

        Re-run every tick rather than once on approach. The reservation table changes
        as peers broadcast, so a decision taken 200 ms ago may no longer hold -- and
        Appendix B's AVOID step explicitly says to recompute ETAs and retry DETECT
        after shedding speed.
        """
        assert self.arbiter is not None
        junction = self.next_node
        assert junction is not None

        if not self.graph.node(junction).is_junction:
            return config.NOMINAL_SPEED_MM_S  # nothing crosses here

        arrival_ms = now_ms + self._remaining_on_current_edge_ms()
        if arrival_ms - now_ms > config.ARBITRATION_LOOKAHEAD_MS:
            return config.NOMINAL_SPEED_MM_S  # too far away to matter yet

        window = crossing_window(arrival_ms=arrival_ms)
        remaining = self.remaining_route
        decision = self.arbiter.arbitrate(
            junction=junction,
            window=window,
            my_priority=self.task_priority,
            table=self.reservations,
            from_node=self.current_node,
            to_node=remaining[2] if len(remaining) >= 3 else -1,
            can_absorb_shift=lambda shift: self._can_absorb(shift),
            has_alternative_route=self._has_alternative_route(junction),
            position_confident=self.position_confident,
        )

        if decision.outcome is Outcome.RESERVE:
            # FR-5.5: broadcast the claim before entering. Sent every tick while the
            # conflict stands, not once: RESERVE is one-shot on a lossy radio, and a
            # peer that missed it would keep believing the junction contested and
            # yield unnecessarily. Repetition is how every other guarantee here is
            # made loss-tolerant.
            result.outbox.append(
                Reserve(
                    junction_id=junction,
                    window_start_ms=window.start_ms,
                    window_end_ms=window.end_ms,
                    priority=self.task_priority,
                )
            )
            self.metrics.yields_won += 1

        if decision.outcome.may_enter:
            if self.state is State.YIELD:
                self.machine.fire_if_possible(Event.CONFLICT_CLEARED, now_ms)
                result.notes.append(f"conflict cleared at J{junction}; resuming")
            return config.NOMINAL_SPEED_MM_S

        # ---- yielding ------------------------------------------------------
        if self.state is not State.YIELD:
            if self.machine.fire_if_possible(Event.CONFLICT_LOST, now_ms):
                self.metrics.yields_lost += 1
                result.notes.append(str(decision))
        self.yielding_until_ms = max(
            self.yielding_until_ms, window.start_ms + decision.shift_ms
        )

        if decision.outcome is Outcome.YIELD_REPLAN:
            # Appendix B's second AVOID branch: mark the edge temporarily costly and
            # replan. The graph is not modified -- the aisle is passable, just
            # contended -- so the cost penalty lives in the robot's own view.
            if self.edge_id is not None:
                self._avoid_edges.add(self.edge_id)
            if self.machine.fire_if_possible(Event.EDGE_BLOCKED, now_ms):
                result.notes.append(
                    f"J{junction} contested and unabsorbable; rerouting"
                )
            return config.YIELD_SPEED_MM_S

        if decision.outcome is Outcome.YIELD_SLOW:
            # Shed speed while there is room, then hold at the line. Crawling all the
            # way in would keep closing on the junction, and two robots converging on
            # one node from different aisles touch before either has entered it.
            if self._distance_to_next_node_mm() <= config.JUNCTION_CLEARANCE_MM:
                return 0
            return config.YIELD_SPEED_MM_S

        # YIELD_HOLD or BLOCKED_LOW_CONFIDENCE: stop short of the junction. Not the
        # emergency braking FR-5.6 forbids -- that rules out hard braking as the
        # normal means of resolution, and this is reached only after anticipation has
        # been tried and found insufficient.
        return 0

    def _distance_to_next_node_mm(self) -> int:
        if self.edge_id is None:
            return 0
        return max(0, self.graph.length_mm(self.edge_id) - self.progress_mm)

    def _can_absorb(self, shift_ms: int) -> bool:
        """Whether shedding speed can delay arrival by ``shift_ms`` in time.

        Slowing from nominal to the yield speed stretches the remaining approach by
        the ratio between them. The comparison is against the time left *before the
        junction*, which is FR-5.10's last passing point for an approach with no
        alternative: once there, nothing can be absorbed any more.
        """
        if self.edge_id is None:
            return False
        remaining_ms = self._remaining_on_current_edge_ms()
        if remaining_ms <= 0:
            return False
        # Integer arithmetic only (CON-6): at YIELD_SPEED the same distance takes
        # remaining * NOMINAL / YIELD, so the delay available is the difference.
        stretched_ms = remaining_ms * config.NOMINAL_SPEED_MM_S // config.YIELD_SPEED_MM_S
        return (stretched_ms - remaining_ms) >= shift_ms

    def _has_alternative_route(self, junction: int) -> bool:
        """Whether a route to the objective exists that avoids ``junction``.

        Cheap structural test rather than a replan: if the node the robot is standing
        on has another way out, an alternative is plausible and the planner will
        settle it. A full search here would run inside every tick of every approach
        and blow the FR-3.5 budget for no gain.
        """
        if self.edge_id is None:
            return False
        ways_out = [
            edge for _, edge in self.graph.neighbours(self.current_node)
            if edge != self.edge_id
        ]
        return len(ways_out) > 0 and self.progress_mm == 0

    def _step_replan(self, now_ms: int, result: StepResult) -> None:
        task = self.task
        if task is None:
            self.machine.fire(Event.ROUTE_UNREACHABLE, now_ms)
            return
        route = self.planner.route(self.current_node, task.target_node)
        if route is None:
            self._release_current(now_ms, unreachable=True)
            self.machine.fire(Event.ROUTE_UNREACHABLE, now_ms)
            result.notes.append("UNREACHABLE after repair; returning task to mesh")
            return
        self._adopt_route(route)
        self.metrics.replans += 1
        self.machine.fire(Event.ROUTE_REPAIRED, now_ms)
        result.notes.append(f"repaired route: {'->'.join(map(str, route))}")

    def _step_at_drop(self, now_ms: int, result: StepResult) -> None:
        task = self.queue.pop_current()
        if task is not None:
            task.complete(now_ms)
            self.metrics.tasks_completed += 1
            self.completed_tasks.append(task)
            # Appendix A: AT_DROP broadcasts COMPLETE. Peers need it so a task
            # they heard claimed is not left looking abandoned forever, and a
            # duplicate holder needs it to stop doing finished work.
            result.outbox.append(Complete(task_id=task.task_id))
            self._completion_echo[task.task_id] = self.COMPLETION_ECHOES
            result.notes.append(f"completed {task} in {task.completion_ms()} ms")
        self.task_priority = 0
        self.machine.fire(Event.TASK_REPORTED, now_ms)

    def _release_current(self, now_ms: int, *, unreachable: bool = False) -> None:
        """Hand the current task back to the mesh and stop holding it.

        The task leaves the queue. That is the whole point: a robot still holding
        a task it cannot route to will keep re-planning it every tick and never
        make progress, and no other robot can take it while it is held.
        """
        task = self.queue.pop_current()
        if task is None:
            return
        task.release(now_ms, unreachable=unreachable)
        self.released_tasks.append(task)
        self.task_priority = 0
        self.route, self.route_index = [], 0
        self.next_node, self.edge_id, self.progress_mm = None, None, 0

    def drain_released_tasks(self) -> list[Task]:
        """Take the tasks awaiting re-announcement, clearing the list."""
        released, self.released_tasks = self.released_tasks, []
        return released

    # -- route bookkeeping ---------------------------------------------------

    def _adopt_route(self, route: list[int]) -> None:
        if not route or route[0] != self.current_node:
            raise ValueError(
                f"robot {self.robot_id}: route {route} does not start at the "
                f"robot's current node {self.current_node}"
            )
        self.route = list(route)
        self.route_index = 0
        self._aim_at_next()

    def _aim_at_next(self) -> None:
        """Point the robot at the next node on its route, or stand down."""
        if self.route_index + 1 >= len(self.route):
            self.next_node, self.edge_id, self.progress_mm = None, None, 0
            return
        following = self.route[self.route_index + 1]
        edge = self.graph.edge_between(self.current_node, following)
        if edge is None:
            # The route crossed an edge that has since been blocked. Treated as a
            # blockage rather than an error: FR-6.1 is exactly this case.
            self.next_node, self.edge_id, self.progress_mm = None, None, 0
            return
        self.next_node, self.edge_id, self.progress_mm = following, edge, 0

    def _finish_leg(self, now_ms: int, result: StepResult) -> None:
        """Reached the end of the current route."""
        task = self.task
        if task is None:
            return
        if task.leg is Leg.TO_PICKUP and self.current_node == task.pickup:
            task.reach_pickup(now_ms)
            result.notes.append(f"picked up task {task.task_id} at {task.pickup}")
            # The pickup leg is done, so a route to the drop is needed. PLANNING
            # is where routes come from; see the LEG_COMPLETE note in
            # core/state_machine.py for why this is not an EDGE_BLOCKED.
            self.machine.fire(Event.LEG_COMPLETE, now_ms)
            return
        if self.current_node == task.drop:
            self.machine.fire(Event.ARRIVED_AT_DROP, now_ms)
            result.notes.append(f"arrived at drop {task.drop}")
            return
        # Route ran out short of the objective, which means an edge vanished
        # underneath it. Repair rather than fault.
        self.machine.fire(Event.EDGE_BLOCKED, now_ms)
        result.notes.append("route ended short of the objective; repairing")

    # -- driven by whatever is moving the robot ------------------------------

    def advance(self, distance_mm: int, elapsed_ms: int) -> int | None:
        """Report that the robot moved ``distance_mm`` along its current edge.

        Returns the last node reached this tick, or None if still between
        junctions. Called by the simulator engine, by the Webots controller, or --
        on real hardware -- by odometry.

        Overshoot is carried into the following edge rather than discarded, and
        the carry loops, so a tick long enough to cross a whole edge cannot make
        a robot lose ground. At the nominal 20 ms tick a robot covers 16 mm and
        this never iterates, but the Webots adapter may be handed a much coarser
        step and correctness there should not depend on tick size.
        """
        if self.edge_id is None or self.next_node is None or distance_mm <= 0:
            return None

        self.metrics.moving_ms += elapsed_ms
        self.metrics.distance_mm += distance_mm
        self.progress_mm += distance_mm

        reached: int | None = None
        while self.edge_id is not None and self.next_node is not None:
            length = self.graph.length_mm(self.edge_id)
            if self.progress_mm < length:
                break
            carry = self.progress_mm - length
            reached = self.next_node
            self.metrics.edges_used.add(self.edge_id)
            self._arrive(reached)
            self.progress_mm = carry
        return reached

    def _arrive(self, node_id: int) -> None:
        self.current_node = node_id
        if self.route_index + 1 < len(self.route) and self.route[self.route_index + 1] == node_id:
            self.route_index += 1
        self._aim_at_next()

    def hold(self, elapsed_ms: int) -> None:
        """Report that the robot did not move this tick (FR-10.3)."""
        self.metrics.stopped_ms += elapsed_ms

    def drain_battery(self, distance_mm: int) -> None:
        """Spend charge for distance travelled.

        Accumulates the remainder rather than discarding it, so discharge is
        independent of tick size: 16 mm per tick would otherwise round to zero
        drain every tick and the battery would never move.
        """
        self._drain_accumulator_mm += distance_mm
        spent, self._drain_accumulator_mm = divmod(
            self._drain_accumulator_mm, config.BATTERY_MM_PER_PERCENT
        )
        if spent:
            self.battery_pct = max(0, self.battery_pct - spent)

    # -- exceptions ----------------------------------------------------------

    def on_edge_blocked(self, edge_id: int, now_ms: int) -> bool:
        """FR-6.1: an edge on the committed route became impassable.

        Returns whether this robot's own route was affected. A robot that hears
        about a blockage it was not going to use records it and carries on -- the
        graph is already updated, so its next plan avoids the edge anyway.
        """
        if edge_id not in self.remaining_edges():
            return False
        return self.machine.fire_if_possible(Event.EDGE_BLOCKED, now_ms)

    def remaining_edges(self) -> set[int]:
        """Edges the committed route still intends to use."""
        remaining = self.remaining_route
        return {
            edge
            for index in range(len(remaining) - 1)
            if (edge := self.graph.edge_between(remaining[index], remaining[index + 1]))
            is not None
        }

    def fault(self, now_ms: int, reason: str) -> StepResult:
        """Enter FAULT. Reachable from every working state (Appendix A)."""
        self.machine.fire_if_possible(Event.FAULT_DETECTED, now_ms)
        return StepResult(
            command=MotionCommand.hold(), notes=[f"FAULT: {reason}"]
        )

    def __str__(self) -> str:
        where = (
            f"@{self.current_node}"
            if self.edge_id is None
            else f"{self.current_node}->{self.next_node}"
        )
        return (
            f"r{self.robot_id}[{self.state.value} {where} "
            f"bat{self.battery_pct}% q{len(self.queue)}]"
        )
