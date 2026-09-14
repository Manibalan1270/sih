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
    Path,
)
from core import config
from core.arbitration import Arbiter
from core.auction import NO_WINNER, Auctioneer
from core.graph import Graph
from core.peers import PeerTable
from core.planner_timewindow import Start, TimeWindowPlanner
from core.state_machine import Event, State, StateMachine
from core.task import Leg, Task, TaskQueue, TaskState
from core.timewindows import (
    CORRIDOR,
    LANE,
    REGION,
    STATION,
    Booking,
    ResourceModel,
    ResourceTable,
    RoutePlan,
    Step,
    conflict_between,
    plan_entries,
    plan_from_entries,
    resting_plan,
)


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


@dataclass(frozen=True, slots=True)
class WaitCause:
    """Why a robot is not moving, and who it is waiting on.

    Recorded so a stalled fleet can be *diagnosed* rather than inferred. Reading a
    freeze off a dump of positions and states was tried repeatedly and misdiagnosed
    it twice: the position tells you a robot has stopped, not which other robot it
    stopped for, and without that the wait-for cycle has to be guessed.
    """

    kind: str
    """``headway``, ``junction``, ``corridor`` or ``confidence``."""

    blocker_id: int
    """The robot being waited on, or -1 where the cause is not another robot."""

    resource: str
    """What is being waited for: a junction, a corridor, or the space ahead."""

    detail: str = ""

    def __str__(self) -> str:
        who = f"r{self.blocker_id}" if self.blocker_id >= 0 else "nobody"
        return f"{self.kind} on {self.resource}, waiting on {who}"


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

    plans_committed: int = 0
    plans_failed: int = 0
    """Route bookings made, and attempts that found no window in the horizon."""
    paths_sent: int = 0
    paths_heard: int = 0
    """PATH frames broadcast and peer PATH frames entered in the table -- the
    per-robot side of the frame accounting NFR-1.12 asks for."""
    precedence_hold_ms: int = 0
    """Time spent stopped at a boundary for a robot booked ahead (FR-10.3)."""


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

    bookings: ResourceTable = field(default_factory=ResourceTable)
    """This robot's own view of every peer's booked route (FR-5.1), built from the
    PATH frames it has heard. The planner books against it; execution asks it who
    is ahead."""

    resources: ResourceModel | None = None
    """The map read as bookable resources. Built in __post_init__."""

    route_planner: TimeWindowPlanner | None = None
    """Free-time-window route planner over ``resources``. Built in __post_init__."""

    plan: RoutePlan | None = None
    """The route this robot has booked and announced, or None between plans."""
    plan_seq: int = 0
    """Sequence number of ``plan``; carried in INTENT and PATH so peers can tell a
    fresh plan from a repeat. Zero until the first plan."""
    plan_index: int = 0
    """Steps of ``plan`` this robot has *released* -- physically left. Carried in
    INTENT; what a peer waiting behind this robot reads."""
    _entered_index: int = 0
    """Steps of ``plan`` this robot has entered. Everything before this is under
    the wheels or behind; the next gate is the first capacity-one step at or
    after it."""
    _plan_goal: int = -1
    _replan_requested: bool = False
    _last_path_ms: int = -10_000
    _last_replan_ms: int = -10_000
    _step_pos: list[int] = field(default_factory=list)
    """Route position of each plan step's starting node; see _index_plan_steps."""
    _rest_since_ms: int = -1
    """When the current rest began, so its booking keeps one start across refreshes."""
    _region_entered_ms: dict[int, int] = field(default_factory=dict)
    """node -> when this robot crossed into that node's region, for the opening
    step of a mid-edge replan."""

    peers: PeerTable = field(default_factory=PeerTable)
    """Neighbours heard from within the liveness window (FR-1.4)."""

    arbiter: Arbiter | None = None
    """Junction arbitration (FE-5). None disables it, which is how Configuration A
    runs the same robot object with no coordination at all (FR-10.5)."""

    wait_cause: WaitCause | None = None
    """Why this robot held position on the last tick, or None if it did not.

    Cleared at the top of every ``step`` so it always describes now, never a stale
    reason from an earlier tick."""

    forward_blocker_id: int = -1
    """Whose footprint the forward sensor is reading, from whatever drives the robot.
    A sensor gives a distance; the simulator also knows the identity, and the
    identity is what makes a stall diagnosable."""

    forward_clearance_mm: int = 1 << 30
    """Distance to the nearest obstacle directly ahead, from the forward sensor
    (IF-2.4). Written every tick by whatever is driving the robot -- the simulator
    here, a real sensor on hardware. Defaults to "clear"."""

    position_confident: bool = True
    """FR-5.14 / NFR-2.3. Phase 10 drives this from core/localization.py; until then
    a robot always knows where it is."""

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

    _now_ms: int = 0
    """Latest tick time, so ``edge_cost`` can honour penalty expiry. The planner's
    cost function takes only an edge id, and core/ may not read a wall clock."""

    _last_speed_mm_s: int = 0
    """Speed commanded on the previous tick, so INTENT can declare an honest ETA."""

    _charge_accumulator_ms: int = 0
    """Sub-percent charging time carried between ticks, mirroring the drain side."""

    _avoid_edges: dict[int, int] = field(default_factory=dict)
    """edge_id -> aligned time the penalty expires.

    Edges this robot is routing around because a peer is parked on them (Appendix B's
    AVOID, second branch). Local to the robot and deliberately not a graph blockage:
    the aisle is passable, merely occupied, and marking the graph would make one
    robot's problem the whole fleet's.

    Read by ``edge_cost``, which is the planner's cost function. That wiring is the
    whole mechanism -- without it the penalty is recorded and ignored, the replan
    returns the same route, and the reroute re-triggers every tick. Measured before it
    was connected: 113,612 replans in a single 12-task run."""

    _last_intent_ms: int = -10_000
    """When INTENT was last broadcast. Negative so the first tick emits one."""

    _drain_accumulator_mm: int = 0
    """Sub-percent battery drain carried between ticks. See ``drain_battery``."""

    def __post_init__(self) -> None:
        if self.current_node < 0:
            self.current_node = self.home_node
        self.graph.node(self.current_node)  # fail now if the spawn node is bogus
        if self.resources is None:
            self.resources = ResourceModel(self.graph)
        if self.route_planner is None:
            self.route_planner = TimeWindowPlanner(self.resources)

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
        self.wait_cause = None
        self._now_ms = now_ms

        # A peer's plan lives exactly as long as the peer does (FR-5.11 via
        # FR-1.5): it is replaced by its next PATH and dropped when the peer is
        # reaped. It is never expired by its own timetable -- a robot running late
        # is still on its route, and forgetting its bookings is what let two robots
        # meet head-on in a corridor: the one behind schedule became invisible to
        # precedence, and the one that had waited for it drove in.
        self._reap_peers(now_ms, result)

        if inbox:
            self._handle_inbox(inbox, now_ms, result)
        if self.auctioneer is not None:
            self._advance_auctions(now_ms, result)
            self._emit_intent(now_ms, result)
        if self.arbiter is not None:
            self._repeat_path(now_ms, result)

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
            self.plan, self._plan_goal = None, -1

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
            elif message.type is MessageType.PATH:
                self._on_peer_path(message.sender, payload, message.timestamp_ms, now_ms, result)
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
        # First leg at the speed actually commanded, not the nominal one.
        #
        # FR-1.2 makes eta_ms a declaration of *arrival*, and a robot holding position
        # that advertises a nominal ETA is telling its peers it will clear the node
        # shortly when it will not. They keep approaching on that promise and then have
        # to brake reactively -- which is precisely the stop-and-wait behaviour being
        # removed. Floored at the yield speed so a halted robot declares a late arrival
        # rather than an infinite one, which would overflow the uint16 field.
        running = self._remaining_on_current_edge_ms()
        if self._last_speed_mm_s < config.NOMINAL_SPEED_MM_S:
            honest = max(config.YIELD_SPEED_MM_S, self._last_speed_mm_s)
            running = running * config.NOMINAL_SPEED_MM_S // honest
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
                plan_seq=self.plan_seq,
                plan_index=min(self.plan_index, 0xFF),
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
            next_nodes=payload.next_nodes,
            eta_ms=tuple(sent_at_ms + eta for eta in payload.eta_ms),
            plan_seq=payload.plan_seq,
            plan_index=payload.plan_index,
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

    def _reap_peers(self, now_ms: int, result: StepResult) -> None:
        """Declare silent peers lost and act on it (FR-1.5, FR-6.5).

        Three consequences, all local: expire the peer's reservations so the fleet
        stops avoiding junctions nobody is coming to, forget its sequence state so a
        rebooted robot is not mistaken for a replay, and re-announce the task it was
        holding so the work is not stranded on a dead robot.
        """
        for lost in self.peers.reap(now_ms):
            removed = self.bookings.drop_robot(lost.robot_id)
            result.notes.append(
                f"{lost}; dropped its plan ({removed} bookings) (FR-6.5)"
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
            if self.arbiter is not None and self.resources is not None and self.resources.is_station(self.current_node):
                self._rest_here(now_ms, result)
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
            if not self._go_to(target, now_ms, result):
                return
            result.notes.append(f"heading to charger {target}")

        self._drive_tick(now_ms, result)

    def _vacate_junction(self, now_ms: int, result: StepResult) -> None:
        """Rest in a bay when idle; withdraw to one from anywhere else.

        **Not in the SRS, and it has to be.** Appendix A has IDLE broadcasting INTENT
        and leaving only on winning an auction; nothing says where a robot idles. An
        idle robot standing on a junction is a permanent obstacle, and an idle robot
        standing on a pickup or drop blocks the very work it is waiting for --
        measured at 6 AMRs as a run stuck at 35 of 36 tasks. Only a bay is somewhere
        standing still obstructs nobody, which is what a bay is for. This is the
        idle rule of Token Passing (Ma et al., AAMAS 2017): rest only where no task
        endpoint is, and choose a bay nobody's plan ends at.

        A resting robot keeps its bay booked by repeating a one-node plan, so no
        peer plans into it.
        """
        if self.current_node in self.graph.parking_nodes and self.next_node is None:
            if self.arbiter is not None:
                self._rest_here(now_ms, result)
            return

        # The test is "nothing left to follow", not "no route". A completed route
        # leaves a stale single-node remainder behind, which is non-empty and would
        # make an idle robot decide it was already on its way somewhere.
        if self.next_node is None:
            parking = self._nearest_parking_node()
            if parking is None or parking == self.current_node:
                return
            if not self._go_to(parking, now_ms, result):
                return
            result.notes.append(
                f"idle on {self.current_node}, which is not a bay; withdrawing to {parking}"
            )

        self._drive_tick(now_ms, result)

    def _nearest_parking_node(self) -> int | None:
        """Closest bay nobody is in or coming to.

        Occupancy is what the table says: a bay at the end of any live plan --
        including a resting robot's one-node plan -- is taken. Taking the nearest
        regardless concentrates the whole idle fleet on one spur, and the queue
        behind it blocks the aisle. Falls back to the nearest bay when all are
        taken: standing in a queue for a bay still beats standing on a junction.
        """
        bays = list(self.graph.parking_nodes)
        if not bays:
            return None
        taken = self.bookings.plan_ends(exclude_robot=self.robot_id)
        taken |= {peer.current_node for peer in self.peers.peers.values()}
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

        self.task_priority = task.priority
        if self.arbiter is None:
            self._adopt_route(route)
        elif not self._commit_plan(goal, now_ms, result):
            return  # reachable but no window yet; PLANNING tries again next tick
        task.begin()
        self.machine.fire(Event.ROUTE_READY, now_ms)
        result.notes.append(
            f"route to {goal} ({task.leg.value}): {'->'.join(map(str, self.route))}"
        )

    def _step_moving(self, now_ms: int, result: StepResult) -> None:
        if self.next_node is None:
            # At the end of the route. Either a leg is finished or the drop is
            # reached; _arrive handles the distinction, so reaching here with no
            # next node means the route was empty.
            self._finish_leg(now_ms, result)
            return
        self._drive_tick(now_ms, result)

    def _follow_traffic_ahead(self, now_ms: int, speed: int, result: StepResult) -> int:
        """Keep station behind a peer on the same edge by shedding speed (FR-5.6).

        Anticipatory, from a peer's own INTENT: ``next_nodes`` and ``eta_ms`` say where
        it is going and when it expects to arrive, so a follower can tell it is closing
        on the robot in front *before* either is near enough for a sensor to matter,
        and open the gap by slowing instead of braking.

        Separation is measured in time, not distance, which is the same reasoning
        junction arbitration already uses. A robot whose arrival at the shared node is
        within FOLLOW_GAP_MS of the leader's sheds speed; one comfortably behind runs at
        cruise.

        Why not simply keep the proximity rule: §1.5 *defines* stop-and-wait as halting
        when a peer comes within a fixed radius, and FR-5.6/NFR-2.4 reserve braking for
        non-cooperative obstacles. A fleet peer broadcasts five times a second, so it is
        not one -- and treating it as one put the very behaviour this system exists to
        replace inside Configuration B, accounting for 44% of its hold-time. It also
        made the wait un-negotiable, which is what let a deadlock cycle close through
        it.
        """
        if self.edge_id is None or self.next_node is None or speed <= 0:
            return speed

        my_arrival = now_ms + self._remaining_on_current_edge_ms()
        tightest: int | None = None
        leader = -1
        for peer in self.peers.live(now_ms):
            if not peer.is_on_edge(self.current_node, self.next_node):
                continue
            their_arrival = peer.arrival_at(self.next_node)
            if their_arrival is None:
                continue
            gap = my_arrival - their_arrival
            if gap <= 0:
                continue  # they are behind us, or level: not traffic to follow
            if tightest is None or gap < tightest:
                tightest, leader = gap, peer.robot_id

        # A peer standing *on* the node ahead is the other way traffic blocks a path,
        # and the sensor was catching it reactively. It is visible in INTENT too:
        # current_node says where a peer is, and its declared ETA says whether it is
        # going anywhere soon.
        #
        # Slowing down does not help here -- a crawl still closes on something that is
        # not moving, so the reflex fires anyway, just later. What anticipation actually
        # buys is the *choice* a sensor can never offer: FR-5.6's second remedy, take
        # another edge. That is also what breaks a wait-for cycle, because a robot that
        # reroutes stops being a link in it.
        # Rerouting around a stationary blocker is the planner's job now: a parked
        # peer holds its station in the table, and a fresh plan books around it.
        blocker = self._stationary_blocker_ahead(now_ms, my_arrival)
        if blocker >= 0 and tightest is None:
            tightest, leader = 0, blocker

        if tightest is None or tightest >= config.FOLLOW_GAP_MS:
            return speed

        # Closing too fast. Shed speed to stretch the remaining approach and open the
        # gap. Not a halt: FR-5.6 wants the ETA shifted, and a crawling robot still
        # clears the node behind it -- which is what stops a waiting robot becoming an
        # obstacle to someone else.
        result.notes.append(
            f"following r{leader} by {tightest} ms on e{self.edge_id}; shedding speed"
        )
        return config.YIELD_SPEED_MM_S

    def _stationary_blocker_ahead(self, now_ms: int, my_arrival: int) -> int:
        """A peer sitting on the node we are heading for that will not clear in time.

        Returns its id, or -1. "In time" means its own declared arrival at its next
        node, plus a following gap, lands before we get there -- if it does, it is
        traffic that will have moved on and no action is needed.
        """
        for peer in self.peers.live(now_ms):
            if peer.current_node != self.next_node:
                continue
            if not peer.next_nodes:
                return peer.robot_id  # declared nowhere to go: parked
            clears_at = peer.arrival_at(peer.heading_to)
            if clears_at is None or clears_at + config.FOLLOW_GAP_MS > my_arrival:
                return peer.robot_id
        return -1

    def edge_cost(self, edge_id: int) -> int:
        """The cost this robot plans on: nominal, plus its own temporary penalties.

        Injected into the planner (FR-3.4), so the planner never has to know why an
        edge is expensive. Phase 8's learned traffic model composes into the same
        place, which is why the planner takes a cost function rather than a graph.
        """
        cost = self.graph.nominal_cost_ms(edge_id)
        expires = self._avoid_edges.get(edge_id)
        if expires is not None and expires > self._now_ms:
            cost += config.CONTENDED_EDGE_PENALTY_MS
        return cost

    def _hold_outside_corner(self, result: StepResult) -> bool:
        """Never come to rest inside a junction's corner. Hold at the line instead.

        The rule is standard in lane-annotated multi-agent navigation: lanes are made to
        end a deliberate gap short of an intersection so that a robot stopped at the end
        of one does not interfere with robots moving through the intersection, and a
        robot that cannot cross is directed to stop *outside* the conflict region rather
        than partway into it (Google/Intrinsic, US 11,709,502 B2, "Roadmap annotation for
        deadlock-free multi-agent navigation"). Entry is conditional on being able to
        leave.

        We had the gap -- YIELD_STANDOFF_MM is 200 mm outside JUNCTION_FOOTPRINT_MM --
        but only arbitration used it. The reactive clearance limiter did not: it stops
        dead wherever the sensor reading happens to fall, which on seed 19 was 800 mm
        from node 2, inside the corner. The geometry then works against the robot,
        because a peer *departing* that junction closes on a robot stopped there as it
        leaves -- separation falls to 100 mm at 700 mm past the node. So a robot that
        halts in a corner is not merely obstructing, it is being driven into.

        Deciding here rather than at the exit is what keeps this from being the blunt
        "do not enter unless your exit is clear" rule that was tried and reverted: that
        one refused entry behind a peer about to leave and cost three of six runs. This
        asks a narrower question -- is passage already in doubt, right now, at the last
        point where stopping is still safe -- and once inside the corner it never holds,
        because a committed robot must clear rather than freeze.
        """
        if self.next_node is None or self.edge_id is None:
            return False
        if not self.graph.node(self.next_node).is_junction:
            return False
        if self.graph.edge(self.edge_id).single_lane:
            return False  # inside a corridor the only safe act is to clear it
        if (
            self.resources is not None
            and self.resources.has_region(self.current_node)
            and self.progress_mm < config.JUNCTION_FOOTPRINT_MM
        ):
            return False  # still inside the region behind: clear it before holding
        to_node = self._distance_to_next_node_mm()
        if to_node <= config.JUNCTION_FOOTPRINT_MM:
            return False  # committed: inside the corner, clearing it is the only safe act
        if to_node > config.YIELD_STANDOFF_MM:
            return False  # not at the line yet; approach it normally
        clearance = self.forward_clearance_mm
        if clearance < 0 or clearance > 2 * config.FOLLOWING_DISTANCE_MM:
            return False  # nothing close enough ahead to halt for

        # Where would the reactive limiter bring this robot to rest? It halts once the
        # clearance falls to FOLLOWING_DISTANCE_MM, and clearance and distance-to-node
        # both shrink together as the robot advances, so the halt lands at
        # ``to_node - clearance + FOLLOWING_DISTANCE_MM`` from the node. If that is inside
        # the corner, hold here, outside it, instead.
        #
        # Testing the current clearance against FOLLOWING_DISTANCE_MM does not work and
        # was measured never to fire: with the obstruction at the node, clearance *equals*
        # the distance to the node, and the two thresholds are both 1200 mm -- so
        # clearance reaches the limit on the same tick the robot crosses the boundary,
        # one tick too late. r6 sat at 1196 mm for a second before r3 swept it.
        halt_at_mm = to_node - clearance + config.FOLLOWING_DISTANCE_MM
        if halt_at_mm > config.YIELD_STANDOFF_MM:
            return False  # it would stop clear of the corner on its own
        self.wait_cause = WaitCause(
            kind="corner",
            blocker_id=self.forward_blocker_id,
            resource=f"J{self.next_node}",
            detail=f"{clearance} mm ahead; holding outside the corner",
        )
        result.notes.append(
            f"holding {to_node} mm short of J{self.next_node}: {clearance} mm clearance "
            f"ahead, so the corner cannot be crossed without stopping in it"
        )
        return True

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
            self.wait_cause = WaitCause(
                kind="headway",
                blocker_id=self.forward_blocker_id,
                resource=f"space ahead on e{self.edge_id}",
                detail=f"{self.forward_clearance_mm} mm clearance",
            )
            return 0
        return min(speed, config.YIELD_SPEED_MM_S)

    def _distance_to_next_node_mm(self) -> int:
        if self.edge_id is None:
            return 0
        return max(0, self.graph.length_mm(self.edge_id) - self.progress_mm)

    def _step_replan(self, now_ms: int, result: StepResult) -> None:
        task = self.task
        if task is None:
            self.machine.fire(Event.ROUTE_UNREACHABLE, now_ms)
            return
        if self.planner.route(self.current_node, task.target_node) is None:
            self._release_current(now_ms, unreachable=True)
            self.machine.fire(Event.ROUTE_UNREACHABLE, now_ms)
            result.notes.append("UNREACHABLE after repair; returning task to mesh")
            return
        if self.arbiter is None:
            route = self.planner.route(self.current_node, task.target_node)
            assert route is not None
            self._adopt_route(route)
        elif not self._commit_plan(task.target_node, now_ms, result):
            return  # no window yet; REPLAN tries again next tick
        self.metrics.replans += 1
        self.machine.fire(Event.ROUTE_REPAIRED, now_ms)
        result.notes.append(f"repaired route: {'->'.join(map(str, self.route))}")

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
        self.plan, self._plan_goal = None, -1

    def drain_released_tasks(self) -> list[Task]:
        """Take the tasks awaiting re-announcement, clearing the list."""
        released, self.released_tasks = self.released_tasks, []
        return released

    # =======================================================================
    # Route-level reservation: commit a plan, share it, execute it by precedence
    # =======================================================================
    #
    # A coordinated robot never negotiates one junction at a time. It books its
    # whole route as time windows against every plan it has heard (PATH), tells
    # the fleet (PATH again, repeated), and then drives by *order*: it enters a
    # junction region, a single-lane corridor or a station only once every robot
    # booked ahead of it there has released it, which their INTENT says. Time
    # decided who is ahead; order is what is enforced, so a slow robot delays the
    # robots behind it and never collides with them.

    def _start_for_plan(self, now_ms: int) -> Start:
        """Where a plan begins, as the robot knows itself right now."""
        model = self.resources
        assert model is not None
        if self.edge_id is None or self.next_node is None:
            node = self.current_node
            if model.is_station(node):
                return Start(node, now_ms, waitable=True)
            # Standing on an aisle node. A bend is a lane and waiting is fine; a
            # junction is a region the robot already occupies, so the plan has to
            # cross it from now without waiting.
            return Start(node, now_ms, waitable=not model.has_region(node))
        # Mid-edge: the plan opens with the edge already under the wheels, so
        # peers behind keep FIFO order and nobody books the region just used.
        remaining_ms = self._remaining_on_current_edge_ms()
        to_node = self._distance_to_next_node_mm()
        edge = self.graph.edge(self.edge_id)
        entered_ms = self._region_entered_ms.get(self.current_node, now_ms)
        # What the plan being replaced booked for the steps under the wheels: those
        # times are the robot's place in every queue and must carry over.
        if model.has_region(self.current_node):
            old_region = self._booked_enter(model.region(self.current_node))
            if old_region >= 0:
                entered_ms = min(entered_ms, old_region)
        old_lane = self._booked_enter(model.lane(self.edge_id, self.next_node))
        footprint = config.JUNCTION_FOOTPRINT_MM
        if model.has_region(self.next_node) and to_node <= footprint:
            # Inside the corner ahead already. The plan holds it and resumes from
            # its far boundary, which the robot reaches at nominal speed.
            out_ms = now_ms + (to_node + footprint) * 1000 // config.NOMINAL_SPEED_MM_S
            return Start(
                self.next_node,
                out_ms,
                waitable=False,
                in_region=True,
                prefix_node=self.current_node,
                prefix_ms=entered_ms,
                lane_entered_ms=old_lane,
                region_entered_ms=self._booked_enter(model.region(self.next_node)),
            )
        if model.has_region(self.next_node):
            at_ms = now_ms + (to_node - footprint) * 1000 // config.NOMINAL_SPEED_MM_S
        else:
            at_ms = now_ms + remaining_ms
        return Start(
            self.next_node,
            at_ms,
            waitable=not edge.single_lane,
            prefix_node=self.current_node,
            prefix_ms=entered_ms,
            lane_entered_ms=old_lane,
        )

    def _booked_enter(self, resource) -> int:
        """When the current plan booked ``resource`` among the steps already
        entered, or -1."""
        plan = self.plan
        if plan is None:
            return -1
        for index in range(min(self._entered_index, len(plan.steps))):
            if plan.steps[index].resource == resource:
                return plan.steps[index].enter_ms
        return -1

    def _commit_plan(self, goal: int, now_ms: int, result: StepResult) -> bool:
        """Book a route to ``goal`` against the table and announce it.

        Returns whether a plan now stands. On failure the previous plan, if any,
        stays in force: a robot mid-corridor cannot stop to think, and a robot at a
        station simply tries again next tick.
        """
        planner = self.route_planner
        assert planner is not None and self.resources is not None
        start = self._start_for_plan(now_ms)
        plan = planner.plan(
            self.bookings,
            start=start,
            goal=goal,
            robot_id=self.robot_id,
            priority=self.task_priority,
            plan_seq=(self.plan_seq + 1) & 0xFF or 1,
            committed_ms=now_ms,
        )
        if plan is None:
            self.metrics.plans_failed += 1
            result.notes.append(f"no window to {goal} within the horizon; holding")
            return False
        self.plan = plan
        self.plan_seq = plan.plan_seq
        self._plan_goal = goal
        self.plan_index = 0
        self._entered_index = 0
        self._replan_requested = False
        self.metrics.plans_committed += 1
        self._adopt_plan_route(plan)
        self._index_plan_steps()
        self._sync_plan_progress(adopting=True)
        self._broadcast_path(now_ms, result)
        result.notes.append(f"committed {plan} ({len(plan.steps)} steps)")
        return True

    def _adopt_plan_route(self, plan: RoutePlan) -> None:
        """Follow the plan's nodes. Mid-edge the plan opens with the edge under the
        wheels, so the physical edge and progress are kept and only what follows
        changes; _adopt_route would reset progress to the node behind."""
        nodes = list(plan.nodes)
        if nodes[0] != self.current_node:
            raise ValueError(f"plan {plan} does not start at node {self.current_node}")
        if self.edge_id is not None and self.next_node is not None:
            if len(nodes) < 2 or nodes[1] != self.next_node:
                raise ValueError(
                    f"plan {plan} does not continue the edge {self.current_node}->{self.next_node}"
                )
            self.route, self.route_index = nodes, 0
            return
        self._adopt_route(nodes)

    def _rest_here(self, now_ms: int, result: StepResult) -> None:
        """Hold the station under the robot, so nobody plans into it."""
        assert self.resources is not None
        if self.plan is not None and len(self.plan.nodes) == 1 and self.plan.nodes[0] == self.current_node:
            return
        self.plan_seq = (self.plan_seq + 1) & 0xFF or 1
        self._rest_since_ms = now_ms
        self.plan = resting_plan(
            self.resources, robot_id=self.robot_id, plan_seq=self.plan_seq,
            priority=self.task_priority, since_ms=now_ms, now_ms=now_ms, leaf=self.current_node,
        )
        self._plan_goal = self.current_node
        self.plan_index = 0
        self._entered_index = 0
        self._index_plan_steps()
        self._broadcast_path(now_ms, result)

    def _broadcast_path(self, now_ms: int, result: StepResult) -> None:
        plan = self.plan
        if plan is None or self.resources is None:
            return
        entries = [(node, t - now_ms) for node, t in plan_entries(self.resources, plan)]
        task = self.task
        result.outbox.append(
            Path(
                plan_seq=plan.plan_seq,
                committed_ms=plan.committed_ms,
                priority=plan.priority,
                entries=tuple(entries),
                goal_task_id=task.task_id if task is not None else -1,
            )
        )
        self._last_path_ms = now_ms
        self.metrics.paths_sent += 1

    def _repeat_path(self, now_ms: int, result: StepResult) -> None:
        """PATH heals by repetition (IF-4.5), like INTENT: once a second while a plan
        is in force. A resting plan is refreshed the same way, which is what keeps a
        parked robot's bay booked."""
        if self.plan is None:
            return
        if now_ms - self._last_path_ms < config.PATH_REPEAT_MS:
            return
        if len(self.plan.nodes) == 1 and self.resources is not None:
            self.plan = resting_plan(
                self.resources, robot_id=self.robot_id, plan_seq=self.plan_seq,
                priority=self.task_priority, since_ms=self._rest_since_ms,
                now_ms=now_ms, leaf=self.plan.nodes[0],
            )
        self._broadcast_path(now_ms, result)

    def _on_peer_path(self, sender: int, payload: Path, sent_at_ms: int, now_ms: int, result: StepResult) -> None:
        """Enter a peer's plan in the table and settle any race with our own."""
        if self.resources is None:
            return
        known = self.bookings.plans.get(sender)
        if known is not None and known.plan_seq == payload.plan_seq and known.committed_ms == payload.committed_ms:
            return  # a repeat of a plan already in the table
        entries = [(node, sent_at_ms + offset) for node, offset in payload.entries]
        try:
            theirs = plan_from_entries(
                self.resources,
                robot_id=sender,
                plan_seq=payload.plan_seq,
                priority=payload.priority,
                committed_ms=payload.committed_ms,
                entries=entries,
            )
        except (ValueError, KeyError) as exc:
            result.notes.append(f"ignored PATH from r{sender}: {exc}")
            return
        self.bookings.replace_plan(theirs)
        self.metrics.paths_heard += 1

        mine = self.plan
        if mine is None or self.arbiter is None:
            return
        clash = conflict_between(mine, theirs, self.bookings.margin_ms)
        if clash is None:
            return
        if self.arbiter.resolve(mine, theirs, clash, now_ms):
            # Ours stands. Say so again now rather than in up to a second: the
            # sooner the peer hears it, the sooner it replans.
            if now_ms - self._last_path_ms >= config.INTENT_PERIOD_MS:
                self._broadcast_path(now_ms, result)
            result.notes.append(f"r{sender}'s plan#{payload.plan_seq} clashes on {clash}; mine stands")
            return
        self._replan_requested = True
        result.notes.append(f"r{sender}'s plan#{payload.plan_seq} clashes on {clash} and precedes mine; replanning")

    # -- where am I in my plan ---------------------------------------------------
    #
    # Everything below is indexed by *route position*, never by node id: a plan
    # may visit a node twice (a robot that turns round at a junction goes
    # 10 -> 2 -> 10), and looking a node up by id found the wrong visit -- the
    # corridor ahead read as already behind, and the robot drove into it.

    def _index_plan_steps(self) -> None:
        """For each step of the plan, the route position of the node it starts at."""
        plan = self.plan
        model = self.resources
        self._step_pos = []
        if plan is None or model is None:
            return
        nodes = plan.nodes
        # ``cursor`` is the earliest position the next step may anchor at. A lane
        # or corridor runs from its anchor to the next node, so whatever follows
        # it anchors at least one position on; a region or station is a point,
        # and the lane leaving it anchors where it is. Without the advance, a
        # route that turns round (62 -> 84 -> 62) matched the lane back on the
        # same edge at its first visit, and the second crossing of 62 read as
        # already behind the robot.
        cursor = 0
        for index, step in enumerate(plan.steps):
            res = step.resource
            pos = cursor
            if res.kind == STATION and index == 0 and nodes[0] == res.key:
                pos = 0
            elif res.kind in (REGION, STATION):
                while pos < len(nodes) and nodes[pos] != res.key:
                    pos += 1
            else:
                while pos + 1 < len(nodes) and self.graph.edge_between(nodes[pos], nodes[pos + 1]) != res.key:
                    pos += 1
            if pos >= len(nodes):
                pos = len(nodes) - 1
            self._step_pos.append(pos)
            cursor = pos + 1 if res.kind in (LANE, CORRIDOR) else pos

    def _signed_distance_to_pos_mm(self, pos: int) -> int:
        """Distance along the route to position ``pos``: positive ahead, negative
        behind, zero at the node under the robot."""
        route = self.route
        if pos <= self.route_index:
            back = self.progress_mm
            for i in range(pos, self.route_index):
                edge = self.graph.edge_between(route[i], route[i + 1])
                back += self.graph.length_mm(edge) if edge is not None else 0
            return -back
        ahead = self._distance_to_next_node_mm() if self.edge_id is not None else 0
        for i in range(self.route_index + 1, pos):
            edge = self.graph.edge_between(route[i], route[i + 1])
            ahead += self.graph.length_mm(edge) if edge is not None else 0
        return ahead

    def _node_margin_mm(self, node: int) -> int:
        """Where a resource that starts at ``node`` really begins, relative to it.

        Past the region, if the node has one. A bend has no region, but the two
        directions still meet at the node point, so a robot waiting to enter must
        stand a footprint short of it: waiting 200 mm from a bend into a corridor
        was measured to stop the robot leaving the corridor in its tracks -- inside
        the corridor -- with the waiter then waiting on it.
        """
        assert self.resources is not None
        if self.resources.has_region(node):
            return config.JUNCTION_FOOTPRINT_MM
        return -config.JUNCTION_FOOTPRINT_MM

    def _boundary_distance_mm(self, index: int) -> int:
        """How far ahead the start of step ``index``'s resource lies. Negative once
        the robot is inside it."""
        plan = self.plan
        model = self.resources
        assert plan is not None and model is not None
        step = plan.steps[index]
        pos = self._step_pos[index]
        res = step.resource
        if res.kind == REGION:
            return self._signed_distance_to_pos_mm(pos) - config.JUNCTION_FOOTPRINT_MM
        if res.kind == STATION:
            if pos == 0 and index == 0:
                return -1  # the station the plan leaves from
            if self.route_index == pos and self.edge_id is None:
                return -1
            anchor_pos = pos - 1
            return self._signed_distance_to_pos_mm(anchor_pos) + self._node_margin_mm(plan.nodes[anchor_pos])
        # A lane or corridor begins where the edge leaves its first node.
        return self._signed_distance_to_pos_mm(pos) + self._node_margin_mm(plan.nodes[pos])

    def _step_left(self, index: int) -> bool:
        """Whether the robot has physically left step ``index``'s resource."""
        plan = self.plan
        model = self.resources
        assert plan is not None and model is not None
        res = plan.steps[index].resource
        pos = self._step_pos[index]
        if res.kind == REGION:
            return self._signed_distance_to_pos_mm(pos) <= -config.JUNCTION_FOOTPRINT_MM
        if res.kind == STATION:
            if self.route_index == pos:
                return self.edge_id is not None and self.progress_mm > 0
            return self.route_index > pos
        to_pos = pos + 1
        if to_pos >= len(plan.nodes):
            return False
        d = self._signed_distance_to_pos_mm(to_pos)
        to_node = plan.nodes[to_pos]
        if res.kind == CORRIDOR:
            # Done at the far node if a region takes over there (the region step
            # releases in its own time); a footprint past it if the far node is a
            # bend, so the node point is clear before the next robot enters.
            return d <= (0 if model.has_region(to_node) else -config.JUNCTION_FOOTPRINT_MM)
        return d <= (config.JUNCTION_FOOTPRINT_MM if model.has_region(to_node) else 0)

    def _sync_plan_progress(self, *, adopting: bool = False) -> None:
        """Advance ``plan_index`` (released) and ``_entered_index`` (entered) to match
        where the robot physically is. Both only ever move forward within a plan.

        When ``adopting`` a plan, every step anchored at or behind the node under
        the robot is under the wheels -- the prefix of a mid-edge plan, the station
        a plan leaves from, the region a plan starts inside -- and counts as
        entered. Asking their boundary distance instead counted the lane a robot
        had just turned onto as still ahead of it, with an entry time in the past,
        and that read as "late" every tick. Only when adopting: applied every tick
        it counted the corridor *beyond* a junction as entered the moment the robot
        reached the junction, and the corridor's gate vanished.
        """
        plan = self.plan
        if plan is None or len(self._step_pos) != len(plan.steps):
            return
        steps = plan.steps
        while self.plan_index < len(steps) and self._step_left(self.plan_index):
            self.plan_index += 1
        if self._entered_index < self.plan_index:
            self._entered_index = self.plan_index
        if adopting:
            while self._entered_index < len(steps) and self._step_pos[self._entered_index] <= self.route_index:
                self._entered_index += 1
        while self._entered_index < len(steps) and self._boundary_distance_mm(self._entered_index) <= 0:
            self._entered_index += 1

    # -- may I enter -------------------------------------------------------------

    def _peer_released(self, booking: Booking, *, entered_is_enough: bool = False) -> bool:
        """Whether the robot holding ``booking`` has left that resource -- or, with
        ``entered_is_enough``, has at least entered it -- by its own account.

        INTENT carries how many steps the peer has released. Released past the
        booking's step means it has left; released everything *before* the step
        means it is on it. A lane asks the second: no overtaking is an order of
        entry, so a robot booked behind another on a lane may not enter until the
        other has, or the one behind on paper ends up ahead on the floor and the
        two wait on each other -- measured as a two-robot deadlock on one lane.
        """
        peer = self.peers.get(booking.robot_id)
        if peer is None:
            return True  # reaped: its bookings are gone with it
        if peer.plan_seq != booking.plan_seq:
            # Its INTENT names a plan we have not got. Until that PATH arrives the
            # peer is opaque and its old booking is treated as still live.
            return False
        if entered_is_enough:
            return peer.plan_index >= booking.step_index
        return peer.plan_index > booking.step_index

    def _may_enter(self, step: Step, now_ms: int) -> tuple[bool, WaitCause | None]:
        """Precedence: everyone booked ahead of me on this resource has released it."""
        plan = self.plan
        assert plan is not None
        res = step.resource
        if res.kind in (REGION, CORRIDOR) and now_ms - plan.committed_ms < config.SETTLE_MS:
            return False, WaitCause(
                kind="settle", blocker_id=-1, resource=str(res),
                detail=f"plan#{plan.plan_seq} is {now_ms - plan.committed_ms} ms old",
            )
        for booking in self.bookings.predecessors(res, step.enter_ms, exclude_robot=self.robot_id):
            if not self._peer_released(booking, entered_is_enough=res.kind == LANE):
                return False, WaitCause(
                    kind="precedence", blocker_id=booking.robot_id, resource=str(res),
                    detail=f"booked ahead at {booking.enter_ms}",
                )
        # An opaque peer -- INTENT heard, its current PATH not -- occupies the
        # region of the node it last reported and nothing beyond.
        if res.kind == REGION:
            for peer in self.peers.live(now_ms):
                known = self.bookings.plans.get(peer.robot_id)
                if known is not None and known.plan_seq == peer.plan_seq:
                    continue
                if peer.current_node == res.key:
                    return False, WaitCause(
                        kind="opaque", blocker_id=peer.robot_id, resource=str(res),
                        detail="its PATH has not arrived",
                    )
        return True, None

    def _next_gate(self) -> tuple[list[Step], int] | None:
        """The next stretch not yet entered, up to and including the next place a
        robot may wait, and the distance to its boundary.

        From a region or corridor the stretch runs through every region and
        corridor that follows and ends with the lane or station it leads into; a
        lane on its own is a stretch of one. The whole stretch is checked at its
        first boundary because nothing inside it is a place to wait: once a robot
        is in a region it must clear it, so it may not enter until everything up
        to the next waiting point is its turn -- including the lane beyond, whose
        order of entry is what no-overtaking means.
        """
        plan = self.plan
        if plan is None:
            return None
        steps = plan.steps
        if len(self._step_pos) != len(steps) or self._entered_index >= len(steps):
            return None
        index = self._entered_index
        group = [steps[index]]
        j = index + 1
        while steps[j - 1].resource.atomic and j < len(steps):
            group.append(steps[j])
            j += 1
        return group, self._boundary_distance_mm(index)

    def _drive_by_precedence(self, now_ms: int, result: StepResult) -> int:
        """Speed for this tick from the plan: nominal, slowed on approach to a
        boundary that is not yet ours, or held at the line. Never holds inside a
        region or corridor -- once entered, the only safe act is to clear it."""
        self._sync_plan_progress()
        speed = config.NOMINAL_SPEED_MM_S
        gate = self._next_gate()
        if gate is None:
            self._leave_yield(now_ms, result)
            return speed
        group, distance = gate
        if distance <= 0:
            self._leave_yield(now_ms, result)
            return speed
        allowed, cause = True, None
        for step in group:
            allowed, cause = self._may_enter(step, now_ms)
            if not allowed:
                break
        if allowed:
            self._leave_yield(now_ms, result)
            return speed
        if distance <= self._hold_line_mm():
            self.wait_cause = cause
            self._enter_yield(now_ms, result, cause)
            return 0
        if distance <= config.APPROACH_SLOW_MM:
            return config.YIELD_SPEED_MM_S
        return speed

    def _hold_line_mm(self) -> int:
        """How far short of the next boundary to stop: HOLD_LINE_MM, or half the
        lane if the lane is shorter than that.

        Where two junctions sit two footprints apart the lane between their
        regions is a few hundred millimetres. Stopping the full HOLD_LINE_MM short
        of the next region left the robot still inside the previous one -- never
        releasing it -- and two robots crossing a short edge in opposite
        directions each waited for the other to release the region behind it.
        """
        if self.edge_id is None or self.next_node is None or self.resources is None:
            return config.HOLD_LINE_MM
        gap = self.graph.length_mm(self.edge_id)
        if self.resources.has_region(self.current_node):
            gap -= config.JUNCTION_FOOTPRINT_MM
        if self.resources.has_region(self.next_node):
            gap -= config.JUNCTION_FOOTPRINT_MM
        return max(20, min(config.HOLD_LINE_MM, gap // 2))

    def _enter_yield(self, now_ms: int, result: StepResult, cause: WaitCause | None) -> None:
        if self.state is State.MOVING and self.machine.fire_if_possible(Event.CONFLICT_LOST, now_ms):
            self.metrics.yields_lost += 1
            result.notes.append(f"holding at the line: {cause}")

    def _leave_yield(self, now_ms: int, result: StepResult) -> None:
        if self.state is State.YIELD:
            self.machine.fire_if_possible(Event.CONFLICT_CLEARED, now_ms)

    def _is_late(self, now_ms: int) -> bool:
        """Behind the plan by more than the slack: the next step's window opened
        REPLAN_SLACK_MS ago and the robot is not yet in it."""
        plan = self.plan
        if plan is None or self._entered_index >= len(plan.steps):
            return False
        return now_ms > plan.steps[self._entered_index].enter_ms + config.REPLAN_SLACK_MS

    def _maybe_replan(self, now_ms: int, result: StepResult) -> None:
        """Re-book after losing a race, or when running late. Never while inside a
        corridor: the plan opens with the edge under the wheels and a corridor
        cannot be planned from its middle.

        Lateness matters not for this robot -- order, not time, gates entry -- but
        for everyone planning around it: bookings in the past say it has been where
        it has not, and a peer replanning behind it on a lane read "already ahead
        of me" off them, booked the junction after it, and the two waited on each
        other. A late robot re-books so its timetable is true again.
        """
        if self.plan is None or self._plan_goal < 0:
            return
        if not (self._replan_requested or self._is_late(now_ms)):
            return
        if now_ms - self._last_replan_ms < config.INTENT_PERIOD_MS:
            return
        if self.edge_id is not None and self.graph.edge(self.edge_id).single_lane:
            return
        self._last_replan_ms = now_ms
        why = "lost a race" if self._replan_requested else "behind the plan"
        if self._commit_plan(self._plan_goal, now_ms, result):
            self.metrics.replans += 1
            result.notes.append(f"replanned: {why}")
        else:
            self._replan_requested = False  # keep the old plan; try again later

    def _drive_tick(self, now_ms: int, result: StepResult) -> None:
        """The moving tick shared by task legs, withdrawing to a bay, and heading to
        a charger. Precedence first, then anticipatory following, then the sensor
        net (FR-6.6) -- the last word, whatever was decided above it."""
        if self.next_node is None:
            return
        if self.arbiter is not None and self.plan is not None:
            self._maybe_replan(now_ms, result)
            speed = self._drive_by_precedence(now_ms, result)
        else:
            speed = self._uncoordinated_speed(now_ms, result)
        speed = self._follow_traffic_ahead(now_ms, speed, result)
        if self._hold_outside_corner(result):
            speed = 0
        speed = self._limit_for_clearance(speed, result)
        self._last_speed_mm_s = speed
        result.command = MotionCommand(speed_mm_s=speed, target_node=self.next_node)

    def _uncoordinated_speed(self, now_ms: int, result: StepResult) -> int:
        """Configuration A: no plan, no precedence. FR-10.5's stop-and-wait."""
        return config.NOMINAL_SPEED_MM_S

    def _go_to(self, goal: int, now_ms: int, result: StepResult) -> bool:
        """Have a route to ``goal`` in force -- booked, for a coordinated robot;
        plain A*, for the baseline. Returns whether one now exists."""
        if self.arbiter is None:
            if self.next_node is None or self.route[-1:] != [goal]:
                route = self.planner.route(self.current_node, goal)
                if route is None or len(route) < 2:
                    return False
                self._adopt_route(route)
            return True
        if self.plan is not None and self._plan_goal == goal and self.next_node is not None:
            return True
        if self.planner.route(self.current_node, goal) is None:
            return False
        return self._commit_plan(goal, now_ms, result)

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
        # The region was entered a footprint ago; a mid-edge replan opens with it.
        self._region_entered_ms[node_id] = (
            self._now_ms - config.JUNCTION_FOOTPRINT_MM * 1000 // config.NOMINAL_SPEED_MM_S
        )
        if len(self._region_entered_ms) > 8:
            oldest = min(self._region_entered_ms, key=self._region_entered_ms.get)
            del self._region_entered_ms[oldest]
        if self.route_index + 1 < len(self.route) and self.route[self.route_index + 1] == node_id:
            self.route_index += 1
        self._aim_at_next()

    def hold(self, elapsed_ms: int) -> None:
        """Report that the robot did not move this tick (FR-10.3)."""
        self.metrics.stopped_ms += elapsed_ms
        if self.wait_cause is not None and self.wait_cause.kind in ("precedence", "settle", "opaque"):
            self.metrics.precedence_hold_ms += elapsed_ms

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
