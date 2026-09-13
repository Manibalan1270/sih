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

from dataclasses import dataclass, field
from typing import Protocol

from core import config
from core.graph import Graph
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
    frames: list[bytes] = field(default_factory=list)
    """Messages to broadcast. Populated from Phase 5 onward."""

    notes: list[str] = field(default_factory=list)
    """Human-readable record of what was decided and why. Feeds the dashboard's
    per-robot decision panel and the benchmark event log."""


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

    def position_mm(self) -> tuple[int, int]:
        """Interpolated position, for the collision check and the dashboard."""
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

    def step(self, now_ms: int) -> StepResult:
        """One tick of the robot's own decision making.

        Deliberately a flat dispatch on state rather than a chain of conditions.
        Appendix A defines behaviour per state, and matching that shape means a
        reviewer can check this against the SRS table line by line.
        """
        result = StepResult(command=MotionCommand.hold())

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
        # BIDDING, CHARGING and FAULT hold position. BIDDING is a 300 ms window
        # during which the robot keeps its committed plan but starts nothing new;
        # the auction itself is driven by messages, not by this loop.

        return result

    def _step_idle(self, now_ms: int, result: StepResult) -> None:
        if self.queue:
            self.machine.fire(Event.QUEUE_HAS_WORK, now_ms)
            result.notes.append(f"queued work: {self.queue.current}")

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

        # Phase 6 inserts arbitration here: detect a conflict on the junction
        # ahead, and either RESERVE or fire CONFLICT_LOST to enter YIELD.
        speed = (
            config.YIELD_SPEED_MM_S
            if self.state is State.YIELD
            else config.NOMINAL_SPEED_MM_S
        )
        result.command = MotionCommand(speed_mm_s=speed, target_node=self.next_node)

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
