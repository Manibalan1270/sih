"""Headless deterministic simulation engine.

This is the primary engine: it is what the benchmark runs on and the only source
of numbers admissible as evidence for AC-2 and AC-3. Webots is a visual path
layered on the same ``core.robot.Robot`` objects, not a replacement for this.

The engine owns two things the robots must not: **time** and **motion**. It
advances a virtual clock in fixed steps and carries out the ``MotionCommand`` each
robot emits. Robots never read a clock and never move themselves, which is what
makes a run a pure function of its seed (NFR-4.4) -- rerun it a year later on
another machine and every collision, yield and completion falls on the same
millisecond.

Determinism rests on four rules, each easy to break by accident:

1. Time is ``tick * tick_ms``. Never a wall clock, never accumulated floats.
2. Robots are stepped in ``robot_id`` order, always.
3. All randomness comes from one seeded ``Random``, drawn in a fixed order.
4. Motion is integer millimetres. No float positions to drift or round.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from core import config
from core.graph import Graph
from core.robot import Robot
from core.state_machine import State
from simulator.collision_detector import CollisionDetector, CollisionEvent, Pose
from simulator.deadlock import StallReport, StallWatcher


@dataclass(frozen=True, slots=True)
class SimEvent:
    """One thing that happened, timestamped on the virtual clock.

    The event log is the benchmark's raw material: FR-10.1 through FR-10.3 are all
    computed from it, and NFR-4.4 requires every safety decision to be
    reconstructable from the inputs that produced it.
    """

    at_ms: int
    kind: str
    robot_id: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        who = "" if self.robot_id is None else f" r{self.robot_id}"
        return f"{self.at_ms:>8} ms  {self.kind}{who}  {self.detail}".rstrip()


@dataclass
class Engine:
    """Fixed-step simulation of a fleet on a graph."""

    graph: Graph
    robots: list[Robot]
    seed: int = 0
    tick_ms: int = config.MOTION_TICK_MS
    tick: int = 0

    mesh: object | None = None
    """The message bus, or None for a fleet that does not communicate.

    None is Configuration A: FR-10.5 requires the baseline to exchange no intent at
    all, and giving it no mesh makes that structural rather than a matter of
    remembering not to send."""

    collision_detector: CollisionDetector = field(default_factory=CollisionDetector)
    events: list[SimEvent] = field(default_factory=list)
    log_notes: bool = True
    """Whether to record each robot's decision notes. Useful for a 3-robot demo,
    switched off for scale100 where it would dominate memory."""

    stall_watcher: StallWatcher = field(default_factory=StallWatcher)
    """Watches for a motionless fleet and explains it (see simulator/deadlock.py).

    Observational only: it never influences a decision, so it cannot affect
    reproducibility or bring state onto the arbitration path."""

    stall_report: StallReport | None = None
    """The first stall diagnosed in this run, if any."""

    rng: random.Random = field(init=False)
    _by_id: dict[int, Robot] = field(init=False, repr=False)
    _disabled: set[int] = field(default_factory=set, repr=False)
    """Robots killed mid-run for TC-5. They stop stepping and stop being
    physically present, which is what makes their peers' reservations expire."""

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        # Sorted so step order is a property of the fleet, not of list order.
        self.robots = sorted(self.robots, key=lambda r: r.robot_id)
        self._by_id = {r.robot_id: r for r in self.robots}
        if len(self._by_id) != len(self.robots):
            raise ValueError("robot ids must be unique within a fleet")

    # -- clock ---------------------------------------------------------------

    @property
    def now_ms(self) -> int:
        """Virtual time. Derived from the tick count, never accumulated."""
        return self.tick * self.tick_ms

    def ticks_for(self, period_ms: int) -> int:
        """How many ticks make up ``period_ms``.

        Protocol periods must be whole numbers of ticks or a 200 ms INTENT would
        drift against the clock it is timestamped on. config.MOTION_TICK_MS is
        chosen to divide every period in Appendix E.
        """
        if period_ms % self.tick_ms:
            raise ValueError(
                f"period {period_ms} ms is not a whole number of {self.tick_ms} ms "
                f"ticks, so its rate would drift against the aligned clock"
            )
        return period_ms // self.tick_ms

    def is_due(self, period_ms: int) -> bool:
        return self.tick % self.ticks_for(period_ms) == 0

    # -- fleet ---------------------------------------------------------------

    @property
    def active_robots(self) -> list[Robot]:
        return [r for r in self.robots if r.robot_id not in self._disabled]

    def robot(self, robot_id: int) -> Robot:
        return self._by_id[robot_id]

    def kill(self, robot_id: int) -> None:
        """Remove a robot from the world without warning (TC-5, FR-6.5).

        It stops deciding and stops being collidable. Peers must notice through
        silence alone -- there is no death notification, because a robot that has
        lost power cannot send one.
        """
        self._disabled.add(robot_id)
        if self.mesh is not None:
            self.mesh.leave(robot_id)
        self.log("robot_killed", robot_id, "powered off mid-run")

    def revive(self, robot_id: int) -> None:
        self._disabled.discard(robot_id)

    # -- logging -------------------------------------------------------------

    def log(self, kind: str, robot_id: int | None = None, detail: str = "") -> None:
        self.events.append(SimEvent(self.now_ms, kind, robot_id, detail))

    def events_of(self, kind: str) -> list[SimEvent]:
        return [e for e in self.events if e.kind == kind]

    # -- the tick ------------------------------------------------------------

    def step(self) -> list[CollisionEvent]:
        """Advance the world by one tick. Returns collisions that began now.

        Order within a tick is fixed and matters: every robot decides from the
        same world state, then all motion is applied, then collisions are checked.
        Deciding and moving robot-by-robot would let a robot see a peer that had
        already moved this tick, which is information a real robot could not have.
        """
        if self.mesh is not None:
            self.mesh.deliver()

        # Sensor readings first, from the world as it stands at the start of the
        # tick. A robot cannot see where a peer will be after this tick's motion.
        self._update_forward_clearance()

        commands = []
        for robot in self.active_robots:
            before = robot.state
            inbox = (
                self.mesh.inbox(robot.robot_id) if self.mesh is not None else None
            )
            result = robot.step(self.now_ms, inbox)
            commands.append((robot, result))
            if before is not robot.state:
                self.log(
                    "state",
                    robot.robot_id,
                    f"{before.value} -> {robot.state.value}",
                )
            if self.log_notes:
                for note in result.notes:
                    self.log("note", robot.robot_id, note)

        # Motion and transmission after every decision, so nothing a robot does
        # this tick can be observed by a peer deciding in the same tick.
        for robot, result in commands:
            self._apply_motion(robot, result)
            if self.mesh is not None and result.outbox:
                self.mesh.send_all(robot.robot_id, result.outbox, self.now_ms)

        self.stall_watcher.observe(self.active_robots, self.now_ms)
        if self.stall_report is None and self.stall_watcher.is_stalled(
            self.active_robots, self.now_ms
        ):
            self.stall_report = self.stall_watcher.report(
                self.active_robots, self.now_ms
            )
            for line in self.stall_report.describe().splitlines():
                self.log("stall", None, line.strip())

        collisions = self.collision_detector.check(self.now_ms, self.poses())
        for collision in collisions:
            self.log(
                "collision",
                collision.robot_a,
                f"with r{collision.robot_b} at ({collision.x_mm},{collision.y_mm}) "
                f"sep {collision.separation_mm} mm, "
                f"{'coordination failure' if collision.is_coordination_failure else 'contact'}",
            )

        self.tick += 1
        return collisions

    def _update_forward_clearance(self) -> None:
        """Feed every robot its forward proximity reading (IF-2.4).

        Modelled rather than messaged. The guide's upward flow has the simulator
        report "collision proximity sensor value" and our code interpret it, and
        keeping headway on a sensor rather than on INTENT means it still works
        against a peer whose radio has failed -- which is the case that matters.

        Two things count as ahead: a robot further along the same edge in the same
        direction, and a robot sitting at the node this robot is heading for.
        Opposing traffic on a bidirectional aisle is in the other lane and is not an
        obstacle; on a single-lane corridor it is, and corridor arbitration is what
        stops them meeting there.
        """
        occupants: dict[tuple[int, int], list[Robot]] = {}
        # node -> [(robot, departing)]. ``departing`` distinguishes a robot on its way
        # out of the node from one on its way in, which decides who gives way.
        at_node: dict[int, list[tuple[Robot, bool]]] = {}
        for robot in self.active_robots:
            if robot.edge_id is not None and robot.next_node is not None:
                occupants.setdefault((robot.edge_id, robot.next_node), []).append(robot)

            if robot.edge_id is None:
                at_node.setdefault(robot.current_node, []).append((robot, False))
                continue

            # Occupying the node behind it, on the way out. The clearance used is the
            # junction footprint, not a few millimetres: a robot 250 mm past a node is
            # still inside it. Section 3.4.1 says as much -- current_node is "the
            # junction most recently occupied *or departed*".
            #
            # This is the case that collided at 6 robots: one robot 264 mm past a node
            # and another 940 mm from it, 497 mm apart. The departing robot's claim had
            # already vanished from its INTENT, and a 150 mm test saw neither.
            if robot.progress_mm <= config.JUNCTION_CLEARANCE_MM:
                at_node.setdefault(robot.current_node, []).append((robot, True))

            # Occupying the node ahead, on the way in -- so two robots converging on
            # one node from different aisles can see each other. Junction arbitration
            # separates them in time; this is the geometric backstop for when its
            # timing is slightly out.
            to_node = robot.graph.length_mm(robot.edge_id) - robot.progress_mm
            if robot.next_node is not None and to_node <= config.JUNCTION_CLEARANCE_MM:
                at_node.setdefault(robot.next_node, []).append((robot, False))

        for robot in self.active_robots:
            robot.forward_clearance_mm = 1 << 30
            robot.forward_blocker_id = -1
            robot.forward_blocker_stopped = False
            if robot.edge_id is None or robot.next_node is None:
                continue
            nearest = 1 << 30
            blocker = -1

            same_lane = occupants.get((robot.edge_id, robot.next_node), ())
            for other in same_lane:
                if other.robot_id == robot.robot_id:
                    continue
                gap = other.progress_mm - robot.progress_mm
                if gap == 0:
                    # Coincident. There is no "behind", so the symmetric rule would
                    # have both hold for each other. Break it with the same total
                    # order the rest of the system uses: the lower id goes first.
                    if robot.robot_id < other.robot_id:
                        continue
                    nearest, blocker = 0, other.robot_id
                    blocker_stopped = other._last_speed_mm_s == 0
                elif 0 < gap < nearest:
                    nearest, blocker = gap, other.robot_id
                    blocker_stopped = other._last_speed_mm_s == 0

            to_node = robot.graph.length_mm(robot.edge_id) - robot.progress_mm
            for other, departing in at_node.get(robot.next_node, ()):
                if other.robot_id == robot.robot_id:
                    continue
                # A robot on its way *out* of the node is an obstacle unconditionally:
                # it is physically in the way and it will clear on its own, so waiting
                # for it always terminates.
                #
                # A robot on its way *in* is a peer converging on the same node. Both
                # treating the other as an obstacle would have both hold for ever, so
                # the same total order used everywhere else decides: the lower id goes
                # first, and the higher gives way.
                if not departing and robot.robot_id < other.robot_id:
                    continue
                # The oncoming lane is not ahead. A robot leaving the node along the
                # edge this one is arriving on, in the other direction, is beside it,
                # not in front of it -- a forward sensor does not see it, and treating
                # it as an obstacle held robots inside the region behind them until
                # three of them waited in a ring. Single-lane edges have no other lane.
                if (
                    other.edge_id == robot.edge_id
                    and other.next_node != robot.next_node
                    and not robot.graph.edge(robot.edge_id).single_lane
                ):
                    continue
                if to_node < nearest:
                    nearest, blocker = to_node, other.robot_id
                    blocker_stopped = other._last_speed_mm_s == 0

            robot.forward_clearance_mm = nearest
            robot.forward_blocker_id = blocker
            robot.forward_blocker_stopped = blocker >= 0 and blocker_stopped


    def _apply_motion(self, robot: Robot, result) -> None:
        command = result.command
        if command.is_hold:
            robot.hold(self.tick_ms)
            return

        # Integer millimetres per tick. At 800 mm/s and a 20 ms tick this is 16.
        distance_mm = (command.speed_mm_s * self.tick_ms) // 1000
        if distance_mm <= 0:
            robot.hold(self.tick_ms)
            return

        reached = robot.advance(distance_mm, self.tick_ms)
        robot.drain_battery(distance_mm)
        if reached is not None:
            self.log("arrive", robot.robot_id, f"node {reached}")

    def poses(self) -> list[Pose]:
        """Physical positions of every robot present in the world."""
        result = []
        for robot in self.active_robots:
            # Lane-adjusted, because two robots passing on a bidirectional aisle are
            # not in contact -- see Robot.footprint_mm.
            x, y = robot.footprint_mm()
            result.append(
                Pose(
                    robot_id=robot.robot_id,
                    x_mm=x,
                    y_mm=y,
                    state=robot.state,
                    node=robot.current_node,
                    edge_id=robot.edge_id,
                )
            )
        return result

    # -- running -------------------------------------------------------------

    def run(
        self,
        *,
        max_ms: int,
        until: Callable[["Engine"], bool] | None = None,
        stop_on_stall: bool = False,
    ) -> bool:
        """Run until ``until`` is satisfied or ``max_ms`` elapses.

        Returns whether ``until`` was satisfied. A hard time limit is mandatory
        rather than optional: a deadlocked fleet would otherwise spin forever, and
        the point of TC-3 is that deadlock cannot happen -- a test that hangs
        instead of failing proves nothing.

        ``stop_on_stall`` returns as soon as a wait-for cycle is diagnosed rather
        than grinding on to the limit. Off by default so it cannot change any
        existing result; worth switching on when investigating, where it turns a
        multi-minute wait into a few seconds and leaves ``stall_report`` set.
        """
        limit_tick = self.tick + max(1, max_ms // self.tick_ms)
        while self.tick < limit_tick:
            if until is not None and until(self):
                return True
            self.step()
            if (
                stop_on_stall
                and self.stall_report is not None
                and self.stall_report.is_deadlocked
            ):
                return False
        return until(self) if until is not None else True

    def run_ticks(self, count: int) -> None:
        for _ in range(count):
            self.step()

    # -- convenience predicates ---------------------------------------------

    def all_idle(self) -> bool:
        return all(r.state is State.IDLE for r in self.active_robots)

    def all_queues_empty(self) -> bool:
        return all(not r.queue for r in self.active_robots)

    def work_finished(self) -> bool:
        """Every active robot is idle with nothing queued."""
        return self.all_idle() and self.all_queues_empty()

    def tasks_completed(self) -> int:
        return sum(r.metrics.tasks_completed for r in self.robots)

    # -- reporting -----------------------------------------------------------

    @property
    def collisions(self) -> list[CollisionEvent]:
        return self.collision_detector.events

    @property
    def coordination_failures(self) -> list[CollisionEvent]:
        return self.collision_detector.coordination_failures

    def route_diversity_q8(self) -> int:
        """Distinct edges used across the fleet over edges available, as Q8.

        A secondary metric in section 6.3. Low diversity on a map with bypasses
        means every robot chose the same aisle and manufactured the congestion the
        traffic model was supposed to avoid.
        """
        used = set()
        for robot in self.robots:
            used |= robot.metrics.edges_used
        available = max(1, len(self.graph.edges))
        return (len(used) * config.Q8_ONE) // available

    def summary(self) -> str:
        failures = len(self.coordination_failures)
        return (
            f"t={self.now_ms} ms  tasks={self.tasks_completed()}  "
            f"collisions={len(self.collisions)} (coordination failures={failures})  "
            f"route diversity={self.route_diversity_q8() * 100 // config.Q8_ONE}%"
        )


def spawn_positions(graph: Graph, count: int, *, seed: int) -> list[int]:
    """Pick ``count`` distinct start nodes deterministically, bays first.

    A fleet starts in its bays: that is where robots are between shifts, it is
    the one place standing still obstructs nobody, and it is what lets every
    robot's first route plan begin at a station rather than inside a junction.
    Only when the fleet outnumbers the bays does the remainder spread across the
    other nodes, seeded and strided so different seeds use different nodes without
    clustering -- an artificial jam at t=0 would flatter or damn Configuration A
    and B unequally.
    """
    bays = sorted(graph.parking_nodes)
    if bays:
        offset = random.Random(seed).randrange(len(bays))
        chosen = [bays[(offset + i) % len(bays)] for i in range(min(count, len(bays)))]
        if len(chosen) == count:
            return chosen
        rest = _spread_positions(graph, count - len(chosen), seed=seed, avoid=set(chosen))
        return chosen + rest
    return _spread_positions(graph, count, seed=seed, avoid=set())


def _spread_positions(graph: Graph, count: int, *, seed: int, avoid: set[int]) -> list[int]:
    nodes = sorted(n for n in graph.nodes if n not in avoid)
    if count > len(nodes):
        raise ValueError(
            f"cannot spawn {count} robots on a {len(nodes)}-node map"
        )
    # Even stride first, so robots start far apart; the seeded rotation keeps
    # different seeds from always using the same nodes without clustering them.
    stride = max(1, len(nodes) // count)
    offset = random.Random(seed).randrange(len(nodes))
    chosen: list[int] = []
    index = 0
    while len(chosen) < count:
        node = nodes[(offset + index * stride) % len(nodes)]
        if node not in chosen:
            chosen.append(node)
        index += 1
        if index > 4 * len(nodes):  # pathological stride; fall back to order
            chosen = nodes[:count]
            break
    return chosen


def build_fleet(
    graph: Graph,
    planner_for: Callable[[Robot], object] | None,
    *,
    count: int,
    seed: int,
    planner_factory: Callable[[Graph], object],
    zone_of: Callable[[int], int] | None = None,
) -> list[Robot]:
    """Create ``count`` robots, each with its own planner instance.

    Each robot gets a *separate* planner object even though they share one graph.
    A shared planner would be shared mutable state between robots, and while it
    would not change any result today, it is exactly the kind of accidental
    centralisation this architecture exists to rule out.
    """
    del planner_for
    homes = spawn_positions(graph, count, seed=seed)
    fleet: list[Robot] = []
    for index, home in enumerate(homes):
        robot = Robot(
            robot_id=index + 1,
            graph=graph,
            planner=planner_factory(graph),  # type: ignore[arg-type]
            home_node=home,
            zone_id=zone_of(home) if zone_of else -1,
        )
        fleet.append(robot)
    return fleet


def summarise_events(events: Iterable[SimEvent], kinds: set[str] | None = None) -> str:
    lines = [str(e) for e in events if kinds is None or e.kind in kinds]
    return "\n".join(lines)
