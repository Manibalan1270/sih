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

    collision_detector: CollisionDetector = field(default_factory=CollisionDetector)
    events: list[SimEvent] = field(default_factory=list)
    log_notes: bool = True
    """Whether to record each robot's decision notes. Useful for a 3-robot demo,
    switched off for scale100 where it would dominate memory."""

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
        commands = []
        for robot in self.active_robots:
            before = robot.state
            result = robot.step(self.now_ms)
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

        for robot, result in commands:
            self._apply_motion(robot, result)

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
            x, y = robot.position_mm()
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
    ) -> bool:
        """Run until ``until`` is satisfied or ``max_ms`` elapses.

        Returns whether ``until`` was satisfied. A hard time limit is mandatory
        rather than optional: a deadlocked fleet would otherwise spin forever, and
        the point of TC-3 is that deadlock cannot happen -- a test that hangs
        instead of failing proves nothing.
        """
        limit_tick = self.tick + max(1, max_ms // self.tick_ms)
        while self.tick < limit_tick:
            if until is not None and until(self):
                return True
            self.step()
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
    """Pick ``count`` distinct, spread-out start nodes deterministically.

    Spread matters: spawning a fleet on adjacent nodes creates an artificial
    traffic jam at t=0 that has nothing to do with the coordination logic, and
    would flatter or damn Configuration A and B unequally.
    """
    nodes = sorted(graph.nodes)
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
