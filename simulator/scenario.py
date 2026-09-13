"""Assembling a runnable simulation from a declared scenario.

``core.scenarios`` says *what* to run; this says how to stand it up. One builder
serves bench3, visual30, scale100 and hardware2, which is what stops four
demonstrations becoming four codebases.

Task allocation is pluggable, and that is the heart of the benchmark rather than a
convenience. Configuration A and Configuration B differ in exactly two things --
how work is allocated and whether intent is exchanged -- so allocation has to be a
swappable part while everything around it stays byte-identical. If the two
configurations shared no code, a measured difference between them could always be
blamed on some incidental difference in the harness.

``RoundRobinAllocator`` is therefore not scaffolding. It is Configuration A's
allocator as FR-10.5 specifies it, and it also serves Phase 4, where three robots
work independently before any auction exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from core import config
from core.graph import Graph
from core.planner_astar import AStarPlanner
from core.robot import Robot
from core.scenarios import Scenario
from core.state_machine import State
from core.task import Task
from core.zones import ZoneMap
from benchmark.task_generator import TaskSet, generate_for_map
from simulator.engine import Engine, spawn_positions


class TaskAllocator(Protocol):
    """How announced work reaches robots.

    Configuration A assigns round-robin. Configuration B announces to the mesh and
    lets robots bid (Phase 5). Both see the same released tasks at the same
    simulated millisecond.
    """

    name: str

    def tick(self, sim: "Simulation", now_ms: int) -> None:
        ...


@dataclass
class RoundRobinAllocator:
    """Configuration A's allocator (FR-10.5): round-robin, no bidding.

    Walks the fleet in id order and gives the next task to the next robot with
    room. It takes no account of where a robot is or what it is already doing,
    which is the point: it is the conventional allocator the auction is being
    measured against, and its weakness is part of the result.
    """

    name: str = "round_robin"
    cursor: int = 0

    def tick(self, sim: "Simulation", now_ms: int) -> None:
        for task in sim.take_pending(now_ms):
            if not self._offer(sim, task, now_ms):
                sim.defer(task)

    def _offer(self, sim: "Simulation", task: Task, now_ms: int) -> bool:
        fleet = sim.engine.active_robots
        if not fleet:
            return False
        for offset in range(len(fleet)):
            robot = fleet[(self.cursor + offset) % len(fleet)]
            if robot.queue.is_full:
                continue
            if robot.battery_pct < config.BATTERY_RESERVE_PCT:
                continue  # BR-4: below reserve an AMR accepts no new work
            if robot.state is State.FAULT:
                continue
            self.cursor = (self.cursor + offset + 1) % len(fleet)
            robot.accept_task(task, now_ms)
            sim.engine.log(
                "assign", robot.robot_id, f"task {task.task_id} ({self.name})"
            )
            return True
        return False


@dataclass
class Simulation:
    """A scenario, built and ready to run."""

    scenario: Scenario
    graph: Graph
    zones: ZoneMap
    engine: Engine
    task_set: TaskSet
    allocator: TaskAllocator
    seed: int

    pending: list[Task] = field(default_factory=list)
    """Announced but unallocated. Tasks arrive here at their release time and
    return here when a robot hands one back (FR-6.2, FR-6.5)."""

    _released: set[int] = field(default_factory=set, repr=False)
    completed: list[Task] = field(default_factory=list)

    # -- task flow -----------------------------------------------------------

    def take_pending(self, now_ms: int) -> list[Task]:
        """Release newly due tasks and return everything awaiting allocation.

        Also recovers tasks robots have let go of. A released task that nobody
        picked up again would simply vanish, and the run would report a makespan
        over fewer tasks than it was given -- a silently flattering result.
        """
        for task in self.task_set.tasks:
            if task.task_id not in self._released and task.created_at_ms <= now_ms:
                self._released.add(task.task_id)
                self.pending.append(task)
                self.engine.log("announce", None, f"task {task.task_id}")

        for robot in self.engine.robots:
            for task in robot.drain_released_tasks():
                task.reannounce()
                self.pending.append(task)
                self.engine.log(
                    "reannounce",
                    robot.robot_id,
                    f"task {task.task_id} (attempt {task.announce_count})",
                )

        ready, self.pending = self.pending, []
        return ready

    def defer(self, task: Task) -> None:
        """Put a task back because no robot could take it now (FR-4.13).

        Its aging term keeps accruing, so it becomes more attractive with every
        cycle it waits. That is what discharges BR-2.
        """
        self.pending.append(task)

    def collect_completions(self) -> None:
        for robot in self.engine.robots:
            while robot.completed_tasks:
                self.completed.append(robot.completed_tasks.pop(0))

    # -- running -------------------------------------------------------------

    def step(self) -> None:
        now = self.engine.now_ms
        self.allocator.tick(self, now)
        self.engine.step()
        self.collect_completions()

    def run(self, *, max_ms: int) -> bool:
        """Run until all work is done or ``max_ms`` of simulated time elapses.

        Returns whether the work finished. A hard limit is mandatory: a fleet that
        cannot finish must fail the run rather than hang, or a deadlock defect
        would present as a stuck test rather than a failing one.
        """
        limit = self.engine.now_ms + max_ms
        while self.engine.now_ms < limit:
            if self.is_finished:
                return True
            self.step()
        return self.is_finished

    @property
    def is_finished(self) -> bool:
        return (
            not self.pending
            and len(self._released) == len(self.task_set)
            and self.engine.work_finished()
        )

    # -- reporting -----------------------------------------------------------

    @property
    def makespan_ms(self) -> int:
        """FR-10.2: last completion minus the first release."""
        if not self.completed:
            return 0
        first_release = min(t.created_at_ms for t in self.task_set)
        last_completion = max(t.completed_at_ms or 0 for t in self.completed)
        return last_completion - first_release

    def report(self) -> str:
        failures = len(self.engine.coordination_failures)
        stopped = sum(r.metrics.stopped_ms for r in self.engine.robots)
        yields = sum(r.metrics.yields_lost for r in self.engine.robots)
        return (
            f"{self.scenario.name}/{self.allocator.name} seed={self.seed}\n"
            f"  robots            {len(self.engine.robots)}\n"
            f"  tasks completed   {len(self.completed)}/{len(self.task_set)}\n"
            f"  makespan          {self.makespan_ms} ms\n"
            f"  collisions        {len(self.engine.collisions)} "
            f"({failures} coordination failures)\n"
            f"  stopped time      {stopped} ms across the fleet\n"
            f"  yields            {yields}\n"
            f"  route diversity   "
            f"{self.engine.route_diversity_q8() * 100 // config.Q8_ONE}%"
        )


def load_map(scenario: Scenario) -> tuple[Graph, dict]:
    """Load a scenario's map, returning the graph and its raw JSON.

    The raw form is kept because maps carry task geography -- ``choke_edge``,
    ``pickup_nodes``, ``drop_nodes`` -- that the graph itself has no business
    knowing about.
    """
    raw = json.loads(scenario.map_path.read_text(encoding="utf-8"))
    graph = Graph.from_dict(raw)
    graph.validate(scenario.max_nodes, scenario.max_edges)
    return graph, raw


def build(
    scenario: Scenario,
    *,
    seed: int,
    allocator: TaskAllocator | None = None,
    robots: int | None = None,
    task_count: int | None = None,
    waves: int = 4,
    log_notes: bool | None = None,
) -> Simulation:
    """Stand up a simulation for ``scenario`` at ``seed``.

    Every robot receives its own planner instance. They share one graph -- the
    warehouse is genuinely shared -- but sharing a planner would be shared mutable
    state between robots, which is the accidental centralisation this whole
    architecture exists to rule out.
    """
    scenario.validate()
    graph, raw = load_map(scenario)
    zones = ZoneMap.from_graph(graph, enabled=scenario.zoned)
    zones.validate(graph)

    fleet_size = robots if robots is not None else scenario.simulated_robots
    if fleet_size < 1:
        raise ValueError(
            f"{scenario.name} has no simulated robots to build; "
            f"physical_robots={scenario.physical_robots}"
        )

    homes = spawn_positions(graph, fleet_size, seed=seed)
    fleet = [
        Robot(
            robot_id=index + 1,
            graph=graph,
            planner=AStarPlanner(graph),
            home_node=home,
            zone_id=zones.assign(index + 1, home),
        )
        for index, home in enumerate(homes)
    ]

    # Log notes for small fleets, where they are the demo; suppress at scale,
    # where they would dominate memory without being read.
    notes = log_notes if log_notes is not None else fleet_size <= 10
    engine = Engine(graph=graph, robots=fleet, seed=seed, log_notes=notes)

    task_set = generate_for_map(
        graph,
        raw,
        seed=seed,
        count=task_count if task_count is not None else 4 * fleet_size,
        waves=waves,
        zone_of=zones.zone_of,
    )

    return Simulation(
        scenario=scenario,
        graph=graph,
        zones=zones,
        engine=engine,
        task_set=task_set,
        allocator=allocator or RoundRobinAllocator(),
        seed=seed,
    )
