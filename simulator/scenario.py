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

from communication.messages import Announce, MessageType
from core import config
from core.arbitration import Arbiter
from core.auction import Auctioneer
from core.graph import Graph
from core.planner_astar import AStarPlanner
from core.robot import Robot
from core.scenarios import Scenario
from core.state_machine import State
from core.task import Task
from core.zones import ZoneMap
from benchmark.task_generator import TaskSet, generate_for_map
from simulator.engine import Engine, spawn_positions
from simulator.mesh import GATEWAY_ID, Mesh, build_mesh


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
class AuctionAllocator:
    """Configuration B's allocator: the order gateway, and nothing more.

    It broadcasts ANNOUNCE and then has no further part in the outcome. FR-4.1
    forbids any component assigning a task to a named robot, and FR-4.6 forbids a
    dispatcher, so this class deliberately has no way to pick a winner -- it cannot
    even see the bids except as frames on the mesh like anyone else.

    IF-3.2 and IF-3.3 confine the gateway to originating ANNOUNCE and the clock
    beacon. It listens for CLAIM only so it can stop re-announcing work that has
    been taken; that is bookkeeping about its own announcements, not a decision
    about allocation.
    """

    name: str = "auction"
    announced_at: dict[int, int] = field(default_factory=dict)
    """task_id -> aligned time of its most recent ANNOUNCE."""

    claimed: set[int] = field(default_factory=set)
    finished: set[int] = field(default_factory=set)
    """Tasks a robot has reported COMPLETE. Never announced again, whatever else
    happens -- announcing finished work would have it delivered twice."""

    reannounce_after_ms: int = config.AUCTION_WINDOW_MS + config.CLAIM_TIMEOUT_MS

    def tick(self, sim: "Simulation", now_ms: int) -> None:
        assert sim.mesh is not None, "the auction allocator needs a mesh"
        self._absorb_claims(sim)

        # A task handed back is unclaimed again, whatever we previously overheard.
        for task_id in sim.reannounced:
            if task_id in self.finished:
                continue
            self.claimed.discard(task_id)
            self.announced_at.pop(task_id, None)
        sim.reannounced.clear()

        for task in sim.take_pending(now_ms):
            if task.task_id in self.finished or task.task_id in self.claimed:
                continue
            last = self.announced_at.get(task.task_id)
            if last is not None and now_ms - last < self.reannounce_after_ms:
                # Already on the mesh and still inside its auction window plus the
                # FR-4.12 claim timeout. Re-announcing now would double the auction
                # traffic for no gain.
                sim.defer(task)
                continue
            self.announced_at[task.task_id] = now_ms
            sim.mesh.send(
                GATEWAY_ID,
                Announce(
                    task_id=task.task_id,
                    pickup_node=task.pickup,
                    drop_node=task.drop,
                    priority=task.priority,
                    created_at_ms=task.created_at_ms,
                    zone_id=task.zone_id,
                ),
                now_ms,
            )
            sim.engine.log("announce", None, f"task {task.task_id} ({self.name})")
            sim.defer(task)  # stays pending until somebody claims it

    def _absorb_claims(self, sim: "Simulation") -> None:
        """Note which tasks are being worked, so they stop being announced.

        CLAIM alone is not enough. It is a one-shot frame, and IF-4.5 forbids
        requiring retransmission, so a gateway that missed one would keep
        announcing work already under way -- and a second robot would win it and do
        the job twice. Measured at 30% loss: 9 completions for 8 distinct tasks.

        INTENT closes it. Every robot broadcasts its held task every 200 ms, so a
        single lost frame is self-healing. Deriving state from overheard INTENT
        rather than asking anyone is the same discipline FR-8.6 sets for the
        dashboard, and it keeps the gateway inside IF-3.3: it still originates
        nothing but ANNOUNCE and the clock beacon.
        """
        assert sim.mesh is not None
        for message in sim.mesh.inbox(GATEWAY_ID):
            if message.type is MessageType.CLAIM:
                self._mark_taken(sim, message.payload.task_id)
                sim.engine.log(
                    "claim", message.sender, f"task {message.payload.task_id}"
                )
            elif message.type is MessageType.INTENT:
                if message.payload.held_task_id >= 0:
                    self._mark_taken(sim, message.payload.held_task_id)
            elif message.type is MessageType.COMPLETE:
                self.finished.add(message.payload.task_id)
                self._mark_taken(sim, message.payload.task_id)
            elif message.type is MessageType.ANNOUNCE:
                # A robot re-announcing under FR-4.12. Treat it as unclaimed again.
                self.claimed.discard(message.payload.task_id)

    def _mark_taken(self, sim: "Simulation", task_id: int) -> None:
        self.claimed.add(task_id)
        sim.pending = [t for t in sim.pending if t.task_id != task_id]


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
    mesh: Mesh | None = None

    pending: list[Task] = field(default_factory=list)
    """Announced but unallocated. Tasks arrive here at their release time and
    return here when a robot hands one back (FR-6.2, FR-6.5)."""

    _released: set[int] = field(default_factory=set, repr=False)
    completed: list[Task] = field(default_factory=list)

    reannounced: set[int] = field(default_factory=set)
    """Tasks a robot handed back this tick.

    The allocator must clear these from whatever "already taken" bookkeeping it
    keeps. Without it a task noted as claimed and then dropped is never announced
    again and never completed -- and the run reports itself finished having done
    less work than it was given, which is a silently flattering result rather than
    a visible failure."""

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
                self.reannounced.add(task.task_id)
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
        """Every task released, completed, and the fleet idle.

        Completion is counted by distinct task id rather than by an empty pending
        list. A defect that loses a task would otherwise satisfy "nothing pending,
        fleet idle" and report success having done less work than it was given --
        which is how a lost task first presented here. Now it runs to the time limit
        and fails, which is the failure mode a benchmark needs.
        """
        if len(self._released) != len(self.task_set):
            return False
        if len({t.task_id for t in self.completed}) != len(self.task_set):
            return False
        return not self.pending and self.engine.work_finished()

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

    chosen_allocator = allocator or AuctionAllocator()
    coordinated = not isinstance(chosen_allocator, RoundRobinAllocator)

    homes = spawn_positions(graph, fleet_size, seed=seed)
    fleet = []
    for index, home in enumerate(homes):
        robot_id = index + 1
        robot_zone = zones.assign(robot_id, home)
        robot = Robot(
                robot_id=robot_id,
                graph=graph,
                planner=AStarPlanner(graph),
                home_node=home,
                zone_id=robot_zone,
                # Configuration A gets no auctioneer at all. FR-10.5 requires the
                # baseline to exchange no intent, and withholding the collaborator
                # makes that structural rather than a matter of remembering not to.
                auctioneer=Auctioneer(robot_id=robot_id) if coordinated else None,
                arbiter=Arbiter(robot_id=robot_id) if coordinated else None,
                zone_eligible=(
                    (lambda task_zone, mine=robot_zone: zones.eligible_to_bid(mine, task_zone))
                    if coordinated
                    else None
                ),
            )
        # The planner costs edges through the robot, so a robot's own temporary
        # penalties reach its planning (FR-3.4). Wired after construction because the
        # two refer to each other.
        robot.planner.cost = robot.edge_cost
        fleet.append(robot)

    # Log notes for small fleets, where they are the demo; suppress at scale,
    # where they would dominate memory without being read.
    notes = log_notes if log_notes is not None else fleet_size <= 10

    mesh: Mesh | None = None
    if coordinated:
        by_id = {r.robot_id: r for r in fleet}

        def position_of(member_id: int) -> tuple[int, int] | None:
            robot = by_id.get(member_id)
            return robot.position_mm() if robot is not None else None

        # Range filtering is what makes INTENT delivery proportional to neighbours
        # rather than to fleet size (FR-9.4), and what keeps 100 robots tractable.
        # On the 30 x 22 m benchmark map a 40 m radio covers everything, so bench3
        # behaves as a flood -- correct, since section 4.9 says a flood auction is
        # right at 3-5 AMRs.
        mesh = build_mesh(
            seed=seed,
            id_bits=scenario.edge_id_bits,
            position_of=position_of,
        )
        for robot in fleet:
            mesh.join(robot.robot_id)
        mesh.join(GATEWAY_ID)

    engine = Engine(
        graph=graph, robots=fleet, seed=seed, log_notes=notes, mesh=mesh
    )

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
        allocator=chosen_allocator,
        seed=seed,
        mesh=mesh,
    )
