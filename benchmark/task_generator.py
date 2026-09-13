"""Deterministic task set generation (FR-10.6, FR-10.9).

The benchmark compares two configurations, so the *task set must be identical
between them*. That makes this module part of the experimental control, not a
convenience: a seed fixes the task set completely, and both Configuration A and
Configuration B are handed the same one (FR-10.6).

FR-10.9 sets the harder constraint. At least 60% of tasks must induce a route
through the choke corridor, because that is what produces the overlapping paths
the success criteria are about. A task set of random pickup/drop pairs would not
do it -- on the benchmark map a pair within one half of the warehouse never goes
near the corridor. So candidate pairs are classified by whether their nominal
least-cost route uses the choke, and sampled to hit the target deliberately.

Tasks are released in waves rather than all at once (section 6.3). A single
release at t=0 lets a fleet of three spread out and never meet again; waves keep
re-injecting contention for the whole run, which is what the comparison needs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.graph import NO_ZONE, Graph
from core.planner_astar import AStarPlanner
from core.task import MAX_PRIORITY, Task

CHOKE_TARGET = 0.75
"""Fraction of tasks aimed through the choke corridor. FR-10.9 requires at least
0.60; aiming higher leaves headroom, because a task whose *nominal* route uses the
corridor may still be diverted at run time once pheromone builds -- and it is the
run-time fraction the requirement is really about."""

PRIORITY_MIX: tuple[tuple[int, int], ...] = (
    (10, 70),
    (100, 20),
    (200, 10),
)
"""(priority, percent) pairs. Deliberately a small set rather than a uniform draw
over 0-255: repeated priorities mean arbitration frequently falls through to the
robot_id tie-break (FR-5.4, BR-1), which is the branch most likely to harbour a
defect and least likely to be exercised by uniform random priorities."""


@dataclass
class TaskSet:
    """A generated task set, with the properties needed to defend it."""

    tasks: list[Task]
    seed: int
    waves: int
    choke_tasks: int
    """Tasks whose nominal least-cost route uses the choke corridor."""

    graph_name: str = ""
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    @property
    def choke_fraction(self) -> float:
        return self.choke_tasks / len(self.tasks) if self.tasks else 0.0

    @property
    def horizon_ms(self) -> int:
        """When the last task is released."""
        return max((t.created_at_ms for t in self.tasks), default=0)

    def released_by(self, now_ms: int) -> list[Task]:
        """Tasks whose release time has arrived."""
        return [t for t in self.tasks if t.created_at_ms <= now_ms]

    def copy(self) -> "TaskSet":
        """An independent copy, so two configurations cannot share task objects."""
        return TaskSet(
            tasks=[t.copy() for t in self.tasks],
            seed=self.seed,
            waves=self.waves,
            choke_tasks=self.choke_tasks,
            graph_name=self.graph_name,
            warnings=list(self.warnings),
        )

    def check_fr_10_9(self, minimum: float = 0.60) -> None:
        """Raise unless the choke fraction meets FR-10.9.

        Called by the benchmark runner before a run, not merely by tests: a task
        set that fails this silently produces a comparison with nothing to
        measure, and the resulting 20% figure would be meaningless rather than
        wrong in a visible way.
        """
        if self.choke_fraction < minimum:
            raise ValueError(
                f"task set (seed {self.seed}) routes {self.choke_fraction:.0%} of "
                f"tasks through the choke corridor; FR-10.9 requires at least "
                f"{minimum:.0%}, or the benchmark has no contention to measure"
            )

    def describe(self) -> str:
        return (
            f"{len(self.tasks)} tasks over {self.waves} waves, "
            f"{self.choke_fraction:.0%} through the choke, "
            f"released to {self.horizon_ms} ms (seed {self.seed})"
        )


def _pick_priority(rng: random.Random) -> int:
    roll = rng.randrange(100)
    cumulative = 0
    for priority, percent in PRIORITY_MIX:
        cumulative += percent
        if roll < cumulative:
            return priority
    return PRIORITY_MIX[-1][0]


def _classify_pairs(
    graph: Graph,
    pickups: list[int],
    drops: list[int],
    choke_edge: int | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split candidate pairs by whether their nominal route uses the choke."""
    planner = AStarPlanner(graph)
    through: list[tuple[int, int]] = []
    around: list[tuple[int, int]] = []
    for pickup in pickups:
        for drop in drops:
            if pickup == drop:
                continue
            route = planner.route(pickup, drop)
            if route is None:
                continue
            if choke_edge is not None and choke_edge in planner.route_edges(route):
                through.append((pickup, drop))
            else:
                around.append((pickup, drop))
    return through, around


def generate(
    graph: Graph,
    *,
    seed: int,
    count: int,
    waves: int = 4,
    wave_interval_ms: int = 20_000,
    choke_edge: int | None = None,
    pickup_nodes: list[int] | None = None,
    drop_nodes: list[int] | None = None,
    zone_of=None,
    both_directions: bool = True,
) -> TaskSet:
    """Build a reproducible task set for one benchmark run.

    ``both_directions`` mirrors each pickup/drop pair, so traffic crosses the
    corridor in both directions. Without it every robot travels the same way and a
    single-lane corridor never sees a head-on approach -- which would quietly
    exclude the case FR-5.10 and TC-2 exist for.
    """
    if count < 1:
        raise ValueError(f"a task set needs at least one task, got {count}")
    if waves < 1:
        raise ValueError(f"waves must be positive, got {waves}")

    rng = random.Random(seed)
    pickups = sorted(pickup_nodes if pickup_nodes is not None else graph.nodes)
    drops = sorted(drop_nodes if drop_nodes is not None else graph.nodes)

    through, around = _classify_pairs(graph, pickups, drops, choke_edge)
    if both_directions:
        # Reversing a pair that crosses the corridor still crosses it.
        through += [(d, p) for p, d in through]
        around += [(d, p) for p, d in around]

    warnings: list[str] = []
    if choke_edge is not None and not through:
        warnings.append(
            "no candidate pickup/drop pair routes through the declared choke "
            "corridor, so FR-10.9 cannot be satisfied on this map"
        )

    wanted_through = round(count * CHOKE_TARGET) if through else 0
    wanted_around = count - wanted_through
    if not around and wanted_around:
        # Every pair crosses the corridor. Fine -- more contention, not less.
        wanted_through, wanted_around = count, 0

    chosen: list[tuple[int, int]] = []
    for pool, quantity in ((through, wanted_through), (around, wanted_around)):
        if not pool or quantity <= 0:
            continue
        # Sampled with replacement: a warehouse genuinely does receive repeat
        # orders for the same pickup and drop, and forbidding them would cap the
        # task count at the number of distinct pairs.
        chosen += [pool[rng.randrange(len(pool))] for _ in range(quantity)]
    rng.shuffle(chosen)

    tasks: list[Task] = []
    per_wave = max(1, (len(chosen) + waves - 1) // waves)
    for index, (pickup, drop) in enumerate(chosen):
        wave = index // per_wave
        task = Task(
            task_id=index + 1,
            pickup=pickup,
            drop=drop,
            priority=_pick_priority(rng),
            created_at_ms=wave * wave_interval_ms,
            zone_id=zone_of(pickup) if zone_of is not None else NO_ZONE,
        )
        tasks.append(task)

    choke_pairs = set(through)
    choke_tasks = sum(1 for t in tasks if (t.pickup, t.drop) in choke_pairs)

    return TaskSet(
        tasks=tasks,
        seed=seed,
        waves=waves,
        choke_tasks=choke_tasks,
        graph_name=graph.name,
        warnings=warnings,
    )


def generate_for_map(
    graph: Graph,
    raw_map: dict,
    *,
    seed: int,
    count: int,
    waves: int = 4,
    wave_interval_ms: int = 20_000,
    zone_of=None,
) -> TaskSet:
    """Convenience wrapper reading the map's own declared task geography.

    Maps carry ``choke_edge``, ``pickup_nodes`` and ``drop_nodes``, so the
    generator does not have to guess which corridor matters or which nodes are
    order points.
    """
    return generate(
        graph,
        seed=seed,
        count=count,
        waves=waves,
        wave_interval_ms=wave_interval_ms,
        choke_edge=raw_map.get("choke_edge"),
        pickup_nodes=raw_map.get("pickup_nodes"),
        drop_nodes=raw_map.get("drop_nodes"),
        zone_of=zone_of,
    )


def validate_priorities(task_set: TaskSet) -> None:
    """ASM-17: every priority must fit the byte the wire format carries."""
    for task in task_set:
        if not 0 <= task.priority <= MAX_PRIORITY:
            raise ValueError(
                f"task {task.task_id} has priority {task.priority}, outside the "
                f"0-{MAX_PRIORITY} range ASM-17 requires"
            )
