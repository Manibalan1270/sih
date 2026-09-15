"""Benchmark runner for Configuration A vs B.

This is the missing Step 6 evidence layer described in README.md.
It creates an identical seeded task set for each configuration, runs the same
scenario under each allocator, and summarizes the measured evidence needed for
AC-2 and AC-3.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from core import scenarios
from simulator.scenario import AuctionAllocator, RoundRobinAllocator, build


@dataclass(frozen=True)
class SeedResult:
    seed: int
    config_name: str
    tasks_total: int
    tasks_completed: int
    makespan_ms: int
    collisions: int
    coordination_failures: int
    deadlocked: bool
    stopped_time_ms: int
    precedence_hold_ms: int
    yields_lost: int
    yields_won: int
    paths_sent: int
    path_frames_seen: int
    completion_mean_ms: float
    """§6.3 primary metric: mean time from announcement to completion, over the
    tasks this seed finished. Makespan says when the fleet stopped; this says how
    long an order waited, which is the quantity a warehouse actually buys."""

    route_diversity: float
    """§6.3 secondary metric: distinct edges traversed per robot, as a fraction of
    the edges in the map. Low diversity means the fleet is funnelling through the
    same corridor, which is what the choke map exists to provoke."""

    auction_frames_per_task: float
    """§6.3 secondary metric, and the NFR-1.12 budget (<=40 at 100 AMRs)."""
    completed_ids: tuple[int, ...]
    task_signature: tuple[tuple[int, int, int, int, int], ...]


def _task_count_for(scenario, seed: int) -> int:
    # Use the same task generation as the scenario builder: a small default fleet
    # task set is enough for the benchmark runner to exercise the A/B comparison.
    return 4 * scenario.simulated_robots


def run_seed(scenario, *, seed: int, config_name: str, allocator) -> SeedResult:
    sim = build(
        scenario,
        seed=seed,
        allocator=allocator,
        task_count=_task_count_for(scenario, seed),
        waves=4,
    )
    finished = sim.run(max_ms=1_800_000)
    if not finished:
        raise RuntimeError(f"benchmark run did not complete for {config_name} seed {seed}")

    stopped = sum(r.metrics.stopped_ms for r in sim.engine.robots)
    paths_sent = sum(r.metrics.paths_sent for r in sim.engine.robots)
    path_frames_seen = sum(r.metrics.paths_heard for r in sim.engine.robots)

    completions = [t.completion_ms() for t in sim.completed]
    completions = [c for c in completions if c is not None]

    # Route diversity: distinct edges any robot used, against the edges available.
    edges_touched = set()
    for robot in sim.engine.robots:
        edges_touched |= robot.metrics.edges_used
    edge_total = len(sim.graph.edges) or 1

    frames = getattr(getattr(sim.mesh, "stats", None), "auction_frames", None)
    auction_frames = frames() if callable(frames) else 0

    return SeedResult(
        seed=seed,
        config_name=config_name,
        tasks_total=len(sim.task_set),
        tasks_completed=len(sim.completed),
        makespan_ms=sim.makespan_ms,
        collisions=len(sim.engine.collisions),
        coordination_failures=len(sim.engine.coordination_failures),
        deadlocked=sim.engine.stall_report is not None and sim.engine.stall_report.is_deadlocked,
        stopped_time_ms=stopped,
        precedence_hold_ms=sum(
            r.metrics.precedence_hold_ms for r in sim.engine.robots
        ),
        yields_lost=sum(r.metrics.yields_lost for r in sim.engine.robots),
        yields_won=sum(r.metrics.yields_won for r in sim.engine.robots),
        paths_sent=paths_sent,
        path_frames_seen=path_frames_seen,
        completion_mean_ms=statistics.mean(completions) if completions else 0.0,
        route_diversity=len(edges_touched) / edge_total,
        auction_frames_per_task=(
            auction_frames / len(sim.completed) if sim.completed else 0.0
        ),
        completed_ids=tuple(sorted(t.task_id for t in sim.completed)),
        task_signature=tuple(
            (t.task_id, t.pickup, t.drop, t.priority, t.created_at_ms)
            for t in sim.task_set
        ),
    )


def _mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and population SD. §6.3: "a single run is not admissible as evidence",
    so every metric is reported with its spread, not just makespan."""
    values = list(values)
    if not values:
        return 0.0, 0.0
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


#: The metrics §6.3 mandates, as (report key, attribute). Primary first.
_METRICS = (
    ("makespan", "makespan_ms"),
    ("completion", "completion_mean_ms"),
    ("collisions", "collisions"),
    ("coordination_failures", "coordination_failures"),
    ("stopped_time", "stopped_time_ms"),
    ("precedence_hold", "precedence_hold_ms"),
    ("yields_lost", "yields_lost"),
    ("yields_won", "yields_won"),
    ("route_diversity", "route_diversity"),
    ("auction_frames_per_task", "auction_frames_per_task"),
    ("paths_sent", "paths_sent"),
    ("path_frames_seen", "path_frames_seen"),
)


def compare_configurations(results_a: list[SeedResult], results_b: list[SeedResult]) -> dict[str, float]:
    """The §6.3 report: mean and standard deviation of every metric, per configuration.

    The pass condition is B recording zero collisions and a makespan at most 80% of
    A's, so those two are also reported reduced. Everything else is reported because
    §6.3 lists it, not because a criterion turns on it -- secondary metrics are how a
    reader tells a real coordination win from a lucky seed set.
    """
    if not results_a or not results_b or len(results_a) != len(results_b):
        raise ValueError("A and B need equally sized, non-empty result sets")
    for result_a, result_b in zip(results_a, results_b):
        if result_a.seed != result_b.seed or result_a.task_signature != result_b.task_signature:
            raise ValueError(f"A and B did not use the same task set for seed {result_a.seed}")

    report: dict[str, float] = {"seeds": float(len(results_a))}
    for key, attr in _METRICS:
        for tag, results in (("a", results_a), ("b", results_b)):
            mean, sd = _mean_sd([getattr(r, attr) for r in results])
            report[f"{key}_mean_{tag}"] = mean
            report[f"{key}_std_{tag}"] = sd

    # Totals the acceptance criteria are stated in terms of.
    report["collision_total_a"] = sum(r.collisions for r in results_a)
    report["collision_total_b"] = sum(r.collisions for r in results_b)
    report["coordination_failures_a"] = sum(r.coordination_failures for r in results_a)
    report["coordination_failures_b"] = sum(r.coordination_failures for r in results_b)
    report["deadlock_cycles_a"] = sum(r.deadlocked for r in results_a)
    report["deadlock_cycles_b"] = sum(r.deadlocked for r in results_b)

    # AC-3 reads on this one number; AC-2 on collision_total_b.
    mean_a = report["makespan_mean_a"]
    report["makespan_ratio"] = report["makespan_mean_b"] / mean_a if mean_a else 0.0
    mean_ca = report["completion_mean_a"]
    report["completion_ratio"] = (
        report["completion_mean_b"] / mean_ca if mean_ca else 0.0
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Configuration A against B.")
    parser.add_argument("--seeds", type=int, default=10, help="seed count to evaluate")
    parser.add_argument("--scenario", default="bench3", help="scenario name")
    parser.add_argument("--output", type=str, default="benchmark_report.json", help="JSON output path")
    args = parser.parse_args(argv)

    scenario = scenarios.get(args.scenario)
    scenario.validate()

    results_a = [
        run_seed(scenario, seed=seed, config_name="A", allocator=RoundRobinAllocator())
        for seed in range(args.seeds)
    ]
    results_b = [
        run_seed(scenario, seed=seed, config_name="B", allocator=AuctionAllocator())
        for seed in range(args.seeds)
    ]

    summary = compare_configurations(results_a, results_b)
    summary["scenario"] = scenario.name
    summary["seeds"] = args.seeds
    summary["makespan_ratio_b_over_a"] = summary["makespan_mean_b"] / summary["makespan_mean_a"] if summary["makespan_mean_a"] else 0.0

    output = Path(args.output)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
