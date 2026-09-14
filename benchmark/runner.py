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
    paths_sent: int
    path_frames_seen: int
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
        paths_sent=paths_sent,
        path_frames_seen=path_frames_seen,
        completed_ids=tuple(sorted(t.task_id for t in sim.completed)),
        task_signature=tuple(
            (t.task_id, t.pickup, t.drop, t.priority, t.created_at_ms)
            for t in sim.task_set
        ),
    )


def compare_configurations(results_a: list[SeedResult], results_b: list[SeedResult]) -> dict[str, float]:
    if not results_a or not results_b or len(results_a) != len(results_b):
        raise ValueError("A and B need equally sized, non-empty result sets")
    for result_a, result_b in zip(results_a, results_b):
        if result_a.seed != result_b.seed or result_a.task_signature != result_b.task_signature:
            raise ValueError(f"A and B did not use the same task set for seed {result_a.seed}")

    makespan_a = [r.makespan_ms for r in results_a]
    makespan_b = [r.makespan_ms for r in results_b]

    return {
        "makespan_mean_a": statistics.mean(makespan_a),
        "makespan_mean_b": statistics.mean(makespan_b),
        "makespan_std_a": statistics.pstdev(makespan_a) if len(makespan_a) > 1 else 0.0,
        "makespan_std_b": statistics.pstdev(makespan_b) if len(makespan_b) > 1 else 0.0,
        "collision_total_a": sum(r.collisions for r in results_a),
        "collision_total_b": sum(r.collisions for r in results_b),
        "coordination_failures_a": sum(r.coordination_failures for r in results_a),
        "coordination_failures_b": sum(r.coordination_failures for r in results_b),
        "deadlock_cycles_a": sum(r.deadlocked for r in results_a),
        "deadlock_cycles_b": sum(r.deadlocked for r in results_b),
        "stopped_time_mean_a": statistics.mean(r.stopped_time_ms for r in results_a),
        "stopped_time_mean_b": statistics.mean(r.stopped_time_ms for r in results_b),
        "precedence_hold_mean_b": statistics.mean(
            r.precedence_hold_ms for r in results_b
        ),
        "paths_sent_mean_b": statistics.mean(r.paths_sent for r in results_b),
        "path_frames_seen_mean_b": statistics.mean(
            r.path_frames_seen for r in results_b
        ),
    }


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
