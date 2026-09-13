"""Run one scenario and report what happened.

    py scripts/run_scenario.py bench3 --seed 42
    py scripts/run_scenario.py bench3 --seed 42 --events collision,assign
    py scripts/run_scenario.py scale100 --seed 1 --max-ms 600000

This is the developer and evaluator entry point (section 2.3's Developer /
Evaluator user class). It is not the benchmark: a single run is not admissible as
evidence under section 6.3, which requires at least ten seeds per configuration
reported as mean and standard deviation. Use ``benchmark/runner.py`` for that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import scenarios  # noqa: E402
from simulator import scenario as scenario_module  # noqa: E402

DEFAULT_MAX_MS = 1_800_000  # 30 minutes of simulated time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a ROBOTON scenario.",
        epilog="A single run is not benchmark evidence; see benchmark/runner.py.",
    )
    parser.add_argument("scenario", choices=sorted(scenarios.SCENARIOS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--robots", type=int, default=None,
                        help="override the scenario's fleet size")
    parser.add_argument("--tasks", type=int, default=None,
                        help="override the task count (default: 4 per robot)")
    parser.add_argument("--waves", type=int, default=4,
                        help="release the task set in this many waves")
    parser.add_argument("--max-ms", type=int, default=DEFAULT_MAX_MS,
                        help="simulated time limit")
    parser.add_argument("--events", default="",
                        help="comma-separated event kinds to print "
                             "(collision, assign, announce, state, arrive, note)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    chosen = scenarios.get(args.scenario)
    if chosen.physical_robots and args.robots is None:
        print(
            f"{chosen.name} is a hardware scenario ({chosen.physical_robots} "
            f"physical AMRs) and has no simulated fleet to run. Pass --robots to "
            f"run it in simulation instead.",
            file=sys.stderr,
        )
        return 2

    clamped = scenarios.clamped(chosen)
    if clamped.robots != chosen.robots:
        print(
            f"note: fleet reduced from {chosen.robots} to {clamped.robots} by the "
            f"probed Webots capacity (scripts/probe_webots.py)"
        )

    sim = scenario_module.build(
        clamped,
        seed=args.seed,
        robots=args.robots,
        task_count=args.tasks,
        waves=args.waves,
    )

    for warning in sim.task_set.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not args.quiet:
        print(f"{chosen.purpose}\n")
        print(f"map     {sim.graph.name} "
              f"({len(sim.graph.nodes)} nodes, {len(sim.graph.edges)} edges, "
              f"{sim.zones.zone_count} zones, zoning "
              f"{'on' if sim.zones.enabled else 'off'})")
        print(f"tasks   {sim.task_set.describe()}")
        print(f"alloc   {sim.allocator.name}")
        print()

    finished = sim.run(max_ms=args.max_ms)
    print(sim.report())

    if args.events:
        wanted = {kind.strip() for kind in args.events.split(",") if kind.strip()}
        print()
        for event in sim.engine.events:
            if event.kind in wanted:
                print(f"  {event}")

    if not finished:
        print(
            f"\nINCOMPLETE: {len(sim.completed)} of {len(sim.task_set)} tasks done "
            f"within {args.max_ms} ms of simulated time.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
