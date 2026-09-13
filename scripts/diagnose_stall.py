"""Run a scenario until it stalls, then say exactly why.

    py scripts/diagnose_stall.py --robots 6 --tasks 36 --waves 1
    py scripts/diagnose_stall.py --robots 10 --tasks 50 --seeds 3
    py scripts/diagnose_stall.py --robots 3 --tasks 24 --config A

Prints the wait-for graph and any cycle in it. A cycle is a deadlock: every member
is waiting on a member, so none can be the one to move first. A chain is not -- it
clears on its own and means congestion rather than breakage.

Exists because reading a frozen fleet off a dump of positions got the diagnosis
wrong twice, at the cost of a reverted fix each time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import scenarios  # noqa: E402
from simulator.scenario import AuctionAllocator, RoundRobinAllocator, build  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose a stalled fleet.")
    parser.add_argument("--scenario", default="bench3", choices=sorted(scenarios.SCENARIOS))
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--tasks", type=int, default=36)
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-ms", type=int, default=1_800_000)
    parser.add_argument(
        "--config",
        choices=("A", "B"),
        default="B",
        help="A = round-robin with no coordination; B = full ROBOTON",
    )
    args = parser.parse_args(argv)

    allocator = RoundRobinAllocator if args.config == "A" else AuctionAllocator
    stalls = 0

    for seed in range(args.seeds):
        sim = build(
            scenarios.get(args.scenario),
            seed=seed,
            allocator=allocator(),
            robots=args.robots,
            task_count=args.tasks,
            waves=args.waves,
        )
        # Driven through Simulation.step, not Engine.run: the allocator is what
        # announces tasks, and stepping the engine alone means nothing is ever
        # announced and the fleet is idle rather than stalled.
        ticks = args.max_ms // sim.engine.tick_ms
        finished = False
        for _ in range(ticks):
            if sim.is_finished:
                finished = True
                break
            sim.step()
            report = sim.engine.stall_report
            if report is not None and report.is_deadlocked:
                break
        report = sim.engine.stall_report

        print(
            f"--- seed {seed}: "
            f"{'finished' if finished else 'DID NOT FINISH'}, "
            f"{len(sim.completed)}/{len(sim.task_set)} tasks, "
            f"t={sim.engine.now_ms} ms ---"
        )
        if report is None:
            print("  no stall detected")
        else:
            stalls += 1
            for line in report.describe().splitlines():
                print(f"  {line}")
            print("  fleet at that moment:")
            for robot in sim.engine.robots:
                graph = sim.graph
                ahead = (
                    graph.length_mm(robot.edge_id) - robot.progress_mm
                    if robot.edge_id is not None
                    else -1
                )
                print(
                    f"    r{robot.robot_id} {robot.state.value:8s} "
                    f"{robot.current_node:3d}->{str(robot.next_node):>4s} "
                    f"{ahead:6d} mm to node  p={robot.task_priority:3d}  "
                    f"wait={robot.wait_cause}"
                )
        print()

    print(f"{stalls} of {args.seeds} seeds stalled")
    return 1 if stalls else 0


if __name__ == "__main__":
    raise SystemExit(main())
