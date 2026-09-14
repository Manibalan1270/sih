"""Did raising YIELD_SPEED_MM_S cost safety at 30-AMR density?

bench3 (3 AMRs) showed zero collisions at yield speeds up to 799, but the AC-2
gate is only three robots. visual30 puts 30 on a zoned map, and the dashboard
reported collisions there. This runs the same seed under both yield speeds so the
comparison isolates that one constant.

    py scripts/check_yield_at_density.py            # seed 1
    py scripts/check_yield_at_density.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import time


def run(scenario_name: str, seed: int, yield_speed: int) -> dict:
    # config is read at call time by the robot, so setting it before build() is
    # enough; each run is otherwise identical and seeded.
    from core import config, scenarios
    from simulator.scenario import AuctionAllocator, build

    config.YIELD_SPEED_MM_S = yield_speed
    scenario = scenarios.get(scenario_name)
    sim = build(scenario, seed=seed, allocator=AuctionAllocator(), task_count=120, waves=4)
    started = time.time()
    finished = sim.run(max_ms=1_800_000)
    return {
        "yield": yield_speed,
        "seed": seed,
        "finished": finished,
        "collisions": len(sim.engine.collisions),
        "coordination_failures": len(sim.engine.coordination_failures),
        "delivered": len({t.task_id for t in sim.completed}),
        "tasks": len(sim.task_set),
        "makespan_ms": sim.makespan_ms,
        "wall_s": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="visual30")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--speeds", type=int, nargs="+", default=[240, 600])
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for speed in args.speeds:
            print(f"running {args.scenario} seed {seed} at yield {speed} mm/s ...", flush=True)
            row = run(args.scenario, seed, speed)
            rows.append(row)
            print(f"  collisions={row['collisions']} "
                  f"coordination_failures={row['coordination_failures']} "
                  f"delivered={row['delivered']}/{row['tasks']} "
                  f"makespan={row['makespan_ms']}ms ({row['wall_s']}s wall)", flush=True)

    print()
    print(f"{'seed':>5} {'yield':>6} {'collisions':>11} {'coord fails':>12} {'makespan_ms':>12}")
    for row in rows:
        print(f"{row['seed']:>5} {row['yield']:>6} {row['collisions']:>11} "
              f"{row['coordination_failures']:>12} {row['makespan_ms']:>12}")

    unsafe = [r for r in rows if r["yield"] != 240 and r["collisions"] > 0]
    baseline_clean = all(r["collisions"] == 0 for r in rows if r["yield"] == 240)
    print()
    if unsafe and baseline_clean:
        print("VERDICT: the raised yield speed costs safety at this density. Revert it.")
    elif unsafe:
        print("VERDICT: collisions at both speeds -- density is the cause, not yield speed.")
    else:
        print("VERDICT: no collisions at either speed on these seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
