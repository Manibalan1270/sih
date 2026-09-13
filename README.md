# ROBOTON — Decentralized Edge-AI Fleet Coordination

Software implementation of ROBOTON-SRS-001 v1.0 (SIH 2026, PS 26123).

A fleet of AMRs coordinates with **no central path planner, task allocator or
traffic model**. Each robot learns its own congestion model, plans its own
routes, bids for its own work, and negotiates right of way directly with its
neighbours. The gateway and dashboard exist for human convenience and hold no
state the fleet needs.

## Quick start

```sh
py -m pip install -r requirements.txt
py -m pytest -q                                        # full verification suite
py scripts/visualize_map.py maps/benchmark_map.json --route 0 7
```

## Layout

| Path | Contents |
|---|---|
| `core/` | decision logic: graph, planners, auction, reservation, arbitration, traffic model. No I/O, no wall clock, no simulator awareness. |
| `communication/` | binary message codec and pluggable transport (in-process bus / UDP / ESP-NOW bridge). |
| `simulator/` | headless deterministic engine, plus the Webots controller adapter. |
| `benchmark/` | Configuration A (stop-and-wait) vs B (ROBOTON), the graded experiment. |
| `web/` | Run Control (launches scenarios) and Fleet Dashboard (strictly read-only). |
| `maps/` | warehouse topologies. |
| `tests/` | unit tests, SRS section 6.2 test cases, architectural invariants. |

## Scenarios

| Scenario | Robots | Engine | What it proves |
|---|---|---|---|
| `bench3` | 3 | headless, seeded | AC-2 (zero collisions) and AC-3 (>=20% makespan reduction) |
| `visual30` | 30 | Webots 3D | coordination visible at fleet density (AC-5, AC-6) |
| `scale100` | 100 | headless | auction traffic stays zone-local (FR-9.6, NFR-1.12) |
| `hardware2` | 2 physical | ESP-NOW bridge | same decision code on real robots (NFR-4.7) |

Scenarios are declared as data in `core/scenarios.py`. The coordination logic is
written once and runs unchanged under all four (IF-3.1).

## Architectural invariants

Enforced by `tests/test_architecture.py`, not by convention:

- **Arbitration is pure.** No learned, random or floating-point state on the
  right-of-way path (CON-7, FR-2.9, FR-5.9). Two robots with divergent learned
  models must never both conclude they hold right of way.
- **Every message fits one 250-byte ESP-NOW frame** (IF-4.1, CON-2). This is why
  framing is binary rather than JSON.
- **The dashboard holds no send path to the mesh** (FR-8.4, IF-1.6, BR-7).
- **`core/` reads no wall clock**, so every run is reproducible from its seed
  (NFR-4.4).
- **Benchmark evidence never comes from Webots**, which is wall-clock bound.

## Known SRS defects

All found while implementing; all should be corrected in v1.1.

1. **Graph size.** ASM-2 and CON-10 cap node and edge ids at 8 bits (<=255
   edges), but Appendix D sizes the memory budget for 400 edges. Worked around
   with a configurable id width: 8-bit everywhere except `scale100`.
2. **Auction frame budget.** NFR-1.12 budgets <=40 frames/task "at 100 AMRs
   across 6 zones", but that figure assumes only the task's own zone bids.
   FR-9.3 and FR-4.14 require adjacent zones to bid too, which on 6 zones
   triples the traffic. Resolved by partitioning the scale map into 24 zones,
   which meets the budget with adjacent-zone eligibility honoured. See
   `ZONE_BUDGET_NOTE` in `scripts/generate_maps.py`.
3. **INTENT carries no task identity.** FR-4.8 requires that where two AMRs claim
   the same task, the lower `robot_id` retains it and the other relinquishes
   within one auction cycle. CLAIM is a one-shot frame and IF-4.5 forbids
   requiring retransmission, so a robot whose peer's CLAIM was lost has no way to
   discover the collision -- it keeps the task and does the work twice. Measured
   at 30% packet loss before the fix: 13 completions for 9 tasks, with one task
   held by two robots at once. Resolved by adding `held_task_id` to INTENT, which
   makes the healing self-repairing on the 200 ms heartbeat the SRS already
   relies on elsewhere. Costs 2 bytes.
4. **Appendix A has no transition for work being taken away.** FR-4.8 creates
   one: a robot that must relinquish may be part-way through the job. The robot
   has not failed, the route is intact, and the task is not unreachable, so none
   of Appendix A's events describe it. Added as `TASK_WITHDRAWN`.
5. **Three required message types are missing from the section 3.4 table**:
   COMPLETE (Appendix A has AT_DROP broadcast it), BLOCKAGE (FR-6.1) and BEACON
   (FR-7.4). All three are implemented.

## Status

Phases 0-1 complete (skeleton, graph, zones, maps, visualiser). See the plan for
the remaining build order.
