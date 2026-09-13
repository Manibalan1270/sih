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

Both found while implementing; both should be corrected in v1.1.

1. **Graph size.** ASM-2 and CON-10 cap node and edge ids at 8 bits (<=255
   edges), but Appendix D sizes the memory budget for 400 edges. Worked around
   with a configurable id width: 8-bit everywhere except `scale100`.
2. **Auction frame budget.** NFR-1.12 budgets <=40 frames/task "at 100 AMRs
   across 6 zones", but that figure assumes only the task's own zone bids.
   FR-9.3 and FR-4.14 require adjacent zones to bid too, which on 6 zones
   triples the traffic. Resolved by partitioning the scale map into 24 zones,
   which meets the budget with adjacent-zone eligibility honoured. See
   `ZONE_BUDGET_NOTE` in `scripts/generate_maps.py`.

## Status

Phases 0-1 complete (skeleton, graph, zones, maps, visualiser). See the plan for
the remaining build order.
