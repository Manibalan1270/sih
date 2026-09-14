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
py -m web.backend.app                                 # fleet dashboard at http://127.0.0.1:8000
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
5. **Appendix C's deadlock proof does not cover multiple resources.** The proof
   concerns one conflict set under one total order and shows it has a unique
   maximum, so exactly one AMR proceeds. A fleet contends for two kinds of
   resource -- junctions and single-lane corridors -- and ranking them
   independently readmits the cycle: r1 yielded a junction to r3 while r3 yielded
   a corridor to r1, each correctly applying the total order to a different
   resource, and neither moved. Resolved for a corridor and its endpoints by
   merging them into one conflict set. **Still open for a ring of distinct
   corridors** (`tests/coordination/test_safety_cases.py`, xfail): each robot is
   the rightful winner of the segment it wants while blocked by another holding
   the next. The fix is sequence-level reservation per [R8], not a local rule.
6. **Nothing says where an AMR idles, and it matters.** Appendix A has IDLE
   broadcasting INTENT and leaving only on winning an auction. An idle robot on a
   junction is a permanent obstacle: peers stop at their following distance and
   wait for a robot with no reason to move. Two separate deadlocks came from this.
   Resolved with staging bays in every map -- one per robot, never a task endpoint
   -- and an idle robot withdraws to one. Both conditions are necessary: too few
   bays and the queue for a bay blocks the aisle; a bay on a pickup node and the
   jam simply relocates.
7. **No requirement covers following distance.** FE-5 resolves who crosses a
   *node*; nothing addresses two robots travelling one aisle in the same
   direction, where the one behind simply drives into the one in front. NFR-2.1
   permits zero collisions, so it has to be handled.

   This was first implemented as a reactive proximity halt on the forward sensor
   IF-2.4 requires -- and that was a conformance problem, because §1.5 *defines*
   stop-and-wait as halting when a peer comes within a fixed radius, and
   FR-5.6/NFR-2.4 reserve braking for **non-cooperative obstacles**. A fleet peer
   broadcasts INTENT five times a second, so it is not one. Measured at 44% of all
   hold-time: the behaviour this project exists to replace, inside Configuration B.

   Now anticipatory. A peer's INTENT already carries `next_nodes` and `eta_ms`, so a
   follower measures separation in *time* -- the same reasoning junction arbitration
   uses -- and opens the gap by shedding speed, which is what FR-5.6 prescribes. The
   proximity sensor remains only as a last-resort net for things that do not
   broadcast (FR-6.6, ASM-13). Reactive halting is down from 44% to 3% of hold-time.
8. **RESERVE carries no direction, and INTENT's does not survive it.** Section
   3.4.2 gives RESERVE no approach or exit field, so an explicit claim arriving
   after an implied one discarded the geometry and made following traffic look
   like a crossing conflict. A robot then outranked one it was queued behind and
   could not pass. Resolved by carrying the known direction forward rather than by
   widening the frame.
9. **Three required message types are missing from the section 3.4 table**:
   COMPLETE (Appendix A has AT_DROP broadcast it), BLOCKAGE (FR-6.1) and BEACON
   (FR-7.4). All three are implemented.
10. **Junction occupancy is specified as a time, but the conflict it must prevent
    is a distance.** Appendix E fixes `JUNCTION_OCCUPANCY_MS` at 700 ms, and
    arbitration.py justifies a constant by ASM-6's homogeneous fleet. ASM-6 gives a
    homogeneous *nominal* speed, and a robot that yields travels at
    `YIELD_SPEED_MM_S` -- a third of it. So 700 ms buys 560 mm of clearance at
    nominal speed and 168 mm at yield speed, while the geometry that has to be
    covered extends 1200 mm each side of the node (derivation in
    `config.JUNCTION_FOOTPRINT_MM`, pinned by `tests/unit/test_geometry.py`). A
    yielding robot therefore outlives its own reservation and crosses the corner
    unclaimed. **Mostly resolved**, in two halves: the claim now begins
    JUNCTION_FOOTPRINT_MM before arrival instead of at arrival, and a robot that has
    already crossed keeps claiming the junction until clear of the corner (arbitration
    only ever looks at `next_node`, so it previously stopped claiming a junction the
    instant it passed it -- while still standing in it). 3-AMR failures over 20 seeds
    fell from 3 to 1, at a cost of 4.1% makespan.

    Two findings are worth keeping. Sizing the window from the speed actually being
    travelled is self-reinforcing and deadlocks -- a yielding robot drops to a third of
    nominal, tripling its own claim, conflicting with more peers and yielding harder, so
    the window is sized at nominal speed and the crawling case is handled by renewal
    instead. And extending the claim *forwards* past arrival, though it is the
    physically complete statement, holds every junction three times as long and turned
    one failure back into three.

    **Still open:** a robot that halts inside the corner. `_limit_for_clearance` stops
    dead at FOLLOWING_DISTANCE_MM, which can bring a robot to rest 800 mm from a
    junction -- inside the footprint -- and the geometry then works against it, because
    a second robot departing that junction gets *closer* as it leaves (separation falls
    to 100 mm at 700 mm past). This is seed 19 and it is the same shape as the
    YIELD_STANDOFF_MM defect: a robot must not come to rest inside a region it does not
    own. A full fix needs entry to the corner conditional on being able to cross it
    without stopping.
11. **Nothing gives a robot already inside a single-lane corridor the right to
    finish traversing it.** FR-5.10 puts the decision at the last passing point and
    ranks the contenders by the Appendix C total order -- but that order is only
    meaningful between robots that both still have a choice. A robot already
    committed to a single-lane aisle cannot reverse and cannot be passed, so being
    outranked cannot make it leave. Observed on seed 6: r2 outranked r3 and entered
    e4 while r3 was 4524 mm of 6000 into it, and they closed to 492 mm. Resolved by
    treating an opposing occupant as a fact rather than a contender.
12. **Appendix C's total order is applied to robots that cannot act, and its proof
    does not cover them.** The proof takes a set of AMRs "in mutual conflict" and shows
    a total order gives it a unique maximum, so exactly one proceeds and a cyclic wait
    cannot form. Mutual conflict is doing unstated work there: it presumes every member
    could take the resource next. Three cases break it, and each produced either a
    collision or a cycle -- a robot already committed to a single-lane corridor
    (defect 11), a robot standing in a junction's corner where it cannot step aside
    (defect 10), and a robot queued behind another on the approach, which cannot reach
    the resource at all until the one ahead moves.

    That last one is a three-robot cycle on bench3 seed 28: r1 yielded the choke
    corridor to r3 because r3 carried priority 100, while r3 was stuck behind r2 on the
    approach; r2, at the mouth, yielded to r1. Every robot applied the order correctly.
    Resolved by collapsing each approach to its nearest member before ranking
    (`reservation.queue_heads`) -- **for ranking only**, since Appendix B's AVOID asks
    the opposite question and needs every window, the whole queue having to pass before
    the resource is free.

    The SRS needs either a precondition on Appendix C -- that the order ranks only
    robots able to take the resource next -- or an explicit rule that physical occupancy
    is not subject to arbitration. The implementation now assumes the latter.
13. **Task endpoints sit on junctions, and nothing in the SRS forbids it.** FR-10.8
    places pickups and drops as nodes of the topology; it says nothing about their degree.
    On the generated maps 47 of 48 endpoints (warehouse_zoned_30) and all 192
    (warehouse_zoned_100) are 3- or 4-way junctions, so a robot dwelling at a pickup is
    parked in an intersection -- the one place the rule "never come to rest inside a
    conflict region" cannot be honoured, because the task requires the rest.

    This is the structural cause beneath the 30-AMR failures that survived every local
    rule. The literature's solvable class for pickup-and-delivery (Ma, Li, Kumar & Koenig,
    AAMAS 2017, "well-formed" MAPD) requires that between any two endpoints a route exists
    crossing no third endpoint. `Graph.is_well_formed()` measures it: 19 of 28 endpoint
    pairs fail on benchmark_map, 996 of 1,128 on warehouse_zoned_30, 17,738 of 18,336 on
    warehouse_zoned_100. **Resolved** by making every station a degree-1 spur, as the
    parking bays already were: the benchmark map's depots were, and four more hang off
    its station junctions; the generated maps are Kiva-style, with pickup stations off
    the left column (inbound dock), drop stations off the right (packing), and bays off
    the rest of a perimeter ring whose edges are split at their midpoints so every leaf
    has its own anchor -- a bay chained behind another would put one robot on another's
    only way out. `Graph.is_well_formed` now reports no failing pair on any map, and it
    is a plain test rather than an xfail. The SRS should state the condition.

## Status

Phases 0-7b complete. `bench3` at 3 AMRs is **clean across 30 seeds**: zero
collisions, every run completing. 6 AMRs on the same map -- twice its design density,
with three bays for six robots -- also completes without collisions or cycles.

That sequence went 3 failures in 20 seeds -> 1 -> 0 collisions in 30, and the fixes that
got there all turned out to be one idea applied in four places: **a robot that has no
move left is a fact, not a contender.** The Appendix C total order is only meaningful
between robots that both still have a choice, and ranking one that does not have a choice
produces either a collision or a cycle. The four places were a robot already inside a
single-lane corridor (defect 11), a robot standing in a junction's corner (defect 10), a
robot idling on a node that tasks need (defect 7), and a robot queued behind another on
the approach to a contended resource (defect 12).

The approach follows the lane/conflict-region treatment in Google/Intrinsic's
US 11,709,502 B2, *Roadmap annotation for deadlock-free multi-agent navigation*, where
lanes deliberately end a gap short of an intersection so a robot stopped at the end of
one cannot interfere with robots crossing, and robots that cannot cross are held
*outside* the conflict region rather than partway into it. The bounded-lookahead framing
-- each robot needing to examine only a few states ahead rather than the whole
configuration space -- follows the distributed higher-order-deadlock literature, and maps
onto INTENT's existing three-node horizon.

That distinction matters and the earlier wording here hid it. The Phase 6 gate was
read as met because the first 8 seeds pass, and the same 8 seeds were used to clear
Phase 7b. Widening to 20 shows AC-2 is **not** met at 3 AMRs. Every failure seen at this
fleet size has been SRS defect 10 -- the junction-corner footprint -- in one guise or
another: two robots converging in the corner unseen, the corner reading as headway and
closing a wait-for cycle against a junction yield, and a robot coming to rest inside a
corner another robot is leaving. The first two are fixed; the last is seed 19.

Both densities tried at 3 AMRs -- 12 tasks over 4 waves and 24 in a single wave --
behave the same way. TC-3's single-lane ring clears.

### Above 3 AMRs it does not hold, and the gap is wide

Measured on this commit, not projected:

| fleet | map | result |
|---|---|---|
| 3 | benchmark_map | clean, 30 of 30 seeds |
| 6 | benchmark_map | clean, no cycle, no collision |
| 30 (`visual30`) | warehouse_zoned_30 | seed 0 **completes 120 of 120** with 1 collision; seed 1 reaches 83 with 2 and a cycle |
| 100 (`scale100`) | warehouse_zoned_100 | not re-measured since the map fix |

### Where local rules stop working, and what was tried past that point

Four further changes were built after the state above, each grounded and each measured,
and each fixed one seed while breaking another. They are recorded because the pattern is
the result:

- **Convoy priority inheritance** (Sha, Rajkumar & Lehoczky 1990): every robot in a
  same-direction queue asserts the queue's highest priority, so a high-priority robot
  stuck behind a low-priority head is not outranked on the head's behalf. Correctly
  targets a measured inversion at J7. seed 0 unchanged; seed 1 rose from 83 to 98 tasks
  and from 2 to 5 collisions.
- **Opposite-lane departures are not corner occupants.** Geometrically exact -- two
  robots on one bidirectional edge going opposite ways are 2 * offset apart, always.
  Targets a measured six-robot cycle on e5. Deadlocked seed 0 at 66 tasks every time it
  was tried, alone or with the others, by exposing a queue-ranking configuration at J7.
- **Queue-head ranking at junctions**, matching what corridors already do. Fixed the J7
  configuration and raised seed 0's collisions from 1 to 5.
- **Approach half of the window at actual speed.** Targets the one signature behind all
  five seed 1 collisions: a robot crawling behind a leader whose 1500 ms approach claim
  covered 360 mm of a 1200 mm corner, so it entered the corner with its claim seconds in
  the future and stopped 4 mm inside -- 500 mm of separation instead of 507. Deadlocked
  seed 0 at 97.

Each is a correct local statement. Together they show that at 30 AMRs the remaining
failures are not one more missing rule about one junction; they are what one-junction-
at-a-time reasoning cannot resolve. The deadlock-freedom argument in the lane-routing
literature (US 11,709,502 B2) rests on two things this implementation does not yet have:
a spare-capacity condition on every lane and cycle, and routes planned sequentially in
priority order over the *whole path*, not negotiated one resource at a time. The second
is the sequence-level reservation per [R8] that `_arbitrate_corridor` already names as
the fix for ring deadlocks. That is the next piece of work, and it is not a patch.

`visual30` began this work at 24 of 120 tasks with 12 collisions. Part of that was never
coordination at all: the bay generator placed bays *on top of* aisle nodes -- fifteen
coordinates on warehouse_zoned_30 held two or more nodes -- so robots collided on lanes
that shared no node, which no junction arbitration can prevent because there is no
junction there. That is now an invariant (`TestLanesDoNotOverlapWithoutAJunction`) and
both generated maps satisfy it.

The `scale100` figures above predate the map fix and should be read as stale.

This does **not** contradict the zoning result. The 16.4 auction frames per task
measured at 100 AMRs against NFR-1.12's budget of 40 is a statement about
*communication* -- that bidding stays inside a zone rather than crossing the fleet --
and it still holds. Coordination collapsing at the same fleet size is a separate
property, and the earlier reading of "scale100 runs" conflated the two. What runs is
the auction; what fails is the traffic.

So AC-5 is met at 3 AMRs only, and AC-6's `visual30` is not close to demonstrable
regardless of whether Webots is installed.

Three defects behind earlier failures, all of which presented as coordination
deadlocks and none of which were:

1. **A junction-footprint blind spot.** A robot's claim on a junction vanished the
   moment it passed the node, while it was still physically inside the junction.
   Both 6-AMR collisions were this: one robot 264 mm past a node, another 940 mm
   from it, 497 mm apart. Section 3.4.1 already calls `current_node` "the junction
   most recently occupied *or departed*" -- the field exists for exactly this and was
   being used only for position.
2. **Flat batteries.** `BATTERY_MM_PER_PERCENT` gave 250 m per charge, an order of
   magnitude too pessimistic, and nothing implemented Appendix A's charging cycle, so
   `CHARGING` was unreachable. Robots hit 0%, refused work, and runs stalled with
   tasks unallocated. FE-6 groups low battery with blocked aisles and peer failure --
   an *exception* -- but at 250 m it was routine.
3. **Chargers on task endpoints.** A robot that finished charging parked on a node
   that was both charger and drop point, blocking the robot whose task was there.
   Charging now lives on the staging bays, which are never task endpoints.

A correction worth recording: the ring deadlock was diagnosed as a resource-ordering
problem needing sequence-level reservation per [R8]. That was wrong. It was defects
2 and 3, and the ring clears with no change to arbitration. The test had been marked
`xfail(strict=True)`, which is the only reason the mistaken diagnosis was caught
rather than acted on.

Not yet met, and known:

- **AC-3 (>=20% makespan reduction) is not met and is not expected to be yet.**
  Configuration B currently runs slightly *slower* than A, because yielding costs
  time while the two things that pay it back are unbuilt: traffic learning to
  spread routes off the choke corridor (Phase 8), and Configuration A's own
  stop-and-wait halting penalty, which FR-10.5 requires and which is what the
  baseline is supposed to be handicapped by (Phase 11).
- **TC-3's ring case deadlocks** -- see SRS defect 5 above. Junction arbitration
  and single-corridor arbitration are each correct and neither is sufficient.
- **Safety and liveness are unverified above 3 AMRs** -- see above. This outranks
  the AC-3 margin: a makespan figure measured on a fleet that sometimes collides
  or stalls is not evidence of anything.
- **`visual30` has never run**: Webots is not installed.
