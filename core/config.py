"""Frozen configuration constants for ROBOTON.

Every value here traces to the SRS (ROBOTON-SRS-001 v1.0). Appendix E is the
authoritative table for the tunable set; section 5.1 for the latency budgets.

Two rules govern this module:

1. Everything is an ``int``. CON-6 (limited ESP32 floating point) and FR-3.8
   forbid floating-point arithmetic on any path executed inside the 5 ms
   arbitration deadline. Weights that are conceptually fractional are stored as
   Q8 fixed point -- the real value multiplied by 256 -- and applied with
   ``q8_mul``.

2. Nothing here is read at run time from a file or the environment. FR-10.10
   requires the tunable constants to ship as fixed constants with no run-time
   search. The offline search that sets them rewrites *this file*.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fixed-point helpers (CON-6, FR-3.8)
# ---------------------------------------------------------------------------

Q8_ONE = 256
"""Scale factor for Q8 fixed point: a stored value of 256 means 1.0."""


def q8(value: float) -> int:
    """Convert a real weight to Q8 fixed point. Author-time use only.

    Takes a float, so it must never be called on the arbitration path. It
    exists so the constants below can be written readably.
    """
    return round(value * Q8_ONE)


def q8_mul(weight_q8: int, x: int) -> int:
    """Multiply ``x`` by a Q8 weight using integer arithmetic only.

    Truncates toward negative infinity, which keeps the result monotonic in
    ``x`` and so keeps bid and cost ordering stable. FR-5.8 requires identical
    inputs to yield identical decisions, and a rounding mode that varied with
    sign would not.
    """
    return (weight_q8 * x) >> 8


# ---------------------------------------------------------------------------
# Protocol periods (Appendix E)
# ---------------------------------------------------------------------------

INTENT_PERIOD_MS = 200
"""FR-1.1 / NFR-1.4: INTENT broadcast period (5 Hz)."""

INTENT_TOLERANCE_MS = 20
"""FR-1.1: permitted jitter on the INTENT period."""

INTENT_HORIZON = 3
"""Section 3.4.1: future junctions declared in INTENT (next_nodes[3])."""

PEER_TIMEOUT_MS = 1500
"""FR-1.5 / NFR-1.11: silence after which a peer is declared lost."""

AUCTION_WINDOW_MS = 300
"""FR-4.4 / NFR-1.8: sealed bid window, measured from the ANNOUNCE stamp."""

CLAIM_TIMEOUT_MS = 2000
"""FR-4.12: winner must CLAIM within this, else second place re-announces."""

DIGEST_PERIOD_MS = 10_000
"""FR-2.5 / NFR-1.6: traffic-model gossip period (0.1 Hz)."""

PHEROMONE_DECAY_MS = 2000
"""FR-2.3: pheromone evaporation period."""

BEACON_PERIOD_MS = 5000
"""FR-7.4: fleet clock beacon period."""

BEACON_STALE_MS = 60_000
"""FR-5.13: beacon silence past which the safety margin is widened."""

TELEMETRY_PERIOD_MS = 1000
"""NFR-1.7 / FR-8.3: telemetry relay period (1 Hz)."""

MOTION_TICK_MS = 20
"""NFR-1.5: motion control period (50 Hz). Also the simulator's fixed dt, so
every protocol period above is an exact whole number of ticks."""

# ---------------------------------------------------------------------------
# Safety margins (Appendix E, FE-5)
# ---------------------------------------------------------------------------

MARGIN_MS = 800
"""FR-5.3: two reservation windows on one junction conflict when separated by
less than this."""

MARGIN_DEGRADED_MS = 1500
"""FR-5.13 / FR-7.6: widened margin once the clock beacon is stale."""

# ---------------------------------------------------------------------------
# Latency budgets (section 5.1) -- asserted by tests, not enforced at run time
# ---------------------------------------------------------------------------

ARBITRATION_DEADLINE_MS = 5
"""FR-5.7 / NFR-1.1: arbitration decision latency."""

PLAN_DEADLINE_MS = 50
"""FR-3.5 / NFR-1.2: initial route planning latency."""

REPAIR_DEADLINE_MS = 50
"""FR-3.6 / NFR-1.3: D* Lite route repair latency."""

# ---------------------------------------------------------------------------
# Task allocation (FE-4, Appendix E, BR-3)
# ---------------------------------------------------------------------------

QUEUE_CAP = 2
"""FR-4.9 / BR-3 / ASM-18: maximum queued tasks per AMR. This bound is what
keeps insertion-cost enumeration from growing combinatorially."""

# ---- provisional tunable weights ------------------------------------------
# OI-2: the five weights below are provisional. FR-10.10 requires them to be
# fixed by offline search in simulation against the makespan objective, then
# frozen before the graded run. Until benchmark/runner.py has produced that
# search, treat every value here as unvalidated.

ALPHA_Q8 = q8(12.0)
"""alpha -- pheromone weight in edge cost, ms of cost per pheromone unit. A
saturated edge (pheromone 255) adds ~3060 ms, comparable to one nominal aisle
traversal, so congestion can divert a route but not dominate it."""

EPSILON_MS = 300
"""epsilon -- idle-preference bid reduction (FR-4.3, BR-5, TC-14). Expressed
directly in ms of bid, so it needs no fixed-point scaling."""

DELTA_MS = 500
"""delta -- reassignment stability margin (FR-4.10, BR-6, TC-15). A claimed
task changes hands only if a challenger beats the holder by more than this."""

W1_Q8 = q8(40.0)
"""w1 -- battery penalty weight, ms of bid per percentage point below the
penalty knee (FR-4.3)."""

W2_Q8 = q8(0.25)
"""w2 -- task aging weight, ms of bid reduction per ms waited (FR-4.11, BR-2).
At 0.25 a task waiting 20 s has its bid cut by 5000 ms, enough to outbid a
convenient rival, which is how the anti-starvation obligation is discharged."""

BATTERY_PENALTY_KNEE_PCT = 50
"""State of charge below which the battery penalty starts to bite. Above the
knee the penalty is zero, so a healthy fleet bids on insertion cost alone."""

# ---------------------------------------------------------------------------
# Battery (FE-6, BR-4)
# ---------------------------------------------------------------------------

BATTERY_RESERVE_PCT = 20
"""FR-4.15 / FR-6.7 / BR-4: below this an AMR stops bidding, finishes held
work, and routes to a charger."""

BATTERY_RESUME_PCT = 80
"""Appendix A: charge above which a CHARGING AMR returns to IDLE."""

BATTERY_MM_PER_PERCENT = 20_000
"""Millimetres travelled per percentage point of charge: 20 m, so a full charge
covers about 2 km.

This was 2,500 mm (250 m per charge) and that was wrong by an order of magnitude,
with consequences. The SRS treats low battery as an *exception* -- FE-6 groups it
with blocked aisles and peer failure -- but at 250 m a robot flattens after about
six tasks, so charging became the dominant dynamic rather than an exception. On a
24-task run every robot reached 0%, stopped accepting work, and the run stalled with
seven tasks unallocated. It presented as a coordination deadlock and was not one.

2 km is modest but defensible for a small AMR on an ESP32-class platform, and it
puts a benchmark run's ~300 m of travel at roughly 15% of charge -- enough that the
battery terms in the bid still matter, not enough to dominate.

Expressed as mm-per-percent rather than as a Q8 percent-per-mm weight on purpose. At
a 20 ms tick a robot covers 16 mm, and a fractional percent-per-mm rounds to zero in
Q8, so the battery would never discharge at all. Accumulating distance and spending a
whole percent each time the threshold is crossed keeps the arithmetic integer and
exact at every tick rate."""

BATTERY_CHARGE_MS_PER_PERCENT = 400
"""Time at a charger to regain one percentage point, so a full charge takes ~40 s.

A simulation parameter, not an SRS one: no requirement fixes a charge rate, and
OI-5 puts docking mechanics out of scope. Chosen fast enough that charging is a
recoverable excursion rather than the end of a robot's run, and slow enough that
FR-6.7's routing to a charger has a visible cost."""

# ---------------------------------------------------------------------------
# Localization and confidence (FE-7)
# ---------------------------------------------------------------------------

CONFIDENCE_MAX = 255
"""Confidence immediately after a successful ground-marker decode (FR-7.2)."""

CONFIDENCE_THRESHOLD = 128
"""FR-5.14 / NFR-2.3: below this an AMR must not claim a reservation."""

CONFIDENCE_DECAY_PER_METRE = 12
"""FR-7.3: confidence lost per metre since the last marker. At 12 per metre
confidence crosses the threshold after ~10.5 m, which on the benchmark map is
longer than one aisle and shorter than two -- so a robot that misses a single
marker degrades, while a robot reading markers stays trusted."""

# ---------------------------------------------------------------------------
# Traffic model (FE-2)
# ---------------------------------------------------------------------------

EWMA_ALPHA_Q8 = q8(0.25)
"""Smoothing factor for the learned edge-time EWMA (FR-2.4)."""

PHEROMONE_MAX = 255
"""Section 3.4.1: the deposit field is a uint8, so this is a wire limit."""

PHEROMONE_DEPOSIT = 40
"""Pheromone added to an edge on traversal (FR-2.3)."""

PHEROMONE_DECAY_STEP = 8
"""Pheromone removed from every edge each PHEROMONE_DECAY_MS period. Decay is
subtractive rather than multiplicative so it stays integer and reaches exactly
zero -- FR-2.3 requires decay *towards zero*, and an integer multiplicative
rule stalls at 1."""

TRAFFIC_MODEL_MAX_BYTES = 1024
"""FR-2.8 / NFR-1.9: the complete traffic model must fit 1 KB of SRAM."""

# ---------------------------------------------------------------------------
# Wire format (section 3.4, IF-4.1, CON-2, CON-10)
# ---------------------------------------------------------------------------

MAX_FRAME_BYTES = 250
"""IF-4.1 / CON-2: one ESP-NOW frame. Asserted for every message type by
tests/test_architecture.py -- this is why framing is binary, not JSON."""

EDGE_ID_BITS_DEFAULT = 8
"""ASM-2 / CON-10: node and edge identifiers are 8-bit on the wire.

The scale100 scenario overrides this to 16 for the fleet-scale map. Note the
SRS defect being worked around: ASM-2 and CON-10 cap edge ids at 255, but
Appendix D sizes the memory budget for 400 edges. Both cannot hold. ASM-2
itself anticipates the revision -- "packet field widths and memory budget must
be revised".
"""

RADIO_RANGE_MM = 40_000
"""Modelled ESP-NOW broadcast range, 40 m.

Load-bearing for ASM-7, the SRS's own "single most critical assumption": any two
AMRs that can physically collide must be within radio range of one another. 40 m
comfortably covers the 30 x 22 m benchmark map, so bench3 behaves as a flood --
which is correct, since section 4.9 says a flood auction is right at 3-5 AMRs. On
the 92 x 55 m scale map it genuinely limits reach, which is what makes INTENT
delivery proportional to neighbours rather than to fleet size (FR-9.4)."""

MAX_NODES_8BIT = 255
MAX_EDGES_8BIT = 255
MAX_NODES_16BIT = 65_535
MAX_EDGES_16BIT = 65_535

# ---------------------------------------------------------------------------
# Robot geometry and motion
# ---------------------------------------------------------------------------

ROBOT_RADIUS_MM = 250
"""Half-width of an AMR footprint, used by the geometric collision check."""

COLLISION_DISTANCE_MM = 2 * ROBOT_RADIUS_MM
"""Centre separation below which two AMRs have physically overlapped. This is
the *geometric* event; whether it means the algorithm failed is decided by
simulator/collision_detector.py, per the guide's division of responsibility."""

NOMINAL_SPEED_MM_S = 800
"""Commanded cruise speed."""

YIELD_SPEED_MM_S = 240
"""Speed while shedding to push an ETA past a conflicting window (FR-5.6).
Non-zero by design: FR-5.6 and NFR-2.4 require yielding by anticipation, not
by braking to a halt."""

MIN_SPEED_MM_S = 80
"""Floor on commanded speed while still notionally moving."""

AISLE_LANE_OFFSET_MM = 700
"""Lateral offset of a travel lane from an aisle's centre line.

ASM-5 labels every aisle single-lane or bidirectional, and that labelling only means
something if a bidirectional aisle physically fits two AMRs abreast. Robots keep to
their own side, so two travelling in opposite directions have centres 1400 mm apart
-- well clear of the 500 mm collision distance -- while two travelling the same way
share a lane and can still rear-end each other.

Without this the collision detector treats every aisle as single-file and reports a
collision every time two robots pass, which is not a coordination failure at all: it
is the model being wrong. On the benchmark map that accounted for every remaining
collision after junction arbitration was working."""

JUNCTION_OCCUPANCY_MS = 700
"""How long one AMR occupies a junction while crossing it.

Roughly the time to clear its own footprint at cruise: 500 mm of diameter at
800 mm/s is 625 ms, rounded up for the approach and exit. Fixed rather than
computed per robot because ASM-6 makes the fleet homogeneous in speed and size;
varying it would imply a heterogeneity the specification does not permit and OI-4
records as out of scope."""

ENTRY_COMMIT_MM = 150
"""Distance into a single-lane corridor past which a robot is committed.

Not zero, because arrival overshoot carries into the next edge: a robot that
reaches a node mid-tick starts the following edge already a few millimetres along,
so a test for "exactly at the entry" never fires. That defect presented as corridor
arbitration being skipped entirely while appearing to be implemented."""

FOLLOW_GAP_MS = 1600
"""Time gap a robot keeps behind a peer it is following along an edge.

Separation in *time*, not distance, and that is the point: it is the same temporal
reasoning junction arbitration already uses (FR-5.3's margin), and it lets a robot
hold station by *shedding speed* rather than by halting -- which is what FR-5.6 and
NFR-2.4 actually prescribe.

Sized at twice FR-5.3's 800 ms margin, so a follower stays outside the window the
leader would claim at the junction ahead and the two never contend for it.

The reactive distance rule below remains as a safety net for things that do not
broadcast (IF-2.4, FR-6.6). The difference matters: reactive proximity halting is
*literally* how §1.5 defines stop-and-wait, so using it against a cooperating peer
puts the behaviour this project exists to replace inside Configuration B. Measured
before this existed: 44% of all hold-time."""

FOLLOWING_DISTANCE_MM = 1200
"""Headway a robot keeps from another ahead of it in the same lane.

Nothing in FE-5 covers this: junction arbitration resolves who crosses *a node*
first and says nothing about two robots travelling the same aisle in the same
direction, where the one behind can simply drive into the one in front. But NFR-2.1
permits zero inter-robot collisions, so something must handle it.

IF-2.4 already requires the firmware to "detect a non-cooperative obstacle ahead
within the stopping distance at maximum commanded speed", and a robot ahead in the
same lane is exactly what that sensor sees. So headway is kept as a local reactive
behaviour driven by a proximity reading, not as a coordination protocol -- which is
also what makes it work against a peer whose radio has failed.

2.4x the collision distance, so a robot brakes with room rather than to the
millimetre."""

JUNCTION_CLEARANCE_MM = 900
"""How far short of a junction a yielding robot stops.

FR-5.6 prefers shedding speed to braking, and this is the other half of doing that
properly: shed speed early, then hold at the line if the conflict has not cleared.
A robot that merely crawls keeps closing on the junction, and two robots converging
on the same node from different aisles come within a footprint of each other before
either has entered it -- which the collision detector correctly reports and which
arbitrating *entry* alone does not prevent."""

JUNCTION_FOOTPRINT_MM = 1200
"""How far from a junction two robots on *perpendicular* edges can still collide.

Derived from the lane offset, not chosen. Robots keep AISLE_LANE_OFFSET_MM to one side
of the aisle centre (ASM-5), and the offset is a consistent perpendicular -- which
means that on two perpendicular aisles the two offset lanes *cross* near the junction.
For one robot d1 short of the node and another d2 past it, separation is

    sqrt((d1 - AISLE_LANE_OFFSET_MM)**2 + (d2 - AISLE_LANE_OFFSET_MM)**2)

which falls below COLLISION_DISTANCE_MM whenever (d1, d2) lies within
COLLISION_DISTANCE_MM of (700, 700) -- so for either distance anywhere up to about
1200 mm. Both robots can be well over a metre from the node, on edges that do not
share a lane, and still be 500 mm apart.

Only the arrive-while-the-other-departs pairing is exposed. Both-arriving gives
sqrt((d1-L)**2 + (d2+L)**2) and both-departing sqrt((d1+L)**2 + (d2-L)**2), and
neither can fall below the lane offset itself. A turn is what crosses the lanes.

**Not yet enforced, and this is the open defect.** JUNCTION_OCCUPANCY_MS reserves a
junction for a fixed 700 ms, which is 560 mm of travel at NOMINAL_SPEED_MM_S and only
168 mm at YIELD_SPEED_MM_S -- against a footprint that extends 1200 mm each side. A
yielding robot therefore outlives its own claim and crosses the corner unreserved.
Every remaining bench3 failure at 3 AMRs is this. See tests/unit/test_geometry.py,
which pins the derivation, and README's defect 10.
"""

YIELD_STANDOFF_MM = JUNCTION_CLEARANCE_MM + COLLISION_DISTANCE_MM
"""How far short of a junction a yielding robot actually stops.

Must be strictly greater than the radius at which a robot counts as *occupying* the
node ahead, and that distinction is the whole point. Those were the same constant,
which meant a yielding robot came to rest exactly on the occupancy boundary of the
junction it had just given away -- so it blocked the winner's approach to it. Measured
as a two-robot wait-for cycle: r1 yielded J2 to r3 on priority, stopped at the line,
and r3 then could not reach J2 because r1 was standing in its footprint.

One robot diameter of separation is enough to keep the yielder clear of the node while
still leaving it at the last passing point FR-5.10 asks for.

Deliberately *not* written as ``NODE_OCCUPANCY_MM + COLLISION_DISTANCE_MM``, even
though it must stay above NODE_OCCUPANCY_MM. Tying the two together means every
widening of the sensor radius also pushes yielders further back, and yielders standing
further back are themselves an obstruction: widening the radius to 1300 mm that way
turned one collision into two stalls. The sensor radius answers "is the node ahead
occupied"; this answers "where do I stop". They are bounded relative to each other,
not equal."""

ARBITRATION_LOOKAHEAD_MS = 4000
"""How far ahead of a junction a robot begins arbitrating.

Must exceed the safety margin plus one INTENT period, or a robot could arrive at a
junction before it had a chance to detect a conflict there. At 800 mm/s this is
3.2 m of approach, which on the benchmark map is most of an aisle -- so a conflict
is seen well before the last passing point FR-5.10 requires the decision at."""

# ---------------------------------------------------------------------------
# Baseline controller, Configuration A (FR-10.5)
# ---------------------------------------------------------------------------

STOP_WAIT_RADIUS_MM = 1800
"""FR-10.5: fixed radius within which the stop-and-wait baseline halts.
Deliberately larger than COLLISION_DISTANCE_MM -- a reactive controller with no
intent exchange must be conservative, and that conservatism is precisely the
cost the benchmark measures."""

STOP_WAIT_RESUME_HYSTERESIS_MM = 200
"""Extra clearance required before a halted baseline AMR resumes, so it does
not chatter on the radius boundary."""

# ---------------------------------------------------------------------------
# Exception handling (FE-6)
# ---------------------------------------------------------------------------

OBSTACLE_WAIT_MS = 3000
"""FR-6.6: dwell before repairing a route around a non-cooperative obstacle."""

CONTENDED_EDGE_PENALTY_MS = 20_000
"""Cost a robot adds, in its own view only, to an edge it is routing around because a
peer is parked on it (Appendix B's AVOID, second branch).

Comparable to a long detour on the benchmark map, so an alternative wins where one
exists and the edge is still chosen when it is the only way. Never applied to the
graph: the aisle is passable, merely occupied, and marking the graph would make one
robot's problem the whole fleet's."""

CONTENDED_EDGE_TTL_MS = 10_000
"""How long such a penalty lasts.

Expiry is what keeps the avoid-set from becoming permanent route damage: a peer that
has since moved on should stop costing the fleet an aisle. It also makes the mechanism
self-healing if a robot's belief about the blockage was wrong."""

BLOCKED_EDGE_PENALTY_MS = 1_000_000
"""Cost applied to an edge known to be impassable. Large but finite, so an
unreachable goal is reported as UNREACHABLE by search exhaustion (FR-3.7)
rather than by arithmetic overflow."""

INFINITE_COST = 1 << 30
"""Sentinel for 'not traversable at all' in the composed edge cost (FE-2).
Chosen to stay well inside a 32-bit signed range after summation."""
