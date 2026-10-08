"""Route-level reservations: resources, plans, and the table that holds them.

This replaces junction-by-junction arbitration with the model the deadlock-freedom
literature actually proves things about. A robot does not claim one junction at a time;
it books its **whole route** as a sequence of time windows on named resources, against
the routes its peers have already booked, and executes by **precedence** -- entering a
resource only after every robot booked ahead of it there has left. Time decides the
*order*; order is what is enforced. A slow robot delays the robots behind it and never
collides with them, because they do not enter until it has gone.

Three kinds of resource, all derived from the map:

**Region** -- a junction plus JUNCTION_FOOTPRINT_MM into every incident edge. Capacity
one, and traversal is *atomic*: a plan may never wait inside one. This is defect 10's
"hold nowhere inside a conflict region" made structural rather than defended by a rule
at every call site; it is also how lane-annotated navigation treats an intersection (US
11,709,502 B2), with lanes ending short of it so a stopped robot cannot block it.

**Lane** -- the stretch of an edge between the regions at its ends. A two-lane edge is
two lanes, one per direction, each of unlimited capacity but **FIFO**: no overtaking, so
the order robots enter is the order they leave. Waiting is planned only here, at the
lane's end, which is a region boundary -- YIELD_STANDOFF_MM geometry. A single-lane
aisle is one **corridor** from junction to junction, however many bends and pickups
lie along it: capacity one, either direction. A robot may wait at a corridor's end,
as on a lane -- it holds the corridor until it enters what comes next, and nobody
else can be in it meanwhile -- and may stop inside one at a pickup on the aisle. One
resource per stretch, not per edge, is what makes that safe: a robot collecting
mid-aisle holds the whole stretch, so nobody can drive in at the far end and meet it.

**Station** -- a dead-end spur and its leaf, beyond the anchor's region: a pickup, a
drop, or a bay. Capacity one. Waiting here is the point, so it is allowed. Maps are
well-formed (Graph.is_well_formed), which is what makes a station never anyone's way
through.

The table is local, holds no learned state, and expires on its own, for the reasons
core/reservation.py gives; those properties carry over unchanged and the same AST test
enforces them here. Everything is integer milliseconds (CON-6, FR-3.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core import config
from core.graph import Graph

REGION = 0
LANE = 1
CORRIDOR = 2
STATION = 3

_KIND_NAMES = ("region", "lane", "corridor", "station")


@dataclass(frozen=True, slots=True)
class Resource:
    """One bookable thing. Hashable and integer-only so it can key a table and, later,
    be named on the wire."""

    kind: int
    key: int
    """Node id for a region or station; edge id for a lane; for a corridor, the lowest
    edge id of the junction-to-junction stretch it covers."""
    to_node: int = -1
    """For a two-lane lane, the direction of travel: which end it runs towards. -1 for
    every other kind, whose identity does not depend on direction."""

    @property
    def capacity_one(self) -> bool:
        return self.kind != LANE

    @property
    def atomic(self) -> bool:
        """No waiting inside. True only for regions."""
        return self.kind == REGION

    def __str__(self) -> str:
        name = _KIND_NAMES[self.kind]
        if self.kind == LANE:
            return f"{name}(e{self.key}->{self.to_node})"
        prefix = "J" if self.kind in (REGION, STATION) else "e"
        return f"{name}({prefix}{self.key})"


@dataclass(frozen=True, slots=True)
class Step:
    """One resource, held for one window."""

    resource: Resource
    enter_ms: int
    exit_ms: int

    def __post_init__(self) -> None:
        if self.exit_ms < self.enter_ms:
            raise ValueError(f"step on {self.resource} exits before it enters")

    def __str__(self) -> str:
        return f"{self.resource}[{self.enter_ms},{self.exit_ms}]"


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """A robot's booked route: the node sequence, and the resources with windows.

    ``committed_ms`` is what settles a race. When two plans overlap on a resource, the
    one committed earlier on the fleet clock stands and the other replans; ties fall to
    the Appendix C total order. Every robot computes the same answer from the same
    integers. ``plan_seq`` lets a peer tell a fresh plan from a repeat of the old one.
    """

    robot_id: int
    plan_seq: int
    priority: int
    committed_ms: int
    nodes: tuple[int, ...]
    steps: tuple[Step, ...]

    @property
    def start_ms(self) -> int:
        return self.steps[0].enter_ms if self.steps else self.committed_ms

    @property
    def end_ms(self) -> int:
        return self.steps[-1].exit_ms if self.steps else self.committed_ms

    def __str__(self) -> str:
        route = "->".join(map(str, self.nodes))
        return f"r{self.robot_id} plan#{self.plan_seq} {route} [{self.start_ms},{self.end_ms}]"


# ---------------------------------------------------------------------------
# Geometry: which resources a map has, and how long each takes
# ---------------------------------------------------------------------------


class ResourceModel:
    """Derives the resources of a map and the time each takes to traverse.

    Built once per graph. Nothing here depends on traffic; it is the map read through
    the lens of "what can conflict with what", which is why regions are sized by the
    corner geometry (config.JUNCTION_FOOTPRINT_MM) and not by the node's dot.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._degree = {n: graph.degree(n) for n in graph.nodes}
        """Undirected: what makes a node a junction is that aisles cross there, not
        how many of them lead out of it."""
        self._corridor_key = self._corridor_stretches()
        """Single-lane edge id -> the key of the corridor it belongs to."""

    def _corridor_stretches(self) -> dict[int, int]:
        """Group single-lane edges into junction-to-junction stretches.

        Edges chain through any node without a region -- a bend, or a pickup on the
        aisle -- so a stretch ends only where a region or a station begins. Each
        stretch is keyed by its lowest edge id, a choice every robot makes the same
        way from the same map.
        """
        graph = self.graph
        keys: dict[int, int] = {}
        for edge in graph.edges.values():
            if not edge.single_lane or edge.id in keys:
                continue
            stretch, frontier = {edge.id}, [edge.u, edge.v]
            while frontier:
                node = frontier.pop()
                if self.has_region(node) or self.is_station(node):
                    continue
                for other, edge_id in graph.neighbours(node):
                    if edge_id not in stretch and graph.edge(edge_id).single_lane:
                        stretch.add(edge_id)
                        frontier.append(other)
            key = min(stretch)
            for edge_id in stretch:
                keys[edge_id] = key
        return keys

    # -- classification ------------------------------------------------------

    def is_station(self, node: int) -> bool:
        """A leaf: exactly one edge. Pickups, drops and bays are all leaves on a
        well-formed map, and a leaf is the one place a robot may stop indefinitely."""
        return self._degree.get(node, 0) == 1

    def has_region(self, node: int) -> bool:
        """Whether crossing ``node`` needs a region booking. Leaves are stations,
        degree-two nodes are just a bend in a lane, and everything else is a junction
        with a corner."""
        return self.graph.node(node).is_junction and self._degree.get(node, 0) >= 3

    def region(self, node: int) -> Resource:
        return Resource(REGION, node)

    def station(self, leaf: int) -> Resource:
        return Resource(STATION, leaf)

    def lane(self, edge_id: int, to_node: int) -> Resource:
        """The resource covering the stretch of ``edge_id`` between the regions at
        its ends, travelled towards ``to_node``.

        A spur answers with its station, whichever way it is being driven. The spur
        and the berth are one stretch of floor and must be one capacity-one
        resource: while they were two -- a station for the robot driving in, a plain
        unbounded lane for the one driving out -- two robots held what they each
        thought was an empty spur and met head-on on it. Worse than the double
        booking, the lane let the leaving robot *release* the station the moment it
        rolled off the berth, so precedence waved the next robot in while the first
        was still on the spur waiting for its junction.
        """
        edge = self.graph.edge(edge_id)
        if self.is_station(edge.u):
            return self.station(edge.u)
        if self.is_station(edge.v):
            return self.station(edge.v)
        if edge.single_lane:
            return Resource(CORRIDOR, self._corridor_key.get(edge_id, edge_id))
        return Resource(LANE, edge_id, to_node)

    def corridor_key(self, edge_id: int) -> int | None:
        """The corridor ``edge_id`` belongs to, or None if it is not single-lane."""
        return self._corridor_key.get(edge_id)

    # -- durations, all at nominal speed --------------------------------------

    @staticmethod
    def _ms_for(distance_mm: int) -> int:
        return max(0, distance_mm) * 1000 // config.NOMINAL_SPEED_MM_S

    @property
    def region_cross_ms(self) -> int:
        """Boundary to boundary through the node: two footprints."""
        return self._ms_for(2 * config.JUNCTION_FOOTPRINT_MM)

    def lane_length_mm(self, edge_id: int) -> int:
        """The edge less the footprint of each junction end. May be as short as zero
        where two regions nearly touch; never negative, which the map invariant
        TestConflictRegionsDoNotOverlap guarantees."""
        edge = self.graph.edge(edge_id)
        length = edge.length_mm
        if self.has_region(edge.u):
            length -= config.JUNCTION_FOOTPRINT_MM
        if self.has_region(edge.v):
            length -= config.JUNCTION_FOOTPRINT_MM
        return max(0, length)

    def lane_ms(self, edge_id: int) -> int:
        return self._ms_for(self.lane_length_mm(edge_id))

    def station_depth_mm(self, leaf: int) -> int:
        """From the anchor's region boundary to the leaf."""
        (anchor, edge_id), = self.graph.neighbours(leaf)
        depth = self.graph.edge(edge_id).length_mm
        if self.has_region(anchor):
            depth -= config.JUNCTION_FOOTPRINT_MM
        return max(0, depth)

    def station_in_ms(self, leaf: int) -> int:
        return self._ms_for(self.station_depth_mm(leaf))

    def anchor_of(self, leaf: int) -> tuple[int, int]:
        """``(anchor_node, spur_edge)`` for a station leaf."""
        (anchor, edge_id), = self.graph.neighbours(leaf)
        return anchor, edge_id


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Booking:
    """One robot's step on one resource, as the table indexes it.

    ``step_index`` is the step's position in that robot's plan. INTENT carries how
    many steps a robot has released (``plan_index``), so "has it left this resource"
    is ``plan_index > step_index`` -- an integer comparison against a broadcast
    counter, which is what makes precedence checkable without a position estimate.
    """

    robot_id: int
    plan_seq: int
    enter_ms: int
    exit_ms: int
    step_index: int = 0

    def overlaps(self, enter_ms: int, exit_ms: int, margin_ms: int) -> bool:
        return self.enter_ms - margin_ms <= exit_ms and enter_ms - margin_ms <= self.exit_ms


@dataclass
class ResourceTable:
    """One robot's view of every peer's booked route, indexed by resource.

    Two questions are asked of it. The planner asks "when is this resource free?", and
    the executor asks "who is booked ahead of me here, and have they left?". Both are
    answered from bookings alone; no learned state, no randomness.
    """

    margin_ms: int = config.MARGIN_MS
    plans: dict[int, RoutePlan] = field(default_factory=dict)
    _index: dict[Resource, list[Booking]] = field(default_factory=dict, repr=False)
    _progress: dict[int, tuple[int, int]] = field(default_factory=dict, repr=False)
    """robot_id -> (plan_seq, plan_index) from its latest INTENT: the step it stands on."""

    def __len__(self) -> int:
        return len(self.plans)

    # -- recording ------------------------------------------------------------

    def replace_plan(self, plan: RoutePlan) -> None:
        """Enter a robot's plan, discarding its previous one entirely.

        A new plan_seq cancels the old plan: a robot that has rerouted no longer needs
        the resources it was going to use, and keeping them booked would make peers
        wait for a robot that is never coming.
        """
        self.drop_robot(plan.robot_id)
        self.plans[plan.robot_id] = plan
        for index, step in enumerate(plan.steps):
            booking = Booking(plan.robot_id, plan.plan_seq, step.enter_ms, step.exit_ms, index)
            bucket = self._index.setdefault(step.resource, [])
            bucket.append(booking)
            bucket.sort(key=lambda b: (b.enter_ms, b.robot_id))

    def note_progress(self, robot_id: int, plan_seq: int, plan_index: int) -> None:
        """Record which step of which plan a robot says it is on (its INTENT)."""
        self._progress[robot_id] = (plan_seq, plan_index)

    def drop_robot(self, robot_id: int) -> int:
        """Forget a robot's plan. Returns how many bookings went with it."""
        self._progress.pop(robot_id, None)
        plan = self.plans.pop(robot_id, None)
        if plan is None:
            return 0
        removed = 0
        for step in plan.steps:
            bucket = self._index.get(step.resource)
            if not bucket:
                continue
            kept = [b for b in bucket if b.robot_id != robot_id]
            removed += len(bucket) - len(kept)
            if kept:
                self._index[step.resource] = kept
            else:
                del self._index[step.resource]
        return removed

    def expire(self, now_ms: int, grace_ms: int = config.PEER_TIMEOUT_MS) -> int:
        """Drop plans that ended more than ``grace_ms`` ago (FR-5.11)."""
        stale = [rid for rid, plan in self.plans.items() if plan.end_ms + grace_ms < now_ms]
        for rid in stale:
            self.drop_robot(rid)
        return len(stale)

    # -- querying -------------------------------------------------------------

    def bookings(self, resource: Resource, *, exclude_robot: int = -1) -> tuple[Booking, ...]:
        """Every booking on ``resource`` in enter order, robot id breaking ties.

        A capacity-one booking its robot is already standing on -- by its INTENT --
        counts as entered from the start of time, whatever window it was booked for.
        A robot may enter early (order gates entry, not time), and its booking then
        lies about where it is: the table offered the gap in front of it as free, a
        peer booked that gap, and precedence put the peer first onto ground the
        early robot was already on (visual30 seed 5, r12 and r25 at C11R02). Pinned,
        the occupant is ahead of everyone and no window before its exit is free.
        Only capacity-one: a lane holds many, and its order is FIFO by entry.
        """
        bucket = [b for b in self._index.get(resource, ()) if b.robot_id != exclude_robot]
        if not resource.capacity_one or not self._progress:
            return tuple(bucket)
        pinned = False
        for i, b in enumerate(bucket):
            if b.enter_ms > 0 and self._holds(b):
                bucket[i] = Booking(b.robot_id, b.plan_seq, 0, b.exit_ms, b.step_index)
                pinned = True
        if pinned:
            bucket.sort(key=lambda b: (b.enter_ms, b.robot_id))
        return tuple(bucket)

    def occupied(self, resource: Resource, *, exclude_robot: int = -1) -> bool:
        """Whether another robot says, by its INTENT, that it is standing on this
        capacity-one resource now -- whatever its booking says about when it leaves.
        """
        if not resource.capacity_one:
            return False
        return any(
            b.robot_id != exclude_robot and self._holds(b) for b in self._index.get(resource, ())
        )

    def _holds(self, b: Booking) -> bool:
        """Whether the robot behind ``b`` is on that step by its INTENT -- or is
        crossing the region just before it, which commits it to this step: a region
        is never a place to stop, so a robot inside one will be on the next step
        whatever its booking says. Counting only the step it reports let a peer book
        the corridor beyond in front of a robot already crossing into it, and both
        drove in (visual30 seed 7, r9 and r29 at C11R00).
        """
        progress = self._progress.get(b.robot_id)
        if progress is None or progress[0] != b.plan_seq:
            return False
        if progress[1] == b.step_index:
            return True
        if progress[1] != b.step_index - 1:
            return False
        plan = self.plans.get(b.robot_id)
        return (
            plan is not None
            and plan.plan_seq == b.plan_seq
            and 0 < b.step_index < len(plan.steps)
            and plan.steps[b.step_index - 1].resource.kind == REGION
        )

    def is_free(
        self, resource: Resource, enter_ms: int, exit_ms: int, *, exclude_robot: int = -1
    ) -> bool:
        """Capacity-one test: no other booking within the margin of the window."""
        for booking in self.bookings(resource, exclude_robot=exclude_robot):
            if booking.overlaps(enter_ms, exit_ms, self.margin_ms):
                return False
        return True

    def blocking(
        self, resource: Resource, enter_ms: int, exit_ms: int, *, exclude_robot: int = -1
    ) -> Booking | None:
        """The earliest-ending booking that makes ``is_free`` false, or None."""
        found: Booking | None = None
        for booking in self.bookings(resource, exclude_robot=exclude_robot):
            if booking.overlaps(enter_ms, exit_ms, self.margin_ms):
                if found is None or booking.exit_ms < found.exit_ms:
                    found = booking
        return found

    def predecessors(
        self, resource: Resource, enter_ms: int, *, exclude_robot: int
    ) -> tuple[Booking, ...]:
        """Bookings that enter ``resource`` before ``enter_ms``: the robots that must
        have *left* before the asker may enter. Precedence, not timing."""
        return tuple(
            b
            for b in self.bookings(resource, exclude_robot=exclude_robot)
            if (b.enter_ms, b.robot_id) < (enter_ms, exclude_robot)
        )

    def lane_exit_after(
        self, resource: Resource, enter_ms: int, *, exclude_robot: int
    ) -> int:
        """FIFO on a lane: the earliest exit consistent with everyone already ahead.

        Whoever entered before must leave before, by at least a following gap. Returns
        0 when nobody is ahead.
        """
        latest = 0
        for booking in self.bookings(resource, exclude_robot=exclude_robot):
            if booking.enter_ms < enter_ms and booking.exit_ms + config.FOLLOW_GAP_MS > latest:
                latest = booking.exit_ms + config.FOLLOW_GAP_MS
        return latest

    def cuts_in_front_of(
        self, resource: Resource, enter_ms: int, exit_ms: int, *, exclude_robot: int
    ) -> Booking | None:
        """FIFO violated from the other side: someone booked to enter *after* me who
        would then leave *before* me. I cannot change their plan, so I must not take
        that slot. Returns the offender, or None."""
        for booking in self.bookings(resource, exclude_robot=exclude_robot):
            if booking.enter_ms > enter_ms and booking.exit_ms < exit_ms + config.FOLLOW_GAP_MS:
                return booking
        return None

    def plan_ends(self, *, exclude_robot: int = -1) -> set[int]:
        """The last node of every live plan: where somebody is going to stand."""
        return {
            plan.nodes[-1]
            for rid, plan in self.plans.items()
            if rid != exclude_robot and plan.nodes
        }

    def summary(self) -> str:
        if not self.plans:
            return "no plans"
        return "; ".join(str(self.plans[rid]) for rid in sorted(self.plans))


def conflict_between(mine: RoutePlan, theirs: RoutePlan, margin_ms: int) -> Resource | None:
    """The first resource on which two plans cannot both stand, or None.

    Two windows on a capacity-one resource within the margin of each other; or, on
    a lane, one plan entering after the other but leaving before it plus the
    following gap -- an overtake the lane's FIFO forbids. Plans built against each
    other never conflict; this catches the races, when two robots committed before
    either heard the other, and the losses, when a PATH frame never arrived.
    """
    by_resource: dict[Resource, list[Step]] = {}
    for step in theirs.steps:
        by_resource.setdefault(step.resource, []).append(step)
    for step in mine.steps:
        for other in by_resource.get(step.resource, ()):
            res = step.resource
            if res.kind == LANE:
                if step.enter_ms < other.enter_ms and other.exit_ms < step.exit_ms + config.FOLLOW_GAP_MS:
                    return res
                if other.enter_ms < step.enter_ms and step.exit_ms < other.exit_ms + config.FOLLOW_GAP_MS:
                    return res
                continue
            if other.enter_ms - margin_ms <= step.exit_ms and step.enter_ms - margin_ms <= other.exit_ms:
                return res
    return None


# ---------------------------------------------------------------------------
# Wire form: a plan is a node list with one time per node, nothing else
# ---------------------------------------------------------------------------

REST_HOLD_MS = 30_000
"""How long a resting robot's station stays booked per PATH frame. A robot standing
in a bay with nothing to do repeats a one-node plan every PATH_REPEAT_MS, so the
booking never lapses while it is there and lapses on its own once it is gone or
silent -- the same self-healing repetition INTENT uses, and the reason no peer
ever plans into an occupied bay."""


def resting_plan(
    model: ResourceModel,
    *,
    robot_id: int,
    plan_seq: int,
    priority: int,
    since_ms: int,
    now_ms: int,
    leaf: int,
) -> RoutePlan:
    """A plan that goes nowhere: hold this station from ``since_ms``, the moment
    the robot came to rest, until REST_HOLD_MS after ``now_ms``.

    The start never moves. Each refresh extends only the end: a robot queued for
    this bay booked its window after the rest began, and if the rest's start crept
    forward with every repeat that robot would one day find its own booking the
    earlier of the two and drive in -- which is how two idle robots met in a bay.
    """
    return RoutePlan(
        robot_id, plan_seq, priority, now_ms, (leaf,),
        (Step(model.station(leaf), since_ms, now_ms + REST_HOLD_MS),),
    )


def plan_entries(model: ResourceModel, plan: RoutePlan) -> list[tuple[int, int]]:
    """``(node, enter_ms)`` per node of the plan -- everything the wire needs.

    Every step is indexed by a node: a region by its junction, a station by its leaf,
    a lane or corridor by the node it leaves from. So one time per node reconstructs
    the whole plan against the map, and PATH need carry nothing about resources. The
    time is when the robot enters that node's resource -- the region boundary for a
    junction, the berth for a station, the lane start for anything else.
    """
    nodes, steps = plan.nodes, plan.steps
    entries: list[tuple[int, int]] = []
    last = len(nodes) - 1
    cursor = 0
    for index, node in enumerate(nodes):
        if index == last:
            if model.is_station(node) and last > 0:
                entries.append((node, steps[-1].enter_ms))
            elif last == 0:
                entries.append((node, steps[0].enter_ms if steps else plan.committed_ms))
            else:
                entries.append((node, steps[-1].exit_ms if steps else plan.committed_ms))
            break
        if model.has_region(node):
            wanted = model.region(node)
        else:
            edge = model.graph.edge_between(node, nodes[index + 1])
            wanted = model.lane(edge, nodes[index + 1]) if edge is not None else None
            if model.is_station(nodes[index + 1]):
                wanted = model.station(nodes[index + 1])
        while cursor < len(steps) and steps[cursor].resource != wanted:
            cursor += 1
        if cursor >= len(steps):
            raise ValueError(f"plan {plan} has no step for node {node}")
        entries.append((node, steps[cursor].enter_ms))
        if wanted.kind != STATION:
            cursor += 1
        if wanted.kind == REGION and not (model.is_station(nodes[index + 1]) and index + 1 == last):
            # The stretch leaving a junction opens at the region's exit and has no
            # time of its own on the wire. Step past it here rather than rely on
            # the next node's stretch being a different resource: along a corridor
            # with bends or pickups in it, it is the same one.
            edge = model.graph.edge_between(node, nodes[index + 1])
            if edge is not None and cursor < len(steps) and steps[cursor].resource == model.lane(edge, nodes[index + 1]):
                cursor += 1
    return entries


def plan_from_entries(
    model: ResourceModel,
    *,
    robot_id: int,
    plan_seq: int,
    priority: int,
    committed_ms: int,
    entries: list[tuple[int, int]],
) -> RoutePlan:
    """Rebuild a plan from ``plan_entries`` output, against this robot's own map.

    Lanes and corridors end when the next node's resource is entered, which carries
    any wait the sender planned at their end; regions take their fixed crossing; the
    closing station is booked for its depth plus the turnaround, and an opening one
    from the commit until the robot is clear of its spur.
    """
    nodes = tuple(n for n, _ in entries)
    times = [t for _, t in entries]
    steps: list[Step] = []
    last = len(nodes) - 1
    for index, node in enumerate(nodes):
        at = times[index]
        if index == last:
            if model.is_station(node) and last > 0:
                steps.append(
                    Step(model.station(node), at, at + model.station_in_ms(node) + config.STATION_HOLD_MS)
                )
            elif last == 0 and model.is_station(node):
                # Resting: held since ``at``, refreshed from this frame's commit.
                steps.append(Step(model.station(node), at, max(at, committed_ms) + REST_HOLD_MS))
            break
        if index == 0 and model.is_station(node):
            # Leaving a station: berth and spur are one step, held from the commit
            # until the robot is off the spur -- which is when it enters the next
            # node's resource, however long it waited at the spur's end for that
            # window. Booking only the drive-out released the station with the robot
            # still standing on it.
            steps.append(
                Step(
                    model.station(node),
                    min(committed_ms, at),
                    max(at + model.station_in_ms(node), times[index + 1]),
                )
            )
            continue
        lane_from = at
        if model.has_region(node):
            steps.append(Step(model.region(node), at, at + model.region_cross_ms))
            lane_from = at + model.region_cross_ms
        nxt = nodes[index + 1]
        if model.is_station(nxt) and index + 1 == last:
            continue  # the berth is booked at the last node
        edge = model.graph.edge_between(node, nxt)
        if edge is None:
            raise ValueError(f"{node}->{nxt} is not an edge on this map")
        steps.append(Step(model.lane(edge, nxt), lane_from, max(lane_from, times[index + 1])))
    return RoutePlan(robot_id, plan_seq, priority, committed_ms, nodes, tuple(steps))
