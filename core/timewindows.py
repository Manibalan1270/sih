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
edge is one **corridor**: capacity one, either direction, and atomic like a region,
because a robot stopped inside a corridor blocks it for whoever it stopped for.

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
    """Node id for a region or station; edge id for a lane or corridor."""
    to_node: int = -1
    """For a two-lane lane, the direction of travel: which end it runs towards. -1 for
    every other kind, whose identity does not depend on direction."""

    @property
    def capacity_one(self) -> bool:
        return self.kind != LANE

    @property
    def atomic(self) -> bool:
        """No waiting inside. True for regions and corridors."""
        return self.kind in (REGION, CORRIDOR)

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
        self._degree = {n: len(graph.neighbours(n)) for n in graph.nodes}

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
        edge = self.graph.edge(edge_id)
        if edge.single_lane:
            return Resource(CORRIDOR, edge_id)
        return Resource(LANE, edge_id, to_node)

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
    """One robot's step on one resource, as the table indexes it."""

    robot_id: int
    plan_seq: int
    enter_ms: int
    exit_ms: int

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
        for step in plan.steps:
            booking = Booking(plan.robot_id, plan.plan_seq, step.enter_ms, step.exit_ms)
            bucket = self._index.setdefault(step.resource, [])
            bucket.append(booking)
            bucket.sort(key=lambda b: (b.enter_ms, b.robot_id))

    def drop_robot(self, robot_id: int) -> int:
        """Forget a robot's plan. Returns how many bookings went with it."""
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
        """Every booking on ``resource`` in enter order, robot id breaking ties."""
        return tuple(
            b for b in self._index.get(resource, ()) if b.robot_id != exclude_robot
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

    def summary(self) -> str:
        if not self.plans:
            return "no plans"
        return "; ".join(str(self.plans[rid]) for rid in sorted(self.plans))


# ---------------------------------------------------------------------------
# Wire form: a plan is a node list with one time per node, nothing else
# ---------------------------------------------------------------------------


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

    Lanes end when the next node's resource is entered, which carries any FIFO delay
    the sender planned; regions and corridors take their fixed traversal time; the
    closing station is booked for its depth plus the turnaround.
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
            break
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
