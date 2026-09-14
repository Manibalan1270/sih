"""Route planning in free time windows (FE-3, FR-3.x, and the fix for defects 5, 10, 12).

Context-aware routing after ter Mors (AAMAS 2009), on the resource model in
core/timewindows.py. A* over states ``(node, time)`` where the cost *is* the time:
arriving earlier is never worse when you are allowed to wait, and the search only ever
stands at points where waiting is allowed. Between two such points the route is an
**atomic group** -- a region, or a corridor and the regions at its ends, ending as the
robot enters a two-lane lane or a station -- and the whole group must fit the free
windows of every resource in it, shifted by one start time. That is what makes "never
wait inside a region or corridor" a property of every plan rather than a rule enforced
at run time.

What the plan means at execution is *order*, not time. A robot enters a resource when
the robots booked ahead of it there have left, however late that is. Time here decides
who is ahead and lets the search prefer a bypass when the choke is booked -- which is
where the benchmark's margin over stop-and-wait comes from.

Integer milliseconds throughout; deterministic tie-breaking; no floats, no randomness,
no traffic model (CON-6, CON-7, FR-2.9, FR-5.9). The AST test in
tests/test_architecture.py enforces the last three.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from core import config
from core.timewindows import (
    CORRIDOR,
    LANE,
    REGION,
    REST_HOLD_MS,
    STATION,
    Resource,
    ResourceModel,
    ResourceTable,
    RoutePlan,
    Step,
)

PLAN_HORIZON_MS = 600_000
"""How far ahead a plan may reach before the search gives up. Ten minutes: a robot that
cannot find a slot inside that has a fleet problem, not a routing one."""

MAX_FIT_ITERATIONS = 64
"""Bound on how many times a group is pushed later to clear a blocking booking. Each
push is monotone, so this bounds the work, not the correctness."""

MAX_FIFO_REPAIRS = 6
"""Bound on search reruns to repair a lane-order violation (see ``plan``)."""


@dataclass(frozen=True, slots=True)
class Start:
    """Where the search begins, as the robot knows itself.

    ``node`` is the node the search continues from: the station leaf a resting robot
    is about to leave, or the node a moving robot will next reach. ``at_ms`` is when it
    will be there -- at that node's region entry boundary if it has one. ``waitable``
    says whether it can hold there: true at a station or at the end of a two-lane lane,
    false inside a region or after a corridor.

    A robot replanning mid-edge names the edge it is on through ``prefix_node``, the
    node it last left, and ``prefix_ms``, when it entered that node's region. The plan
    then begins with that crossing and the lane it is on, so peers behind it on the
    lane keep FIFO order and nobody books the region it just used. With ``in_region``
    the robot is already inside ``node``'s region and ``at_ms`` is when it will be out
    the far side; the region is held as an opening step, never re-fitted. Everything a plan
    contains is indexed by node this way, which is what lets the wire carry it as a
    node list with times and nothing else.
    """

    node: int
    at_ms: int
    waitable: bool = True
    in_region: bool = False
    prefix_node: int = -1
    prefix_ms: int = 0
    lane_entered_ms: int = -1
    region_entered_ms: int = -1
    """Booked entry times of the lane under the wheels and, with ``in_region``, of
    the region ahead, carried over from the plan being replaced. -1 derives them
    from ``prefix_ms``. Precedence is the order of booked entry times, so a step
    the robot has already entered must keep the time it was booked at: re-timing
    it to "now" moved a robot that was crossing a junction behind a peer that was
    waiting for it, and the peer drove in.
    """


@dataclass(frozen=True, slots=True)
class _Rel:
    """A resource inside a group, at an offset from the group's start."""

    resource: Resource
    enter: int
    exit: int


@dataclass(frozen=True, slots=True)
class _Group:
    """An atomic stretch of route between two points where waiting is allowed."""

    nodes: tuple[int, ...]
    """Nodes passed, ending at the node the group arrives at."""
    rels: tuple[_Rel, ...]
    end_node: int
    ends_at_station: bool


@dataclass
class PlanStats:
    expansions: int = 0
    groups_tried: int = 0
    fits_pushed: int = 0
    searches: int = 0
    unplannable: int = 0
    fifo_repairs: int = 0


@dataclass
class TimeWindowPlanner:
    model: ResourceModel
    stats: PlanStats = field(default_factory=PlanStats)

    # -- group enumeration ---------------------------------------------------

    def _groups_from(self, node: int, came_from: int, need_region: bool) -> list[_Group]:
        """Every atomic group leaving ``node``.

        Depth-first through corridors and the regions between them until a two-lane
        lane is entered (its far end is a waiting point) or a station is entered. On a
        single-lane column this walks the whole column, branching at every cross-aisle,
        because once inside a corridor a robot cannot stop until it is back on a
        two-lane lane -- and the plan has to say so.
        """
        model = self.model
        out: list[_Group] = []

        def walk(here: int, prev: int, offset: int, rels: list[_Rel], nodes: list[int], region_needed: bool) -> None:
            if region_needed and model.has_region(here):
                cross = model.region_cross_ms
                rels = rels + [_Rel(model.region(here), offset, offset + cross)]
                offset += cross
            for neighbour, edge_id in model.graph.neighbours(here):
                if neighbour == prev or neighbour in nodes:
                    continue
                if model.is_station(neighbour):
                    depth = model.station_in_ms(neighbour)
                    step = _Rel(
                        model.station(neighbour),
                        offset,
                        offset + depth + config.STATION_HOLD_MS,
                    )
                    out.append(_Group(tuple(nodes + [neighbour]), tuple(rels + [step]), neighbour, True))
                    continue
                lane = model.lane(edge_id, neighbour)
                run = model.lane_ms(edge_id)
                step = _Rel(lane, offset, offset + run)
                if lane.kind == CORRIDOR:
                    walk(neighbour, here, offset + run, rels + [step], nodes + [neighbour], True)
                else:
                    out.append(_Group(tuple(nodes + [neighbour]), tuple(rels + [step]), neighbour, False))

        walk(node, came_from, 0, [], [node], need_region)
        return out

    # -- fitting a group into free windows -------------------------------------

    def _fit(
        self,
        group: _Group,
        table: ResourceTable,
        earliest: int,
        *,
        fixed: bool,
        robot_id: int,
        constraints: dict[Resource, int] | None = None,
    ) -> tuple[int, int] | None:
        """Earliest start ``s >= earliest`` at which every resource in the group is
        free at its offset, or None. Returns ``(s, end_ms)``; ``end_ms`` includes any
        FIFO delay on a closing lane. With ``fixed`` the start must be exactly
        ``earliest`` -- the robot cannot wait -- so the answer is that or nothing.
        ``constraints`` are lanes this plan may not enter before a given time (see
        ``plan``)."""
        self.stats.groups_tried += 1
        s = earliest
        margin = table.margin_ms
        model = self.model
        for _ in range(MAX_FIT_ITERATIONS):
            pushed_to = s
            end = s
            for index, rel in enumerate(group.rels):
                enter, exit_ = s + rel.enter, s + rel.exit
                res = rel.resource
                if res.kind == LANE:
                    floor = constraints.get(res, 0) if constraints else 0
                    if enter < floor:
                        pushed_to = max(pushed_to, floor - rel.enter)
                        break
                    exit_ = max(exit_, table.lane_exit_after(res, enter, exclude_robot=robot_id))
                    if index == len(group.rels) - 1 and model.has_region(group.end_node):
                        # The lane ends at a junction. I stand on the lane until the
                        # region takes me, so my real exit is when the region is
                        # next free -- and that, not the bare traversal, is what
                        # the robots behind me on the lane have to be able to follow.
                        region = model.region(group.end_node)
                        blocker = table.blocking(
                            region, exit_, exit_ + model.region_cross_ms, exclude_robot=robot_id
                        )
                        if blocker is not None:
                            exit_ = max(exit_, blocker.exit_ms + margin + 1)
                    ahead = table.cuts_in_front_of(res, enter, exit_, exclude_robot=robot_id)
                    if ahead is not None:
                        # Someone already booked to enter after me would then leave
                        # before me. Take the slot after them instead.
                        pushed_to = max(pushed_to, ahead.enter_ms + 1 - rel.enter)
                        break
                    end = max(s + rel.exit, table.lane_exit_after(res, enter, exclude_robot=robot_id))
                    continue
                blocker = table.blocking(res, enter, exit_, exclude_robot=robot_id)
                if blocker is not None:
                    pushed_to = max(pushed_to, blocker.exit_ms + margin - rel.enter + 1)
                    break
                end = exit_
            else:
                return s, end
            if fixed:
                return None
            if pushed_to <= s:
                pushed_to = s + 1
            s = pushed_to
            self.stats.fits_pushed += 1
            if s > earliest + PLAN_HORIZON_MS:
                return None
        return None

    # -- the search -----------------------------------------------------------

    def plan(
        self,
        table: ResourceTable,
        *,
        start: Start,
        goal: int,
        robot_id: int,
        priority: int,
        plan_seq: int,
        committed_ms: int,
    ) -> RoutePlan | None:
        """A* over ``(node, time)``. Returns the plan or None if no route fits the
        horizon.

        A found plan is checked lane by lane for the order it implies: nobody
        booked to enter a lane after this robot may leave it before this robot
        does. The search fits one group at a time and a later group can stretch
        an earlier lane past such a robot; when that happens the lane gets a
        floor -- enter it after that robot -- and the search runs again. Bounded,
        and rare: the look-ahead in ``_fit`` already sees the common case, a
        lane feeding a busy junction.
        """
        start = self._respect_lane_fifo(start, table, robot_id)
        constraints: dict[Resource, int] = {}
        for _ in range(MAX_FIFO_REPAIRS):
            plan = self._search(
                table, start=start, goal=goal, robot_id=robot_id, priority=priority,
                plan_seq=plan_seq, committed_ms=committed_ms, constraints=constraints,
            )
            if plan is None:
                return None
            offence = self._lane_order_offence(plan, table, robot_id)
            if offence is None:
                return plan
            lane, floor = offence
            if constraints.get(lane, 0) >= floor:
                break
            constraints[lane] = floor
            self.stats.fifo_repairs += 1
        self.stats.unplannable += 1
        return None

    def _lane_order_offence(
        self, plan: RoutePlan, table: ResourceTable, robot_id: int
    ) -> tuple[Resource, int] | None:
        """The first lane whose booked window overtakes someone, and when this
        robot would have to enter it instead."""
        for step in plan.steps:
            if step.resource.kind != LANE:
                continue
            ahead = table.cuts_in_front_of(
                step.resource, step.enter_ms, step.exit_ms, exclude_robot=robot_id
            )
            if ahead is not None:
                return step.resource, ahead.enter_ms + 1
        return None

    def _search(
        self,
        table: ResourceTable,
        *,
        start: Start,
        goal: int,
        robot_id: int,
        priority: int,
        plan_seq: int,
        committed_ms: int,
        constraints: dict[Resource, int],
    ) -> RoutePlan | None:
        self.stats.searches += 1
        model = self.model
        graph = model.graph
        if start.node == goal:
            nodes, steps = self._prefix(start)
            if not nodes and model.is_station(goal):
                # Already there: hold the station, as the wire form of a one-node
                # plan says (a resting plan, refreshed while the robot stays).
                steps.append(
                    Step(model.station(goal), start.at_ms, max(start.at_ms, committed_ms) + REST_HOLD_MS)
                )
            return RoutePlan(
                robot_id, plan_seq, priority, committed_ms, tuple(nodes + [goal]), tuple(steps)
            )

        # (f, g, counter, node, waitable, came_from)
        counter = 0
        best_g: dict[int, int] = {start.node: start.at_ms}
        parent: dict[int, tuple[int, _Group, int, int] | None] = {start.node: None}
        # node -> (parent_node, group, group_start, group_end)
        heap: list[tuple[int, int, int, int, bool, int, bool]] = [
            (
                start.at_ms + graph.straight_line_ms(start.node, goal),
                start.at_ms,
                counter,
                start.node,
                start.waitable,
                -1,
                start.in_region,
            )
        ]
        closed: set[int] = set()

        while heap:
            _, g, _, node, waitable, came_from, in_region = heapq.heappop(heap)
            if node in closed:
                continue
            closed.add(node)
            self.stats.expansions += 1
            if node == goal:
                return self._assemble(
                    start, goal, parent, robot_id, priority, plan_seq, committed_ms
                )
            need_region = not in_region  # inside already means the crossing is underway
            for group in self._groups_from(node, came_from, need_region):
                fit = self._fit(
                    group, table, g, fixed=not waitable, robot_id=robot_id, constraints=constraints
                )
                if fit is None:
                    continue
                s, end = fit
                nxt = group.end_node
                if nxt in closed:
                    continue
                if nxt in best_g and best_g[nxt] <= end:
                    continue
                best_g[nxt] = end
                parent[nxt] = (node, group, s, end)
                counter += 1
                h = 0 if group.ends_at_station else graph.straight_line_ms(nxt, goal)
                heapq.heappush(heap, (end + h, end, counter, nxt, True, node, False))
        self.stats.unplannable += 1
        return None

    def _respect_lane_fifo(self, start: Start, table: ResourceTable, robot_id: int) -> Start:
        """A mid-edge start on a two-lane lane leaves it no earlier than everyone
        already ahead on it, plus the following gap.

        The opening lane step is a prefix, not a fitted group, so without this a
        robot replanning behind a queued peer booked the region at the lane's end
        *before* that peer -- and then waited for it by precedence while the peer
        waited for it by headway. Measured as a two-robot deadlock on one lane.
        """
        if start.prefix_node < 0 or start.in_region:
            return start
        model = self.model
        edge_id = model.graph.edge_between(start.prefix_node, start.node)
        if edge_id is None:
            return start
        lane = model.lane(edge_id, start.node)
        if lane.kind != LANE:
            return start
        entered = start.prefix_ms + (model.region_cross_ms if model.has_region(start.prefix_node) else 0)
        if start.lane_entered_ms >= 0:
            entered = min(entered, start.lane_entered_ms)
        earliest_exit = table.lane_exit_after(lane, entered, exclude_robot=robot_id)
        if earliest_exit <= start.at_ms:
            return start
        return Start(
            start.node, earliest_exit, waitable=start.waitable, in_region=start.in_region,
            prefix_node=start.prefix_node, prefix_ms=start.prefix_ms,
            lane_entered_ms=start.lane_entered_ms, region_entered_ms=start.region_entered_ms,
        )

    def _prefix(self, start: Start) -> tuple[list[int], list[Step]]:
        """The edge a replanning robot is already on, as the plan's opening steps."""
        if start.prefix_node < 0:
            return [], []
        model = self.model
        edge_id = model.graph.edge_between(start.prefix_node, start.node)
        if edge_id is None:
            return [], []
        steps: list[Step] = []
        cross = model.region_cross_ms
        lane_from = start.prefix_ms
        if model.has_region(start.prefix_node):
            steps.append(Step(model.region(start.prefix_node), start.prefix_ms, start.prefix_ms + cross))
            lane_from += cross
        elif model.is_station(start.prefix_node):
            # Leaving a station mid-spur: the departure step the wire form carries
            # for every plan that starts at a station. It must be here too, or the
            # sender's step indices and every receiver's disagree -- and a peer then
            # waits for this robot to release a step it never had.
            steps.append(
                Step(
                    model.station(start.prefix_node),
                    start.prefix_ms,
                    start.prefix_ms + model.station_in_ms(start.prefix_node),
                )
            )
        if start.lane_entered_ms >= 0:
            lane_from = min(lane_from, start.lane_entered_ms)
        if start.in_region and model.has_region(start.node):
            # Already inside the region ahead: it is held from when the robot
            # crossed its boundary until it leaves the far side, and the search
            # continues from there. Re-fitting it could book it *after* a peer that
            # is waiting for this very robot to clear it -- measured as a collision.
            entered = start.at_ms - cross
            if start.region_entered_ms >= 0:
                entered = min(entered, start.region_entered_ms)
            steps.append(Step(model.lane(edge_id, start.node), lane_from, max(lane_from, entered)))
            steps.append(Step(model.region(start.node), entered, start.at_ms))
        elif model.is_station(start.node):
            # Driving into a station: the spur is the station's, held from here
            # until the robot has stood its turnaround there.
            steps.append(
                Step(
                    model.station(start.node),
                    lane_from,
                    lane_from + model.station_in_ms(start.node) + config.STATION_HOLD_MS,
                )
            )
        else:
            steps.append(Step(model.lane(edge_id, start.node), lane_from, max(lane_from, start.at_ms)))
        return [start.prefix_node], steps

    def _assemble(
        self,
        start: Start,
        goal: int,
        parent: dict,
        robot_id: int,
        priority: int,
        plan_seq: int,
        committed_ms: int,
    ) -> RoutePlan:
        chain: list[tuple[_Group, int, int]] = []
        node = goal
        while parent[node] is not None:
            prev, group, s, end = parent[node]
            chain.append((group, s, end))
            node = prev
        chain.reverse()

        nodes, steps = self._prefix(start)
        nodes.append(start.node)
        if steps and steps[-1].resource.kind == LANE and chain:
            # The lane under the wheels is held until the first group starts,
            # which includes any wait at its end for that group's window.
            last = steps[-1]
            steps[-1] = Step(last.resource, last.enter_ms, max(last.exit_ms, chain[0][1]))
        if start.prefix_node < 0 and self.model.is_station(start.node) and chain:
            # Leaving a station: hold it from the start until the robot has driven
            # out to the anchor's region boundary, however long it waits for its
            # first window. Peers then never plan into an occupied station.
            depart = chain[0][1]
            steps.append(
                Step(
                    self.model.station(start.node),
                    min(committed_ms, start.at_ms),
                    depart + self.model.station_in_ms(start.node),
                )
            )
        for position, (group, s, end) in enumerate(chain):
            nodes.extend(group.nodes[1:])
            # A lane is held until the robot enters what comes next. The closing lane
            # of a group ends when the *following* group starts, which includes any
            # wait at the lane's end for that group's first window -- the robot is
            # standing on the lane throughout. Booking only the traversal under-booked
            # the lane and let a follower plan to leave it through a robot sitting at
            # its end; the round-trip through the wire form caught the discrepancy.
            follows_at = chain[position + 1][1] if position + 1 < len(chain) else end
            for index, rel in enumerate(group.rels):
                exit_ms = s + rel.exit
                if index == len(group.rels) - 1 and rel.resource.kind == LANE:
                    exit_ms = max(end, follows_at)
                steps.append(Step(rel.resource, s + rel.enter, exit_ms))
        return RoutePlan(robot_id, plan_seq, priority, committed_ms, tuple(nodes), tuple(steps))
