"""Route planning in free time windows.

Each test states one property a plan must have for execution by precedence to be
collision-free and deadlock-free, and checks it on the smallest graph that can exhibit
the conflict. The properties are what core/timewindows.py's docstring promises: no plan
ever waits inside a region or a corridor; two plans never hold a capacity-one resource
within the margin of each other; a lane keeps its FIFO order; and the ring that used to
deadlock plans conflict-free.
"""

from __future__ import annotations

import time

import pytest

from core import config
from core.graph import Graph
from core.planner_timewindow import Start, TimeWindowPlanner
from core.timewindows import (
    CORRIDOR,
    LANE,
    REGION,
    STATION,
    ResourceModel,
    ResourceTable,
    RoutePlan,
)


def build(nodes: dict[int, tuple[int, int]], edges: list[tuple[int, int, bool]]) -> Graph:
    """A toy map. ``edges`` are ``(u, v, single_lane)``. Leaves are stations."""
    return Graph.from_dict(
        {
            "name": "toy",
            "nodes": [{"id": n, "x": x, "y": y} for n, (x, y) in nodes.items()],
            "edges": [
                {"id": i, "u": u, "v": v, **({"single_lane": True} if sl else {})}
                for i, (u, v, sl) in enumerate(edges)
            ],
        }
    )


def from_station(model: ResourceModel, leaf: int, at_ms: int = 0) -> Start:
    """A robot resting at a station, about to leave."""
    anchor, _ = model.anchor_of(leaf)
    return Start(anchor, at_ms + model.station_in_ms(leaf), waitable=True)


def steps_of(plan: RoutePlan, kind: int):
    return [s for s in plan.steps if s.resource.kind == kind]


def assert_no_waiting_inside(plan: RoutePlan, model: ResourceModel) -> None:
    """Regions and corridors are held for exactly their traversal time."""
    for step in plan.steps:
        if step.resource.kind == REGION:
            assert step.exit_ms - step.enter_ms == model.region_cross_ms, str(step)
        elif step.resource.kind == CORRIDOR:
            assert step.exit_ms - step.enter_ms == model.lane_ms(step.resource.key), str(step)


def assert_disjoint_on_capacity_one(a: RoutePlan, b: RoutePlan, margin: int) -> None:
    for sa in a.steps:
        if not sa.resource.capacity_one:
            continue
        for sb in b.steps:
            if sb.resource != sa.resource:
                continue
            apart = sa.enter_ms - margin > sb.exit_ms or sb.enter_ms - margin > sa.exit_ms
            assert apart, f"{a.robot_id}:{sa} overlaps {b.robot_id}:{sb} within {margin} ms"


class TestHeadOnInACorridor:
    """Two junctions joined by one single-lane corridor, a station at each end."""

    def setup_method(self) -> None:
        #  a1 -- A ===corridor=== B -- b1      (A and B need a third arm to be junctions)
        #        |                 |
        #        a2                b2
        self.graph = build(
            {0: (0, 0), 1: (10000, 0), 10: (-3000, 0), 11: (0, 3000), 20: (13000, 0), 21: (10000, 3000)},
            [(0, 1, True), (10, 0, False), (11, 0, False), (20, 1, False), (21, 1, False)],
        )
        self.model = ResourceModel(self.graph)
        self.planner = TimeWindowPlanner(self.model)
        self.table = ResourceTable()

    def test_the_second_robot_waits_at_its_mouth_never_inside(self) -> None:
        first = self.planner.plan(
            self.table, start=from_station(self.model, 10), goal=20,
            robot_id=1, priority=10, plan_seq=1, committed_ms=0,
        )
        assert first is not None
        self.table.replace_plan(first)
        second = self.planner.plan(
            self.table, start=from_station(self.model, 20), goal=10,
            robot_id=2, priority=10, plan_seq=1, committed_ms=1,
        )
        assert second is not None

        (c1,) = steps_of(first, CORRIDOR)
        (c2,) = steps_of(second, CORRIDOR)
        assert c2.enter_ms >= c1.exit_ms + config.MARGIN_MS, "head-on in the corridor"
        assert_no_waiting_inside(second, self.model)
        assert_disjoint_on_capacity_one(first, second, config.MARGIN_MS)

        # The wait shows up before B's region, on the spur, not in the corridor:
        # the second robot's first region crossing starts later than it could have.
        start = from_station(self.model, 20)
        (rb,) = [s for s in steps_of(second, REGION) if s.resource.key == 1]
        assert rb.enter_ms > start.at_ms, "the second robot did not wait at the mouth"

    def test_the_corridor_and_its_exit_region_are_booked_as_one(self) -> None:
        """A robot may not enter the corridor unless it can also leave it: the exit
        region must be free on arrival, because it cannot wait inside."""
        # Park a third robot's plan across B's region right when robot 1 would exit.
        first_free = self.planner.plan(
            self.table, start=from_station(self.model, 10), goal=20,
            robot_id=1, priority=10, plan_seq=1, committed_ms=0,
        )
        (rb,) = [s for s in steps_of(first_free, REGION) if s.resource.key == 1]
        blocker = self.planner.plan(
            self.table, start=Start(1, rb.enter_ms - 500, waitable=True), goal=21,
            robot_id=3, priority=10, plan_seq=1, committed_ms=0,
        )
        assert blocker is not None
        self.table.replace_plan(blocker)

        first = self.planner.plan(
            self.table, start=from_station(self.model, 10), goal=20,
            robot_id=1, priority=10, plan_seq=1, committed_ms=1,
        )
        assert first is not None
        assert_no_waiting_inside(first, self.model)
        assert_disjoint_on_capacity_one(first, blocker, config.MARGIN_MS)
        # It entered the corridor later so as to reach B after the blocker had left.
        (c1,) = steps_of(first, CORRIDOR)
        (c0,) = steps_of(first_free, CORRIDOR)
        assert c1.enter_ms > c0.enter_ms


class TestCrossingAtAJunction:
    """A four-way region: the second robot's crossing starts after the first's ends."""

    def setup_method(self) -> None:
        self.graph = build(
            {0: (0, 0), 1: (0, 5000), 2: (0, -5000), 3: (5000, 0), 4: (-5000, 0)},
            [(1, 0, False), (2, 0, False), (3, 0, False), (4, 0, False)],
        )
        self.model = ResourceModel(self.graph)
        self.planner = TimeWindowPlanner(self.model)
        self.table = ResourceTable()

    def test_regions_are_taken_in_turn_with_the_margin(self) -> None:
        west_east = self.planner.plan(
            self.table, start=from_station(self.model, 4), goal=3,
            robot_id=1, priority=10, plan_seq=1, committed_ms=0,
        )
        self.table.replace_plan(west_east)
        north_south = self.planner.plan(
            self.table, start=from_station(self.model, 1), goal=2,
            robot_id=2, priority=10, plan_seq=1, committed_ms=1,
        )
        assert north_south is not None
        (r1,) = steps_of(west_east, REGION)
        (r2,) = steps_of(north_south, REGION)
        assert r2.enter_ms >= r1.exit_ms + config.MARGIN_MS
        assert_no_waiting_inside(north_south, self.model)

    def test_a_third_robot_takes_the_next_free_window(self) -> None:
        plans = []
        for rid, (a, b) in enumerate(((4, 3), (1, 2), (3, 4)), start=1):
            plan = self.planner.plan(
                self.table, start=from_station(self.model, a), goal=b,
                robot_id=rid, priority=10, plan_seq=1, committed_ms=rid,
            )
            assert plan is not None
            self.table.replace_plan(plan)
            plans.append(plan)
        for i, a in enumerate(plans):
            for b in plans[i + 1:]:
                assert_disjoint_on_capacity_one(a, b, config.MARGIN_MS)


class TestFollowingOnATwoLaneEdge:
    """Two robots down one long aisle: the one that enters first leaves first."""

    def setup_method(self) -> None:
        #  a1, a2 -- A ==== long two-lane ==== B -- b1, b2
        self.graph = build(
            {0: (0, 0), 1: (20000, 0), 10: (-3000, 0), 11: (0, 3000), 20: (23000, 0), 21: (20000, 3000)},
            [(0, 1, False), (10, 0, False), (11, 0, False), (20, 1, False), (21, 1, False)],
        )
        self.model = ResourceModel(self.graph)
        self.planner = TimeWindowPlanner(self.model)
        self.table = ResourceTable()

    def test_fifo_is_kept_when_the_leader_is_slow(self) -> None:
        leader = self.planner.plan(
            self.table, start=from_station(self.model, 10), goal=20,
            robot_id=1, priority=10, plan_seq=1, committed_ms=0,
        )
        # Make the leader dawdle on the lane: stretch its lane step.
        (lane,) = steps_of(leader, LANE)
        slow = RoutePlan(
            leader.robot_id, leader.plan_seq, leader.priority, leader.committed_ms, leader.nodes,
            tuple(
                s if s is not lane else type(s)(s.resource, s.enter_ms, s.exit_ms + 10_000)
                for s in leader.steps
            ),
        )
        self.table.replace_plan(slow)

        follower = self.planner.plan(
            self.table, start=from_station(self.model, 11, at_ms=500), goal=21,
            robot_id=2, priority=10, plan_seq=1, committed_ms=1,
        )
        assert follower is not None
        (f_lane,) = steps_of(follower, LANE)
        assert f_lane.enter_ms > lane.enter_ms, "the follower should enter second"
        assert f_lane.exit_ms >= lane.exit_ms + 10_000 + config.FOLLOW_GAP_MS, (
            "the follower planned to overtake a slow leader on a lane"
        )

    def test_a_faster_arrival_does_not_cut_in_front(self) -> None:
        """The second planner cannot rewrite the first's plan, so if entering ahead of
        it would mean leaving ahead of it, it does not enter ahead of it."""
        booked = self.planner.plan(
            self.table, start=from_station(self.model, 10, at_ms=3000), goal=20,
            robot_id=1, priority=10, plan_seq=1, committed_ms=0,
        )
        (lane,) = steps_of(booked, LANE)
        # Book it as if it will crawl the whole lane, exiting very late.
        crawl = RoutePlan(
            booked.robot_id, 1, 10, 0, booked.nodes,
            tuple(
                s if s is not lane else type(s)(s.resource, s.enter_ms, s.enter_ms + 60_000)
                for s in booked.steps
            ),
        )
        self.table.replace_plan(crawl)
        later = self.planner.plan(
            self.table, start=from_station(self.model, 11, at_ms=0), goal=21,
            robot_id=2, priority=10, plan_seq=1, committed_ms=1,
        )
        assert later is not None
        (l_lane,) = steps_of(later, LANE)
        (crawl_lane,) = steps_of(crawl, LANE)
        # Either in first and out first, or in second and out second -- never in
        # first and out second, which would be overtaking on paper.
        if l_lane.enter_ms < crawl_lane.enter_ms:
            assert l_lane.exit_ms + config.FOLLOW_GAP_MS <= crawl_lane.exit_ms, (
                "entered ahead of the crawler but planned to leave behind it"
            )
        else:
            assert l_lane.exit_ms >= crawl_lane.exit_ms + config.FOLLOW_GAP_MS, (
                "entered behind the crawler but planned to leave ahead of it"
            )


class TestTheRing:
    """TC-3 / Appendix C: three robots each wanting the corridor the next one holds.

    Junction-by-junction arbitration cannot escape this (defect 5); each robot is the
    rightful winner of the segment it wants and blocked by another holding the next.
    Booking whole routes, the third plan simply takes the slot after the other two.
    """

    def test_three_plans_round_the_loop_are_pairwise_disjoint(self) -> None:
        graph = Graph.load("maps/loop_map.json")
        model = ResourceModel(graph)
        planner = TimeWindowPlanner(model)
        table = ResourceTable()
        # S0 -> S1 -> S2 -> S0 are degree-two stations on spurs; bays are the leaves.
        plans = []
        for rid, (a, b) in enumerate(((6, 7), (7, 8), (8, 6)), start=1):
            plan = planner.plan(
                table, start=from_station(model, a), goal=b,
                robot_id=rid, priority=10, plan_seq=1, committed_ms=rid,
            )
            assert plan is not None, f"robot {rid} could not plan {a}->{b}"
            assert_no_waiting_inside(plan, model)
            table.replace_plan(plan)
            plans.append(plan)
        for i, a in enumerate(plans):
            for b in plans[i + 1:]:
                assert_disjoint_on_capacity_one(a, b, config.MARGIN_MS)


class TestPlanShape:
    def test_steps_are_contiguous_except_where_waiting_is_allowed(self) -> None:
        """A gap between consecutive steps is a wait, and a wait is legal only at the
        end of a two-lane lane or in a station."""
        graph = Graph.load("maps/warehouse_zoned_30.json")
        model = ResourceModel(graph)
        planner = TimeWindowPlanner(model)
        table = ResourceTable()
        plans = []
        for rid in range(1, 8):
            pick = graph.pickup_nodes[rid % len(graph.pickup_nodes)]
            drop = graph.drop_nodes[(rid * 3) % len(graph.drop_nodes)]
            plan = planner.plan(
                table, start=from_station(model, pick), goal=drop,
                robot_id=rid, priority=10, plan_seq=1, committed_ms=rid,
            )
            assert plan is not None
            table.replace_plan(plan)
            plans.append(plan)
        for plan in plans:
            assert_no_waiting_inside(plan, model)
            for before, after in zip(plan.steps, plan.steps[1:]):
                gap = after.enter_ms - before.exit_ms
                assert gap >= 0, f"{before} and {after} overlap in one plan"
                if gap > 0:
                    assert before.resource.kind in (LANE, STATION), (
                        f"waited {gap} ms after {before}, which is not a waiting point"
                    )
        for i, a in enumerate(plans):
            for b in plans[i + 1:]:
                assert_disjoint_on_capacity_one(a, b, config.MARGIN_MS)

    def test_the_route_is_the_node_sequence_the_steps_traverse(self) -> None:
        graph = Graph.load("maps/benchmark_map.json")
        model = ResourceModel(graph)
        planner = TimeWindowPlanner(model)
        plan = planner.plan(
            ResourceTable(), start=from_station(model, 15), goal=18,
            robot_id=1, priority=10, plan_seq=1, committed_ms=0,
        )
        assert plan is not None
        assert plan.nodes[-1] == 18
        for u, v in zip(plan.nodes, plan.nodes[1:]):
            assert graph.edge_between(u, v) is not None, f"{u}->{v} is not an edge"


class TestBudget:
    """FR-3.6: planning inside 50 ms, on the largest map, against a loaded table."""

    def test_planning_on_the_hundred_robot_map_stays_inside_the_deadline(self) -> None:
        graph = Graph.load("maps/warehouse_zoned_100.json")
        model = ResourceModel(graph)
        planner = TimeWindowPlanner(model)
        table = ResourceTable()
        for rid in range(1, 25):
            pick = graph.pickup_nodes[rid % len(graph.pickup_nodes)]
            drop = graph.drop_nodes[(rid * 5) % len(graph.drop_nodes)]
            plan = planner.plan(
                table, start=from_station(model, pick, at_ms=rid * 700), goal=drop,
                robot_id=rid, priority=10, plan_seq=1, committed_ms=rid,
            )
            assert plan is not None
            table.replace_plan(plan)

        timings = []
        for rid in range(25, 31):
            pick = graph.pickup_nodes[rid % len(graph.pickup_nodes)]
            drop = graph.drop_nodes[(rid * 7) % len(graph.drop_nodes)]
            t0 = time.perf_counter()
            plan = planner.plan(
                table, start=from_station(model, pick, at_ms=rid * 700), goal=drop,
                robot_id=rid, priority=10, plan_seq=1, committed_ms=rid,
            )
            timings.append((time.perf_counter() - t0) * 1000)
            assert plan is not None
            table.replace_plan(plan)
        typical = sorted(timings)[len(timings) // 2]
        assert typical < config.PLAN_DEADLINE_MS, (
            f"median plan took {typical:.1f} ms against a 24-plan table; "
            f"FR-3.6 allows {config.PLAN_DEADLINE_MS}"
        )
