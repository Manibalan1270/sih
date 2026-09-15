"""SRS section 6.2 cases for the safety core: TC-1, TC-2, TC-3.

These are the Phase 6 and Phase 7 gates. TC-3 in particular is the executable
counterpart of the Appendix C proof: the proof shows a conflict set under a total
order has a unique maximum, so exactly one robot proceeds and a cyclic wait cannot
form. Here that is checked three ways -- the key is a total order, every small
conflict set has exactly one winner, and three robots on a single-lane ring
actually clear.

Coordination is route-level: a robot books its whole route as time windows against
the routes it has heard (core/timewindows.py, core/planner_timewindow.py) and enters
each junction region only after everyone booked ahead of it there has left. So TC-1's
"overlap detected before approach" is the planner refusing to book a window inside
another robot's, and "the lower-ranked AMR sheds speed" is the race rule in
core/arbitration.py deciding whose plan stands when two were committed before
either heard the other.
"""

from __future__ import annotations

import itertools

import pytest

from core import config, scenarios
from core.arbitration import (
    Arbiter,
    outranks,
    plan_precedence,
    ranking_key,
    unique_winner,
)
from core.graph import Graph
from core.planner_timewindow import Start, TimeWindowPlanner
from core.state_machine import State
from core.timewindows import (
    REGION,
    ResourceModel,
    ResourceTable,
    RoutePlan,
    Step,
    conflict_between,
)
from benchmark.runner import compare_configurations, run_seed
from simulator.scenario import AuctionAllocator, build
from simulator.scenario import RoundRobinAllocator


def coordinated(seed: int = 0, *, tasks: int | None = None):
    """A fleet with the full coordination stack: auction, mesh, arbitration."""
    return build(
        scenarios.get("bench3"),
        seed=seed,
        allocator=AuctionAllocator(),
        task_count=tasks,
    )


def crossing() -> tuple[Graph, ResourceModel, TimeWindowPlanner]:
    """A four-way junction with a station on every arm: the TC-1 geometry. The
    west arm has a bend (node 5) halfway, so a second robot can be placed on the
    approach behind the one leaving the station."""
    graph = Graph.from_dict(
        {
            "name": "cross",
            "nodes": [
                {"id": 0, "x": 0, "y": 0},
                {"id": 1, "x": -16000, "y": 0},
                {"id": 2, "x": 16000, "y": 0},
                {"id": 3, "x": 0, "y": -16000},
                {"id": 4, "x": 0, "y": 16000},
                {"id": 5, "x": -8000, "y": 0},
            ],
            "edges": [
                {"id": 0, "u": 1, "v": 5},
                {"id": 1, "u": 0, "v": 2},
                {"id": 2, "u": 3, "v": 0},
                {"id": 3, "u": 0, "v": 4},
                {"id": 4, "u": 5, "v": 0},
            ],
        }
    )
    model = ResourceModel(graph)
    return graph, model, TimeWindowPlanner(model)


def plan_across(
    planner: TimeWindowPlanner, table: ResourceTable, *, robot_id: int, start: int,
    goal: int, at_ms: int, priority: int = 10, committed_ms: int | None = None,
) -> RoutePlan:
    plan = planner.plan(
        table, start=Start(start, at_ms, waitable=True), goal=goal, robot_id=robot_id,
        priority=priority, plan_seq=1, committed_ms=at_ms if committed_ms is None else committed_ms,
    )
    assert plan is not None
    return plan


def region_window(plan: RoutePlan, node: int) -> Step:
    return next(s for s in plan.steps if s.resource.kind == REGION and s.resource.key == node)


@pytest.mark.tc
class TestTC1CrossingJunction:
    """Two AMRs plan routes crossing one junction with overlapping ETA windows.

    Expected: overlap detected before approach; the lower-ranked AMR sheds speed;
    no collision (FR-5.1 to FR-5.7).
    """

    def test_an_overlapping_window_is_detected_before_approach(self) -> None:
        """The second robot to plan books the junction after the first, plus the
        margin -- at planning time, before either has left its station."""
        _, _, planner = crossing()
        table = ResourceTable()
        first = plan_across(planner, table, robot_id=2, start=1, goal=2, at_ms=0)
        table.replace_plan(first)
        second = plan_across(planner, table, robot_id=3, start=3, goal=4, at_ms=200)
        a, b = region_window(first, 0), region_window(second, 0)
        assert b.enter_ms >= a.exit_ms + config.MARGIN_MS
        assert conflict_between(first, second, config.MARGIN_MS) is None

    def test_windows_beyond_the_margin_do_not_conflict(self) -> None:
        """FR-5.3 separates by the margin, not merely by intersection."""
        _, _, planner = crossing()
        table = ResourceTable()
        first = plan_across(planner, table, robot_id=2, start=1, goal=2, at_ms=0)
        table.replace_plan(first)
        clear_at = region_window(first, 0).exit_ms + config.MARGIN_MS + 100
        # Leaving late enough that the crossing is free: no wait is planned.
        second = plan_across(planner, table, robot_id=3, start=3, goal=4, at_ms=clear_at)
        lane_in = second.steps[1]
        assert lane_in.exit_ms == region_window(second, 0).enter_ms
        assert lane_in.enter_ms == clear_at

    def test_a_race_gives_both_robots_opposite_answers(self) -> None:
        """Appendix C's Theorem in miniature: two plans committed in the same
        millisecond, overlapping on the junction -- exactly one stands."""
        _, _, planner = crossing()
        mine = plan_across(planner, ResourceTable(), robot_id=2, start=1, goal=2, at_ms=0)
        theirs = plan_across(planner, ResourceTable(), robot_id=5, start=3, goal=4, at_ms=0)
        clash = conflict_between(mine, theirs, config.MARGIN_MS)
        assert clash is not None and clash.kind == REGION
        assert plan_precedence(mine, theirs) is True
        assert plan_precedence(theirs, mine) is False

    def test_the_earlier_commit_stands(self) -> None:
        """Committed first, stands first, whatever the rank -- the later committer
        planned against a table that should have held the earlier plan."""
        _, _, planner = crossing()
        early = plan_across(planner, ResourceTable(), robot_id=9, start=1, goal=2, at_ms=0, priority=10, committed_ms=0)
        late = plan_across(planner, ResourceTable(), robot_id=1, start=3, goal=4, at_ms=0, priority=200, committed_ms=5)
        assert plan_precedence(early, late)
        assert not plan_precedence(late, early)

    def test_priority_beats_identifier_on_a_tie(self) -> None:
        """BR-1: on the same fleet-clock millisecond, higher task priority stands."""
        _, _, planner = crossing()
        low = plan_across(planner, ResourceTable(), robot_id=1, start=1, goal=2, at_ms=0, priority=10)
        high = plan_across(planner, ResourceTable(), robot_id=9, start=3, goal=4, at_ms=0, priority=200)
        assert plan_precedence(high, low)
        assert not plan_precedence(low, high)

    def test_the_loser_of_a_race_replans_after_the_winner(self) -> None:
        """The race resolved, the loser's new plan books the junction after the
        winner's window: the yield is a later window, not a slower crossing."""
        _, _, planner = crossing()
        winner = plan_across(planner, ResourceTable(), robot_id=2, start=1, goal=2, at_ms=0)
        table = ResourceTable()
        table.replace_plan(winner)
        loser = plan_across(planner, table, robot_id=5, start=3, goal=4, at_ms=0, committed_ms=1)
        assert region_window(loser, 0).enter_ms >= region_window(winner, 0).exit_ms + config.MARGIN_MS

    def test_the_arbiter_records_the_decision(self) -> None:
        _, _, planner = crossing()
        mine = plan_across(planner, ResourceTable(), robot_id=2, start=1, goal=2, at_ms=0)
        theirs = plan_across(planner, ResourceTable(), robot_id=5, start=3, goal=4, at_ms=0)
        arbiter = Arbiter(robot_id=2)
        clash = conflict_between(mine, theirs, config.MARGIN_MS)
        assert arbiter.resolve(mine, theirs, clash, now_ms=100) is True
        assert arbiter.stood == 1 and arbiter.replanned == 0
        assert arbiter.last is not None and arbiter.last.other_robot == 5

    def test_following_traffic_is_not_a_conflict(self) -> None:
        """Two robots entering from the same side queue; they do not contend for
        the lane, only take turns in the junction, and the one behind never
        leaves the lane before the one in front."""
        _, _, planner = crossing()
        table = ResourceTable()
        leader = plan_across(planner, table, robot_id=1, start=1, goal=2, at_ms=0)
        table.replace_plan(leader)
        lane_leader = next(s for s in leader.steps if s.resource.key == 4)  # the last lane in
        # Already on the approach lane past the bend, a following gap behind the leader.
        behind = Start(
            0, lane_leader.exit_ms, waitable=True,
            prefix_node=5, prefix_ms=lane_leader.enter_ms + config.FOLLOW_GAP_MS,
        )
        follower = planner.plan(
            table, start=behind, goal=2, robot_id=2, priority=10, plan_seq=1,
            committed_ms=config.FOLLOW_GAP_MS,
        )
        assert follower is not None
        assert conflict_between(leader, follower, config.MARGIN_MS) is None
        lane_follower = follower.steps[0]
        assert lane_follower.resource == lane_leader.resource
        assert lane_follower.enter_ms > lane_leader.enter_ms
        assert lane_follower.exit_ms >= lane_leader.exit_ms + config.FOLLOW_GAP_MS
        assert region_window(follower, 0).enter_ms >= region_window(leader, 0).exit_ms + config.MARGIN_MS

    @pytest.mark.slow
    def test_repeated_crossings_produce_no_collisions(self) -> None:
        """The Phase 6 gate, end to end (NFR-2.1)."""
        for seed in range(3):
            sim = coordinated(seed)
            assert sim.run(max_ms=1_800_000), f"seed {seed} did not finish"
            assert sim.engine.coordination_failures == [], (
                f"seed {seed}: {len(sim.engine.coordination_failures)} collisions"
            )

    @pytest.mark.slow
    def test_yielding_actually_happens(self) -> None:
        """Zero collisions reached by nobody ever contending would prove nothing."""
        sim = coordinated(0)
        sim.run(max_ms=1_800_000)
        assert sum(r.metrics.yields_lost for r in sim.engine.robots) > 0

    @pytest.mark.slow
    def test_ac2_zero_collisions_across_thirty_bench3_seeds(self) -> None:
        """AC-2: Configuration B is safe across the required seed floor."""
        for seed in range(30):
            sim = coordinated(seed)
            assert sim.run(max_ms=1_800_000), f"seed {seed} did not finish"
            assert sim.engine.coordination_failures == [], (
                f"seed {seed}: {len(sim.engine.coordination_failures)} coordination failures"
            )
            assert sim.engine.stall_report is None or not sim.engine.stall_report.is_deadlocked, (
                f"seed {seed}: deadlock cycle reported"
            )

    @pytest.mark.slow
    @pytest.mark.xfail(
        strict=True,
        reason="Open defect 16: a station is a point, while every other resource is "
        "sized to contain a robot. Releasing a station step therefore does not mean "
        "having physically left it, and a robot that stops just down the spur is "
        "rear-ended by the next robot to take the station. Carried as a strict xfail "
        "rather than deleted so the gap stays visible and the day it is fixed this "
        "fails loudly. See README defect 16 for the measurement and the fix.",
    )
    def test_ac2_holds_at_thirty_amr_density(self) -> None:
        """AC-2 where it is actually hard: thirty AMRs, not three.

        The thirty-seed gate above is `bench3`, and three robots on that map rarely
        produce the conflicts a full floor does -- which is how a real collision sat
        undetected while AC-2 passed. It was found by running `visual30` directly:
        r2 left station J110, stopped 322 mm down the spur on headway, and r12 then
        entered J110 on a booking that had been legitimately released, overlapping it
        at 322 mm against a COLLISION_DISTANCE_MM of 500.

        One seed and a capped clock, because this has to stay runnable, but the full
        task count: at 40 tasks the floor never loads enough and the case passes
        without proving anything. Seed 0 collides at 346,800 ms, so the cap sits
        just past it.
        """
        from core import scenarios as _scenarios
        from simulator.scenario import build as _build

        sim = _build(
            _scenarios.get("visual30"), seed=0,
            allocator=AuctionAllocator(), task_count=120, waves=4,
        )
        sim.run(max_ms=400_000)
        assert sim.engine.coordination_failures == [], (
            f"visual30 seed 0: {len(sim.engine.coordination_failures)} coordination "
            f"failures at 30-AMR density -- {sim.engine.coordination_failures[:1]}"
        )
        assert sim.engine.collisions == [], (
            f"visual30 seed 0: {len(sim.engine.collisions)} collisions"
        )

    @pytest.mark.slow
    def test_ac3_b_reduces_mean_makespan_by_twenty_percent(self) -> None:
        """AC-3: Configuration B must beat A on the same ten task sets."""
        results_a = [
            run_seed(
                scenarios.get("bench3"),
                seed=seed,
                config_name="A",
                allocator=RoundRobinAllocator(),
            )
            for seed in range(10)
        ]
        results_b = [
            run_seed(
                scenarios.get("bench3"),
                seed=seed,
                config_name="B",
                allocator=AuctionAllocator(),
            )
            for seed in range(10)
        ]
        summary = compare_configurations(results_a, results_b)
        assert summary["collision_total_b"] == 0
        assert summary["coordination_failures_b"] == 0
        assert summary["makespan_mean_b"] <= 0.8 * summary["makespan_mean_a"], summary


class TestYieldSpeedDoesNotGovernRegionSafety:
    """Why it is safe to raise YIELD_SPEED_MM_S.

    A robot does slow to yield speed while numerically within JUNCTION_FOOTPRINT_MM
    of a node it just departed -- once its region step is entered, _next_gate moves
    on to gate the *next* step (a lane, say), and if that lane is still booked to
    someone else the robot sheds to yield speed on approach to it, before it has
    physically cleared the region's footprint. Measured directly: at seed 0, task
    count 6, robot 1 sits at YIELD_SPEED_MM_S while 4mm past node 4, 1196mm short of
    the footprint radius.

    That is not the collision exposure JUNCTION_FOOTPRINT_MM exists to bound,
    though. The exposure is two robots occupying the same node's footprint from
    perpendicular edges at once, and a second robot cannot get a region booking
    while a live peer still *reports* that node as current -- _may_enter's REGION
    branch (core/robot.py) blocks on any peer whose current_node equals the region,
    independent of the time-window bookkeeping and independent of speed. So the
    thing that actually needs checking is not "does speed ever dip near a
    footprint" (it does, structurally, regardless of YIELD_SPEED_MM_S's value) but
    "do collisions and coordination failures stay at zero regardless of the value"
    -- checked here at both the shipped value and a substantially higher one.
    """

    @pytest.mark.parametrize("yield_speed", [50, 600, 799])
    def test_zero_collisions_across_seeds_at_this_yield_speed(
        self, monkeypatch: pytest.MonkeyPatch, yield_speed: int
    ) -> None:
        monkeypatch.setattr(config, "YIELD_SPEED_MM_S", yield_speed)
        for seed in range(5):
            sim = coordinated(seed=seed)
            assert sim.run(max_ms=1_800_000), f"seed {seed} did not finish"
            assert sim.engine.collisions == [], (
                f"seed {seed} at yield {yield_speed}: {len(sim.engine.collisions)} collisions"
            )
            assert sim.engine.coordination_failures == [], (
                f"seed {seed} at yield {yield_speed}: "
                f"{len(sim.engine.coordination_failures)} coordination failures"
            )


@pytest.mark.tc
class TestTC2SingleLaneCorridor:
    """Two AMRs approach a single-lane corridor from opposite ends.

    Expected: the yield decision is taken at the last passing point before entry;
    no head-on standoff (FR-5.10, ASM-5).
    """

    @pytest.mark.slow
    def test_no_collision_inside_the_corridor(self) -> None:
        for seed in range(3):
            sim = coordinated(seed)
            sim.run(max_ms=1_800_000)
            choke = sim.graph.single_lane_edges[0]
            edge = sim.graph.edge(choke)
            span = sorted(
                (sim.graph.node(edge.u).x_mm, sim.graph.node(edge.v).x_mm)
            )
            for event in sim.engine.coordination_failures:
                assert not (span[0] <= event.x_mm <= span[1]), (
                    f"seed {seed}: collision inside the single-lane corridor"
                )

    @pytest.mark.slow
    def test_nobody_ever_yields_from_inside_the_corridor(self) -> None:
        """The decision must be complete before entry. Deciding once inside leaves no
        outcome available but a standoff, which is what FR-5.10 forbids."""
        sim = coordinated(0)
        choke = sim.graph.single_lane_edges[0]
        offences = 0
        for _ in range(90_000):
            sim.step()
            for robot in sim.engine.robots:
                if (
                    robot.edge_id == choke
                    and robot.state is State.YIELD
                    and robot.progress_mm > config.ENTRY_COMMIT_MM
                ):
                    offences += 1
            if sim.is_finished:
                break
        assert offences == 0, f"yielded {offences} times from inside the corridor"

    @pytest.mark.slow
    def test_the_corridor_is_used_in_both_directions(self) -> None:
        """Otherwise the head-on case was never exercised at all."""
        sim = coordinated(0)
        choke = sim.graph.single_lane_edges[0]
        directions: set[int] = set()
        for _ in range(90_000):
            sim.step()
            for robot in sim.engine.robots:
                if robot.edge_id == choke and robot.next_node is not None:
                    directions.add(robot.next_node)
            if sim.is_finished:
                break
        assert len(directions) == 2, f"corridor only traversed toward {directions}"

    @pytest.mark.slow
    def test_the_corridor_never_holds_opposing_traffic(self) -> None:
        """The invariant the corridor mechanism exists to hold.

        Not "one robot at a time": a single-lane aisle is one *lane*, so robots may
        queue through it nose to tail exactly as on a one-lane road, with headway
        (IF-2.4) keeping them apart. What it cannot hold is two robots facing each
        other, because neither can pass and neither can turn round -- that is the
        head-on FR-5.10 forbids.
        """
        sim = coordinated(0)
        choke = sim.graph.single_lane_edges[0]
        for _ in range(90_000):
            sim.step()
            inside = [
                r
                for r in sim.engine.robots
                if r.edge_id == choke and r.progress_mm > config.ENTRY_COMMIT_MM
            ]
            headings = {r.next_node for r in inside}
            assert len(headings) <= 1, (
                f"opposing traffic inside the corridor at {sim.engine.now_ms} ms: "
                f"{[(r.robot_id, r.current_node, r.next_node) for r in inside]}"
            )
            if sim.is_finished:
                break


@pytest.mark.tc
class TestTC3CyclicConflict:
    """Three AMRs form a cyclic conflict, each waiting on the next.

    Expected: the total order yields a unique maximum; the cycle cannot form; all
    three clear (FR-5.12, Appendix C).
    """

    def test_the_ranking_key_is_a_total_order(self) -> None:
        """Appendix C's Lemma 1: no ties, no incomparable pairs."""
        contenders = [(p, r) for p in (0, 10, 200, 255) for r in range(1, 12)]
        keys = [ranking_key(p, r) for p, r in contenders]
        assert len(set(keys)) == len(keys), "two distinct robots share a ranking key"

    def test_every_conflict_set_has_exactly_one_maximum(self) -> None:
        """Appendix C's Theorem, checked over every subset of a small population."""
        population = [(10, 1), (10, 2), (10, 3), (200, 4), (0, 5)]
        for size in range(1, len(population) + 1):
            for contenders in itertools.combinations(population, size):
                winner = unique_winner(list(contenders))
                assert winner is not None
                others = [c for c in contenders if c != winner]
                assert all(
                    ranking_key(*winner) > ranking_key(*other) for other in others
                ), f"{winner} is not the unique maximum of {contenders}"

    def test_no_cycle_can_form_in_a_three_way_conflict(self) -> None:
        """A cyclic wait needs every member waiting on a strictly greater element --
        a strictly increasing cycle in a total order, which cannot exist."""
        trio = [(10, 1), (10, 2), (10, 3)]
        waiting_on = {
            me: [o for o in trio if o != me and outranks(o[0], o[1], me[0], me[1])]
            for me in trio
        }
        proceeding = [me for me, rivals in waiting_on.items() if not rivals]
        assert len(proceeding) == 1, f"expected exactly one to proceed, got {proceeding}"

    def test_the_ordering_is_antisymmetric(self) -> None:
        """Both directions can never hold, or two robots would both proceed."""
        population = [(p, r) for p in (0, 10, 200) for r in (1, 2, 7)]
        for a, b in itertools.permutations(population, 2):
            assert not (
                outranks(a[0], a[1], b[0], b[1]) and outranks(b[0], b[1], a[0], a[1])
            )

    def test_an_empty_conflict_set_has_no_winner(self) -> None:
        assert unique_winner([]) is None

    @pytest.mark.slow
    def test_three_robots_on_a_single_lane_ring_all_clear(self) -> None:
        """The executable counterpart of Appendix C, on the map built for it.

        The loop map is three single-lane junction-to-junction edges in a ring with a
        spur at each junction, so robots sent round it must contend for every segment.
        If a cyclic wait were possible, this is where it would form.

        This was xfail for a while, on the belief that a ring of distinct corridors
        needed sequence-level reservation because each robot is the rightful winner of
        the segment it wants while blocked by another holding the next. That diagnosis
        was wrong. The ring was not deadlocking on resource ordering at all: robots
        were running flat -- nothing implemented Appendix A's charging cycle -- and
        finished robots were parking on charger nodes that were also task endpoints.
        Fixing those two made the ring clear with no change to arbitration.
        """
        ring = scenarios.Scenario(
            name="loop3",
            purpose=(
                "Three AMRs on a single-lane ring, the geometry Appendix C's "
                "deadlock-freedom proof is about."
            ),
            robots=3,
            map_name="loop_map",
            engine=scenarios.Engine.HEADLESS,
            transport=scenarios.TransportKind.INPROC,
            requirements=("FR-5.12", "TC-3"),
        )
        sim = build(ring, seed=1, allocator=AuctionAllocator(), task_count=6)
        assert sim.run(max_ms=1_800_000), (
            "three robots on a single-lane ring did not clear; a cyclic wait formed"
        )
        assert sim.engine.coordination_failures == []

    @pytest.mark.slow
    def test_arbitration_decisions_repeat_exactly(self) -> None:
        """FR-5.8 / NFR-2.5. Reproducibility is what makes the Appendix C argument
        checkable rather than merely plausible."""
        traces = []
        for _ in range(3):
            sim = coordinated(5)
            sim.run(max_ms=1_800_000)
            traces.append(
                [
                    (robot.robot_id, robot.plan_seq, robot.arbiter.stood, robot.arbiter.replanned)
                    for robot in sim.engine.robots
                ]
                + [str(d) for robot in sim.engine.robots for d in robot.arbiter.decisions]
            )
        assert traces[0] == traces[1] == traces[2]
