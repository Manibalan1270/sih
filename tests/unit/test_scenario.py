"""Scenario assembly, and the Phase 4 stop/go gate.

Phase 4's gate is "tasks complete with no central planner". That is the
architectural claim the whole project rests on, so it is tested structurally --
by checking no planner or plan is shared -- rather than by observing that the
robots happened to behave independently.

Phase 4 also expects collisions, and ``TestTheProblemExists`` asserts they happen.
The roadmap is explicit that this is intentional: three robots planning
independently with no coordination layer *must* collide, or there is nothing for
Phases 6 and 7 to fix and the benchmark has no improvement to demonstrate.
"""

from __future__ import annotations

import pytest

from core import config, scenarios
from core.robot import Robot
from core.state_machine import State
from simulator import scenario as scenario_module
from simulator.scenario import RoundRobinAllocator, build


@pytest.fixture
def sim():
    """Configuration A: round-robin, no mesh, no arbitration (FR-10.5).

    Phase 4's subject. Pinned explicitly rather than left to the default, because the
    default allocator is now the full auction and these tests are about what an
    uncoordinated fleet does.
    """
    return build(scenarios.get("bench3"), seed=42, allocator=RoundRobinAllocator())


class TestPhase4Gate:
    """Tasks complete with no central path computation."""

    def test_three_robots_complete_every_task(self, sim) -> None:
        assert sim.run(max_ms=1_800_000), "the fleet did not finish its work"
        assert len(sim.completed) == len(sim.task_set)
        assert all(t.is_finished for t in sim.completed)

    def test_no_planner_is_shared_between_robots(self, sim) -> None:
        """The structural form of "no central planner". A shared planner would be
        shared mutable state between robots even if it changed no result today."""
        planners = [id(r.planner) for r in sim.engine.robots]
        assert len(set(planners)) == len(planners)

    def test_every_robot_plans_for_itself(self, sim) -> None:
        """Not one robot planning for the fleet: each planner must do real work."""
        sim.run(max_ms=1_800_000)
        for robot in sim.engine.robots:
            assert robot.planner.stats.searches > 0, (
                f"robot {robot.robot_id} never planned, so something else routed it"
            )

    def test_no_robot_shares_a_task_object_with_the_announced_set(self, sim) -> None:
        """A shared Task would let one robot observe another's claim directly,
        which is a back channel the mesh protocol does not provide."""
        sim.run(max_ms=200_000)
        announced = {id(t) for t in sim.task_set.tasks}
        for robot in sim.engine.robots:
            for task in robot.queue:
                assert id(task) not in announced

    def test_every_robot_does_some_work(self, sim) -> None:
        sim.run(max_ms=1_800_000)
        for robot in sim.engine.robots:
            assert robot.metrics.distance_mm > 0
            assert robot.metrics.tasks_completed > 0

    def test_the_fleet_ends_idle_and_empty(self, sim) -> None:
        sim.run(max_ms=1_800_000)
        assert all(r.state is State.IDLE for r in sim.engine.robots)
        assert all(not r.queue for r in sim.engine.robots)
        assert sim.pending == []


class TestTheProblemExists:
    """Phase 4 establishes the problem the coordination layer must solve.

    The roadmap: collisions here "are intentional: you are establishing the
    problem the coordination layer must solve". These numbers are the "before"
    against which Phases 6 and 7 are judged.
    """

    def test_uncoordinated_robots_collide(self, sim) -> None:
        sim.run(max_ms=1_800_000)
        assert sim.engine.coordination_failures, (
            "three robots planning independently through a single-lane choke "
            "corridor did not collide once. Either the map is not contended or "
            "the collision detector is not working -- and Phases 6 and 7 would "
            "then have nothing to demonstrate."
        )

    def test_collisions_concentrate_on_the_contended_corridor(self, sim) -> None:
        """If collisions were scattered uniformly the map would not be doing its
        job of forcing overlap where FR-10.8 intends."""
        sim.run(max_ms=1_800_000)
        corridor_y = sim.graph.node(10).y_mm
        on_corridor = sum(
            1
            for event in sim.engine.coordination_failures
            if abs(event.y_mm - corridor_y) < 4000
        )
        assert on_corridor >= len(sim.engine.coordination_failures) // 2

    def test_an_uncoordinated_fleet_never_yields(self, sim) -> None:
        """Configuration A exchanges no intent (FR-10.5), so it cannot yield: it has
        no arbiter and no mesh. A yield here would mean the control condition is
        quietly doing some of the coordination it exists to be measured against."""
        sim.run(max_ms=1_800_000)
        assert sum(r.metrics.yields_lost for r in sim.engine.robots) == 0
        assert all(r.arbiter is None for r in sim.engine.robots)
        assert sim.mesh is None


class TestDeterminism:
    def test_the_same_seed_gives_an_identical_run(self) -> None:
        """NFR-4.4. This is the property AC-2 and AC-3 depend on: a comparison
        across ten seeds means nothing if a seed does not fix the run."""
        traces = []
        for _ in range(3):
            sim = build(scenarios.get("bench3"), seed=11)
            sim.run(max_ms=1_800_000)
            traces.append(
                (
                    sim.makespan_ms,
                    len(sim.engine.collisions),
                    [str(e) for e in sim.engine.events],
                )
            )
        assert traces[0] == traces[1] == traces[2]

    def test_different_seeds_give_different_runs(self) -> None:
        results = set()
        for seed in (1, 2, 3, 4):
            sim = build(scenarios.get("bench3"), seed=seed)
            sim.run(max_ms=1_800_000)
            results.add(sim.makespan_ms)
        assert len(results) > 1, "the seed is not influencing the run"


class TestRoundRobinAllocator:
    """Configuration A's allocator (FR-10.5), and Phase 4's stand-in."""

    def test_work_is_spread_across_the_fleet(self, sim) -> None:
        sim.run(max_ms=1_800_000)
        assigned = [r.metrics.tasks_completed for r in sim.engine.robots]
        assert min(assigned) > 0
        # Round-robin takes no account of position or load, so the spread is even
        # by construction. That evenness is exactly its weakness: a robot far from
        # a pickup is as likely to get it as one passing by.
        assert max(assigned) - min(assigned) <= 2

    def test_it_takes_no_account_of_position(self, sim) -> None:
        """The property that makes it a valid control for the auction.

        If round-robin preferred nearby robots it would already be doing part of
        the auction's job, and the measured improvement in Phase 11 would
        understate the difference between the two configurations.

        Set up the case the auction is supposed to get right: robot 1 sits at the
        far end of the warehouse, robot 2 is parked on the pickup node itself.
        Round-robin must still hand the task to robot 1, because its cursor is
        there and it knows nothing about geography.
        """
        allocator = RoundRobinAllocator()
        far, near = sim.engine.robots[0], sim.engine.robots[1]
        far.current_node, near.current_node = 7, 1  # R_DEPOT and L_TOP
        task = sim.task_set.tasks[0]
        task.pickup, task.drop = 1, 6
        assert allocator._offer(sim, task, 0) is True
        assert far.queue.holds(task.task_id), (
            "round-robin preferred the closer robot, so it is not a clean control"
        )
        assert not near.queue

    def test_a_full_robot_is_skipped(self, sim) -> None:
        """FR-4.9 / BR-3: at most two queued tasks."""
        for robot in sim.engine.robots[:2]:
            while not robot.queue.is_full:
                robot.accept_task(sim.task_set.tasks[0].copy(), 0)
        sim.allocator.tick(sim, 0)
        for robot in sim.engine.robots[:2]:
            assert len(robot.queue) <= config.QUEUE_CAP

    def test_a_low_battery_robot_is_skipped(self, sim) -> None:
        """BR-4 / FR-4.15: below reserve an AMR accepts no new work."""
        flat = sim.engine.robots[0]
        flat.battery_pct = config.BATTERY_RESERVE_PCT - 1
        sim.allocator.tick(sim, 0)
        assert not flat.queue

    def test_a_faulted_robot_is_skipped(self, sim) -> None:
        broken = sim.engine.robots[0]
        broken.fault(0, "test")
        sim.allocator.tick(sim, 0)
        assert not broken.queue

    def test_unallocatable_tasks_are_deferred_not_lost(self, sim) -> None:
        """FR-4.13: the task is re-announced and its aging term keeps accruing.

        A dropped task would make the run report a makespan over fewer tasks than
        it was given, which is a silently flattering result.
        """
        for robot in sim.engine.robots:
            robot.battery_pct = 0
        sim.allocator.tick(sim, 0)
        assert sim.pending, "tasks nobody could take were dropped"


class TestTaskRecovery:
    def test_a_released_task_returns_to_the_pending_pool(self, sim) -> None:
        """FR-6.2 / FR-6.5. A task a robot let go of must be re-announced, not
        silently vanish."""
        sim.run(max_ms=60_000)
        robot: Robot = sim.engine.robots[0]
        held = robot.queue.current
        if held is None:
            pytest.skip("robot held no task at this point in the run")
        robot._release_current(sim.engine.now_ms)
        before = held.announce_count
        sim.take_pending(sim.engine.now_ms)
        assert held.announce_count == before + 1

    def test_reannouncements_are_logged(self, sim) -> None:
        sim.run(max_ms=60_000)
        robot = sim.engine.robots[0]
        if robot.queue.current is None:
            pytest.skip("robot held no task at this point in the run")
        robot._release_current(sim.engine.now_ms)
        sim.take_pending(sim.engine.now_ms)
        assert sim.engine.events_of("reannounce")


class TestBuild:
    def test_every_scenario_builds(self) -> None:
        for name in scenarios.SCENARIOS:
            chosen = scenarios.get(name)
            robots = 3 if chosen.physical_robots else None
            sim = build(chosen, seed=1, robots=robots, task_count=4)
            assert sim.engine.robots
            assert len(sim.task_set) == 4

    def test_zoning_follows_the_scenario(self) -> None:
        """Section 4.9: at 3-5 AMRs a flood auction is correct, so bench3 is
        unzoned by design rather than by omission."""
        assert not build(scenarios.get("bench3"), seed=1).zones.enabled
        assert build(scenarios.get("scale100"), seed=1, task_count=4).zones.enabled

    def test_robots_are_assigned_zones_from_their_spawn(self) -> None:
        """FR-9.1 / ASM-16: static, derived from the map, no negotiation."""
        sim = build(scenarios.get("scale100"), seed=1, robots=6, task_count=4)
        for robot in sim.engine.robots:
            assert robot.zone_id == sim.zones.zone_of(robot.home_node)

    def test_a_hardware_only_scenario_cannot_be_built_without_robots(self) -> None:
        with pytest.raises(ValueError, match="no simulated robots"):
            build(scenarios.get("hardware2"), seed=1)

    def test_notes_are_suppressed_at_scale(self) -> None:
        """Decision notes are the demo for a small fleet and would dominate memory
        at 100 robots without being read."""
        small = build(scenarios.get("bench3"), seed=1)
        large = build(scenarios.get("scale100"), seed=1, task_count=4)
        assert small.engine.log_notes
        assert not large.engine.log_notes

    def test_the_map_is_validated_against_the_scenario_id_width(self) -> None:
        graph, _ = scenario_module.load_map(scenarios.get("scale100"))
        assert len(graph.edges) > config.MAX_EDGES_8BIT


class TestReporting:
    def test_makespan_is_measured_from_the_first_release(self, sim) -> None:
        """FR-10.2. Measuring from the first *completion* would hide the time the
        earliest orders spent waiting."""
        sim.run(max_ms=1_800_000)
        assert sim.makespan_ms > 0
        latest = max(t.completed_at_ms for t in sim.completed)
        earliest_release = min(t.created_at_ms for t in sim.task_set)
        assert sim.makespan_ms == latest - earliest_release

    def test_report_names_the_numbers_that_matter(self, sim) -> None:
        sim.run(max_ms=1_800_000)
        report = sim.report()
        for expected in ("makespan", "collisions", "stopped time", "route diversity"):
            assert expected in report

    def test_an_unfinished_run_is_not_reported_as_finished(self) -> None:
        sim = build(scenarios.get("bench3"), seed=1)
        assert sim.run(max_ms=1000) is False
        assert not sim.is_finished
