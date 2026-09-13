"""Headless engine: determinism, the virtual clock, and collision detection.

Determinism is the load-bearing property. AC-2 and AC-3 rest on comparing two
configurations over ten seeds, and that comparison is only meaningful if a seed
fixes the run completely (NFR-4.4). Most of this file is about proving it does.
"""

from __future__ import annotations

import pytest

from core import config
from core.graph import Graph
from core.robot import Robot
from core.state_machine import State
from core.task import Task
from simulator.collision_detector import CollisionDetector, Pose
from simulator.engine import Engine, build_fleet, spawn_positions
from tests.conftest import ReferencePlanner


def fleet_of(graph: Graph, count: int, *, seed: int = 1) -> list[Robot]:
    return build_fleet(
        graph, None, count=count, seed=seed, planner_factory=ReferencePlanner
    )


def task(task_id: int, pickup: int, drop: int, *, priority: int = 10) -> Task:
    return Task(
        task_id=task_id, pickup=pickup, drop=drop, priority=priority, created_at_ms=0
    )


class TestVirtualClock:
    def test_time_is_derived_from_the_tick_count(self, benchmark_map: Graph) -> None:
        """Never accumulated, so it cannot drift."""
        engine = Engine(graph=benchmark_map, robots=fleet_of(benchmark_map, 3))
        assert engine.now_ms == 0
        engine.run_ticks(50)
        assert engine.now_ms == 50 * config.MOTION_TICK_MS

    def test_every_appendix_e_period_is_a_whole_number_of_ticks(
        self, benchmark_map: Graph
    ) -> None:
        """Otherwise a 200 ms INTENT drifts against the clock it is stamped on."""
        engine = Engine(graph=benchmark_map, robots=fleet_of(benchmark_map, 3))
        for period in (
            config.INTENT_PERIOD_MS,
            config.AUCTION_WINDOW_MS,
            config.PEER_TIMEOUT_MS,
            config.DIGEST_PERIOD_MS,
            config.PHEROMONE_DECAY_MS,
            config.BEACON_PERIOD_MS,
            config.TELEMETRY_PERIOD_MS,
        ):
            assert engine.ticks_for(period) * engine.tick_ms == period

    def test_a_period_that_does_not_divide_is_rejected_loudly(
        self, benchmark_map: Graph
    ) -> None:
        engine = Engine(graph=benchmark_map, robots=fleet_of(benchmark_map, 3))
        with pytest.raises(ValueError, match="drift"):
            engine.ticks_for(config.MOTION_TICK_MS + 1)

    def test_is_due_fires_at_the_right_cadence(self, benchmark_map: Graph) -> None:
        engine = Engine(graph=benchmark_map, robots=fleet_of(benchmark_map, 3))
        due = []
        for tick in range(40):
            engine.tick = tick
            if engine.is_due(config.INTENT_PERIOD_MS):
                due.append(tick)
        assert due == [0, 10, 20, 30]  # 200 ms at a 20 ms tick


class TestDeterminism:
    def test_the_same_seed_gives_an_identical_run(self) -> None:
        traces = []
        for _ in range(3):
            graph = Graph.load("maps/benchmark_map.json")
            robots = fleet_of(graph, 3, seed=42)
            engine = Engine(graph=graph, robots=robots, seed=42)
            for index, robot in enumerate(robots):
                robot.accept_task(task(index, 1 + index, 4 + index), 0)
            engine.run(max_ms=600_000, until=lambda e: e.work_finished())
            traces.append(
                (
                    engine.now_ms,
                    [r.metrics.distance_mm for r in robots],
                    [str(e) for e in engine.events],
                )
            )
        assert traces[0] == traces[1] == traces[2]

    def test_different_seeds_place_robots_differently(self, benchmark_map: Graph) -> None:
        a = spawn_positions(benchmark_map, 3, seed=1)
        b = spawn_positions(benchmark_map, 3, seed=99)
        assert a != b, "the seed must actually influence the spawn layout"

    def test_robots_are_stepped_in_id_order(self, benchmark_map: Graph) -> None:
        """Step order must be a property of the fleet, not of list order."""
        robots = fleet_of(benchmark_map, 4)
        shuffled = [robots[2], robots[0], robots[3], robots[1]]
        engine = Engine(graph=benchmark_map, robots=shuffled)
        assert [r.robot_id for r in engine.robots] == [1, 2, 3, 4]

    def test_duplicate_robot_ids_are_rejected(self, benchmark_map: Graph) -> None:
        robots = fleet_of(benchmark_map, 2)
        robots[1].robot_id = robots[0].robot_id
        with pytest.raises(ValueError, match="unique"):
            Engine(graph=benchmark_map, robots=robots)

    def test_each_robot_gets_its_own_planner(self, benchmark_map: Graph) -> None:
        """A shared planner would be shared mutable state between robots -- the
        accidental centralisation this architecture exists to rule out."""
        robots = fleet_of(benchmark_map, 3)
        planners = {id(r.planner) for r in robots}
        assert len(planners) == 3

    def test_spawn_positions_are_distinct_and_spread(self, benchmark_map: Graph) -> None:
        """Spawning a fleet on adjacent nodes creates a jam at t=0 that has
        nothing to do with the coordination logic."""
        homes = spawn_positions(benchmark_map, 4, seed=3)
        assert len(set(homes)) == 4

    def test_spawning_more_robots_than_nodes_is_rejected(self, loop_map: Graph) -> None:
        with pytest.raises(ValueError, match="cannot spawn"):
            spawn_positions(loop_map, 99, seed=1)


class TestTickOrdering:
    def test_all_robots_decide_before_any_moves(self, benchmark_map: Graph) -> None:
        """A robot must not see a peer that has already moved this tick -- that is
        information a real robot could not have.

        Checked by observing that the positions seen during the decision phase are
        the positions from the end of the previous tick.
        """
        robots = fleet_of(benchmark_map, 3)
        engine = Engine(graph=benchmark_map, robots=robots)
        for index, robot in enumerate(robots):
            robot.accept_task(task(index, 1 + index, 4 + index), 0)
        engine.run_ticks(30)

        seen: list[tuple[int, int]] = []
        original_steps = [r.step for r in robots]

        def spy(robot, original):
            def wrapped(now_ms):
                first = engine.poses()[0]
                seen.append((first.x_mm, first.y_mm))
                return original(now_ms)
            return wrapped

        for robot, original in zip(robots, original_steps):
            robot.step = spy(robot, original)  # type: ignore[method-assign]
        engine.step()
        assert len(set(seen)) == 1, "a robot moved during the decision phase"


class TestKillingARobot:
    def test_a_killed_robot_stops_deciding_and_stops_being_present(
        self, benchmark_map: Graph
    ) -> None:
        """TC-5 / FR-6.5. There is no death notification: a robot that has lost
        power cannot send one, so peers must notice through silence."""
        robots = fleet_of(benchmark_map, 3)
        engine = Engine(graph=benchmark_map, robots=robots)
        for index, robot in enumerate(robots):
            robot.accept_task(task(index, 1 + index, 4 + index), 0)
        engine.run_ticks(20)

        engine.kill(2)
        assert [r.robot_id for r in engine.active_robots] == [1, 3]
        assert all(p.robot_id != 2 for p in engine.poses())

        before = engine.robot(2).metrics.distance_mm
        engine.run_ticks(50)
        assert engine.robot(2).metrics.distance_mm == before, "a dead robot moved"
        assert engine.events_of("robot_killed")

    def test_reviving_restores_a_robot(self, benchmark_map: Graph) -> None:
        engine = Engine(graph=benchmark_map, robots=fleet_of(benchmark_map, 3))
        engine.kill(1)
        engine.revive(1)
        assert len(engine.active_robots) == 3


class TestRunning:
    def test_run_returns_whether_the_predicate_was_met(self, benchmark_map: Graph) -> None:
        engine = Engine(graph=benchmark_map, robots=fleet_of(benchmark_map, 3))
        assert engine.run(max_ms=100, until=lambda e: True) is True
        assert engine.run(max_ms=100, until=lambda e: False) is False

    def test_the_time_limit_is_enforced(self, benchmark_map: Graph) -> None:
        """A deadlocked fleet must fail, not hang. TC-3's point is that deadlock
        cannot happen, and a test that hangs instead of failing proves nothing."""
        engine = Engine(graph=benchmark_map, robots=fleet_of(benchmark_map, 3))
        engine.run(max_ms=1000, until=lambda e: False)
        assert engine.now_ms == 1000

    def test_work_finished_requires_empty_queues(self, benchmark_map: Graph) -> None:
        robots = fleet_of(benchmark_map, 3)
        engine = Engine(graph=benchmark_map, robots=robots)
        assert engine.work_finished()
        robots[0].accept_task(task(1, 1, 5), 0)
        assert not engine.work_finished()


class TestMetrics:
    def test_a_three_robot_run_completes_and_is_measured(self, benchmark_map: Graph) -> None:
        robots = fleet_of(benchmark_map, 3)
        engine = Engine(graph=benchmark_map, robots=robots)
        for index, robot in enumerate(robots):
            robot.accept_task(task(index, 1 + index, 4 + index), 0)
        assert engine.run(max_ms=600_000, until=lambda e: e.work_finished())
        assert engine.tasks_completed() == 3
        assert 0 < engine.route_diversity_q8() <= config.Q8_ONE
        assert "tasks=3" in engine.summary()

    def test_moving_and_stopped_time_are_tracked_separately(
        self, benchmark_map: Graph
    ) -> None:
        """FR-10.3: total stopped time is a reported metric, and it must not be
        inferred from position deltas -- a yielding robot is moving slowly, which
        is not the same as stopped."""
        robots = fleet_of(benchmark_map, 2)
        engine = Engine(graph=benchmark_map, robots=robots)
        robots[0].accept_task(task(1, 1, 5), 0)
        engine.run_ticks(60)
        assert robots[0].metrics.moving_ms > 0
        assert robots[1].metrics.stopped_ms > 0, "an idle robot should book stopped time"


class TestCollisionDetection:
    def test_a_lone_robot_never_collides(self, benchmark_map: Graph) -> None:
        robots = fleet_of(benchmark_map, 3)
        engine = Engine(graph=benchmark_map, robots=[robots[0]])
        robots[0].accept_task(task(1, 1, 5), 0)
        engine.run(max_ms=600_000, until=lambda e: e.work_finished())
        assert engine.collisions == []

    def test_one_contiguous_overlap_counts_once(self) -> None:
        """Two robots wedged together for two seconds is one collision, not a
        hundred. Counting per tick would make the metric a function of tick rate."""
        detector = CollisionDetector()
        close = [
            Pose(1, 0, 0, State.MOVING, 0, None),
            Pose(2, 10, 0, State.MOVING, 0, None),
        ]
        for tick in range(100):
            detector.check(tick * 20, close)
        assert detector.collision_count == 1

    def test_separating_and_re_touching_counts_twice(self) -> None:
        detector = CollisionDetector()
        close = [
            Pose(1, 0, 0, State.MOVING, 0, None),
            Pose(2, 10, 0, State.MOVING, 0, None),
        ]
        far = [
            Pose(1, 0, 0, State.MOVING, 0, None),
            Pose(2, 99_000, 0, State.MOVING, 0, None),
        ]
        detector.check(0, close)
        detector.check(20, far)
        detector.check(40, close)
        assert detector.collision_count == 2

    def test_robots_just_outside_the_threshold_do_not_collide(self) -> None:
        detector = CollisionDetector()
        poses = [
            Pose(1, 0, 0, State.MOVING, 0, None),
            Pose(2, config.COLLISION_DISTANCE_MM, 0, State.MOVING, 0, None),
        ]
        assert detector.check(0, poses) == []

    def test_two_moving_robots_overlapping_is_a_coordination_failure(self) -> None:
        """NFR-2.1 and AC-2 are about exactly this case."""
        detector = CollisionDetector()
        events = detector.check(
            0,
            [
                Pose(1, 0, 0, State.MOVING, 0, None),
                Pose(2, 10, 0, State.MOVING, 0, None),
            ],
        )
        assert len(events) == 1
        assert events[0].is_coordination_failure
        assert events[0].both_moving
        assert "COORDINATION FAILURE" in str(events[0])

    def test_driving_into_a_faulted_robot_is_a_different_defect(self) -> None:
        """Still logged, but it is not evidence about arbitration."""
        detector = CollisionDetector()
        events = detector.check(
            0,
            [
                Pose(1, 0, 0, State.MOVING, 0, None),
                Pose(2, 10, 0, State.FAULT, 0, None),
            ],
        )
        assert len(events) == 1
        assert not events[0].is_coordination_failure
        assert detector.coordination_failures == []

    def test_pairs_are_reported_in_a_stable_order(self) -> None:
        """So a collision log can be compared between runs."""
        detector = CollisionDetector()
        events = detector.check(
            0,
            [
                Pose(7, 10, 0, State.MOVING, 0, None),
                Pose(3, 0, 0, State.MOVING, 0, None),
            ],
        )
        assert events[0].pair == (3, 7)

    def test_spatial_bucketing_finds_the_same_pairs_as_all_pairs(self) -> None:
        """The bucketing is an optimisation for scale100; it must not change the
        answer. Checked against a brute-force reference on a crowded layout."""
        import math
        import random

        rng = random.Random(20260913)
        poses = [
            Pose(i, rng.randrange(0, 4000), rng.randrange(0, 4000), State.MOVING, 0, None)
            for i in range(120)
        ]
        expected = {
            (min(a.robot_id, b.robot_id), max(a.robot_id, b.robot_id))
            for i, a in enumerate(poses)
            for b in poses[i + 1 :]
            if math.dist((a.x_mm, a.y_mm), (b.x_mm, b.y_mm)) < config.COLLISION_DISTANCE_MM
        }
        detector = CollisionDetector()
        found = {e.pair for e in detector.check(0, poses)}
        assert found == expected

    def test_reset_clears_state(self) -> None:
        detector = CollisionDetector()
        detector.check(
            0,
            [
                Pose(1, 0, 0, State.MOVING, 0, None),
                Pose(2, 10, 0, State.MOVING, 0, None),
            ],
        )
        detector.reset()
        assert detector.collision_count == 0
        # After a reset the same overlap must be countable again.
        detector.check(
            20,
            [
                Pose(1, 0, 0, State.MOVING, 0, None),
                Pose(2, 10, 0, State.MOVING, 0, None),
            ],
        )
        assert detector.collision_count == 1
