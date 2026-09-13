"""Robot decision loop, and the Phase 2 stop/go gate.

The roadmap's gate for Phase 2 is "20 repeated A->B runs, no state bugs". That is
``TestPhase2Gate`` below. The repetition is the point: a state machine defect that
shows up one run in ten is far more likely than one that shows up every time, and
a single successful run proves very little.
"""

from __future__ import annotations

import pytest

from core import config
from core.graph import Graph
from core.robot import MotionCommand, Robot
from core.state_machine import State
from core.task import Leg, Task, TaskQueue, TaskState
from simulator.engine import Engine
from tests.conftest import ReferencePlanner


def make_robot(graph: Graph, *, robot_id: int = 1, home: int = 0) -> Robot:
    return Robot(
        robot_id=robot_id,
        graph=graph,
        planner=ReferencePlanner(graph),
        home_node=home,
    )


def make_task(task_id: int = 1, *, pickup: int = 1, drop: int = 5, priority: int = 10) -> Task:
    return Task(
        task_id=task_id,
        pickup=pickup,
        drop=drop,
        priority=priority,
        created_at_ms=0,
    )


def run_one_task(graph: Graph, *, seed: int, pickup: int, drop: int) -> tuple[Engine, Robot]:
    """Drive one robot through one complete task. Returns the engine and robot."""
    robot = make_robot(graph, home=0)
    engine = Engine(graph=graph, robots=[robot], seed=seed)
    robot.accept_task(make_task(pickup=pickup, drop=drop), engine.now_ms)
    finished = engine.run(max_ms=600_000, until=lambda e: e.work_finished())
    assert finished, f"seed {seed}: task did not complete within the time limit"
    return engine, robot


class TestPhase2Gate:
    """20 repeated A->B runs, no state bugs."""

    @pytest.mark.parametrize("seed", range(20))
    def test_repeated_runs_complete_cleanly(self, benchmark_map: Graph, seed: int) -> None:
        engine, robot = run_one_task(benchmark_map, seed=seed, pickup=1, drop=5)
        assert robot.state is State.IDLE
        assert robot.metrics.tasks_completed == 1
        assert not robot.queue
        assert robot.current_node == 5, "robot did not finish at the drop node"
        assert engine.collisions == [], "a lone robot collided with something"

    def test_repeated_runs_are_identical(self, benchmark_map: Graph) -> None:
        """NFR-4.4: a run is a pure function of its seed.

        The same seed twice must agree to the millisecond, not merely reach the
        same place. A wall-clock read or an unseeded dict iteration inside core/
        would show up here and nowhere else.
        """
        traces = []
        for _ in range(3):
            graph = Graph.load("maps/benchmark_map.json")
            engine, robot = run_one_task(graph, seed=7, pickup=1, drop=5)
            traces.append(
                (
                    engine.now_ms,
                    robot.metrics.distance_mm,
                    [str(t) for t in robot.machine.history],
                )
            )
        assert traces[0] == traces[1] == traces[2]

    def test_the_state_sequence_is_exactly_appendix_a(self, benchmark_map: Graph) -> None:
        _, robot = run_one_task(benchmark_map, seed=1, pickup=1, drop=5)
        states = [t.to_state for t in robot.machine.history]
        assert states == [
            State.PLANNING,   # queued work
            State.MOVING,     # route to pickup
            State.PLANNING,   # pickup reached, plan the drop leg
            State.MOVING,     # route to drop
            State.AT_DROP,
            State.IDLE,
        ]

    def test_both_legs_are_travelled(self, benchmark_map: Graph) -> None:
        """A task is a journey in two halves; visiting only the drop is a bug that
        still ends with the robot in the right place."""
        graph = benchmark_map
        robot = make_robot(graph, home=0)
        engine = Engine(graph=graph, robots=[robot], seed=1)
        task = make_task(pickup=3, drop=5)
        robot.accept_task(task, 0)

        engine.run(max_ms=600_000, until=lambda e: e.robot(1).current_node == 3)
        assert robot.current_node == 3, "never reached the pickup"

        # Arrival is physical and happens in the motion phase; switching legs is a
        # *decision* and happens on the following tick's step(). One tick of lag
        # between the two is the correct behaviour, not a race.
        assert robot.queue.current is not None
        assert robot.queue.current.leg is Leg.TO_PICKUP
        engine.step()
        assert robot.queue.current.leg is Leg.TO_DROP
        assert robot.queue.current.picked_up_at_ms is not None

        engine.run(max_ms=600_000, until=lambda e: e.work_finished())
        assert robot.current_node == 5


class TestMovement:
    def test_advance_reports_the_node_reached(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        robot._adopt_route([0, 2])
        length = benchmark_map.length_mm(robot.edge_id)
        assert robot.advance(length - 10, 20) is None
        assert robot.advance(20, 20) == 2
        assert robot.current_node == 2

    def test_overshoot_carries_into_the_next_edge(self, benchmark_map: Graph) -> None:
        """A coarse tick must not make a robot lose ground. The Webots adapter may
        be handed a much larger step than 20 ms."""
        robot = make_robot(benchmark_map, home=0)
        robot._adopt_route([0, 2, 10, 11])
        total = sum(
            benchmark_map.length_mm(benchmark_map.edge_between(a, b))
            for a, b in ((0, 2), (2, 10))
        )
        reached = robot.advance(total + 500, 20)
        assert reached == 10, "should have crossed two whole edges in one call"
        assert robot.current_node == 10
        assert robot.progress_mm == 500, "carry was discarded"

    def test_progress_fraction_is_monotonic_and_bounded(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        robot._adopt_route([0, 2])
        seen = [robot.progress_fraction_q8]
        for _ in range(100):
            robot.advance(80, 20)
            seen.append(robot.progress_fraction_q8)
            if robot.edge_id is None:
                break
        assert all(0 <= value <= config.Q8_ONE for value in seen)

    def test_position_is_interpolated_along_the_edge(self, benchmark_map: Graph) -> None:
        """The dashboard animates between junctions from this."""
        robot = make_robot(benchmark_map, home=0)
        start = robot.position_mm()
        assert start == (2000, 12000)  # L_DEPOT
        robot._adopt_route([0, 2])
        robot.advance(benchmark_map.length_mm(robot.edge_id) // 2, 20)
        midway = robot.position_mm()
        assert 2000 < midway[0] < 6000
        assert midway[1] == 12000

    def test_holding_accumulates_stopped_time(self, benchmark_map: Graph) -> None:
        """FR-10.3: total stopped time is a reported metric."""
        robot = make_robot(benchmark_map, home=0)
        robot.hold(20)
        robot.hold(20)
        assert robot.metrics.stopped_ms == 40
        assert robot.metrics.moving_ms == 0

    def test_route_must_start_where_the_robot_is(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        with pytest.raises(ValueError, match="does not start at"):
            robot._adopt_route([2, 10])

    def test_distinct_edges_are_recorded_for_route_diversity(
        self, benchmark_map: Graph
    ) -> None:
        _, robot = run_one_task(benchmark_map, seed=1, pickup=1, drop=5)
        assert len(robot.metrics.edges_used) >= 2


class TestPlanning:
    def test_unreachable_goal_returns_the_task_to_the_mesh(self) -> None:
        """FR-3.7 / FR-6.2: declare UNREACHABLE and go back to IDLE, not FAULT."""
        graph = Graph.load("maps/benchmark_map.json")
        robot = make_robot(graph, home=0)
        # Sever the right half of the warehouse entirely.
        for edge_id in (4, 7, 10):
            graph.block(edge_id)
        robot.accept_task(make_task(pickup=5, drop=6), 0)
        result = robot.step(0)
        assert robot.state is State.IDLE
        assert any("UNREACHABLE" in note for note in result.notes)

        # The task must *leave* the queue, not merely be flagged. A robot still
        # holding a task it cannot route to re-fails the same plan every tick, and
        # no other robot can take it while it is held.
        assert not robot.queue
        released = robot.drain_released_tasks()
        assert [t.state for t in released] == [TaskState.UNREACHABLE]
        assert released[0].holder == -1

    def test_an_unreachable_task_does_not_spin_the_state_machine(self) -> None:
        """Regression: releasing the task is what stops an IDLE -> PLANNING ->
        IDLE loop that would burn every tick re-failing the same plan."""
        graph = Graph.load("maps/benchmark_map.json")
        robot = make_robot(graph, home=0)
        for edge_id in (4, 7, 10):
            graph.block(edge_id)
        robot.accept_task(make_task(pickup=5, drop=6), 0)
        engine = Engine(graph=graph, robots=[robot], seed=1)
        engine.run_ticks(50)
        assert robot.state is State.IDLE
        assert robot.machine.transition_count <= 3, (
            f"{robot.machine.transition_count} transitions in 50 ticks suggests "
            f"the robot is cycling on an unroutable task"
        )

    def test_planner_is_consulted_once_per_leg(self, benchmark_map: Graph) -> None:
        """FR-3.5 budgets 50 ms per plan; replanning every tick would blow it."""
        robot = make_robot(benchmark_map, home=0)
        engine = Engine(graph=benchmark_map, robots=[robot], seed=1)
        robot.accept_task(make_task(pickup=1, drop=5), 0)
        engine.run(max_ms=600_000, until=lambda e: e.work_finished())
        assert robot.planner.calls == 2, "expected one plan per leg"

    def test_blocked_edge_on_the_route_triggers_a_replan(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        robot.accept_task(make_task(pickup=4, drop=5), 0)
        robot.step(0)
        robot.step(20)
        assert robot.state is State.MOVING
        on_route = robot.remaining_edges()
        assert 4 in on_route, "expected the route to use the choke corridor"
        assert robot.on_edge_blocked(4, 40) is True
        assert robot.state is State.REPLAN

    def test_blockage_elsewhere_is_ignored(self, benchmark_map: Graph) -> None:
        """A robot that hears about a blockage it was not going to use carries on;
        the graph is already updated so its next plan avoids it anyway."""
        robot = make_robot(benchmark_map, home=0)
        robot.accept_task(make_task(pickup=4, drop=5), 0)
        robot.step(0)
        robot.step(20)
        off_route = next(e for e in benchmark_map.edges if e not in robot.remaining_edges())
        assert robot.on_edge_blocked(off_route, 40) is False
        assert robot.state is State.MOVING


class TestTaskIntake:
    def test_accepting_work_while_idle_starts_planning(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        robot.accept_task(make_task(), 0)
        assert robot.state is State.PLANNING

    def test_a_robot_never_shares_a_task_object(self, benchmark_map: Graph) -> None:
        """Sharing one Task between robots would smuggle in shared state and make
        the simulation quietly centralised."""
        shared = make_task()
        a = make_robot(benchmark_map, robot_id=1, home=0)
        b = make_robot(benchmark_map, robot_id=2, home=7)
        a.accept_task(shared, 0)
        b.accept_task(shared, 0)
        assert a.queue.current is not b.queue.current
        assert a.queue.current.holder == 1
        assert b.queue.current.holder == 2
        assert shared.holder == -1, "the announced task was mutated by a claim"

    def test_queue_capacity_is_enforced(self, benchmark_map: Graph) -> None:
        """FR-4.9 / BR-3: at most two queued tasks."""
        robot = make_robot(benchmark_map, home=0)
        robot.accept_task(make_task(1), 0)
        robot.accept_task(make_task(2), 0)
        assert robot.queue.is_full
        with pytest.raises(ValueError, match="FR-4.9"):
            robot.accept_task(make_task(3), 0)

    def test_idle_only_while_truly_free(self, benchmark_map: Graph) -> None:
        """FR-4.3 / BR-5: the idle bid preference must not go to a busy robot."""
        robot = make_robot(benchmark_map, home=0)
        assert robot.is_idle
        robot.accept_task(make_task(), 0)
        assert not robot.is_idle

    def test_priority_is_adopted_while_working_and_dropped_after(
        self, benchmark_map: Graph
    ) -> None:
        """BR-1: priority is the first key of the arbitration total order, so a
        working robot must outrank an idle one at a junction."""
        graph = benchmark_map
        robot = make_robot(graph, home=0)
        engine = Engine(graph=graph, robots=[robot], seed=1)
        robot.accept_task(make_task(priority=200, pickup=1, drop=5), 0)
        robot.step(0)
        assert robot.task_priority == 200
        engine.run(max_ms=600_000, until=lambda e: e.work_finished())
        assert robot.task_priority == 0


class TestBattery:
    def test_battery_drains_with_distance(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        robot.drain_battery(100_000)  # 100 m
        assert robot.battery_pct < 100
        assert robot.battery_pct > 50

    def test_battery_never_goes_negative(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        robot.drain_battery(10_000_000)
        assert robot.battery_pct == 0


class TestFault:
    def test_fault_holds_position(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        robot.accept_task(make_task(), 0)
        result = robot.fault(100, "position confidence lost")
        assert robot.state is State.FAULT
        assert result.command.is_hold
        assert robot.step(120).command.is_hold


class TestTaskQueueModel:
    def test_insertion_points_cover_every_slot(self) -> None:
        """The auction prices a task at its best insertion point, so the set of
        candidate points must include before, between and after."""
        queue = TaskQueue(capacity=2)
        assert list(queue.insertion_points()) == [0]
        queue.add(make_task(1))
        assert list(queue.insertion_points()) == [0, 1]
        queue.add(make_task(2))
        assert list(queue.insertion_points()) == [0, 1, 2]

    def test_priority_outside_the_byte_range_is_rejected(self) -> None:
        """ASM-17: outside 0-255 the arbitration total order is undefined."""
        with pytest.raises(ValueError, match="ASM-17"):
            make_task(priority=256)

    def test_a_task_to_nowhere_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a journey"):
            Task(task_id=1, pickup=3, drop=3, priority=0, created_at_ms=0)

    def test_age_cannot_go_negative(self) -> None:
        """A backwards clock correction must not make a stale task less
        attractive, which is what a negative age would do to the bid."""
        task = make_task()
        task.created_at_ms = 5000
        assert task.age_ms(1000) == 0
        assert task.age_ms(9000) == 4000

    def test_completion_is_measured_from_announcement(self) -> None:
        """FR-10.2: time unclaimed is time the customer waited."""
        task = make_task()
        task.created_at_ms = 1000
        task.claim(1, 0, 4000)
        task.complete(9000)
        assert task.completion_ms() == 8000

    def test_release_resets_the_leg(self) -> None:
        """FR-6.5: a task re-announced after its holder failed must restart from
        pickup, since nothing guarantees the goods were collected."""
        task = make_task()
        task.claim(1, 0, 100)
        task.reach_pickup(200)
        assert task.leg is Leg.TO_DROP
        task.release(300)
        assert task.leg is Leg.TO_PICKUP
        assert task.holder == -1
        assert task.picked_up_at_ms is None

    def test_reannounce_counts(self) -> None:
        """Re-announcements are auction traffic that produced no work, reported
        separately by the benchmark."""
        task = make_task()
        assert task.announce_count == 1
        task.release(0)
        task.reannounce()
        assert task.announce_count == 2

    def test_a_task_cannot_begin_unheld(self) -> None:
        with pytest.raises(ValueError, match="no holder"):
            make_task().begin()


class TestMotionCommand:
    def test_hold_is_recognised(self) -> None:
        assert MotionCommand.hold().is_hold
        assert MotionCommand(speed_mm_s=0, target_node=3).is_hold
        assert MotionCommand(speed_mm_s=800, target_node=None).is_hold
        assert not MotionCommand(speed_mm_s=800, target_node=3).is_hold
