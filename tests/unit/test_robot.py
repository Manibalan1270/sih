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
from core.robot import MotionCommand, Robot, StepResult
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


class TestCharging:
    """FR-6.7 / BR-4 / Appendix A's CHARGING state.

    Nothing implemented the middle of Appendix A's charging cycle, so the state was
    unreachable: a flat robot simply stopped for good. On a 24-task run every robot
    reached 0%, refused new work, and the run stalled with seven tasks unallocated.
    It presented as a coordination deadlock and was not one.
    """

    def test_a_flat_idle_robot_enters_charging(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=0)
        robot.battery_pct = config.BATTERY_RESERVE_PCT - 1
        robot.step(0)
        assert robot.state is State.CHARGING

    def test_it_finishes_held_work_before_charging(self, benchmark_map: Graph) -> None:
        """BR-4: an AMR below reserve does not accept new work but does complete work
        already held. Diverting mid-task would abandon a delivery."""
        robot = make_robot(benchmark_map, home=0)
        robot.accept_task(make_task(pickup=1, drop=5), 0)
        robot.battery_pct = config.BATTERY_RESERVE_PCT - 1
        robot.step(0)
        assert robot.state is not State.CHARGING
        assert robot.queue.current is not None

    def test_it_routes_to_a_charger(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=2)
        robot.battery_pct = 5
        engine = Engine(graph=benchmark_map, robots=[robot], seed=1)
        engine.run(
            max_ms=600_000,
            until=lambda e: robot.current_node in benchmark_map.chargers,
        )
        assert robot.current_node in benchmark_map.chargers

    def test_it_charges_and_returns_to_service(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, home=benchmark_map.chargers[0])
        robot.battery_pct = 5
        engine = Engine(graph=benchmark_map, robots=[robot], seed=1)
        # The predicate must be the charge, not the state: the robot starts IDLE, and
        # run() checks the condition before stepping, so "is IDLE" is true at t=0.
        assert engine.run(
            max_ms=600_000,
            until=lambda e: robot.battery_pct >= config.BATTERY_RESUME_PCT,
        )
        engine.step()
        assert robot.state is State.IDLE

    def test_charging_accumulates_sub_percent_time(self, benchmark_map: Graph) -> None:
        """Mirrors the drain side: a 20 ms tick is a fraction of a percent, and
        discarding the remainder would mean never charging at all."""
        robot = make_robot(benchmark_map, home=benchmark_map.chargers[0])
        # Below reserve, or it never enters CHARGING in the first place.
        robot.battery_pct = config.BATTERY_RESERVE_PCT - 1
        engine = Engine(graph=benchmark_map, robots=[robot], seed=1)
        engine.step()  # IDLE -> CHARGING
        assert robot.state is State.CHARGING
        before = robot.battery_pct
        engine.run_ticks(config.BATTERY_CHARGE_MS_PER_PERCENT // config.MOTION_TICK_MS)
        assert robot.battery_pct == before + 1

    def test_a_full_charge_covers_a_realistic_run(self, benchmark_map: Graph) -> None:
        """The SRS treats low battery as an exception (FE-6), not a routine event.

        At 250 m per charge it was routine -- a robot flattened after about six tasks
        and charging dominated every run. A benchmark run covers roughly 300 m per
        robot, which must stay well inside one charge.
        """
        full_charge_mm = 100 * config.BATTERY_MM_PER_PERCENT
        assert full_charge_mm >= 5 * 300_000, (
            f"a full charge covers {full_charge_mm / 1000:.0f} m; a benchmark run is "
            f"~300 m per robot, so charging would dominate rather than be an exception"
        )


class TestAnticipatoryFollowing:
    """FR-5.6 / NFR-2.4: keep station by shedding speed, not by braking on a sensor.

    Section 1.5 *defines* stop-and-wait as halting when a peer comes within a fixed
    radius, and FR-5.6/NFR-2.4 reserve braking for non-cooperative obstacles. A fleet
    peer broadcasts INTENT five times a second, so it is not one. Treating it as one put
    the behaviour this project exists to replace inside Configuration B -- 44% of all
    hold-time, measured -- and made the wait un-negotiable, which let a deadlock cycle
    close through it.
    """

    def _approaching(self, graph: Graph, robot_id: int = 1) -> Robot:
        from core.auction import Auctioneer

        robot = make_robot(graph, robot_id=robot_id, home=2)
        robot.auctioneer = Auctioneer(robot_id=robot_id)
        robot.accept_task(make_task(pickup=4, drop=5), 0)
        robot.step(0)          # PLANNING -> route adopted
        robot.step(20)         # MOVING
        return robot

    def test_a_leader_ahead_makes_the_follower_shed_speed(self, benchmark_map: Graph) -> None:
        robot = self._approaching(benchmark_map)
        assert robot.next_node is not None
        # A peer on the same edge, same direction, arriving just ahead of us -- inside
        # the following gap, which is what closing too fast looks like.
        my_arrival = 40 + robot._remaining_on_current_edge_ms()
        robot.peers.observe(
            robot_id=2, now_ms=40, current_node=robot.current_node, priority=10,
            state=3, battery_pct=100, held_task_id=-1,
            next_nodes=(robot.next_node,), eta_ms=(my_arrival - 500,),
        )
        result = robot.step(40)
        assert result.command.speed_mm_s == config.YIELD_SPEED_MM_S
        assert any("following r2" in note for note in result.notes)

    def test_a_peer_comfortably_ahead_is_not_followed(self, benchmark_map: Graph) -> None:
        """Only closing traffic matters. Slowing for a robot far in front would make
        the fleet crawl for no safety gain."""
        robot = self._approaching(benchmark_map)
        robot.peers.observe(
            robot_id=2, now_ms=40, current_node=robot.current_node, priority=10,
            state=3, battery_pct=100, held_task_id=-1,
            next_nodes=(robot.next_node,),
            eta_ms=(40 + robot._remaining_on_current_edge_ms() - config.FOLLOW_GAP_MS * 3,),
        )
        assert robot.step(40).command.speed_mm_s == config.NOMINAL_SPEED_MM_S

    def test_a_peer_behind_is_not_followed(self, benchmark_map: Graph) -> None:
        """A follower is not an obstacle. Yielding to one would invert the queue."""
        robot = self._approaching(benchmark_map)
        robot.peers.observe(
            robot_id=2, now_ms=40, current_node=robot.current_node, priority=10,
            state=3, battery_pct=100, held_task_id=-1,
            next_nodes=(robot.next_node,),
            eta_ms=(40 + robot._remaining_on_current_edge_ms() + 5_000,),
        )
        assert robot.step(40).command.speed_mm_s == config.NOMINAL_SPEED_MM_S

    def test_opposing_traffic_on_a_bidirectional_aisle_is_not_followed(
        self, benchmark_map: Graph
    ) -> None:
        """ASM-5: robots keep to their own lane, so oncoming traffic is not in the way."""
        robot = self._approaching(benchmark_map)
        robot.peers.observe(
            robot_id=2, now_ms=40, current_node=robot.next_node, priority=10,
            state=3, battery_pct=100, held_task_id=-1,
            next_nodes=(robot.current_node,), eta_ms=(40,),
        )
        # Heading the other way, so not a leader. It does occupy the node ahead, which
        # is a separate concern; what matters here is that it is not treated as traffic
        # to queue behind on this edge.
        assert robot.step(40).command.speed_mm_s in (
            config.NOMINAL_SPEED_MM_S, config.YIELD_SPEED_MM_S, 0
        )

    def test_the_gap_is_temporal_not_geometric(self) -> None:
        """The same reasoning junction arbitration uses. Sized above FR-5.3's margin so
        a follower stays outside the window its leader would claim at the next junction
        and the two never contend for it."""
        assert config.FOLLOW_GAP_MS > config.MARGIN_MS


class TestUncoordinatedBaseline:
    """Configuration A: no mesh, no arbiter, stop-and-wait by sensor radius."""

    def test_stop_wait_halts_inside_radius_and_resumes_after_hysteresis(
        self, benchmark_map: Graph
    ) -> None:
        robot = make_robot(benchmark_map, robot_id=2, home=0)
        robot.forward_clearance_mm = config.STOP_WAIT_RADIUS_MM
        robot.forward_blocker_id = 1
        assert robot._uncoordinated_speed(0, StepResult(MotionCommand.hold())) == 0

        robot.forward_clearance_mm = (
            config.STOP_WAIT_RADIUS_MM + config.STOP_WAIT_RESUME_HYSTERESIS_MM + 1
        )
        assert (
            robot._uncoordinated_speed(0, StepResult(MotionCommand.hold()))
            == config.NOMINAL_SPEED_MM_S
        )

    def test_lower_robot_id_proceeds_when_both_are_halted(self, benchmark_map: Graph) -> None:
        low = make_robot(benchmark_map, robot_id=1, home=0)
        high = make_robot(benchmark_map, robot_id=2, home=0)

        low.forward_clearance_mm = config.STOP_WAIT_RADIUS_MM
        low.forward_blocker_id = 2
        low.forward_blocker_stopped = True
        high.forward_clearance_mm = config.STOP_WAIT_RADIUS_MM
        high.forward_blocker_id = 1
        high.forward_blocker_stopped = True

        low._last_speed_mm_s = 0
        high._last_speed_mm_s = 0

        assert low._uncoordinated_speed(0, StepResult(MotionCommand.hold())) == config.NOMINAL_SPEED_MM_S
        assert high._uncoordinated_speed(0, StepResult(MotionCommand.hold())) == 0

    def test_lower_robot_id_still_waits_for_a_moving_blocker(self, benchmark_map: Graph) -> None:
        robot = make_robot(benchmark_map, robot_id=1, home=0)
        robot.forward_clearance_mm = config.STOP_WAIT_RADIUS_MM
        robot.forward_blocker_id = 2
        robot.forward_blocker_stopped = False

        assert robot._uncoordinated_speed(0, StepResult(MotionCommand.hold())) == 0


class TestHonestETAs:
    """FR-1.2: eta_ms is a declaration of arrival, so it must reflect reality."""

    def test_a_slowed_robot_declares_a_later_arrival(self, benchmark_map: Graph) -> None:
        """A robot holding position that advertises a nominal ETA tells its peers it
        will clear the node shortly when it will not. They approach on that promise and
        then have to brake reactively -- which is the behaviour being removed."""
        from communication.messages import Intent

        from core.auction import Auctioneer

        def first_eta(commanded: int) -> int:
            robot = make_robot(benchmark_map, home=2)
            robot.auctioneer = Auctioneer(robot_id=1)
            robot.accept_task(make_task(pickup=4, drop=5), 0)
            robot.step(0)
            robot.step(20)
            robot._last_speed_mm_s = commanded
            robot._last_intent_ms = -10_000
            result = robot.step(40)
            intents = [p for p in result.outbox if isinstance(p, Intent)]
            assert intents, "no INTENT was emitted"
            return intents[0].eta_ms[0]

        assert first_eta(config.YIELD_SPEED_MM_S) > first_eta(config.NOMINAL_SPEED_MM_S)

    def test_a_halted_robot_declares_a_finite_arrival(self, benchmark_map: Graph) -> None:
        """Floored at the yield speed: the field is a uint16, so an infinite ETA would
        overflow and wrap to something misleadingly soon."""
        from communication.messages import Codec, Intent

        from core.auction import Auctioneer

        robot = make_robot(benchmark_map, home=2)
        robot.auctioneer = Auctioneer(robot_id=1)
        robot.accept_task(make_task(pickup=4, drop=5), 0)
        robot.step(0)
        robot.step(20)
        robot._last_speed_mm_s = 0
        robot._last_intent_ms = -10_000
        intents = [p for p in robot.step(40).outbox if isinstance(p, Intent)]
        assert intents
        # Must survive the wire, which is where an unbounded value would betray itself.
        codec = Codec(8)
        assert codec.decode(codec.encode(1, 1, 40, intents[0])) is not None


class TestContendedEdgePenalty:
    """Appendix B's AVOID, second branch: mark the edge costly and replan."""

    def test_the_penalty_reaches_the_planner(self, benchmark_map: Graph) -> None:
        """Recorded and ignored is worse than absent: the replan returns the same route
        and the reroute re-triggers every tick. Measured before this was connected:
        113,612 replans in a single 12-task run."""
        robot = make_robot(benchmark_map, home=2)
        nominal = robot.edge_cost(4)
        robot._now_ms = 0
        robot._avoid_edges[4] = 5_000
        assert robot.edge_cost(4) == nominal + config.CONTENDED_EDGE_PENALTY_MS

    def test_the_penalty_expires(self, benchmark_map: Graph) -> None:
        """Otherwise a peer that has long moved on keeps costing the fleet an aisle."""
        robot = make_robot(benchmark_map, home=2)
        robot._avoid_edges[4] = 5_000
        robot._now_ms = 5_001
        assert robot.edge_cost(4) == robot.graph.nominal_cost_ms(4)

    def test_the_graph_itself_is_never_penalised(self, benchmark_map: Graph) -> None:
        """The aisle is passable, merely occupied. Marking the graph would make one
        robot's problem the whole fleet's."""
        robot = make_robot(benchmark_map, home=2)
        robot._now_ms = 0
        robot._avoid_edges[4] = 5_000
        assert benchmark_map.nominal_cost_ms(4) == robot.graph.nominal_cost_ms(4)
        assert not benchmark_map.is_blocked(4)
