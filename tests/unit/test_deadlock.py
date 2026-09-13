"""Deadlock diagnosis.

The detector's job is to tell a *cycle* from a *chain*, because they mean opposite
things: a cycle cannot resolve however long you wait, a chain clears on its own. A
detector that conflated them would be worse than none, since it would report every
queue as a deadlock and train the reader to ignore it.
"""

from __future__ import annotations

import pytest

from core.graph import Graph
from core.robot import Robot, WaitCause
from core.state_machine import State
from simulator.deadlock import (
    StallWatcher,
    WaitEdge,
    diagnose,
    find_cycles,
    wait_edges,
)
from simulator.engine import Engine
from tests.conftest import ReferencePlanner


def robot_at(graph: Graph, robot_id: int, node: int) -> Robot:
    return Robot(
        robot_id=robot_id, graph=graph, planner=ReferencePlanner(graph), home_node=node
    )


def waiting(graph: Graph, robot_id: int, node: int, *, on: int, kind: str = "corridor"):
    robot = robot_at(graph, robot_id, node)
    robot.wait_cause = WaitCause(kind=kind, blocker_id=on, resource="e4")
    robot.machine.state = State.YIELD
    return robot


def edge(waiter: int, blocker: int, kind: str = "corridor") -> WaitEdge:
    return WaitEdge(waiter=waiter, blocker=blocker, kind=kind, resource="e4")


class TestCycleDetection:
    def test_a_two_robot_loop_is_a_cycle(self) -> None:
        cycles = find_cycles([edge(1, 2), edge(2, 1)])
        assert len(cycles) == 1
        assert set(cycles[0].robots) == {1, 2}

    def test_a_three_robot_loop_is_a_cycle(self) -> None:
        """The shape actually observed at 6 AMRs."""
        cycles = find_cycles([edge(1, 3), edge(3, 5), edge(5, 1)])
        assert len(cycles) == 1
        assert set(cycles[0].robots) == {1, 3, 5}

    def test_a_chain_is_not_a_cycle(self) -> None:
        """A waits on B waits on C, and C is free. This clears on its own, and
        calling it a deadlock would make the detector cry wolf on every queue."""
        assert find_cycles([edge(1, 2), edge(2, 3)]) == []

    def test_a_chain_that_ends_in_a_cycle_reports_only_the_cycle(self) -> None:
        """The robots in the loop are the ones that can never move; the ones queued
        behind it are merely delayed, and naming them would obscure the cause."""
        cycles = find_cycles([edge(9, 1), edge(1, 2), edge(2, 1)])
        assert len(cycles) == 1
        assert set(cycles[0].robots) == {1, 2}

    def test_two_independent_cycles_are_both_found(self) -> None:
        cycles = find_cycles([edge(1, 2), edge(2, 1), edge(5, 6), edge(6, 5)])
        assert len(cycles) == 2

    def test_no_edges_means_no_cycle(self) -> None:
        assert find_cycles([]) == []

    def test_a_self_loop_cannot_be_built(self) -> None:
        """A robot waiting on itself would be a bookkeeping error, not a deadlock."""
        graph = Graph.load("maps/benchmark_map.json")
        robot = waiting(graph, 1, 2, on=1)
        assert wait_edges([robot]) == []


class TestWaitGraph:
    def test_edges_come_from_recorded_causes(self, benchmark_map: Graph) -> None:
        robots = [
            waiting(benchmark_map, 1, 2, on=3),
            waiting(benchmark_map, 3, 4, on=1),
        ]
        edges = wait_edges(robots)
        assert {(e.waiter, e.blocker) for e in edges} == {(1, 3), (3, 1)}

    def test_a_robot_with_no_cause_contributes_nothing(self, benchmark_map: Graph) -> None:
        assert wait_edges([robot_at(benchmark_map, 1, 2)]) == []

    def test_a_wait_on_something_that_is_not_a_robot_is_dropped(
        self, benchmark_map: Graph
    ) -> None:
        """Low position confidence stops a robot, but no peer can release it, so it
        cannot be part of a cycle."""
        robot = robot_at(benchmark_map, 1, 2)
        robot.wait_cause = WaitCause(kind="confidence", blocker_id=-1, resource="J10")
        assert wait_edges([robot]) == []

    def test_a_wait_on_an_absent_robot_is_dropped(self, benchmark_map: Graph) -> None:
        """A killed peer cannot be waited on: its claims expire (FR-5.11)."""
        robot = waiting(benchmark_map, 1, 2, on=99)
        assert wait_edges([robot]) == []

    def test_edges_are_ordered_by_waiter(self, benchmark_map: Graph) -> None:
        """So two reports of the same stall read identically."""
        robots = [
            waiting(benchmark_map, 5, 2, on=1),
            waiting(benchmark_map, 1, 4, on=5),
        ]
        assert [e.waiter for e in wait_edges(robots)] == [1, 5]


class TestReport:
    def test_a_cycle_is_named_a_deadlock(self, benchmark_map: Graph) -> None:
        robots = [
            waiting(benchmark_map, 1, 2, on=3),
            waiting(benchmark_map, 3, 4, on=1),
        ]
        report = diagnose(robots, at_ms=1000, moved_recently=set())
        assert report.is_deadlocked
        text = report.describe()
        assert "deadlock" in text
        assert "r1" in text and "r3" in text

    def test_a_chain_is_described_as_clearing_itself(self, benchmark_map: Graph) -> None:
        robots = [
            waiting(benchmark_map, 1, 2, on=3),
            waiting(benchmark_map, 3, 4, on=5),
        ]
        report = diagnose(robots, at_ms=1000, moved_recently=set())
        assert not report.is_deadlocked
        assert "no cycle" in report.describe()

    def test_a_robot_stopped_with_no_reason_is_flagged(self, benchmark_map: Graph) -> None:
        """A different and usually worse defect: some path holds position without
        going through any declared wait point."""
        mute = robot_at(benchmark_map, 7, 2)
        mute.machine.state = State.MOVING
        report = diagnose([mute], at_ms=1000, moved_recently=set())
        assert report.unexplained == (7,)
        assert "no recorded cause" in report.describe()

    def test_the_mixed_cycle_shape_is_reported_in_full(self, benchmark_map: Graph) -> None:
        """The observed cycle mixed two corridor yields with one headway block.

        That mix is the whole point of recording the kind: the loop closes through a
        *geometric* link, not a right-of-way decision, so no change to arbitration
        can break it. Reporting only the robots would hide that.
        """
        robots = [
            waiting(benchmark_map, 1, 2, on=3, kind="corridor"),
            waiting(benchmark_map, 3, 4, on=5, kind="headway"),
            waiting(benchmark_map, 5, 6, on=1, kind="corridor"),
        ]
        report = diagnose(robots, at_ms=1000, moved_recently=set())
        assert report.is_deadlocked
        assert report.cycles[0].kinds == ("corridor", "headway")


class TestStallWatcher:
    def test_a_moving_fleet_is_not_stalled(self, benchmark_map: Graph) -> None:
        from core.task import Task

        robot = robot_at(benchmark_map, 1, 0)
        robot.accept_task(
            Task(task_id=1, pickup=1, drop=5, priority=10, created_at_ms=0), 0
        )
        engine = Engine(graph=benchmark_map, robots=[robot])
        engine.run_ticks(200)
        assert engine.stall_report is None

    def test_an_idle_fleet_is_not_a_stall(self, benchmark_map: Graph) -> None:
        """A fleet with no work is motionless and perfectly healthy. Reporting that
        as a deadlock would fire on every completed run."""
        robot = robot_at(benchmark_map, 1, benchmark_map.parking_nodes[0])
        engine = Engine(graph=benchmark_map, robots=[robot])
        engine.run_ticks(1000)
        assert engine.stall_report is None

    def test_movement_resets_the_clock(self, benchmark_map: Graph) -> None:
        watcher = StallWatcher(window_ms=1000)
        robot = robot_at(benchmark_map, 1, 0)
        watcher.observe([robot], 0)
        robot.metrics.distance_mm += 100
        watcher.observe([robot], 5000)
        assert 1 in watcher.moved_recently(5000)
        assert 1 not in watcher.moved_recently(9000)

    def test_the_window_is_longer_than_a_legitimate_hold(self) -> None:
        """A robot waiting out a junction conflict pauses for a safety margin, and a
        corridor traverse takes a few seconds. The window must exceed both or the
        detector reports normal operation as a deadlock."""
        from core import config

        assert StallWatcher().window_ms > config.MARGIN_DEGRADED_MS
        assert StallWatcher().window_ms > config.JUNCTION_OCCUPANCY_MS

    def test_a_motionless_working_fleet_reads_as_stalled(self, benchmark_map: Graph) -> None:
        """Checked through the watcher rather than by stepping the engine: step()
        clears each robot's wait cause and re-derives it, which is correct -- the
        cause must describe now, not an earlier tick -- but it means a manually
        planted cause cannot survive a step."""
        robots = [
            waiting(benchmark_map, 1, 2, on=2),
            waiting(benchmark_map, 2, 4, on=1),
        ]
        watcher = StallWatcher(window_ms=100)
        watcher.observe(robots, 0)
        assert watcher.is_stalled(robots, 5000)
        report = watcher.report(robots, 5000)
        assert report.is_deadlocked


@pytest.mark.slow
class TestAgainstTheRealFleet:
    """End to end, on the fleet size that used to deadlock.

    These tests previously asserted the opposite: that 6 AMRs *did* form a wait-for
    cycle, and that the cycle closed through a geometric headway link rather than a
    right-of-way decision. Both were true and both are now fixed -- the geometric link
    was the junction corner (SRS defect 10), and a robot standing in a corner is now a
    fact that peers hold outside rather than a contender they can outrank.

    They are kept pointing at the same run rather than deleted, because a regression here
    is exactly what would be easy to miss: the detector's synthetic tests above would
    still pass while the fleet quietly started deadlocking again.
    """

    def build_six(self, seed: int = 0):
        from core import scenarios
        from simulator.scenario import AuctionAllocator, build

        return build(
            scenarios.get("bench3"), seed=seed, allocator=AuctionAllocator(),
            robots=6, task_count=36, waves=1,
        )

    def test_six_robots_no_longer_form_a_wait_for_cycle(self) -> None:
        sim = self.build_six()
        for _ in range(60_000):
            sim.step()
            report = sim.engine.stall_report
            if report is not None and report.is_deadlocked:
                break
            if sim.is_finished:
                break

        report = sim.engine.stall_report
        if report is not None:
            assert not report.is_deadlocked, (
                "the 6-AMR wait-for cycle is back: " + report.describe()
            )

    def test_six_robots_do_the_work_without_colliding(self) -> None:
        """Collisions are the part that must hold. Completion at this fleet size is
        limited by the map rather than by coordination: benchmark_map has three parking
        bays, so six robots cannot all stand somewhere harmless, and a fleet with nowhere
        to idle is the capacity constraint that deadlock-free lane routing assumes away
        (every lane keeping room for one more agent). bench3 is a three-robot scenario;
        this exercises the safety layer above its intended density, not its throughput.
        """
        sim = self.build_six()
        for _ in range(60_000):
            sim.step()
            if sim.is_finished:
                break
        assert not sim.engine.coordination_failures, (
            f"{len(sim.engine.coordination_failures)} collisions at 6 AMRs"
        )
        assert len(sim.completed) >= 20, (
            f"only {len(sim.completed)} of {len(sim.task_set)} tasks done; throughput "
            f"has collapsed even allowing for the bay shortage"
        )


class TestDiagnosticsAreObservationalOnly:
    def test_instrumentation_does_not_change_a_run(self, benchmark_map: Graph) -> None:
        """NFR-4.4 and CON-7: a diagnostic that could alter behaviour would be worse
        than none. Two runs of one seed must stay identical to the millisecond."""
        from core import scenarios
        from simulator.scenario import AuctionAllocator, build

        traces = []
        for _ in range(2):
            sim = build(scenarios.get("bench3"), seed=4, allocator=AuctionAllocator())
            sim.run(max_ms=600_000)
            traces.append(
                (sim.makespan_ms, [str(e) for e in sim.engine.events if e.kind != "stall"])
            )
        assert traces[0] == traces[1]
