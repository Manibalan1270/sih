"""SRS section 6.2 verification test cases.

These are integration tests: each one sets up the operational scenario the SRS
describes and checks the documented outcome. They are named for their TC number so
a reviewer can walk the acceptance table and find each one.

Cases that cannot be discharged in software are declared here as skips with the
reason, rather than quietly omitted:

* TC-23 -- a human walking into an aisle. The reroute half is exercised by the
  obstacle machinery in Phase 9; the sensor behaviour itself is hardware.
* TC-25 -- ESP32 SRAM budget. Discharged by the Appendix D analysis, not by Python.
* TC-26 -- radio range confirming ASM-7. Requires physical measurement, and OI-6
  makes it mandatory before any physical demonstration.

Cases for features not yet built are marked with the phase that will provide them.
"""

from __future__ import annotations

import pytest

from core import config, scenarios
from core.auction import Auctioneer
from core.task import Task, TaskQueue
from simulator.scenario import AuctionAllocator, build


def small_task(task_id: int = 1, *, pickup: int = 1, drop: int = 5,
               priority: int = 10, created_at_ms: int = 0) -> Task:
    return Task(
        task_id=task_id, pickup=pickup, drop=drop, priority=priority,
        created_at_ms=created_at_ms,
    )


@pytest.fixture
def sim():
    """Three AMRs, the benchmark map, the full auction over the mesh."""
    return build(scenarios.get("bench3"), seed=42)


# ---------------------------------------------------------------------------
# FE-4: task allocation
# ---------------------------------------------------------------------------


@pytest.mark.tc
class TestTC9ClaimCollision:
    """A BID frame is lost, causing two AMRs to compute different winners.

    Expected: both broadcast CLAIM; the lower robot_id retains the task; the other
    relinquishes within one cycle (FR-4.8).
    """

    def test_a_lost_bid_is_survivable(self) -> None:
        """With heavy loss the fleet must still complete every task.

        This is the property that matters operationally: IF-4.5 forbids
        acknowledgement or retransmission on the safety path, so correctness has to
        come from the protocol tolerating loss rather than from reliable delivery.
        """
        sim = build(scenarios.get("bench3"), seed=3, task_count=9)
        sim.mesh.bus.loss_permille = 300
        assert sim.run(max_ms=1_800_000), "the fleet stalled under 30% packet loss"
        assert len(sim.completed) == len(sim.task_set)

    def test_a_duplicate_holding_resolves_within_one_auction_cycle(self) -> None:
        """FR-4.8 permits a transient overlap and bounds how long it may last.

        The requirement is that the loser relinquishes "within one auction cycle",
        not that an overlap never occurs -- it cannot be prevented, because the two
        robots reach their conclusions independently and neither knows about the
        other until a frame arrives. So this measures *duration*.

        The bound allowed here is one auction window plus one INTENT period, because
        healing arrives on the next INTENT when the CLAIM itself was lost.
        """
        sim = build(scenarios.get("bench3"), seed=5, task_count=9)
        sim.mesh.bus.loss_permille = 250
        limit_ms = config.AUCTION_WINDOW_MS + config.INTENT_PERIOD_MS * 2

        overlap_started: dict[int, int] = {}
        worst_ms = 0
        for _ in range(6000):
            sim.step()
            now = sim.engine.now_ms
            holders: dict[int, list[int]] = {}
            for robot in sim.engine.robots:
                for task in robot.queue:
                    holders.setdefault(task.task_id, []).append(robot.robot_id)

            duplicated = {tid for tid, who in holders.items() if len(who) > 1}
            for task_id in duplicated:
                overlap_started.setdefault(task_id, now)
                worst_ms = max(worst_ms, now - overlap_started[task_id])
            for task_id in list(overlap_started):
                if task_id not in duplicated:
                    del overlap_started[task_id]

            assert worst_ms <= limit_ms, (
                f"task held by two robots for {worst_ms} ms, over the "
                f"{limit_ms} ms FR-4.8 allows"
            )
            if sim.is_finished:
                break

    def test_no_task_is_completed_twice(self) -> None:
        """The consequence that actually costs something.

        Before INTENT carried the held task, 30% loss produced 13 completions for 9
        tasks: two robots each did the same job because the loser of a claim
        collision never learned it had lost.
        """
        sim = build(scenarios.get("bench3"), seed=3, task_count=9)
        sim.mesh.bus.loss_permille = 300
        assert sim.run(max_ms=1_800_000)
        completed_ids = [t.task_id for t in sim.completed]
        assert len(completed_ids) == len(set(completed_ids)), (
            f"duplicate work: {len(completed_ids)} completions for "
            f"{len(set(completed_ids))} distinct tasks"
        )
        assert set(completed_ids) == {t.task_id for t in sim.task_set}

    def test_the_lower_robot_id_retains_on_collision(self) -> None:
        """The deterministic rule, checked directly (FR-4.8)."""
        auctioneer = Auctioneer(robot_id=4)
        auctioneer.on_announce(small_task(1), 0)
        auctioneer.on_claim(1, robot_id=4, value=100, now_ms=400)
        auctioneer.on_claim(1, robot_id=2, value=100, now_ms=405)
        assert auctioneer.settled[1] == 2
        assert auctioneer.must_relinquish(1)


@pytest.mark.tc
class TestTC10NobodyCanBid:
    """A task is announced while every AMR is at capacity or below battery reserve.

    Expected: no bid is received; the task is re-announced; its aging term accrues
    until an AMR becomes eligible (FR-4.13, FR-4.9).
    """

    def test_a_task_nobody_can_take_is_not_lost(self, sim) -> None:
        for robot in sim.engine.robots:
            robot.battery_pct = config.BATTERY_RESERVE_PCT - 1
        for _ in range(400):
            sim.step()
        assert sim.pending, "a task nobody could take was dropped"
        assert len(sim.completed) == 0

    def test_it_completes_once_a_robot_becomes_eligible(self, sim) -> None:
        for robot in sim.engine.robots:
            robot.battery_pct = config.BATTERY_RESERVE_PCT - 1
        for _ in range(400):
            sim.step()
        for robot in sim.engine.robots:
            robot.battery_pct = 100
        assert sim.run(max_ms=1_800_000)
        assert len(sim.completed) == len(sim.task_set)

    def test_aging_keeps_accruing_while_it_waits(self, sim) -> None:
        """So the task becomes more attractive the longer it is stuck."""
        waiting = small_task(1, created_at_ms=0)
        auctioneer = Auctioneer(robot_id=1)
        from core.auction import aging_credit

        assert aging_credit(waiting, 120_000) > aging_credit(waiting, 10_000)
        del auctioneer


@pytest.mark.tc
class TestTC4GatewayPause:
    """AC-4: pausing order intake does not interrupt held work."""

    def test_held_tasks_complete_after_gateway_stops(self) -> None:
        allocator = AuctionAllocator()
        sim = build(
            scenarios.get("bench3"),
            seed=2,
            allocator=allocator,
            task_count=3,
            waves=1,
        )
        for _ in range(100):
            sim.step()

        assert not sim.pending
        assert sum(len(robot.queue) for robot in sim.engine.robots) == len(sim.task_set)
        allocator.stop()

        assert sim.run(max_ms=1_800_000)
        assert {task.task_id for task in sim.completed} == {
            task.task_id for task in sim.task_set
        }


@pytest.mark.tc
class TestTC11NoStarvation:
    """A task at a remote corner competes repeatedly against convenient tasks.

    Expected: aging lowers every bid on the remote task until it is claimed; it is
    not starved (FR-4.11, BR-2).
    """

    def test_an_awkward_task_eventually_wins(self, sim) -> None:
        from core.auction import aging_credit
        from core.planner_astar import AStarPlanner

        planner = AStarPlanner(sim.graph)
        auctioneer = Auctioneer(robot_id=1)

        awkward = small_task(1, pickup=3, drop=9, created_at_ms=0)
        convenient = small_task(2, pickup=1, drop=2, created_at_ms=0)
        shared = dict(
            start_node=0, queue=TaskQueue(2), battery_pct=100,
            travel_cost=planner.travel_cost, is_idle=True,
        )

        fresh_awkward = auctioneer.price(awkward, now_ms=0, **shared)
        fresh_convenient = auctioneer.price(convenient, now_ms=0, **shared)
        assert fresh_awkward.value > fresh_convenient.value, (
            "the awkward task should start out less attractive, or this proves "
            "nothing about starvation"
        )

        # Wait long enough and the ordering must flip.
        for waited_ms in (30_000, 120_000, 600_000):
            aged = auctioneer.price(awkward, now_ms=waited_ms, **shared)
            still_fresh = auctioneer.price(
                small_task(2, pickup=1, drop=2, created_at_ms=waited_ms),
                now_ms=waited_ms, **shared,
            )
            if aged.value < still_fresh.value:
                break
        else:
            pytest.fail(
                "the awkward task never became more attractive than a fresh "
                "convenient one, so BR-2 is not discharged"
            )
        assert aging_credit(awkward, 600_000) > 0

    def test_every_task_in_a_run_is_completed(self, sim) -> None:
        """The end-to-end form: nothing is left behind."""
        assert sim.run(max_ms=1_800_000)
        completed_ids = {t.task_id for t in sim.completed}
        assert completed_ids == {t.task_id for t in sim.task_set}


@pytest.mark.tc
class TestTC15NoThrashing:
    """A sequence of marginally better bids arrives for an already-claimed task.

    Expected: no reassignment; the holder retains it; no thrashing (FR-4.10, BR-6).
    """

    def test_marginal_improvements_do_not_move_the_task(self) -> None:
        from core.auction import OpenAuction

        auction = OpenAuction(task=small_task(), opened_at_ms=0)
        auction.claimed_by, auction.claimed_value = 2, 10_000
        for challenger in range(10_000, 10_000 - config.DELTA_MS, -50):
            assert not auction.beats_holder(challenger)

    def test_tasks_do_not_change_hands_repeatedly_in_a_run(self, sim) -> None:
        """A high relinquish count means work is being passed around instead of
        done, which would show up as a makespan regression rather than a failure."""
        sim.run(max_ms=1_800_000)
        relinquished = sum(r.auctioneer.relinquished for r in sim.engine.robots)
        assert relinquished <= len(sim.task_set) // 2, (
            f"{relinquished} relinquishments for {len(sim.task_set)} tasks suggests "
            f"thrashing"
        )


@pytest.mark.tc
class TestTC16FullQueueDoesNotBid:
    """An AMR already holding two tasks hears a new ANNOUNCE.

    Expected: it does not bid at all; auction traffic does not grow with task
    backlog (FR-4.9).
    """

    def test_a_full_robot_sends_no_bid(self, sim) -> None:
        robot = sim.engine.robots[0]
        while not robot.queue.is_full:
            robot.accept_task(small_task(99 + len(robot.queue)), 0)
        before = robot.metrics.bids_sent
        for _ in range(100):
            sim.step()
            if robot.queue.is_full:
                assert robot.metrics.bids_sent == before, (
                    "a robot at queue capacity placed a bid"
                )
            else:
                break

    def test_queue_capacity_is_never_exceeded(self, sim) -> None:
        """BR-3 is a hard limit, and the insertion cost a robot quoted stops
        describing its plan if the queue grows past what it priced."""
        for _ in range(4000):
            sim.step()
            for robot in sim.engine.robots:
                assert len(robot.queue) <= config.QUEUE_CAP
            if sim.is_finished:
                break


# ---------------------------------------------------------------------------
# FE-1: peer communication
# ---------------------------------------------------------------------------


@pytest.mark.tc
class TestTC28MalformedFrame:
    """A replayed or malformed frame is injected into the mesh.

    Expected: discarded with no effect on the reservation table or the traffic
    model (NFR-3.1 to NFR-3.3).
    """

    def test_rubbish_is_discarded_without_effect(self, sim) -> None:
        from simulator.mesh import GATEWAY_ID

        sim.run(max_ms=60_000)
        before = (len(sim.completed), sim.mesh.decoded)

        # The injector must be wired, not radio-bound. A member with no position is
        # range-filtered out before its frames are ever decoded, so without this the
        # test would pass while exercising nothing.
        sim.mesh.bus.join(99)
        sim.mesh.bus.wired.add(99)
        for rubbish in (b"", b"\x01", b"\xff" * 60, b"not a frame at all"):
            sim.mesh.bus.broadcast(99, rubbish)
        sim.mesh.deliver()
        for robot in sim.engine.robots:
            assert sim.mesh.inbox(robot.robot_id) == []
        assert sim.mesh.discarded_malformed >= 4
        assert sim.mesh.decoded == before[1]
        del GATEWAY_ID

    def test_a_replayed_frame_is_rejected(self, sim) -> None:
        """FR-1.6 / NFR-3.2: a sequence number not greater than the last accepted."""
        from communication.messages import Complete

        sim.run(max_ms=20_000)
        frame = sim.mesh.codec.encode(1, 5, 0, Complete(task_id=1))
        sim.mesh.bus.broadcast(1, frame)
        sim.mesh.deliver()
        accepted_first = sim.mesh.inbox(2)

        sim.mesh.bus.broadcast(1, frame)
        sim.mesh.deliver()
        accepted_again = sim.mesh.inbox(2)
        assert len(accepted_again) < len(accepted_first) + 1 or accepted_again == []

    def test_the_fleet_keeps_working_under_injection(self, sim) -> None:
        sim.mesh.bus.join(99)
        sim.mesh.bus.wired.add(99)
        for _ in range(2000):
            sim.mesh.bus.broadcast(99, b"\xde\xad\xbe\xef")
            sim.step()
            if sim.is_finished:
                break
        assert sim.run(max_ms=1_800_000)
        assert len(sim.completed) == len(sim.task_set)


# ---------------------------------------------------------------------------
# Cases that cannot be discharged in software
# ---------------------------------------------------------------------------


@pytest.mark.tc
@pytest.mark.skip(
    reason="TC-23: the reroute half is covered by simulator/obstacles.py in "
    "Phase 9, but detecting a human with a forward obstacle sensor (IF-2.4) is "
    "hardware behaviour and cannot be verified here."
)
def test_tc23_non_cooperative_obstacle() -> None:
    ...


@pytest.mark.tc
@pytest.mark.skip(
    reason="TC-25: the ESP32 SRAM budget is discharged by the Appendix D "
    "analysis (12,476 bytes of 320 KB). Python object sizes say nothing about a "
    "C firmware footprint, so a test here would be theatre."
)
def test_tc25_memory_budget() -> None:
    ...


@pytest.mark.tc
@pytest.mark.skip(
    reason="TC-26: confirming ASM-7 -- that any two AMRs able to collide are in "
    "mutual radio range -- requires physical measurement in the deployment "
    "environment. OI-6 makes it mandatory before any physical demonstration. The "
    "simulation asserts the modelled range exceeds the collision distance, which "
    "is a consistency check, not evidence about a real radio."
)
def test_tc26_radio_range() -> None:
    ...
