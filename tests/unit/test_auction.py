"""Distributed task allocation (FE-4), and TC-9 through TC-16.

The central claim is that insertion-cost bidding resolves the choice between a
busy robot and an idle one *correctly, without a hand-written rule*. TC-12, TC-13
and TC-14 are the three cases that pin that down, and they are the reason the bid
is marginal cost rather than distance.
"""

from __future__ import annotations

import pytest

from core import config
from core.auction import (
    NO_WINNER,
    Auctioneer,
    OpenAuction,
    SubmittedBid,
    aging_credit,
    battery_penalty,
    committed_cost,
)
from core.graph import Graph
from core.planner_astar import AStarPlanner
from core.task import Leg, Task, TaskQueue


def task(task_id: int = 1, *, pickup: int = 1, drop: int = 5, priority: int = 10,
         created_at_ms: int = 0, zone: int = -1) -> Task:
    return Task(
        task_id=task_id, pickup=pickup, drop=drop, priority=priority,
        created_at_ms=created_at_ms, zone_id=zone,
    )


@pytest.fixture
def planner(benchmark_map: Graph) -> AStarPlanner:
    return AStarPlanner(benchmark_map)


class TestBidFormula:
    """Section 4.4's formula, term by term."""

    def test_an_idle_robot_bids_its_full_travel_cost(self, planner) -> None:
        auctioneer = Auctioneer(robot_id=1)
        bid = auctioneer.price(
            task(pickup=1, drop=5), start_node=0, queue=TaskQueue(2),
            battery_pct=100, travel_cost=planner.travel_cost, now_ms=0, is_idle=True,
        )
        assert bid is not None
        expected = planner.travel_cost(0, 1) + planner.travel_cost(1, 5)
        assert bid.insertion_cost == expected

    def test_the_idle_credit_lowers_the_bid(self, planner) -> None:
        """FR-4.3 / BR-5: on a near tie, prefer the idle robot."""
        auctioneer = Auctioneer(robot_id=1)
        shared = dict(
            start_node=0, queue=TaskQueue(2), battery_pct=100,
            travel_cost=planner.travel_cost, now_ms=0,
        )
        idle = auctioneer.price(task(), is_idle=True, **shared)
        busy = auctioneer.price(task(), is_idle=False, **shared)
        assert idle.value == busy.value - config.EPSILON_MS

    def test_low_battery_raises_the_bid(self, planner) -> None:
        auctioneer = Auctioneer(robot_id=1)
        shared = dict(
            start_node=0, queue=TaskQueue(2), travel_cost=planner.travel_cost,
            now_ms=0, is_idle=True,
        )
        healthy = auctioneer.price(task(), battery_pct=100, **shared)
        flat = auctioneer.price(task(), battery_pct=25, **shared)
        assert flat.value > healthy.value
        assert healthy.battery_penalty == 0

    def test_the_battery_penalty_is_zero_above_the_knee(self) -> None:
        """So a healthy fleet bids on insertion cost alone and the term does not
        quietly distort ordinary allocation."""
        assert battery_penalty(100) == 0
        assert battery_penalty(config.BATTERY_PENALTY_KNEE_PCT) == 0
        assert battery_penalty(config.BATTERY_PENALTY_KNEE_PCT - 10) > 0

    def test_waiting_lowers_the_bid(self, planner) -> None:
        """FR-4.11 / BR-2: the whole anti-starvation mechanism."""
        auctioneer = Auctioneer(robot_id=1)
        shared = dict(
            start_node=0, queue=TaskQueue(2), battery_pct=100,
            travel_cost=planner.travel_cost, is_idle=True,
        )
        fresh = auctioneer.price(task(created_at_ms=0), now_ms=0, **shared)
        stale = auctioneer.price(task(created_at_ms=0), now_ms=60_000, **shared)
        assert stale.value < fresh.value

    def test_a_bid_may_legitimately_go_negative(self, planner) -> None:
        """Clamping at zero would flatten exactly the ordering that makes a very
        old task finally win."""
        auctioneer = Auctioneer(robot_id=1)
        ancient = auctioneer.price(
            task(created_at_ms=0), start_node=0, queue=TaskQueue(2), battery_pct=100,
            travel_cost=planner.travel_cost, now_ms=10_000_000, is_idle=True,
        )
        assert ancient.value < 0

    def test_aging_credit_grows_with_age(self) -> None:
        assert aging_credit(task(created_at_ms=0), 0) == 0
        assert aging_credit(task(created_at_ms=0), 40_000) > aging_credit(
            task(created_at_ms=0), 10_000
        )

    def test_the_breakdown_explains_itself(self, planner) -> None:
        """NFR-4.4 wants decisions reconstructable; the dashboard renders this."""
        auctioneer = Auctioneer(robot_id=3)
        bid = auctioneer.price(
            task(7), start_node=0, queue=TaskQueue(2), battery_pct=100,
            travel_cost=planner.travel_cost, now_ms=0, is_idle=True,
        )
        text = str(bid)
        assert "r3" in text and "task 7" in text and "insertion" in text


class TestInsertionCost:
    """Why the bid is marginal cost and not distance."""

    def test_committed_cost_chains_pickup_and_drop(self, planner) -> None:
        queue = TaskQueue(2)
        queue.add(task(1, pickup=1, drop=5))
        total = committed_cost(0, queue, planner.travel_cost)
        assert total == planner.travel_cost(0, 1) + planner.travel_cost(1, 5)

    def test_a_collected_task_is_priced_from_where_it_actually_is(self, planner) -> None:
        """Pricing a half-finished task from its pickup again would inflate every
        bid a busy robot makes, and hand work to idle robots that should not win."""
        queue = TaskQueue(2)
        held = task(1, pickup=1, drop=5)
        held.leg = Leg.TO_DROP
        queue.add(held)
        assert committed_cost(0, queue, planner.travel_cost) == planner.travel_cost(0, 5)

    def test_remaining_objectives_shrink_after_pickup(self) -> None:
        held = task(1, pickup=1, drop=5)
        assert held.remaining_objectives() == (1, 5)
        held.reach_pickup(0)
        assert held.remaining_objectives() == (5,)

    def test_every_insertion_point_is_priced(self, planner) -> None:
        """FR-4.3's "best point". The queue cap of two makes this at most three
        candidate positions, which is what keeps it inside FR-3.5's budget --
        ASM-18 exists because it grows combinatorially otherwise."""
        queue = TaskQueue(2)
        queue.add(task(1, pickup=1, drop=2))
        auctioneer = Auctioneer(robot_id=1)
        bid = auctioneer.price(
            task(2, pickup=4, drop=6), start_node=0, queue=queue, battery_pct=100,
            travel_cost=planner.travel_cost, now_ms=0, is_idle=False,
        )
        assert bid is not None
        assert bid.insert_at in (0, 1)

    def test_an_unroutable_task_produces_no_bid(self, benchmark_map: Graph) -> None:
        for edge_id in (4, 7, 10):
            benchmark_map.block(edge_id)
        planner = AStarPlanner(benchmark_map)
        auctioneer = Auctioneer(robot_id=1)
        assert auctioneer.price(
            task(pickup=5, drop=6), start_node=0, queue=TaskQueue(2), battery_pct=100,
            travel_cost=planner.travel_cost, now_ms=0, is_idle=True,
        ) is None


class TestInsertionCostCases:
    """TC-12, TC-13, TC-14 -- the three cases the mechanism exists for."""

    def _bid(self, planner, *, start: int, queue: TaskQueue, idle: bool, candidate: Task):
        return Auctioneer(robot_id=1).price(
            candidate, start_node=start, queue=queue, battery_pct=100,
            travel_cost=planner.travel_cost, now_ms=0, is_idle=idle,
        )

    def test_tc12_busy_robot_already_passing_the_pickup_wins(self, planner) -> None:
        """A task on a busy robot's way costs it almost nothing, and total fleet
        cost is lower than sending a distant idle robot."""
        busy_queue = TaskQueue(2)
        busy_queue.add(task(1, pickup=2, drop=4))  # L_MID -> R_MID, through the choke
        busy = self._bid(
            planner, start=0, queue=busy_queue, idle=False,
            candidate=task(2, pickup=10, drop=11),  # a hop inside that same corridor
        )
        idle = self._bid(
            planner, start=7, queue=TaskQueue(2), idle=True,  # parked at R_DEPOT
            candidate=task(2, pickup=10, drop=11),
        )
        assert busy.value < idle.value, (
            "a robot already traversing the corridor should outbid a distant idle "
            "robot for a task inside it"
        )

    def test_tc13_busy_robot_that_must_backtrack_loses(self, planner) -> None:
        """The case a raw-distance bid gets wrong. A busy robot near the pickup
        would win on distance while charging the detour to work it had promised."""
        busy_queue = TaskQueue(2)
        busy_queue.add(task(1, pickup=1, drop=8))  # committed to the top bypass
        busy = self._bid(
            planner, start=1, queue=busy_queue, idle=False,
            candidate=task(2, pickup=3, drop=9),  # the bottom of the warehouse
        )
        idle = self._bid(
            planner, start=3, queue=TaskQueue(2), idle=True,
            candidate=task(2, pickup=3, drop=9),
        )
        assert idle.value < busy.value, (
            "a robot that would have to backtrack should lose to a well-placed "
            "idle robot"
        )

    def test_tc14_a_near_tie_goes_to_the_idle_robot(self, planner) -> None:
        """FR-4.3 / BR-5 / TC-14: where bids differ by less than epsilon, the idle
        robot wins.

        Constructing an exact tie is the cleanest way to show only the idle term
        separates them. Both robots sit at R_MID. The busy one holds a task it has
        already collected whose drop is R_MID itself, so servicing it costs nothing
        more and its insertion cost for the candidate is identical to the idle
        robot's. Any difference in the final bid is therefore the idle credit and
        nothing else.
        """
        candidate = task(2, pickup=4, drop=5)

        collected = task(1, pickup=1, drop=4)
        collected.leg = Leg.TO_DROP  # already picked up; only the drop remains
        busy_queue = TaskQueue(2)
        busy_queue.add(collected)

        busy = self._bid(planner, start=4, queue=busy_queue, idle=False, candidate=candidate)
        idle = self._bid(planner, start=4, queue=TaskQueue(2), idle=True, candidate=candidate)

        assert busy.insertion_cost == idle.insertion_cost, (
            "the setup did not produce a tie, so this is not testing the idle term"
        )
        assert idle.value == busy.value - config.EPSILON_MS
        assert idle.value < busy.value


class TestEligibility:
    def test_tc16_a_full_robot_does_not_bid(self) -> None:
        """FR-4.9 / BR-3: auction traffic must not grow with task backlog."""
        auctioneer = Auctioneer(robot_id=1)
        allowed, reason = auctioneer.may_bid(
            queue_length=config.QUEUE_CAP, battery_pct=100,
            zone_eligible=True, faulted=False,
        )
        assert not allowed
        assert "FR-4.9" in reason

    def test_a_flat_robot_does_not_bid(self) -> None:
        """FR-4.15 / BR-4."""
        allowed, reason = Auctioneer(robot_id=1).may_bid(
            queue_length=0, battery_pct=config.BATTERY_RESERVE_PCT - 1,
            zone_eligible=True, faulted=False,
        )
        assert not allowed
        assert "FR-4.15" in reason

    def test_an_out_of_zone_robot_does_not_bid(self) -> None:
        """FR-9.3, which is what bounds auction traffic at fleet scale."""
        allowed, reason = Auctioneer(robot_id=1).may_bid(
            queue_length=0, battery_pct=100, zone_eligible=False, faulted=False,
        )
        assert not allowed
        assert "FR-9.3" in reason

    def test_a_faulted_robot_does_not_bid(self) -> None:
        allowed, _ = Auctioneer(robot_id=1).may_bid(
            queue_length=0, battery_pct=100, zone_eligible=True, faulted=True,
        )
        assert not allowed

    def test_a_healthy_idle_robot_may_bid(self) -> None:
        allowed, reason = Auctioneer(robot_id=1).may_bid(
            queue_length=0, battery_pct=100, zone_eligible=True, faulted=False,
        )
        assert allowed and reason == ""


class TestWinnerDetermination:
    """FR-4.5, FR-4.6: lowest bid, ties to the lower id, decided independently."""

    def test_the_lowest_bid_wins(self) -> None:
        auction = OpenAuction(task=task(), opened_at_ms=0)
        for robot_id, value in ((1, 5000), (2, 3000), (3, 7000)):
            auction.record(SubmittedBid(robot_id, value, False))
        assert auction.winner() == 2

    def test_a_tie_goes_to_the_lower_robot_id(self) -> None:
        auction = OpenAuction(task=task(), opened_at_ms=0)
        for robot_id in (3, 1, 2):
            auction.record(SubmittedBid(robot_id, 4000, False))
        assert auction.winner() == 1

    def test_no_bids_means_no_winner(self) -> None:
        """FR-4.13: re-announced with its aging term still accruing."""
        assert OpenAuction(task=task(), opened_at_ms=0).winner() == NO_WINNER

    def test_the_runner_up_is_identified(self) -> None:
        """FR-4.12 makes them responsible for re-announcing if the winner never
        claims, so responsibility is fixed rather than left to whoever notices --
        which would produce a burst of duplicate announcements."""
        auction = OpenAuction(task=task(), opened_at_ms=0)
        for robot_id, value in ((1, 5000), (2, 3000), (3, 7000)):
            auction.record(SubmittedBid(robot_id, value, False))
        assert auction.runner_up() == 1

    def test_a_robot_counts_its_own_bid(self) -> None:
        """A robot that left its own bid out would conclude a peer had won an
        auction it actually won itself."""
        from core.auction import BidBreakdown

        auction = OpenAuction(task=task(), opened_at_ms=0)
        auction.record(SubmittedBid(2, 9000, False))
        auction.record_own(
            BidBreakdown(1, 1, 1000, 0, 0, 0, 0, True)
        )
        assert auction.winner() == 1

    def test_a_resent_bid_replaces_the_earlier_one(self) -> None:
        auction = OpenAuction(task=task(), opened_at_ms=0)
        auction.record(SubmittedBid(1, 9000, False))
        auction.record(SubmittedBid(1, 1000, False))
        assert auction.winning_value() == 1000


class TestWindow:
    def test_the_window_is_measured_from_the_announcement(self) -> None:
        """FR-4.4. From the ANNOUNCE, not from when this robot heard it, so every
        robot closes at the same aligned-clock moment."""
        auction = OpenAuction(task=task(), opened_at_ms=1000)
        assert not auction.is_window_closed(1000 + config.AUCTION_WINDOW_MS - 20)
        assert auction.is_window_closed(1000 + config.AUCTION_WINDOW_MS)

    def test_a_claim_is_overdue_after_the_timeout(self) -> None:
        auction = OpenAuction(task=task(), opened_at_ms=0)
        closed = config.AUCTION_WINDOW_MS
        assert not auction.claim_overdue(closed + config.CLAIM_TIMEOUT_MS - 20)
        assert auction.claim_overdue(closed + config.CLAIM_TIMEOUT_MS)

    def test_a_claimed_auction_is_never_overdue(self) -> None:
        auction = OpenAuction(task=task(), opened_at_ms=0)
        auction.claimed_by = 2
        assert not auction.claim_overdue(1_000_000)


class TestClaimCollision:
    """TC-9 / FR-4.8: one lost BID is enough to make two robots both claim."""

    def test_the_lower_robot_id_retains_the_task(self) -> None:
        auctioneer = Auctioneer(robot_id=5)
        auctioneer.on_announce(task(1), 0)
        auctioneer.on_claim(1, robot_id=3, value=100, now_ms=400)
        note = auctioneer.on_claim(1, robot_id=2, value=90, now_ms=410)
        assert "r2 retains" in note
        assert auctioneer.settled[1] == 2

    def test_a_robot_that_lost_the_collision_must_let_go(self) -> None:
        auctioneer = Auctioneer(robot_id=7)
        auctioneer.on_announce(task(1), 0)
        auctioneer.on_claim(1, robot_id=7, value=100, now_ms=400)
        assert not auctioneer.must_relinquish(1)
        auctioneer.on_claim(1, robot_id=2, value=100, now_ms=410)
        assert auctioneer.must_relinquish(1)

    def test_a_duplicate_claim_changes_nothing(self) -> None:
        auctioneer = Auctioneer(robot_id=5)
        auctioneer.on_announce(task(1), 0)
        auctioneer.on_claim(1, robot_id=3, value=100, now_ms=400)
        assert auctioneer.on_claim(1, robot_id=3, value=100, now_ms=420) == "duplicate claim"

    def test_a_claim_for_an_unheard_auction_is_still_recorded(self) -> None:
        """A robot out of range during the ANNOUNCE must not later bid on work that
        is already being done."""
        auctioneer = Auctioneer(robot_id=5)
        auctioneer.on_claim(9, robot_id=3, value=100, now_ms=0)
        assert auctioneer.settled[9] == 3


class TestStability:
    """FR-4.10 / BR-6 / TC-15: no thrashing."""

    def test_a_marginal_improvement_does_not_move_a_claimed_task(self) -> None:
        auction = OpenAuction(task=task(), opened_at_ms=0)
        auction.claimed_by, auction.claimed_value = 2, 5000
        assert not auction.beats_holder(5000 - config.DELTA_MS + 1)

    def test_a_clear_improvement_does(self) -> None:
        auction = OpenAuction(task=task(), opened_at_ms=0)
        auction.claimed_by, auction.claimed_value = 2, 5000
        assert auction.beats_holder(5000 - config.DELTA_MS - 1)

    def test_an_unclaimed_task_is_always_beatable(self) -> None:
        assert OpenAuction(task=task(), opened_at_ms=0).beats_holder(999_999)


class TestReannouncement:
    def test_a_duplicate_announce_does_not_reopen_a_decided_auction(self) -> None:
        """Otherwise a second winner could be produced for work already claimed."""
        auctioneer = Auctioneer(robot_id=1)
        assert auctioneer.on_announce(task(1), 0) is not None
        auctioneer.conclude(1, winner=2)
        assert auctioneer.on_announce(task(1), 100) is None

    def test_a_genuine_reannouncement_is_auctioned_afresh(self) -> None:
        """FR-4.12, FR-4.13 and FR-6.5 all put a task back on the mesh. If a robot
        kept it settled, the task would be attributed to a robot that is not doing
        it and nobody would ever bid again -- which showed up as an infinite
        re-announce loop before this was handled.

        Section 3.4.2's ANNOUNCE carries no attempt counter, so the frame timestamp
        is what distinguishes the two cases (IF-4.4 guarantees one is present).
        """
        auctioneer = Auctioneer(robot_id=1)
        auctioneer.on_announce(task(1), 0)
        auctioneer.conclude(1, winner=2)
        stale_after = config.AUCTION_WINDOW_MS + config.CLAIM_TIMEOUT_MS
        assert auctioneer.on_announce(task(1), stale_after + 1) is not None

    def test_forgetting_a_task_allows_a_fresh_auction(self) -> None:
        auctioneer = Auctioneer(robot_id=1)
        auctioneer.on_announce(task(1), 0)
        auctioneer.conclude(1, winner=2)
        auctioneer.forget(1)
        assert auctioneer.on_announce(task(1), 10) is not None

    def test_closing_lists_only_undecided_auctions(self) -> None:
        auctioneer = Auctioneer(robot_id=1)
        auctioneer.on_announce(task(1), 0)
        auctioneer.on_announce(task(2), 0)
        assert len(auctioneer.closing(config.AUCTION_WINDOW_MS)) == 2
        auctioneer.conclude(1, winner=1)
        assert [a.task.task_id for a in auctioneer.closing(config.AUCTION_WINDOW_MS)] == [2]

    def test_auctions_are_processed_in_task_id_order(self) -> None:
        """FR-5.8's determinism requirement reaches here too: two robots must
        process a batch of closing auctions in the same order or they can reach
        different conclusions about a full queue."""
        auctioneer = Auctioneer(robot_id=1)
        for task_id in (5, 2, 9, 1):
            auctioneer.on_announce(task(task_id), 0)
        order = [a.task.task_id for a in auctioneer.closing(config.AUCTION_WINDOW_MS)]
        assert order == sorted(order)
