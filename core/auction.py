"""Distributed task allocation by sealed-window reverse auction (FE-4).

Tasks are not assigned. A task is announced, every eligible robot works out what
that task would cost *it*, and the lowest bidder takes it. The mechanism follows
the Contract Net Protocol [R9], with one crucial choice: the bid is the *marginal
insertion cost* of the task into the bidder's existing plan, not raw distance.

Section 4.4 explains why that matters, and it is worth restating because it is the
single idea the whole feature turns on. An idle robot bids its full travel cost,
because it has nothing to disrupt. A busy robot bids only the detour the task adds
to its committed plan: small where the task lies on its way, large where it would
have to backtrack. A raw-distance bid would let a busy robot near the pickup outbid
every idle robot while silently charging the delay to the task it had already
promised.

The other thing to understand is that **there is no auctioneer**. FR-4.6 requires
every robot to determine the winner independently from the bids it happened to
hear, with no dispatcher and nobody informed of the outcome. So every robot runs
its own copy of the bookkeeping below and reaches its own conclusion. Usually they
agree. When a lost BID makes them disagree, two robots both CLAIM, and FR-4.8
resolves it in favour of the lower robot_id within one cycle -- which is TC-9.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core import config
from core.task import Task, TaskQueue

NO_WINNER = -1


@dataclass(frozen=True, slots=True)
class BidBreakdown:
    """A bid and every term that produced it.

    Kept in full rather than reduced to one number because the dashboard's
    per-robot decision panel renders it (so a jury can see three separate brains
    reaching three different prices), and because NFR-4.4 requires a decision to be
    reconstructable from its inputs.
    """

    task_id: int
    robot_id: int
    insertion_cost: int
    battery_penalty: int
    aging_credit: int
    idle_credit: int
    insert_at: int
    """Queue position the insertion cost was priced at. The robot must insert the
    task *there* if it wins, or it has committed to a plan it did not price."""

    is_idle: bool

    @property
    def value(self) -> int:
        """The bid itself, per section 4.4's formula.

        Not clamped at zero. The aging and idle terms subtract, so a long-waiting
        task genuinely can price below zero, and clamping would flatten exactly
        the ordering that discharges FR-4.11's anti-starvation guarantee.
        """
        return (
            self.insertion_cost
            + self.battery_penalty
            - self.aging_credit
            - self.idle_credit
        )

    def __str__(self) -> str:
        return (
            f"r{self.robot_id} bids {self.value} on task {self.task_id} "
            f"(insertion {self.insertion_cost}, battery +{self.battery_penalty}, "
            f"aging -{self.aging_credit}, idle -{self.idle_credit}, "
            f"at queue slot {self.insert_at})"
        )


@dataclass(slots=True)
class SubmittedBid:
    robot_id: int
    value: int
    is_idle: bool


@dataclass
class OpenAuction:
    """One robot's view of one auction in progress.

    "One robot's view" is the point. This is not shared state -- every robot holds
    its own instance, populated only from frames it received, and two robots'
    copies can legitimately differ if a BID was lost.
    """

    task: Task
    opened_at_ms: int
    bids: dict[int, SubmittedBid] = field(default_factory=dict)
    own_bid: BidBreakdown | None = None
    claimed_by: int = NO_WINNER
    claimed_value: int | None = None
    claimed_at_ms: int | None = None
    settled: bool = False

    def record(self, bid: SubmittedBid) -> None:
        """Record a peer's bid. A resent bid replaces the earlier one."""
        self.bids[bid.robot_id] = bid

    def record_own(self, bid: BidBreakdown) -> None:
        """Enter this robot's own bid into the same pool as its peers'.

        Necessary, not merely tidy: ``winner`` ranks whatever is in ``bids``, and a
        robot that left its own bid out would conclude a peer had won an auction it
        actually won itself.
        """
        self.bids[bid.robot_id] = SubmittedBid(
            robot_id=bid.robot_id, value=bid.value, is_idle=bid.is_idle
        )

    def is_window_closed(self, now_ms: int) -> bool:
        """FR-4.4: the window is 300 ms from the ANNOUNCE timestamp.

        Measured from the announcement, not from when this robot first heard it, so
        every robot closes the window at the same aligned-clock moment even if the
        ANNOUNCE reached them a tick apart.
        """
        return now_ms - self.opened_at_ms >= config.AUCTION_WINDOW_MS

    def winner(self) -> int:
        """The lowest bid heard, ties to the lower robot_id (FR-4.5).

        Sorting on ``(value, robot_id)`` makes the tie-break part of the ordering
        rather than a separate rule, so it cannot be forgotten at a call site.
        """
        if not self.bids:
            return NO_WINNER
        return min(self.bids.values(), key=lambda b: (b.value, b.robot_id)).robot_id

    def runner_up(self) -> int:
        """Second place, which FR-4.12 makes responsible for re-announcing if the
        winner never claims."""
        ranked = sorted(self.bids.values(), key=lambda b: (b.value, b.robot_id))
        return ranked[1].robot_id if len(ranked) > 1 else NO_WINNER

    def winning_value(self) -> int | None:
        winner = self.winner()
        return self.bids[winner].value if winner != NO_WINNER else None

    def claim_overdue(self, now_ms: int) -> bool:
        """FR-4.12: no CLAIM within 2000 ms of the window closing."""
        if self.claimed_by != NO_WINNER:
            return False
        closed_at = self.opened_at_ms + config.AUCTION_WINDOW_MS
        return now_ms - closed_at >= config.CLAIM_TIMEOUT_MS

    def beats_holder(self, challenger_value: int) -> bool:
        """FR-4.10 / BR-6: a claimed task moves only on a clear improvement.

        The stability margin is what stops thrashing. Without it a stream of
        marginally better bids would pass a task from robot to robot, each handover
        discarding the planning already done -- TC-15 exists for exactly this.
        """
        if self.claimed_value is None:
            return True
        return challenger_value < self.claimed_value - config.DELTA_MS


class PlanCostFn:
    """Protocol-ish helper type: see ``committed_cost``."""


def committed_cost(
    start_node: int,
    queue: TaskQueue | list[Task],
    travel_cost,
    *,
    extra: Task | None = None,
    insert_at: int | None = None,
) -> int:
    """Cost of servicing a queue of tasks in order from ``start_node``.

    The robot's committed plan is a chain: go to the first task's remaining
    objective, then its drop, then the next task's pickup, and so on. ``extra``
    inserted at ``insert_at`` prices the plan *with* a candidate task, which is the
    other half of the insertion-cost subtraction.

    A task already part-done is priced from where it actually is: if its pickup is
    already made, only the run to the drop remains. Pricing a half-finished task
    from its pickup again would inflate every bid a busy robot makes and hand work
    to idle robots that should not have won it.
    """
    tasks = list(queue)
    if extra is not None:
        position = len(tasks) if insert_at is None else insert_at
        tasks.insert(position, extra)

    total = 0
    at = start_node
    for task in tasks:
        for objective in task.remaining_objectives():
            leg = travel_cost(at, objective)
            if leg >= config.INFINITE_COST:
                return config.INFINITE_COST
            total += leg
            at = objective
    return total


def battery_penalty(battery_pct: int) -> int:
    """FR-4.3: low charge raises the bid.

    Zero above the knee, so a healthy fleet bids on insertion cost alone and the
    term does not quietly distort ordinary allocation.
    """
    shortfall = max(0, config.BATTERY_PENALTY_KNEE_PCT - battery_pct)
    return config.q8_mul(config.W1_Q8, shortfall)


def aging_credit(task: Task, now_ms: int) -> int:
    """FR-4.11 / BR-2: a waiting task becomes cheaper until somebody takes it.

    This is the whole anti-starvation mechanism. A task in an awkward corner loses
    every auction against convenient work, and without this it would lose forever;
    with it, its bid falls until it wins. TC-11 is the test.
    """
    return config.q8_mul(config.W2_Q8, task.age_ms(now_ms))


@dataclass
class Auctioneer:
    """One robot's participation in task allocation.

    Named for what it does locally, not for a role in the fleet: there is no
    auctioneer in the protocol (FR-4.6). Every robot runs one of these.
    """

    robot_id: int
    queue_cap: int = config.QUEUE_CAP
    open_auctions: dict[int, OpenAuction] = field(default_factory=dict)
    settled: dict[int, int] = field(default_factory=dict)
    """task_id -> winning robot_id, for auctions this robot has concluded."""

    _settled_at: dict[int, int] = field(default_factory=dict, repr=False)
    """task_id -> aligned time the auction for it was opened. Used by
    ``on_announce`` to tell a duplicate ANNOUNCE from a genuine re-announcement."""

    bids_placed: int = 0
    auctions_won: int = 0
    relinquished: int = 0
    """Tasks given up under FR-4.8 after a CLAIM collision. Reported because a
    high count means BIDs are being lost, which is a radio problem masquerading as
    an allocation problem."""

    # -- eligibility ---------------------------------------------------------

    def may_bid(
        self,
        *,
        queue_length: int,
        battery_pct: int,
        zone_eligible: bool,
        faulted: bool,
        pickup_contested: bool = False,
    ) -> tuple[bool, str]:
        """Whether this robot may bid, and why not if it may not.

        The reason is returned rather than logged here so the caller can put it in
        the run log; a robot that silently declines to bid is very hard to debug
        from the outside.
        """
        if faulted:
            return False, "faulted"
        if queue_length >= self.queue_cap:
            return False, f"queue full ({queue_length}/{self.queue_cap}, FR-4.9)"
        if battery_pct < config.BATTERY_RESERVE_PCT:
            return False, f"battery {battery_pct}% below reserve (FR-4.15)"
        if not zone_eligible:
            return False, "task is not in this robot's zone or an adjacent one (FR-9.3)"
        if pickup_contested:
            return False, "another robot is already routed to this pickup (deferred bid filter)"
        return True, ""

    # -- bidding -------------------------------------------------------------

    def price(
        self,
        task: Task,
        *,
        start_node: int,
        queue: TaskQueue,
        battery_pct: int,
        travel_cost,
        now_ms: int,
        is_idle: bool,
    ) -> BidBreakdown | None:
        """Compute this robot's bid, or None if the task is unroutable for it.

        Every insertion point is priced and the cheapest is kept (FR-4.3's "best
        point"). The queue cap of two makes that at most three candidate positions,
        which is what keeps the enumeration inside FR-3.5's planning budget --
        ASM-18 exists precisely because this grows combinatorially otherwise.
        """
        base = committed_cost(start_node, queue, travel_cost)
        if base >= config.INFINITE_COST:
            return None

        best_cost: int | None = None
        best_at = 0
        for position in range(len(list(queue)) + 1):
            with_task = committed_cost(
                start_node, queue, travel_cost, extra=task, insert_at=position
            )
            if with_task >= config.INFINITE_COST:
                continue
            marginal = with_task - base
            if best_cost is None or marginal < best_cost:
                best_cost, best_at = marginal, position

        if best_cost is None:
            return None  # no insertion point yields a routable plan

        return BidBreakdown(
            task_id=task.task_id,
            robot_id=self.robot_id,
            insertion_cost=best_cost,
            battery_penalty=battery_penalty(battery_pct),
            aging_credit=aging_credit(task, now_ms),
            idle_credit=config.EPSILON_MS if is_idle else 0,
            insert_at=best_at,
            is_idle=is_idle,
        )

    # -- protocol events -----------------------------------------------------

    def on_announce(self, task: Task, announced_at_ms: int) -> OpenAuction | None:
        """Open an auction for an announced task, or None to ignore it.

        The hard part is telling a *duplicate* ANNOUNCE from a *re-announcement*.
        A duplicate within the same auction window must not reopen a decided
        auction and produce a second winner. But a genuine re-announcement -- under
        FR-4.12 when a winner never claimed, FR-4.13 when nobody bid, or FR-6.5
        when a holder was lost -- must be auctioned afresh, or the task is settled
        against a robot that is not doing it and nobody ever bids again.

        Section 3.4.2's ANNOUNCE carries no attempt counter, so the two cannot be
        distinguished by payload. The frame's own timestamp settles it, which
        IF-4.4 guarantees is present: an announcement arriving after the previous
        auction's window plus the FR-4.12 claim timeout can only be a fresh
        attempt, because by then the earlier one was over.
        """
        settled_at = self._settled_at.get(task.task_id)
        if settled_at is not None:
            stale_after = settled_at + config.AUCTION_WINDOW_MS + config.CLAIM_TIMEOUT_MS
            if announced_at_ms <= stale_after:
                return None  # a duplicate of an auction already decided
            self.forget(task.task_id)

        existing = self.open_auctions.get(task.task_id)
        if existing is not None:
            return existing
        auction = OpenAuction(task=task.copy(), opened_at_ms=announced_at_ms)
        self.open_auctions[task.task_id] = auction
        self._settled_at[task.task_id] = announced_at_ms
        return auction

    def on_bid(self, task_id: int, robot_id: int, value: int, is_idle: bool) -> None:
        auction = self.open_auctions.get(task_id)
        if auction is None:
            return  # a bid for something we never heard announced
        auction.record(SubmittedBid(robot_id=robot_id, value=value, is_idle=is_idle))

    def on_claim(self, task_id: int, robot_id: int, value: int, now_ms: int) -> str:
        """Record a peer's CLAIM. Returns what this robot concluded.

        FR-4.8: where two robots claim the same task, the lower robot_id retains
        it. This is reachable in normal operation, not only under attack -- a
        single lost BID is enough to make two robots compute different winners
        (TC-9).
        """
        auction = self.open_auctions.get(task_id)
        if auction is None:
            # Already concluded, or never heard announced. The lower id still wins:
            # without this the last CLAIM to arrive would take the task regardless
            # of id, and a robot that had correctly won would hand it to a higher
            # id -- which is how it presented, with the lowest-numbered robot
            # relinquishing to a higher one.
            previous = self.settled.get(task_id)
            self.settled[task_id] = (
                robot_id if previous is None else min(previous, robot_id)
            )
            return (
                f"claim for a settled auction; task {task_id} stands with "
                f"r{self.settled[task_id]}"
            )

        if auction.claimed_by == NO_WINNER:
            auction.claimed_by = robot_id
            auction.claimed_value = value
            auction.claimed_at_ms = now_ms
            self.settled[task_id] = robot_id
            return f"task {task_id} claimed by r{robot_id}"

        if robot_id == auction.claimed_by:
            return "duplicate claim"

        # A collision. Lower id wins, and that is decided identically by every
        # robot that heard both claims, so no negotiation is needed.
        winner = min(auction.claimed_by, robot_id)
        loser = max(auction.claimed_by, robot_id)
        auction.claimed_by = winner
        if winner == robot_id:
            auction.claimed_value = value
        self.settled[task_id] = winner
        return f"claim collision on task {task_id}: r{winner} retains, r{loser} yields"

    def must_relinquish(self, task_id: int) -> bool:
        """FR-4.8: whether this robot lost a CLAIM collision and must let go."""
        auction = self.open_auctions.get(task_id)
        if auction is None:
            return self.settled.get(task_id, self.robot_id) != self.robot_id
        return auction.claimed_by != NO_WINNER and auction.claimed_by != self.robot_id

    def conclude(self, task_id: int, winner: int) -> None:
        self.settled[task_id] = winner
        auction = self.open_auctions.pop(task_id, None)
        if auction is not None:
            auction.settled = True
        if winner == self.robot_id:
            self.auctions_won += 1

    def forget(self, task_id: int) -> None:
        """Drop all memory of a task, so a re-announcement is auctioned afresh.

        Needed when a task comes back to the mesh after its holder failed
        (FR-6.5): the previous auction's conclusion is no longer relevant, and
        leaving it in ``settled`` would stop this robot ever bidding again.
        """
        self.open_auctions.pop(task_id, None)
        self.settled.pop(task_id, None)
        self._settled_at.pop(task_id, None)

    def closing(self, now_ms: int) -> list[OpenAuction]:
        """Auctions whose window has closed and which still need a decision."""
        return [
            auction
            for auction in sorted(self.open_auctions.values(), key=lambda a: a.task.task_id)
            if auction.is_window_closed(now_ms) and not auction.settled
        ]

    def overdue_claims(self, now_ms: int) -> list[OpenAuction]:
        """FR-4.12: auctions whose winner never claimed."""
        return [
            auction
            for auction in sorted(self.open_auctions.values(), key=lambda a: a.task.task_id)
            if auction.claim_overdue(now_ms) and not auction.settled
        ]
