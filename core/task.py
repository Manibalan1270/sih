"""Task data model (FE-4).

A task is a journey: collect at one node, deliver at another. It carries no
assignment -- FR-4.1 is explicit that no component assigns a task to a named
robot. A task is announced, robots decide for themselves what it would cost them,
and the lowest bidder takes it. So ``holder`` here records what *happened*, never
what was *instructed*.

Two fields exist purely to make the auction correct rather than merely plausible:
``created_at_ms`` drives the aging term that discharges BR-2 (no task starves),
and ``zone_id`` scopes the announcement under FR-9.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

MAX_PRIORITY = 255
"""ASM-17: operator-supplied priority is an integer 0-255. Outside that range the
arbitration total order is undefined, so values are rejected rather than clamped."""

NO_HOLDER = -1


class TaskState(str, Enum):
    """Lifecycle of a task as any single robot understands it.

    Note this is one robot's *belief*, not a global truth: there is no central
    registry, so two robots can briefly disagree about who holds a task. FR-4.8
    resolves that in favour of the lower robot_id within one auction cycle.
    """

    ANNOUNCED = "ANNOUNCED"
    """On the mesh, unclaimed. Bids may be forming."""

    CLAIMED = "CLAIMED"
    """A robot has broadcast CLAIM. May still change hands under FR-4.10 if a
    challenger beats the holder by more than the stability margin."""

    IN_PROGRESS = "IN_PROGRESS"
    """The holder is en route. Reached once it starts moving toward pickup."""

    COMPLETE = "COMPLETE"

    UNREACHABLE = "UNREACHABLE"
    """FR-3.7: no traversable route exists. Returned to the mesh under FR-6.2
    rather than dropped, because the blockage that caused it may clear."""


class Leg(str, Enum):
    """Which half of the journey the holder is on."""

    TO_PICKUP = "TO_PICKUP"
    TO_DROP = "TO_DROP"


@dataclass(slots=True)
class Task:
    """One pickup-to-drop job."""

    task_id: int
    pickup: int
    drop: int
    priority: int
    created_at_ms: int
    """Aligned-clock time the order was accepted. The aging term measures from
    here, not from when a given robot first heard about it, so a task announced
    during a radio outage is not penalised for the outage."""

    zone_id: int = -1
    state: TaskState = TaskState.ANNOUNCED
    holder: int = NO_HOLDER
    claimed_at_ms: int | None = None
    claimed_bid: int | None = None
    """The winning bid. Retained because FR-4.10 needs the holder's own bid to
    decide whether a challenger has beaten it by more than delta."""

    leg: Leg = Leg.TO_PICKUP
    picked_up_at_ms: int | None = None
    completed_at_ms: int | None = None
    announce_count: int = 1
    """Times this task has been announced. Greater than one means it was
    re-announced under FR-4.12, FR-4.13 or FR-6.5, which the benchmark reports
    separately -- re-announcements are auction traffic that produced no work."""

    def __post_init__(self) -> None:
        if not 0 <= self.priority <= MAX_PRIORITY:
            raise ValueError(
                f"task {self.task_id}: priority {self.priority} outside 0-"
                f"{MAX_PRIORITY}; ASM-17 makes the arbitration total order "
                f"undefined outside that range"
            )
        if self.pickup == self.drop:
            raise ValueError(
                f"task {self.task_id}: pickup and drop are both node "
                f"{self.pickup}, which is not a journey"
            )

    # -- auction inputs ------------------------------------------------------

    def age_ms(self, now_ms: int) -> int:
        """How long this task has been waiting (FR-4.11, BR-2).

        Clamped at zero so a clock correction that moves time backwards cannot
        produce a negative age, which would *raise* the bid and make a stale task
        less attractive -- the exact opposite of the anti-starvation intent.
        """
        return max(0, now_ms - self.created_at_ms)

    def completion_ms(self) -> int | None:
        """Time from announcement to completion (FR-10.2).

        Measured from announcement rather than from claim, because the metric the
        success criteria care about is how long the *order* took, and time spent
        unclaimed is time the customer waited.
        """
        if self.completed_at_ms is None:
            return None
        return self.completed_at_ms - self.created_at_ms

    # -- current objective ---------------------------------------------------

    @property
    def target_node(self) -> int:
        """Where the holder should be heading right now."""
        return self.pickup if self.leg is Leg.TO_PICKUP else self.drop

    @property
    def is_open(self) -> bool:
        """Whether this task still needs a robot."""
        return self.state in (TaskState.ANNOUNCED, TaskState.UNREACHABLE)

    @property
    def is_finished(self) -> bool:
        return self.state is TaskState.COMPLETE

    # -- transitions ---------------------------------------------------------

    def claim(self, robot_id: int, bid: int, now_ms: int) -> None:
        self.state = TaskState.CLAIMED
        self.holder = robot_id
        self.claimed_at_ms = now_ms
        self.claimed_bid = bid

    def begin(self) -> None:
        if self.holder == NO_HOLDER:
            raise ValueError(f"task {self.task_id} cannot begin with no holder")
        self.state = TaskState.IN_PROGRESS

    def reach_pickup(self, now_ms: int) -> None:
        self.leg = Leg.TO_DROP
        self.picked_up_at_ms = now_ms

    def complete(self, now_ms: int) -> None:
        self.state = TaskState.COMPLETE
        self.completed_at_ms = now_ms

    def release(self, now_ms: int, *, unreachable: bool = False) -> None:
        """Return this task to the mesh (FR-6.2, FR-6.5, FR-4.12).

        Resets the leg as well as the holder: a task re-announced after its
        holder failed part-way must be re-run from pickup, since nothing
        guarantees the goods were ever collected.
        """
        del now_ms
        self.state = TaskState.UNREACHABLE if unreachable else TaskState.ANNOUNCED
        self.holder = NO_HOLDER
        self.claimed_at_ms = None
        self.claimed_bid = None
        self.leg = Leg.TO_PICKUP
        self.picked_up_at_ms = None

    def reannounce(self) -> None:
        self.state = TaskState.ANNOUNCED
        self.announce_count += 1

    def copy(self) -> "Task":
        """An independent copy.

        Each robot holds its own view of a task, updated only from messages it
        received. Sharing one object between robot objects in the in-process
        transport would smuggle in shared state and quietly make the simulation
        centralised -- the single thing this architecture exists to avoid.
        """
        return Task(
            task_id=self.task_id,
            pickup=self.pickup,
            drop=self.drop,
            priority=self.priority,
            created_at_ms=self.created_at_ms,
            zone_id=self.zone_id,
            state=self.state,
            holder=self.holder,
            claimed_at_ms=self.claimed_at_ms,
            claimed_bid=self.claimed_bid,
            leg=self.leg,
            picked_up_at_ms=self.picked_up_at_ms,
            completed_at_ms=self.completed_at_ms,
            announce_count=self.announce_count,
        )

    def __str__(self) -> str:
        held = "unheld" if self.holder == NO_HOLDER else f"r{self.holder}"
        return (
            f"task{self.task_id}[{self.pickup}->{self.drop} p{self.priority} "
            f"{self.state.value} {held} {self.leg.value}]"
        )


@dataclass
class TaskQueue:
    """A robot's committed work, capped at two tasks.

    The cap is not a convenience: ASM-18 and FR-4.9 rely on it, because insertion
    cost means enumerating where a new task could slot into the committed plan,
    and that enumeration grows combinatorially with queue length. At two tasks
    there are three insertion points and the computation stays inside the
    50 ms planning budget (FR-3.5).
    """

    capacity: int
    tasks: list[Task] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __bool__(self) -> bool:
        return bool(self.tasks)

    @property
    def is_full(self) -> bool:
        """FR-4.9 / BR-3: a robot at capacity does not bid at all."""
        return len(self.tasks) >= self.capacity

    @property
    def current(self) -> Task | None:
        """The task being worked now. Order is commitment order."""
        return self.tasks[0] if self.tasks else None

    def add(self, task: Task) -> None:
        if self.is_full:
            raise ValueError(
                f"queue already holds {len(self.tasks)} tasks (cap {self.capacity}); "
                f"FR-4.9 requires a full robot not to bid"
            )
        self.tasks.append(task)

    def insert(self, index: int, task: Task) -> None:
        """Insert at a chosen point, which is what the auction bid priced."""
        if self.is_full:
            raise ValueError(f"queue full (cap {self.capacity})")
        self.tasks.insert(index, task)

    def pop_current(self) -> Task | None:
        return self.tasks.pop(0) if self.tasks else None

    def remove(self, task_id: int) -> Task | None:
        for index, task in enumerate(self.tasks):
            if task.task_id == task_id:
                return self.tasks.pop(index)
        return None

    def holds(self, task_id: int) -> bool:
        return any(t.task_id == task_id for t in self.tasks)

    def insertion_points(self) -> range:
        """Positions a new task could occupy, for insertion-cost enumeration."""
        return range(len(self.tasks) + 1)
