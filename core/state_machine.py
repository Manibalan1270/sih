"""Robot state machine (Appendix A).

The SRS gives the state model as a table of entered-when / exited-when /
broadcasting. This module makes that table executable and, more importantly,
makes the *absence* of a transition an error rather than a silent no-op. A robot
that slides from MOVING straight to IDLE without passing through AT_DROP has lost
a task, and the symptom would otherwise be a missing completion record hundreds
of ticks later.

Appendix A's own note is worth keeping in mind while reading the table: YIELD and
REPLAN are excursions from MOVING and return to it directly. They are normal
operating states on a busy floor, not faults. An AMR that never enters them is an
AMR operating alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    """The nine states of Appendix A."""

    IDLE = "IDLE"
    BIDDING = "BIDDING"
    PLANNING = "PLANNING"
    MOVING = "MOVING"
    YIELD = "YIELD"
    REPLAN = "REPLAN"
    AT_DROP = "AT_DROP"
    CHARGING = "CHARGING"
    FAULT = "FAULT"


class Event(str, Enum):
    """What can happen to a robot. One event, one cause."""

    TASK_ANNOUNCED = "TASK_ANNOUNCED"
    AUCTION_WON = "AUCTION_WON"
    AUCTION_LOST = "AUCTION_LOST"
    QUEUE_HAS_WORK = "QUEUE_HAS_WORK"
    ROUTE_READY = "ROUTE_READY"
    ROUTE_UNREACHABLE = "ROUTE_UNREACHABLE"
    LEG_COMPLETE = "LEG_COMPLETE"
    CONFLICT_LOST = "CONFLICT_LOST"
    CONFLICT_CLEARED = "CONFLICT_CLEARED"
    EDGE_BLOCKED = "EDGE_BLOCKED"
    ROUTE_REPAIRED = "ROUTE_REPAIRED"
    ARRIVED_AT_DROP = "ARRIVED_AT_DROP"
    TASK_REPORTED = "TASK_REPORTED"
    BATTERY_LOW = "BATTERY_LOW"
    CHARGED = "CHARGED"
    FAULT_DETECTED = "FAULT_DETECTED"
    RECOVERED = "RECOVERED"


# The transition table. Read it as: in this state, this event moves you there.
#
# Two entries deserve explanation because they are not literally in Appendix A:
#
# IDLE + QUEUE_HAS_WORK -> PLANNING.  Appendix A describes the single-task path,
#   where IDLE is left only by winning an auction. BR-3 permits two queued tasks,
#   so a robot that finishes one must be able to plan the next without winning it
#   again. Routing that through BIDDING would re-auction work it already holds.
#
# YIELD + EDGE_BLOCKED -> REPLAN.  Appendix B's AVOID step has two outcomes: shed
#   speed, or mark the edge costly and invoke D* Lite. The second is a replan, so
#   YIELD must be able to reach REPLAN directly rather than via MOVING.
#
# MOVING + LEG_COMPLETE -> PLANNING.  Appendix A says MOVING is exited when "the
#   drop node is reached", which silently assumes one leg. A task is a journey in
#   two halves (ASM-10: pickup and drop are both graph nodes), so reaching the
#   *pickup* also ends a route and needs a new one. PLANNING is where routes come
#   from, so that is where it goes. Without this the pickup arrival would have to
#   masquerade as a blockage, which would corrupt the replan metric.
TRANSITIONS: dict[State, dict[Event, State]] = {
    State.IDLE: {
        Event.TASK_ANNOUNCED: State.BIDDING,
        Event.QUEUE_HAS_WORK: State.PLANNING,
        Event.BATTERY_LOW: State.CHARGING,
        Event.FAULT_DETECTED: State.FAULT,
    },
    State.BIDDING: {
        Event.AUCTION_WON: State.PLANNING,
        Event.AUCTION_LOST: State.IDLE,
        Event.FAULT_DETECTED: State.FAULT,
    },
    State.PLANNING: {
        Event.ROUTE_READY: State.MOVING,
        # FR-3.7 / FR-6.2: an unreachable task returns to the mesh and the robot
        # returns to IDLE. It is not a fault -- the map changed, not the robot.
        Event.ROUTE_UNREACHABLE: State.IDLE,
        Event.FAULT_DETECTED: State.FAULT,
    },
    State.MOVING: {
        Event.CONFLICT_LOST: State.YIELD,
        Event.EDGE_BLOCKED: State.REPLAN,
        Event.LEG_COMPLETE: State.PLANNING,
        Event.ARRIVED_AT_DROP: State.AT_DROP,
        Event.FAULT_DETECTED: State.FAULT,
    },
    State.YIELD: {
        Event.CONFLICT_CLEARED: State.MOVING,
        Event.EDGE_BLOCKED: State.REPLAN,
        Event.FAULT_DETECTED: State.FAULT,
    },
    State.REPLAN: {
        Event.ROUTE_REPAIRED: State.MOVING,
        Event.ROUTE_UNREACHABLE: State.IDLE,
        Event.FAULT_DETECTED: State.FAULT,
    },
    State.AT_DROP: {
        Event.TASK_REPORTED: State.IDLE,
        Event.FAULT_DETECTED: State.FAULT,
    },
    State.CHARGING: {
        Event.CHARGED: State.IDLE,
        Event.FAULT_DETECTED: State.FAULT,
    },
    State.FAULT: {
        Event.RECOVERED: State.IDLE,
    },
}

BROADCASTING: dict[State, tuple[str, ...]] = {
    State.IDLE: ("INTENT",),
    State.BIDDING: ("BID",),
    State.PLANNING: ("CLAIM", "INTENT"),
    State.MOVING: ("INTENT", "RESERVE"),
    State.YIELD: ("INTENT",),
    State.REPLAN: ("INTENT", "BLOCKAGE"),
    State.AT_DROP: ("COMPLETE", "DIGEST"),
    State.CHARGING: ("INTENT",),
    State.FAULT: ("INTENT",),
}
"""Appendix A's broadcasting column. Used by tests to check a robot is not silent
in a state where the SRS requires it to be heard -- a robot that stops sending
INTENT is indistinguishable from a dead one after PEER_TIMEOUT_MS (FR-1.5)."""

MOVEMENT_STATES = frozenset({State.MOVING, State.YIELD})
"""States in which the robot is physically in motion. YIELD is included because
FR-5.6 requires yielding by shedding speed, not by stopping -- a robot in YIELD
is still moving, just more slowly."""

ACTIVE_TASK_STATES = frozenset(
    {State.PLANNING, State.MOVING, State.YIELD, State.REPLAN, State.AT_DROP}
)
"""States in which the robot holds work. FR-6.5 re-announces the task of a peer
declared lost while in one of these."""


class IllegalTransition(RuntimeError):
    """An event arrived in a state that does not accept it."""


@dataclass(frozen=True, slots=True)
class Transition:
    """One recorded state change, for reconstruction and the dashboard."""

    at_ms: int
    from_state: State
    to_state: State
    event: Event

    def __str__(self) -> str:
        return (
            f"{self.at_ms:>8} ms  {self.from_state.value:>8} "
            f"--{self.event.value}--> {self.to_state.value}"
        )


@dataclass
class StateMachine:
    """Enforces Appendix A for one robot.

    History is retained because NFR-4.4 requires every safety decision to be
    reconstructable from the logged inputs that produced it, and because the
    dashboard's per-robot decision panel renders it directly (section 8 of the
    implementation guide: three separate brains, not one shared status).
    """

    state: State = State.IDLE
    history: list[Transition] = field(default_factory=list)
    history_limit: int = 256
    """Bounded so a long scale100 run cannot grow history without limit. The
    dashboard shows only the last few entries, and the benchmark reads its
    metrics from the event log rather than from here."""

    transition_count: int = 0
    """Total transitions, including those dropped from bounded history."""

    def can(self, event: Event) -> bool:
        return event in TRANSITIONS[self.state]

    def target(self, event: Event) -> State | None:
        return TRANSITIONS[self.state].get(event)

    def fire(self, event: Event, now_ms: int) -> State:
        """Apply ``event``. Raises ``IllegalTransition`` if it does not apply.

        Raising rather than ignoring is deliberate. An ignored event means the
        robot's state silently stops describing what it is doing, and the first
        visible symptom is a lost task or a missing reservation far downstream.
        """
        destination = TRANSITIONS[self.state].get(event)
        if destination is None:
            allowed = ", ".join(sorted(e.value for e in TRANSITIONS[self.state]))
            raise IllegalTransition(
                f"{event.value} is not accepted in {self.state.value}; "
                f"accepted here: {allowed or '(none)'}"
            )
        record = Transition(now_ms, self.state, destination, event)
        self.state = destination
        self.transition_count += 1
        self.history.append(record)
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]
        return self.state

    def fire_if_possible(self, event: Event, now_ms: int) -> bool:
        """Apply ``event`` only where legal, reporting whether it applied.

        For events that are genuinely advisory -- a BATTERY_LOW that arrives
        mid-task should be remembered and acted on at IDLE (BR-4: an AMR below
        reserve finishes work already held), not forced through immediately.
        """
        if not self.can(event):
            return False
        self.fire(event, now_ms)
        return True

    def reset(self) -> None:
        self.state = State.IDLE
        self.history.clear()
        self.transition_count = 0

    # -- derived properties --------------------------------------------------

    @property
    def is_moving(self) -> bool:
        return self.state in MOVEMENT_STATES

    @property
    def holds_task(self) -> bool:
        return self.state in ACTIVE_TASK_STATES

    @property
    def broadcasts(self) -> tuple[str, ...]:
        return BROADCASTING[self.state]

    @property
    def last_transition(self) -> Transition | None:
        return self.history[-1] if self.history else None

    def recent(self, count: int = 5) -> list[Transition]:
        return self.history[-count:]


def validate_table() -> None:
    """Check the transition table is well formed. Called by tests.

    Three properties matter: every state is reachable, every state can be left
    (a state with no exit is a deadlock in the controller itself), and FAULT is
    reachable from everywhere that could plausibly fail.
    """
    missing = set(State) - set(TRANSITIONS)
    if missing:
        raise AssertionError(f"states with no transition entry: {sorted(missing)}")

    dead_ends = [s.value for s, moves in TRANSITIONS.items() if not moves]
    if dead_ends:
        raise AssertionError(f"states that cannot be left: {dead_ends}")

    reachable = {State.IDLE}
    frontier = [State.IDLE]
    while frontier:
        for destination in TRANSITIONS[frontier.pop()].values():
            if destination not in reachable:
                reachable.add(destination)
                frontier.append(destination)
    unreachable = set(State) - reachable
    if unreachable:
        raise AssertionError(f"states unreachable from IDLE: {sorted(unreachable)}")

    # FAULT must be reachable from every state that can hold work or move;
    # Appendix A enters it on lost position confidence, hardware fault or an
    # unrecoverable route, none of which respect what the robot was doing.
    for state in ACTIVE_TASK_STATES | {State.IDLE, State.CHARGING}:
        if TRANSITIONS[state].get(Event.FAULT_DETECTED) is not State.FAULT:
            raise AssertionError(f"{state.value} cannot reach FAULT")

    missing_broadcast = set(State) - set(BROADCASTING)
    if missing_broadcast:
        raise AssertionError(
            f"states with no broadcasting entry: {sorted(missing_broadcast)}"
        )
