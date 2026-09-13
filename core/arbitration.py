"""Deterministic junction arbitration (FE-5, Appendix B).

This is the safety core. Everything else in the project can be wrong and produce a
slow fleet; if this is wrong, robots collide.

Appendix B gives the procedure as DETECT, PRIORITIZE, RESERVE, AVOID, and this file
follows it step for step so a reviewer can check one against the other. Appendix C
then proves deadlock freedom from two facts about this code, both of which are
tested rather than asserted:

1. The ranking key ``k(r) = (priority, -robot_id)`` is a *total order* -- no ties,
   no incomparable pairs -- because priority is an integer and robot_id is unique.
   A finite non-empty set under a total order has a unique maximum, so exactly one
   robot in any conflict set does not yield.
2. Every robot computes the *same* ranking, because the only inputs are priority
   and robot_id, both carried in every peer's INTENT, and because nothing local,
   learned or randomized participates.

Point 2 is why this module imports neither ``core.traffic_model`` nor ``random``,
and why ``tests/test_architecture.py`` enforces that with an AST check. The failure
it prevents is specific and silent: two robots with divergent learned cost models
could each conclude they hold right of way, both proceed, and collide -- while every
behavioural test still passed. CON-7 and FR-5.9 exist for that reason, and a comment
saying "do not import the traffic model here" would decay where a test will not.

All arithmetic is integer milliseconds (CON-6, FR-3.8). There are no float literals
in this file, and that too is enforced by test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core import config
from core.reservation import Reservation, ReservationTable, Window


class Outcome(str, Enum):
    """What arbitration decided."""

    PROCEED = "PROCEED"
    """No conflict. Enter on schedule, no RESERVE needed."""

    RESERVE = "RESERVE"
    """Won the conflict. Broadcast RESERVE, then enter on schedule (FR-5.5)."""

    YIELD_SLOW = "YIELD_SLOW"
    """Lost, and shedding speed can push our ETA past the conflict (FR-5.6).
    The preferred yield: NFR-2.4 wants conflicts resolved by anticipation."""

    YIELD_REPLAN = "YIELD_REPLAN"
    """Lost, slowing cannot absorb the shift, but another route exists. Mark the
    edge costly and replan (Appendix B's second AVOID branch)."""

    YIELD_HOLD = "YIELD_HOLD"
    """Lost, slowing cannot absorb the shift, and no alternative route exists.

    Not in Appendix B, and necessary. Appendix B's AVOID offers exactly two
    outcomes -- shed speed, or replan -- but in a single-lane corridor with no
    bypass both can be unavailable, and the procedure as written has nothing left
    to do. Something must: FR-5.3's separation is a Critical requirement and
    NFR-2.1 permits zero collisions.

    A controlled stop short of the junction is the safe resolution, and it is not
    the emergency braking FR-5.6 and NFR-2.4 forbid -- those rule out hard braking
    as the *normal* means of resolution, and this is reached only when anticipation
    has already been tried and found insufficient. It costs throughput, which is
    the correct thing to spend."""

    BLOCKED_LOW_CONFIDENCE = "BLOCKED_LOW_CONFIDENCE"
    """FR-5.14 / NFR-2.3: position confidence is below threshold, so no reservation
    may be claimed until a marker read restores it. An AMR that cannot say where it
    is cannot promise when it will be somewhere."""

    @property
    def is_yield(self) -> bool:
        return self in (
            Outcome.YIELD_SLOW,
            Outcome.YIELD_REPLAN,
            Outcome.YIELD_HOLD,
            Outcome.BLOCKED_LOW_CONFIDENCE,
        )

    @property
    def may_enter(self) -> bool:
        return self in (Outcome.PROCEED, Outcome.RESERVE)


def ranking_key(priority: int, robot_id: int) -> tuple[int, int]:
    """Appendix C's k(r) = (priority, -robot_id).

    Higher priority wins (BR-1); on equal priority the lower robot_id wins. Negating
    the id turns those two rules into one maximum, so no call site can apply the
    first and forget the second.
    """
    return (priority, -robot_id)


def outranks(
    my_priority: int, my_robot_id: int, other_priority: int, other_robot_id: int
) -> bool:
    """Whether the first robot ranks above the second. Never equal.

    Totality matters as much as ordering here: robot_id is unique across the fleet
    by construction, so no two distinct robots share a key. Appendix C's Lemma 1.
    """
    return ranking_key(my_priority, my_robot_id) > ranking_key(
        other_priority, other_robot_id
    )


@dataclass(frozen=True, slots=True)
class Decision:
    """One arbitration result, with everything needed to explain it.

    The detail is not decoration. NFR-2.5 requires every safety-path decision to be
    reproducible from logged inputs, and NFR-4.4 to be reconstructable. The
    dashboard also renders this per robot, so a jury can see two robots reaching
    opposite conclusions from the same table (section 8 of the implementation
    guide).
    """

    outcome: Outcome
    junction: int
    window: Window
    conflicts: tuple[Reservation, ...] = ()
    rival: Reservation | None = None
    """The highest-ranked contender. None when unopposed."""

    shift_ms: int = 0
    """How much later our arrival must be to clear the conflict (Appendix B's
    ``shift``). Zero when proceeding."""

    reason: str = ""

    @property
    def yielded(self) -> bool:
        return self.outcome.is_yield

    def __str__(self) -> str:
        rival = f" to {self.rival}" if self.rival is not None else ""
        shift = f", shift {self.shift_ms} ms" if self.shift_ms else ""
        return (
            f"J{self.junction} {self.window} -> {self.outcome.value}{rival}{shift}"
            f"{': ' + self.reason if self.reason else ''}"
        )


@dataclass
class Arbiter:
    """Runs Appendix B for one robot.

    Stateless with respect to decisions -- it holds counters only -- so the same
    table and window always produce the same answer (FR-5.8).
    """

    robot_id: int
    decisions: int = 0
    reserves_won: int = 0
    yields: int = 0
    holds: int = 0
    history: list[Decision] = field(default_factory=list)
    history_limit: int = 32

    def arbitrate(
        self,
        *,
        junction: int,
        window: Window,
        my_priority: int,
        table: ReservationTable,
        can_absorb_shift,
        has_alternative_route: bool = False,
        position_confident: bool = True,
        from_node: int = -1,
        to_node: int = -1,
    ) -> Decision:
        """Decide whether to enter ``junction`` during ``window``.

        ``can_absorb_shift`` is a predicate taking a shift in milliseconds and
        answering whether slowing down can delay arrival by that much *before the
        last passing point*. It is injected rather than computed here because the
        answer depends on geometry -- distance remaining, speed floor, whether an
        alternative edge is still reachable -- which belongs to the robot, while the
        arbitration rule belongs here.
        """
        self.decisions += 1

        # ---- 1. DETECT -----------------------------------------------------
        conflicts = table.conflicts(
            junction,
            window,
            exclude_robot=self.robot_id,
            from_node=from_node,
            to_node=to_node,
        )
        if not conflicts:
            return self._record(
                Decision(
                    outcome=Outcome.PROCEED,
                    junction=junction,
                    window=window,
                    reason="no conflicting claim within the safety margin",
                )
            )

        # FR-5.14 / NFR-2.3. Checked after DETECT so an unopposed junction can
        # still be crossed: the requirement forbids *claiming a reservation*
        # without confidence, not moving at all. Checked before PRIORITIZE because
        # winning a contest with an unreliable ETA is worse than losing one.
        if not position_confident:
            rival = max(conflicts, key=Reservation.ranking_key)
            return self._record(
                Decision(
                    outcome=Outcome.BLOCKED_LOW_CONFIDENCE,
                    junction=junction,
                    window=window,
                    conflicts=conflicts,
                    rival=rival,
                    reason=(
                        "position confidence below threshold; not claiming a "
                        "contested junction (FR-5.14)"
                    ),
                )
            )

        # ---- 2. PRIORITIZE -------------------------------------------------
        rival = max(conflicts, key=Reservation.ranking_key)
        if outranks(my_priority, self.robot_id, rival.priority, rival.robot_id):
            # Appendix C's Theorem: the unique maximum of the conflict set is the
            # one robot that does not yield. Every contender computes this same
            # comparison from the same broadcast facts and agrees.
            self.reserves_won += 1
            return self._record(
                Decision(
                    outcome=Outcome.RESERVE,
                    junction=junction,
                    window=window,
                    conflicts=conflicts,
                    rival=rival,
                    reason=(
                        f"outrank every contender: "
                        f"(p{my_priority}, r{self.robot_id}) beats "
                        f"(p{rival.priority}, r{rival.robot_id})"
                    ),
                )
            )

        # ---- 3. AVOID ------------------------------------------------------
        # Taken from the conflict set already computed, not re-queried: re-querying
        # would drop the direction filter and reintroduce following traffic.
        latest_end = max(held.window.end_ms for held in conflicts)
        shift_ms = latest_end + table.margin_ms - window.start_ms
        if shift_ms <= 0:
            # Can happen when a conflict straddles the margin on the early side:
            # the rival is already past, and separation is satisfied by waiting no
            # time at all. Treat as clear rather than manufacturing a yield.
            return self._record(
                Decision(
                    outcome=Outcome.PROCEED,
                    junction=junction,
                    window=window,
                    conflicts=conflicts,
                    rival=rival,
                    reason="conflicting window already past the margin",
                )
            )

        self.yields += 1
        if can_absorb_shift(shift_ms):
            return self._record(
                Decision(
                    outcome=Outcome.YIELD_SLOW,
                    junction=junction,
                    window=window,
                    conflicts=conflicts,
                    rival=rival,
                    shift_ms=shift_ms,
                    reason=(
                        f"yield to (p{rival.priority}, r{rival.robot_id}); "
                        f"shedding speed absorbs {shift_ms} ms"
                    ),
                )
            )

        if has_alternative_route:
            return self._record(
                Decision(
                    outcome=Outcome.YIELD_REPLAN,
                    junction=junction,
                    window=window,
                    conflicts=conflicts,
                    rival=rival,
                    shift_ms=shift_ms,
                    reason=(
                        f"cannot absorb {shift_ms} ms before the last passing "
                        f"point; taking an alternative route"
                    ),
                )
            )

        self.holds += 1
        return self._record(
            Decision(
                outcome=Outcome.YIELD_HOLD,
                junction=junction,
                window=window,
                conflicts=conflicts,
                rival=rival,
                shift_ms=shift_ms,
                reason=(
                    f"cannot absorb {shift_ms} ms and no alternative exists; "
                    f"holding short of the junction"
                ),
            )
        )

    def _record(self, decision: Decision) -> Decision:
        self.history.append(decision)
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]
        return decision

    @property
    def last(self) -> Decision | None:
        return self.history[-1] if self.history else None

    def summary(self) -> str:
        return (
            f"r{self.robot_id}: {self.decisions} decisions, "
            f"{self.reserves_won} reserved, {self.yields} yielded "
            f"({self.holds} of them held)"
        )


def crossing_window(
    *,
    arrival_ms: int,
    occupancy_ms: int = config.JUNCTION_OCCUPANCY_MS,
) -> Window:
    """The window a robot needs at a junction it will reach at ``arrival_ms``.

    Occupancy is a fixed footprint-crossing time rather than a computed one. ASM-6
    makes the fleet homogeneous in speed and size, so a robot crossing a junction
    always takes the same time, and computing it per robot would imply a
    heterogeneity the specification does not permit (and OI-4 records as out of
    scope).
    """
    return Window(arrival_ms, arrival_ms + occupancy_ms)


def unique_winner(contenders: list[tuple[int, int]]) -> tuple[int, int] | None:
    """The single winner of a conflict set of ``(priority, robot_id)`` pairs.

    Exists so Appendix C's claim can be tested directly rather than only through
    behaviour: a finite non-empty set under a total order has exactly one maximum,
    therefore exactly one robot proceeds and a cyclic wait cannot form.
    """
    if not contenders:
        return None
    return max(contenders, key=lambda pair: ranking_key(pair[0], pair[1]))
