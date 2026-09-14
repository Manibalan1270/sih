"""Deterministic arbitration (FE-5, Appendix B and C): who stands when plans collide.

This is the safety core. Everything else in the project can be wrong and produce a
slow fleet; if this is wrong, robots collide.

Under route-level reservation (core/timewindows.py) a robot never negotiates one
junction at a time. It books its whole route against the routes it has heard, and
executes by precedence -- entering a resource only after every robot booked ahead
of it there has left. Two plans therefore conflict only through a race (both
committed before either heard the other) or a loss (a PATH frame never arrived).
That leaves arbitration one job: given two plans that cannot both stand, say which
does. The answer must be the same on both robots, from integers both hold:

1. The plan committed **earlier on the fleet clock** stands. Every PATH carries its
   ``committed_ms``; the later committer replans against a table that now includes
   the earlier plan, so the outcome converges in one exchange.
2. On the same millisecond, Appendix C's ranking key ``k(r) = (priority, -robot_id)``
   decides. It is a *total order* -- priority is an integer and robot_id is unique --
   so a finite non-empty set under it has a unique maximum: exactly one robot in any
   conflict set does not yield, and a cyclic wait cannot form.

Both inputs are carried in every PATH and INTENT, and nothing local, learned or
randomized participates. That is why this module imports neither
``core.traffic_model`` nor ``random``, and why ``tests/test_architecture.py``
enforces it with an AST check: two robots with divergent learned models could each
conclude they held right of way, both proceed, and collide -- while every
behavioural test still passed (CON-7, FR-2.9, FR-5.9).

All arithmetic is integer milliseconds (CON-6, FR-3.8). There are no float literals
in this file, and that too is enforced by test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.timewindows import Resource, RoutePlan


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


def plan_precedence(mine: RoutePlan, theirs: RoutePlan) -> bool:
    """Whether ``mine`` stands and ``theirs`` must replan.

    Earlier commit first; a tie on the fleet clock falls to the total order. The
    result is antisymmetric by construction -- swap the arguments and it flips --
    which is what lets both robots compute it independently and agree.
    """
    if mine.committed_ms != theirs.committed_ms:
        return mine.committed_ms < theirs.committed_ms
    return outranks(mine.priority, mine.robot_id, theirs.priority, theirs.robot_id)


def unique_winner(contenders: list[tuple[int, int]]) -> tuple[int, int] | None:
    """The single winner of a conflict set of ``(priority, robot_id)`` pairs.

    Exists so Appendix C's claim can be tested directly rather than only through
    behaviour: a finite non-empty set under a total order has exactly one maximum,
    therefore exactly one robot proceeds and a cyclic wait cannot form.
    """
    if not contenders:
        return None
    return max(contenders, key=lambda pair: ranking_key(pair[0], pair[1]))


@dataclass(frozen=True, slots=True)
class Decision:
    """One resolved race, with what is needed to explain it."""

    now_ms: int
    resource: Resource
    other_robot: int
    other_plan_seq: int
    stood: bool
    """True if this robot's plan stood; False if it had to replan."""

    def __str__(self) -> str:
        verdict = "stood" if self.stood else "replanned"
        return f"t={self.now_ms} {self.resource} vs r{self.other_robot}#{self.other_plan_seq}: {verdict}"


@dataclass
class Arbiter:
    """Resolves races for one robot and keeps the record of them.

    Present on every Configuration B robot; absent (None) on Configuration A, which
    is how the baseline runs the same robot object with no coordination (FR-10.5).
    """

    robot_id: int
    decisions: list[Decision] = field(default_factory=list)
    stood: int = 0
    replanned: int = 0

    def resolve(
        self, mine: RoutePlan, theirs: RoutePlan, resource: Resource, now_ms: int
    ) -> bool:
        """Whether this robot's plan stands against ``theirs`` on ``resource``."""
        stands = plan_precedence(mine, theirs)
        if stands:
            self.stood += 1
        else:
            self.replanned += 1
        self.decisions.append(
            Decision(now_ms, resource, theirs.robot_id, theirs.plan_seq, stands)
        )
        if len(self.decisions) > 64:
            del self.decisions[:-64]
        return stands

    @property
    def last(self) -> Decision | None:
        return self.decisions[-1] if self.decisions else None

    def summary(self) -> str:
        return f"r{self.robot_id}: stood {self.stood}, replanned {self.replanned}"
