"""Junction reservations (FE-5, FR-5.1, FR-5.3, FR-5.11).

Conflicts are resolved in *time*, not in space. Two robots do not conflict because
they both use a junction; they conflict because they would use it within the same
time window. So this module is a table of claimed windows per junction, built
entirely from what peers broadcast.

Three properties are load-bearing:

**It is local.** Every robot holds its own table, populated only from frames it
received. Two robots' tables can legitimately differ -- one may have missed an
INTENT -- and the safety argument survives that, because a robot that has not heard
of a claim behaves *more* conservatively, not less: it will still detect the
conflict on the next INTENT 200 ms later, and the safety margin is sized to cover
exactly that gap.

**It holds no learned state.** CON-7, FR-2.9 and FR-5.9 keep the traffic model, and
anything random, out of the arbitration path. This module therefore imports neither,
and an AST test enforces it. The reason is not tidiness: two robots with divergent
learned models could otherwise both conclude they hold right of way, and the
Appendix C proof would stop applying.

**Reservations expire by themselves.** FR-5.11 forbids requiring a release message.
That is what makes a robot that loses power harmless -- its claims lapse on their
own, so it cannot hold a junction hostage from beyond the grave. The Appendix C
proof depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core import config

IMPLIED = "INTENT"
"""A window inferred from a peer's declared intent. Soft: the peer has said it
means to be there, but has not claimed it."""

EXPLICIT = "RESERVE"
"""A window a peer has explicitly claimed (FR-5.5). Takes precedence over an
implied window from the same robot for the same junction."""


@dataclass(frozen=True, slots=True)
class Window:
    """A closed time interval on the aligned fleet clock, in milliseconds."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"window [{self.start_ms}, {self.end_ms}] ends before it starts"
            )

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def overlaps(self, other: "Window", margin_ms: int) -> bool:
        """FR-5.3: two windows conflict if separated by less than the margin.

        Note this is *separation*, not intersection. Windows 100 ms apart do not
        intersect but do conflict, because 100 ms is not enough clearance for two
        robots whose position estimates carry error. Expanding one side by the
        margin and testing intersection is the same test, written so the margin
        cannot be forgotten.
        """
        return (
            self.start_ms - margin_ms <= other.end_ms
            and other.start_ms - margin_ms <= self.end_ms
        )

    def shifted(self, delta_ms: int) -> "Window":
        return Window(self.start_ms + delta_ms, self.end_ms + delta_ms)

    def __str__(self) -> str:
        return f"[{self.start_ms},{self.end_ms}]"


@dataclass(frozen=True, slots=True)
class Reservation:
    """One robot's claim on one junction for one bounded window."""

    junction: int
    window: Window
    robot_id: int
    priority: int
    """Composite priority the sender precomputed (section 3.4.1). The first key of
    the arbitration total order (FR-5.4, BR-1)."""

    kind: str = IMPLIED
    received_at_ms: int = 0

    from_node: int = -1
    to_node: int = -1
    """How the robot passes through: approach node and exit node.

    Needed to tell a *crossing* conflict from *following* traffic. Two robots
    entering a junction from the same side and leaving by the same side are not in
    conflict at all -- one is simply behind the other, and safe separation there is a
    matter of headway (IF-2.4), not of right of way.

    Arbitrating them anyway is actively harmful. Observed directly: a leading robot
    yielded a junction to a higher-priority robot following it through, then stood in
    the follower's way, so the winner could not advance and the loser would not.
    Neither ever moved. Section 3.4.1 carries the whole node sequence, so direction is
    available without adding a wire field."""

    @property
    def is_explicit(self) -> bool:
        return self.kind == EXPLICIT

    def ranking_key(self) -> tuple[int, int]:
        """Appendix C's k(r) = (priority, -robot_id), ordered lexicographically.

        Negating the id makes "higher priority wins, and on equal priority the
        *lower* id wins" a single maximum rather than two rules, so the tie-break
        cannot be forgotten at a call site.
        """
        return (self.priority, -self.robot_id)

    def __str__(self) -> str:
        return (
            f"r{self.robot_id} p{self.priority} holds J{self.junction} "
            f"{self.window} ({self.kind})"
        )


@dataclass
class ReservationTable:
    """One robot's view of who holds which junction, and when.

    Keyed by junction because that is how it is queried: the only question ever
    asked is "who else wants this junction around the time I do?".
    """

    margin_ms: int = config.MARGIN_MS
    """FR-5.3. Widened to MARGIN_DEGRADED_MS when the clock beacon goes stale
    (FR-5.13, FR-7.6) -- reduced throughput, never reduced safety."""

    entries: dict[int, dict[int, Reservation]] = field(default_factory=dict)
    """junction -> robot_id -> its current claim. One claim per robot per junction:
    a fresh INTENT supersedes the robot's previous declaration rather than adding
    to it, or a robot that changed its mind would appear to want the junction
    twice and block itself out."""

    corridors: dict[int, dict[int, Reservation]] = field(default_factory=dict)
    """edge_id -> robot_id -> claim on a single-lane corridor (FR-5.10, ASM-5).

    A separate namespace from ``entries`` because junction and edge identifiers
    overlap numerically -- junction 4 and edge 4 are different things, and sharing
    one dict would have a robot claiming a junction block a corridor.

    Corridors need their own arbitration because junction arbitration cannot
    prevent a head-on. Two robots entering a single-lane corridor from opposite
    ends want *different* junctions -- each wants the far end -- so they never
    contend for the same one and both proceed. Measured on the benchmark map with
    junction arbitration working and this missing: zero junction collisions, and
    every remaining collision a head-on inside the choke corridor."""

    def __len__(self) -> int:
        return sum(len(h) for h in self.entries.values()) + sum(
            len(h) for h in self.corridors.values()
        )

    # -- recording -----------------------------------------------------------

    def record(self, reservation: Reservation) -> None:
        """Enter a claim, superseding this robot's previous one for the junction.

        An explicit RESERVE is never overwritten by an implied INTENT window from
        the same robot: the robot has committed, and downgrading that back to a
        mere intention would discard a stronger statement in favour of a weaker
        one arriving later.
        """
        holders = self.entries.setdefault(reservation.junction, {})
        existing = holders.get(reservation.robot_id)
        if (
            existing is not None
            and existing.is_explicit
            and not reservation.is_explicit
            and reservation.received_at_ms < existing.window.end_ms
        ):
            return

        # RESERVE asserts *timing*; INTENT supplies *geometry*. Section 3.4.2 gives
        # RESERVE no approach or exit field, so an explicit claim arriving after an
        # implied one would otherwise discard the direction and leave the entry
        # looking like a crossing conflict. It then outranked a robot it was queued
        # behind, which could not pass it -- a permanent standoff on the approach to
        # the choke corridor. Carrying the known direction forward costs nothing and
        # needs no wire change.
        if reservation.from_node < 0 and existing is not None and existing.from_node >= 0:
            reservation = Reservation(
                junction=reservation.junction,
                window=reservation.window,
                robot_id=reservation.robot_id,
                priority=reservation.priority,
                kind=reservation.kind,
                received_at_ms=reservation.received_at_ms,
                from_node=existing.from_node,
                to_node=existing.to_node,
            )
        holders[reservation.robot_id] = reservation

    def record_intent(
        self,
        *,
        robot_id: int,
        priority: int,
        junctions: tuple[int, ...],
        eta_ms: tuple[int, ...],
        sent_at_ms: int,
        now_ms: int,
        occupancy_ms: int = config.JUNCTION_OCCUPANCY_MS,
        is_junction=None,
        current_node: int | None = None,
        corridor_of=None,
    ) -> list[Reservation]:
        """Build implied windows from a peer's INTENT (FR-5.1, FR-1.8).

        FR-1.8 requires a single INTENT to update both the reservation table and the
        traffic model with no second message. This is that half of it.

        ETAs arrive relative to the frame's own timestamp, so they are rebased onto
        the aligned clock here. ``now_ms`` is accepted rather than assumed equal to
        ``sent_at_ms`` because a frame takes time to arrive, and treating a 200 ms
        old ETA as current would place the peer later than it really is -- the
        dangerous direction to err.
        """
        recorded: list[Reservation] = []
        path = (current_node, *junctions) if current_node is not None else junctions
        base = 1 if current_node is not None else 0
        for index, (junction, eta) in enumerate(zip(junctions, eta_ms)):
            if is_junction is not None and not is_junction(junction):
                continue  # a non-junction node needs no arbitration
            arrival = sent_at_ms + eta
            position = index + base
            approach = path[position - 1] if position >= 1 else -1
            exit_node = path[position + 1] if position + 1 < len(path) else -1
            reservation = Reservation(
                junction=junction,
                window=Window(arrival, arrival + occupancy_ms),
                robot_id=robot_id,
                priority=priority,
                kind=IMPLIED,
                received_at_ms=now_ms,
                from_node=approach,
                to_node=exit_node,
            )
            self.record(reservation)
            recorded.append(reservation)

        if corridor_of is not None and current_node is not None:
            recorded += self._record_corridors_from_path(
                robot_id=robot_id,
                priority=priority,
                path=(current_node, *junctions),
                eta_ms=eta_ms,
                sent_at_ms=sent_at_ms,
                now_ms=now_ms,
                occupancy_ms=occupancy_ms,
                corridor_of=corridor_of,
            )
        return recorded

    def _record_corridors_from_path(
        self,
        *,
        robot_id: int,
        priority: int,
        path: tuple[int, ...],
        eta_ms: tuple[int, ...],
        sent_at_ms: int,
        now_ms: int,
        occupancy_ms: int,
        corridor_of,
    ) -> list[Reservation]:
        """Derive single-lane corridor claims from a peer's declared node sequence.

        No new wire field is needed, and that is deliberate. A corridor is implied by
        two consecutive nodes the peer says it will cross, so INTENT already carries
        it -- which means the claim inherits INTENT's 200 ms repetition and is
        self-healing under packet loss, exactly like every other guarantee here. An
        explicit corridor RESERVE would be one-shot and IF-4.5 forbids requiring
        retransmission.

        The window spans from entering the corridor to clearing it: the ETA of the
        node *before* it through the ETA of the node after, plus occupancy.
        """
        recorded: list[Reservation] = []
        for index in range(len(path) - 1):
            edge_id = corridor_of(path[index], path[index + 1])
            if edge_id is None:
                continue  # not a single-lane corridor, or no such edge
            # ETA of the entry node: index 0 is the peer's current node, which it is
            # at or leaving now, so its arrival is the frame's own timestamp.
            entry_eta = 0 if index == 0 else eta_ms[index - 1]
            exit_eta = eta_ms[index] if index < len(eta_ms) else entry_eta
            reservation = Reservation(
                junction=edge_id,
                window=Window(sent_at_ms + entry_eta, sent_at_ms + exit_eta + occupancy_ms),
                robot_id=robot_id,
                priority=priority,
                kind=IMPLIED,
                received_at_ms=now_ms,
                from_node=path[index],
                to_node=path[index + 1],
            )
            self.record_corridor(reservation)
            recorded.append(reservation)
        return recorded

    def record_corridor(self, reservation: Reservation) -> None:
        """Enter a claim on a single-lane corridor. ``junction`` holds the edge id."""
        holders = self.corridors.setdefault(reservation.junction, {})
        holders[reservation.robot_id] = reservation

    def corridor_conflicts(
        self,
        edge_id: int,
        window: Window,
        *,
        exclude_robot: int,
        from_node: int = -1,
    ) -> tuple[Reservation, ...]:
        """Other robots claiming this corridor in an overlapping window.

        The case that must be prevented is the head-on: two robots entering from
        opposite ends, where neither can pass and neither can turn round.

        Robots entering from the *same* end are not that. They queue through the
        corridor one behind the other, and their separation is headway (IF-2.4). This
        filter matters more than it looks: without it a robot already at the corridor
        mouth yields to a higher-priority robot queued behind it, then blocks the very
        approach that robot needs. Observed as a permanent standoff at the choke
        corridor with both robots in YIELD.
        """
        holders = self.corridors.get(edge_id)
        if not holders:
            return ()
        return tuple(
            holders[robot_id]
            for robot_id in sorted(holders)
            if robot_id != exclude_robot
            and holders[robot_id].window.overlaps(window, self.margin_ms)
            and not (
                from_node >= 0
                and holders[robot_id].from_node == from_node
            )
        )

    def corridor_holders(self, edge_id: int) -> tuple[Reservation, ...]:
        holders = self.corridors.get(edge_id)
        if not holders:
            return ()
        return tuple(holders[robot_id] for robot_id in sorted(holders))

    def record_reserve(
        self,
        *,
        robot_id: int,
        priority: int,
        junction: int,
        window_start_ms: int,
        window_end_ms: int,
        now_ms: int,
    ) -> Reservation:
        """Enter an explicit claim from a peer's RESERVE (FR-5.5)."""
        reservation = Reservation(
            junction=junction,
            window=Window(window_start_ms, window_end_ms),
            robot_id=robot_id,
            priority=priority,
            kind=EXPLICIT,
            received_at_ms=now_ms,
        )
        self.record(reservation)
        return reservation

    # -- querying ------------------------------------------------------------

    def holders(self, junction: int) -> tuple[Reservation, ...]:
        """Every current claim on a junction, in robot_id order.

        Sorted so two robots iterating the same table see the same sequence --
        FR-5.8 requires identical contents to produce identical decisions, and dict
        order would make that true only by accident.
        """
        holders = self.entries.get(junction)
        if not holders:
            return ()
        return tuple(holders[robot_id] for robot_id in sorted(holders))

    def conflicts(
        self,
        junction: int,
        window: Window,
        *,
        exclude_robot: int,
        from_node: int = -1,
        to_node: int = -1,
    ) -> tuple[Reservation, ...]:
        """Claims on ``junction`` overlapping ``window`` within the margin.

        Appendix B's DETECT step. ``exclude_robot`` is the asking robot: its own
        declared intent is in the table too, and a robot that conflicted with itself
        would yield forever.

        ``from_node`` and ``to_node`` describe how the asker passes through, so
        following traffic can be excluded -- see ``Reservation.from_node``. Omitting
        them falls back to treating every overlap as a conflict, which costs
        throughput and never safety.
        """
        return tuple(
            held
            for held in self.holders(junction)
            if held.robot_id != exclude_robot
            and held.window.overlaps(window, self.margin_ms)
            and not _is_following(held, from_node, to_node)
        )

    def is_free(self, junction: int, window: Window, *, exclude_robot: int) -> bool:
        return not self.conflicts(junction, window, exclude_robot=exclude_robot)

    def latest_conflict_end(
        self, junction: int, window: Window, *, exclude_robot: int
    ) -> int | None:
        """When the last conflicting window ends. Appendix B's AVOID input."""
        conflicting = self.conflicts(junction, window, exclude_robot=exclude_robot)
        if not conflicting:
            return None
        return max(held.window.end_ms for held in conflicting)

    # -- expiry --------------------------------------------------------------

    def expire(self, now_ms: int) -> int:
        """Drop windows that have passed. Returns how many went.

        FR-5.11: reservations expire automatically, with no release message. This is
        what makes a failed robot harmless rather than a permanent obstruction, and
        the Appendix C proof cites it directly.
        """
        removed = 0
        for table in (self.entries, self.corridors):
            for key in list(table):
                holders = table[key]
                for robot_id in list(holders):
                    if holders[robot_id].window.end_ms < now_ms:
                        del holders[robot_id]
                        removed += 1
                if not holders:
                    del table[key]
        return removed

    def drop_robot(self, robot_id: int) -> int:
        """Remove every claim by one robot (FR-6.5, peer declared lost).

        Separate from expiry because a lost peer's claims must go *now*: waiting for
        them to lapse would leave the fleet avoiding junctions nobody is coming to.
        """
        removed = 0
        for table in (self.entries, self.corridors):
            for key in list(table):
                if table[key].pop(robot_id, None) is not None:
                    removed += 1
                if not table[key]:
                    del table[key]
        return removed

    def clear(self) -> None:
        self.entries.clear()
        self.corridors.clear()

    def set_margin(self, margin_ms: int) -> None:
        """Widen or restore the safety margin (FR-5.13, FR-7.6)."""
        self.margin_ms = margin_ms

    # -- reporting -----------------------------------------------------------

    def summary(self) -> str:
        if not self.entries:
            return "reservation table empty"
        lines = [f"margin {self.margin_ms} ms, {len(self)} claims:"]
        for junction in sorted(self.entries):
            for held in self.holders(junction):
                lines.append(f"  {held}")
        return "\n".join(lines)


def _is_following(held: Reservation, from_node: int, to_node: int) -> bool:
    """Whether a claim is following traffic rather than a crossing conflict.

    The decisive fact is the *approach*: two robots entering a junction from the same
    node are in the same lane, one behind the other, and their separation is a matter
    of headway (IF-2.4) rather than right of way. Where they go afterwards does not
    change that -- they may diverge on the far side, and diverging traffic has already
    stopped contending.

    Requiring the exits to match as well was tried and deadlocks. INTENT declares
    three junctions (FR-1.2), so the *last* one has no known exit, and treating an
    unknown exit as a crossing conflict made a leading robot yield to a follower it
    was standing in front of. Neither moved. Matching on approach alone is both
    simpler and correct: an unknown exit is not evidence of a crossing.
    """
    del to_node  # the exit does not decide this; see above
    if from_node < 0 or held.from_node < 0:
        return False
    return held.from_node == from_node
