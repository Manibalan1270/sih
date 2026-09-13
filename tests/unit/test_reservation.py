"""Reservation table (FE-5: FR-5.1, FR-5.3, FR-5.11).

The table is the input to every arbitration decision, so a defect here produces a
wrong decision that looks perfectly reasonable. Two behaviours carry most of the
risk: the margin being *separation* rather than intersection, and the direction
bookkeeping that tells a crossing conflict from following traffic.
"""

from __future__ import annotations

import pytest

from core import config
from core.reservation import (
    EXPLICIT,
    IMPLIED,
    Reservation,
    ReservationTable,
    Window,
)


def claim(
    *,
    junction: int = 5,
    robot_id: int = 2,
    priority: int = 10,
    start: int = 10_000,
    end: int | None = None,
    kind: str = IMPLIED,
    from_node: int = 1,
    to_node: int = 9,
    received_at_ms: int = 0,
) -> Reservation:
    return Reservation(
        junction=junction,
        window=Window(start, end if end is not None else start + config.JUNCTION_OCCUPANCY_MS),
        robot_id=robot_id,
        priority=priority,
        kind=kind,
        from_node=from_node,
        to_node=to_node,
        received_at_ms=received_at_ms,
    )


class TestWindow:
    def test_a_backwards_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ends before it starts"):
            Window(9000, 1000)

    def test_overlapping_windows_conflict(self) -> None:
        assert Window(1000, 1700).overlaps(Window(1500, 2200), margin_ms=0)

    def test_the_margin_is_separation_not_intersection(self) -> None:
        """FR-5.3 conflicts windows *separated by less than* the margin. Two windows
        400 ms apart do not intersect but do conflict, because 400 ms is not enough
        clearance for robots whose position estimates carry error."""
        early, late = Window(0, 1000), Window(1400, 2400)
        assert not early.overlaps(late, margin_ms=0)
        assert early.overlaps(late, margin_ms=config.MARGIN_MS)

    def test_windows_beyond_the_margin_are_clear(self) -> None:
        early = Window(0, 1000)
        far = Window(1000 + config.MARGIN_MS + 1, 5000)
        assert not early.overlaps(far, margin_ms=config.MARGIN_MS)

    def test_overlap_is_symmetric(self) -> None:
        """Two robots must reach the same conclusion from the same pair of windows,
        or FR-5.8's determinism fails in the worst possible place."""
        a, b = Window(0, 1000), Window(900, 2000)
        for margin in (0, 400, config.MARGIN_MS, config.MARGIN_DEGRADED_MS):
            assert a.overlaps(b, margin) == b.overlaps(a, margin)

    def test_shifting_moves_both_ends(self) -> None:
        assert Window(1000, 2000).shifted(500) == Window(1500, 2500)


class TestRecording:
    def test_a_claim_is_stored_and_found(self) -> None:
        table = ReservationTable()
        table.record(claim())
        assert len(table) == 1
        assert table.holders(5)[0].robot_id == 2

    def test_a_fresh_claim_supersedes_the_same_robot_s_previous_one(self) -> None:
        """A robot that changed its mind must not appear to want the junction twice,
        or it would block itself out."""
        table = ReservationTable()
        table.record(claim(start=10_000))
        table.record(claim(start=20_000))
        assert len(table) == 1
        assert table.holders(5)[0].window.start_ms == 20_000

    def test_different_robots_are_stored_separately(self) -> None:
        table = ReservationTable()
        table.record(claim(robot_id=2))
        table.record(claim(robot_id=3))
        assert len(table.holders(5)) == 2

    def test_holders_are_returned_in_robot_id_order(self) -> None:
        """So two robots iterating the same table see the same sequence (FR-5.8)."""
        table = ReservationTable()
        for robot_id in (7, 2, 5):
            table.record(claim(robot_id=robot_id))
        assert [h.robot_id for h in table.holders(5)] == [2, 5, 7]

    def test_an_explicit_claim_is_not_downgraded_by_a_later_intent(self) -> None:
        """A RESERVE is a commitment; an INTENT window is an intention. Replacing the
        stronger statement with a weaker one arriving later loses information."""
        table = ReservationTable()
        table.record(claim(kind=EXPLICIT, start=10_000, end=12_000))
        table.record(claim(kind=IMPLIED, start=30_000, received_at_ms=11_000))
        held = table.holders(5)[0]
        assert held.is_explicit
        assert held.window.start_ms == 10_000

    def test_an_explicit_claim_inherits_direction_from_what_is_known(self) -> None:
        """Section 3.4.2 gives RESERVE no approach or exit field. Without carrying the
        direction forward, an explicit claim looks like a crossing conflict and a
        robot outranks one it is queued behind and cannot pass."""
        table = ReservationTable()
        table.record(claim(kind=IMPLIED, from_node=1, to_node=9))
        table.record_reserve(
            robot_id=2, priority=10, junction=5,
            window_start_ms=11_000, window_end_ms=11_700, now_ms=1000,
        )
        held = table.holders(5)[0]
        assert held.is_explicit
        assert (held.from_node, held.to_node) == (1, 9)


class TestConflictDetection:
    def test_an_overlapping_claim_is_a_conflict(self) -> None:
        table = ReservationTable()
        table.record(claim(robot_id=2, from_node=1, to_node=9))
        conflicts = table.conflicts(
            5, Window(10_200, 10_900), exclude_robot=3, from_node=4, to_node=8
        )
        assert [c.robot_id for c in conflicts] == [2]

    def test_a_robot_never_conflicts_with_itself(self) -> None:
        """Its own declared intent is in the table too, and a robot that conflicted
        with itself would yield forever."""
        table = ReservationTable()
        table.record(claim(robot_id=3))
        assert table.conflicts(5, Window(10_000, 10_700), exclude_robot=3) == ()

    def test_following_traffic_is_not_a_conflict(self) -> None:
        table = ReservationTable()
        table.record(claim(robot_id=2, from_node=1, to_node=9))
        assert table.conflicts(
            5, Window(10_000, 10_700), exclude_robot=3, from_node=1, to_node=9
        ) == ()

    def test_following_is_judged_on_approach_alone(self) -> None:
        """INTENT declares three junctions, so the last has no known exit. Requiring
        both ends to match treated an unknown exit as a crossing and deadlocked."""
        table = ReservationTable()
        table.record(claim(robot_id=2, from_node=1, to_node=-1))
        assert table.conflicts(
            5, Window(10_000, 10_700), exclude_robot=3, from_node=1, to_node=9
        ) == ()

    def test_an_unknown_approach_falls_back_to_treating_it_as_a_conflict(self) -> None:
        """Conservative in the safe direction: costs throughput, never safety."""
        table = ReservationTable()
        table.record(claim(robot_id=2, from_node=-1, to_node=-1))
        assert len(
            table.conflicts(5, Window(10_000, 10_700), exclude_robot=3, from_node=1)
        ) == 1

    def test_a_different_approach_is_a_crossing_conflict(self) -> None:
        table = ReservationTable()
        table.record(claim(robot_id=2, from_node=1, to_node=9))
        assert len(
            table.conflicts(
                5, Window(10_000, 10_700), exclude_robot=3, from_node=6, to_node=9
            )
        ) == 1

    def test_a_free_junction_reports_free(self) -> None:
        table = ReservationTable()
        assert table.is_free(5, Window(0, 700), exclude_robot=1)

    def test_the_latest_conflict_end_drives_the_avoid_shift(self) -> None:
        """Appendix B's AVOID computes the shift from the last conflicting window."""
        table = ReservationTable()
        table.record(claim(robot_id=2, start=10_000, end=10_700, from_node=1))
        table.record(claim(robot_id=3, start=10_400, end=12_000, from_node=1))
        latest = table.latest_conflict_end(
            5, Window(10_500, 11_200), exclude_robot=9, from_node=6
        )
        assert latest == 12_000

    def test_a_widened_margin_finds_more_conflicts(self) -> None:
        """FR-5.13 / FR-7.6: a stale clock costs throughput, never safety."""
        table = ReservationTable()
        table.record(claim(robot_id=2, start=0, end=700, from_node=1))
        probe = Window(700 + config.MARGIN_MS + 100, 900 + config.MARGIN_MS + 100)
        assert table.conflicts(5, probe, exclude_robot=3, from_node=6) == ()
        table.set_margin(config.MARGIN_DEGRADED_MS)
        assert len(table.conflicts(5, probe, exclude_robot=3, from_node=6)) == 1


class TestIntentDerivedWindows:
    def test_windows_are_rebased_onto_the_aligned_clock(self) -> None:
        """ETAs arrive relative to the frame's own timestamp."""
        table = ReservationTable()
        table.record_intent(
            robot_id=2, priority=10, junctions=(5, 6, 7),
            eta_ms=(1000, 2000, 3000), sent_at_ms=50_000, now_ms=50_020,
        )
        assert table.holders(5)[0].window.start_ms == 51_000
        assert table.holders(7)[0].window.start_ms == 53_000

    def test_non_junction_nodes_are_skipped(self) -> None:
        """Nothing crosses at a spur, so it needs no arbitration."""
        table = ReservationTable()
        table.record_intent(
            robot_id=2, priority=10, junctions=(5, 6), eta_ms=(1000, 2000),
            sent_at_ms=0, now_ms=0, is_junction=lambda n: n != 6,
        )
        assert table.holders(6) == ()
        assert len(table.holders(5)) == 1

    def test_direction_is_derived_from_the_node_sequence(self) -> None:
        """No wire field needed: section 3.4.1 already carries the whole sequence."""
        table = ReservationTable()
        table.record_intent(
            robot_id=2, priority=10, junctions=(5, 6, 7), eta_ms=(1000, 2000, 3000),
            sent_at_ms=0, now_ms=0, current_node=4,
        )
        assert (table.holders(5)[0].from_node, table.holders(5)[0].to_node) == (4, 6)
        assert (table.holders(6)[0].from_node, table.holders(6)[0].to_node) == (5, 7)

    def test_the_last_declared_junction_has_no_known_exit(self) -> None:
        """Which is exactly why following is judged on approach alone."""
        table = ReservationTable()
        table.record_intent(
            robot_id=2, priority=10, junctions=(5, 6, 7), eta_ms=(1, 2, 3),
            sent_at_ms=0, now_ms=0, current_node=4,
        )
        assert table.holders(7)[0].to_node == -1

    def test_single_lane_corridors_are_claimed_from_the_same_frame(self) -> None:
        """FR-5.10's protection rides on INTENT's 200 ms repetition, so it is
        self-healing under loss -- an explicit corridor RESERVE would be one-shot."""
        corridors = {(4, 5): 40}
        table = ReservationTable()
        table.record_intent(
            robot_id=2, priority=10, junctions=(5, 6), eta_ms=(1000, 2000),
            sent_at_ms=0, now_ms=0, current_node=4,
            corridor_of=lambda u, v: corridors.get((u, v)),
        )
        held = table.corridor_holders(40)
        assert len(held) == 1
        assert (held[0].from_node, held[0].to_node) == (4, 5)

    def test_a_corridor_window_spans_the_whole_traverse(self) -> None:
        corridors = {(4, 5): 40}
        table = ReservationTable()
        table.record_intent(
            robot_id=2, priority=10, junctions=(5,), eta_ms=(7500,),
            sent_at_ms=0, now_ms=0, current_node=4,
            corridor_of=lambda u, v: corridors.get((u, v)),
        )
        window = table.corridor_holders(40)[0].window
        assert window.start_ms == 0
        assert window.end_ms >= 7500


class TestCorridorConflicts:
    def test_opposing_traffic_conflicts(self) -> None:
        table = ReservationTable()
        table.record_corridor(claim(junction=40, robot_id=2, from_node=11, to_node=10))
        assert len(
            table.corridor_conflicts(
                40, Window(10_000, 18_000), exclude_robot=3, from_node=10
            )
        ) == 1

    def test_same_end_traffic_does_not(self) -> None:
        """A single-lane aisle is one lane, not one robot: they queue through it, and
        headway keeps them apart. Without this a robot at the mouth yields to one
        queued behind it and then blocks the approach it needs."""
        table = ReservationTable()
        table.record_corridor(claim(junction=40, robot_id=2, from_node=10, to_node=11))
        assert table.corridor_conflicts(
            40, Window(10_000, 18_000), exclude_robot=3, from_node=10
        ) == ()

    def test_corridors_and_junctions_are_separate_namespaces(self) -> None:
        """Junction 4 and edge 4 are different things; sharing one dict would have a
        junction claim block a corridor."""
        table = ReservationTable()
        table.record(claim(junction=4, robot_id=2))
        assert table.corridor_holders(4) == ()
        assert len(table.holders(4)) == 1


class TestExpiry:
    def test_past_windows_expire(self) -> None:
        """FR-5.11: no release message. This is what makes a failed robot harmless
        rather than a permanent obstruction, and Appendix C cites it."""
        table = ReservationTable()
        table.record(claim(start=1000, end=2000))
        assert table.expire(1500) == 0
        assert table.expire(2500) == 1
        assert len(table) == 0

    def test_expiry_covers_corridors_too(self) -> None:
        table = ReservationTable()
        table.record_corridor(claim(junction=40, start=1000, end=2000))
        assert table.expire(3000) == 1
        assert table.corridor_holders(40) == ()

    def test_dropping_a_lost_peer_clears_it_immediately(self) -> None:
        """FR-6.5: waiting for a dead robot's claims to lapse leaves the fleet
        avoiding junctions nobody is coming to."""
        table = ReservationTable()
        table.record(claim(junction=5, robot_id=2, start=0, end=999_999))
        table.record_corridor(claim(junction=40, robot_id=2, start=0, end=999_999))
        table.record(claim(junction=5, robot_id=3, start=0, end=999_999))
        assert table.drop_robot(2) == 2
        assert [h.robot_id for h in table.holders(5)] == [3]
        assert table.corridor_holders(40) == ()

    def test_dropping_an_unknown_robot_is_harmless(self) -> None:
        table = ReservationTable()
        table.record(claim())
        assert table.drop_robot(99) == 0

    def test_clear_empties_both_namespaces(self) -> None:
        table = ReservationTable()
        table.record(claim())
        table.record_corridor(claim(junction=40))
        table.clear()
        assert len(table) == 0


class TestReporting:
    def test_the_ranking_key_matches_appendix_c(self) -> None:
        assert claim(priority=10, robot_id=3).ranking_key() == (10, -3)

    def test_a_lower_id_outranks_on_equal_priority(self) -> None:
        assert claim(priority=10, robot_id=2).ranking_key() > claim(
            priority=10, robot_id=5
        ).ranking_key()

    def test_higher_priority_wins_regardless_of_id(self) -> None:
        assert claim(priority=200, robot_id=9).ranking_key() > claim(
            priority=10, robot_id=1
        ).ranking_key()

    def test_summary_is_readable_when_empty_and_when_not(self) -> None:
        table = ReservationTable()
        assert "empty" in table.summary()
        table.record(claim())
        assert "J5" in table.summary()
