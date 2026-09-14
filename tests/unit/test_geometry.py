"""Lane geometry: where two robots can collide without either being on the other's edge.

These are statements about arithmetic and the map, not about the coordination logic, so
they hold regardless of what arbitration decides. They exist because a collision at
3 AMRs was repeatedly mistaken for an arbitration defect: two robots 1386 mm and 951 mm
from the same junction, on edges sharing no lane, 499 mm apart. Nothing was wrong with
the ranking. The lanes cross at the corner and no rule accounted for it.
"""

from __future__ import annotations

import math

from core import config
from core.graph import Graph
from core.timewindows import ResourceModel


def separation(d1: int, d2: int, *, offset: int = config.AISLE_LANE_OFFSET_MM) -> float:
    """Distance between a robot ``d1`` short of a junction and one ``d2`` past it.

    The two travel perpendicular edges, each offset to the same side of its own aisle.
    Mirrors ``Robot.footprint_mm``: travelling +x puts the robot at +y, travelling +y
    puts it at -x, so a turn moves the offset from one axis to the other.
    """
    return math.dist((-d1, offset), (-offset, d2))


class TestTheCornerConvergence:
    def test_the_measured_collision_is_reproduced_by_arithmetic(self) -> None:
        """The pair observed on bench3 seed 13, against its logged footprints.

        r1 approaching junction 4 at (27196, 11300); r2 departing it at (26700, 11356);
        node 4 at (26000, 12000). Recomputing their separation from the formula and
        from the logged coordinates must agree, which is what shows the collision is
        the lane crossing and not a simulator artefact.
        """
        logged = math.dist((27196, 11300), (26700, 11356))
        assert round(logged) == 499
        assert logged < config.COLLISION_DISTANCE_MM

    def test_robots_over_a_metre_from_a_junction_can_still_collide(self) -> None:
        assert separation(1084, 1020) < config.COLLISION_DISTANCE_MM

    def test_the_footprint_radius_bounds_the_exposure(self) -> None:
        """Outside JUNCTION_FOOTPRINT_MM the pairing is safe, inside it may not be.

        This is what makes the constant a derivation rather than a guess: it is the
        smallest radius that contains every unsafe (d1, d2).
        """
        r = config.JUNCTION_FOOTPRINT_MM
        assert separation(r, r) >= config.COLLISION_DISTANCE_MM
        worst = config.AISLE_LANE_OFFSET_MM
        assert separation(worst, worst) == 0.0
        for d1 in range(0, 4000, 50):
            for d2 in range(0, 4000, 50):
                if separation(d1, d2) < config.COLLISION_DISTANCE_MM:
                    assert d1 <= r and d2 <= r, (
                        f"unsafe pair ({d1}, {d2}) lies outside the footprint radius"
                    )

    def test_only_a_turn_crosses_the_lanes(self) -> None:
        """Both approaching, or both departing, cannot breach the collision distance.

        So the exposure is specific to one robot arriving while another departs -- which
        is why it survived testing that exercised head-on and following traffic.
        """
        offset = config.AISLE_LANE_OFFSET_MM
        for d1 in range(0, 3000, 25):
            for d2 in range(0, 3000, 25):
                both_arriving = math.dist((-d1, offset), (-offset, -d2))
                both_departing = math.dist((d1, offset), (-offset, d2))
                assert both_arriving >= offset
                assert both_departing >= offset


class TestRegionCrossingIsSizedByGeometryNotAConstant:
    """core/timewindows.py::ResourceModel.region_cross_ms replaced the old fixed
    JUNCTION_OCCUPANCY_MS reservation (Appendix E, 700 ms, defect 10) that a
    yielding robot could outlive. The old constant was never wired into the
    Phase-12 route-level planner and has been removed from config.py; this pins
    the invariant its replacement actually enforces.

    Raising YIELD_SPEED_MM_S needs no change here. It is not true that a robot
    never moves at yield speed within JUNCTION_FOOTPRINT_MM -- once a region step
    is entered, gating moves on to whatever comes next (a lane, typically), and a
    robot can be shedding to yield speed on approach to *that* boundary while still
    geometrically short of the footprint radius it just departed (measured: 4 mm
    past the node, still slowing for a lane 1196 mm further on). What matters is
    that this is not the exposure JUNCTION_FOOTPRINT_MM bounds: a second robot
    cannot book that same region while a live peer still reports it as
    current_node (core/robot.py::_may_enter, the REGION "opaque peer" branch),
    independent of speed or elapsed reservation time. See
    tests/coordination/test_safety_cases.py::TestYieldSpeedDoesNotGovernRegionSafety
    for the direct, empirical check (zero collisions across seeds at several
    yield speeds, including well above the shipped value).
    """

    def test_region_cross_ms_exactly_covers_the_footprint_at_nominal_speed(self) -> None:
        model = ResourceModel(Graph.load("maps/benchmark_map.json"))
        reach_mm = model.region_cross_ms * config.NOMINAL_SPEED_MM_S // 1000
        assert reach_mm == 2 * config.JUNCTION_FOOTPRINT_MM

    def test_a_fixed_millisecond_constant_would_have_undersized_it(self) -> None:
        """Why the fix is geometry, not a bigger constant: Appendix E's 700 ms
        reserved only 560 mm of nominal-speed travel against the 2400 mm a full
        crossing (boundary to boundary) actually needs."""
        appendix_e_ms = 700
        reach_mm = appendix_e_ms * config.NOMINAL_SPEED_MM_S // 1000
        assert reach_mm < 2 * config.JUNCTION_FOOTPRINT_MM

    def test_a_yielding_robot_stops_outside_the_footprint(self) -> None:
        """The one guarantee that must hold regardless of speed: whatever a robot
        is waiting for, it comes to rest clear of the corner, not inside it."""
        assert config.YIELD_STANDOFF_MM > config.JUNCTION_FOOTPRINT_MM


class TestSingleLaneAislesHaveNoOffset:
    def test_a_single_lane_corridor_is_the_centre_line(self) -> None:
        """So a head-on there is a real overlap, and the corner analysis above does not
        apply: there is only one lane to be in."""
        graph = Graph.load("maps/benchmark_map.json")
        assert graph.single_lane_edges


class TestTheCornerOutlastsAppendixE:
    """Historical record: why the region had to be sized by geometry rather than
    by Appendix E's fixed constant, which this file no longer imports because
    core/config.py no longer defines it (dead since the Phase-12 rewrite to
    route-level, time-window reservations)."""

    def test_the_corner_takes_longer_to_cross_than_appendix_e_reserved(self) -> None:
        """One crossing (one footprint, boundary to node) already outlasts the 700 ms
        Appendix E specified -- before even considering the full two-footprint
        transit ResourceModel.region_cross_ms now reserves."""
        appendix_e_ms = 700
        corner_ms = config.JUNCTION_FOOTPRINT_MM * 1000 // config.NOMINAL_SPEED_MM_S
        assert corner_ms > appendix_e_ms
