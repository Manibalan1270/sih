"""Zone partitioning (FE-9).

Two properties matter most here and neither is obvious from reading zones.py:
that an unzoned fleet behaves as a flood auction rather than as a broken zoned
one, and that eligibility fails *open* -- because a task nobody is eligible for
is re-announced forever under FR-4.13 and never completed.
"""

from __future__ import annotations

import pytest

from core.graph import NO_ZONE, Edge, Graph, Node
from core.zones import ZoneMap


class TestPartition:
    def test_zones_are_read_from_the_map(self, benchmark_map: Graph) -> None:
        zones = ZoneMap.from_graph(benchmark_map, enabled=True)
        assert zones.zone_count == 2
        assert zones.zone_of(0) == 0  # L_DEPOT, left block
        assert zones.zone_of(7) == 1  # R_DEPOT, right block

    def test_large_map_partition_matches_its_declared_bands(self) -> None:
        graph = Graph.load("maps/warehouse_zoned_100.json")
        zones = ZoneMap.from_graph(graph, enabled=True)
        assert zones.zone_count == 24  # 6 column bands x 4 row bands
        assert sum(len(ns) for ns in zones.zone_nodes.values()) == len(graph.nodes)

    def test_every_node_belongs_to_exactly_one_zone(self) -> None:
        graph = Graph.load("maps/warehouse_zoned_30.json")
        zones = ZoneMap.from_graph(graph, enabled=True)
        seen: set[int] = set()
        for nodes in zones.zone_nodes.values():
            assert not seen & set(nodes), "a node appears in two zones"
            seen |= set(nodes)
        assert seen == set(graph.nodes)

    def test_a_zone_is_its_own_neighbour(self, benchmark_map: Graph) -> None:
        """So eligible_to_bid needs no same-zone special case."""
        zones = ZoneMap.from_graph(benchmark_map, enabled=True)
        for zone in zones.zone_ids:
            assert zone in zones.neighbours_of(zone)

    def test_zones_joined_by_an_edge_are_adjacent(self, benchmark_map: Graph) -> None:
        # The choke corridor spans C_WEST (zone 0) and C_EAST (zone 1).
        zones = ZoneMap.from_graph(benchmark_map, enabled=True)
        assert 1 in zones.neighbours_of(0)
        assert 0 in zones.neighbours_of(1)

    def test_a_single_zone_map_disables_zoning(self, loop_map: Graph) -> None:
        """Zoning one zone is not zoning; section 4.9 says flood is correct."""
        zones = ZoneMap.from_graph(loop_map, enabled=True)
        assert zones.enabled is False


class TestEligibility:
    """FR-4.14 / FR-9.3: own zone plus adjacent zones may bid."""

    def test_unzoned_fleet_lets_everyone_bid(self, benchmark_map: Graph) -> None:
        zones = ZoneMap.from_graph(benchmark_map, enabled=False)
        assert zones.eligible_to_bid(robot_zone=0, task_zone=1) is True
        assert zones.eligible_to_bid(robot_zone=99, task_zone=42) is True

    def test_same_zone_is_eligible(self) -> None:
        zones = ZoneMap.from_graph(
            Graph.load("maps/warehouse_zoned_100.json"), enabled=True
        )
        assert zones.eligible_to_bid(robot_zone=7, task_zone=7)

    def test_adjacent_zone_is_eligible(self) -> None:
        zones = ZoneMap.from_graph(
            Graph.load("maps/warehouse_zoned_100.json"), enabled=True
        )
        neighbour = next(z for z in zones.neighbours_of(7) if z != 7)
        assert zones.eligible_to_bid(robot_zone=7, task_zone=neighbour)

    def test_distant_zone_is_not_eligible(self) -> None:
        """The whole point: traffic is bounded, not fleet-wide."""
        zones = ZoneMap.from_graph(
            Graph.load("maps/warehouse_zoned_100.json"), enabled=True
        )
        far = max(
            zones.zone_ids,
            key=lambda z: 0 if z in zones.neighbours_of(0) else 1,
        )
        assert far not in zones.neighbours_of(0)
        assert zones.eligible_to_bid(robot_zone=0, task_zone=far) is False

    def test_unknown_zone_fails_open(self) -> None:
        """An unzoned task must not become unallocatable.

        FR-4.13 re-announces a task nobody bid on, so excluding every robot from
        a NO_ZONE task would produce an infinite re-announce loop rather than a
        clean failure.
        """
        zones = ZoneMap.from_graph(
            Graph.load("maps/warehouse_zoned_100.json"), enabled=True
        )
        assert zones.eligible_to_bid(robot_zone=3, task_zone=NO_ZONE) is True
        assert zones.eligible_to_bid(robot_zone=NO_ZONE, task_zone=3) is True


class TestFrameBudget:
    """FR-9.6 / NFR-1.12: auction traffic scales with fleet/zones, not fleet."""

    def test_scale_map_stays_inside_the_frame_budget(self) -> None:
        graph = Graph.load("maps/warehouse_zoned_100.json")
        zones = ZoneMap.from_graph(graph, enabled=True)
        bidders = zones.expected_bidders(100)
        frames = 2 * bidders  # section 4.9: a flood auction costs ~2N frames
        assert frames <= 40, (
            f"{frames} frames/task at 100 AMRs across {zones.zone_count} zones "
            f"exceeds the NFR-1.12 budget of 40"
        )

    def test_zoning_beats_flooding_at_scale(self) -> None:
        graph = Graph.load("maps/warehouse_zoned_100.json")
        zoned = ZoneMap.from_graph(graph, enabled=True)
        flooded = ZoneMap.from_graph(graph, enabled=False)
        assert zoned.expected_bidders(100) < flooded.expected_bidders(100) / 2

    def test_flood_eligibility_is_the_whole_fleet(self, benchmark_map: Graph) -> None:
        flooded = ZoneMap.from_graph(benchmark_map, enabled=False)
        assert flooded.expected_bidders(100) == 100

    def test_small_fleet_is_unzoned_by_design(self, bench3) -> None:
        """Section 4.9: at 3-5 AMRs zoning adds complexity without benefit."""
        assert bench3.zoned is False


class TestAssignment:
    def test_assignment_follows_the_home_node(self, benchmark_map: Graph) -> None:
        """FR-9.1 / ASM-16: static, derived from the map, no negotiation."""
        zones = ZoneMap.from_graph(benchmark_map, enabled=True)
        assert zones.assign(robot_id=1, home_node=0) == 0
        assert zones.assign(robot_id=2, home_node=7) == 1

    def test_assignment_is_deterministic(self, benchmark_map: Graph) -> None:
        """FR-9.7: every robot computes the same table, so there is no zone
        server whose loss could stop the fleet."""
        a = ZoneMap.from_graph(benchmark_map, enabled=True)
        b = ZoneMap.from_graph(benchmark_map, enabled=True)
        assert a == b
        assert all(a.assign(1, n) == b.assign(1, n) for n in benchmark_map.nodes)


class TestValidation:
    def test_partially_zoned_map_is_rejected(self) -> None:
        nodes = {
            0: Node(0, "A", 0, 0, zone_id=0),
            1: Node(1, "B", 8000, 0, zone_id=1),
            2: Node(2, "C", 16000, 0),  # no zone
        }
        edges = {
            0: Edge(0, 0, 1, nominal_cost_ms=10_000, length_mm=8000),
            1: Edge(1, 1, 2, nominal_cost_ms=10_000, length_mm=8000),
        }
        graph = Graph(name="partial", nodes=nodes, edges=edges)
        zones = ZoneMap.from_graph(graph, enabled=True)
        with pytest.raises(ValueError, match="FR-9.2"):
            zones.validate(graph)

    def test_shipped_zoned_maps_validate(self) -> None:
        for name in ("warehouse_zoned_30", "warehouse_zoned_100"):
            graph = Graph.load(f"maps/{name}.json")
            ZoneMap.from_graph(graph, enabled=True).validate(graph)

    def test_unzoned_map_skips_validation(self, loop_map: Graph) -> None:
        ZoneMap.from_graph(loop_map, enabled=False).validate(loop_map)
