"""Graph behaviour and the Phase 1 stop/go gate.

The gate the roadmap sets for Phase 1 is: given any start and goal, valid
connected paths are returned, and blocked edges are handled. Those are the first
two test classes here. The rest guard invariants that other modules will quietly
depend on -- heuristic admissibility above all, because losing it does not raise
an error, it just silently stops A* being least-cost.
"""

from __future__ import annotations

import itertools

import pytest

from core import config
from core.graph import Edge, Graph, MapError, Node
from tests.conftest import MAPS_DIR, reference_shortest_cost, to_networkx


def _tiny_graph() -> Graph:
    """Three nodes in a line, costs consistent with their spacing."""
    nodes = {
        0: Node(0, "A", 0, 0),
        1: Node(1, "B", 8000, 0),
        2: Node(2, "C", 16000, 0),
    }
    edges = {
        0: Edge(0, 0, 1, nominal_cost_ms=10_000, length_mm=8000),
        1: Edge(1, 1, 2, nominal_cost_ms=10_000, length_mm=8000),
    }
    return Graph(name="tiny", nodes=nodes, edges=edges)


class TestPhase1Gate:
    """Valid connected paths for any start/goal (Phase 1 definition of done)."""

    def test_every_node_pair_has_a_route(self, any_map: Graph) -> None:
        nodes = sorted(any_map.nodes)
        # Full pairwise on the 288-node map would be 82k Dijkstra runs; a
        # deterministic sample of pairs plus the connectivity check below gives
        # the same assurance without the cost.
        sample = nodes if len(nodes) <= 12 else nodes[:: max(1, len(nodes) // 12)]
        for start, goal in itertools.permutations(sample, 2):
            cost = reference_shortest_cost(any_map, start, goal)
            assert cost is not None, f"{any_map.name}: no route {start} -> {goal}"
            assert cost > 0

    def test_map_is_connected(self, any_map: Graph) -> None:
        assert any_map.is_connected()

    def test_route_to_self_costs_nothing(self, benchmark_map: Graph) -> None:
        assert reference_shortest_cost(benchmark_map, 0, 0) == 0

    def test_costs_agree_with_an_independent_implementation(self, any_map: Graph) -> None:
        """Cross-check against networkx (FR-3.1 requires we own graph.py)."""
        import networkx as nx

        reference = to_networkx(any_map)
        nodes = sorted(any_map.nodes)
        sample = nodes if len(nodes) <= 10 else nodes[:: max(1, len(nodes) // 10)]
        for start, goal in itertools.permutations(sample, 2):
            ours = reference_shortest_cost(any_map, start, goal)
            theirs = nx.dijkstra_path_length(reference, start, goal, weight="weight")
            assert ours == theirs, f"{any_map.name}: {start}->{goal} {ours} != {theirs}"


class TestBlockage:
    """FR-6.1: a blocked edge disappears from the topology."""

    def test_blocked_edge_is_not_offered_as_a_neighbour(self, benchmark_map: Graph) -> None:
        choke = 4
        before = benchmark_map.neighbours(10)
        assert any(edge_id == choke for _, edge_id in before)
        benchmark_map.block(choke)
        after = benchmark_map.neighbours(10)
        assert all(edge_id != choke for _, edge_id in after)

    def test_blocking_the_choke_forces_a_longer_route(self, benchmark_map: Graph) -> None:
        via_choke = reference_shortest_cost(benchmark_map, 2, 4)
        benchmark_map.block(4)
        via_bypass = reference_shortest_cost(benchmark_map, 2, 4)
        assert via_bypass is not None
        assert via_bypass > via_choke

    def test_blocking_is_idempotent_and_reversible(self, benchmark_map: Graph) -> None:
        benchmark_map.block(4)
        benchmark_map.block(4)
        assert benchmark_map.blocked_edges == frozenset({4})
        benchmark_map.unblock(4)
        assert benchmark_map.blocked_edges == frozenset()
        benchmark_map.unblock(4)  # unblocking twice must not raise

    def test_blocking_can_make_a_goal_unreachable(self) -> None:
        """FR-3.7 needs unreachability to be detectable, not to crash."""
        graph = _tiny_graph()
        graph.block(1)
        assert reference_shortest_cost(graph, 0, 2) is None
        assert not graph.is_connected()

    def test_blocking_an_unknown_edge_raises(self, benchmark_map: Graph) -> None:
        with pytest.raises(KeyError):
            benchmark_map.block(9999)


class TestHeuristicAdmissibility:
    """FR-3.2 requires an admissible heuristic, and admissibility is fragile."""

    def test_no_edge_undercuts_its_straight_line_time(self, any_map: Graph) -> None:
        for edge in any_map.edges.values():
            floor = any_map.straight_line_ms(edge.u, edge.v)
            assert edge.nominal_cost_ms >= floor, (
                f"{any_map.name}: edge {edge.id} cost {edge.nominal_cost_ms} < "
                f"straight-line {floor}"
            )

    def test_heuristic_never_exceeds_true_cost(self, any_map: Graph) -> None:
        """The defining property: h(a, b) <= actual least cost from a to b."""
        nodes = sorted(any_map.nodes)
        sample = nodes if len(nodes) <= 10 else nodes[:: max(1, len(nodes) // 10)]
        for start, goal in itertools.permutations(sample, 2):
            actual = reference_shortest_cost(any_map, start, goal)
            assert any_map.straight_line_ms(start, goal) <= actual

    def test_a_too_cheap_edge_is_rejected_at_validation(self) -> None:
        nodes = {0: Node(0, "A", 0, 0), 1: Node(1, "B", 40_000, 0)}
        # 40 m at 800 mm/s is 50 s; claiming 1 s would break admissibility.
        edges = {0: Edge(0, 0, 1, nominal_cost_ms=1000, length_mm=40_000)}
        graph = Graph(name="bad", nodes=nodes, edges=edges)
        with pytest.raises(MapError, match="inadmissible"):
            graph.validate(config.MAX_NODES_8BIT, config.MAX_EDGES_8BIT)


class TestSingleLaneGeometry:
    """FR-5.10: the yield decision happens before committing to the corridor."""

    def test_last_passing_point_is_where_an_alternative_still_exists(
        self, benchmark_map: Graph
    ) -> None:
        route = [0, 2, 10, 11, 4, 7]
        passing_point = benchmark_map.last_passing_point(route, corridor_edge=4)
        # C_WEST (10) is the corridor entrance but is a pass-through with only
        # two neighbours, so a robot there can no longer divert. The real
        # decision point is L_MID (2), which still offers the top and bottom
        # bypasses.
        assert passing_point == 2
        assert len(benchmark_map.neighbours(2)) >= 3

    def test_returns_none_when_the_route_avoids_the_corridor(
        self, benchmark_map: Graph
    ) -> None:
        assert benchmark_map.last_passing_point([0, 2, 1, 8], corridor_edge=4) is None

    def test_loop_map_corridors_are_all_single_lane(self, loop_map: Graph) -> None:
        """TC-3's cyclic conflict needs the triangle itself to be contended."""
        assert set(loop_map.single_lane_edges) == {0, 1, 2}


class TestQueries:
    def test_neighbours_are_sorted_for_determinism(self, benchmark_map: Graph) -> None:
        """FR-5.8: identical inputs must give identical decisions, and A*
        tie-breaking depends on neighbour iteration order."""
        for node_id in benchmark_map.nodes:
            neighbours = benchmark_map.neighbours(node_id)
            assert list(neighbours) == sorted(neighbours)

    def test_edge_between_is_directional_aware(self, benchmark_map: Graph) -> None:
        assert benchmark_map.edge_between(10, 11) == 4
        assert benchmark_map.edge_between(11, 10) == 4  # bidirectional
        assert benchmark_map.edge_between(0, 4) is None

    def test_one_way_edge_is_traversable_in_one_direction_only(self) -> None:
        nodes = {0: Node(0, "A", 0, 0), 1: Node(1, "B", 8000, 0)}
        edges = {
            0: Edge(0, 0, 1, nominal_cost_ms=10_000, length_mm=8000, bidirectional=False)
        }
        graph = Graph(name="oneway", nodes=nodes, edges=edges)
        assert graph.edge_between(0, 1) == 0
        assert graph.edge_between(1, 0) is None
        assert graph.neighbours(1) == ()

    def test_other_end_rejects_a_non_endpoint(self, benchmark_map: Graph) -> None:
        assert benchmark_map.edge(4).other_end(10) == 11
        with pytest.raises(KeyError):
            benchmark_map.edge(4).other_end(0)

    def test_chargers_and_junctions_are_reported(self, benchmark_map: Graph) -> None:
        # Charging lives at the staging bays, not at the depots: the depots are task
        # endpoints, and a robot that finished charging on one blocked the robot whose
        # task was there.
        assert set(benchmark_map.parking_nodes) == {12, 13, 14, 19, 20, 21}
        assert set(benchmark_map.chargers) == set(benchmark_map.parking_nodes)
        # Depots and parking bays are spurs, so they are not arbitration points.
        spurs = {0, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}  # depots, bays, station spurs
        assert set(benchmark_map.junctions) == set(benchmark_map.nodes) - spurs

    def test_parking_bays_are_not_task_endpoints(self, benchmark_map: Graph) -> None:
        """Parking on a pickup node simply moves the jam somewhere else."""
        import json

        raw = json.loads((MAPS_DIR / "benchmark_map.json").read_text(encoding="utf-8"))
        endpoints = set(raw["pickup_nodes"]) | set(raw["drop_nodes"])
        assert not set(benchmark_map.parking_nodes) & endpoints

    def test_unknown_ids_raise_with_a_useful_message(self, benchmark_map: Graph) -> None:
        with pytest.raises(KeyError, match="benchmark_map"):
            benchmark_map.node(9999)
        with pytest.raises(KeyError, match="benchmark_map"):
            benchmark_map.edge(9999)


class TestValidation:
    def test_shipped_maps_validate_at_their_scenario_width(self, any_map: Graph) -> None:
        wide = (
            len(any_map.edges) > config.MAX_EDGES_8BIT
            or len(any_map.nodes) > config.MAX_NODES_8BIT
        )
        max_nodes = config.MAX_NODES_16BIT if wide else config.MAX_NODES_8BIT
        max_edges = config.MAX_EDGES_16BIT if wide else config.MAX_EDGES_8BIT
        any_map.validate(max_nodes, max_edges)

    def test_wide_map_is_rejected_at_8_bit(self) -> None:
        """CON-10 / ASM-2: the same map can be legal for scale100 and not bench3."""
        graph = Graph.load("maps/warehouse_zoned_100.json")
        with pytest.raises(MapError, match="ASM-2"):
            graph.validate(config.MAX_NODES_8BIT, config.MAX_EDGES_8BIT)

    def test_self_loop_is_rejected(self) -> None:
        nodes = {0: Node(0, "A", 0, 0)}
        edges = {0: Edge(0, 0, 0, nominal_cost_ms=100, length_mm=0)}
        with pytest.raises(MapError, match="self-loop"):
            Graph(name="loop", nodes=nodes, edges=edges).validate(255, 255)

    def test_duplicate_edge_is_rejected(self) -> None:
        nodes = {0: Node(0, "A", 0, 0), 1: Node(1, "B", 8000, 0)}
        edges = {
            0: Edge(0, 0, 1, nominal_cost_ms=10_000, length_mm=8000),
            1: Edge(1, 1, 0, nominal_cost_ms=10_000, length_mm=8000),
        }
        with pytest.raises(MapError, match="duplicate edge"):
            Graph(name="dup", nodes=nodes, edges=edges).validate(255, 255)

    def test_disconnected_map_is_rejected(self) -> None:
        nodes = {0: Node(0, "A", 0, 0), 1: Node(1, "B", 8000, 0), 2: Node(2, "C", 99_000, 0)}
        edges = {0: Edge(0, 0, 1, nominal_cost_ms=10_000, length_mm=8000)}
        with pytest.raises(MapError, match="not connected"):
            Graph(name="split", nodes=nodes, edges=edges).validate(255, 255)

    def test_empty_map_is_rejected(self) -> None:
        with pytest.raises(MapError, match="no nodes"):
            Graph(name="empty", nodes={}, edges={}).validate(255, 255)

    def test_missing_file_reports_the_path(self) -> None:
        with pytest.raises(MapError, match="no map file"):
            Graph.load("maps/does_not_exist.json")


class TestLoading:
    def test_cost_is_derived_from_distance_when_omitted(self) -> None:
        """Hand-authored maps omit cost_ms; deriving it cannot break FR-3.2."""
        data = {
            "name": "derived",
            "nodes": [
                {"id": 0, "x": 0, "y": 0},
                {"id": 1, "x": 8000, "y": 0},
            ],
            "edges": [{"id": 0, "u": 0, "v": 1}],
        }
        graph = Graph.from_dict(data)
        expected = (8000 * 1000) // config.NOMINAL_SPEED_MM_S
        assert graph.nominal_cost_ms(0) == expected
        assert graph.length_mm(0) == 8000

    def test_explicit_cost_overrides_the_derived_one(self) -> None:
        data = {
            "name": "explicit",
            "nodes": [{"id": 0, "x": 0, "y": 0}, {"id": 1, "x": 8000, "y": 0}],
            "edges": [{"id": 0, "u": 0, "v": 1, "cost_ms": 30_000}],
        }
        assert Graph.from_dict(data).nominal_cost_ms(0) == 30_000

    def test_duplicate_node_id_is_rejected(self) -> None:
        data = {
            "name": "dup",
            "nodes": [{"id": 0, "x": 0, "y": 0}, {"id": 0, "x": 1, "y": 1}],
            "edges": [],
        }
        with pytest.raises(MapError, match="duplicate node id"):
            Graph.from_dict(data)

    def test_edge_to_missing_node_is_rejected(self) -> None:
        data = {
            "name": "dangling",
            "nodes": [{"id": 0, "x": 0, "y": 0}],
            "edges": [{"id": 0, "u": 0, "v": 5}],
        }
        with pytest.raises(MapError, match="missing node"):
            Graph.from_dict(data)
