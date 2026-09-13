"""Shared test fixtures and helpers."""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import scenarios  # noqa: E402
from core.graph import Graph  # noqa: E402
from core.zones import ZoneMap  # noqa: E402

MAPS_DIR = REPO_ROOT / "maps"

ALL_MAP_NAMES = ("benchmark_map", "loop_map", "warehouse_zoned_30", "warehouse_zoned_100")


@pytest.fixture
def benchmark_map() -> Graph:
    return Graph.load(MAPS_DIR / "benchmark_map.json")


@pytest.fixture
def loop_map() -> Graph:
    return Graph.load(MAPS_DIR / "loop_map.json")


@pytest.fixture(params=ALL_MAP_NAMES)
def any_map(request) -> Graph:
    """Every shipped map, so structural invariants are checked against all."""
    return Graph.load(MAPS_DIR / f"{request.param}.json")


@pytest.fixture
def bench3():
    return scenarios.get("bench3")


def zone_map_for(graph: Graph, *, enabled: bool = True) -> ZoneMap:
    return ZoneMap.from_graph(graph, enabled=enabled)


def reference_shortest_cost(graph: Graph, start: int, goal: int) -> int | None:
    """Dijkstra over nominal costs, independent of core/planner_astar.py.

    Exists so graph behaviour can be verified before the planner is written, and
    afterwards as a second opinion the planner is checked against.
    """
    best: dict[int, int] = {start: 0}
    queue: list[tuple[int, int]] = [(0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            return cost
        if cost > best.get(node, 1 << 60):
            continue
        for neighbour, edge_id in graph.neighbours(node):
            new_cost = cost + graph.nominal_cost_ms(edge_id)
            if new_cost < best.get(neighbour, 1 << 60):
                best[neighbour] = new_cost
                heapq.heappush(queue, (new_cost, neighbour))
    return None


class ReferencePlanner:
    """Dijkstra route planner satisfying ``core.robot.RoutePlanner``.

    Exists so the robot and engine can be exercised before core/planner_astar.py
    is written, and afterwards as the independent implementation A* is checked
    against. Dijkstra needs no heuristic, so it cannot share a bug with A*.
    """

    def __init__(self, graph: Graph, *, cost=None) -> None:
        self.graph = graph
        self._cost = cost or graph.nominal_cost_ms
        self.calls = 0

    def route(self, start: int, goal: int) -> list[int] | None:
        self.calls += 1
        if start == goal:
            return [start]
        best: dict[int, int] = {start: 0}
        came: dict[int, int] = {}
        queue: list[tuple[int, int]] = [(0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(came[path[-1]])
                return list(reversed(path))
            if cost > best.get(node, 1 << 60):
                continue
            for neighbour, edge_id in self.graph.neighbours(node):
                new_cost = cost + self._cost(edge_id)
                if new_cost < best.get(neighbour, 1 << 60):
                    best[neighbour] = new_cost
                    came[neighbour] = node
                    heapq.heappush(queue, (new_cost, neighbour))
        return None


def to_networkx(graph: Graph):
    """Mirror a Graph into networkx for reference cross-checks.

    Tests only. FR-3.1 requires the project to own its graph implementation, so
    networkx exists here purely as an independent second implementation to check
    ours against -- never as a runtime dependency.
    """
    import networkx as nx

    reference = nx.DiGraph()
    reference.add_nodes_from(graph.nodes)
    for edge in graph.edges.values():
        if graph.is_blocked(edge.id):
            continue
        reference.add_edge(edge.u, edge.v, weight=edge.nominal_cost_ms)
        if edge.bidirectional:
            reference.add_edge(edge.v, edge.u, weight=edge.nominal_cost_ms)
    return reference
