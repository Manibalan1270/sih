"""A* correctness and the Phase 3 stop/go gate.

The gate is "correct least-cost routes on known maps". Correctness here means
provably least-cost, not merely short, so the central tests compare A* against
Dijkstra (which needs no heuristic and so cannot share a heuristic bug) and
against networkx as a third opinion.
"""

from __future__ import annotations

import itertools
import time

import pytest

from core import config
from core.graph import Graph
from core.planner_astar import AStarPlanner, Unreachable
from tests.conftest import ReferencePlanner, reference_shortest_cost, to_networkx


class TestPhase3Gate:
    """Correct least-cost routes on known maps."""

    def test_costs_match_dijkstra_everywhere(self, any_map: Graph) -> None:
        """A* must return the same *cost* as Dijkstra for every pair.

        Not the same route: equal-cost alternatives exist all over a warehouse
        grid and either is correct. Cost equality is the real claim.
        """
        astar = AStarPlanner(any_map)
        nodes = sorted(any_map.nodes)
        sample = nodes if len(nodes) <= 12 else nodes[:: max(1, len(nodes) // 12)]
        for start, goal in itertools.permutations(sample, 2):
            route = astar.route(start, goal)
            assert route is not None, f"{any_map.name}: no route {start}->{goal}"
            assert astar.route_cost(route) == reference_shortest_cost(
                any_map, start, goal
            ), f"{any_map.name}: {start}->{goal} is not least-cost"

    def test_costs_match_networkx(self, benchmark_map: Graph) -> None:
        """A third implementation, to catch a bug shared by the first two."""
        import networkx as nx

        astar = AStarPlanner(benchmark_map)
        reference = to_networkx(benchmark_map)
        for start, goal in itertools.permutations(sorted(benchmark_map.nodes), 2):
            route = astar.route(start, goal)
            assert astar.route_cost(route) == nx.dijkstra_path_length(
                reference, start, goal, weight="weight"
            )

    def test_known_route_on_the_benchmark_map(self, benchmark_map: Graph) -> None:
        """The depot-to-depot route must go through the choke corridor. If this
        ever changes, the section 6.3 experiment stops forcing path overlap."""
        astar = AStarPlanner(benchmark_map)
        assert astar.route(0, 7) == [0, 2, 10, 11, 4, 7]

    def test_route_is_a_connected_walk(self, any_map: Graph) -> None:
        """Every consecutive pair must be joined by a real edge."""
        astar = AStarPlanner(any_map)
        nodes = sorted(any_map.nodes)
        for start, goal in itertools.permutations(nodes[:6], 2):
            route = astar.route(start, goal)
            assert route is not None
            assert route[0] == start and route[-1] == goal
            for index in range(len(route) - 1):
                assert any_map.edge_between(route[index], route[index + 1]) is not None

    def test_route_visits_no_node_twice(self, benchmark_map: Graph) -> None:
        astar = AStarPlanner(benchmark_map)
        for start, goal in itertools.permutations(sorted(benchmark_map.nodes), 2):
            route = astar.route(start, goal)
            assert len(route) == len(set(route)), f"route {route} revisits a node"


class TestDeterminism:
    def test_identical_inputs_give_identical_routes(self, any_map: Graph) -> None:
        """FR-5.8 needs identical inputs to produce identical decisions, and
        equal-cost ties are common enough on a grid that this is a real risk."""
        first = AStarPlanner(any_map)
        second = AStarPlanner(any_map)
        nodes = sorted(any_map.nodes)
        sample = nodes if len(nodes) <= 10 else nodes[:: max(1, len(nodes) // 10)]
        for start, goal in itertools.permutations(sample, 2):
            assert first.route(start, goal) == second.route(start, goal)

    def test_ties_are_broken_the_same_way_every_time(self) -> None:
        """A symmetric diamond: both routes cost exactly the same."""
        data = {
            "name": "diamond",
            "nodes": [
                {"id": 0, "x": 0, "y": 0},
                {"id": 1, "x": 8000, "y": 8000},
                {"id": 2, "x": 8000, "y": -8000},
                {"id": 3, "x": 16000, "y": 0},
            ],
            "edges": [
                {"id": 0, "u": 0, "v": 1},
                {"id": 1, "u": 0, "v": 2},
                {"id": 2, "u": 1, "v": 3},
                {"id": 3, "u": 2, "v": 3},
            ],
        }
        graph = Graph.from_dict(data)
        planner = AStarPlanner(graph)
        routes = {tuple(planner.route(0, 3)) for _ in range(25)}
        assert len(routes) == 1, f"tie broken inconsistently: {routes}"

    def test_no_floating_point_enters_the_search(self, benchmark_map: Graph) -> None:
        """CON-6 / FR-3.8. A float g-score would drift differently on different
        hardware and break FR-5.8 in a way no behavioural test would catch."""
        astar = AStarPlanner(benchmark_map)
        route = astar.route(0, 7)
        assert all(isinstance(node, int) for node in route)
        assert isinstance(astar.route_cost(route), int)
        assert isinstance(benchmark_map.straight_line_ms(0, 7), int)


class TestHeuristic:
    def test_heuristic_is_admissible_for_every_pair(self, any_map: Graph) -> None:
        """The defining property of A*'s optimality guarantee (FR-3.2)."""
        nodes = sorted(any_map.nodes)
        sample = nodes if len(nodes) <= 10 else nodes[:: max(1, len(nodes) // 10)]
        for start, goal in itertools.permutations(sample, 2):
            assert any_map.straight_line_ms(start, goal) <= reference_shortest_cost(
                any_map, start, goal
            )

    def test_heuristic_is_exact_on_a_grid_map(self) -> None:
        """On a rectilinear grid the Manhattan bound is not merely admissible, it
        is exact: no route can be shorter and none is longer. This is the property
        that makes planning cheap enough for 100 robots inside FR-3.5's budget.

        With the Euclidean bound this was badly loose -- the diagonal is a poor
        estimate of a path forced to follow the aisles -- and A* degenerated to
        Dijkstra on every query.
        """
        graph = Graph.load("maps/warehouse_zoned_100.json")
        astar = AStarPlanner(graph)
        cols = 24
        for start, goal in ((0, 23), (0, 11 * cols), (10 * cols + 5, 6 * cols + 14)):
            route = astar.route(start, goal)
            assert graph.straight_line_ms(start, goal) == astar.route_cost(route)

    def test_heuristic_prunes_the_search_for_typical_queries(self) -> None:
        """A heuristic that guides nothing is a bug no correctness test catches."""
        graph = Graph.load("maps/warehouse_zoned_100.json")
        cols = 24
        for start, goal in ((0, 23), (0, 11 * cols), (10 * cols + 5, 6 * cols + 14)):
            astar = AStarPlanner(graph)
            astar.route(start, goal)
            blind = _expansions_with_zero_heuristic(graph, start, goal)
            assert astar.stats.expansions * 3 < blind, (
                f"{start}->{goal}: A* expanded {astar.stats.expansions} against "
                f"Dijkstra's {blind}; the heuristic is barely guiding the search"
            )

    def test_the_diagonal_worst_case_cannot_be_pruned(self) -> None:
        """Corner to opposite corner on a uniform grid is the one case where no
        heuristic can help, and it is worth pinning down so a future reader does
        not mistake it for a regression.

        Every monotone staircase between opposite corners costs exactly the same,
        so every node in the enclosing rectangle ties at the optimal f and A* must
        expand all of them. The heuristic is exact here; the graph simply has an
        enormous number of equally-optimal routes.
        """
        graph = Graph.load("maps/warehouse_zoned_100.json")
        # The far corner of the *floor*, not max(nodes): staging bays are appended
        # after the grid, and a bay is a spur rather than a corner.
        cols, rows = 24, 12
        goal = (rows - 1) * cols + (cols - 1)
        astar = AStarPlanner(graph)
        route = astar.route(0, goal)
        assert graph.straight_line_ms(0, goal) == astar.route_cost(route), (
            "the heuristic should be exact even in the case it cannot prune"
        )
        # Every floor node ties, so every floor node is expanded: the whole rectangle
        # less the goal itself. The staging bays hang off the perimeter as dead-end
        # spurs, and those the heuristic *does* prune -- a bay leads away from the goal
        # -- while a zero heuristic wanders into every one within the cost radius.
        #
        # This used to assert equality with the zero-heuristic count, which held only
        # because the bays were generated on top of floor nodes (see
        # TestLanesDoNotOverlapWithoutAJunction in test_maps.py). Once the bays sat where
        # they belong the equality broke in the heuristic's favour.
        floor = cols * rows - 1
        assert astar.stats.expansions == floor, (
            f"expected the full floor rectangle ({floor}) and no more; "
            f"got {astar.stats.expansions}"
        )
        assert _expansions_with_zero_heuristic(graph, 0, goal) >= floor

    def test_grid_maps_use_the_manhattan_bound(self) -> None:
        """Valid only because every edge in these maps is axis-parallel."""
        grid = Graph.load("maps/warehouse_zoned_100.json")
        assert grid._axis_aligned

    def test_maps_with_diagonal_aisles_keep_the_euclidean_bound(
        self, benchmark_map: Graph
    ) -> None:
        """The benchmark map's bypasses run diagonally, so Manhattan would
        overestimate and A* would stop being least-cost."""
        assert not benchmark_map._axis_aligned


def _expansions_with_zero_heuristic(graph: Graph, start: int, goal: int) -> int:
    """Dijkstra expansion count, for comparison against guided A*."""
    import heapq

    best = {start: 0}
    queue = [(0, start)]
    closed: set[int] = set()
    expansions = 0
    while queue:
        cost, node = heapq.heappop(queue)
        if node in closed:
            continue
        if node == goal:
            return expansions
        closed.add(node)
        expansions += 1
        for neighbour, edge_id in graph.neighbours(node):
            new_cost = cost + graph.nominal_cost_ms(edge_id)
            if new_cost < best.get(neighbour, 1 << 62):
                best[neighbour] = new_cost
                heapq.heappush(queue, (new_cost, neighbour))
    return expansions


class TestUnreachable:
    def test_unreachable_goal_returns_none(self, benchmark_map: Graph) -> None:
        """FR-3.7. A normal operating condition on a floor with blocked aisles,
        so it is a return value rather than an exception."""
        for edge_id in (4, 7, 10):
            benchmark_map.block(edge_id)
        astar = AStarPlanner(benchmark_map)
        assert astar.route(0, 5) is None
        assert astar.stats.unreachable == 1

    def test_route_or_raise_names_the_requirement(self, benchmark_map: Graph) -> None:
        for edge_id in (4, 7, 10):
            benchmark_map.block(edge_id)
        astar = AStarPlanner(benchmark_map)
        with pytest.raises(Unreachable, match="FR-3.7"):
            astar.route_or_raise(0, 5)

    def test_travel_cost_reports_infinity_rather_than_failing(
        self, benchmark_map: Graph
    ) -> None:
        """The auction asks for prices, not routes, and must not have to handle an
        exception to discover a task is unbiddable."""
        for edge_id in (4, 7, 10):
            benchmark_map.block(edge_id)
        astar = AStarPlanner(benchmark_map)
        assert astar.travel_cost(0, 5) == config.INFINITE_COST

    def test_unknown_node_raises_rather_than_returning_none(
        self, benchmark_map: Graph
    ) -> None:
        """A bogus node id is a programming error, not an unreachable goal, and
        conflating the two would hide it."""
        astar = AStarPlanner(benchmark_map)
        with pytest.raises(KeyError):
            astar.route(0, 9999)

    def test_route_to_self_is_the_single_node(self, benchmark_map: Graph) -> None:
        """A route is a sequence of places to be, and the robot is at one."""
        assert AStarPlanner(benchmark_map).route(4, 4) == [4]


class TestBlockageAndHardBlocks:
    def test_blocked_edge_is_routed_around(self, benchmark_map: Graph) -> None:
        astar = AStarPlanner(benchmark_map)
        assert 10 in astar.route(0, 7)
        benchmark_map.block(4)
        detour = astar.route(0, 7)
        assert 10 not in detour and 11 not in detour
        assert astar.route_cost(detour) > 0

    def test_an_infinite_cost_edge_is_never_planned_through(
        self, benchmark_map: Graph
    ) -> None:
        """FR-3.9: a route shall not be planned through an edge subject to a
        conflicting reservation. Skipping the edge makes that structural rather
        than a matter of how large the penalty happens to be."""

        def cost(edge_id: int) -> int:
            return (
                config.INFINITE_COST
                if edge_id == 4
                else benchmark_map.nominal_cost_ms(edge_id)
            )

        astar = AStarPlanner(benchmark_map, cost=cost)
        route = astar.route(0, 7)
        assert route is not None
        assert 4 not in astar.route_edges(route)

    def test_a_hard_block_can_make_a_goal_unreachable(self, benchmark_map: Graph) -> None:
        blocked = {4, 7, 10}

        def cost(edge_id: int) -> int:
            return (
                config.INFINITE_COST
                if edge_id in blocked
                else benchmark_map.nominal_cost_ms(edge_id)
            )

        assert AStarPlanner(benchmark_map, cost=cost).route(0, 5) is None


class TestInjectedCost:
    def test_the_planner_does_not_know_learning_exists(self, benchmark_map: Graph) -> None:
        """FR-3.4: the planner consumes the composed cost without special-casing
        its layers, which is what lets Phase 8 add EWMA and pheromone without
        touching this module."""
        expensive_choke = {4: 10_000_000}

        def cost(edge_id: int) -> int:
            return expensive_choke.get(edge_id, benchmark_map.nominal_cost_ms(edge_id))

        astar = AStarPlanner(benchmark_map, cost=cost)
        route = astar.route(0, 7)
        assert 4 not in astar.route_edges(route), (
            "a heavily penalised edge was still chosen, so the injected cost is "
            "not reaching the search"
        )

    def test_default_cost_is_the_nominal_map_cost(self, benchmark_map: Graph) -> None:
        """FR-2.7: a robot joining a fleet starts on nominal costs."""
        astar = AStarPlanner(benchmark_map)
        assert astar.cost(4) == benchmark_map.nominal_cost_ms(4)

    def test_route_cost_of_a_disconnected_sequence_is_infinite(
        self, benchmark_map: Graph
    ) -> None:
        astar = AStarPlanner(benchmark_map)
        assert astar.route_cost([0, 5]) == config.INFINITE_COST


class TestLatencyBudget:
    def test_planning_the_largest_map_stays_inside_the_budget(self) -> None:
        """FR-3.5 / NFR-1.2: 50 ms for any graph conforming to ASM-2.

        Timed on the development host, which is far faster than an ESP32, so this
        is a smoke test rather than proof of the embedded budget. The
        machine-independent claim is the expansion count asserted below: A*
        expands a small fraction of the graph, which is what makes the budget
        achievable on the target at all.
        """
        graph = Graph.load("maps/warehouse_zoned_100.json")
        astar = AStarPlanner(graph)
        corner = max(graph.nodes)

        started = time.perf_counter()
        for _ in range(20):
            astar.route(0, corner)
        elapsed_ms = (time.perf_counter() - started) * 1000 / 20

        assert elapsed_ms < config.PLAN_DEADLINE_MS, (
            f"{elapsed_ms:.1f} ms per plan on a {len(graph.nodes)}-node map "
            f"exceeds the {config.PLAN_DEADLINE_MS} ms budget (FR-3.5)"
        )

    def test_expansions_are_a_small_fraction_of_the_graph(self) -> None:
        graph = Graph.load("maps/warehouse_zoned_100.json")
        astar = AStarPlanner(graph)
        astar.route(0, max(graph.nodes))
        assert astar.stats.expansions < len(graph.nodes), (
            f"expanded {astar.stats.expansions} of {len(graph.nodes)} nodes; the "
            f"heuristic should keep this well below the whole graph"
        )


class TestAgainstTheReferencePlanner:
    def test_drop_in_replacement_for_the_reference(self, benchmark_map: Graph) -> None:
        """The robot takes a RoutePlanner protocol; both must satisfy it, so
        Phase 3 substitutes A* without any change to core/robot.py."""
        astar = AStarPlanner(benchmark_map)
        reference = ReferencePlanner(benchmark_map)
        for start, goal in itertools.permutations(sorted(benchmark_map.nodes)[:6], 2):
            mine, theirs = astar.route(start, goal), reference.route(start, goal)
            assert astar.route_cost(mine) == astar.route_cost(theirs)

    def test_the_robot_runs_on_astar(self, benchmark_map: Graph) -> None:
        from core.robot import Robot
        from core.state_machine import State
        from core.task import Task
        from simulator.engine import Engine

        robot = Robot(
            robot_id=1,
            graph=benchmark_map,
            planner=AStarPlanner(benchmark_map),
            home_node=0,
        )
        engine = Engine(graph=benchmark_map, robots=[robot], seed=1)
        robot.accept_task(
            Task(task_id=1, pickup=1, drop=5, priority=10, created_at_ms=0), 0
        )
        assert engine.run(max_ms=600_000, until=lambda e: e.work_finished())
        assert robot.state is State.IDLE
        assert robot.metrics.tasks_completed == 1
        assert robot.current_node == 5
