"""A* route planning (FE-3, FR-3.2).

Each robot plans its own routes. There is no shared plan and no server to ask,
which is the whole architectural point: once the traffic model is per-robot
(FR-2.1), only the robot holds the cost information needed to route itself well.

Three properties of this implementation are requirements rather than choices:

* **Integer arithmetic throughout** (CON-6, FR-3.8). Costs are milliseconds. No
  float ever enters the open set, so two robots given identical inputs produce
  byte-identical routes -- which FR-5.8 needs and float accumulation would
  quietly break on different hardware.
* **An admissible heuristic** (FR-3.2). ``Graph.straight_line_ms`` is the
  straight-line time at cruise speed, and ``Graph.validate`` guarantees no edge
  costs less than its own straight-line time, so the heuristic can never
  overestimate. That is what makes the returned route provably least-cost rather
  than merely short.
* **A deterministic tie-break.** Equal-cost routes are extremely common on a
  warehouse grid, and ``heapq`` alone would order ties by whatever object landed
  in the heap first. Pushing an explicit insertion counter makes the choice a
  function of the map, so a rerun cannot pick a different equal-cost route.

The cost function is injected. The planner does not know that learning exists: it
consumes the composed edge cost of FR-2.2 and never inspects its layers (FR-3.4).
That is what lets Phase 8 add EWMA and pheromone without touching this file, and
what keeps the traffic model out of any path the arbitration layer depends on.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field

from core import config
from core.graph import Graph

CostFn = Callable[[int], int]
"""Edge id -> traversal cost in milliseconds."""


class Unreachable(Exception):
    """No traversable route exists (FR-3.7).

    Raised only by ``route_or_raise``. The ordinary ``route`` returns None,
    because an unreachable goal is a normal operating condition on a warehouse
    floor with blocked aisles -- not an exceptional one.
    """


@dataclass
class PlanStats:
    """What the last search cost. Compared against FR-3.5's 50 ms budget."""

    expansions: int = 0
    """Nodes popped from the open set. The machine-independent measure of work --
    wall-clock timings vary with the host, this does not."""

    frontier_peak: int = 0
    searches: int = 0
    unreachable: int = 0


@dataclass
class AStarPlanner:
    """A* over a warehouse graph, satisfying ``core.robot.RoutePlanner``.

    One instance per robot. It holds no state that affects a result -- only
    counters -- but keeping instances separate means no robot can observe
    another's planning, which is a property worth having structurally rather than
    by inspection.
    """

    graph: Graph
    cost: CostFn | None = None
    """Composed edge cost (FR-2.2). Defaults to the map's nominal costs, which is
    what a robot uses before its traffic model has learned anything (FR-2.7)."""

    stats: PlanStats = field(default_factory=PlanStats)

    def __post_init__(self) -> None:
        if self.cost is None:
            self.cost = self.graph.nominal_cost_ms

    # -- the search ----------------------------------------------------------

    def route(self, start: int, goal: int) -> list[int] | None:
        """Least-cost node sequence from ``start`` to ``goal``, or None.

        Returns ``[start]`` when already at the goal, rather than an empty list:
        a route is a sequence of places to be, and the robot is already at one.
        """
        self.stats.searches += 1
        self.graph.node(start)
        self.graph.node(goal)

        if start == goal:
            return [start]

        cost_of = self.cost
        assert cost_of is not None  # set in __post_init__

        # (f, insertion_order, node). The counter is the tie-break: equal-cost
        # routes are common on a grid, and without it heapq would order ties by
        # whatever was pushed first, making the choice an accident of iteration.
        counter = 0
        open_set: list[tuple[int, int, int]] = [
            (self.graph.straight_line_ms(start, goal), counter, start)
        ]
        g_score: dict[int, int] = {start: 0}
        came_from: dict[int, int] = {}
        closed: set[int] = set()
        expansions = 0
        peak = 1

        while open_set:
            _, _, node = heapq.heappop(open_set)
            if node in closed:
                continue  # a stale entry superseded by a cheaper path
            if node == goal:
                self.stats.expansions += expansions
                self.stats.frontier_peak = max(self.stats.frontier_peak, peak)
                return _reconstruct(came_from, start, goal)

            closed.add(node)
            expansions += 1
            node_g = g_score[node]

            for neighbour, edge_id in self.graph.neighbours(node):
                if neighbour in closed:
                    continue
                step = cost_of(edge_id)
                if step >= config.INFINITE_COST:
                    # A hard block from the reservation layer (FE-2). Skipped
                    # rather than added at huge cost, so FR-3.9's "shall not be
                    # planned through" is structural, not a matter of degree.
                    continue
                tentative = node_g + step
                if tentative >= g_score.get(neighbour, 1 << 62):
                    continue
                g_score[neighbour] = tentative
                came_from[neighbour] = node
                counter += 1
                heapq.heappush(
                    open_set,
                    (
                        tentative + self.graph.straight_line_ms(neighbour, goal),
                        counter,
                        neighbour,
                    ),
                )
            peak = max(peak, len(open_set))

        self.stats.expansions += expansions
        self.stats.unreachable += 1
        return None

    def route_or_raise(self, start: int, goal: int) -> list[int]:
        """As ``route``, but raises ``Unreachable`` instead of returning None."""
        found = self.route(start, goal)
        if found is None:
            raise Unreachable(
                f"no traversable route from {start} to {goal} on map "
                f"{self.graph.name!r} (FR-3.7)"
            )
        return found

    # -- derived queries -----------------------------------------------------

    def route_cost(self, route: list[int]) -> int:
        """Cost of an explicit route under the current cost function.

        The auction needs this to price a committed plan with and without a task
        inserted (FR-4.3), which is the definition of insertion cost.
        """
        assert self.cost is not None
        total = 0
        for index in range(len(route) - 1):
            edge_id = self.graph.edge_between(route[index], route[index + 1])
            if edge_id is None:
                return config.INFINITE_COST
            total += self.cost(edge_id)
        return total

    def travel_cost(self, start: int, goal: int) -> int:
        """Least cost between two nodes, or ``INFINITE_COST`` if unreachable.

        Used by the auction, which cares about the price of a journey rather than
        its shape and should not have to discard a route to find out.
        """
        found = self.route(start, goal)
        return config.INFINITE_COST if found is None else self.route_cost(found)

    def route_edges(self, route: list[int]) -> list[int]:
        edges = []
        for index in range(len(route) - 1):
            edge_id = self.graph.edge_between(route[index], route[index + 1])
            if edge_id is not None:
                edges.append(edge_id)
        return edges


def _reconstruct(came_from: dict[int, int], start: int, goal: int) -> list[int]:
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path
