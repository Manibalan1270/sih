"""Warehouse topology (FR-3.1).

The warehouse is a topological graph: vertices are junctions, edges are aisle
segments carrying a traversal cost. It is deliberately not an occupancy grid --
section 2.5 of the SRS records the reasoning, and Appendix D quantifies it (a
500x500 grid needs megabytes for the planner alone, against ~3.4 KB here).

This module owns *meaning*. The simulator sees geometry only; that a pair of
coordinates is a junction, that an aisle is single-lane, that a node holds a
charger -- none of that exists outside this file.

Scope boundary: the graph knows nominal costs only. Learned costs, pheromone and
reservation blocks are composed on top by ``core.traffic_model`` and passed into
the planner as a cost function (FR-3.4), so nothing here has to know that
learning exists.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from core import config

NO_ZONE = -1
"""Zone id for maps that are not partitioned (small fleets, section 4.9)."""


class MapError(ValueError):
    """A map file is malformed or violates a structural requirement."""


@dataclass(frozen=True, slots=True)
class Node:
    """A junction, or a pickup/drop/charge point sitting on the topology."""

    id: int
    name: str
    x_mm: int
    y_mm: int
    is_junction: bool = True
    """Whether coordination happens here. Non-junction nodes are stubs such as
    a depot spur, where no arbitration is needed because nothing crosses."""

    has_marker: bool = False
    """ASM-3 / FR-7.2: a machine-readable ground marker is installed here, so a
    robot crossing it re-zeroes its position estimate."""

    is_charger: bool = False
    """FR-6.7: a low-battery AMR routes here. Docking itself is out of scope
    for release 1.0 (OI-5)."""

    is_parking: bool = False
    """A staging bay: somewhere an idle AMR can stand without obstructing anyone.

    Not in the SRS, and the omission has teeth. Appendix A never says where a robot
    idles, and an idle robot on a junction is a permanent obstacle to every peer that
    needs it. Parking bays are how real warehouses solve that, and they must be
    distinct from task endpoints -- parking on a pickup node simply moves the jam."""

    zone_id: int = NO_ZONE


@dataclass(frozen=True, slots=True)
class Edge:
    """An aisle segment between two junctions."""

    id: int
    u: int
    v: int
    nominal_cost_ms: int
    """Traversal time with an empty aisle. The base cost the learned model
    replaces once it has samples (FE-2, 'Learned edge time' layer)."""

    length_mm: int
    single_lane: bool = False
    """ASM-5 / FR-5.10: only one AMR may occupy this segment at a time, so a
    yield decision must be taken at the last passing point before entry."""

    bidirectional: bool = True
    """False makes the edge traversable u->v only.

    A one-way aisle is single-lane in the physical sense -- one file of traffic on
    the centre line -- without being capacity one: robots still follow each other
    nose to tail. That distinction is ``single_file`` below."""

    @property
    def single_file(self) -> bool:
        """Whether this aisle carries one file of traffic rather than two abreast.

        True for an exclusive corridor and for a one-way aisle, and the two get there
        differently: a corridor admits one robot at a time in either direction, a
        one-way aisle admits many but all travelling the same way. Either way nothing
        ever passes anything head-on, so there is no second lane to keep to and the
        robot drives the aisle's centre line (``Robot.footprint_mm``).

        Only a two-way aisle wide enough for two AMRs abreast (ASM-5) is not single
        file, and after the one-way conversion of the maps no aisle is."""
        return self.single_lane or not self.bidirectional

    def other_end(self, node_id: int) -> int:
        if node_id == self.u:
            return self.v
        if node_id == self.v:
            return self.u
        raise KeyError(f"node {node_id} is not an endpoint of edge {self.id}")


@dataclass
class Graph:
    """A warehouse map, queryable by the planner and the auction.

    Mutable in exactly one respect: edge passability (FR-6.1). Everything else
    is fixed at load, per ASM-1 -- the topology is known in advance and static;
    only traversability changes at run time.
    """

    name: str
    nodes: dict[int, Node]
    edges: dict[int, Edge]
    description: str = ""
    _adjacency: dict[int, tuple[tuple[int, int], ...]] = field(
        default_factory=dict, repr=False
    )
    _pair_index: dict[tuple[int, int], int] = field(default_factory=dict, repr=False)
    _blocked: set[int] = field(default_factory=set, repr=False)
    _degree: dict[int, int] = field(default_factory=dict, repr=False)
    """Undirected degree per node, cached at load. See ``degree``."""

    _spur_edges: frozenset[int] = field(default_factory=frozenset, repr=False)
    """Edges into a dead end, cached at load: the sensor model and the collision
    geometry ask about them every tick."""

    _axis_aligned: bool = field(default=False, repr=False)
    """Whether every edge runs purely horizontally or vertically. Determines which
    geometric lower bound ``straight_line_ms`` may use; see its docstring."""

    pickup_nodes: tuple[int, ...] = ()
    drop_nodes: tuple[int, ...] = ()
    """Where tasks begin and end. Carried on the graph rather than only consumed by
    the task generator, because where a robot must *stop* is a structural property of
    the map -- see ``is_well_formed``."""

    # -- construction --------------------------------------------------------

    def __post_init__(self) -> None:
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        adjacency: dict[int, list[tuple[int, int]]] = {n: [] for n in self.nodes}
        self._pair_index = {}
        for edge in self.edges.values():
            adjacency[edge.u].append((edge.v, edge.id))
            self._pair_index[(edge.u, edge.v)] = edge.id
            if edge.bidirectional:
                adjacency[edge.v].append((edge.u, edge.id))
                self._pair_index[(edge.v, edge.u)] = edge.id
        # Sorted so neighbour iteration order is a property of the map rather
        # than of dict insertion. A* tie-breaking depends on it, and FR-5.8
        # requires identical inputs to give identical decisions.
        self._adjacency = {
            node: tuple(sorted(neighbours)) for node, neighbours in adjacency.items()
        }
        self._axis_aligned = all(
            self.nodes[edge.u].x_mm == self.nodes[edge.v].x_mm
            or self.nodes[edge.u].y_mm == self.nodes[edge.v].y_mm
            for edge in self.edges.values()
        )
        degree: dict[int, int] = {n: 0 for n in self.nodes}
        for edge in self.edges.values():
            degree[edge.u] += 1
            degree[edge.v] += 1
        self._degree = degree
        self._spur_edges = frozenset(
            edge.id
            for edge in self.edges.values()
            if degree[edge.u] == 1 or degree[edge.v] == 1
        )

    # -- queries -------------------------------------------------------------

    def node(self, node_id: int) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"no node {node_id} in map {self.name!r}") from None

    def edge(self, edge_id: int) -> Edge:
        try:
            return self.edges[edge_id]
        except KeyError:
            raise KeyError(f"no edge {edge_id} in map {self.name!r}") from None

    def neighbours(self, node_id: int) -> tuple[tuple[int, int], ...]:
        """``((neighbour_id, edge_id), ...)`` reachable from ``node_id``.

        Blocked edges are omitted, so a planner needs no special case for
        blockage -- it simply cannot see the edge (FR-6.1).
        """
        if node_id not in self._adjacency:
            raise KeyError(f"no node {node_id} in map {self.name!r}")
        return tuple(
            (neighbour, edge_id)
            for neighbour, edge_id in self._adjacency[node_id]
            if edge_id not in self._blocked
        )

    def edge_between(self, u: int, v: int) -> int | None:
        """Edge id for the directed step ``u -> v``, or None."""
        return self._pair_index.get((u, v))

    def nominal_cost_ms(self, edge_id: int) -> int:
        return self.edges[edge_id].nominal_cost_ms

    def straight_line_ms(self, a: int, b: int) -> int:
        """Admissible A* heuristic: lower bound on travel time between two nodes.

        Admissible because ``validate`` guarantees no edge costs less than its own
        straight-line time, so no path can beat the geometric bound between its
        endpoints (FR-3.2).

        Which bound is used depends on the map, and the difference is large.
        Euclidean distance is the only safe bound for arbitrary geometry, but on a
        rectilinear warehouse grid it is a *weak* bound -- a robot cannot travel
        the diagonal, it must go along the aisles -- and a weak heuristic
        degenerates A* into Dijkstra. Measured corner to corner on the 288-node
        grid, Euclidean expanded 287 nodes: the search was doing no better than an
        unguided one.

        When every edge is axis-parallel, Manhattan distance is also a valid lower
        bound, and a far tighter one. Each edge then contributes its whole length
        to exactly one axis, so the summed length of any path is at least
        ``|dx| + |dy|`` between its endpoints. ``_axis_aligned`` records whether
        that holds, so mixed maps with diagonal aisles keep the Euclidean bound and
        stay correct.
        """
        na, nb = self.node(a), self.node(b)
        gap_x, gap_y = abs(na.x_mm - nb.x_mm), abs(na.y_mm - nb.y_mm)
        distance_mm = (
            gap_x + gap_y if self._axis_aligned else math.dist((0, 0), (gap_x, gap_y))
        )
        return int(distance_mm * 1000) // config.NOMINAL_SPEED_MM_S

    def length_mm(self, edge_id: int) -> int:
        return self.edges[edge_id].length_mm

    # -- run-time blockage (FR-6.1) -----------------------------------------

    def block(self, edge_id: int) -> None:
        """Mark an edge impassable. Idempotent."""
        self.edge(edge_id)
        self._blocked.add(edge_id)

    def unblock(self, edge_id: int) -> None:
        self._blocked.discard(edge_id)

    def is_blocked(self, edge_id: int) -> bool:
        return edge_id in self._blocked

    @property
    def blocked_edges(self) -> frozenset[int]:
        return frozenset(self._blocked)

    def clear_blockages(self) -> None:
        self._blocked.clear()

    # -- structural helpers --------------------------------------------------

    @property
    def junctions(self) -> tuple[int, ...]:
        return tuple(n.id for n in self.nodes.values() if n.is_junction)

    @property
    def chargers(self) -> tuple[int, ...]:
        return tuple(n.id for n in self.nodes.values() if n.is_charger)

    @property
    def parking_nodes(self) -> tuple[int, ...]:
        """Bays first, then chargers, then any node where nothing crosses."""
        bays = tuple(n.id for n in self.nodes.values() if n.is_parking)
        if bays:
            return bays
        chargers = self.chargers
        if chargers:
            return chargers
        return tuple(n.id for n in self.nodes.values() if not n.is_junction)

    @property
    def single_lane_edges(self) -> tuple[int, ...]:
        """Exclusive corridors: capacity one, either direction."""
        return tuple(e.id for e in self.edges.values() if e.single_lane)

    @property
    def one_way_edges(self) -> tuple[int, ...]:
        return tuple(e.id for e in self.edges.values() if not e.bidirectional)

    @property
    def single_file_edges(self) -> tuple[int, ...]:
        """Every aisle that carries one file of traffic -- corridors and one-way
        aisles both. See ``Edge.single_file``."""
        return tuple(e.id for e in self.edges.values() if e.single_file)

    def two_lane_edges(self, exclude: frozenset[int] = frozenset()) -> tuple[int, ...]:
        """Aisles that are neither a corridor nor one-way, so two AMRs may pass on
        them abreast. ``exclude`` drops edges the caller accounts for otherwise --
        station spurs, whose capacity-one resource already admits one robot at a
        time."""
        return tuple(
            e.id
            for e in self.edges.values()
            if not self.is_single_file(e.id) and e.id not in exclude
        )

    def degree(self, node_id: int) -> int:
        """How many aisles meet at ``node_id``, ignoring which way they run.

        Structural degree, not out-degree, and the distinction is load-bearing on a
        one-way map: a crossing of two one-way aisles has two ways out and four
        aisles meeting, and it is a junction with a conflict region either way. Read
        off the directed adjacency instead, every such crossing on the one-way grid
        stopped counting as a junction and stopped being booked at all.
        """
        return self._degree.get(node_id, 0)

    @property
    def station_spur_edges(self) -> tuple[int, ...]:
        """Edges into a dead end: the spur to a pickup, a drop or a bay."""
        return tuple(sorted(self._spur_edges))

    def is_single_file(self, edge_id: int) -> bool:
        """Whether ``edge_id`` carries one file of traffic rather than two abreast.

        The one predicate the collision geometry and the sensor model must agree on,
        which is why it lives here and not at either call site. Three ways to be
        single file: an exclusive corridor, a one-way aisle, and a spur into a dead
        end -- the last structurally rather than by flag, because a spur is two-way
        but ends somewhere only one robot fits. Getting the spur wrong is not
        academic: with the flag alone the sensor model read a robot driving out of a
        station as oncoming traffic safely in another lane, and drove a second robot
        into it on four of five bench3 seeds.
        """
        return edge_id in self._spur_edges or self.edge(edge_id).single_file

    def reachable_from(self, start: int) -> frozenset[int]:
        """Nodes reachable from ``start``, honouring blockages and direction."""
        seen = {start}
        stack = [start]
        while stack:
            for neighbour, _ in self.neighbours(stack.pop()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return frozenset(seen)

    @property
    def task_endpoints(self) -> frozenset[int]:
        return frozenset(self.pickup_nodes) | frozenset(self.drop_nodes)

    def is_well_formed(self) -> list[tuple[int, int]]:
        """Endpoint pairs with no route between them that avoids every other endpoint.

        Empty means the map is *well-formed* in the sense of Ma, Li, Kumar and Koenig
        (Lifelong Multi-Agent Path Finding for Online Pickup and Delivery Tasks, AAMAS
        2017): the solvable class of pickup-and-delivery instances, defined by two
        conditions -- at least as many parking places as robots, none of them a task
        endpoint (defect 6 gave us that), and between any two endpoints a path that
        crosses no third. The second is what makes every task eventually assignable
        with no deadlock: a robot resting at an endpoint never stands on anyone's only
        way through.

        Our maps violate it wholesale. Pickups and drops sit on 3- and 4-way junctions
        -- 47 of 48 endpoints on warehouse_zoned_30, all 192 on warehouse_zoned_100 --
        so a robot dwelling at a pickup is parked in an intersection. That is the one
        place the rule "never come to rest inside a conflict region" cannot be honoured,
        because the task requires the rest. Every collision and cycle at 30 AMRs that
        survived the local rules traces back to it. Recorded as SRS defect 13.

        BFS from each endpoint over the graph with every *other* endpoint removed. Pairs
        are returned in id order so the report is stable.
        """
        endpoints = self.task_endpoints
        failing: list[tuple[int, int]] = []
        for start in sorted(endpoints):
            seen = {start}
            stack = [start]
            while stack:
                for neighbour, _ in self.neighbours(stack.pop()):
                    if neighbour in seen:
                        continue
                    seen.add(neighbour)
                    if neighbour in endpoints:
                        continue  # reached, but do not pass through
                    stack.append(neighbour)
            for goal in sorted(endpoints):
                if goal > start and goal not in seen:
                    failing.append((start, goal))
        return failing

    def is_connected(self) -> bool:
        """Whether every node can reach every other, following edge directions.

        Strong connectivity, not weak: on a one-way map "reachable from node 0" and
        "able to reach node 0" are different questions, and a map that answers only
        the first has a node a robot can drive into and never leave. Checking one
        direction from one node would have passed such a map.
        """
        if not self.nodes:
            return False
        first = next(iter(self.nodes))
        every = frozenset(self.nodes)
        return self.reachable_from(first) == every and self.reaching(first) == every

    def reaching(self, target: int) -> frozenset[int]:
        """Every node from which ``target`` can be reached. The reverse of
        ``reachable_from``, and identical to it on an all-bidirectional map."""
        incoming: dict[int, list[int]] = {n: [] for n in self.nodes}
        for edge in self.edges.values():
            if edge.id in self._blocked:
                continue
            incoming[edge.v].append(edge.u)
            if edge.bidirectional:
                incoming[edge.u].append(edge.v)
        seen = {target}
        stack = [target]
        while stack:
            for previous in incoming[stack.pop()]:
                if previous not in seen:
                    seen.add(previous)
                    stack.append(previous)
        return frozenset(seen)

    def last_passing_point(self, route: list[int], corridor_edge: int) -> int | None:
        """The node on ``route`` where a single-lane yield must be decided.

        FR-5.10 requires the decision to be taken *before* committing to entry,
        because once inside a single-lane corridor neither robot can divert and
        the only remaining outcome is a head-on standoff. The last passing point
        is the corridor's entry junction: the final node from which an
        alternative edge still exists.

        Returns None if the route does not use the corridor.
        """
        edge = self.edge(corridor_edge)
        for index in range(len(route) - 1):
            if self.edge_between(route[index], route[index + 1]) == corridor_edge:
                entry = route[index]
                # An entry node with only the corridor and the way we came is
                # not a passing point -- the real decision was one node earlier.
                if len(self._adjacency[entry]) <= 2 and index > 0:
                    return route[index - 1]
                return entry
        del edge
        return None

    # -- validation ----------------------------------------------------------

    def validate(self, max_nodes: int, max_edges: int) -> None:
        """Check structural requirements. Raises ``MapError`` on failure.

        ``max_nodes`` and ``max_edges`` come from the scenario's edge-id width
        (ASM-2, CON-10), so the same map can be legal for scale100 and illegal
        for bench3.
        """
        if not self.nodes:
            raise MapError(f"{self.name}: map has no nodes")
        if len(self.nodes) > max_nodes:
            raise MapError(
                f"{self.name}: {len(self.nodes)} nodes exceeds the {max_nodes} "
                f"addressable by this scenario's id width (ASM-2, CON-10)"
            )
        if len(self.edges) > max_edges:
            raise MapError(
                f"{self.name}: {len(self.edges)} edges exceeds the {max_edges} "
                f"addressable by this scenario's id width (ASM-2, CON-10)"
            )
        for node_id, node in self.nodes.items():
            if node_id != node.id:
                raise MapError(f"{self.name}: node {node_id} keyed as {node.id}")
            if not 0 <= node_id < max_nodes:
                raise MapError(f"{self.name}: node id {node_id} out of range")

        seen_pairs: set[frozenset[int]] = set()
        for edge_id, edge in self.edges.items():
            if edge_id != edge.id:
                raise MapError(f"{self.name}: edge {edge_id} keyed as {edge.id}")
            if edge.u not in self.nodes or edge.v not in self.nodes:
                raise MapError(
                    f"{self.name}: edge {edge_id} references a missing node "
                    f"({edge.u} -> {edge.v})"
                )
            if edge.u == edge.v:
                raise MapError(f"{self.name}: edge {edge_id} is a self-loop")
            pair = frozenset((edge.u, edge.v))
            if pair in seen_pairs:
                raise MapError(
                    f"{self.name}: duplicate edge between {edge.u} and {edge.v}"
                )
            seen_pairs.add(pair)
            if edge.nominal_cost_ms <= 0:
                raise MapError(
                    f"{self.name}: edge {edge_id} has non-positive cost "
                    f"{edge.nominal_cost_ms}"
                )
            floor = self.straight_line_ms(edge.u, edge.v)
            if edge.nominal_cost_ms < floor:
                # Without this the straight-line heuristic overestimates and A*
                # stops being admissible, so FR-3.2's least-cost guarantee is
                # lost silently -- routes would still be returned, just not
                # optimal ones.
                raise MapError(
                    f"{self.name}: edge {edge_id} cost {edge.nominal_cost_ms} ms "
                    f"is below its straight-line time {floor} ms, which would "
                    f"make the A* heuristic inadmissible (FR-3.2)"
                )

        if not self.is_connected():
            unreachable = sorted(set(self.nodes) - self.reachable_from(next(iter(self.nodes))))
            raise MapError(
                f"{self.name}: map is not connected; unreachable from node "
                f"{next(iter(self.nodes))}: {unreachable}"
            )

    # -- loading -------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        """Build a graph from parsed map JSON.

        ``cost_ms`` may be omitted on an edge, in which case it is derived from
        the straight-line distance at cruise speed. Deriving it is preferred for
        hand-authored maps: a typo in a cost that happens to fall below the
        straight-line time would break heuristic admissibility, and letting the
        loader compute it makes that impossible.
        """
        nodes: dict[int, Node] = {}
        for raw in data.get("nodes", ()):
            node = Node(
                id=int(raw["id"]),
                name=str(raw.get("name", f"n{raw['id']}")),
                x_mm=int(raw["x"]),
                y_mm=int(raw["y"]),
                is_junction=bool(raw.get("junction", True)),
                has_marker=bool(raw.get("marker", raw.get("junction", True))),
                is_charger=bool(raw.get("charger", False)),
                is_parking=bool(raw.get("parking", False)),
                zone_id=int(raw.get("zone", NO_ZONE)),
            )
            if node.id in nodes:
                raise MapError(f"duplicate node id {node.id}")
            nodes[node.id] = node

        edges: dict[int, Edge] = {}
        for raw in data.get("edges", ()):
            u, v = int(raw["u"]), int(raw["v"])
            if u not in nodes or v not in nodes:
                raise MapError(f"edge {raw.get('id')} references a missing node")
            length_mm = int(
                round(
                    math.dist(
                        (nodes[u].x_mm, nodes[u].y_mm), (nodes[v].x_mm, nodes[v].y_mm)
                    )
                )
            )
            derived_cost = (length_mm * 1000) // config.NOMINAL_SPEED_MM_S
            edge = Edge(
                id=int(raw["id"]),
                u=u,
                v=v,
                nominal_cost_ms=int(raw.get("cost_ms", derived_cost)),
                length_mm=length_mm,
                single_lane=bool(raw.get("single_lane", False)),
                bidirectional=bool(raw.get("bidirectional", True)),
            )
            if edge.id in edges:
                raise MapError(f"duplicate edge id {edge.id}")
            edges[edge.id] = edge

        return cls(
            name=str(data.get("name", "unnamed")),
            description=str(data.get("description", "")),
            nodes=nodes,
            edges=edges,
            pickup_nodes=tuple(int(n) for n in data.get("pickup_nodes", ())),
            drop_nodes=tuple(int(n) for n in data.get("drop_nodes", ())),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Graph":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise MapError(f"no map file at {path}") from None
        except json.JSONDecodeError as exc:
            raise MapError(f"{path} is not valid JSON: {exc}") from None
        return cls.from_dict(data)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"Graph({self.name!r}, {len(self.nodes)} nodes, {len(self.edges)} edges, "
            f"{len(self.single_lane_edges)} corridors, "
            f"{len(self.one_way_edges)} one-way, "
            f"{len(self._blocked)} blocked)"
        )
