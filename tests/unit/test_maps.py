"""Structural requirements the shipped maps must satisfy.

These are not tests of code -- they are tests of *data*, and they exist because a
map that looks plausible can still make the section 6.3 experiment meaningless.
FR-10.8 and FR-10.9 put real constraints on the benchmark map: without a choke
corridor that routes actually prefer, Configuration A and Configuration B would
barely differ and the 20% claim would have nothing to measure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import scenarios
from core.graph import Graph
from tests.conftest import ALL_MAP_NAMES, MAPS_DIR, reference_shortest_cost


def _raw(name: str) -> dict:
    return json.loads((MAPS_DIR / f"{name}.json").read_text(encoding="utf-8"))


class TestBenchmarkMapStructure:
    """FR-10.8: one single-lane choke corridor and >=2 longer bypasses."""

    def test_has_exactly_one_declared_choke_corridor(self) -> None:
        raw = _raw("benchmark_map")
        graph = Graph.from_dict(raw)
        choke = raw["choke_edge"]
        assert graph.edge(choke).single_lane, "the choke corridor must be single-lane"
        assert graph.single_lane_edges == (choke,)

    def test_at_least_two_longer_alternatives_bypass_the_choke(self) -> None:
        import networkx as nx

        from tests.conftest import to_networkx

        raw = _raw("benchmark_map")
        graph = Graph.from_dict(raw)
        choke = raw["choke_edge"]
        west, east = graph.edge(choke).u, graph.edge(choke).v

        # Junctions flanking the corridor, i.e. the real decision points.
        entry = next(n for n, e in graph.neighbours(west) if e != choke)
        exit_ = next(n for n, e in graph.neighbours(east) if e != choke)

        paths = list(nx.all_simple_paths(to_networkx(graph), entry, exit_))

        def uses_choke(path: list[int]) -> bool:
            return any(
                graph.edge_between(path[i], path[i + 1]) == choke
                for i in range(len(path) - 1)
            )

        def cost(path: list[int]) -> int:
            return sum(
                graph.nominal_cost_ms(graph.edge_between(path[i], path[i + 1]))
                for i in range(len(path) - 1)
            )

        through = [p for p in paths if uses_choke(p)]
        around = [p for p in paths if not uses_choke(p)]

        assert len(through) == 1, "there should be exactly one route via the choke"
        assert len(around) >= 2, (
            f"FR-10.8 requires at least two bypasses; found {len(around)}"
        )
        choke_cost = cost(through[0])
        for path in around:
            assert cost(path) > choke_cost, (
                "a bypass is not longer than the choke route, so nothing forces "
                "contention for the corridor"
            )

    def test_most_cross_warehouse_tasks_prefer_the_choke(self) -> None:
        """FR-10.9 requires a task set where at least 60% of tasks route through
        the choke corridor. The generator can only achieve that if the map's
        candidate pickup/drop pairs mostly prefer it in the first place, so this
        measures the headroom the generator has to work with.

        Note it is 60%, not 100%: the two corner-to-corner pairs (top-to-top and
        bottom-to-bottom) are an exact cost tie between the choke and their own
        bypass, which is a property of the map's deliberate symmetry rather than
        a defect. What must never happen is a pair for which the bypass is
        strictly cheaper -- that would mean the corridor is not a chokepoint.
        """
        raw = _raw("benchmark_map")
        graph = Graph.from_dict(raw)
        choke = raw["choke_edge"]

        prefer_choke = 0
        pairs = 0
        for pickup in raw["pickup_nodes"]:
            for drop in raw["drop_nodes"]:
                pairs += 1
                via_choke = reference_shortest_cost(graph, pickup, drop)
                graph.block(choke)
                via_bypass = reference_shortest_cost(graph, pickup, drop)
                graph.unblock(graph.edge(choke).id)
                assert via_bypass >= via_choke, (
                    f"task {pickup}->{drop} is cheaper avoiding the corridor, so "
                    f"the corridor is not a chokepoint"
                )
                if via_bypass > via_choke:
                    prefer_choke += 1

        fraction = prefer_choke / pairs
        assert fraction >= 0.60, (
            f"only {fraction:.0%} of candidate task pairs prefer the choke; "
            f"FR-10.9 needs at least 60% of generated tasks to route through it"
        )

    def test_pickup_and_drop_sets_are_disjoint_and_on_opposite_sides(self) -> None:
        raw = _raw("benchmark_map")
        pickups, drops = set(raw["pickup_nodes"]), set(raw["drop_nodes"])
        assert pickups and drops
        assert not pickups & drops, "a task with pickup == drop is not a journey"

    def test_has_chargers_reachable_from_everywhere(self) -> None:
        """FR-6.7 routes a low-battery AMR to a charger; it must be able to
        get there from wherever it happens to be."""
        graph = Graph.from_dict(_raw("benchmark_map"))
        assert graph.chargers
        for node_id in graph.nodes:
            assert any(
                reference_shortest_cost(graph, node_id, charger) is not None
                for charger in graph.chargers
            )


class TestLoopMapStructure:
    """TC-3 / Appendix C: a genuine three-way cyclic conflict must be possible."""

    def test_contains_a_three_junction_cycle(self, loop_map: Graph) -> None:
        junction_cycle = [0, 1, 2]
        for index, node in enumerate(junction_cycle):
            following = junction_cycle[(index + 1) % 3]
            edge = loop_map.edge_between(node, following)
            assert edge is not None, f"no edge {node}->{following}"
            assert loop_map.edge(edge).single_lane

    def test_each_junction_has_a_spur_and_a_bay(self, loop_map: Graph) -> None:
        """Three task spurs to start from, and three staging bays to idle in.

        The bays are separate from the spurs on purpose: a robot parked on a task
        endpoint blocks the task, which is how the depot jam arose.
        """
        raw = _raw("loop_map")
        spurs = set(raw["pickup_nodes"])
        bays = set(loop_map.parking_nodes)
        assert len(spurs) == 3
        assert len(bays) == 3
        assert not spurs & bays
        for node in spurs | bays:
            assert not loop_map.node(node).is_junction
            assert len(loop_map.neighbours(node)) >= 1

    def test_cyclic_task_set_routes_through_the_triangle(self, loop_map: Graph) -> None:
        """Robots at S0/S1/S2 heading to S1/S2/S0 must contend for the ring."""
        spurs = sorted(n.id for n in loop_map.nodes.values() if not n.is_junction)
        for index, start in enumerate(spurs):
            goal = spurs[(index + 1) % len(spurs)]
            assert reference_shortest_cost(loop_map, start, goal) is not None


class TestStagingBays:
    """Every map needs somewhere idle robots can stand without obstructing anyone.

    Not an SRS requirement, and it should be. Appendix A never says where a robot
    idles, and one left on a junction is a permanent obstacle: peers stop at their
    following distance and wait for a robot that has no reason to move. Two distinct
    deadlocks came from this before bays existed.
    """

    @pytest.mark.parametrize("name", ALL_MAP_NAMES)
    def test_bays_are_never_task_endpoints(self, name: str) -> None:
        """Parking on a pickup node relocates the jam rather than removing it."""
        graph = Graph.load(MAPS_DIR / f"{name}.json")
        raw = _raw(name)
        endpoints = set(raw.get("pickup_nodes", ())) | set(raw.get("drop_nodes", ()))
        assert not set(graph.parking_nodes) & endpoints

    @pytest.mark.parametrize("name", ALL_MAP_NAMES)
    def test_bays_are_not_junctions(self, name: str) -> None:
        graph = Graph.load(MAPS_DIR / f"{name}.json")
        assert graph.parking_nodes
        for bay in graph.parking_nodes:
            assert not graph.node(bay).is_junction

    def test_every_scenario_has_a_bay_per_robot(self) -> None:
        """A bay is a dead-end spur holding one robot, so too few means the queue for
        a bay blocks the aisle instead -- which is the same failure in a new place."""
        for scenario in scenarios.SCENARIOS.values():
            if scenario.simulated_robots == 0:
                continue
            graph = Graph.load(scenario.map_path)
            assert len(graph.parking_nodes) >= scenario.simulated_robots, (
                f"{scenario.name}: {len(graph.parking_nodes)} bays for "
                f"{scenario.simulated_robots} AMRs"
            )


class TestZonedMapStructure:
    @pytest.mark.parametrize("name", ("warehouse_zoned_30", "warehouse_zoned_100"))
    def test_has_single_lane_chokepoints(self, name: str) -> None:
        graph = Graph.load(MAPS_DIR / f"{name}.json")
        assert graph.single_lane_edges, (
            f"{name} has no single-lane aisle, so arbitration has nothing "
            f"interesting to resolve"
        )

    @pytest.mark.parametrize("name", ("warehouse_zoned_30", "warehouse_zoned_100"))
    def test_grid_dimensions_match_the_node_count(self, name: str) -> None:
        """The floor is the grid; bays hang off it and are counted separately."""
        raw = _raw(name)
        grid = raw["grid"]
        bays = [n for n in raw["nodes"] if n.get("parking")]
        assert len(raw["nodes"]) == grid["cols"] * grid["rows"] + len(bays)

    def test_scale_map_needs_16_bit_ids(self) -> None:
        """This is what forces scale100's edge_id_bits override."""
        from core import config

        graph = Graph.load(MAPS_DIR / "warehouse_zoned_100.json")
        assert len(graph.edges) > config.MAX_EDGES_8BIT


class TestEveryMap:
    @pytest.mark.parametrize("name", ALL_MAP_NAMES)
    def test_file_exists_and_parses(self, name: str) -> None:
        path: Path = MAPS_DIR / f"{name}.json"
        assert path.exists(), f"{path} is missing"
        assert Graph.load(path).nodes

    def test_carries_a_description(self, any_map: Graph) -> None:
        """A map whose reasoning is not written down cannot be reviewed."""
        assert len(any_map.description) > 40

    def test_declares_markers_at_every_junction(self, any_map: Graph) -> None:
        """ASM-3: a marker at every junction. Violating it makes odometry drift
        unbounded and every reservation built on an ETA invalid."""
        for node in any_map.nodes.values():
            if node.is_junction:
                assert node.has_marker, f"{any_map.name}: junction {node.id} has no marker"


class TestScenarioMapPairing:
    def test_every_scenario_map_exists_and_fits_its_id_width(self) -> None:
        for scenario in scenarios.SCENARIOS.values():
            graph = Graph.load(scenario.map_path)
            graph.validate(scenario.max_nodes, scenario.max_edges)

    def test_scenario_fleet_fits_its_map(self) -> None:
        """A fleet denser than roughly one robot per two nodes gridlocks, and a
        gridlocked scale demo proves the opposite of what it is for."""
        for scenario in scenarios.SCENARIOS.values():
            graph = Graph.load(scenario.map_path)
            if scenario.simulated_robots == 0:
                continue
            nodes_per_robot = len(graph.nodes) / scenario.simulated_robots
            assert nodes_per_robot >= 2.0, (
                f"{scenario.name}: {scenario.simulated_robots} AMRs on "
                f"{len(graph.nodes)} nodes is {nodes_per_robot:.1f} nodes/robot"
            )


class TestLanesDoNotOverlapWithoutAJunction:
    """A roadmap is only arbitrable where its conflicts are marked.

    Every safety guarantee here resolves conflicts *at nodes*: junction reservations,
    corridor claims and the corner footprint all key off a node two robots share. Two
    lanes that pass through the same ground while sharing no node therefore have no
    resource to contend for, and no amount of correct arbitration can separate the robots
    on them. Annotating a roadmap so that every place lanes can conflict is represented is
    a precondition of deadlock-free lane-based routing, not an optimisation.

    This was not hypothetical. The bay generator offset every bay along +x regardless of
    which side of the floor its anchor sat on, so bays anchored on the top and bottom rows
    marched *along* the cross-aisle instead of away from it. On warehouse_zoned_30 fifteen
    coordinates were occupied by two or more nodes -- one point held four -- and bay edges
    lay directly on top of cross-aisle edges. At 30 AMRs that produced collisions between
    robots whose edges shared no node, which read as a coordination defect and was not one.
    """

    def separation_mm(self, graph, a: int, b: int) -> float:
        import math

        def point_to_segment(p, s, e) -> float:
            sx, sy = s
            ex, ey = e
            px, py = p
            dx, dy = ex - sx, ey - sy
            length_sq = dx * dx + dy * dy
            if length_sq == 0:
                return math.dist(p, s)
            t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
            return math.dist(p, (sx + t * dx, sy + t * dy))

        def ends(edge_id):
            edge = graph.edge(edge_id)
            u, v = graph.node(edge.u), graph.node(edge.v)
            return (u.x_mm, u.y_mm), (v.x_mm, v.y_mm)

        (a1, a2), (b1, b2) = ends(a), ends(b)
        return min(
            point_to_segment(a1, b1, b2),
            point_to_segment(a2, b1, b2),
            point_to_segment(b1, a1, a2),
            point_to_segment(b2, a1, a2),
        )

    @pytest.mark.parametrize("map_name", ALL_MAP_NAMES)
    def test_no_two_nodes_occupy_the_same_point(self, map_name: str) -> None:
        from core.graph import Graph

        graph = Graph.load(f"maps/{map_name}.json")
        seen: dict[tuple[int, int], int] = {}
        for node_id in graph.nodes:
            node = graph.node(node_id)
            point = (node.x_mm, node.y_mm)
            assert point not in seen, (
                f"{map_name}: nodes {seen[point]} and {node_id} are both at {point}"
            )
            seen[point] = node_id

    @pytest.mark.parametrize("map_name", ALL_MAP_NAMES)
    def test_lanes_sharing_no_node_stay_clear_of_each_other(self, map_name: str) -> None:
        """Wide enough for two robots to pass in opposite lanes without touching.

        Robots keep AISLE_LANE_OFFSET_MM to one side of the centre line, so two lanes
        need that twice over plus the collision distance between them.
        """
        import itertools

        from core import config
        from core.graph import Graph

        graph = Graph.load(f"maps/{map_name}.json")
        required = config.COLLISION_DISTANCE_MM + 2 * config.AISLE_LANE_OFFSET_MM
        for a, b in itertools.combinations(sorted(graph.edges), 2):
            edge_a, edge_b = graph.edge(a), graph.edge(b)
            if {edge_a.u, edge_a.v} & {edge_b.u, edge_b.v}:
                continue  # they meet at a node, which is what arbitration is for
            gap = self.separation_mm(graph, a, b)
            assert gap >= required, (
                f"{map_name}: edges {a} ({edge_a.u}->{edge_a.v}) and {b} "
                f"({edge_b.u}->{edge_b.v}) share no node but pass {gap:.0f} mm apart; "
                f"{required} mm is needed. Robots there cannot be separated by "
                f"arbitration, because they contend for no common node."
            )


class TestWellFormedness:
    """Ma, Li, Kumar & Koenig's solvable class for pickup-and-delivery (AAMAS 2017).

    Two conditions. Enough parking places, none a task endpoint -- defect 6 gave us that
    and ``test_bays_are_never_task_endpoints`` holds it. And between any two endpoints a
    route that crosses no third, so a robot resting at an endpoint is never on someone's
    only way through. That one we violate on every real map, and it is the structural
    reason a robot at a pickup is parked in an intersection.
    """

    @pytest.mark.parametrize(
        "map_name",
        [
            pytest.param(
                name,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "SRS defect 13: task endpoints sit on junctions, so most endpoint "
                        "pairs have no route avoiding a third endpoint. Fixed by the "
                        "Kiva-style maps that make every endpoint a degree-1 spur."
                    ),
                ),
            )
            if name != "loop_map"
            else name
            for name in ALL_MAP_NAMES
        ],
    )
    def test_every_endpoint_pair_has_a_route_avoiding_other_endpoints(
        self, map_name: str
    ) -> None:
        graph = Graph.load(f"maps/{map_name}.json")
        failing = graph.is_well_formed()
        pairs = len(graph.task_endpoints) * (len(graph.task_endpoints) - 1) // 2
        assert not failing, (
            f"{map_name}: {len(failing)} of {pairs} endpoint pairs are not well-formed; "
            f"first: {failing[:5]}"
        )
