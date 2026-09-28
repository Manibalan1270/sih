"""Generate the large zoned warehouse maps (FR-9.1, FR-10.8).

The benchmark and loop maps are hand-authored -- they are small and every node
in them is load-bearing for a specific test case. The fleet-scale maps are not:
they are regular grids of cross-aisles and pick-aisles, and hand-authoring 240
nodes would only introduce typos.

Run this when the layout parameters change; commit the JSON it writes.

    py scripts/generate_maps.py

Zone counts are chosen so that measured auction traffic meets NFR-1.12, and that
choice is not arbitrary -- see ``ZONE_BUDGET_NOTE`` below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import config  # noqa: E402
from core.graph import Graph  # noqa: E402
from core.zones import ZoneMap  # noqa: E402

MAPS_DIR = REPO_ROOT / "maps"

COL_SPACING_MM = 6000
"""Equal to the row spacing on purpose. Perimeter edges are split at their midpoints to
give every station and bay its own anchor, and a midpoint with a bay is a junction whose
conflict region extends JUNCTION_FOOTPRINT_MM (1200) along the aisle. Two such regions
must not overlap or they would have to merge into one capacity-1 resource -- and with
bays on most of the ring that would chain whole rows together. At 4000 the midpoints sat
2000 mm from the grid junctions and 24 region pairs overlapped on the 30-map, 88 on the
100-map. At 5000 they sat 2500 apart, 100 mm clear -- but a robot holding for its turn
stops HOLD_LINE_MM short of the next region, and 100 mm of lane put that stop inside
the region behind it, where a robot turning through the neighbouring junction in its
700 mm lane passed 495 mm from it. At 6000 the midpoints sit 3000 apart: 600 mm of lane,
room for the hold line outside both regions (core.graph.MIN_JUNCTION_SPACING_MM)."""
ROW_SPACING_MM = 6000

ZONE_BUDGET_NOTE = """
The SRS is internally inconsistent about auction traffic at fleet scale, and the
zone count here is what reconciles it.

Section 4.9 states a flood auction costs ~2N frames per task, and that a zoned
auction at 100 AMRs costs ~34 frames per task. NFR-1.12 then budgets <=40 frames
per task "at 100 AMRs across 6 zones". But 34 frames means ~17 eligible bidders,
which is 100/6 -- the task's own zone only. FR-9.3 and FR-4.14 require robots in
*adjacent* zones to bid too, which on a 6-zone partition raises eligibility to
~55 bidders and the frame count to ~110, nearly triple the budget.

Both cannot hold. Rather than quietly dropping adjacent-zone eligibility (which
would distort allocation quality, since a robot just over a boundary may hold the
lowest insertion cost) this map uses 24 zones. With adjacent-zone eligibility
honoured that yields ~35 frames per task -- inside NFR-1.12's budget and
matching section 4.9's own stated figure. The "6 zones" in NFR-1.12 is the part
that does not survive contact with FR-9.3, and should be corrected in SRS v1.1.
""".strip()


def build_grid(
    *,
    name: str,
    description: str,
    cols: int,
    rows: int,
    col_bands: int,
    row_bands: int,
    bays: int = 0,
) -> dict:
    """Build a warehouse grid map as a JSON-ready dict.

    Nodes sit at aisle intersections. Horizontal edges are cross-aisles, vertical
    edges are pick-aisles, and **every one of them is a single lane travelled in one
    direction only** -- the layout Kiva-style floors actually use.

    The orientation is not a free choice; a one-way grid is easy to make unreachable.
    Alternating every row and every column (the textbook Manhattan grid) puts a sink
    at one corner and a source at the opposite one: at ``(cols-1, 0)`` an eastbound
    top row and a southbound last column both point off the floor, so a robot that
    drives in can never drive out. So the perimeter is laid out as a single directed
    cycle -- top row east, last column north, bottom row west, first column south --
    and only the interior aisles alternate. Every interior aisle then begins and ends
    on that cycle, which makes the map strongly connected: leave any node along its
    row or column to reach the ring, and enter any node by joining its column or row
    where that aisle meets the ring. ``Graph.is_connected`` checks this both ways
    round, and ``write_map`` fails the build if it does not hold.

    One-way is what buys collision freedom cheaply. An aisle that cannot carry
    opposing traffic cannot produce a head-on, and the lane resource in
    ``core.timewindows`` is FIFO, so robots on one still follow each other nose to
    tail rather than serialising one at a time the way an exclusive corridor does.
    """
    if cols % col_bands or rows % row_bands:
        raise ValueError(
            f"{name}: {cols}x{rows} grid does not divide evenly into "
            f"{col_bands}x{row_bands} zone bands"
        )

    cols_per_band = cols // col_bands
    rows_per_band = rows // row_bands

    def node_id(col: int, row: int) -> int:
        return row * cols + col

    nodes = []
    for row in range(rows):
        for col in range(cols):
            zone = (row // rows_per_band) * col_bands + (col // cols_per_band)
            # No chargers on the floor itself. Charging happens at the staging bays
            # below, which are deliberately not task endpoints: a robot that finished
            # charging on a pickup node stood on a task endpoint and blocked the robot
            # whose task was there.
            nodes.append(
                {
                    "id": node_id(col, row),
                    "name": f"C{col:02d}R{row:02d}",
                    "x": col * COL_SPACING_MM,
                    "y": row * ROW_SPACING_MM,
                    "marker": True,
                    "zone": zone,
                }
            )

    edges = []

    def add_edge(u: int, v: int, *, one_way: bool = True) -> None:
        """``u -> v`` is the direction of travel when ``one_way``."""
        entry: dict[str, object] = {"id": len(edges), "u": u, "v": v}
        if one_way:
            entry["bidirectional"] = False
        edges.append(entry)

    # Perimeter edges are split at their midpoint. Every station and every bay must be
    # its own degree-1 leaf off its own anchor -- a bay chained behind another puts one
    # robot on another's only way out, which is the parking half of well-formedness
    # violated -- and the bare perimeter has too few nodes: 32 on this grid for 30 bays
    # and 12 stations, 68 on the 24x12 grid for 100 and 24. A midpoint on each perimeter
    # edge roughly doubles the ring. Spurs from neighbouring anchors are then 2000 mm
    # (top and bottom) or 2500 mm (sides) apart, clear of the lane-overlap floor.
    mid_of: dict[tuple[int, int], int] = {}

    def add_mid(a: int, b: int) -> int:
        na, nb = nodes[a], nodes[b]
        mid = {
            "id": len(nodes),
            "name": f"M{a:03d}_{b:03d}",
            "x": (na["x"] + nb["x"]) // 2,
            "y": (na["y"] + nb["y"]) // 2,
            "marker": True,
            "junction": False,  # degree 2 until a bay hangs off it
            "zone": na["zone"],
        }
        nodes.append(mid)
        mid_of[(a, b)] = mid["id"]
        return mid["id"]

    def add_aisle(u: int, v: int, *, split: bool, forward: bool) -> None:
        """One aisle from ``u`` to ``v``, travelled ``u -> v`` when ``forward``.

        A split aisle is two segments either side of a midpoint, and both carry the
        aisle's direction -- a midpoint is a bend in the lane, not a place to turn
        round. The midpoint is always created under the caller's ``(u, v)`` order so
        the ring walk below can find it whichever way the traffic runs.
        """
        if not split:
            add_edge(u, v) if forward else add_edge(v, u)
            return
        m = add_mid(u, v)
        if forward:
            add_edge(u, m)
            add_edge(m, v)
        else:
            add_edge(v, m)
            add_edge(m, u)

    # Perimeter as a directed cycle, interior aisles alternating. See the docstring:
    # alternating the perimeter too would strand the corners.
    def cross_aisle_runs_east(row: int) -> bool:
        if row == 0:
            return True  # top of the ring
        if row == rows - 1:
            return False  # bottom of the ring, returning west
        return row % 2 == 0

    def pick_aisle_runs_up(col: int) -> bool:
        if col == cols - 1:
            return True  # right side of the ring, climbing
        if col == 0:
            return False  # left side, descending
        return col % 2 == 0

    for row in range(rows):  # cross-aisles
        for col in range(cols - 1):
            add_aisle(
                node_id(col, row), node_id(col + 1, row),
                split=row in (0, rows - 1),
                forward=cross_aisle_runs_east(row),
            )
    for col in range(cols):  # pick-aisles
        for row in range(rows - 1):
            add_aisle(
                node_id(col, row), node_id(col, row + 1),
                split=col in (0, cols - 1),
                forward=pick_aisle_runs_up(col),
            )

    # ---- stations and bays: leaves off the perimeter ring ----------------------
    #
    # A Kiva-style floor. The grid is the aisle network; nothing a robot must *stop*
    # for is on it. Pickup stations hang off the left column (the inbound dock), drop
    # stations off the right (packing), each a degree-1 spur pointing away from the
    # floor. That is the well-formedness condition of Ma, Li, Kumar and Koenig (AAMAS
    # 2017): between any two endpoints there is a route crossing no third, because a
    # leaf cannot be crossed. Before this every station was a 3- or 4-way junction and
    # a robot at a pickup was parked in an intersection -- SRS defect 13.
    #
    # Bays take the rest of the ring: top and bottom rows and the side midpoints. One
    # per robot, each its own leaf, chargers here, never a task endpoint (defect 6).
    x_max, y_max = (cols - 1) * COL_SPACING_MM, (rows - 1) * ROW_SPACING_MM

    def outward(node: dict) -> tuple[int, int]:
        if node["x"] == 0:
            return (-(COL_SPACING_MM // 2), 0)
        if node["x"] == x_max:
            return (COL_SPACING_MM // 2, 0)
        if node["y"] == 0:
            return (0, -(ROW_SPACING_MM // 2))
        return (0, ROW_SPACING_MM // 2)

    def add_spur(anchor: int, name: str, **flags) -> int:
        a = nodes[anchor]
        dx, dy = outward(a)
        spur = {
            "id": len(nodes),
            "name": name,
            "x": a["x"] + dx,
            "y": a["y"] + dy,
            "junction": False,
            "marker": True,
            "zone": a["zone"],
            **flags,
        }
        nodes.append(spur)
        a["junction"] = True  # something now turns off here
        # Spurs stay two-way: a dead end has to be left the way it was entered, and
        # the station resource admits one robot at a time so nothing meets there.
        add_edge(anchor, spur["id"], one_way=False)
        return spur["id"]

    pickups = [add_spur(node_id(0, row), f"PICK{row:02d}") for row in range(rows)]
    drops = [add_spur(node_id(cols - 1, row), f"DROP{row:02d}") for row in range(rows)]

    # Bay anchors, walked round the ring so the chosen subset is spread out rather
    # than clustered at one corner: bottom row left to right, right side upward, top
    # row right to left, left side downward. Corners belong to the stations.
    ring: list[int] = []
    for col in range(cols - 1):
        if col > 0:
            ring.append(node_id(col, 0))
        ring.append(mid_of[(node_id(col, 0), node_id(col + 1, 0))])
    for row in range(rows - 1):
        ring.append(mid_of[(node_id(cols - 1, row), node_id(cols - 1, row + 1))])
    for col in range(cols - 1, 0, -1):
        if col < cols - 1:
            ring.append(node_id(col, rows - 1))
        ring.append(mid_of[(node_id(col - 1, rows - 1), node_id(col, rows - 1))])
    for row in range(rows - 1, 0, -1):
        ring.append(mid_of[(node_id(0, row - 1), node_id(0, row))])
    if bays > len(ring):
        raise ValueError(
            f"{name}: {bays} bays requested but the ring has only {len(ring)} anchors"
        )
    for index in range(bays):
        anchor = ring[index * len(ring) // bays]
        add_spur(anchor, f"BAY{index:03d}", parking=True, charger=True)

    return {
        "name": name,
        "description": description,
        "grid": {"cols": cols, "rows": rows},
        "zone_bands": {"cols": col_bands, "rows": row_bands},
        "nodes": nodes,
        "edges": edges,
        "one_way_aisles": True,
        "pickup_nodes": pickups,
        "drop_nodes": drops,
    }


WAREHOUSE_30 = dict(
    name="warehouse_zoned_30",
    description=(
        "Zoned warehouse for the 30-AMR visual scenario. 12x6 grid of aisle "
        "intersections (44 m x 25 m), partitioned into 6 zones as 3 column "
        "bands x 2 row bands. Every aisle is a single lane travelled one way: the "
        "perimeter circulates as a directed ring and the interior rows and columns "
        "alternate, so no two robots ever meet head-on and contention is confined "
        "to the junctions between them. Fits 8-bit node and edge ids."
    ),
    cols=12,
    rows=6,
    col_bands=3,
    row_bands=2,
    bays=30,
)

WAREHOUSE_100 = dict(
    name="warehouse_zoned_100",
    description=(
        "Zoned warehouse for the 100-AMR scale scenario. 24x12 grid of aisle "
        "intersections (138 m x 66 m), partitioned into 24 zones as 6 column "
        "bands x 4 row bands. Every aisle is a single lane travelled one way, on "
        "the same directed-perimeter layout as the 30-AMR floor. Requires 16-bit "
        "ids (540 edges exceeds the 8-bit ceiling of 255). Zone count is set by "
        "the NFR-1.12 frame budget under FR-9.3 adjacent-zone eligibility -- see "
        "ZONE_BUDGET_NOTE in scripts/generate_maps.py."
    ),
    cols=24,
    rows=12,
    col_bands=6,
    row_bands=4,
    bays=100,
)


def write_map(spec: dict) -> Path:
    data = build_grid(**spec)
    path = MAPS_DIR / f"{data['name']}.json"
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")

    graph = Graph.from_dict(data)
    wide = len(graph.edges) > config.MAX_EDGES_8BIT or len(graph.nodes) > config.MAX_NODES_8BIT
    max_nodes = config.MAX_NODES_16BIT if wide else config.MAX_NODES_8BIT
    max_edges = config.MAX_EDGES_16BIT if wide else config.MAX_EDGES_8BIT
    graph.validate(max_nodes, max_edges)

    zones = ZoneMap.from_graph(graph, enabled=True)
    zones.validate(graph)

    fleet = 30 if "30" in data["name"] else 100
    bidders = zones.expected_bidders(fleet)
    spurs = frozenset(graph.station_spur_edges)
    two_lane = graph.two_lane_edges(exclude=spurs)
    if two_lane:
        raise ValueError(
            f"{data['name']}: {len(two_lane)} aisles are neither one-way nor an "
            f"exclusive corridor, so two AMRs could meet abreast on them: {two_lane[:8]}"
        )
    print(
        f"{data['name']:22s} {len(graph.nodes):3d} nodes  {len(graph.edges):3d} edges  "
        f"{len(graph.one_way_edges):3d} one-way  {len(spurs):3d} spurs  "
        f"{zones.zone_count:2d} zones  ids={'16' if wide else '8':>2s}b"
    )
    print(
        f"{'':22s} at {fleet} AMRs: ~{bidders} eligible bidders "
        f"-> ~{2 * bidders} auction frames/task "
        f"(NFR-1.12 budget 40){'  OK' if 2 * bidders <= 40 else '  OVER BUDGET'}"
    )
    return path


def main() -> int:
    MAPS_DIR.mkdir(exist_ok=True)
    for spec in (WAREHOUSE_30, WAREHOUSE_100):
        write_map(spec)
    print()
    print(ZONE_BUDGET_NOTE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
