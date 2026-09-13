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

COL_SPACING_MM = 4000
ROW_SPACING_MM = 5000

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
    single_lane_cols: tuple[int, ...] = (),
    bays: int = 0,
) -> dict:
    """Build a warehouse grid map as a JSON-ready dict.

    Nodes sit at aisle intersections. Horizontal edges are cross-aisles,
    vertical edges are pick-aisles. Columns listed in ``single_lane_cols`` have
    their vertical segments marked single-lane, which creates the genuine
    contention points arbitration has to resolve -- a grid with no narrow aisle
    is a grid with no interesting conflicts.
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

    def add_edge(u: int, v: int, *, single_lane: bool = False) -> None:
        entry: dict[str, object] = {"id": len(edges), "u": u, "v": v}
        if single_lane:
            entry["single_lane"] = True
        edges.append(entry)

    for row in range(rows):  # cross-aisles
        for col in range(cols - 1):
            add_edge(node_id(col, row), node_id(col + 1, row))
    for col in range(cols):  # pick-aisles
        for row in range(rows - 1):
            add_edge(
                node_id(col, row),
                node_id(col, row + 1),
                single_lane=col in single_lane_cols,
            )

    # Pickups on the left half, drops on the right half, so generated tasks
    # traverse the floor and actually overlap rather than staying local.
    pickups = [n["id"] for n in nodes if n["id"] % cols < cols // 3]
    drops = [n["id"] for n in nodes if n["id"] % cols >= cols - cols // 3]

    # ---- staging bays --------------------------------------------------------
    # One per robot, and none of them a task endpoint.
    #
    # An idle AMR standing on a junction is a permanent obstacle: peers stop at their
    # following distance and wait for a robot with no reason to move. So idle robots
    # withdraw to a bay -- but that only helps if there are enough bays and none of
    # them is somewhere a task sends a robot. With too few, the queue for a bay blocks
    # the aisle instead; with a bay on a pickup node, the jam simply relocates. Both
    # were observed before this existed.
    #
    # Bays hang off perimeter nodes and sit outside the floor, which is where a real
    # warehouse puts its charging hall.
    perimeter = [
        node_id(col, row)
        for row in range(rows)
        for col in range(cols)
        if col in (0, cols - 1) or row in (0, rows - 1)
    ]
    endpoints = set(pickups) | set(drops)
    anchors = [n for n in perimeter if n not in endpoints] or perimeter
    for index in range(bays):
        anchor = anchors[index % len(anchors)]
        anchor_node = nodes[anchor]
        depth = 1 + index // len(anchors)
        outward = -1 if anchor_node["x"] == 0 else 1
        bay_id = len(nodes)
        nodes.append(
            {
                "id": bay_id,
                "name": f"BAY{index:03d}",
                "x": anchor_node["x"] + outward * depth * (COL_SPACING_MM // 2),
                "y": anchor_node["y"],
                "junction": False,
                "marker": True,
                "parking": True,
                "charger": True,
                "zone": anchor_node["zone"],
            }
        )
        add_edge(anchor, bay_id)

    return {
        "name": name,
        "description": description,
        "grid": {"cols": cols, "rows": rows},
        "zone_bands": {"cols": col_bands, "rows": row_bands},
        "nodes": nodes,
        "edges": edges,
        "single_lane_cols": list(single_lane_cols),
        "pickup_nodes": pickups,
        "drop_nodes": drops,
    }


WAREHOUSE_30 = dict(
    name="warehouse_zoned_30",
    description=(
        "Zoned warehouse for the 30-AMR visual scenario. 12x6 grid of aisle "
        "intersections (44 m x 25 m), partitioned into 6 zones as 3 column "
        "bands x 2 row bands. Columns 4 and 7 are single-lane pick-aisles, so "
        "there are two interior chokepoints rather than one, and robots have a "
        "real choice of which to contend for. Fits 8-bit node and edge ids."
    ),
    cols=12,
    rows=6,
    col_bands=3,
    row_bands=2,
    single_lane_cols=(4, 7),
    bays=30,
)

WAREHOUSE_100 = dict(
    name="warehouse_zoned_100",
    description=(
        "Zoned warehouse for the 100-AMR scale scenario. 24x12 grid of aisle "
        "intersections (92 m x 55 m), partitioned into 24 zones as 6 column "
        "bands x 4 row bands. Requires 16-bit ids (540 edges exceeds the "
        "8-bit ceiling of 255). Zone count is set by the NFR-1.12 frame budget "
        "under FR-9.3 adjacent-zone eligibility -- see ZONE_BUDGET_NOTE in "
        "scripts/generate_maps.py."
    ),
    cols=24,
    rows=12,
    col_bands=6,
    row_bands=4,
    single_lane_cols=(5, 11, 17),
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
    print(
        f"{data['name']:22s} {len(graph.nodes):3d} nodes  {len(graph.edges):3d} edges  "
        f"{len(graph.single_lane_edges):3d} single-lane  {zones.zone_count:2d} zones  "
        f"ids={'16' if wide else '8':>2s}b"
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
