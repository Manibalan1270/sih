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
    pick_every: int = 2,
    pick_on: tuple[int, ...] = (0,),
) -> dict:
    """Build a warehouse grid map as a JSON-ready dict.

    Nodes sit at aisle intersections. Horizontal edges are cross-aisles, vertical
    edges are pick-aisles, and **every one of them is a single lane, driven either
    way**: an exclusive corridor (``single_lane``), one robot at a time from one
    junction to the next. No aisle is wide enough for two AMRs abreast, and none
    has a direction -- a robot takes the shortest way and waits its turn at the
    corridor's mouth when someone is coming the other way. ``core.timewindows``
    books each junction-to-junction stretch as one resource, which is what makes a
    head-on impossible: two robots never hold the same stretch at once.

    Consignments are collected in the aisle itself. A pickup is a plain node at the
    midpoint of an interior pick-aisle segment, beside the shelving it is collected
    from -- no spur, no alcove, nothing cut into a rack. The segments chosen are
    those with ``(row + col) % pick_every`` in ``pick_on``: a diagonal spread, so no
    stretch of the floor is all pickups and none is without. A robot collecting
    there holds its stretch of aisle while it does, and traffic for that stretch
    goes round by the next pick-aisle or waits at the junction.
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

    def add_edge(u: int, v: int, *, aisle: bool = True) -> None:
        """An aisle is a two-way single-lane corridor; a spur is a station's own."""
        entry: dict[str, object] = {"id": len(edges), "u": u, "v": v}
        if aisle:
            entry["single_lane"] = True
        edges.append(entry)

    # Perimeter edges are split at their midpoint. Every station and every bay must be
    # its own degree-1 leaf off its own anchor -- a bay chained behind another puts one
    # robot on another's only way out, which is the parking half of well-formedness
    # violated -- and the bare perimeter has too few nodes: 32 on this grid for 30 bays
    # and 12 stations, 68 on the 24x12 grid for 100 and 24. A midpoint on each perimeter
    # edge roughly doubles the ring. Spurs from neighbouring anchors are then 2000 mm
    # (top and bottom) or 2500 mm (sides) apart, clear of the lane-overlap floor.
    mid_of: dict[tuple[int, int], int] = {}

    def add_mid(a: int, b: int, name: str | None = None) -> int:
        na, nb = nodes[a], nodes[b]
        mid = {
            "id": len(nodes),
            "name": name or f"M{a:03d}_{b:03d}",
            "x": (na["x"] + nb["x"]) // 2,
            "y": (na["y"] + nb["y"]) // 2,
            "marker": True,
            "junction": False,  # degree 2 until a bay hangs off it
            "zone": na["zone"],
        }
        nodes.append(mid)
        mid_of[(a, b)] = mid["id"]
        return mid["id"]

    def add_aisle(u: int, v: int, *, split: bool, name: str | None = None) -> int | None:
        """One aisle from ``u`` to ``v``; split, two segments either side of a
        midpoint, which is returned. A midpoint is a bend in the corridor, or a
        pickup on it -- not a junction, unless a spur is hung off it later."""
        if not split:
            add_edge(u, v)
            return None
        m = add_mid(u, v, name)
        add_edge(u, m)
        add_edge(m, v)
        return m

    for row in range(rows):  # cross-aisles
        for col in range(cols - 1):
            add_aisle(node_id(col, row), node_id(col + 1, row), split=row in (0, rows - 1))
    # Pickups sit at interior pick-aisle midpoints; one segment, one pickup.
    pick_segments = sorted(
        (col, row)
        for col in range(1, cols - 1)
        for row in range(rows - 1)
        if (row + col) % pick_every in pick_on
    )
    pickups: list[int] = []
    for col in range(cols):  # pick-aisles
        for row in range(rows - 1):
            picking = (col, row) in pick_segments
            mid = add_aisle(
                node_id(col, row), node_id(col, row + 1),
                split=col in (0, cols - 1) or picking,
                name=f"PICK{col:02d}{row:02d}" if picking else None,
            )
            if picking:
                assert mid is not None
                pickups.append(mid)

    # ---- drop stations and bays: leaves ---------------------------------------
    #
    # Consignments are collected in the pick-aisles (above) and taken to drop
    # stations off the right of the ring (packing), each a degree-1 spur. No endpoint
    # is a junction -- a robot collecting is parked mid-aisle, never in an
    # intersection (SRS defect 13) -- and between any two endpoints there is a route
    # crossing no third, the well-formedness condition of Ma, Li, Kumar and Koenig
    # (AAMAS 2017): every pick-aisle segment has a parallel one a column over.
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
        # The station resource admits one robot at a time, spur and berth, so
        # nothing meets on a spur however it is flagged.
        add_edge(anchor, spur["id"], aisle=False)
        return spur["id"]

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
        "single_lane_aisles": True,
        "pickup_nodes": pickups,
        "drop_nodes": drops,
    }


WAREHOUSE_30 = dict(
    name="warehouse_zoned_30",
    description=(
        "Zoned warehouse for the 30-AMR visual scenario. 12x6 grid of aisle "
        "intersections (66 m x 30 m), partitioned into 6 zones as 3 column "
        "bands x 2 row bands. Every aisle is a single lane driven either way, one "
        "AMR at a time from junction to junction, so no two robots ever meet "
        "head-on or pass abreast. Consignments are collected in the aisle itself, "
        "at a pickup midway along every interior pick-aisle segment (50), and "
        "dropped at stations off the right of the floor. Fits 8-bit node and edge ids."
    ),
    cols=12,
    rows=6,
    col_bands=3,
    row_bands=2,
    bays=30,
    pick_every=1,
    pick_on=(0,),
)

WAREHOUSE_100 = dict(
    name="warehouse_zoned_100",
    description=(
        "Zoned warehouse for the 100-AMR scale scenario. 24x12 grid of aisle "
        "intersections (138 m x 66 m), partitioned into 24 zones as 6 column "
        "bands x 4 row bands. Every aisle is a single lane driven either way, as "
        "on the 30-AMR floor, with consignments collected in the aisle at 121 "
        "pickups along the pick-aisles. Requires 16-bit ids (its "
        "edges exceed the 8-bit ceiling of 255). Zone count is set by the NFR-1.12 "
        "frame budget under FR-9.3 adjacent-zone eligibility -- see "
        "ZONE_BUDGET_NOTE in scripts/generate_maps.py."
    ),
    cols=24,
    rows=12,
    col_bands=6,
    row_bands=4,
    bays=100,
    pick_every=2,
    pick_on=(0,),
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
        f"{len(graph.single_lane_edges):3d} single-lane  {len(spurs):3d} spurs  "
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
