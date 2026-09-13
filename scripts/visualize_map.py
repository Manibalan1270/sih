"""Plot a warehouse map (Phase 1 deliverable).

The first thing in this project with a visible, checkable output. If the choke
corridor is not obvious in this picture, the benchmark map is wrong and the
section 6.3 experiment will not force the path overlap the success criteria
require -- so this is a verification tool, not decoration.

    py scripts/visualize_map.py maps/benchmark_map.json
    py scripts/visualize_map.py maps/benchmark_map.json --route 0 7
    py scripts/visualize_map.py maps/warehouse_zoned_100.json --out results/

Output is SVG; see scripts/svgplot.py for why that is not matplotlib.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.graph import NO_ZONE, Graph  # noqa: E402
from core.zones import ZoneMap  # noqa: E402
from scripts.svgplot import Canvas  # noqa: E402

ZONE_COLOURS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#8C8C8C", "#2E7EBB", "#D67C2E",
    "#3D8C46", "#C4453F", "#5FA39E", "#CFAE30", "#9B6790", "#DE8C93",
    "#86644F", "#767676", "#27699E", "#B96A25", "#347A3C", "#A83A35",
]

CHOKE_COLOUR = "#D62728"
ROUTE_COLOUR = "#1F77B4"


def least_cost_route(graph: Graph, start: int, goal: int) -> list[int] | None:
    """Nominal-cost route, for overlaying a path on the plot.

    Deliberately a local Dijkstra rather than an import of the real planner: this
    script must stay usable at Phase 1, before core/planner_astar.py exists.
    """
    best: dict[int, int] = {start: 0}
    queue: list[tuple[int, int, list[int]]] = [(0, start, [start])]
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == goal:
            return path
        if cost > best.get(node, 1 << 60):
            continue
        for neighbour, edge_id in graph.neighbours(node):
            new_cost = cost + graph.nominal_cost_ms(edge_id)
            if new_cost < best.get(neighbour, 1 << 60):
                best[neighbour] = new_cost
                heapq.heappush(queue, (new_cost, neighbour, path + [neighbour]))
    return None


def plot(
    graph: Graph,
    *,
    choke_edge: int | None,
    route: list[int] | None,
    out_path: Path,
) -> Path:
    zones = ZoneMap.from_graph(graph, enabled=True)
    big = len(graph.nodes) > 100

    xs = [n.x_mm / 1000 for n in graph.nodes.values()]
    ys = [n.y_mm / 1000 for n in graph.nodes.values()]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    margin = max(span_x, span_y) * 0.04 + 1

    # Size the canvas to the map's aspect ratio so aisles stay square.
    plot_px = 1320 if big else 1120
    aspect = (span_y + 2 * margin) / (span_x + 2 * margin)
    canvas = Canvas(
        width=plot_px,
        height=int(plot_px * aspect) + 130,
        xmin=min(xs) - margin, xmax=max(xs) + margin,
        ymin=min(ys) - margin, ymax=max(ys) + margin,
        pad=70,
    )

    # Edges first, so nodes and the route overlay sit on top of them.
    for edge in graph.edges.values():
        a, b = graph.node(edge.u), graph.node(edge.v)
        ax, ay = a.x_mm / 1000, a.y_mm / 1000
        bx, by = b.x_mm / 1000, b.y_mm / 1000
        if edge.id == choke_edge:
            canvas.line(ax, ay, bx, by, colour=CHOKE_COLOUR, width=7)
        elif edge.single_lane:
            canvas.line(ax, ay, bx, by, colour=CHOKE_COLOUR, width=2.4, dash="7 4")
        else:
            canvas.line(ax, ay, bx, by, colour="#AAAAAA", width=1.3)

    if route and len(route) > 1:
        canvas.polyline(
            [(graph.node(n).x_mm / 1000, graph.node(n).y_mm / 1000) for n in route],
            colour=ROUTE_COLOUR, width=5, opacity=0.5,
        )

    node_size = 4.0 if big else 9.0
    for node in graph.nodes.values():
        zone = zones.zone_of(node.id)
        colour = ZONE_COLOURS[zone % len(ZONE_COLOURS)] if zone != NO_ZONE else "#BBBBBB"
        x, y = node.x_mm / 1000, node.y_mm / 1000
        if node.is_charger:
            canvas.square(x, y, node_size * 1.15, fill=colour)
        elif not node.is_junction:
            canvas.triangle(x, y, node_size * 1.3, fill=colour)
        else:
            canvas.circle(x, y, node_size, fill=colour)
        if not big:
            canvas.text(
                x, y, f"{node.name} #{node.id}",
                size=10, dy=-node_size - 7, colour="#333333",
            )

    # Axes and title in pixel space, outside the data area.
    canvas.text(
        canvas.width / 2, 30, graph.name,
        size=19, weight="bold", anchor="middle", data_space=False,
    )
    canvas.text(
        canvas.width / 2, 52,
        f"{len(graph.nodes)} nodes  |  {len(graph.edges)} edges  |  "
        f"{len(graph.single_lane_edges)} single-lane  |  {zones.zone_count} zones",
        size=12, colour="#666666", anchor="middle", data_space=False,
    )
    canvas.text(
        canvas.width / 2, canvas.height - 16, "metres",
        size=11, colour="#666666", anchor="middle", data_space=False,
    )

    legend = []
    if choke_edge is not None:
        legend.append("thick red  = choke corridor")
    legend += [
        "dashed red = single-lane aisle",
        "square     = charger node",
        "triangle   = non-junction (no arbitration)",
        "fill colour= zone",
    ]
    if route:
        legend.append("blue       = nominal least-cost route")
    canvas.legend_px(84, canvas.height - 132, legend, size=11)

    saved = canvas.save(out_path, title=f"{graph.name} - ROBOTON warehouse map")
    print(f"wrote {saved}")
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot a ROBOTON warehouse map.")
    parser.add_argument("map", type=Path, help="path to a map JSON file")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: results/)")
    parser.add_argument("--route", nargs=2, type=int, metavar=("START", "GOAL"),
                        help="overlay the nominal least-cost route")
    args = parser.parse_args(argv)

    raw = json.loads(args.map.read_text(encoding="utf-8"))
    graph = Graph.from_dict(raw)

    route = None
    if args.route:
        start, goal = args.route
        route = least_cost_route(graph, start, goal)
        if route is None:
            print(f"no route from {start} to {goal}", file=sys.stderr)
            return 1
        print(
            f"route {start} -> {goal}: "
            + " -> ".join(graph.node(n).name for n in route)
        )

    out_dir = args.out or REPO_ROOT / "results"
    plot(
        graph,
        choke_edge=raw.get("choke_edge"),
        route=route,
        out_path=out_dir / f"{graph.name}.svg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
