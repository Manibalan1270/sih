"""Generate a Webots world for a scenario from its map.

    py scripts/generate_world.py visual30            # -> webots/worlds/visual30.wbt
    py scripts/generate_world.py bench3 --seed 3

The world is the map's geometry rendered in 3D and one Solid per robot; the
robots have no physics and no controller of their own. A single Supervisor
(webots/controllers/fleet_supervisor) steps the headless engine and places each
robot where the engine says it is, so every decision a robot takes in Webots is
the same code path as the benchmark (IF-3.1). Webots is the picture, not the
physics -- which is what core/scenarios.py already says about Engine.WEBOTS.

Only base nodes are used (Solid, Shape, Box, Cylinder, PBRAppearance), never
EXTERNPROTO, so the world opens without network access.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import config, scenarios  # noqa: E402
from core.graph import Graph  # noqa: E402
from simulator.engine import spawn_positions  # noqa: E402

WORLDS_DIR = REPO_ROOT / "webots" / "worlds"
WEBOTS_VERSION = "R2025a"

MM = 0.001
LANE_WIDTH_M = (2 * config.AISLE_LANE_OFFSET_MM + 800) * MM
SINGLE_LANE_WIDTH_M = 1.4
STATION_PAD_M = 1.4
ROBOT_RADIUS_M = config.ROBOT_RADIUS_MM * MM
ROBOT_HEIGHT_M = 0.45
RACK_HEIGHT_M = 1.6
RACK_INSET_M = LANE_WIDTH_M / 2 + 0.3
# Floor markings are a few centimetres proud of the slab: at the viewing
# distances of a 60 m warehouse the depth buffer cannot separate millimetres.
LANE_THICK, LANE_Z = 0.03, 0.015
PAD_THICK, PAD_Z = 0.06, 0.03

# Colours, matching the dashboard so the two views read as one system.
FLOOR = (0.20, 0.23, 0.28)
LANE = (0.27, 0.30, 0.35)
LANE_SINGLE = (0.24, 0.27, 0.32)
HAZARD = (0.95, 0.66, 0.23)
RACK = (0.16, 0.19, 0.24)
PICKUP = (0.25, 0.72, 0.69)
DROP = (0.84, 0.42, 0.63)
BAY = (0.42, 0.47, 0.53)
CHARGER = (0.48, 0.79, 0.44)
ROBOT_IDLE = (0.42, 0.48, 0.58)

_GRID_NAME = re.compile(r"^C\d+R\d+$")


def _look_from_south(pitch: float) -> str:
    """Axis-angle for a viewpoint that faces +y and tilts down by ``pitch``.

    Webots' camera looks along its +x with +z up, so this is a yaw of 90 degrees
    about z followed by a pitch about the camera's own y, composed as quaternions.
    """
    half_yaw, half_pitch = math.pi / 4, pitch / 2
    # q = qz(90deg) * qy(pitch)
    w = math.cos(half_yaw) * math.cos(half_pitch)
    x = -math.sin(half_yaw) * math.sin(half_pitch)
    y = math.cos(half_yaw) * math.sin(half_pitch)
    z = math.sin(half_yaw) * math.cos(half_pitch)
    angle = 2 * math.acos(max(-1.0, min(1.0, w)))
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    return f"{_f(x / norm)} {_f(y / norm)} {_f(z / norm)} {_f(angle)}"


def _f(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


def _rgb(colour: tuple[float, float, float]) -> str:
    return " ".join(_f(c) for c in colour)


def _box(
    x: float, y: float, z: float, sx: float, sy: float, sz: float,
    colour: tuple[float, float, float], *, yaw: float = 0.0, name: str = "",
    def_name: str = "", roughness: float = 0.9,
) -> str:
    prefix = f"DEF {def_name} " if def_name else ""
    rotation = f"  rotation 0 0 1 {_f(yaw)}\n" if yaw else ""
    return (
        f"{prefix}Solid {{\n"
        f"  translation {_f(x)} {_f(y)} {_f(z)}\n"
        f"{rotation}"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance PBRAppearance {{ baseColor {_rgb(colour)} roughness {_f(roughness)} metalness 0 }}\n"
        f"      geometry Box {{ size {_f(sx)} {_f(sy)} {_f(sz)} }}\n"
        f"    }}\n"
        f"  ]\n"
        f"  name \"{name or def_name or 'solid'}\"\n"
        f"}}\n"
    )


def _robot(robot_id: int, x: float, y: float) -> str:
    """One AMR: a cylinder body, a heading bar, and a DEF on the appearance so
    the supervisor can recolour it by state."""
    return (
        f"DEF AMR_{robot_id} Solid {{\n"
        f"  translation {_f(x)} {_f(y)} {_f(ROBOT_HEIGHT_M / 2)}\n"
        f"  rotation 0 0 1 0\n"
        f"  children [\n"
        f"    Shape {{\n"
        f"      appearance DEF AMR_{robot_id}_COLOR PBRAppearance {{ baseColor {_rgb(ROBOT_IDLE)} roughness 0.5 metalness 0.1 }}\n"
        f"      geometry Cylinder {{ radius {_f(ROBOT_RADIUS_M)} height {_f(ROBOT_HEIGHT_M)} }}\n"
        f"    }}\n"
        f"    Transform {{\n"
        f"      translation {_f(ROBOT_RADIUS_M * 0.55)} 0 {_f(ROBOT_HEIGHT_M / 2 + 0.01)}\n"
        f"      children [\n"
        f"        Shape {{\n"
        f"          appearance PBRAppearance {{ baseColor 0.08 0.10 0.14 roughness 1 metalness 0 }}\n"
        f"          geometry Box {{ size {_f(ROBOT_RADIUS_M * 0.9)} 0.06 0.02 }}\n"
        f"        }}\n"
        f"      ]\n"
        f"    }}\n"
        f"    DEF AMR_{robot_id}_CARGO Transform {{\n"
        f"      translation 0 0 {_f(ROBOT_HEIGHT_M / 2 + 0.15)}\n"
        f"      scale 0.001 0.001 0.001\n"
        f"      children [\n"
        f"        Shape {{\n"
        f"          appearance PBRAppearance {{ baseColor 0.85 0.72 0.45 roughness 1 metalness 0 }}\n"
        f"          geometry Box {{ size 0.34 0.34 0.3 }}\n"
        f"        }}\n"
        f"      ]\n"
        f"    }}\n"
        f"  ]\n"
        f"  name \"AMR_{robot_id}\"\n"
        f"}}\n"
    )


def _racks(graph: Graph, raw: dict) -> list[str]:
    """Storage blocks in every grid cell no node touches -- the Kiva pods."""
    if "grid" not in raw:
        return []
    # The aisle grid is the nodes the generator named CxxRyy; ring midpoints and
    # spurs would otherwise slice the cells into slivers.
    grid = [n for n in graph.nodes.values() if _GRID_NAME.match(n.name)]
    xs = sorted({n.x_mm for n in grid})
    ys = sorted({n.y_mm for n in grid})
    out = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            if any(x0 < n.x_mm < x1 and y0 < n.y_mm < y1 for n in graph.nodes.values()):
                continue
            sx = (x1 - x0) * MM - 2 * RACK_INSET_M
            sy = (y1 - y0) * MM - 2 * RACK_INSET_M
            if sx < 0.6 or sy < 0.6:
                continue
            out.append(
                _box(
                    (x0 + x1) / 2 * MM, (y0 + y1) / 2 * MM, RACK_HEIGHT_M / 2,
                    sx, sy, RACK_HEIGHT_M, RACK, name=f"rack_{i}_{j}",
                )
            )
    return out


def build_world(scenario: scenarios.Scenario, *, seed: int, robots: int | None = None) -> str:
    raw = json.loads(scenario.map_path.read_text(encoding="utf-8"))
    graph = Graph.from_dict(raw)
    count = robots or scenario.simulated_robots
    pickups, drops = set(raw.get("pickup_nodes", ())), set(raw.get("drop_nodes", ()))
    choke = raw.get("choke_edge")

    xs = [n.x_mm for n in graph.nodes.values()]
    ys = [n.y_mm for n in graph.nodes.values()]
    min_x, max_x = min(xs) * MM - 4, max(xs) * MM + 4
    min_y, max_y = min(ys) * MM - 4, max(ys) * MM + 4
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    span = max(max_x - min_x, max_y - min_y)

    parts: list[str] = [
        f"#VRML_SIM {WEBOTS_VERSION} utf8\n",
        f"# Generated by scripts/generate_world.py from {scenario.map_path.name}; do not edit.\n",
        f"# Scenario {scenario.name}: {count} AMRs, seed {seed}.\n\n",
        "WorldInfo {\n"
        f"  info [ \"ROBOTON {scenario.name}: {scenario.purpose}\" ]\n"
        f"  title \"ROBOTON {scenario.name}\"\n"
        f"  basicTimeStep {config.MOTION_TICK_MS}\n"
        "  coordinateSystem \"ENU\"\n"
        "}\n",
        "Viewpoint {\n"
        f"  position {_f(cx)} {_f(cy - span * 0.55)} {_f(span * 0.75)}\n"
        f"  orientation {_look_from_south(math.atan2(0.75, 0.55))}\n"
        "  followType \"None\"\n"
        "}\n",
        "Background { skyColor [ 0.08 0.10 0.14 ] }\n",
        "DirectionalLight { direction 0.3 -0.5 -1 intensity 2.2 castShadows TRUE }\n",
        "DirectionalLight { direction -0.6 0.4 -1 intensity 0.8 }\n",
        _box(cx, cy, -0.01, max_x - min_x, max_y - min_y, 0.02, FLOOR, name="floor"),
    ]

    # Lanes stop half a lane width short of each node and a pad fills the node,
    # so no two strips overlap (overlapping coplanar boxes z-fight in the
    # renderer) -- and it is the real geometry: lanes end before the junction.
    for edge in graph.edges.values():
        u, v = graph.node(edge.u), graph.node(edge.v)
        full = math.hypot(v.x_mm - u.x_mm, v.y_mm - u.y_mm) * MM
        yaw = math.atan2(v.y_mm - u.y_mm, v.x_mm - u.x_mm)
        width = SINGLE_LANE_WIDTH_M if edge.single_lane else LANE_WIDTH_M
        colour = LANE_SINGLE if edge.single_lane else LANE
        length = max(0.2, full - LANE_WIDTH_M)
        mx, my = (u.x_mm + v.x_mm) / 2 * MM, (u.y_mm + v.y_mm) / 2 * MM
        parts.append(_box(mx, my, LANE_Z, length, width, LANE_THICK, colour, yaw=yaw, name=f"lane_{edge.id}"))
        if edge.single_lane or edge.id == choke:
            # Hazard edging on a corridor only one robot may hold.
            for side in (-1, 1):
                ox = -math.sin(yaw) * side * (width / 2 + 0.06)
                oy = math.cos(yaw) * side * (width / 2 + 0.06)
                parts.append(
                    _box(mx + ox, my + oy, LANE_Z + 0.01, length, 0.12, LANE_THICK, HAZARD, yaw=yaw, name=f"hazard_{edge.id}_{side}")
                )

    for node in graph.nodes.values():
        if node.is_junction and not (node.id in pickups or node.id in drops or node.is_parking):
            parts.append(
                _box(node.x_mm * MM, node.y_mm * MM, LANE_Z, LANE_WIDTH_M, LANE_WIDTH_M, LANE_THICK, LANE, name=f"junction_{node.id}")
            )

    parts.extend(_racks(graph, raw))

    for node in graph.nodes.values():
        x, y = node.x_mm * MM, node.y_mm * MM
        if node.id in pickups:
            parts.append(_box(x, y, PAD_Z, STATION_PAD_M, STATION_PAD_M, PAD_THICK, PICKUP, name=f"pickup_{node.name}"))
        elif node.id in drops:
            parts.append(_box(x, y, PAD_Z, STATION_PAD_M, STATION_PAD_M, PAD_THICK, DROP, name=f"drop_{node.name}"))
        elif node.is_parking:
            parts.append(_box(x, y, PAD_Z, STATION_PAD_M, STATION_PAD_M, PAD_THICK, BAY, name=f"bay_{node.name}"))
        if node.is_charger:
            parts.append(
                "Solid {\n"
                f"  translation {_f(x)} {_f(y)} 0.1\n"
                "  children [ Shape {\n"
                f"    appearance PBRAppearance {{ baseColor {_rgb(CHARGER)} roughness 0.6 metalness 0.2 }}\n"
                "    geometry Cylinder { radius 0.12 height 0.2 }\n"
                "  } ]\n"
                f"  name \"charger_{node.name}\"\n"
                "}\n"
            )

    spawns = spawn_positions(graph, count, seed=seed)
    for index, node_id in enumerate(spawns, start=1):
        node = graph.node(node_id)
        parts.append(_robot(index, node.x_mm * MM, node.y_mm * MM))

    parts.append(
        "Robot {\n"
        "  name \"fleet_supervisor\"\n"
        "  controller \"fleet_supervisor\"\n"
        f"  controllerArgs [ \"{scenario.name}\", \"--seed\", \"{seed}\"" + (f", \"--robots\", \"{count}\"" if robots else "") + " ]\n"
        "  supervisor TRUE\n"
        "}\n"
    )
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Webots world for a scenario.")
    parser.add_argument("scenario", choices=sorted(scenarios.SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robots", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    scenario = scenarios.get(args.scenario)
    if scenario.physical_robots and args.robots is None:
        print(f"{scenario.name} is a hardware scenario; pass --robots", file=sys.stderr)
        return 2
    world = build_world(scenario, seed=args.seed, robots=args.robots)
    out = args.out or WORLDS_DIR / f"{scenario.name}.wbt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(world, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(world.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
