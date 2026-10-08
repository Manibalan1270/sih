"""Where the storage racks stand, and which shelving each pickup is collected from.

The graph is the aisle network and says nothing about shelving: a rack is not a
place a robot can be, so core/ has no business knowing it is there. But three
views draw the warehouse -- the plan, the 3D floor and the Webots world -- and
they have to put the same racks in the same places, so the rule lives here once.

A rack fills every cell of the aisle grid that no aisle node lies inside, set
back RACK_INSET_MM from each aisle centre so a robot in the aisle clears it.
Pickups are in the aisles themselves (scripts/generate_maps.py): a robot stops on
the aisle centre line beside the rack and collects from its face. Nothing is cut
into a rack for it. The bay of shelving it collects from is that rack's **pick
face** -- where the pick light hangs and the order's carton waits.

Maps without an aisle grid (bench3, the loop map) have no racks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core import config
from core.graph import Graph

GRID_NAME = re.compile(r"^C\d+R\d+$")
"""The aisle-grid nodes generate_maps.py names CxxRyy. Midpoints and spurs would
otherwise slice the cells into slivers."""

RACK_INSET_MM = (2 * config.AISLE_LANE_OFFSET_MM + 800) // 2 + 300
"""Aisle centre to rack face: half a two-lane aisle's width plus 300 mm, as
scripts/generate_world.py has always built it."""

MIN_RACK_MM = 600
"""A cell narrower than this after the inset gets no rack."""

FACE_WIDTH_MM = 1100
"""Along the aisle: one bay of shelving, centred on the robot's stopping point."""

FACE_DEPTH_MM = 600
"""Into the rack from its face: the front of the bay, within an arm's reach."""


@dataclass(frozen=True, slots=True)
class PickFace:
    """The bay of shelving a pickup on the aisle is collected from, in mm."""

    node: int
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True, slots=True)
class Rack:
    """One rack block, filling one grid cell less the inset, in mm."""

    x0: int
    y0: int
    x1: int
    y1: int
    faces: tuple[PickFace, ...] = ()


def racks(graph: Graph) -> list[Rack]:
    """Every rack on ``graph``'s floor, with the pick face of each pickup on it."""
    grid = [n for n in graph.nodes.values() if GRID_NAME.match(n.name)]
    if not grid:
        return []
    xs = sorted({n.x_mm for n in grid})
    ys = sorted({n.y_mm for n in grid})
    pickups = [graph.node(p) for p in graph.pickup_nodes]
    out: list[Rack] = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx0, cx1, cy0, cy1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            if any(cx0 < n.x_mm < cx1 and cy0 < n.y_mm < cy1 for n in graph.nodes.values()):
                continue  # an aisle runs through this cell
            x0, x1 = cx0 + RACK_INSET_MM, cx1 - RACK_INSET_MM
            y0, y1 = cy0 + RACK_INSET_MM, cy1 - RACK_INSET_MM
            if x1 - x0 < MIN_RACK_MM or y1 - y0 < MIN_RACK_MM:
                continue
            faces = tuple(
                _face(p.id, p.x_mm, p.y_mm, x0, y0, x1, y1)
                for p in pickups
                if _faces_west(p.x_mm, p.y_mm, cx0, cy0, cy1) or _faces_south(p.x_mm, p.y_mm, cx0, cx1, cy0)
            )
            out.append(Rack(x0, y0, x1, y1, faces))
    return out


def _faces_west(x: int, y: int, cx0: int, cy0: int, cy1: int) -> bool:
    """A pickup on the pick-aisle along the cell's west side collects from it."""
    return x == cx0 and cy0 < y < cy1


def _faces_south(x: int, y: int, cx0: int, cx1: int, cy0: int) -> bool:
    """One on the cross-aisle along its low-y side does too -- no generated map puts
    pickups there, but a hand-made one may."""
    return y == cy0 and cx0 < x < cx1


def _face(node: int, x: int, y: int, x0: int, y0: int, x1: int, y1: int) -> PickFace:
    half = FACE_WIDTH_MM // 2
    if x < x0:  # the aisle is west of the rack: the face is its west side
        return PickFace(node, x0, y - half, x0 + FACE_DEPTH_MM, y + half)
    return PickFace(node, x - half, y0, x + half, y0 + FACE_DEPTH_MM)


def pick_faces(graph: Graph) -> frozenset[int]:
    """The pickups collected from a rack's shelving, rather than from a station."""
    return frozenset(f.node for rack in racks(graph) for f in rack.faces)


def racks_payload(graph: Graph) -> list[dict]:
    """``racks`` as plain data, for the dashboard and the Webots supervisor."""
    return [
        {
            "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1,
            "faces": [
                {"node": f.node, "x0": f.x0, "y0": f.y0, "x1": f.x1, "y1": f.y1}
                for f in r.faces
            ],
        }
        for r in racks(graph)
    ]
