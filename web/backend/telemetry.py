"""What the dashboard is shown: a plain-data snapshot of a running simulation.

One function turns a ``Simulation`` into JSON-ready dictionaries. The FastAPI app
streams these over a WebSocket and the Webots supervisor pushes the same shape, so
both views describe one run in one vocabulary. Nothing here can touch a robot: the
snapshot reads engine state and returns copies (FR-8.4, IF-1.6, BR-7).

Positions are ``Robot.position_mm`` -- the aisle centre line -- rather than the
lane-adjusted footprint. The dashboard animates the centre line and draws the lane
offset itself, which keeps the picture readable when a robot sits at a node.
"""

from __future__ import annotations

import math
from typing import Any

from core import config
from core.robot import Robot
from simulator.scenario import Simulation, load_map


def map_payload(sim: Simulation) -> dict[str, Any]:
    """The static geometry, sent once when a run starts.

    Re-reads the map file for the task geography (pickup and drop nodes, the
    benchmark choke) that the graph deliberately does not carry.
    """
    graph, zones = sim.graph, sim.zones
    _, raw = load_map(sim.scenario)
    pickups = set(raw.get("pickup_nodes", ()))
    drops = set(raw.get("drop_nodes", ()))
    nodes = []
    for node in graph.nodes.values():
        nodes.append(
            {
                "id": node.id,
                "name": node.name,
                "x": node.x_mm,
                "y": node.y_mm,
                "junction": node.is_junction,
                "marker": node.has_marker,
                "charger": node.is_charger,
                "parking": node.is_parking,
                "pickup": node.id in pickups,
                "drop": node.id in drops,
                "zone": zones.zone_of(node.id) if zones.enabled else -1,
            }
        )
    edges = [
        {
            "id": edge.id,
            "u": edge.u,
            "v": edge.v,
            "length": edge.length_mm,
            "single_lane": edge.single_lane,
            "choke": edge.id == raw.get("choke_edge"),
        }
        for edge in graph.edges.values()
    ]
    return {
        "name": graph.name,
        "description": raw.get("description", ""),
        "nodes": nodes,
        "edges": edges,
        "zones": list(zones.zone_ids) if zones.enabled else [],
        "robot_radius_mm": config.ROBOT_RADIUS_MM,
        "lane_offset_mm": config.AISLE_LANE_OFFSET_MM,
        "junction_footprint_mm": config.JUNCTION_FOOTPRINT_MM,
    }


def _heading_deg(robot: Robot) -> int:
    """Direction of travel in degrees, 0 = +x, counter-clockwise. Standing robots
    keep no heading; the client keeps the last one it saw."""
    if robot.edge_id is None or robot.next_node is None:
        return -1
    here = robot.graph.node(robot.current_node)
    there = robot.graph.node(robot.next_node)
    return int(math.degrees(math.atan2(there.y_mm - here.y_mm, there.x_mm - here.x_mm))) % 360


def robot_payload(robot: Robot) -> dict[str, Any]:
    x, y = robot.position_mm()
    task = robot.task
    wait = robot.wait_cause
    return {
        "id": robot.robot_id,
        "x": x,
        "y": y,
        "heading": _heading_deg(robot),
        "state": robot.state.value,
        "battery": robot.battery_pct,
        "task": task.task_id if task is not None else None,
        "leg": task.leg.value if task is not None else None,
        "node": robot.current_node,
        "next": robot.next_node,
        "edge": robot.edge_id,
        "route": list(robot.remaining_route),
        "queue": len(robot.queue),
        "wait": (
            {"kind": wait.kind, "blocker": wait.blocker_id, "resource": wait.resource}
            if wait is not None
            else None
        ),
        "stopped_ms": robot.metrics.stopped_ms,
        "distance_mm": robot.metrics.distance_mm,
    }


def snapshot(sim: Simulation, *, finished: bool | None = None) -> dict[str, Any]:
    """Everything the fleet view needs for one frame."""
    engine = sim.engine
    robots = [robot_payload(r) for r in engine.active_robots]
    completed_ids = {t.task_id for t in sim.completed}
    # The announced set holds its own copies of every task; the robots hold
    # theirs. Who holds what is therefore read off the robots, not the set.
    holders = {
        task.task_id: robot.robot_id for robot in engine.robots for task in robot.queue
    }
    tasks = []
    for task in sim.task_set.tasks:
        if task.task_id in completed_ids:
            status = "COMPLETED"
        elif task.created_at_ms > engine.now_ms:
            status = "SCHEDULED"
        elif task.task_id in holders:
            status = "HELD"
        else:
            status = "PENDING"
        tasks.append(
            {
                "id": task.task_id,
                "pickup": task.pickup,
                "drop": task.drop,
                "priority": task.priority,
                "holder": holders.get(task.task_id),
                "status": status,
            }
        )
    stats = sim.mesh.stats if sim.mesh is not None else None
    return {
        "scenario": sim.scenario.name,
        "seed": sim.seed,
        "allocator": sim.allocator.name,
        "now_ms": engine.now_ms,
        "finished": sim.is_finished if finished is None else finished,
        "robots": robots,
        "tasks": tasks,
        "counters": {
            "tasks_total": len(sim.task_set),
            "tasks_completed": len(completed_ids),
            "tasks_pending": len(sim.pending),
            "makespan_ms": sim.makespan_ms,
            "collisions": len(engine.collisions),
            "coordination_failures": len(engine.coordination_failures),
            "stopped_ms": sum(r.metrics.stopped_ms for r in engine.robots),
            "yields": sum(r.metrics.yields_lost for r in engine.robots),
            "frames_sent": stats.total_sent if stats is not None else 0,
            "frames_by_type": dict(stats.sent) if stats is not None else {},
        },
    }
