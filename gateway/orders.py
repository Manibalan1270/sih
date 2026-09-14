"""The order gateway: where work enters the fleet (FR-4.1, IF-3.2, IF-3.3).

An order is a journey an operator wants done. Accepting one means putting it on
the mesh as an ANNOUNCE and then having no further part in the outcome -- the
fleet decides who does it. Nothing here names a robot, ranks robots, or chooses
a winner, because FR-4.1 forbids any component assigning a task to a named AMR
and FR-4.6 forbids a dispatcher. That restriction is the whole reason this
module is separate from the allocator: the gateway's only power is to say that
work exists.

Mechanically an accepted order is appended to the running simulation's task set
with ``created_at_ms`` set to now. ``Simulation.take_pending`` releases it on the
next tick, the allocator announces it like any other task, and from that moment
it is indistinguishable from generated work. There is deliberately no separate
"operator task" path through the fleet: an order that behaved differently from a
benchmark task would not be evidence of anything.

The dashboard cannot reach this module, and this module holds no transport. The
gateway originates ANNOUNCE and the clock beacon and nothing else (IF-3.2,
IF-3.3), which in simulation means appending to the pending queue that the
allocator already reads. FR-8.4's read-only guarantee for the *dashboard* is
unaffected: posting an order is not commanding a robot, and no robot is named.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

from core.task import MAX_PRIORITY, Task
from simulator.scenario import Simulation, load_map

MAX_TASK_ID = 0xFFFE
"""Task ids are a uint16 on the wire (ANNOUNCE, BID, CLAIM, COMPLETE), and
``communication.messages.NO_TASK`` reserves 0xFFFF as the "holding nothing"
sentinel in INTENT. So the last usable id is one below it."""


class OrderRejected(ValueError):
    """The order cannot be accepted. Carries an operator-readable reason.

    Rejection is preferred to clamping throughout: a priority silently clamped
    into range, or a pickup silently moved to the nearest legal node, would make
    the fleet do something the operator did not ask for.
    """


@dataclass(frozen=True, slots=True)
class Station:
    """A node an order may name as an endpoint."""

    node: int
    name: str
    kind: str
    """``pickup``, ``drop`` or ``station`` -- what the map declares it to be."""

    x_mm: int
    y_mm: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "name": self.name,
            "kind": self.kind,
            "x": self.x_mm,
            "y": self.y_mm,
        }


@dataclass(slots=True)
class OrderLog:
    """Which tasks arrived as operator orders rather than generated work.

    Kept because the two are deliberately identical once announced: the fleet
    cannot tell them apart and must not, so the only place the distinction can
    survive is here. Reset whenever the run changes, since task ids restart with
    it and a stale receipt would point at someone else's task.
    """

    run: "weakref.ReferenceType[Simulation] | None" = None
    receipts: list[dict[str, Any]] = field(default_factory=list)

    def bind(self, sim: Simulation | None) -> None:
        """Point at ``sim``, discarding receipts if it is a different run.

        Held weakly and compared by identity rather than by ``id()``: a finished
        run is garbage before the next one starts, and CPython reuses addresses,
        so an id comparison can silently decide that a brand new run is the old
        one and keep receipts pointing at somebody else's task ids.
        """
        current = self.run() if self.run is not None else None
        if current is not sim:
            self.run = weakref.ref(sim) if sim is not None else None
            self.receipts = []

    def record(self, task: Task) -> dict[str, Any]:
        receipt = {
            "task_id": task.task_id,
            "pickup": task.pickup,
            "drop": task.drop,
            "priority": task.priority,
            "accepted_at_ms": task.created_at_ms,
        }
        self.receipts.append(receipt)
        return receipt

    def ids(self) -> set[int]:
        return {receipt["task_id"] for receipt in self.receipts}


def _degree(sim: Simulation, node: int) -> int:
    return len(sim.graph.neighbours(node))


def stations(sim: Simulation) -> list[Station]:
    """Every node an order may legally name, in id order.

    A legal endpoint is a degree-one spur that is not a parking bay or charger.
    Both halves matter and both are defects this project has already paid for:

    * Degree one is the well-formedness condition of defect 13. A task endpoint
      on a junction parks a robot in an intersection, which is the one place the
      rule "never come to rest inside a conflict region" cannot be honoured,
      because the task requires the rest. Every station on the shipped maps is a
      spur precisely so this holds.
    * Bays and chargers are excluded because they are where idle robots live. An
      order delivering to a bay would have a robot park on another robot's only
      way out, and `Robot._nearest_parking_node` would then route the fleet into
      the cargo.
    """
    _, raw = load_map(sim.scenario)
    pickups = set(raw.get("pickup_nodes", ()))
    drops = set(raw.get("drop_nodes", ()))
    out = []
    for node_id in sorted(sim.graph.nodes):
        node = sim.graph.node(node_id)
        if node.is_parking or node.is_charger:
            continue
        if _degree(sim, node_id) != 1:
            continue
        kind = "pickup" if node_id in pickups else "drop" if node_id in drops else "station"
        out.append(Station(node=node_id, name=node.name, kind=kind, x_mm=node.x_mm, y_mm=node.y_mm))
    return out


def _check_endpoint(sim: Simulation, node: int, role: str, legal: set[int]) -> None:
    if node not in sim.graph.nodes:
        raise OrderRejected(f"{role} node {node} is not on map {sim.graph.name!r}")
    if node in legal:
        return
    where = sim.graph.node(node)
    if where.is_parking or where.is_charger:
        raise OrderRejected(
            f"{role} node {node} ({where.name}) is a bay or charger. Bays are where "
            f"idle AMRs wait; delivering there would block another robot's only exit"
        )
    raise OrderRejected(
        f"{role} node {node} ({where.name}) has {_degree(sim, node)} edges. A task "
        f"endpoint must be a degree-one spur, or the AMR that dwells there is parked "
        f"in a conflict region it cannot step out of"
    )


def next_task_id(sim: Simulation) -> int:
    """One past the highest id in the task set.

    Reused ids would be catastrophic rather than merely confusing: every robot
    keys its auction state, its held task and its completion record by id, so a
    repeat would look like a re-announcement of work already done.
    """
    highest = max((task.task_id for task in sim.task_set.tasks), default=-1)
    candidate = highest + 1
    if candidate > MAX_TASK_ID:
        raise OrderRejected(
            f"task id space exhausted at {MAX_TASK_ID}; ids are a uint16 on the wire"
        )
    return candidate


def submit(sim: Simulation, *, pickup: int, drop: int, priority: int) -> Task:
    """Accept an order and hand it to the fleet. Returns the task created.

    The caller is responsible for holding whatever lock guards ``sim`` -- this
    mutates the task set, and a snapshot taken mid-append would show a task the
    engine has not seen.
    """
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise OrderRejected(f"priority must be a whole number, got {priority!r}")
    if not 0 <= priority <= MAX_PRIORITY:
        raise OrderRejected(
            f"priority {priority} is outside 0-{MAX_PRIORITY}; ASM-17 leaves the "
            f"arbitration total order undefined beyond that range"
        )
    legal = {station.node for station in stations(sim)}
    _check_endpoint(sim, pickup, "pickup", legal)
    _check_endpoint(sim, drop, "drop", legal)
    if pickup == drop:
        raise OrderRejected(f"pickup and drop are both node {pickup}, which is not a journey")

    task = Task(
        task_id=next_task_id(sim),
        pickup=pickup,
        drop=drop,
        priority=priority,
        # Now, not zero: the aging term that discharges BR-2 measures from here,
        # and back-dating an order would have it outrank work already waiting.
        created_at_ms=sim.engine.now_ms,
        zone_id=sim.zones.zone_of(pickup) if sim.zones.enabled else -1,
    )
    sim.task_set.tasks.append(task)
    return task
