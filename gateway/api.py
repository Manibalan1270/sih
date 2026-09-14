"""HTTP surface for the order gateway (FR-4.1, IF-3.2).

Separate from ``web/backend/app.py`` on purpose. The dashboard's read-only
guarantee (FR-8.4, IF-1.6, BR-7) is structural -- ``tests/test_architecture.py``
reads that module's source and fails if it so much as names a transport -- so
order entry lives here instead of being bolted onto the spectator. The router is
built against whatever is hosting the run rather than importing the dashboard's
controller, which keeps the dependency pointing one way: the dashboard may mount
the gateway, the gateway knows nothing about the dashboard.

What an operator can do here is exactly one thing: say that a journey needs
doing. There is no endpoint that names a robot, and adding one would contradict
FR-4.1 rather than merely being unwise -- see ``tests/test_architecture.py``.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.task import MAX_PRIORITY
from gateway import orders
from simulator.scenario import Simulation


class RunSource(Protocol):
    """The bit of a run host the gateway needs: the simulation and its lock."""

    lock: threading.Lock
    sim: Simulation | None
    finished: bool


class OrderRequest(BaseModel):
    pickup: int
    drop: int
    priority: int = Field(default=10, ge=0, le=MAX_PRIORITY)
    """ASM-17's operator-supplied priority. Ten matches the generated default, so
    an order posted without thinking about priority competes on equal terms."""


def build_router(source: RunSource) -> APIRouter:
    router = APIRouter()
    log = orders.OrderLog()

    def _require_run() -> Simulation:
        sim = source.sim
        if sim is None:
            raise HTTPException(409, "no run in progress -- start one first")
        if source.finished:
            raise HTTPException(409, "this run has finished; start another to post work")
        return sim

    @router.get("/stations")
    def list_stations() -> dict[str, Any]:
        """Where an order may collect and deliver.

        Served as well as streamed because the page can be opened before the
        first telemetry frame arrives, and an order form with empty dropdowns
        looks like a broken feature rather than an idle one.
        """
        with source.lock:
            sim = _require_run()
            return {
                "map": sim.graph.name,
                "stations": [station.as_dict() for station in orders.stations(sim)],
            }

    @router.get("")
    def list_orders() -> dict[str, Any]:
        with source.lock:
            log.bind(source.sim)
            return {"orders": list(log.receipts)}

    @router.post("")
    def post_order(request: OrderRequest) -> dict[str, Any]:
        with source.lock:
            sim = _require_run()
            log.bind(sim)
            try:
                task = orders.submit(
                    sim,
                    pickup=request.pickup,
                    drop=request.drop,
                    priority=request.priority,
                )
            except orders.OrderRejected as rejected:
                raise HTTPException(400, str(rejected)) from rejected
            receipt = log.record(task)
        return {"accepted": True, "order": receipt}

    return router
