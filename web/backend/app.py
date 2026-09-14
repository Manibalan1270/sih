"""Run Control and the Fleet Dashboard backend (FR-8, Phase 12).

    py -m uvicorn web.backend.app:app --reload
    py -m web.backend.app                      # same, without reload

Runs a headless scenario in a background thread, paced to the wall clock, and
streams ``telemetry.snapshot`` frames to every connected browser. Run Control
starts a run (scenario, seed, speed) and can pause or resume it; that is the whole
of what an operator can do. There is no path from here to a robot: this module
constructs simulations and reads them, and never holds a transport (FR-8.4,
IF-1.6, BR-7 -- enforced by tests/test_architecture.py).

The fleet does not need this process. Kill it mid-run and the robots are unaffected,
because the simulation's robots talk to each other over the in-process mesh and
never to the dashboard; the dashboard is a spectator that happens to host the run
loop for convenience. In the hardware scenario it would only ever be a spectator.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import config, scenarios
from simulator import scenario as scenario_module
from web.backend import telemetry

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

TELEMETRY_HZ = 10
"""Dashboard frame rate. FR-8.2 asks for at least 1 Hz; ten is smooth on 100 robots
and still a few kilobytes a frame."""

DEFAULT_MAX_MS = 1_800_000
"""Simulated-time cap per run, matching scripts/run_scenario.py."""


class RunRequest(BaseModel):
    scenario: str = "bench3"
    seed: int = 1
    robots: int | None = Field(default=None, ge=1)
    tasks: int | None = Field(default=None, ge=1)
    waves: int = Field(default=4, ge=1)
    speed: float = Field(default=1.0, gt=0, le=64)
    """Simulated seconds per wall-clock second."""


@dataclass
class RunController:
    """Owns the simulation thread and the only copy of the run's state.

    The simulation is stepped on its own thread, ``speed`` simulated seconds per
    wall second; snapshots are taken under ``lock`` so a frame never sees a
    half-stepped tick. Everything the outside world gets is a copy.
    """

    sim: scenario_module.Simulation | None = None
    request: RunRequest | None = None
    speed: float = 1.0
    paused: bool = False
    finished: bool = False
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _generation: int = 0

    # -- control ---------------------------------------------------------------

    def start(self, request: RunRequest) -> dict[str, Any]:
        self.stop()
        chosen = scenarios.get(request.scenario)
        if chosen.physical_robots and request.robots is None:
            raise HTTPException(
                400,
                f"{chosen.name} is a hardware scenario; pass robots to simulate it",
            )
        sim = scenario_module.build(
            scenarios.clamped(chosen),
            seed=request.seed,
            robots=request.robots,
            task_count=request.tasks,
            waves=request.waves,
        )
        with self.lock:
            self.sim = sim
            self.request = request
            self.speed = request.speed
            self.paused = False
            self.finished = False
            self.error = None
            self._generation += 1
            generation = self._generation
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(sim, self._stop, generation), daemon=True,
            name=f"sim-{chosen.name}-{request.seed}",
        )
        self._thread.start()
        return self.status()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None

    def set_paused(self, paused: bool) -> dict[str, Any]:
        with self.lock:
            self.paused = paused
        return self.status()

    def set_speed(self, speed: float) -> dict[str, Any]:
        with self.lock:
            self.speed = max(0.05, min(64.0, speed))
        return self.status()

    # -- the run loop ----------------------------------------------------------

    def _run(self, sim: scenario_module.Simulation, stop: threading.Event, generation: int) -> None:
        tick_ms = config.MOTION_TICK_MS
        limit_ms = sim.engine.now_ms + DEFAULT_MAX_MS
        sim_origin = sim.engine.now_ms
        wall_origin = time.perf_counter()
        sim_elapsed_ms = 0
        try:
            while not stop.is_set():
                with self.lock:
                    paused, speed = self.paused, self.speed
                if paused:
                    # Freeze both clocks so resuming does not sprint to catch up.
                    time.sleep(0.05)
                    wall_origin = time.perf_counter()
                    sim_origin = sim.engine.now_ms
                    sim_elapsed_ms = 0
                    continue
                wall_ms = (time.perf_counter() - wall_origin) * 1000.0
                target_ms = int(wall_ms * speed)
                stepped = 0
                while sim_elapsed_ms < target_ms and stepped < 200:
                    with self.lock:
                        if sim.is_finished or sim.engine.now_ms >= limit_ms:
                            self.finished = True
                            return
                        sim.step()
                    sim_elapsed_ms = sim.engine.now_ms - sim_origin
                    stepped += 1
                if stepped == 200:
                    # Could not keep up; re-base rather than accumulate debt.
                    wall_origin = time.perf_counter()
                    sim_origin = sim.engine.now_ms
                    sim_elapsed_ms = 0
                else:
                    time.sleep(tick_ms / 1000.0 / max(speed, 1.0))
        except Exception as exc:  # noqa: BLE001 -- surfaced to the dashboard
            with self.lock:
                if self._generation == generation:
                    self.error = f"{type(exc).__name__}: {exc}"

    # -- views -----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self.lock:
            sim = self.sim
            return {
                "running": sim is not None and not self.finished and self.error is None,
                "paused": self.paused,
                "finished": self.finished,
                "speed": self.speed,
                "error": self.error,
                "scenario": sim.scenario.name if sim else None,
                "seed": sim.seed if sim else None,
                "robots": len(sim.engine.robots) if sim else 0,
                "now_ms": sim.engine.now_ms if sim else 0,
            }

    def frame(self) -> dict[str, Any] | None:
        with self.lock:
            if self.sim is None:
                return None
            data = telemetry.snapshot(self.sim, finished=self.finished)
        data["paused"] = self.paused
        data["speed"] = self.speed
        data["error"] = self.error
        return data

    def map(self) -> dict[str, Any] | None:
        with self.lock:
            if self.sim is None:
                return None
            return telemetry.map_payload(self.sim)


controller = RunController()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield
    controller.stop()


app = FastAPI(title="ROBOTON Fleet Dashboard", version="1.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    out = []
    for name in sorted(scenarios.SCENARIOS):
        s = scenarios.get(name)
        out.append(
            {
                "name": s.name,
                "purpose": s.purpose,
                "robots": s.robots,
                "physical_robots": s.physical_robots,
                "engine": s.engine.value,
                "map": s.map_path.name,
                "benchmarked": s.benchmarked,
            }
        )
    return out


@app.get("/api/status")
def status() -> dict[str, Any]:
    return controller.status()


@app.get("/api/map")
def current_map() -> dict[str, Any]:
    data = controller.map()
    if data is None:
        raise HTTPException(404, "no run in progress")
    return data


@app.get("/api/state")
def current_state() -> dict[str, Any]:
    data = controller.frame()
    if data is None:
        raise HTTPException(404, "no run in progress")
    return data


@app.post("/api/run")
def start_run(request: RunRequest) -> dict[str, Any]:
    if request.scenario not in scenarios.SCENARIOS:
        raise HTTPException(404, f"unknown scenario {request.scenario!r}")
    return controller.start(request)


@app.post("/api/pause")
def pause() -> dict[str, Any]:
    return controller.set_paused(True)


@app.post("/api/resume")
def resume() -> dict[str, Any]:
    return controller.set_paused(False)


class SpeedRequest(BaseModel):
    speed: float = Field(gt=0, le=64)


@app.post("/api/speed")
def speed(request: SpeedRequest) -> dict[str, Any]:
    return controller.set_speed(request.speed)


@app.websocket("/ws/telemetry")
async def telemetry_stream(socket: WebSocket) -> None:
    """Push one frame per period. The map is sent whenever the run changes so a
    client that connects mid-run, or watches a restart, always has the geometry."""
    await socket.accept()
    period = 1.0 / TELEMETRY_HZ
    seen_generation = -1
    try:
        while True:
            generation = controller._generation
            if generation != seen_generation:
                seen_generation = generation
                map_data = controller.map()
                if map_data is not None:
                    await socket.send_text(json.dumps({"type": "map", "data": map_data}))
            frame = controller.frame()
            if frame is None:
                await socket.send_text(json.dumps({"type": "idle", "data": controller.status()}))
            else:
                await socket.send_text(json.dumps({"type": "frame", "data": frame}))
            await asyncio.sleep(period)
    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn

    uvicorn.run("web.backend.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
