"""Run Control and the Fleet Dashboard backend (FR-8, Phase 12).

    py -m uvicorn web.backend.app:app --reload
    py -m web.backend.app                      # same, without reload

Runs a headless scenario in a background thread, paced to the wall clock, and
streams ``telemetry.snapshot`` frames to every connected browser. Run Control
starts a run (scenario, seed, speed) and can pause or resume it. This process
also hosts the order gateway's page and router (``gateway/``), which is how work
is posted -- but that is the gateway's surface, not the dashboard's.

There is no path from here to a robot: this module constructs simulations and
reads them, and never holds a transport (FR-8.4, IF-1.6, BR-7 -- enforced by
tests/test_architecture.py). Posting an order is not an exception to that. An
order says a journey needs doing and goes onto the mesh as an ANNOUNCE; which
AMR takes it is the fleet's decision and nothing here can influence it (FR-4.1).

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

from benchmark import runner as benchmark_runner
from core import config, scenarios
from gateway.api import build_router as build_order_router
from simulator import scenario as scenario_module
from simulator.scenario import AuctionAllocator, RoundRobinAllocator
from web.backend import telemetry

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

TELEMETRY_HZ = 10
"""Dashboard frame rate. FR-8.2 asks for at least 1 Hz; ten is smooth on 100 robots
and still a few kilobytes a frame."""

DEFAULT_MAX_MS = 1_800_000
"""Simulated-time cap per run, matching scripts/run_scenario.py."""

EXTERNAL_TIMEOUT_S = 3.0


CONFIGURATIONS = {
    "A": {
        "name": "Stop-and-wait",
        "detail": (
            "The conventional baseline (FR-10.5). Tasks go round the fleet in turn, "
            "robots exchange nothing, and the only thing keeping them apart is each "
            "one's own forward sensor: see something ahead, stop, wait for it to go."
        ),
    },
    "B": {
        "name": "ROBOTON",
        "detail": (
            "Route-level time-window reservation executed by precedence. Each AMR "
            "bids for work, books its whole route against every plan it has heard, "
            "and enters a junction, corridor or station only after the robots booked "
            "ahead of it there have left."
        ),
    },
}
"""The two things the benchmark compares. Configuration A is not a crippled B: it
is the same robot object built without an auctioneer or an arbiter, which is what
makes the comparison about coordination rather than about two codebases."""


def allocator_for(config_name: str):
    return RoundRobinAllocator() if config_name == "A" else AuctionAllocator()


class RunRequest(BaseModel):
    scenario: str = "bench3"
    seed: int = 1
    robots: int | None = Field(default=None, ge=1)
    tasks: int | None = Field(default=None, ge=1)
    waves: int = Field(default=4, ge=1)
    speed: float = Field(default=1.0, gt=0, le=64)
    """Simulated seconds per wall-clock second."""

    config: str = Field(default="B", pattern="^[AB]$")
    """Which algorithm drives this run: A stop-and-wait, B ROBOTON. Run Control
    offers both so the difference can be watched on the floor and not only read
    off the benchmark table."""


class BenchmarkRequest(BaseModel):
    scenario: str = "bench3"
    seeds: int = Field(default=5, ge=1, le=20)
    robots: int | None = Field(default=None, ge=1)


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

    directive: dict[str, Any] | None = None
    """What a human last asked for, published for the Webots supervisor to read.

    The dashboard still holds no transport and opens no connection (FR-8.4): this
    is a value the supervisor *fetches*. It names a run, never a robot -- which
    AMR does what remains the fleet's decision (FR-4.1)."""

    directive_generation: int = 0
    external_ack: int = 0
    """The directive generation Webots last confirmed it is running. Until these
    agree, frames from Webots belong to the previous run and must not be shown."""

    dashboard_owned: bool = False
    """Set the moment a human clicks Start run, and never cleared while this
    process lives. A supervisor that has lost the feed re-announces its map on a
    timer, and that resend is indistinguishable from a fresh Webots run -- so a
    finished Webots run kept seizing the view back within seconds of every run
    started here. A person clicking a button outranks an automatic retry; to hand
    the floor back to Webots, restart this process."""

    external_map: dict[str, Any] | None = None
    external_frame: dict[str, Any] | None = None
    external_at: float = 0.0
    """A run hosted elsewhere -- the Webots supervisor -- pushing its own frames.
    While those keep arriving the dashboard shows them instead of its own run;
    when they stop for EXTERNAL_TIMEOUT_S it falls back."""

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
            allocator=allocator_for(request.config),
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
            self.external_map = self.external_frame = None
            self.dashboard_owned = True
            self._generation += 1
            generation = self._generation
            # Published for Webots, and hosted here at the same time. The local
            # run is what the browser shows until Webots confirms it has taken
            # the same one up, so a supervisor that is absent, busy or on another
            # world can never leave Run Control looking dead.
            self.directive_generation += 1
            self.external_ack = 0
            self.directive = {
                "generation": self.directive_generation,
                "scenario": chosen.name,
                "seed": request.seed,
                "robots": request.robots,
                "tasks": request.tasks,
                "waves": request.waves,
                "speed": request.speed,
                "config": request.config,
            }
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

    # -- an external run pushing frames --------------------------------------

    def _stamped_for_current_run(self, data: dict[str, Any]) -> bool:
        """True when a push carries the directive generation we last published."""
        stamp = data.get("directive")
        # Generation 0 is "no run was ever asked for": never a match, or a
        # supervisor that has adopted nothing would look like an acknowledgement.
        return bool(stamp) and stamp == self.directive_generation

    def accept_external_map(self, data: dict[str, Any]) -> bool:
        with self.lock:
            if self.dashboard_owned and not self._stamped_for_current_run(data):
                return False
        # A dashboard-started run left running in the background must not be able
        # to resurface later if Webots' feed lapses for a moment -- once Webots
        # is here, it is the only source until it stops sending.
        self.stop()
        with self.lock:
            self.sim = None
            self.external_map = data
            self.external_frame = None
            self.external_at = time.time()
            self._generation += 1
        return True

    def accept_external_frame(self, data: dict[str, Any]) -> bool:
        with self.lock:
            if self.external_map is None:
                return False
            if self.dashboard_owned and not self._stamped_for_current_run(data):
                return False
            self.external_frame = data
            self.external_at = time.time()
            if self._stamped_for_current_run(data):
                self.external_ack = self.directive_generation
        if (
            self.directive_generation > 0
            and self.external_ack == self.directive_generation
            and self._thread is not None
        ):
            # Webots has the run. One copy is the run; stop stepping the spare so
            # the two views can never disagree about what time it is.
            self.stop()
        return True

    def _external_live(self) -> bool:
        if self.dashboard_owned and self.external_ack != self.directive_generation:
            return False
        return (
            self.external_map is not None
            and self.external_frame is not None
            and time.time() - self.external_at < EXTERNAL_TIMEOUT_S
        )

    # -- views -----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self.lock:
            if self._external_live():
                frame = self.external_frame
                return {
                    "running": not frame.get("finished", False),
                    "paused": False,
                    "finished": frame.get("finished", False),
                    "speed": None,
                    "error": None,
                    "source": frame.get("source", "external"),
                    "mirrored_by_webots": self.external_ack == self.directive_generation
                    and self.directive_generation > 0,
                    "scenario": frame.get("scenario"),
                    "seed": frame.get("seed"),
                    "robots": len(frame.get("robots", ())),
                    "now_ms": frame.get("now_ms", 0),
                }
            sim = self.sim
            return {
                "source": "dashboard",
                "mirrored_by_webots": self.external_ack == self.directive_generation
                and self.directive_generation > 0,
                "running": sim is not None and not self.finished and self.error is None,
                "paused": self.paused,
                "finished": self.finished,
                "speed": self.speed,
                "error": self.error,
                "scenario": sim.scenario.name if sim else None,
                "seed": sim.seed if sim else None,
                "robots": len(sim.engine.robots) if sim else 0,
                "config": self.request.config if self.request else None,
                "algorithm": (
                    CONFIGURATIONS[self.request.config]["name"] if self.request else None
                ),
                "now_ms": sim.engine.now_ms if sim else 0,
            }

    def frame(self) -> dict[str, Any] | None:
        with self.lock:
            if self._external_live():
                data = dict(self.external_frame)
                data.setdefault("source", "external")
                data["paused"] = False
                data["speed"] = None
                data["error"] = None
                return data
            if self.sim is None:
                return None
            data = telemetry.snapshot(self.sim, finished=self.finished)
        data["source"] = "dashboard"
        data["paused"] = self.paused
        data["speed"] = self.speed
        data["error"] = self.error
        return data

    def map(self) -> dict[str, Any] | None:
        with self.lock:
            if self._external_live() or (self.external_map is not None and self.sim is None):
                return self.external_map
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

# Order entry is the gateway's, not the dashboard's: this process hosts the page,
# but the logic that accepts work lives in gateway/ and holds no transport either
# (FR-4.1, IF-3.2). Mounting it here does not weaken FR-8.4 -- posting an order
# announces that work exists; it never names the robot that will do it.
app.include_router(build_order_router(controller), prefix="/api/orders", tags=["orders"])


@dataclass
class BenchmarkController:
    """Runs the section 6.3 A/B protocol on a worker thread and holds its result.

    Both configurations get the *same* seeds and therefore the same task sets --
    ``compare_configurations`` refuses the comparison otherwise -- so the only
    difference between the two columns is the coordination. The work happens off
    the request thread because ten seeds of two configurations is minutes, not
    milliseconds, and the page has to stay live while it runs.

    Read-only like the rest of this module: it constructs simulations and reads
    them, and holds no way to reach a robot (FR-8.4).
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    state: str = "idle"
    """idle | running | done | error."""

    request: BenchmarkRequest | None = None
    done_runs: int = 0
    total_runs: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    _thread: threading.Thread | None = None

    def start(self, request: BenchmarkRequest) -> dict[str, Any]:
        with self.lock:
            if self.state == "running":
                raise HTTPException(409, "a benchmark is already running")
            self.request = request
            self.state = "running"
            self.done_runs = 0
            self.total_runs = request.seeds * 2
            self.result = None
            self.error = None
        self._thread = threading.Thread(
            target=self._run, args=(request,), daemon=True, name="benchmark"
        )
        self._thread.start()
        return self.status()

    def _run(self, request: BenchmarkRequest) -> None:
        try:
            scenario = scenarios.get(request.scenario)
            per_config: dict[str, list] = {}
            for config_name in ("A", "B"):
                rows = []
                for seed in range(request.seeds):
                    rows.append(
                        benchmark_runner.run_seed(
                            scenario,
                            seed=seed,
                            config_name=config_name,
                            allocator=allocator_for(config_name),
                            robots=request.robots,
                            # A configuration that jams is a result, not an error:
                            # it is the headline the comparison exists to produce.
                            allow_unfinished=True,
                        )
                    )
                    with self.lock:
                        self.done_runs += 1
                per_config[config_name] = rows
            summary = benchmark_runner.compare_configurations(
                per_config["A"], per_config["B"]
            )
            with self.lock:
                self.result = self._present(summary, per_config, request)
                self.state = "done"
        except Exception as exc:  # noqa: BLE001 - reported to the operator verbatim
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.state = "error"

    @staticmethod
    def _present(
        summary: dict[str, float], per_config: dict[str, list], request: BenchmarkRequest
    ) -> dict[str, Any]:
        """Shape the comparison for the page, deltas included.

        ``makespan_valid`` is the honest caveat and the reason the page does not
        simply print a speed-up. Two things can make the two times incomparable:

        * **Collisions.** The simulator lets robots overlap and carry on, so a
          configuration that collides finishes in a time it could not have achieved
          without driving through its own fleet.
        * **An unfinished run.** ``makespan_ms`` is the last completion minus the
          first release, so a configuration that jams with work outstanding reports
          the time of the last task it managed -- a *smaller* number the longer it
          is stuck. Printing that beside a configuration that finished would make
          the jam look like a win.

        Either way the page says so rather than quietly comparing against it.
        """
        makespan_a = summary["makespan_mean_a"]
        makespan_b = summary["makespan_mean_b"]
        collisions_a = int(summary["collision_total_a"])
        collisions_b = int(summary["collision_total_b"])

        def delta(a: float, b: float) -> float | None:
            return None if not a else round((b - a) / a * 100.0, 1)

        return {
            "scenario": request.scenario,
            "seeds": request.seeds,
            "robots": int(summary["robots"]),
            "tasks_total": int(summary["tasks_total"]),
            "configurations": [
                {
                    "id": "A",
                    **CONFIGURATIONS["A"],
                    "makespan_mean_ms": round(makespan_a),
                    "makespan_std_ms": round(summary["makespan_std_a"]),
                    "collisions": collisions_a,
                    "coordination_failures": int(summary["coordination_failures_a"]),
                    "deadlocks": int(summary["deadlock_cycles_a"]),
                    "runs_finished": int(summary["runs_finished_a"]),
                    "finished_all": int(summary["runs_finished_a"]) == request.seeds,
                    "tasks_done": int(summary["tasks_done_a"]),
                    "stopped_mean_ms": round(summary["stopped_time_mean_a"]),
                },
                {
                    "id": "B",
                    **CONFIGURATIONS["B"],
                    "makespan_mean_ms": round(makespan_b),
                    "makespan_std_ms": round(summary["makespan_std_b"]),
                    "collisions": collisions_b,
                    "coordination_failures": int(summary["coordination_failures_b"]),
                    "deadlocks": int(summary["deadlock_cycles_b"]),
                    "runs_finished": int(summary["runs_finished_b"]),
                    "finished_all": int(summary["runs_finished_b"]) == request.seeds,
                    "tasks_done": int(summary["tasks_done_b"]),
                    "stopped_mean_ms": round(summary["stopped_time_mean_b"]),
                },
            ],
            "delta": {
                "makespan_pct": delta(makespan_a, makespan_b),
                "collisions_pct": delta(collisions_a, collisions_b),
                "stopped_pct": delta(
                    summary["stopped_time_mean_a"], summary["stopped_time_mean_b"]
                ),
                "collisions_avoided": collisions_a - collisions_b,
            },
            "makespan_valid": (
                collisions_a == 0
                and int(summary["runs_finished_a"]) == request.seeds
                and int(summary["runs_finished_b"]) == request.seeds
            ),
            # AC-3's bar: B's mean makespan at or under 80% of A's.
            "target_ratio": 0.8,
            "meets_ac3": bool(makespan_a) and makespan_b <= 0.8 * makespan_a,
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "done_runs": self.done_runs,
                "total_runs": self.total_runs,
                "scenario": self.request.scenario if self.request else None,
                "seeds": self.request.seeds if self.request else None,
                "robots": self.request.robots if self.request else None,
                "error": self.error,
                "result": self.result,
            }


benchmark = BenchmarkController()


@app.post("/api/benchmark")
def start_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    if request.scenario not in scenarios.SCENARIOS:
        raise HTTPException(404, f"unknown scenario {request.scenario!r}")
    return benchmark.start(request)


@app.get("/api/benchmark")
def benchmark_status() -> dict[str, Any]:
    return benchmark.status()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/orders")
def orders_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "orders.html")


@app.get("/simulation")
def simulation_page() -> FileResponse:
    """The floor itself. Split from the overview because watching and reading are
    different jobs: the overview answers "is the fleet healthy", this answers
    "what is it doing right now"."""
    return FileResponse(FRONTEND_DIR / "simulation.html")


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


@app.get("/api/external/directive")
def external_directive() -> dict[str, Any]:
    """What the Webots supervisor should be running, if it can.

    Polled by webots/controllers/fleet_supervisor. A GET, deliberately: the
    dashboard answers questions and never places a call, so nothing here is a
    command channel into the world (FR-8.4, BR-7).
    """
    with controller.lock:
        directive = controller.directive
    return directive or {"generation": 0}


@app.post("/api/external/map")
def external_map(data: dict[str, Any]) -> dict[str, Any]:
    """The Webots supervisor announcing a run it hosts. Reading only: the
    dashboard receives geometry and frames; it sends nothing back."""
    if not controller.accept_external_map(data):
        raise HTTPException(409, "a run started from this dashboard holds the floor")
    return {"accepted": True}


@app.post("/api/external/frame")
def external_frame(data: dict[str, Any]) -> dict[str, Any]:
    if not controller.accept_external_frame(data):
        raise HTTPException(409, "send the map first, or this run is not the one the dashboard asked for")
    return {"accepted": True}


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
