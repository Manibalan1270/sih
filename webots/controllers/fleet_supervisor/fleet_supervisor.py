"""Webots supervisor: renders a headless ROBOTON run in 3D.

Started by the world (see scripts/generate_world.py) with the scenario name and
seed as controllerArgs. Each Webots step advances the headless engine by the
basic time step and moves every AMR Solid to where the engine says it is. No
robot in the world has a controller of its own: the decision code is core/,
unchanged, the same code the benchmark measures (IF-3.1, NFR-4.7). Webots
contributes the picture and the wall clock, nothing else.

If the dashboard is up (web/backend/app.py, default http://127.0.0.1:8000) the
supervisor pushes the same telemetry frames to it, so the browser view and the
3D view describe one run.

    --probe   measure the real-time factor this host sustains at the world's
              fleet size for a few seconds, write config/webots_capacity.json
              (read by core.scenarios.clamped), and quit.

Webots runs this with the ``python`` on its PATH (Preferences > General >
Python command); any Python 3.10+ works, the controller library is pure Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from controller import Supervisor  # noqa: E402  (Webots' package)

from core import config, scenarios  # noqa: E402
from simulator import scenario as scenario_module  # noqa: E402
from web.backend import telemetry  # noqa: E402

MM = 0.001
ROBOT_Z = 0.15
PUSH_PERIOD_S = 0.1
DIRECTIVE_PERIOD_S = 1.0
"""How often to ask the dashboard what a human has asked for."""
LABEL_PERIOD_S = 0.25
PROBE_SECONDS = 8.0
SCREENSHOT_AFTER_S = 6.0

STATE_COLOURS = {
    "MOVING": (0.95, 0.66, 0.23),
    "PLANNING": (0.79, 0.60, 0.29),
    "BIDDING": (0.79, 0.60, 0.29),
    "YIELD": (0.50, 0.82, 0.91),
    "REPLAN": (0.50, 0.82, 0.91),
    "IDLE": (0.33, 0.39, 0.48),
    "AT_DROP": (0.84, 0.42, 0.63),
    "CHARGING": (0.48, 0.79, 0.44),
    "FAULT": (0.89, 0.33, 0.23),
}
LOW_BATTERY_PCT = 20


def world_file(name: str | None) -> Path | None:
    """The world that renders a scenario, if one has been generated for it.

    A world file fixes both the floor geometry and the number of AMR bodies in
    it, so a scenario can only be shown in its own world -- switching scenarios
    means loading another file, not rebuilding this one.
    """
    if not name:
        return None
    path = REPO_ROOT / "webots" / "worlds" / f"{name}.wbt"
    return path if path.exists() else None


class DashboardPush:
    """Fire-and-forget POSTs to the dashboard on a worker thread.

    A dashboard that is down costs one failed connect and then silence for a
    while; it never slows the simulation.
    """

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self._latest: dict | None = None
        self._map: dict | None = None
        self._last_map: dict | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._backoff_until = 0.0
        self.connected = False
        self.directive = 0
        """Generation of the run this supervisor is currently showing. Stamped on
        every push so the dashboard can tell an acknowledged run from a stale one."""
        threading.Thread(target=self._loop, daemon=True, name="dashboard-push").start()

    def send_map(self, data: dict) -> None:
        with self._lock:
            data = dict(data, directive=self.directive)
            self._map = data
            self._last_map = data
        self._wake.set()

    def send_frame(self, data: dict) -> None:
        with self._lock:
            data["directive"] = self.directive
            self._latest = data
        self._wake.set()

    def fetch_directive(self) -> dict | None:
        """What Run Control last asked for, or None if the dashboard is not there.

        A GET the supervisor makes: the dashboard never opens a connection to the
        world, so nothing about this is a channel from a browser to a robot.
        """
        try:
            with urllib.request.urlopen(self.base + "/api/external/directive", timeout=0.5) as reply:
                return json.loads(reply.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            return None

    def _post(self, path: str, data: dict) -> bool:
        body = json.dumps(data).encode("utf-8")
        request = urllib.request.Request(
            self.base + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=0.5):
                return True
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def _loop(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if time.time() < self._backoff_until:
                continue
            with self._lock:
                map_data, frame, self._latest = self._map, self._latest, None
            if map_data is not None:
                if self._post("/api/external/map", map_data):
                    with self._lock:
                        if self._map is map_data:
                            self._map = None
                    self.connected = True
                else:
                    self.connected = False
                    self._backoff_until = time.time() + 3.0
                    continue
            if frame is not None:
                ok = self._post("/api/external/frame", frame)
                self.connected = ok
                if not ok:
                    self._backoff_until = time.time() + 3.0
                    with self._lock:
                        if self._map is None:
                            self._map = self._last_map  # resend the map when it comes back


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROBOTON fleet supervisor")
    parser.add_argument("scenario", nargs="?", default="visual30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robots", type=int, default=None)
    parser.add_argument("--dashboard", default=os.environ.get("ROBOTON_DASHBOARD", "http://127.0.0.1:8000"))
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--max-ms", type=int, default=1_800_000)
    args = parser.parse_args(argv)
    # scripts/probe_webots.py sets this: the world's controllerArgs are fixed
    # text, so the probe switch has to come from outside the world file.
    args.probe = args.probe or os.environ.get("ROBOTON_PROBE") == "1"
    args.screenshot = os.environ.get("ROBOTON_SCREENSHOT") or None
    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    supervisor = Supervisor()
    step_ms = int(supervisor.getBasicTimeStep())
    ticks_per_step = max(1, step_ms // config.MOTION_TICK_MS)

    scenario = scenarios.get(args.scenario)
    push = None if (args.no_dashboard or args.probe) else DashboardPush(args.dashboard)

    # A world loaded to satisfy a directive adopts it before building anything,
    # so it comes up already showing the run that was asked for rather than the
    # seed frozen into its controllerArgs.
    seed, robot_count = args.seed, args.robots
    opening = push.fetch_directive() if push is not None else None
    if opening and opening.get("generation") and opening.get("scenario") == scenario.name:
        seed = opening.get("seed", seed)
        robot_count = opening.get("robots") or robot_count
        push.directive = opening["generation"]

    def build_run(seed: int, robots: int | None):
        return scenario_module.build(
            scenarios.clamped(scenario) if robots is None else scenario,
            seed=seed,
            robots=robots,
        )

    sim = build_run(seed, robot_count)
    fleet = list(sim.engine.robots)

    bodies, colours, cargo, last_state = {}, {}, {}, {}
    for robot in fleet:
        node = supervisor.getFromDef(f"AMR_{robot.robot_id}")
        if node is None:
            print(f"world has no AMR_{robot.robot_id}; regenerate with scripts/generate_world.py", file=sys.stderr)
            continue
        bodies[robot.robot_id] = (node.getField("translation"), node.getField("rotation"))
        colour = supervisor.getFromDef(f"AMR_{robot.robot_id}_COLOR")
        colours[robot.robot_id] = colour.getField("baseColor") if colour else None
        box = supervisor.getFromDef(f"AMR_{robot.robot_id}_CARGO")
        cargo[robot.robot_id] = box.getField("scale") if box else None

    if push is not None:
        push.send_map(telemetry.map_payload(sim))

    headings: dict[int, float] = {}
    last_push = last_label = last_directive = 0.0
    declined = 0   # a generation this world cannot show; do not keep retrying it
    screenshot_taken = False
    finished = False
    wall_start = time.perf_counter()
    sim_start_ms = sim.engine.now_ms
    limit_ms = sim.engine.now_ms + args.max_ms

    while supervisor.step(step_ms) != -1:
        # Run Control is the one place a run is started. This world follows it.
        if push is not None and time.perf_counter() - last_directive >= DIRECTIVE_PERIOD_S:
            last_directive = time.perf_counter()
            wanted = push.fetch_directive() or {}
            generation = wanted.get("generation", 0)
            if generation > push.directive and generation != declined:
                if wanted.get("scenario") == scenario.name:
                    # Same world: rebuild the run in place and re-drive the same
                    # bodies from the new engine. Nothing is respawned.
                    asked = wanted.get("robots") or args.robots
                    sim = build_run(wanted.get("seed", 0), min(asked, len(bodies)) if asked else None)
                    fleet = list(sim.engine.robots)
                    headings.clear()
                    last_state.clear()
                    finished = False
                    wall_start = time.perf_counter()
                    sim_start_ms = sim.engine.now_ms
                    limit_ms = sim.engine.now_ms + args.max_ms
                    push.directive = generation
                    push.send_map(telemetry.map_payload(sim))
                    print(f"adopted run {generation}: {scenario.name} seed {wanted.get('seed')}")
                elif world_file(wanted.get("scenario")) is not None:
                    # Another scenario means another world. Load it; the
                    # supervisor that comes up there reads the same directive.
                    other = world_file(wanted.get("scenario"))
                    print(f"loading {other.name} for run {generation}")
                    supervisor.worldLoad(str(other))
                    return
                else:
                    # Never stamp a generation this world is not showing: the
                    # dashboard would take these frames for the run it asked for.
                    declined = generation
                    print(
                        f"cannot show {wanted.get('scenario')}: no world file for it; "
                        f"staying on {scenario.name}. Generate one with "
                        f"scripts/generate_world.py {wanted.get('scenario')}",
                        file=sys.stderr,
                    )

        if not finished:
            for _ in range(ticks_per_step):
                if sim.is_finished or sim.engine.now_ms >= limit_ms:
                    finished = True
                    break
                sim.step()

        for robot in fleet:
            fields = bodies.get(robot.robot_id)
            if fields is None:
                continue
            translation, rotation = fields
            x, y = robot.footprint_mm()
            translation.setSFVec3f([x * MM, y * MM, ROBOT_Z])
            if robot.edge_id is not None and robot.next_node is not None:
                here, there = robot.graph.node(robot.current_node), robot.graph.node(robot.next_node)
                headings[robot.robot_id] = math.atan2(there.y_mm - here.y_mm, there.x_mm - here.x_mm)
            rotation.setSFRotation([0, 0, 1, headings.get(robot.robot_id, 0.0)])

            state = robot.state.value
            low = robot.battery_pct <= LOW_BATTERY_PCT
            key = (state, low, robot.task is not None and robot.task.leg.value == "TO_DROP")
            if last_state.get(robot.robot_id) != key:
                last_state[robot.robot_id] = key
                field = colours.get(robot.robot_id)
                if field is not None:
                    field.setSFColor(list((0.89, 0.33, 0.23) if low and state != "CHARGING" else STATE_COLOURS.get(state, STATE_COLOURS["IDLE"])))
                box = cargo.get(robot.robot_id)
                if box is not None:
                    box.setSFVec3f([1, 1, 1] if key[2] else [0.001, 0.001, 0.001])

        now = time.perf_counter()
        if push is not None and now - last_push >= PUSH_PERIOD_S:
            last_push = now
            frame = telemetry.snapshot(sim, finished=finished)
            frame["source"] = "webots"
            push.send_frame(frame)

        if now - last_label >= LABEL_PERIOD_S:
            last_label = now
            done = len({t.task_id for t in sim.completed})
            collisions = len(sim.engine.collisions)
            sim_s = sim.engine.now_ms / 1000
            factor = (sim.engine.now_ms - sim_start_ms) / 1000 / max(1e-6, now - wall_start)
            supervisor.setLabel(
                0,
                f"ROBOTON {scenario.name}  seed {args.seed}  {len(fleet)} AMRs   "
                f"t = {sim_s:7.1f} s   tasks {done}/{len(sim.task_set)}   "
                f"collisions {collisions}   x{factor:.1f} real time"
                + ("   FINISHED" if finished else "")
                + ("   dashboard linked" if push is not None and push.connected else ""),
                0.01, 0.01, 0.07, 0xF2A93B if collisions == 0 else 0xE4553B, 0.0, "Arial",
            )

        if args.screenshot and not screenshot_taken and now - wall_start >= SCREENSHOT_AFTER_S:
            screenshot_taken = True
            supervisor.exportImage(args.screenshot, 95)
            print(f"exported {args.screenshot}")

        if args.probe and now - wall_start >= PROBE_SECONDS:
            factor = (sim.engine.now_ms - sim_start_ms) / 1000 / (now - wall_start)
            sustained = len(fleet) if factor >= 1.0 else max(1, int(len(fleet) * factor))
            out = scenarios.WEBOTS_CAPACITY_FILE
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "max_robots": sustained,
                        "measured_robots": len(fleet),
                        "real_time_factor": round(factor, 3),
                        "scenario": scenario.name,
                        "webots_step_ms": step_ms,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"probe: {len(fleet)} AMRs at x{factor:.2f} real time -> max_robots {sustained}; wrote {out}")
            supervisor.simulationQuit(0)
            return


if __name__ == "__main__":
    main(sys.argv[1:])
