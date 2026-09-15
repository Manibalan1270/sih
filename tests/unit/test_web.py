"""The Fleet Dashboard backend (FR-8, Phase 12).

What is asserted here is the contract the browser relies on: the map payload
matches the map file, every robot in a frame carries a position and a battery,
and a run can be started, watched and paused. FR-8.4's read-only guarantee is
covered structurally in tests/test_architecture.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core import scenarios
from simulator import scenario as scenario_module
from web.backend import telemetry
from web.backend.app import RunRequest, app, controller


@pytest.fixture
def sim():
    run = scenario_module.build(scenarios.get("bench3"), seed=1)
    for _ in range(300):
        run.step()
    return run


class TestSnapshot:
    def test_every_robot_has_a_position_and_a_battery(self, sim) -> None:
        frame = telemetry.snapshot(sim)
        assert len(frame["robots"]) == len(sim.engine.robots)
        for robot in frame["robots"]:
            assert isinstance(robot["x"], int) and isinstance(robot["y"], int)
            assert 0 <= robot["battery"] <= 100
            assert robot["state"] in {
                "IDLE", "BIDDING", "PLANNING", "MOVING", "YIELD", "REPLAN",
                "AT_DROP", "CHARGING", "FAULT",
            }

    def test_task_holders_come_from_the_robots_not_the_announced_set(self, sim) -> None:
        """The announced set never learns who holds what; the robots do."""
        frame = telemetry.snapshot(sim)
        held = {t["id"]: t["holder"] for t in frame["tasks"] if t["status"] == "HELD"}
        expected = {
            task.task_id: robot.robot_id
            for robot in sim.engine.robots
            for task in robot.queue
        }
        assert held == expected
        assert held, "after 300 ticks of bench3 somebody holds a task"

    def test_map_payload_matches_the_map_file(self, sim) -> None:
        raw = json.loads(sim.scenario.map_path.read_text(encoding="utf-8"))
        payload = telemetry.map_payload(sim)
        assert {n["id"] for n in payload["nodes"]} == {n["id"] for n in raw["nodes"]}
        assert {e["id"] for e in payload["edges"]} == {e["id"] for e in raw["edges"]}
        assert {n["id"] for n in payload["nodes"] if n["pickup"]} == set(raw["pickup_nodes"])
        assert {n["id"] for n in payload["nodes"] if n["drop"]} == set(raw["drop_nodes"])
        assert sum(e["choke"] for e in payload["edges"]) == 1

    def test_a_frame_is_json_serialisable(self, sim) -> None:
        json.dumps(telemetry.snapshot(sim))
        json.dumps(telemetry.map_payload(sim))


class TestRunControl:
    @pytest.fixture(autouse=True)
    def _stop_after(self):
        yield
        controller.stop()

    def test_scenarios_are_listed(self) -> None:
        with TestClient(app) as client:
            names = {s["name"] for s in client.get("/api/scenarios").json()}
        assert names == set(scenarios.SCENARIOS)

    def test_no_run_means_no_map_and_no_state(self) -> None:
        with TestClient(app) as client:
            assert client.get("/api/map").status_code == 404
            assert client.get("/api/state").status_code == 404

    def test_a_run_starts_streams_and_pauses(self) -> None:
        with TestClient(app) as client:
            started = client.post(
                "/api/run", json={"scenario": "bench3", "seed": 1, "speed": 64}
            ).json()
            assert started["running"] and started["robots"] == 3

            deadline = time.time() + 5
            while client.get("/api/status").json()["now_ms"] == 0 and time.time() < deadline:
                time.sleep(0.05)

            with client.websocket_connect("/ws/telemetry") as socket:
                first = socket.receive_json()
                second = socket.receive_json()
            assert first["type"] == "map"
            assert second["type"] == "frame"
            assert len(second["data"]["robots"]) == 3
            assert second["data"]["now_ms"] > 0

            assert client.post("/api/pause").json()["paused"] is True
            frozen = client.get("/api/status").json()["now_ms"]
            time.sleep(0.2)
            assert client.get("/api/status").json()["now_ms"] == frozen
            assert client.post("/api/resume").json()["paused"] is False

    def test_an_unknown_scenario_is_refused(self) -> None:
        with TestClient(app) as client:
            assert client.post("/api/run", json={"scenario": "nope"}).status_code == 404

    def test_a_hardware_scenario_needs_a_fleet_size(self) -> None:
        with TestClient(app) as client:
            assert client.post("/api/run", json={"scenario": "hardware2"}).status_code == 400
            ok = client.post("/api/run", json={"scenario": "hardware2", "robots": 2})
            assert ok.status_code == 200 and ok.json()["robots"] == 2


def test_run_request_defaults_match_the_cli() -> None:
    request = RunRequest()
    assert (request.scenario, request.seed, request.waves) == ("bench3", 1, 4)


class TestExternalSource:
    """The Webots supervisor hosts its own run and pushes frames here."""

    @pytest.fixture(autouse=True)
    def _stop_after(self):
        yield
        controller.stop()
        controller.external_map = controller.external_frame = None

    def test_pushed_frames_are_what_the_browser_sees(self, sim) -> None:
        with TestClient(app) as client:
            assert client.post("/api/external/frame", json={"now_ms": 1}).status_code == 409
            assert client.post("/api/external/map", json=telemetry.map_payload(sim)).status_code == 200
            frame = telemetry.snapshot(sim)
            frame["source"] = "webots"
            assert client.post("/api/external/frame", json=frame).status_code == 200

            shown = client.get("/api/state").json()
            assert shown["source"] == "webots"
            assert shown["now_ms"] == sim.engine.now_ms
            assert client.get("/api/status").json()["source"] == "webots"
            assert len(client.get("/api/map").json()["nodes"]) == len(sim.graph.nodes)

    def test_a_dashboard_run_replaces_a_stale_external_one(self, sim) -> None:
        with TestClient(app) as client:
            client.post("/api/external/map", json=telemetry.map_payload(sim))
            client.post("/api/external/frame", json=telemetry.snapshot(sim))
            client.post("/api/run", json={"scenario": "bench3", "seed": 2, "speed": 64})
            assert client.get("/api/status").json()["source"] == "dashboard"


class TestTelemetryFallback:
    """The link to a run must not fail silently (FR-8.2).

    uvicorn started without a WebSocket implementation answers /ws/telemetry
    with 404. The socket then closes before it ever opens, and a page that only
    retries in silence shows "no run in progress" while the fleet is working --
    which is what this dashboard did. The REST views therefore have to carry
    everything the socket carries, and every page has to go through the shared
    client that falls back to them.
    """

    @pytest.fixture(autouse=True)
    def _stop_after(self):
        yield
        controller.stop()

    def test_rest_views_carry_what_the_socket_carries(self) -> None:
        """A page polling /api/map and /api/state can draw the same floor."""
        with TestClient(app) as client:
            client.post("/api/run", json={"scenario": "bench3", "seed": 1, "speed": 64})
            for _ in range(100):
                if client.get("/api/state").status_code == 200:
                    break
                time.sleep(0.02)

            frame = client.get("/api/state").json()
            map_data = client.get("/api/map").json()

            # Whatever the socket sends under "map" and "frame" is exactly these
            # two payloads, so a polling page is not a degraded one.
            assert map_data["nodes"] and map_data["edges"]
            assert frame["robots"] and "counters" in frame
            for robot in frame["robots"]:
                assert {"id", "x", "y", "state", "battery"} <= robot.keys()
            # The fallback refetches the map only when scenario+seed changes.
            assert frame["scenario"] == "bench3" and frame["seed"] == 1

    def test_every_page_uses_the_shared_client(self) -> None:
        """No page may open its own socket: the one in telemetry.js is the only
        one that falls back, so a private socket would reintroduce the silence."""
        frontend = Path(__file__).resolve().parents[2] / "web" / "frontend"
        shared = frontend / "telemetry.js"
        assert shared.is_file(), "web/frontend/telemetry.js is the shared client"
        for page in ("app.js", "dashboard.js", "orders.js"):
            source = (frontend / page).read_text(encoding="utf-8")
            assert "new WebSocket(" not in source, (
                f"web/frontend/{page} opens its own WebSocket. Page scripts must "
                f"go through Telemetry.connect() so that a socket which never "
                f"opens falls back to the REST views instead of showing an empty "
                f"floor."
            )
            assert "Telemetry.connect(" in source, f"{page} must use the shared client"

    def test_every_page_loads_the_shared_client_first(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "web" / "frontend"
        for page, script in (
            ("simulation.html", "app.js"),
            ("index.html", "dashboard.js"),
            ("orders.html", "orders.js"),
        ):
            html = (frontend / page).read_text(encoding="utf-8")
            assert html.index("telemetry.js") < html.index(script), (
                f"{page} must load telemetry.js before {script}"
            )
