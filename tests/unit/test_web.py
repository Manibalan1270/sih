"""The Fleet Dashboard backend (FR-8, Phase 12).

What is asserted here is the contract the browser relies on: the map payload
matches the map file, every robot in a frame carries a position and a battery,
and a run can be started, watched and paused. FR-8.4's read-only guarantee is
covered structurally in tests/test_architecture.py.
"""

from __future__ import annotations

import json
import time

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

    def test_run_control_offers_both_algorithms(self) -> None:
        """Run Control can drive either configuration, so the difference between
        them can be watched on the floor and not only read off the benchmark."""
        with TestClient(app) as client:
            default = client.post("/api/run", json={"scenario": "bench3", "speed": 64}).json()
            assert (default["config"], default["algorithm"]) == ("B", "ROBOTON")

            baseline = client.post(
                "/api/run", json={"scenario": "bench3", "speed": 64, "config": "A"}
            ).json()
            assert (baseline["config"], baseline["algorithm"]) == ("A", "Stop-and-wait")
            # Structural, not a flag the robot consults: configuration A is built
            # with no auctioneer and no arbiter at all (FR-10.5).
            fleet = controller.sim.engine.robots
            assert all(r.auctioneer is None and r.arbiter is None for r in fleet)

            assert client.post(
                "/api/run", json={"scenario": "bench3", "config": "C"}
            ).status_code == 422

    def test_an_unknown_scenario_is_refused(self) -> None:
        with TestClient(app) as client:
            assert client.post("/api/run", json={"scenario": "nope"}).status_code == 404

    def test_a_hardware_scenario_needs_a_fleet_size(self) -> None:
        with TestClient(app) as client:
            assert client.post("/api/run", json={"scenario": "hardware2"}).status_code == 400
            ok = client.post("/api/run", json={"scenario": "hardware2", "robots": 2})
            assert ok.status_code == 200 and ok.json()["robots"] == 2


class TestTheBenchmarkComparesLikeWithLike:
    """AC-2 / AC-3's A-against-B protocol, surfaced on the dashboard."""

    def test_both_configurations_run_the_same_task_sets(self) -> None:
        """``compare_configurations`` refuses a comparison whose two halves did not
        see the same work, which is what makes the difference attributable to the
        coordination rather than to luck."""
        with TestClient(app) as client:
            started = client.post(
                "/api/benchmark", json={"scenario": "bench3", "seeds": 1, "robots": 3}
            )
            assert started.status_code == 200
            assert started.json()["total_runs"] == 2

            deadline = time.time() + 120
            while time.time() < deadline:
                status = client.get("/api/benchmark").json()
                if status["state"] in ("done", "error"):
                    break
                time.sleep(0.2)

            assert status["state"] == "done", status.get("error")
            result = status["result"]
            a, b = result["configurations"]
            assert (a["id"], b["id"]) == ("A", "B")
            assert a["tasks_done"] and b["tasks_done"]
            assert result["delta"]["collisions_avoided"] == a["collisions"] - b["collisions"]

    def test_a_colliding_baseline_is_not_offered_as_a_makespan(self) -> None:
        """The honesty rule the page depends on. Robots overlap and carry on in this
        simulator, so a configuration that collides finishes in a time it could not
        have reached without driving through its own fleet; ``makespan_valid`` is
        what stops the page reporting that as a speed-up."""
        from web.backend.app import BenchmarkController, BenchmarkRequest

        def summary(collisions_a: int) -> dict:
            return {
                "makespan_mean_a": 300_000, "makespan_mean_b": 400_000,
                "makespan_std_a": 0, "makespan_std_b": 0,
                "collision_total_a": collisions_a, "collision_total_b": 0,
                "coordination_failures_a": collisions_a, "coordination_failures_b": 0,
                "deadlock_cycles_a": 0, "deadlock_cycles_b": 0,
                "stopped_time_mean_a": 1000, "stopped_time_mean_b": 2000,
                "tasks_total": 12, "tasks_done_a": 12, "tasks_done_b": 12,
                "runs_finished_a": 1, "runs_finished_b": 1, "robots": 3,
            }

        request = BenchmarkRequest(scenario="bench3", seeds=1, robots=3)
        dirty = BenchmarkController._present(summary(66), {}, request)
        assert dirty["makespan_valid"] is False
        assert dirty["delta"]["collisions_avoided"] == 66

        clean = BenchmarkController._present(summary(0), {}, request)
        assert clean["makespan_valid"] is True
        assert clean["meets_ac3"] is False  # 400 s is not within 80% of 300 s

    def test_a_jammed_baseline_is_not_offered_as_a_makespan_either(self) -> None:
        """``makespan_ms`` is the last completion minus the first release, so a
        configuration that jams with work outstanding reports the time of the last
        task it managed -- a smaller number the longer it is stuck. Beside a
        configuration that finished, that would read as a win."""
        from web.backend.app import BenchmarkController, BenchmarkRequest

        jammed = {
            "makespan_mean_a": 120_000, "makespan_mean_b": 400_000,
            "makespan_std_a": 0, "makespan_std_b": 0,
            "collision_total_a": 0, "collision_total_b": 0,
            "coordination_failures_a": 0, "coordination_failures_b": 0,
            "deadlock_cycles_a": 0, "deadlock_cycles_b": 0,
            "stopped_time_mean_a": 9000, "stopped_time_mean_b": 2000,
            "tasks_total": 12, "tasks_done_a": 7, "tasks_done_b": 12,
            "runs_finished_a": 0, "runs_finished_b": 1, "robots": 30,
        }
        result = BenchmarkController._present(
            jammed, {}, BenchmarkRequest(scenario="visual30", seeds=1, robots=30)
        )
        a, b = result["configurations"]
        assert a["finished_all"] is False and b["finished_all"] is True
        assert result["makespan_valid"] is False, (
            "a baseline that never delivered the task set has no makespan to compare"
        )


def test_run_request_defaults_match_the_cli() -> None:
    request = RunRequest()
    assert (request.scenario, request.seed, request.waves) == ("bench3", 1, 4)
    assert request.config == "B", "Run Control defaults to the coordinated fleet"


class TestExternalSource:
    """The Webots supervisor hosts its own run and pushes frames here."""

    @pytest.fixture(autouse=True)
    def _stop_after(self):
        # The controller is a process-wide singleton and ``dashboard_owned`` is
        # deliberately sticky for the life of a process, so a run started by an
        # earlier test would otherwise lock Webots out of this one.
        controller.dashboard_owned = False
        yield
        controller.stop()
        controller.external_map = controller.external_frame = None
        controller.dashboard_owned = False

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

    def test_a_resent_map_cannot_seize_a_run_started_here(self, sim) -> None:
        """A supervisor whose feed was refused re-announces its map on a timer.
        That retry used to stop the dashboard's own run and take the view back
        within seconds of every Start run; a human's click outranks it."""
        with TestClient(app) as client:
            client.post("/api/external/map", json=telemetry.map_payload(sim))
            client.post("/api/external/frame", json=telemetry.snapshot(sim))
            client.post("/api/run", json={"scenario": "bench3", "seed": 2, "speed": 64})

            assert client.post("/api/external/map", json=telemetry.map_payload(sim)).status_code == 409
            assert client.post("/api/external/frame", json=telemetry.snapshot(sim)).status_code == 409
            status = client.get("/api/status").json()
            assert status["source"] == "dashboard" and status["scenario"] == "bench3"

    def test_webots_takes_over_the_run_the_dashboard_asked_for(self, sim) -> None:
        """Start run publishes a directive and hosts the run meanwhile. A push
        stamped with that generation is Webots showing the same run, so the
        browser switches to it and the local copy is stood down -- one run, two
        views, rather than two runs disagreeing about the time."""
        with TestClient(app) as client:
            client.post("/api/external/map", json=telemetry.map_payload(sim))
            client.post("/api/external/frame", json=telemetry.snapshot(sim))
            client.post("/api/run", json={"scenario": "bench3", "seed": 2, "speed": 64})

            directive = client.get("/api/external/directive").json()
            assert directive["scenario"] == "bench3" and directive["seed"] == 2
            assert directive["generation"] > 0
            assert client.get("/api/status").json()["mirrored_by_webots"] is False

            stamped_map = dict(telemetry.map_payload(sim), directive=directive["generation"])
            assert client.post("/api/external/map", json=stamped_map).status_code == 200
            frame = dict(telemetry.snapshot(sim), source="webots",
                         directive=directive["generation"])
            assert client.post("/api/external/frame", json=frame).status_code == 200

            status = client.get("/api/status").json()
            assert status["source"] == "webots"
            assert status["mirrored_by_webots"] is True
            assert controller.sim is None or controller._thread is None

    def test_a_dashboard_run_replaces_a_stale_external_one(self, sim) -> None:
        with TestClient(app) as client:
            client.post("/api/external/map", json=telemetry.map_payload(sim))
            client.post("/api/external/frame", json=telemetry.snapshot(sim))
            client.post("/api/run", json={"scenario": "bench3", "seed": 2, "speed": 64})
            assert client.get("/api/status").json()["source"] == "dashboard"
