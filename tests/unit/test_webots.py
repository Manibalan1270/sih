"""The Webots visual path: a generated world and the supervisor that drives it.

Webots itself is not needed here. The world is text, and the supervisor's only
Webots dependency is the ``controller`` package, which a stub stands in for so
the placement logic can be exercised headless.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

from core import scenarios
from scripts import generate_world
from simulator import scenario as scenario_module
from simulator.engine import spawn_positions

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def world() -> str:
    return generate_world.build_world(scenarios.get("visual30"), seed=0)


class TestGeneratedWorld:

    def test_header_and_no_network_protos(self, world: str) -> None:
        assert world.startswith("#VRML_SIM R2025a utf8")
        assert "EXTERNPROTO" not in world, "a world that needs the network cannot be demoed offline"

    def test_one_solid_per_robot_in_engine_order(self, world: str) -> None:
        scenario = scenarios.get("visual30")
        sim = scenario_module.build(scenario, seed=0)
        defs = re.findall(r"^DEF AMR_(\d+) Solid \{", world, flags=re.MULTILINE)
        assert [int(d) for d in defs] == [r.robot_id for r in sim.engine.robots]
        # And each starts where the engine spawns it.
        homes = spawn_positions(sim.graph, len(sim.engine.robots), seed=0)
        for robot_id, home in zip(defs, homes):
            node = sim.graph.node(home)
            block = world.split(f"DEF AMR_{robot_id} Solid {{")[1].split("}")[0]
            assert f"translation {node.x_mm / 1000:g} {node.y_mm / 1000:g}" in block.replace("\n", " ")

    def test_every_station_has_a_pad_at_its_map_coordinate(self, world: str) -> None:
        raw = json.loads(scenarios.get("visual30").map_path.read_text(encoding="utf-8"))
        by_id = {n["id"]: n for n in raw["nodes"]}
        for kind, ids in (("pickup", raw["pickup_nodes"]), ("drop", raw["drop_nodes"])):
            for node_id in ids:
                node = by_id[node_id]
                pattern = (
                    rf"translation {node['x'] / 1000:g} {node['y'] / 1000:g} [\d.]+\s+children.*?"
                    rf'name "{kind}_{node["name"]}"'
                )
                assert re.search(pattern, world, flags=re.DOTALL), f"{kind} {node['name']} has no pad"

    def test_racks_fill_the_storage_grid(self, world: str) -> None:
        raw = json.loads(scenarios.get("visual30").map_path.read_text(encoding="utf-8"))
        cols, rows = raw["grid"]["cols"], raw["grid"]["rows"]
        assert world.count('name "rack_') == (cols - 1) * (rows - 1)

    def test_bench3_world_has_a_hazard_edged_choke(self) -> None:
        world = generate_world.build_world(scenarios.get("bench3"), seed=1)
        raw = json.loads(scenarios.get("bench3").map_path.read_text(encoding="utf-8"))
        assert f'name "hazard_{raw["choke_edge"]}_1"' in world
        assert world.count('name "rack_') == 0

    def test_the_supervisor_is_the_only_controller(self, world: str) -> None:
        assert world.count("controller ") == 1
        assert 'controller "fleet_supervisor"' in world
        assert "supervisor TRUE" in world


# ---------------------------------------------------------------------------
# Supervisor against a stub of Webots' controller package
# ---------------------------------------------------------------------------


class _Field:
    def __init__(self) -> None:
        self.values: list = []

    def setSFVec3f(self, value):  # noqa: N802 -- Webots API name
        self.values.append(list(value))

    def setSFRotation(self, value):  # noqa: N802
        self.values.append(list(value))

    def setSFColor(self, value):  # noqa: N802
        self.values.append(list(value))


class _Node:
    def __init__(self) -> None:
        self.fields: dict[str, _Field] = {}

    def getField(self, name):  # noqa: N802
        return self.fields.setdefault(name, _Field())


class _Supervisor:
    """Enough of Supervisor to run the loop for a fixed number of steps."""

    steps_allowed = 25
    nodes: dict[str, _Node] = {}
    labels: list[str] = []

    def __init__(self) -> None:
        self.stepped = 0

    def getBasicTimeStep(self):  # noqa: N802
        return 20.0

    def getFromDef(self, name):  # noqa: N802
        return self.nodes.setdefault(name, _Node())

    def step(self, _ms):
        self.stepped += 1
        return 0 if self.stepped <= self.steps_allowed else -1

    def setLabel(self, _id, text, *_rest):  # noqa: N802
        self.labels.append(text)

    def simulationQuit(self, _code):  # noqa: N802
        self.steps_allowed = 0

    def exportImage(self, path, _quality):  # noqa: N802
        Path(path).write_bytes(b"")


@pytest.fixture
def supervisor_module(monkeypatch):
    stub = types.ModuleType("controller")
    stub.Supervisor = _Supervisor
    _Supervisor.nodes = {}
    _Supervisor.labels = []
    monkeypatch.setitem(sys.modules, "controller", stub)
    path = REPO_ROOT / "webots" / "controllers" / "fleet_supervisor" / "fleet_supervisor.py"
    spec = importlib.util.spec_from_file_location("fleet_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSupervisor:
    def test_it_places_every_robot_every_step(self, supervisor_module) -> None:
        supervisor_module.main(["bench3", "--seed", "1", "--no-dashboard"])
        placed = {
            name: node.fields["translation"].values
            for name, node in _Supervisor.nodes.items()
            if name.startswith("AMR_") and not name.endswith(("_COLOR", "_CARGO"))
        }
        assert set(placed) == {"AMR_1", "AMR_2", "AMR_3"}
        for values in placed.values():
            assert len(values) == _Supervisor.steps_allowed
            assert all(v[2] == supervisor_module.ROBOT_Z for v in values)
        assert any("tasks" in label for label in _Supervisor.labels)

    def test_probe_writes_the_capacity_file(self, supervisor_module, monkeypatch, tmp_path) -> None:
        target = tmp_path / "webots_capacity.json"
        monkeypatch.setattr(scenarios, "WEBOTS_CAPACITY_FILE", target)
        monkeypatch.setattr(supervisor_module, "PROBE_SECONDS", 0.0)
        supervisor_module.main(["bench3", "--seed", "1", "--probe"])
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["measured_robots"] == 3
        assert 1 <= data["max_robots"] <= 3
