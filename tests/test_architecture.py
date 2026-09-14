"""Architectural invariants enforced as tests rather than as comments.

Four claims in this project are structural: they hold because of how the code is
arranged, not because a value came out right at run time. A comment asserting
them decays; a test does not. Each one is a Critical requirement whose violation
would be silent:

* CON-7 / FR-2.9 / FR-5.9 -- no learned or random state on the arbitration path.
  Violating this does not break any test of behaviour. It breaks the *safety
  argument*: two robots with divergent learned models could both conclude they
  hold right of way, and the Appendix C proof would no longer apply.
* IF-4.1 / CON-2 -- every coordination message fits one 250-byte ESP-NOW frame.
* FR-8.4 / IF-1.6 / BR-7 -- the dashboard has no path that commands a robot.
* NFR-4.4 -- benchmark evidence comes from the seeded engine, not from Webots.

Tests for modules that do not exist yet skip with the phase that creates them,
so this file grows teeth as the build proceeds rather than passing vacuously.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core import config, scenarios

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = REPO_ROOT / "core"


def imported_modules(path: Path) -> set[str]:
    """Every module name imported by a Python file, from its AST.

    Static rather than dynamic because the question is whether the *source*
    admits a dependency, not whether one happened to be reached at run time.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def referenced_names(path: Path) -> set[str]:
    """Every bare and dotted name appearing in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def require_module(relative: str) -> Path:
    """Return a source path, or skip naming the phase that creates it."""
    phase = {
        "core/arbitration.py": "Phase 6",
        "core/reservation.py": "Phase 6",
        "core/planner_astar.py": "Phase 3",
        "core/traffic_model.py": "Phase 8",
        "communication/messages.py": "Phase 5",
        "web/backend/app.py": "Phase 13",
    }.get(relative, "a later phase")
    path = REPO_ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} is built in {phase}")
    return path


class TestArbitrationPurity:
    """CON-7, FR-2.9, FR-5.9: nothing learned or random decides right of way."""

    FORBIDDEN_MODULES = {
        "random",
        "secrets",
        "numpy",
        "numpy.random",
        "core.traffic_model",
        "traffic_model",
    }

    def test_arbitration_imports_no_learned_or_random_module(self) -> None:
        path = require_module("core/arbitration.py")
        offending = imported_modules(path) & self.FORBIDDEN_MODULES
        assert not offending, (
            f"core/arbitration.py imports {sorted(offending)}. FR-2.9 forbids the "
            f"traffic model from influencing arbitration, and FR-5.9 forbids any "
            f"randomized component, because desynchronised learned models could "
            f"otherwise let two robots both claim right of way (Appendix C)."
        )

    def test_reservation_imports_no_learned_or_random_module(self) -> None:
        """The reservation table feeds arbitration, so it inherits the rule."""
        path = require_module("core/reservation.py")
        offending = imported_modules(path) & self.FORBIDDEN_MODULES
        assert not offending, (
            f"core/reservation.py imports {sorted(offending)}; it feeds the "
            f"arbitration path and so must stay free of learned state (FR-2.9)"
        )

    def test_arbitration_uses_no_floating_point_literals(self) -> None:
        """CON-6 / FR-3.8: integer or fixed-point arithmetic inside the 5 ms
        deadline. A float literal is the cheapest detectable symptom."""
        path = require_module("core/arbitration.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        floats = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]
        assert not floats, (
            f"core/arbitration.py contains float literals {floats}; CON-6 and "
            f"FR-3.8 require integer or fixed-point arithmetic on this path"
        )

    SAFETY_PATH = (
        "core/arbitration.py",
        "core/reservation.py",
        "core/timewindows.py",
        "core/planner_timewindow.py",
    )
    """Every module a route booking or an entry decision passes through. The
    time-window planner and the resource table are now the safety path -- a plan is
    what keeps robots apart -- so they inherit every rule arbitration had."""

    @pytest.mark.parametrize("module", SAFETY_PATH)
    def test_safety_path_imports_no_learned_or_random_module(self, module: str) -> None:
        path = require_module(module)
        offending = imported_modules(path) & self.FORBIDDEN_MODULES
        assert not offending, (
            f"{module} imports {sorted(offending)}; it is on the safety path and must "
            f"stay free of learned state and randomness (FR-2.9, FR-5.9, CON-7)"
        )

    @pytest.mark.parametrize("module", SAFETY_PATH)
    def test_safety_path_uses_no_floating_point_literals(self, module: str) -> None:
        path = require_module(module)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        floats = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]
        assert not floats, (
            f"{module} contains float literals {floats}; CON-6 and FR-3.8 require "
            f"integer or fixed-point arithmetic on this path"
        )

    @pytest.mark.parametrize("module", SAFETY_PATH)
    def test_safety_path_never_consults_the_traffic_model(self, module: str) -> None:
        path = require_module(module)
        leaked = referenced_names(path) & {"ewma", "pheromone", "traffic", "learned"}
        assert not leaked, (
            f"{module} references {sorted(leaked)}, which suggests learned traffic "
            f"state has reached the safety path (FR-2.9)"
        )

    def test_traffic_model_is_never_consulted_by_arbitration(self) -> None:
        path = require_module("core/arbitration.py")
        names = referenced_names(path)
        leaked = names & {"ewma", "pheromone", "edge_cost", "traffic", "learned"}
        assert not leaked, (
            f"core/arbitration.py references {sorted(leaked)}, which suggests "
            f"learned traffic state has reached the safety path (FR-2.9)"
        )


class TestFrameSizeBudget:
    """IF-4.1 / CON-2: one ESP-NOW frame, 250 bytes of payload."""

    def test_every_message_type_fits_one_frame_at_both_id_widths(self) -> None:
        require_module("communication/messages.py")
        from communication import messages

        for bits in (8, 16):
            for name, size in messages.packed_sizes(edge_id_bits=bits).items():
                assert size <= config.MAX_FRAME_BYTES, (
                    f"{name} packs to {size} bytes at {bits}-bit ids, over the "
                    f"{config.MAX_FRAME_BYTES}-byte ESP-NOW limit (IF-4.1)"
                )

    def test_intent_stays_small(self) -> None:
        """INTENT is the only message on the continuous safety path, at 5 Hz per
        robot, so its size sets the airtime floor for the whole fleet."""
        require_module("communication/messages.py")
        from communication import messages

        size = messages.packed_sizes(edge_id_bits=8)["INTENT"]
        assert size <= 48, f"INTENT packs to {size} bytes; expected well under 48"


class TestDashboardIsReadOnly:
    """FR-8.4 / IF-1.6 / BR-7: no affordance that can command a robot."""

    def test_backend_has_no_broadcast_path_to_the_mesh(self) -> None:
        path = require_module("web/backend/app.py")
        source = path.read_text(encoding="utf-8")
        for forbidden in ("transport.broadcast", "UdpBroadcast(", "EspNowBridge("):
            assert forbidden not in source, (
                f"web/backend/app.py references {forbidden!r}. The read-only "
                f"guarantee of FR-8.4 is structural: the dashboard must hold no "
                f"send path to the robot mesh, not merely decline to use one."
            )


class TestScenarioIntegrity:
    def test_every_scenario_validates(self) -> None:
        for name in scenarios.SCENARIOS:
            scenarios.get(name)

    def test_benchmark_evidence_never_comes_from_webots(self) -> None:
        """NFR-4.4: every run must be reproducible from its seed."""
        for scenario in scenarios.SCENARIOS.values():
            if scenario.benchmarked:
                assert scenario.engine is scenarios.Engine.HEADLESS

    def test_benchmarked_scenarios_meet_the_seed_floor(self) -> None:
        """FR-10.7: at least ten seeds, reported as mean and std dev."""
        for scenario in scenarios.SCENARIOS.values():
            if scenario.benchmarked:
                assert scenario.seeds >= 10

    def test_exactly_one_scenario_produces_the_graded_evidence(self) -> None:
        benchmarked = [s.name for s in scenarios.SCENARIOS.values() if s.benchmarked]
        assert benchmarked == ["bench3"]

    def test_simulated_scenarios_meet_the_three_amr_mandate(self) -> None:
        """CON-9 / AC-5. hardware2 is exempt: the roadmap caps the physical
        demonstrator at two or three robots."""
        for scenario in scenarios.SCENARIOS.values():
            if scenario.physical_robots == 0:
                assert scenario.robots >= 3

    def test_wide_ids_are_used_only_where_the_map_needs_them(self) -> None:
        """The 16-bit override is a documented deviation from ASM-2, so it must
        not spread to scenarios that fit the 8-bit wire format."""
        from core.graph import Graph

        for scenario in scenarios.SCENARIOS.values():
            graph = Graph.load(scenario.map_path)
            needs_wide = (
                len(graph.edges) > config.MAX_EDGES_8BIT
                or len(graph.nodes) > config.MAX_NODES_8BIT
            )
            assert (scenario.edge_id_bits == 16) == needs_wide, (
                f"{scenario.name}: edge_id_bits={scenario.edge_id_bits} but its "
                f"map has {len(graph.nodes)} nodes / {len(graph.edges)} edges"
            )

    def test_every_scenario_names_the_requirements_it_discharges(self) -> None:
        for scenario in scenarios.SCENARIOS.values():
            assert scenario.requirements, f"{scenario.name} traces to nothing"
            assert len(scenario.purpose) > 40

    def test_with_robots_rejects_a_nonsense_count(self) -> None:
        with pytest.raises(ValueError):
            scenarios.BENCH3.with_robots(0)

    def test_clamping_leaves_headless_scenarios_alone(self) -> None:
        for scenario in (scenarios.BENCH3, scenarios.SCALE100):
            assert scenarios.clamped(scenario).robots == scenario.robots

    def test_clamping_an_unprobed_host_does_not_invent_a_number(self) -> None:
        """An unprobed host should be told to run the probe, not handed a guess."""
        if scenarios.probed_webots_capacity() is None:
            assert scenarios.clamped(scenarios.VISUAL30).robots == 30


class TestCoreIsPure:
    """core/ holds decision logic only: no I/O, no wall clock, no simulator.

    This is what makes NFR-4.4 (reproducible from seed) achievable at all. A
    single time.time() call inside core/ would make runs differ between
    machines while every behavioural test still passed.
    """

    ALLOWED_WALL_CLOCK_FILES: set[str] = set()

    def test_core_never_reads_a_wall_clock(self) -> None:
        offenders = []
        for path in sorted(CORE_DIR.glob("*.py")):
            if path.name in self.ALLOWED_WALL_CLOCK_FILES:
                continue
            imports = imported_modules(path)
            if imports & {"time", "datetime", "time.time", "time.monotonic"}:
                offenders.append(path.name)
        assert not offenders, (
            f"{offenders} read a wall clock. All time in core/ must arrive as an "
            f"injected now_ms on the aligned fleet clock, or runs stop being "
            f"reproducible from their seed (NFR-4.4)."
        )

    def test_core_never_imports_the_simulator_or_the_web_app(self) -> None:
        """The dependency arrow points one way: simulator drives core, never the
        reverse. IF-3.1 requires the same decision code under Webots, the
        headless engine and real hardware."""
        offenders = {}
        for path in sorted(CORE_DIR.glob("*.py")):
            leaked = {
                name
                for name in imported_modules(path)
                if name.split(".")[0] in {"simulator", "web", "benchmark", "gateway"}
            }
            if leaked:
                offenders[path.name] = sorted(leaked)
        assert not offenders, f"core/ imports outward: {offenders}"

    def test_core_does_no_network_or_file_io(self) -> None:
        """Map loading is the one exception, and it is confined to graph.py."""
        allowed = {"graph.py", "scenarios.py"}
        offenders = {}
        for path in sorted(CORE_DIR.glob("*.py")):
            if path.name in allowed:
                continue
            leaked = imported_modules(path) & {"socket", "json", "pathlib", "urllib"}
            if leaked:
                offenders[path.name] = sorted(leaked)
        assert not offenders, (
            f"core/ performs I/O: {offenders}. Decision logic must be a pure "
            f"function of its inputs so it can run identically on the headless "
            f"engine, under Webots, and on an ESP32 (IF-3.1, NFR-4.7)."
        )
