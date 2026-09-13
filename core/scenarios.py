"""Scenario declarations.

The project must demonstrate four different things, and they have genuinely
different engineering needs: a controlled experiment, a jury-facing visual, a
fleet-scale proof, and a physical demonstrator. They are *configurations of one
system*, not four systems. Everything downstream -- the simulator, the
transport, the message codec, the benchmark runner, the web app -- reads a
``Scenario`` object and adapts, so the coordination logic in ``core/`` is
written exactly once (IF-3.1, NFR-4.7).

Adding a fifth demonstration should mean adding an entry to ``SCENARIOS`` and
nothing else. If it ever requires a branch inside ``core/``, that is a design
defect, not a new scenario.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core import config

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS_DIR = REPO_ROOT / "maps"
CONFIG_DIR = REPO_ROOT / "config"

WEBOTS_CAPACITY_FILE = CONFIG_DIR / "webots_capacity.json"
"""Written by scripts/probe_webots.py. Absent until the probe has run, which is
why every consumer treats it as optional rather than required."""


class Engine(str, Enum):
    """What moves the robots."""

    HEADLESS = "headless"
    """Our own deterministic 2D kinematic engine. Owns a virtual clock, is
    seeded, and is the only engine whose output is admissible as benchmark
    evidence (NFR-4.4)."""

    WEBOTS = "webots"
    """Webots 3D. Visual only. Wall-clock bound and not reproducible from a
    seed, so benchmark numbers never come from here."""


class TransportKind(str, Enum):
    """How robots reach each other. Same messages in every case."""

    INPROC = "inproc"
    """Deterministic in-process bus. Message-passing only -- no shared state
    between robots -- so decentralization is preserved while runs stay
    reproducible."""

    UDP = "udp"
    """Real UDP broadcast between OS processes, one process per robot. This is
    the configuration that demonstrates genuine decentralization, and it mirrors
    ESP-NOW's connectionless broadcast nature (IF-4.2, IF-4.5)."""

    ESPNOW = "espnow"
    """Serial bridge to a gateway MCU that relays onto real ESP-NOW, so
    physical robots join the same mesh as simulated ones."""


@dataclass(frozen=True)
class Scenario:
    """One demonstration configuration.

    Frozen because a scenario is an input to a reproducible run. Anything that
    varies per run -- the seed, an operator-chosen robot count -- is passed
    alongside it rather than mutated into it.
    """

    name: str
    purpose: str
    robots: int
    map_name: str
    engine: Engine
    transport: TransportKind
    edge_id_bits: int = config.EDGE_ID_BITS_DEFAULT
    zoned: bool = False
    """Whether ANNOUNCE is zone-scoped (FR-9.3). False for small fleets, where
    section 4.9 of the SRS says zoning adds complexity without benefit and a
    flood auction is the correct choice."""

    benchmarked: bool = False
    """Whether this scenario runs the section 6.3 A/B protocol. Only bench3
    produces AC-2 / AC-3 evidence."""

    seeds: int = 1
    """Seeds per configuration. FR-10.7 requires at least ten wherever the run
    is used as evidence."""

    physical_robots: int = 0
    """AMRs that are real hardware rather than simulated."""

    requirements: tuple[str, ...] = field(default_factory=tuple)
    """SRS identifiers this scenario exists to discharge. Carried so the run
    report can state what it was evidence *for*."""

    @property
    def map_path(self) -> Path:
        return MAPS_DIR / f"{self.map_name}.json"

    @property
    def max_nodes(self) -> int:
        return (
            config.MAX_NODES_8BIT
            if self.edge_id_bits == 8
            else config.MAX_NODES_16BIT
        )

    @property
    def max_edges(self) -> int:
        return (
            config.MAX_EDGES_8BIT
            if self.edge_id_bits == 8
            else config.MAX_EDGES_16BIT
        )

    @property
    def simulated_robots(self) -> int:
        return self.robots - self.physical_robots

    def with_robots(self, robots: int) -> "Scenario":
        """Return a copy with a different fleet size.

        Used by the Webots clamp and by Run Control, which lets an operator
        choose a smaller fleet than the scenario nominally declares.
        """
        if robots < 1:
            raise ValueError(f"robot count must be positive, got {robots}")
        return Scenario(**{**self.__dict__, "robots": robots})

    def validate(self) -> None:
        """Check the scenario is internally coherent. Raises on failure.

        Called by tests and by the run entry points, so an impossible scenario
        fails at start-up rather than halfway through a benchmark.
        """
        if self.edge_id_bits not in (8, 16):
            raise ValueError(
                f"{self.name}: edge_id_bits must be 8 or 16, got "
                f"{self.edge_id_bits}"
            )
        if self.robots < 3 and self.physical_robots == 0:
            # CON-9 / AC-5: the mandated deliverable is at least three AMRs.
            # A purely physical scenario is exempt; the roadmap explicitly caps
            # the hardware demonstrator at two or three robots.
            raise ValueError(
                f"{self.name}: simulated scenarios need at least 3 AMRs "
                f"(CON-9, AC-5), got {self.robots}"
            )
        if self.benchmarked and self.seeds < 10:
            raise ValueError(
                f"{self.name}: FR-10.7 requires at least 10 seeds for a "
                f"benchmarked scenario, got {self.seeds}"
            )
        if self.physical_robots > self.robots:
            raise ValueError(
                f"{self.name}: physical_robots ({self.physical_robots}) "
                f"exceeds robots ({self.robots})"
            )
        if self.engine is Engine.WEBOTS and self.benchmarked:
            # Webots is wall-clock bound and not seed-reproducible, so it
            # cannot satisfy NFR-4.4. Guarding this in code stops a future
            # change from quietly sourcing graded numbers from the 3D engine.
            raise ValueError(
                f"{self.name}: benchmark evidence must come from the headless "
                f"engine (NFR-4.4), not Webots"
            )


BENCH3 = Scenario(
    name="bench3",
    purpose=(
        "Controlled experiment producing the graded evidence: Configuration A "
        "(stop-and-wait) against Configuration B (full ROBOTON) on an "
        "identical map, task set and seed set."
    ),
    robots=3,
    map_name="benchmark_map",
    engine=Engine.HEADLESS,
    transport=TransportKind.INPROC,
    zoned=False,  # section 4.9: at 3-5 AMRs a flood auction is correct
    benchmarked=True,
    seeds=10,
    requirements=("AC-2", "AC-3", "AC-5", "FR-10.1", "FR-10.4", "FR-10.7", "NFR-2.1"),
)

VISUAL30 = Scenario(
    name="visual30",
    purpose=(
        "Jury-facing visual proof that coordination holds at fleet density, "
        "rendered in Webots 3D at whatever robot count the host sustains."
    ),
    robots=30,
    map_name="warehouse_zoned_30",
    engine=Engine.WEBOTS,
    transport=TransportKind.INPROC,
    zoned=True,
    seeds=1,
    requirements=("AC-5", "AC-6", "FR-9.3", "IF-3.1"),
)

SCALE100 = Scenario(
    name="scale100",
    purpose=(
        "Fleet-scale proof that auction traffic stays local to a zone, so cost "
        "per task scales with fleet size divided by zone count rather than "
        "with fleet size."
    ),
    robots=100,
    map_name="warehouse_zoned_100",
    engine=Engine.HEADLESS,
    transport=TransportKind.INPROC,
    edge_id_bits=16,  # see config.EDGE_ID_BITS_DEFAULT for the SRS defect note
    zoned=True,
    seeds=3,
    requirements=("FR-9.1", "FR-9.5", "FR-9.6", "NFR-1.12"),
)

HARDWARE2 = Scenario(
    name="hardware2",
    purpose=(
        "Physical demonstrator: two real AMRs joining the same mesh through "
        "the ESP-NOW bridge, running the same decision code as simulation."
    ),
    robots=2,
    physical_robots=2,
    map_name="benchmark_map",
    engine=Engine.HEADLESS,
    transport=TransportKind.ESPNOW,
    zoned=False,
    seeds=1,
    requirements=("IF-3.2", "IF-3.3", "FR-7.4", "NFR-4.7"),
)

SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in (BENCH3, VISUAL30, SCALE100, HARDWARE2)
}


def get(name: str) -> Scenario:
    """Look up a scenario by name, validating it before returning."""
    try:
        scenario = SCENARIOS[name]
    except KeyError:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"unknown scenario {name!r}; known scenarios: {known}") from None
    scenario.validate()
    return scenario


def probed_webots_capacity() -> int | None:
    """Max Webots robot count this host sustains, or None if unprobed.

    Written by scripts/probe_webots.py. Returning None rather than a guessed
    default is deliberate: an unprobed host should be told to run the probe, not
    handed a number nobody measured.
    """
    if not WEBOTS_CAPACITY_FILE.exists():
        return None
    try:
        data = json.loads(WEBOTS_CAPACITY_FILE.read_text(encoding="utf-8"))
        value = int(data["max_robots"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return value if value >= 1 else None


def clamped(scenario: Scenario) -> Scenario:
    """Reduce a Webots scenario to the fleet size this host can sustain.

    Non-Webots scenarios pass through untouched -- the headless engine has no
    equivalent ceiling, and scale100's whole point is the count it declares.
    """
    if scenario.engine is not Engine.WEBOTS:
        return scenario
    capacity = probed_webots_capacity()
    if capacity is None or capacity >= scenario.robots:
        return scenario
    return scenario.with_robots(capacity)
