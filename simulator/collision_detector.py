"""Geometric collision detection (FR-10.1).

The division of labour here is the one the implementation guide insists on: the
simulator reports that two robots overlapped, and *our* code decides whether that
overlap means the coordination algorithm failed. So this module produces facts
with enough context attached to support that judgement -- which robots, where,
when, and what each was doing -- and does not itself rule on blame.

Two design points matter for the metric to mean anything:

* **One event per contiguous overlap, not one per tick.** Two robots wedged
  together for two seconds is one collision, not a hundred. Counting per tick
  would make the collision count a function of the tick rate, and AC-2's "zero
  collisions" claim would rest on a number nobody could interpret.
* **Spatial bucketing rather than all-pairs.** At 100 robots an all-pairs check is
  4,950 distance computations every 20 ms tick. Bucketing by the collision
  distance makes it proportional to the number of robots actually near each
  other, which is what keeps scale100 tractable in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core import config
from core.state_machine import MOVEMENT_STATES, State


@dataclass(frozen=True, slots=True)
class Pose:
    """Where one robot is, and what it was doing there."""

    robot_id: int
    x_mm: int
    y_mm: int
    state: State
    node: int
    edge_id: int | None


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    """One contiguous overlap between two robots.

    ``both_moving`` and ``either_yielding`` are recorded so the benchmark can
    distinguish a genuine arbitration failure from an artefact. A robot parked in
    FAULT that another robot drives into is a different defect from two MOVING
    robots that both believed they held right of way, and NFR-2.1 is about the
    second.
    """

    at_ms: int
    robot_a: int
    robot_b: int
    x_mm: int
    y_mm: int
    separation_mm: int
    state_a: State
    state_b: State
    node_a: int
    node_b: int

    @property
    def both_moving(self) -> bool:
        return self.state_a in MOVEMENT_STATES and self.state_b in MOVEMENT_STATES

    @property
    def either_yielding(self) -> bool:
        return State.YIELD in (self.state_a, self.state_b)

    @property
    def is_coordination_failure(self) -> bool:
        """Whether this overlap indicts the coordination layer.

        Both robots moving under their own decisions means arbitration let two
        robots into the same space at the same time, which is exactly what FE-5
        exists to prevent and what NFR-2.1 counts. An overlap involving a faulted
        or charging robot is a different problem -- still logged, but it is not
        evidence about arbitration.
        """
        return self.both_moving

    @property
    def pair(self) -> tuple[int, int]:
        return (self.robot_a, self.robot_b)

    def __str__(self) -> str:
        verdict = "COORDINATION FAILURE" if self.is_coordination_failure else "contact"
        return (
            f"{self.at_ms:>8} ms  {verdict}: r{self.robot_a}({self.state_a.value}) "
            f"and r{self.robot_b}({self.state_b.value}) at "
            f"({self.x_mm},{self.y_mm}) separated by {self.separation_mm} mm"
        )


@dataclass
class CollisionDetector:
    """Stateful detector: remembers which pairs are currently overlapping."""

    threshold_mm: int = config.COLLISION_DISTANCE_MM
    events: list[CollisionEvent] = field(default_factory=list)
    _active: set[tuple[int, int]] = field(default_factory=set, repr=False)

    @property
    def collision_count(self) -> int:
        """Total overlaps logged, including non-coordination contacts."""
        return len(self.events)

    @property
    def coordination_failures(self) -> list[CollisionEvent]:
        """The subset NFR-2.1 and AC-2 are about."""
        return [e for e in self.events if e.is_coordination_failure]

    def check(self, at_ms: int, poses: list[Pose]) -> list[CollisionEvent]:
        """Find overlaps among ``poses``, returning only newly begun ones.

        Robots not physically present -- a killed peer, for TC-5 -- are simply
        omitted from ``poses`` by the caller rather than flagged here.
        """
        overlapping: set[tuple[int, int]] = set()
        new_events: list[CollisionEvent] = []

        for a, b, separation in self._candidate_pairs(poses):
            pair = (min(a.robot_id, b.robot_id), max(a.robot_id, b.robot_id))
            overlapping.add(pair)
            if pair in self._active:
                continue  # same contiguous overlap, already counted
            first, second = (a, b) if a.robot_id < b.robot_id else (b, a)
            event = CollisionEvent(
                at_ms=at_ms,
                robot_a=first.robot_id,
                robot_b=second.robot_id,
                x_mm=(first.x_mm + second.x_mm) // 2,
                y_mm=(first.y_mm + second.y_mm) // 2,
                separation_mm=separation,
                state_a=first.state,
                state_b=second.state,
                node_a=first.node,
                node_b=second.node,
            )
            self.events.append(event)
            new_events.append(event)

        # Pairs that have separated become eligible to count again.
        self._active = overlapping
        return new_events

    def _candidate_pairs(self, poses: list[Pose]):
        """Yield overlapping pairs using a uniform spatial hash.

        Bucket width is the collision threshold, so any overlapping pair shares a
        bucket or lies in adjacent ones, and checking a bucket against its
        forward neighbours is sufficient.
        """
        if len(poses) < 2:
            return

        width = max(1, self.threshold_mm)
        buckets: dict[tuple[int, int], list[Pose]] = {}
        for pose in poses:
            buckets.setdefault((pose.x_mm // width, pose.y_mm // width), []).append(pose)

        # Forward-only neighbour offsets, so each pair of buckets is visited once.
        neighbourhood = ((0, 0), (1, 0), (0, 1), (1, 1), (1, -1))
        threshold_sq = self.threshold_mm * self.threshold_mm

        for (cx, cy), here in buckets.items():
            for dx, dy in neighbourhood:
                there = buckets.get((cx + dx, cy + dy))
                if there is None:
                    continue
                if dx == 0 and dy == 0:
                    pairs = (
                        (here[i], here[j])
                        for i in range(len(here))
                        for j in range(i + 1, len(here))
                    )
                else:
                    pairs = ((a, b) for a in here for b in there)
                for a, b in pairs:
                    gap_x = a.x_mm - b.x_mm
                    gap_y = a.y_mm - b.y_mm
                    distance_sq = gap_x * gap_x + gap_y * gap_y
                    if distance_sq < threshold_sq:
                        yield a, b, _isqrt(distance_sq)

    def reset(self) -> None:
        self.events.clear()
        self._active.clear()


def _isqrt(value: int) -> int:
    """Integer square root. Kept integer to match the rest of the stack."""
    import math

    return math.isqrt(value)
