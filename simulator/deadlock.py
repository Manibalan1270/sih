"""Deadlock diagnosis: who is waiting on whom, and is it a cycle.

Built because reading a frozen fleet off a dump of positions and states was tried
repeatedly and got the diagnosis wrong twice. A position tells you a robot has
stopped; it does not tell you which other robot it stopped *for*, and without that
the wait-for cycle has to be guessed. Two guesses cost a reverted fix each.

The wait-for graph is the standard construction from deadlock detection: an edge
from A to B means "A cannot move until B moves". A cycle in that graph is a
deadlock in the strict sense -- every member is waiting on a member, so no member
can ever be the one to move first, and no amount of waiting resolves it.

A chain is not a deadlock. A waiting on B waiting on C, with C free to move, clears
on its own. Distinguishing the two matters: a chain means "slow", a cycle means
"broken", and they call for completely different responses.

This is diagnostic only. Nothing here influences a robot's decisions, so it cannot
affect NFR-4.4's reproducibility or bring non-deterministic state onto the
arbitration path (CON-7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.robot import Robot
from core.state_machine import State


@dataclass(frozen=True, slots=True)
class WaitEdge:
    """``waiter`` cannot move until ``blocker`` does."""

    waiter: int
    blocker: int
    kind: str
    resource: str
    detail: str = ""

    def __str__(self) -> str:
        return (
            f"r{self.waiter} --{self.kind}--> r{self.blocker} "
            f"({self.resource}{'; ' + self.detail if self.detail else ''})"
        )


@dataclass(frozen=True, slots=True)
class WaitCycle:
    """A closed loop in the wait-for graph: a genuine deadlock."""

    edges: tuple[WaitEdge, ...]

    @property
    def robots(self) -> tuple[int, ...]:
        return tuple(edge.waiter for edge in self.edges)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted({edge.kind for edge in self.edges}))

    def describe(self) -> str:
        loop = " -> ".join(f"r{edge.waiter}" for edge in self.edges)
        lines = [f"  cycle: {loop} -> r{self.edges[0].waiter}"]
        for edge in self.edges:
            lines.append(f"    {edge}")
        return "\n".join(lines)


@dataclass
class StallReport:
    """What a frozen fleet is doing, and whether it can ever recover."""

    at_ms: int
    stationary: tuple[int, ...]
    cycles: tuple[WaitCycle, ...]
    edges: tuple[WaitEdge, ...]
    unexplained: tuple[int, ...] = ()
    """Robots that are stationary with no recorded reason.

    Worth its own field: a robot stopped for no reason it can name is a different
    and usually worse defect than one caught in a cycle -- it means some path holds
    position without going through any of the deliberate wait points."""

    @property
    def is_deadlocked(self) -> bool:
        return bool(self.cycles)

    @property
    def waiting_robots(self) -> tuple[int, ...]:
        return tuple(sorted({edge.waiter for edge in self.edges}))

    def describe(self) -> str:
        lines = [
            f"stall at {self.at_ms} ms: {len(self.stationary)} robots stationary "
            f"({', '.join(f'r{i}' for i in self.stationary)})"
        ]
        if self.cycles:
            lines.append(
                f"{len(self.cycles)} wait-for cycle(s) -- this is a deadlock, not "
                f"congestion:"
            )
            for cycle in self.cycles:
                lines.append(cycle.describe())
        elif self.edges:
            lines.append(
                "no cycle: every waiting robot is waiting on one that can still "
                "move, so this should clear on its own."
            )
            for edge in self.edges:
                lines.append(f"    {edge}")
        else:
            lines.append("no robot reported a reason for waiting.")
        if self.unexplained:
            lines.append(
                "stationary with no recorded cause: "
                + ", ".join(f"r{i}" for i in self.unexplained)
                + " -- some path holds position without a declared wait point."
            )
        return "\n".join(lines)


def wait_edges(robots: list[Robot]) -> list[WaitEdge]:
    """Build the wait-for graph from what each robot recorded this tick.

    Only edges naming a real robot are kept. A robot holding for something that is
    not a peer -- low position confidence, say -- cannot be part of a cycle, because
    nothing another robot does will release it.
    """
    present = {robot.robot_id for robot in robots}
    edges: list[WaitEdge] = []
    for robot in sorted(robots, key=lambda r: r.robot_id):
        cause = robot.wait_cause
        if cause is None or cause.blocker_id < 0:
            continue
        if cause.blocker_id not in present or cause.blocker_id == robot.robot_id:
            continue
        edges.append(
            WaitEdge(
                waiter=robot.robot_id,
                blocker=cause.blocker_id,
                kind=cause.kind,
                resource=cause.resource,
                detail=cause.detail,
            )
        )
    return edges


def find_cycles(edges: list[WaitEdge]) -> list[WaitCycle]:
    """Every distinct cycle in the wait-for graph.

    Each robot has at most one outgoing edge -- it waits on one thing at a time --
    so the graph is a functional graph and cycle-finding is a walk from each node
    until it repeats. That is far simpler than general cycle enumeration and is why
    the wait cause is recorded as a single value rather than a set.
    """
    out: dict[int, WaitEdge] = {edge.waiter: edge for edge in edges}
    cycles: list[WaitCycle] = []
    seen_members: set[int] = set()

    for start in sorted(out):
        if start in seen_members:
            continue
        path: list[WaitEdge] = []
        visited: dict[int, int] = {}
        node = start
        while node in out and node not in visited:
            visited[node] = len(path)
            edge = out[node]
            path.append(edge)
            node = edge.blocker
        if node in visited:
            loop = tuple(path[visited[node] :])
            if not any(set(loop) == set(existing.edges) for existing in cycles):
                cycles.append(WaitCycle(edges=loop))
            seen_members |= {edge.waiter for edge in loop}
    return cycles


def stationary_robots(robots: list[Robot], moved_recently: set[int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            robot.robot_id
            for robot in robots
            if robot.robot_id not in moved_recently and robot.state is not State.IDLE
        )
    )


def diagnose(
    robots: list[Robot], *, at_ms: int, moved_recently: set[int]
) -> StallReport:
    """Produce a report for a fleet that appears to have stopped."""
    edges = wait_edges(robots)
    still = stationary_robots(robots, moved_recently)
    explained = {edge.waiter for edge in edges}
    return StallReport(
        at_ms=at_ms,
        stationary=still,
        cycles=tuple(find_cycles(edges)),
        edges=tuple(edges),
        unexplained=tuple(sorted(set(still) - explained)),
    )


@dataclass
class StallWatcher:
    """Tracks which robots have moved lately, and reports when none have.

    Kept outside the engine's decision path on purpose: it observes, and a
    diagnostic that could change behaviour would be worse than none at all.
    """

    window_ms: int = 5000
    """How long a fleet must be motionless before this counts as a stall.

    Longer than any legitimate hold. A robot waiting out a junction conflict pauses
    for a safety margin (800 ms) or a corridor traverse (a few seconds); five
    seconds of *nothing* moving anywhere is not normal operation."""

    _last_distance: dict[int, int] = field(default_factory=dict, repr=False)
    _last_moved_ms: dict[int, int] = field(default_factory=dict, repr=False)

    def observe(self, robots: list[Robot], now_ms: int) -> None:
        for robot in robots:
            travelled = robot.metrics.distance_mm
            if self._last_distance.get(robot.robot_id) != travelled:
                self._last_distance[robot.robot_id] = travelled
                self._last_moved_ms[robot.robot_id] = now_ms
            self._last_moved_ms.setdefault(robot.robot_id, now_ms)

    def moved_recently(self, now_ms: int) -> set[int]:
        return {
            robot_id
            for robot_id, when in self._last_moved_ms.items()
            if now_ms - when < self.window_ms
        }

    def is_stalled(self, robots: list[Robot], now_ms: int) -> bool:
        """Whether nothing has moved for the whole window.

        Robots parked in IDLE or CHARGING are excluded: a fleet with no work is
        motionless and perfectly healthy, and reporting that as a deadlock would
        make the detector cry wolf on every completed run.
        """
        working = [
            robot
            for robot in robots
            if robot.state not in (State.IDLE, State.CHARGING, State.FAULT)
        ]
        if not working:
            return False
        recent = self.moved_recently(now_ms)
        return all(robot.robot_id not in recent for robot in working)

    def report(self, robots: list[Robot], now_ms: int) -> StallReport:
        return diagnose(
            robots, at_ms=now_ms, moved_recently=self.moved_recently(now_ms)
        )
