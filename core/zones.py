"""Zone partitioning for fleet-scale auctions (FE-9).

At fleet scale the architecture does not change; only the *scope* of a task
announcement does. A flood auction costs roughly 2N frames per task, and because
2.4 GHz is a shared medium (CON-4) contention degrades worse than linearly, so
at 100 AMRs bids start being lost to the air rather than to better rivals.
Scoping ANNOUNCE to the task's zone and its neighbours bounds that cost.

Two things are deliberately *not* zoned:

* INTENT and RESERVE (FR-9.4). Their scope is the physical radio neighbourhood,
  which is already exactly the set of robots capable of colliding. Zoning them
  would shrink the safety horizon for no gain.
* Anything authoritative. A zone table is static data derived from the map and
  held identically by every robot (ASM-16), so there is no zone server to lose.
  FR-9.7 requires precisely that, and it is why zone assignment is a pure
  function of the map rather than a negotiated protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.graph import NO_ZONE, Graph


@dataclass(frozen=True, slots=True)
class ZoneMap:
    """Static zone partition of a warehouse graph.

    Every robot computes this independently from the same map file and gets the
    same answer, which is what lets it be consulted on the auction path without
    introducing shared state.
    """

    enabled: bool
    """False for small fleets. Section 4.9 of the SRS is explicit that at 3-5
    AMRs zoning adds complexity without benefit and a flood auction is correct,
    so bench3 and hardware2 run unzoned by design rather than by omission."""

    node_zone: dict[int, int]
    zone_nodes: dict[int, tuple[int, ...]]
    zone_adjacency: dict[int, frozenset[int]]
    """Zones joined by at least one edge. A zone is always its own neighbour, so
    ``eligible_to_bid`` needs no same-zone special case."""

    @property
    def zone_count(self) -> int:
        return len(self.zone_nodes)

    @property
    def zone_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.zone_nodes))

    def zone_of(self, node_id: int) -> int:
        """Zone containing ``node_id``, or ``NO_ZONE`` when unzoned."""
        return self.node_zone.get(node_id, NO_ZONE)

    def neighbours_of(self, zone_id: int) -> frozenset[int]:
        return self.zone_adjacency.get(zone_id, frozenset())

    def eligible_to_bid(self, robot_zone: int, task_zone: int) -> bool:
        """FR-4.14 / FR-9.3: may a robot in ``robot_zone`` bid on this task?

        Adjacent zones are included, not just the task's own, because a robot
        sitting just across a zone boundary may still hold the lowest insertion
        cost. Excluding it would make the partition distort allocation quality
        rather than merely bound traffic.
        """
        if not self.enabled:
            return True  # flood auction: every robot is eligible
        if task_zone == NO_ZONE or robot_zone == NO_ZONE:
            # A task or robot with no zone cannot be excluded on zone grounds.
            # Failing open matters: an unzoned task that nobody bids on would be
            # re-announced forever (FR-4.13) and never completed.
            return True
        return task_zone == robot_zone or task_zone in self.neighbours_of(robot_zone)

    def expected_bidders(self, fleet_size: int) -> int:
        """Rough count of robots hearing one ANNOUNCE, for the FR-9.6 budget.

        Assumes robots spread evenly across zones, which ASM-16's static
        assignment approximates. Used by the scale100 report to compare measured
        auction frames against the NFR-1.12 budget.
        """
        if not self.enabled or self.zone_count <= 1:
            return fleet_size
        mean_neighbourhood = sum(
            len(self.neighbours_of(z)) for z in self.zone_ids
        ) / self.zone_count
        per_zone = fleet_size / self.zone_count
        return max(1, round(per_zone * mean_neighbourhood))

    def assign(self, robot_id: int, home_node: int) -> int:
        """Boot-time zone assignment for a robot (FR-9.1, ASM-16).

        Derived from where the robot starts, so it is a pure function of the map
        and the spawn layout. ``robot_id`` is accepted for traceability in logs
        and to keep the signature stable if a future release needs it.
        """
        del robot_id
        return self.zone_of(home_node)

    @classmethod
    def from_graph(cls, graph: Graph, *, enabled: bool) -> "ZoneMap":
        """Build the partition declared by the map's per-node ``zone`` fields."""
        node_zone = {
            node.id: node.zone_id
            for node in graph.nodes.values()
            if node.zone_id != NO_ZONE
        }

        buckets: dict[int, list[int]] = {}
        for node_id, zone_id in node_zone.items():
            buckets.setdefault(zone_id, []).append(node_id)
        zone_nodes = {z: tuple(sorted(ns)) for z, ns in sorted(buckets.items())}

        adjacency: dict[int, set[int]] = {z: {z} for z in zone_nodes}
        for edge in graph.edges.values():
            zu, zv = node_zone.get(edge.u, NO_ZONE), node_zone.get(edge.v, NO_ZONE)
            if zu == NO_ZONE or zv == NO_ZONE or zu == zv:
                continue
            adjacency[zu].add(zv)
            adjacency[zv].add(zu)

        return cls(
            enabled=enabled and len(zone_nodes) > 1,
            node_zone=node_zone,
            zone_nodes=zone_nodes,
            zone_adjacency={z: frozenset(ns) for z, ns in adjacency.items()},
        )

    def validate(self, graph: Graph) -> None:
        """Check the partition is usable. Raises ``ValueError`` on failure."""
        if not self.enabled:
            return
        unassigned = sorted(set(graph.nodes) - set(self.node_zone))
        if unassigned:
            raise ValueError(
                f"{graph.name}: zoning enabled but nodes {unassigned} carry no "
                f"zone; ANNOUNCE for a task there could not be scoped (FR-9.2)"
            )
        isolated = [z for z in self.zone_ids if self.neighbours_of(z) == {z}]
        if isolated and self.zone_count > 1:
            raise ValueError(
                f"{graph.name}: zones {isolated} touch no other zone, so no "
                f"robot outside them can ever bid on their tasks (FR-9.3)"
            )
