"""Peer table (FE-1: FR-1.4, FR-1.5, FR-6.5) and the route horizon it retains.

The horizon fields are what make *anticipation* possible. Without them a robot can
only discover traffic ahead with a proximity sensor, and reactive proximity halting is
exactly how section 1.5 defines stop-and-wait -- the behaviour this project exists to
replace. Everything needed to avoid it is already in the INTENT frame.
"""

from __future__ import annotations

from core import config
from core.peers import Peer, PeerTable


def seen(table: PeerTable, robot_id: int, *, now_ms: int = 0, node: int = 2,
         nodes: tuple[int, ...] = (), etas: tuple[int, ...] = (),
         state: int = 3, priority: int = 10) -> Peer:
    return table.observe(
        robot_id=robot_id, now_ms=now_ms, current_node=node, priority=priority,
        state=state, battery_pct=100, held_task_id=-1,
        next_nodes=nodes, eta_ms=etas,
    )


class TestLiveness:
    def test_a_new_peer_is_discovered_from_its_first_heartbeat(self) -> None:
        """FR-2.7's counterpart, and NFR-4.6: adding an AMR reconfigures nothing."""
        table = PeerTable()
        seen(table, 3)
        assert 3 in table
        assert table.ids == (3,)

    def test_a_silent_peer_is_declared_lost(self) -> None:
        """FR-1.5: 1500 ms of silence. There is no death notification -- a robot that
        lost power cannot send one -- so silence is the only signal."""
        table = PeerTable()
        seen(table, 3, now_ms=0)
        assert table.reap(config.PEER_TIMEOUT_MS) == []
        lost = table.reap(config.PEER_TIMEOUT_MS + 1)
        assert [entry.robot_id for entry in lost] == [3]
        assert 3 not in table

    def test_a_lost_peer_carries_what_fr_6_5_needs(self) -> None:
        """The last node and held task must survive the removal to be usable."""
        table = PeerTable()
        table.observe(
            robot_id=4, now_ms=0, current_node=7, priority=10, state=3,
            battery_pct=90, held_task_id=12,
        )
        lost = table.reap(config.PEER_TIMEOUT_MS + 1)[0]
        assert (lost.last_node, lost.held_task_id) == (7, 12)

    def test_peers_are_reported_in_id_order(self) -> None:
        """FR-5.8: iteration order must not vary between robots."""
        table = PeerTable()
        for robot_id in (7, 2, 5):
            seen(table, robot_id)
        assert table.ids == (2, 5, 7)

    def test_forgetting_a_peer_is_idempotent(self) -> None:
        table = PeerTable()
        seen(table, 3)
        table.forget(3)
        table.forget(3)
        assert len(table) == 0


class TestRouteHorizon:
    def test_the_declared_route_is_retained(self) -> None:
        table = PeerTable()
        peer = seen(table, 3, nodes=(10, 11, 4), etas=(1000, 8000, 15000))
        assert peer.next_nodes == (10, 11, 4)
        assert peer.heading_to == 10

    def test_a_peer_declaring_nothing_is_heading_nowhere(self) -> None:
        """A parked robot. Distinguishable from one about to move, which is what lets
        a follower tell traffic from an obstacle."""
        table = PeerTable()
        assert seen(table, 3).heading_to == -1

    def test_arrival_is_looked_up_by_node(self) -> None:
        table = PeerTable()
        peer = seen(table, 3, nodes=(10, 11, 4), etas=(1000, 8000, 15000))
        assert peer.arrival_at(11) == 8000
        assert peer.arrival_at(99) is None

    def test_direction_distinguishes_following_from_opposing(self) -> None:
        """A peer coming the other way along a bidirectional aisle is in the other lane
        and is not traffic to follow."""
        table = PeerTable()
        peer = seen(table, 3, node=2, nodes=(10,), etas=(5000,))
        assert peer.is_on_edge(2, 10)
        assert not peer.is_on_edge(10, 2)

    def test_a_horizon_shorter_than_three_is_handled(self) -> None:
        """Near the end of a route there are fewer than three junctions left."""
        table = PeerTable()
        peer = seen(table, 3, nodes=(10,), etas=(1000,))
        assert peer.next_nodes == (10,)
        assert peer.arrival_at(10) == 1000

    def test_a_fresh_heartbeat_replaces_the_previous_horizon(self) -> None:
        """A robot that replanned must not appear to intend both routes."""
        table = PeerTable()
        seen(table, 3, nodes=(10, 11), etas=(1000, 2000))
        peer = seen(table, 3, now_ms=200, nodes=(1, 8), etas=(1200, 2200))
        assert peer.next_nodes == (1, 8)
