"""The order gateway (FR-4.1, IF-3.2, IF-3.3).

Two things are being defended here. First, that an order actually becomes work:
posting one has to end with a robot delivering it, or the page is a decoration.
Second, that the gateway refuses the endpoints the project has already been
bitten by -- a task endpoint on a junction (defect 13) parks a robot in a
conflict region it cannot step out of, and one in a bay puts cargo where idle
AMRs live.

Nothing here names a robot, and ``tests/test_architecture.py`` enforces that the
module cannot.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core import scenarios
from core.task import MAX_PRIORITY
from gateway import orders
from simulator import scenario as scenario_module
from simulator.scenario import AuctionAllocator
from web.backend.app import app, controller


@pytest.fixture
def sim():
    return scenario_module.build(
        scenarios.get("bench3"), seed=1, allocator=AuctionAllocator(), task_count=2, waves=1
    )


def two_stations(sim) -> tuple[int, int]:
    legal = orders.stations(sim)
    assert len(legal) >= 2, "the benchmark map must offer at least two endpoints"
    return legal[0].node, legal[-1].node


class TestStations:
    def test_every_offered_station_is_a_spur(self, sim) -> None:
        """Defect 13's well-formedness condition, applied to what an operator
        can even select: a degree-one leaf is the one place a robot may dwell."""
        for station in orders.stations(sim):
            assert len(list(sim.graph.neighbours(station.node))) == 1

    def test_bays_and_chargers_are_not_offered(self, sim) -> None:
        offered = {station.node for station in orders.stations(sim)}
        for node_id in sim.graph.nodes:
            node = sim.graph.node(node_id)
            if node.is_parking or node.is_charger:
                assert node_id not in offered

    def test_declared_pickups_and_drops_are_labelled(self, sim) -> None:
        kinds = {station.kind for station in orders.stations(sim)}
        assert {"pickup", "drop"} <= kinds


class TestAnOrderBecomesWork:
    def test_a_posted_order_is_announced_claimed_and_delivered(self, sim) -> None:
        """End to end: the gateway only appends to the task set, and everything
        after that is the fleet's own machinery -- announcement, auction, route,
        delivery. If any link were missing this would time out rather than pass."""
        pickup, drop = two_stations(sim)
        task = orders.submit(sim, pickup=pickup, drop=drop, priority=200)

        assert sim.run(max_ms=1_800_000), "the fleet did not finish the posted order"
        delivered = {t.task_id for t in sim.completed}
        assert task.task_id in delivered
        done = next(t for t in sim.completed if t.task_id == task.task_id)
        assert done.pickup == pickup and done.drop == drop

    def test_the_order_ages_from_when_it_was_accepted(self, sim) -> None:
        """BR-2's aging term measures from created_at_ms. Back-dating an order to
        zero would have it instantly outrank work that has genuinely been
        waiting, which is the opposite of anti-starvation."""
        for _ in range(200):
            sim.step()
        assert sim.engine.now_ms > 0
        pickup, drop = two_stations(sim)
        task = orders.submit(sim, pickup=pickup, drop=drop, priority=10)
        assert task.created_at_ms == sim.engine.now_ms

    def test_ids_never_repeat(self, sim) -> None:
        """A reused id reads as a re-announcement of finished work to every robot
        that keyed its auction state by it."""
        pickup, drop = two_stations(sim)
        posted = [
            orders.submit(sim, pickup=pickup, drop=drop, priority=10).task_id
            for _ in range(5)
        ]
        everything = [t.task_id for t in sim.task_set.tasks]
        assert len(posted) == len(set(posted))
        assert len(everything) == len(set(everything))

    def test_the_task_set_grows_so_the_run_is_not_finished_early(self, sim) -> None:
        """Simulation.is_finished counts distinct ids against the task set, so an
        order appended anywhere else would let a run report success while the
        operator's work sat undone."""
        before = len(sim.task_set)
        pickup, drop = two_stations(sim)
        orders.submit(sim, pickup=pickup, drop=drop, priority=10)
        assert len(sim.task_set) == before + 1


class TestRefusals:
    def test_a_junction_endpoint_is_refused(self, sim) -> None:
        junction = next(
            node for node in sim.graph.nodes
            if len(list(sim.graph.neighbours(node))) >= 3
        )
        _, drop = two_stations(sim)
        with pytest.raises(orders.OrderRejected, match="degree-one spur"):
            orders.submit(sim, pickup=junction, drop=drop, priority=10)

    def test_a_bay_endpoint_is_refused(self, sim) -> None:
        bay = next(iter(sim.graph.parking_nodes))
        pickup, _ = two_stations(sim)
        with pytest.raises(orders.OrderRejected, match="bay or charger"):
            orders.submit(sim, pickup=pickup, drop=bay, priority=10)

    def test_a_journey_to_itself_is_refused(self, sim) -> None:
        pickup, _ = two_stations(sim)
        with pytest.raises(orders.OrderRejected, match="not a journey"):
            orders.submit(sim, pickup=pickup, drop=pickup, priority=10)

    def test_an_unknown_node_is_refused(self, sim) -> None:
        _, drop = two_stations(sim)
        with pytest.raises(orders.OrderRejected, match="not on map"):
            orders.submit(sim, pickup=9999, drop=drop, priority=10)

    @pytest.mark.parametrize("priority", [-1, MAX_PRIORITY + 1])
    def test_a_priority_outside_asm17_is_refused_not_clamped(self, sim, priority) -> None:
        """Clamping would have the fleet run an order the operator did not place."""
        pickup, drop = two_stations(sim)
        with pytest.raises(orders.OrderRejected, match="outside"):
            orders.submit(sim, pickup=pickup, drop=drop, priority=priority)

    def test_a_refused_order_leaves_the_task_set_alone(self, sim) -> None:
        before = len(sim.task_set)
        _, drop = two_stations(sim)
        with pytest.raises(orders.OrderRejected):
            orders.submit(sim, pickup=9999, drop=drop, priority=10)
        assert len(sim.task_set) == before


class TestOrderLog:
    def test_receipts_reset_when_the_run_changes(self, sim) -> None:
        log = orders.OrderLog()
        log.bind(sim)
        pickup, drop = two_stations(sim)
        log.record(orders.submit(sim, pickup=pickup, drop=drop, priority=10))
        assert len(log.receipts) == 1

        other = scenario_module.build(scenarios.get("bench3"), seed=2, task_count=2)
        log.bind(other)
        assert log.receipts == [], "receipts from a previous run point at other ids"


class TestTheHttpSurface:
    @pytest.fixture(autouse=True)
    def _run(self, sim):
        # Assigned rather than started so the simulation does not advance under
        # the test: what is being checked is the gateway, not the pacing thread.
        controller.sim = sim
        controller.finished = False
        yield
        controller.stop()
        controller.sim = None

    def test_an_order_is_accepted_and_listed(self, sim) -> None:
        pickup, drop = two_stations(sim)
        with TestClient(app) as client:
            posted = client.post(
                "/api/orders", json={"pickup": pickup, "drop": drop, "priority": 42}
            )
            assert posted.status_code == 200
            receipt = posted.json()["order"]
            assert (receipt["pickup"], receipt["drop"], receipt["priority"]) == (pickup, drop, 42)

            listed = client.get("/api/orders").json()["orders"]
            assert [o["task_id"] for o in listed] == [receipt["task_id"]]

    def test_stations_are_served_for_the_form(self, sim) -> None:
        with TestClient(app) as client:
            body = client.get("/api/orders/stations").json()
        assert body["map"] == sim.graph.name
        assert {s["node"] for s in body["stations"]} == {
            s.node for s in orders.stations(sim)
        }

    def test_a_rejected_order_says_why(self, sim) -> None:
        _, drop = two_stations(sim)
        with TestClient(app) as client:
            refused = client.post(
                "/api/orders", json={"pickup": 9999, "drop": drop, "priority": 10}
            )
        assert refused.status_code == 400
        assert "not on map" in refused.json()["detail"]

    def test_a_priority_outside_the_range_never_reaches_the_fleet(self, sim) -> None:
        before = len(sim.task_set)
        pickup, drop = two_stations(sim)
        with TestClient(app) as client:
            refused = client.post(
                "/api/orders", json={"pickup": pickup, "drop": drop, "priority": 900}
            )
        assert refused.status_code == 422  # rejected by the schema, before the gateway
        assert len(sim.task_set) == before


class TestWithoutARun:
    @pytest.fixture(autouse=True)
    def _no_run(self):
        controller.sim = None
        controller.finished = False
        yield
        controller.stop()

    def test_posting_without_a_run_is_refused(self) -> None:
        with TestClient(app) as client:
            refused = client.post("/api/orders", json={"pickup": 1, "drop": 2})
        assert refused.status_code == 409

    def test_stations_without_a_run_are_refused(self) -> None:
        with TestClient(app) as client:
            assert client.get("/api/orders/stations").status_code == 409
