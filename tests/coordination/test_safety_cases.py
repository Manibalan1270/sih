"""SRS section 6.2 cases for the safety core: TC-1, TC-2, TC-3.

These are the Phase 6 and Phase 7 gates. TC-3 in particular is the executable
counterpart of the Appendix C proof: the proof shows a conflict set under a total
order has a unique maximum, so exactly one robot proceeds and a cyclic wait cannot
form. Here that is checked three ways -- the key is a total order, every small
conflict set has exactly one winner, and three robots on a single-lane ring
actually clear.
"""

from __future__ import annotations

import itertools

import pytest

from core import config, scenarios
from core.arbitration import (
    Arbiter,
    Outcome,
    crossing_window,
    outranks,
    ranking_key,
    unique_winner,
)
from core.reservation import IMPLIED, Reservation, ReservationTable
from core.state_machine import State
from simulator.scenario import AuctionAllocator, build


def coordinated(seed: int = 0, *, tasks: int | None = None):
    """A fleet with the full coordination stack: auction, mesh, arbitration."""
    return build(
        scenarios.get("bench3"),
        seed=seed,
        allocator=AuctionAllocator(),
        task_count=tasks,
    )


def table_with(*, junction: int, robot_id: int, priority: int, arrival_ms: int,
               from_node: int = 1, to_node: int = 9) -> ReservationTable:
    table = ReservationTable()
    table.record(
        Reservation(
            junction=junction,
            window=crossing_window(arrival_ms=arrival_ms),
            robot_id=robot_id,
            priority=priority,
            kind=IMPLIED,
            from_node=from_node,
            to_node=to_node,
        )
    )
    return table


@pytest.mark.tc
class TestTC1CrossingJunction:
    """Two AMRs plan routes crossing one junction with overlapping ETA windows.

    Expected: overlap detected before approach; the lower-ranked AMR sheds speed;
    no collision (FR-5.1 to FR-5.7).
    """

    def test_an_overlapping_window_is_detected(self) -> None:
        table = table_with(junction=5, robot_id=2, priority=10, arrival_ms=10_000)
        decision = Arbiter(robot_id=3).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=10_200),
            my_priority=10,
            table=table,
            can_absorb_shift=lambda _: True,
            from_node=3,
            to_node=8,
        )
        assert decision.yielded
        assert decision.shift_ms > 0

    def test_windows_beyond_the_margin_do_not_conflict(self) -> None:
        """FR-5.3 separates by the margin, not merely by intersection."""
        table = table_with(junction=5, robot_id=2, priority=10, arrival_ms=0)
        clear_at = config.JUNCTION_OCCUPANCY_MS + config.MARGIN_MS + 100
        decision = Arbiter(robot_id=3).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=clear_at),
            my_priority=10,
            table=table,
            can_absorb_shift=lambda _: True,
            from_node=3,
            to_node=8,
        )
        assert decision.outcome is Outcome.PROCEED

    def test_the_same_table_gives_both_robots_opposite_answers(self) -> None:
        """Appendix C's Theorem in miniature: exactly one of them may proceed."""
        answers: dict[int, Outcome] = {}
        for me, rival in ((2, 5), (5, 2)):
            table = table_with(
                junction=7, robot_id=rival, priority=10, arrival_ms=10_000
            )
            answers[me] = Arbiter(robot_id=me).arbitrate(
                junction=7,
                window=crossing_window(arrival_ms=10_000),
                my_priority=10,
                table=table,
                can_absorb_shift=lambda _: True,
                from_node=3,
                to_node=8,
            ).outcome
        assert answers[2] is Outcome.RESERVE
        assert answers[5].is_yield

    def test_priority_beats_identifier(self) -> None:
        """BR-1: higher task priority takes precedence at a junction."""
        table = table_with(junction=5, robot_id=1, priority=10, arrival_ms=10_000)
        decision = Arbiter(robot_id=9).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=10_000),
            my_priority=200,
            table=table,
            can_absorb_shift=lambda _: True,
            from_node=3,
            to_node=8,
        )
        assert decision.outcome is Outcome.RESERVE

    def test_a_yield_sheds_speed_rather_than_braking(self) -> None:
        """FR-5.6 / NFR-2.4: resolve by anticipation where anticipation suffices."""
        table = table_with(junction=5, robot_id=1, priority=200, arrival_ms=10_000)
        decision = Arbiter(robot_id=9).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=10_100),
            my_priority=10,
            table=table,
            can_absorb_shift=lambda _: True,
            from_node=3,
            to_node=8,
        )
        assert decision.outcome is Outcome.YIELD_SLOW

    def test_it_reroutes_when_slowing_cannot_absorb_the_shift(self) -> None:
        """Appendix B's second AVOID branch."""
        table = table_with(junction=5, robot_id=1, priority=200, arrival_ms=10_000)
        decision = Arbiter(robot_id=9).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=10_100),
            my_priority=10,
            table=table,
            can_absorb_shift=lambda _: False,
            has_alternative_route=True,
            from_node=3,
            to_node=8,
        )
        assert decision.outcome is Outcome.YIELD_REPLAN

    def test_it_holds_when_neither_slowing_nor_rerouting_is_available(self) -> None:
        """Not in Appendix B, and necessary: in a corridor with no bypass both of its
        AVOID branches are unavailable and the procedure has nothing left to do. A
        controlled stop is the only outcome that keeps FR-5.3's separation."""
        table = table_with(junction=5, robot_id=1, priority=200, arrival_ms=10_000)
        decision = Arbiter(robot_id=9).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=10_100),
            my_priority=10,
            table=table,
            can_absorb_shift=lambda _: False,
            has_alternative_route=False,
            from_node=3,
            to_node=8,
        )
        assert decision.outcome is Outcome.YIELD_HOLD

    def test_low_confidence_declines_a_contested_junction(self) -> None:
        """FR-5.14 / NFR-2.3: an AMR that cannot say where it is cannot promise when
        it will be somewhere."""
        table = table_with(junction=5, robot_id=1, priority=10, arrival_ms=10_000)
        decision = Arbiter(robot_id=2).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=10_000),
            my_priority=200,
            table=table,
            can_absorb_shift=lambda _: True,
            position_confident=False,
            from_node=3,
            to_node=8,
        )
        assert decision.outcome is Outcome.BLOCKED_LOW_CONFIDENCE

    def test_low_confidence_still_allows_an_uncontested_junction(self) -> None:
        """FR-5.14 forbids *claiming a reservation* without confidence, not moving."""
        decision = Arbiter(robot_id=2).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=10_000),
            my_priority=10,
            table=ReservationTable(),
            can_absorb_shift=lambda _: True,
            position_confident=False,
        )
        assert decision.outcome is Outcome.PROCEED

    def test_following_traffic_is_not_a_conflict(self) -> None:
        """Two robots entering from the same side queue; they do not contend."""
        table = table_with(
            junction=5, robot_id=1, priority=200, arrival_ms=10_000,
            from_node=3, to_node=8,
        )
        decision = Arbiter(robot_id=9).arbitrate(
            junction=5,
            window=crossing_window(arrival_ms=10_000),
            my_priority=10,
            table=table,
            can_absorb_shift=lambda _: True,
            from_node=3,
            to_node=8,
        )
        assert decision.outcome is Outcome.PROCEED

    @pytest.mark.slow
    def test_repeated_crossings_produce_no_collisions(self) -> None:
        """The Phase 6 gate, end to end (NFR-2.1)."""
        for seed in range(3):
            sim = coordinated(seed)
            assert sim.run(max_ms=1_800_000), f"seed {seed} did not finish"
            assert sim.engine.coordination_failures == [], (
                f"seed {seed}: {len(sim.engine.coordination_failures)} collisions"
            )

    @pytest.mark.slow
    def test_yielding_actually_happens(self) -> None:
        """Zero collisions reached by nobody ever contending would prove nothing."""
        sim = coordinated(0)
        sim.run(max_ms=1_800_000)
        assert sum(r.metrics.yields_lost for r in sim.engine.robots) > 0


@pytest.mark.tc
class TestTC2SingleLaneCorridor:
    """Two AMRs approach a single-lane corridor from opposite ends.

    Expected: the yield decision is taken at the last passing point before entry;
    no head-on standoff (FR-5.10, ASM-5).
    """

    @pytest.mark.slow
    def test_no_collision_inside_the_corridor(self) -> None:
        for seed in range(3):
            sim = coordinated(seed)
            sim.run(max_ms=1_800_000)
            choke = sim.graph.single_lane_edges[0]
            edge = sim.graph.edge(choke)
            span = sorted(
                (sim.graph.node(edge.u).x_mm, sim.graph.node(edge.v).x_mm)
            )
            for event in sim.engine.coordination_failures:
                assert not (span[0] <= event.x_mm <= span[1]), (
                    f"seed {seed}: collision inside the single-lane corridor"
                )

    @pytest.mark.slow
    def test_nobody_ever_yields_from_inside_the_corridor(self) -> None:
        """The decision must be complete before entry. Deciding once inside leaves no
        outcome available but a standoff, which is what FR-5.10 forbids."""
        sim = coordinated(0)
        choke = sim.graph.single_lane_edges[0]
        offences = 0
        for _ in range(90_000):
            sim.step()
            for robot in sim.engine.robots:
                if (
                    robot.edge_id == choke
                    and robot.state is State.YIELD
                    and robot.progress_mm > config.ENTRY_COMMIT_MM
                ):
                    offences += 1
            if sim.is_finished:
                break
        assert offences == 0, f"yielded {offences} times from inside the corridor"

    @pytest.mark.slow
    def test_the_corridor_is_used_in_both_directions(self) -> None:
        """Otherwise the head-on case was never exercised at all."""
        sim = coordinated(0)
        choke = sim.graph.single_lane_edges[0]
        directions: set[int] = set()
        for _ in range(90_000):
            sim.step()
            for robot in sim.engine.robots:
                if robot.edge_id == choke and robot.next_node is not None:
                    directions.add(robot.next_node)
            if sim.is_finished:
                break
        assert len(directions) == 2, f"corridor only traversed toward {directions}"

    @pytest.mark.slow
    def test_the_corridor_never_holds_opposing_traffic(self) -> None:
        """The invariant the corridor mechanism exists to hold.

        Not "one robot at a time": a single-lane aisle is one *lane*, so robots may
        queue through it nose to tail exactly as on a one-lane road, with headway
        (IF-2.4) keeping them apart. What it cannot hold is two robots facing each
        other, because neither can pass and neither can turn round -- that is the
        head-on FR-5.10 forbids.
        """
        sim = coordinated(0)
        choke = sim.graph.single_lane_edges[0]
        for _ in range(90_000):
            sim.step()
            inside = [
                r
                for r in sim.engine.robots
                if r.edge_id == choke and r.progress_mm > config.ENTRY_COMMIT_MM
            ]
            headings = {r.next_node for r in inside}
            assert len(headings) <= 1, (
                f"opposing traffic inside the corridor at {sim.engine.now_ms} ms: "
                f"{[(r.robot_id, r.current_node, r.next_node) for r in inside]}"
            )
            if sim.is_finished:
                break


@pytest.mark.tc
class TestTC3CyclicConflict:
    """Three AMRs form a cyclic conflict, each waiting on the next.

    Expected: the total order yields a unique maximum; the cycle cannot form; all
    three clear (FR-5.12, Appendix C).
    """

    def test_the_ranking_key_is_a_total_order(self) -> None:
        """Appendix C's Lemma 1: no ties, no incomparable pairs."""
        contenders = [(p, r) for p in (0, 10, 200, 255) for r in range(1, 12)]
        keys = [ranking_key(p, r) for p, r in contenders]
        assert len(set(keys)) == len(keys), "two distinct robots share a ranking key"

    def test_every_conflict_set_has_exactly_one_maximum(self) -> None:
        """Appendix C's Theorem, checked over every subset of a small population."""
        population = [(10, 1), (10, 2), (10, 3), (200, 4), (0, 5)]
        for size in range(1, len(population) + 1):
            for contenders in itertools.combinations(population, size):
                winner = unique_winner(list(contenders))
                assert winner is not None
                others = [c for c in contenders if c != winner]
                assert all(
                    ranking_key(*winner) > ranking_key(*other) for other in others
                ), f"{winner} is not the unique maximum of {contenders}"

    def test_no_cycle_can_form_in_a_three_way_conflict(self) -> None:
        """A cyclic wait needs every member waiting on a strictly greater element --
        a strictly increasing cycle in a total order, which cannot exist."""
        trio = [(10, 1), (10, 2), (10, 3)]
        waiting_on = {
            me: [o for o in trio if o != me and outranks(o[0], o[1], me[0], me[1])]
            for me in trio
        }
        proceeding = [me for me, rivals in waiting_on.items() if not rivals]
        assert len(proceeding) == 1, f"expected exactly one to proceed, got {proceeding}"

    def test_the_ordering_is_antisymmetric(self) -> None:
        """Both directions can never hold, or two robots would both proceed."""
        population = [(p, r) for p in (0, 10, 200) for r in (1, 2, 7)]
        for a, b in itertools.permutations(population, 2):
            assert not (
                outranks(a[0], a[1], b[0], b[1]) and outranks(b[0], b[1], a[0], a[1])
            )

    def test_an_empty_conflict_set_has_no_winner(self) -> None:
        assert unique_winner([]) is None

    @pytest.mark.slow
    def test_three_robots_on_a_single_lane_ring_all_clear(self) -> None:
        """The executable counterpart of Appendix C, on the map built for it.

        The loop map is three single-lane junction-to-junction edges in a ring with a
        spur at each junction, so robots sent round it must contend for every segment.
        If a cyclic wait were possible, this is where it would form.

        This was xfail for a while, on the belief that a ring of distinct corridors
        needed sequence-level reservation because each robot is the rightful winner of
        the segment it wants while blocked by another holding the next. That diagnosis
        was wrong. The ring was not deadlocking on resource ordering at all: robots
        were running flat -- nothing implemented Appendix A's charging cycle -- and
        finished robots were parking on charger nodes that were also task endpoints.
        Fixing those two made the ring clear with no change to arbitration.
        """
        ring = scenarios.Scenario(
            name="loop3",
            purpose=(
                "Three AMRs on a single-lane ring, the geometry Appendix C's "
                "deadlock-freedom proof is about."
            ),
            robots=3,
            map_name="loop_map",
            engine=scenarios.Engine.HEADLESS,
            transport=scenarios.TransportKind.INPROC,
            requirements=("FR-5.12", "TC-3"),
        )
        sim = build(ring, seed=1, allocator=AuctionAllocator(), task_count=6)
        assert sim.run(max_ms=1_800_000), (
            "three robots on a single-lane ring did not clear; a cyclic wait formed"
        )
        assert sim.engine.coordination_failures == []

    @pytest.mark.slow
    def test_arbitration_decisions_repeat_exactly(self) -> None:
        """FR-5.8 / NFR-2.5. Reproducibility is what makes the Appendix C argument
        checkable rather than merely plausible."""
        traces = []
        for _ in range(3):
            sim = coordinated(5)
            sim.run(max_ms=1_800_000)
            traces.append(
                [
                    str(decision)
                    for robot in sim.engine.robots
                    for decision in robot.arbiter.history
                ]
            )
        assert traces[0] == traces[1] == traces[2]
