"""Task set generation (FR-10.6, FR-10.9).

The task set is part of the experimental control. If it differs between
Configuration A and Configuration B, or if it fails to put enough traffic through
the choke corridor, the 20% claim measures something other than coordination.
"""

from __future__ import annotations

import json

import pytest

from benchmark import task_generator
from core.graph import Graph
from core.planner_astar import AStarPlanner
from core.task import MAX_PRIORITY
from tests.conftest import MAPS_DIR


def raw_map(name: str = "benchmark_map") -> dict:
    return json.loads((MAPS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def generate(graph: Graph, *, seed: int = 1, count: int = 24, **kwargs):
    return task_generator.generate_for_map(
        graph, raw_map(graph.name), seed=seed, count=count, **kwargs
    )


class TestFR109:
    """At least 60% of tasks must route through the choke corridor."""

    def test_choke_fraction_meets_the_requirement(self, benchmark_map: Graph) -> None:
        for seed in range(10):
            task_set = generate(benchmark_map, seed=seed)
            task_set.check_fr_10_9()
            assert task_set.choke_fraction >= 0.60

    def test_the_declared_fraction_matches_the_actual_routes(
        self, benchmark_map: Graph
    ) -> None:
        """The reported figure must be measured, not assumed."""
        task_set = generate(benchmark_map, seed=3, count=40)
        planner = AStarPlanner(benchmark_map)
        choke = raw_map()["choke_edge"]
        measured = sum(
            1
            for task in task_set
            if choke in planner.route_edges(planner.route(task.pickup, task.drop))
        )
        assert measured == task_set.choke_tasks

    def test_a_failing_task_set_is_rejected_loudly(self, benchmark_map: Graph) -> None:
        """A silently under-contended task set would produce a meaningless
        comparison rather than a visibly wrong one."""
        # Pickups and drops all on the left half never approach the corridor.
        task_set = task_generator.generate(
            benchmark_map, seed=1, count=10,
            choke_edge=4, pickup_nodes=[1, 2, 3], drop_nodes=[1, 2, 3],
        )
        assert task_set.choke_fraction == 0.0
        with pytest.raises(ValueError, match="FR-10.9"):
            task_set.check_fr_10_9()

    def test_a_map_with_no_choke_route_warns(self, benchmark_map: Graph) -> None:
        task_set = task_generator.generate(
            benchmark_map, seed=1, count=5,
            choke_edge=4, pickup_nodes=[1], drop_nodes=[2],
        )
        assert task_set.warnings
        assert "FR-10.9" in task_set.warnings[0]


class TestDeterminism:
    def test_the_same_seed_gives_the_same_task_set(self, benchmark_map: Graph) -> None:
        """FR-10.6: both configurations must be handed an identical task set."""
        a = generate(benchmark_map, seed=99)
        b = generate(benchmark_map, seed=99)
        assert [(t.task_id, t.pickup, t.drop, t.priority, t.created_at_ms) for t in a] == [
            (t.task_id, t.pickup, t.drop, t.priority, t.created_at_ms) for t in b
        ]

    def test_different_seeds_give_different_task_sets(self, benchmark_map: Graph) -> None:
        a = generate(benchmark_map, seed=1)
        b = generate(benchmark_map, seed=2)
        assert [(t.pickup, t.drop) for t in a] != [(t.pickup, t.drop) for t in b]

    def test_a_copy_shares_no_task_objects(self, benchmark_map: Graph) -> None:
        """Two configurations running the same task set must not be able to
        observe each other's claims through a shared object."""
        original = generate(benchmark_map, seed=5)
        duplicate = original.copy()
        assert all(a is not b for a, b in zip(original.tasks, duplicate.tasks))
        original.tasks[0].claim(robot_id=9, bid=100, now_ms=0)
        assert duplicate.tasks[0].holder == -1


class TestWaveRelease:
    def test_tasks_arrive_in_waves_not_all_at_once(self, benchmark_map: Graph) -> None:
        """A single release at t=0 lets a fleet of three spread out and never meet
        again; waves keep re-injecting contention for the whole run."""
        task_set = generate(benchmark_map, seed=1, count=20, waves=4)
        release_times = sorted({t.created_at_ms for t in task_set})
        assert len(release_times) == 4
        assert release_times[0] == 0

    def test_wave_interval_is_respected(self, benchmark_map: Graph) -> None:
        task_set = generate(
            benchmark_map, seed=1, count=20, waves=3, wave_interval_ms=5000
        )
        assert sorted({t.created_at_ms for t in task_set}) == [0, 5000, 10_000]

    def test_released_by_is_monotonic(self, benchmark_map: Graph) -> None:
        task_set = generate(benchmark_map, seed=1, count=20, waves=4)
        counts = [len(task_set.released_by(t)) for t in (0, 20_000, 40_000, 60_000)]
        assert counts == sorted(counts)
        assert counts[-1] == len(task_set)

    def test_a_single_wave_is_allowed(self, benchmark_map: Graph) -> None:
        task_set = generate(benchmark_map, seed=1, count=6, waves=1)
        assert {t.created_at_ms for t in task_set} == {0}


class TestTaskProperties:
    def test_every_task_is_a_real_journey(self, benchmark_map: Graph) -> None:
        for task in generate(benchmark_map, seed=1, count=30):
            assert task.pickup != task.drop

    def test_every_task_is_routable(self, benchmark_map: Graph) -> None:
        planner = AStarPlanner(benchmark_map)
        for task in generate(benchmark_map, seed=1, count=30):
            assert planner.route(task.pickup, task.drop) is not None

    def test_priorities_fit_the_wire_format(self, benchmark_map: Graph) -> None:
        """ASM-17: outside 0-255 the arbitration total order is undefined."""
        task_set = generate(benchmark_map, seed=1, count=40)
        task_generator.validate_priorities(task_set)
        assert all(0 <= t.priority <= MAX_PRIORITY for t in task_set)

    def test_priorities_repeat_so_the_tie_break_is_exercised(
        self, benchmark_map: Graph
    ) -> None:
        """Arbitration falling through to the robot_id tie-break is the branch most
        likely to harbour a defect. Uniform random priorities would almost never
        reach it."""
        task_set = generate(benchmark_map, seed=1, count=40)
        distinct = {t.priority for t in task_set}
        assert len(distinct) <= len(task_generator.PRIORITY_MIX)
        assert len(task_set) > len(distinct) * 3, "priorities are too varied to tie"

    def test_task_ids_are_unique_and_sequential(self, benchmark_map: Graph) -> None:
        task_set = generate(benchmark_map, seed=1, count=15)
        assert [t.task_id for t in task_set] == list(range(1, 16))

    def test_traffic_crosses_the_corridor_in_both_directions(
        self, benchmark_map: Graph
    ) -> None:
        """Otherwise a single-lane corridor never sees a head-on approach, which
        would quietly exclude the case FR-5.10 and TC-2 exist for."""
        task_set = generate(benchmark_map, seed=1, count=40)
        left = {1, 2, 3, 0}
        westbound = any(t.pickup not in left and t.drop in left for t in task_set)
        eastbound = any(t.pickup in left and t.drop not in left for t in task_set)
        assert westbound and eastbound

    def test_zones_are_stamped_when_zoning_is_on(self) -> None:
        """FR-9.2: ANNOUNCE carries the zone identifier of the task."""
        from core.zones import ZoneMap

        graph = Graph.load(MAPS_DIR / "warehouse_zoned_30.json")
        zones = ZoneMap.from_graph(graph, enabled=True)
        task_set = task_generator.generate_for_map(
            graph, raw_map("warehouse_zoned_30"), seed=1, count=20,
            zone_of=zones.zone_of,
        )
        assert all(t.zone_id >= 0 for t in task_set)


class TestValidation:
    def test_zero_tasks_is_rejected(self, benchmark_map: Graph) -> None:
        with pytest.raises(ValueError, match="at least one task"):
            generate(benchmark_map, count=0)

    def test_zero_waves_is_rejected(self, benchmark_map: Graph) -> None:
        with pytest.raises(ValueError, match="waves must be positive"):
            generate(benchmark_map, count=5, waves=0)

    def test_describe_reports_what_matters(self, benchmark_map: Graph) -> None:
        description = generate(benchmark_map, seed=4, count=12).describe()
        assert "12 tasks" in description
        assert "through the choke" in description
        assert "seed 4" in description
