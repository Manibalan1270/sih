from core import scenarios
from benchmark.runner import compare_configurations, run_seed
from simulator.scenario import AuctionAllocator, RoundRobinAllocator


def test_run_seed_and_compare_use_seeded_same_task_set() -> None:
    scenario = scenarios.get("bench3")

    a = run_seed(scenario, seed=0, config_name="A", allocator=RoundRobinAllocator())
    b = run_seed(scenario, seed=0, config_name="B", allocator=AuctionAllocator())

    assert a.seed == b.seed == 0
    assert a.tasks_total == b.tasks_total

    result = compare_configurations([a], [b])
    assert "makespan_mean_a" in result
    assert "makespan_mean_b" in result
    assert result["makespan_mean_a"] >= 0
    assert result["makespan_mean_b"] >= 0


class TestSectionSixThreeMetrics:
    """The §6.3 report is mandatory, and its output is the evidence for §8.

    Everything here was specified from the start and simply not reported: the
    runner measured makespan and collisions and dropped the rest on the floor,
    so a reader could not tell a real coordination win from a lucky seed set.
    """

    def test_every_metric_carries_a_mean_and_a_standard_deviation(self) -> None:
        """§6.3: "a single run is not admissible as evidence"."""
        scenario = scenarios.get("bench3")
        a = [run_seed(scenario, seed=s, config_name="A", allocator=RoundRobinAllocator())
             for s in (0, 1)]
        b = [run_seed(scenario, seed=s, config_name="B", allocator=AuctionAllocator())
             for s in (0, 1)]
        report = compare_configurations(a, b)

        for metric in (
            "makespan", "completion", "collisions", "coordination_failures",
            "stopped_time", "precedence_hold", "yields_lost", "yields_won",
            "route_diversity", "auction_frames_per_task",
        ):
            for tag in ("a", "b"):
                assert f"{metric}_mean_{tag}" in report, f"{metric} has no mean for {tag}"
                assert f"{metric}_std_{tag}" in report, f"{metric} has no SD for {tag}"
        assert report["seeds"] == 2

    def test_mean_per_task_completion_time_is_measured(self) -> None:
        """A §6.3 *primary* metric, alongside makespan and collisions.

        Makespan says when the fleet stopped; this says how long an order waited.
        Task.completion_ms() has always computed it -- nothing read it.
        """
        scenario = scenarios.get("bench3")
        result = run_seed(scenario, seed=0, config_name="B", allocator=AuctionAllocator())
        assert result.completion_mean_ms > 0
        # An order cannot take longer than the run that contained it.
        assert result.completion_mean_ms <= result.makespan_ms

    def test_secondary_metrics_are_reported(self) -> None:
        scenario = scenarios.get("bench3")
        result = run_seed(scenario, seed=0, config_name="B", allocator=AuctionAllocator())
        assert 0.0 < result.route_diversity <= 1.0, "fraction of the map's edges used"
        assert result.auction_frames_per_task >= 0
        assert result.yields_won >= 0

    def test_the_ten_amr_rung_of_the_scalability_ladder_exists(self) -> None:
        """§6.3 names three, ten and one hundred AMRs. Ten was missing, so there
        was no evidence for how the result moves with fleet size."""
        ladder = sorted(
            s.robots for s in scenarios.SCENARIOS.values() if not s.physical_robots
        )
        for size in (3, 10, 100):
            assert size in ladder, f"§6.3 requires a {size}-AMR configuration"
