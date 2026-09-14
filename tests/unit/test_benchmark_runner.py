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
