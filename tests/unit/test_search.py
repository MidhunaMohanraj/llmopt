from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.profile.factory import make_config
from llmopt.schemas import (
    AccuracyResult,
    Config,
    Metrics,
    Objective,
    SettingValue,
    Status,
    TrialResult,
)
from llmopt.settings.table import BY_NAME
from llmopt.tune.search import (
    Evaluated,
    candidate_values,
    default_candidates,
    feasibility_problem,
    pareto_front,
    run_search,
)

CONFIG = make_config("vllm", Path("/m"), {})
SEQS = "max_num_seqs"
TOKENS = "max_num_batched_tokens"
CANDIDATES: dict[str, list[SettingValue]] = {SEQS: [512, 2048], TOKENS: [4096, 16384]}


def make_trial(
    config: Config, throughput: float, ttft: float = 100.0, status: Status = Status.OK
) -> TrialResult:
    if status is not Status.OK:
        return TrialResult(config=config, status=status, reason="boom")
    metrics = Metrics(
        ttft_ms_p50=ttft,
        ttft_ms_p95=ttft,
        tpot_ms=10.0,
        throughput_tps=throughput,
        latency_ms_p50=500.0,
        latency_ms_p95=600.0,
        peak_memory_gb=2.0,
    )
    return TrialResult(config=config, status=Status.OK, metrics=metrics)


def evaluate(config: Config) -> TrialResult:
    """Made-up response, not a measurement: 2048 seqs adds 50 tok/s, 16384 tokens
    adds 20 tok/s and 30 ms of TTFT."""
    settings = config.settings
    throughput = 100.0
    ttft = 100.0
    if settings.get(SEQS) == 2048:
        throughput += 50.0
    if settings.get(TOKENS) == 16384:
        throughput += 20.0
        ttft += 30.0
    return make_trial(config, throughput, ttft)


def clashing(config: Config) -> TrialResult:
    settings = config.settings
    if settings.get(SEQS) == 2048 and settings.get(TOKENS) == 16384:
        return make_trial(config, 90.0)
    return evaluate(config)


def rate(item: Evaluated) -> float:
    assert item.trial.metrics is not None
    return item.trial.metrics.throughput_tps


def test_finds_and_combines_winners() -> None:
    result = run_search(evaluate, CONFIG, Objective(), CANDIDATES)
    assert result.best.settings == {SEQS: 2048, TOKENS: 16384}
    assert rate(result.best) == pytest.approx(170.0)
    assert rate(result.baseline) == pytest.approx(100.0)
    # baseline, four single values and the one combination
    assert len(result.evaluated) == 6
    assert not result.stopped_early


def test_ttft_limit_excludes_candidate() -> None:
    result = run_search(evaluate, CONFIG, Objective(max_ttft_p95_ms=120.0), CANDIDATES)
    assert result.best.settings == {SEQS: 2048}
    assert rate(result.best) == pytest.approx(150.0)


def test_small_gain_is_ignored() -> None:
    def small(config: Config) -> TrialResult:
        return make_trial(config, 100.5 if config.settings.get(SEQS) == 2048 else 100.0)

    result = run_search(small, CONFIG, Objective(), {SEQS: [2048]})
    assert result.best.settings == {}
    assert len(result.evaluated) == 2


def test_combination_that_clashes_is_pruned() -> None:
    result = run_search(clashing, CONFIG, Objective(), CANDIDATES)
    assert result.best.settings == {SEQS: 2048}
    assert rate(result.best) == pytest.approx(150.0)


def test_max_trials_stops_early() -> None:
    result = run_search(evaluate, CONFIG, Objective(), CANDIDATES, max_trials=3)
    assert result.stopped_early
    assert len(result.evaluated) == 3


def test_failed_baseline_raises() -> None:
    def broken(config: Config) -> TrialResult:
        return make_trial(config, 0.0, status=Status.FAILED)

    with pytest.raises(ValueError, match="starting config"):
        run_search(broken, CONFIG, Objective(), CANDIDATES)


def test_unmeasured_accuracy_makes_everything_unusable() -> None:
    result = run_search(
        evaluate, CONFIG, Objective(max_accuracy_drop_pct=1.0), CANDIDATES
    )
    assert not result.best.feasible
    assert result.best.problem == "accuracy not measured"
    assert result.best.settings == {}


def test_accuracy_limit_checks_the_drop() -> None:
    base = make_trial(CONFIG, 100.0)

    def with_drop(drop: float) -> TrialResult:
        item = AccuracyResult(task="t", score=1.0, baseline_score=1.0, drop_pct=drop)
        return base.model_copy(update={"accuracy": [item]})

    objective = Objective(max_accuracy_drop_pct=1.0)
    assert feasibility_problem(with_drop(0.5), objective) is None
    assert "accuracy drop" in str(feasibility_problem(with_drop(2.0), objective))


def test_repeated_value_is_measured_once() -> None:
    result = run_search(evaluate, CONFIG, Objective(), {SEQS: [2048, 2048]})
    assert len(result.evaluated) == 2


def test_unknown_objective_raises() -> None:
    with pytest.raises(ValueError, match="optimize"):
        run_search(evaluate, CONFIG, Objective(optimize="speed"), CANDIDATES)


def test_lower_is_better_metric() -> None:
    def faster_start(config: Config) -> TrialResult:
        return make_trial(config, 100.0, 80.0 if config.settings.get(TOKENS) == 4096 else 100.0)

    result = run_search(faster_start, CONFIG, Objective(optimize="ttft"), {TOKENS: [4096, 16384]})
    assert result.best.settings == {TOKENS: 4096}


def test_pareto_front_drops_dominated() -> None:
    def item(throughput: float, ttft: float) -> Evaluated:
        return Evaluated(settings={}, trial=make_trial(CONFIG, throughput, ttft))

    points = [item(100, 100), item(150, 130), item(120, 140), item(90, 90)]
    assert [rate(p) for p in pareto_front(points)] == [90.0, 100.0, 150.0]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gpu_memory_utilization", [0.45, 0.95]),
        ("max_num_seqs", [512, 2048]),
        ("enable_chunked_prefill", [False]),
        ("enable_prefix_caching", [True, False]),
        ("async_scheduling", [True]),
        ("cudagraph_mode", ["NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY"]),
    ],
)
def test_candidate_values(name: str, expected: list[SettingValue]) -> None:
    assert candidate_values(BY_NAME[name]) == expected


def test_default_candidates_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown setting"):
        default_candidates(["nope"])


def test_default_candidates_covers_known_settings() -> None:
    result = default_candidates()
    assert result["max_num_seqs"] == [512, 2048]
    assert "enable_prefix_caching" in result