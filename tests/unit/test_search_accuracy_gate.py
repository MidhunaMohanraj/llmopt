"""A faster setup that loses too much accuracy must not be chosen (Gate G2 mechanics).

These tests use a stand-in runner, so they prove the search logic only. They say nothing
about real accuracy on any model or GPU.
"""

from llmopt.schemas import (
    AccuracyResult,
    Config,
    Engine,
    Metrics,
    Objective,
    Precision,
    SettingValue,
    Status,
    TrialResult,
)
from llmopt.tune.search import run_search

SEQS = "max_num_seqs"
CONFIG = Config(model="/models/example", engine=Engine.VLLM, precision=Precision.BF16)
CANDIDATES: dict[str, list[SettingValue]] = {SEQS: [512, 2048]}
BAD_DROP_PCT = 3.0


def runner(config: Config) -> TrialResult:
    """2048 sequences is twice as fast but loses 3% accuracy; the rest match the baseline."""
    bad = config.settings.get(SEQS) == 2048
    metrics = Metrics(
        ttft_ms_p50=40.0,
        ttft_ms_p95=50.0,
        tpot_ms=30.0,
        throughput_tps=200.0 if bad else 100.0,
        latency_ms_p50=500.0,
        latency_ms_p95=550.0,
        peak_memory_gb=8.0,
    )
    accuracy = AccuracyResult(
        task="t", score=1.0, baseline_score=1.0, drop_pct=BAD_DROP_PCT if bad else 0.0
    )
    return TrialResult(config=config, status=Status.OK, metrics=metrics, accuracy=[accuracy])


def test_without_an_accuracy_limit_the_faster_setup_wins() -> None:
    result = run_search(runner, CONFIG, Objective(), CANDIDATES)
    assert result.best.settings == {SEQS: 2048}


def test_the_accuracy_limit_rejects_the_faster_setup() -> None:
    result = run_search(runner, CONFIG, Objective(max_accuracy_drop_pct=1.0), CANDIDATES)
    assert result.best.settings == {}
    rejected = [item for item in result.evaluated if item.settings == {SEQS: 2048}]
    assert len(rejected) == 1
    assert not rejected[0].feasible
    assert "accuracy drop 3.00%" in str(rejected[0].problem)


def test_a_looser_limit_accepts_it_again() -> None:
    result = run_search(runner, CONFIG, Objective(max_accuracy_drop_pct=5.0), CANDIDATES)
    assert result.best.settings == {SEQS: 2048}