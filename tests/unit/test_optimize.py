from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.profile.factory import make_config
from llmopt.schemas import Config, Metrics, Objective, SettingValue, Status, TrialResult
from llmopt.tune.optimize import (
    GridCheck,
    OptimizeReport,
    compare_to_grid,
    format_report,
    grid_points,
    run_grid,
    save_optimize,
)
from llmopt.tune.search import Evaluated, SearchResult

CONFIG = make_config("vllm", Path("/m"), {})
SEQS = "max_num_seqs"
TOKENS = "max_num_batched_tokens"
CANDIDATES: dict[str, list[SettingValue]] = {SEQS: [512, 2048], TOKENS: [4096, 16384]}


def make_trial(config: Config, throughput: float, ttft: float = 100.0) -> TrialResult:
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


def item(
    throughput: float,
    ttft: float = 100.0,
    settings: dict[str, SettingValue] | None = None,
    problem: str | None = None,
) -> Evaluated:
    return Evaluated(
        settings=settings or {}, trial=make_trial(CONFIG, throughput, ttft), problem=problem
    )


def search_with(best: Evaluated, baseline: Evaluated | None = None) -> SearchResult:
    base = baseline or item(100.0)
    return SearchResult(
        baseline=base, best=best, evaluated=[base, best], front=[best], stopped_early=False
    )


def test_grid_points_cover_every_combination() -> None:
    points = grid_points({}, CANDIDATES, 64)
    assert len(points) == 9
    assert {} in points
    assert {SEQS: 2048, TOKENS: 16384} in points


def test_grid_points_start_from_the_starting_value() -> None:
    base: dict[str, SettingValue] = {"max_model_len": 2048, SEQS: 1024}
    points = grid_points(base, {SEQS: [512, 1024, 2048]}, 64)
    assert [point[SEQS] for point in points] == [1024, 512, 2048]
    assert all(point["max_model_len"] == 2048 for point in points)


def test_grid_points_refuse_a_big_grid() -> None:
    with pytest.raises(ValueError, match="grid"):
        grid_points({}, CANDIDATES, 8)


def test_grid_points_without_candidates_is_the_start() -> None:
    assert grid_points({"a": 1}, {}, 4) == [{"a": 1}]


def evaluate_fast(config: Config) -> TrialResult:
    return make_trial(config, 150.0 if config.settings.get(SEQS) == 2048 else 100.0)


def test_run_grid_measures_every_point() -> None:
    points = grid_points({}, CANDIDATES, 64)
    results = run_grid(evaluate_fast, CONFIG, Objective(), points)
    assert len(results) == 9
    assert all(result.feasible for result in results)


def test_run_grid_marks_points_over_the_ttft_limit() -> None:
    points = grid_points({}, CANDIDATES, 64)
    results = run_grid(evaluate_fast, CONFIG, Objective(max_ttft_p95_ms=50.0), points)
    assert not any(result.feasible for result in results)


def test_compare_within_the_limit() -> None:
    check = compare_to_grid(search_with(item(108.0)), [item(110.0), item(100.0)], Objective())
    assert check.within_limit
    assert check.gap_pct == pytest.approx(1.818, abs=0.01)
    assert check.grid_best is not None
    assert check.grid_best.trial.metrics is not None
    assert check.grid_best.trial.metrics.throughput_tps == 110.0


def test_compare_outside_the_limit() -> None:
    check = compare_to_grid(search_with(item(100.0)), [item(110.0)], Objective())
    assert not check.within_limit
    assert check.gap_pct == pytest.approx(9.09, abs=0.01)


def test_compare_custom_limit() -> None:
    check = compare_to_grid(search_with(item(100.0)), [item(110.0)], Objective(), limit_pct=10.0)
    assert check.within_limit


def test_compare_tuner_ahead_of_the_grid() -> None:
    check = compare_to_grid(search_with(item(120.0)), [item(110.0)], Objective())
    assert check.within_limit
    assert check.gap_pct is not None
    assert check.gap_pct < 0


@pytest.mark.parametrize(("tuner_ttft", "expected"), [(82.0, True), (100.0, False)])
def test_compare_when_lower_is_better(tuner_ttft: float, expected: bool) -> None:
    grid = [item(100.0, ttft=80.0), item(100.0, ttft=100.0)]
    check = compare_to_grid(
        search_with(item(100.0, ttft=tuner_ttft)), grid, Objective(optimize="ttft")
    )
    assert check.within_limit is expected


def test_compare_with_no_usable_grid_result() -> None:
    check = compare_to_grid(search_with(item(100.0)), [item(110.0, problem="x")], Objective())
    assert check.grid_best is None
    assert check.gap_pct is None
    assert check.within_limit


def test_compare_when_the_tuner_has_no_usable_result() -> None:
    check = compare_to_grid(
        search_with(item(100.0, problem="over limit")), [item(110.0)], Objective()
    )
    assert check.grid_best is not None
    assert check.gap_pct is None
    assert not check.within_limit


def make_report(grid: GridCheck | None, best: Evaluated) -> OptimizeReport:
    return OptimizeReport(
        config=CONFIG,
        workload="smoke",
        objective=Objective(),
        search=search_with(best),
        grid=grid,
    )


def test_format_report_shows_best_and_grid() -> None:
    best = item(150.0, ttft=120.0, settings={SEQS: 2048})
    check = compare_to_grid(search_with(best), [item(150.0)], Objective())
    text = format_report(make_report(check, best))
    assert "objective: throughput" in text
    assert "(+50.0%)" in text
    assert '"max_num_seqs": 2048' in text
    assert "grid check" in text
    assert "OK" in text


def test_format_report_without_a_usable_best() -> None:
    text = format_report(make_report(None, item(100.0, problem="over limit")))
    assert "no usable result (over limit)" in text
    assert "grid check" not in text


def test_save_optimize_round_trips(tmp_path: Path) -> None:
    best = item(150.0, settings={SEQS: 2048})
    check = compare_to_grid(search_with(best), [item(150.0)], Objective())
    path = tmp_path / "deep" / "o.json"
    save_optimize(make_report(check, best), path)
    loaded = OptimizeReport.model_validate_json(path.read_text())
    assert loaded.search.best.settings == {SEQS: 2048}
    assert loaded.grid is not None
    assert loaded.grid.within_limit