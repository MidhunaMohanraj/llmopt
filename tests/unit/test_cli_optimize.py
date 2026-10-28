from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import llmopt.optimize_cli as optimize_cli_module
from llmopt.cli import app
from llmopt.schemas import Config, Metrics, Status, TrialResult, Workload
from llmopt.tune.optimize import OptimizeReport

runner = CliRunner()
SEQS = "max_num_seqs"
TOKENS = "max_num_batched_tokens"


def use_fake(monkeypatch: pytest.MonkeyPatch, fail: bool = False) -> None:
    """Made-up response, not a measurement: 2048 seqs adds 50 tok/s, 16384 tokens
    adds 20 tok/s and 30 ms of TTFT."""

    def fake_make_engine(name: str) -> object:
        return object()

    def fake_run_bench(
        engine: object, config: Config, workload: Workload, **kwargs: Any
    ) -> TrialResult:
        if fail:
            return TrialResult(config=config, status=Status.FAILED, reason="boom")
        throughput = 100.0
        ttft = 100.0
        if config.settings.get(SEQS) == 2048:
            throughput += 50.0
        if config.settings.get(TOKENS) == 16384:
            throughput += 20.0
            ttft += 30.0
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

    monkeypatch.setattr(optimize_cli_module, "make_engine", fake_make_engine)
    monkeypatch.setattr(optimize_cli_module, "run_bench", fake_run_bench)


def optimize_args(
    tmp_path: Path, *extra: str, engine: str = "vllm", tune: str = f"{SEQS},{TOKENS}"
) -> list[str]:
    return [
        "optimize", "--engine", engine, "--model", "/m",
        "--tune", tune, "--output", str(tmp_path / "o.json"), *extra,
    ]


def load(tmp_path: Path) -> OptimizeReport:
    return OptimizeReport.model_validate_json((tmp_path / "o.json").read_text())


def test_optimize_saves_the_best_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(app, optimize_args(tmp_path))
    assert result.exit_code == 0
    assert "best:" in result.output
    report = load(tmp_path)
    assert report.search.best.settings == {SEQS: 2048, TOKENS: 16384}
    assert report.grid is None


def test_optimize_with_grid_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(app, optimize_args(tmp_path, "--check-grid"))
    assert result.exit_code == 0
    assert "grid check" in result.output
    report = load(tmp_path)
    assert report.grid is not None
    assert report.grid.within_limit


def test_optimize_respects_the_ttft_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(app, optimize_args(tmp_path, "--max-ttft-p95-ms", "120"))
    assert result.exit_code == 0
    assert load(tmp_path).search.best.settings == {SEQS: 2048}


def test_optimize_refuses_a_big_grid_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(app, optimize_args(tmp_path, "--check-grid", "--max-grid", "4"))
    assert result.exit_code == 1
    assert not (tmp_path / "o.json").exists()


def test_optimize_rejects_unknown_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(app, optimize_args(tmp_path, "--objective", "speed"))
    assert result.exit_code == 1


def test_optimize_rejects_unknown_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(app, optimize_args(tmp_path, tune="nope"))
    assert result.exit_code == 1


def test_optimize_rejects_unknown_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(app, optimize_args(tmp_path, engine="nope"))
    assert result.exit_code == 1


def test_optimize_rejects_bad_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch)
    result = runner.invoke(app, optimize_args(tmp_path, "--settings", "novalue"))
    assert result.exit_code == 1


def test_optimize_fails_when_the_starting_config_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake(monkeypatch, fail=True)
    result = runner.invoke(app, optimize_args(tmp_path))
    assert result.exit_code == 1
    assert not (tmp_path / "o.json").exists()