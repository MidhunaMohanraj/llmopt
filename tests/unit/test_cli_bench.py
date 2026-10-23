from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import llmopt.cli as cli_module
from llmopt.cli import app
from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.profile.bench import run_bench
from llmopt.report.store import load_trials
from llmopt.schemas import Config, HardwareProfile, Precision, Status, TrialResult, Workload

runner = CliRunner()
real_run_bench = run_bench


class ScriptedEngine(EngineAdapter):
    """Test double with made-up timings. Never use its numbers as a benchmark."""

    name = "scripted"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def supports(
        self, hardware: HardwareProfile, model: str, precision: Precision
    ) -> SupportResult:
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        return None

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        if self.fail:
            raise RuntimeError("engine crashed")
        return [
            GenResult(
                request_id=request.request_id,
                token_times_ms=[10.0 * (k + 1) for k in range(4)],
            )
            for request in requests
        ]

    def teardown(self) -> None:
        return None


def use_engine(monkeypatch: pytest.MonkeyPatch, fail: bool = False) -> None:
    def fake_make_engine(name: str) -> EngineAdapter:
        return ScriptedEngine(fail=fail)

    def fake_run_bench(
        engine: EngineAdapter, config: Config, workload: Workload, **kwargs: Any
    ) -> TrialResult:
        return real_run_bench(
            engine, config, workload, read_memory_bytes=lambda: 2**30, **kwargs
        )

    monkeypatch.setattr(cli_module, "make_engine", fake_make_engine)
    monkeypatch.setattr(cli_module, "run_bench", fake_run_bench)


def bench_args(tmp_path: Path, engine: str = "pytorch", extra: list[str] | None = None) -> list[str]:
    args = ["bench", "--engine", engine, "--model", "/m", "--results", str(tmp_path / "res")]
    return args + (extra or [])


def test_bench_saves_an_ok_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_engine(monkeypatch)
    result = runner.invoke(app, bench_args(tmp_path))
    assert result.exit_code == 0
    assert "status: ok" in result.output
    assert "Saved to" in result.output
    trials = load_trials(tmp_path / "res")
    assert len(trials) == 1
    assert trials[0].status is Status.OK


def test_bench_failure_exits_1_but_still_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_engine(monkeypatch, fail=True)
    result = runner.invoke(app, bench_args(tmp_path))
    assert result.exit_code == 1
    trials = load_trials(tmp_path / "res")
    assert len(trials) == 1
    assert trials[0].status is Status.FAILED


def test_bench_rejects_unknown_engine(tmp_path: Path) -> None:
    result = runner.invoke(app, bench_args(tmp_path, engine="nope"))
    assert result.exit_code == 1


def test_bench_rejects_bad_settings(tmp_path: Path) -> None:
    result = runner.invoke(app, bench_args(tmp_path, extra=["--settings", "novalue"]))
    assert result.exit_code == 1