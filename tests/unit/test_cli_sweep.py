from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import llmopt.profile.sweep as sweep_module
import llmopt.sweep_cli as sweep_cli_module
from llmopt.cli import app
from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.profile.bench import run_bench
from llmopt.profile.sweep import SweepResult
from llmopt.schemas import Config, HardwareProfile, Precision, TrialResult, Workload

runner = CliRunner()
real_run_bench = run_bench
TIMES = [250.0, 500.0, 750.0, 1000.0]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class WaveEngine(EngineAdapter):
    """Pretend engine with made-up timings. Never use its numbers as a benchmark."""

    name = "wave"

    def __init__(self, clock: FakeClock, concurrent: bool = True) -> None:
        self.clock = clock
        self.supports_concurrency = concurrent

    def supports(
        self, hardware: HardwareProfile, model: str, precision: Precision
    ) -> SupportResult:
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        return None

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        self.clock.now += 1.0 if self.supports_concurrency else float(len(requests))
        return [
            GenResult(request_id=r.request_id, token_times_ms=list(TIMES)) for r in requests
        ]

    def teardown(self) -> None:
        return None


def use_engine(monkeypatch: pytest.MonkeyPatch, concurrent: bool = True) -> None:
    clock = FakeClock()

    def fake_make_engine(name: str) -> EngineAdapter:
        return WaveEngine(clock, concurrent)

    def fake_run_bench(
        engine: EngineAdapter, config: Config, workload: Workload, **kwargs: Any
    ) -> TrialResult:
        kwargs.update(read_memory_bytes=lambda: 2**30, clock=clock)
        return real_run_bench(engine, config, workload, **kwargs)

    monkeypatch.setattr(sweep_cli_module, "make_engine", fake_make_engine)
    monkeypatch.setattr(sweep_module, "run_bench", fake_run_bench)


def sweep_args(tmp_path: Path, levels: str = "1,2,4", engine: str = "vllm") -> list[str]:
    return [
        "sweep", "--engine", engine, "--model", "/m",
        "--levels", levels, "--output", str(tmp_path / "s.json"),
    ]


def test_sweep_saves_one_row_per_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_engine(monkeypatch)
    result = runner.invoke(app, sweep_args(tmp_path))
    assert result.exit_code == 0
    assert "tok/s" in result.output
    saved = SweepResult.model_validate_json((tmp_path / "s.json").read_text())
    assert [row.concurrency for row in saved.rows] == [1, 2, 4]


def test_sweep_rejects_bad_levels(tmp_path: Path) -> None:
    result = runner.invoke(app, sweep_args(tmp_path, levels="4,2"))
    assert result.exit_code == 1


def test_sweep_rejects_unknown_engine(tmp_path: Path) -> None:
    result = runner.invoke(app, sweep_args(tmp_path, engine="nope"))
    assert result.exit_code == 1


def test_sweep_rejects_sequential_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_engine(monkeypatch, concurrent=False)
    result = runner.invoke(app, sweep_args(tmp_path, levels="1,2"))
    assert result.exit_code == 1
    assert not (tmp_path / "s.json").exists()