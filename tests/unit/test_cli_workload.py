from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import llmopt.cli as cli_module
from llmopt.cli import app
from llmopt.schemas import Config, Status, TrialResult, Workload

runner = CliRunner()


def capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Workload]:
    seen: dict[str, Workload] = {}

    def fake_run_bench(
        engine: Any, config: Config, workload: Workload, **kwargs: Any
    ) -> TrialResult:
        seen["workload"] = workload
        return TrialResult(config=config, status=Status.OK)

    monkeypatch.setattr(cli_module, "make_engine", lambda name: object())
    monkeypatch.setattr(cli_module, "run_bench", fake_run_bench)
    return seen


def run_bench_cmd(tmp_path: Path, *extra: str, engine: str = "pytorch") -> Any:
    args = ["bench", "--engine", engine, "--model", "/m"]
    args += ["--results", str(tmp_path / "res"), *extra]
    return runner.invoke(app, args)


def test_default_workload_is_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = capture(monkeypatch)
    result = run_bench_cmd(tmp_path)
    assert result.exit_code == 0
    workload = seen["workload"]
    assert workload.name == "smoke"
    assert (workload.prompt_tokens, workload.output_tokens, workload.num_requests) == (32, 16, 8)


def test_named_workload_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = capture(monkeypatch)
    result = run_bench_cmd(tmp_path, "--workload", "chat")
    assert result.exit_code == 0
    workload = seen["workload"]
    assert workload.name == "chat"
    assert (workload.prompt_tokens, workload.output_tokens, workload.num_requests) == (
        128,
        256,
        16,
    )


def test_explicit_option_overrides_the_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = capture(monkeypatch)
    result = run_bench_cmd(tmp_path, "--workload", "chat", "--prompt-tokens", "64")
    assert result.exit_code == 0
    assert seen["workload"].prompt_tokens == 64
    assert seen["workload"].output_tokens == 256


def test_unknown_workload_exits_1_without_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = capture(monkeypatch)
    result = run_bench_cmd(tmp_path, "--workload", "nope")
    assert result.exit_code == 1
    assert seen == {}


def test_vllm_long_is_rejected_when_context_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = capture(monkeypatch)
    result = run_bench_cmd(tmp_path, "--workload", "long", engine="vllm")
    assert result.exit_code == 1
    assert seen == {}


def test_vllm_long_runs_with_a_larger_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = capture(monkeypatch)
    result = run_bench_cmd(
        tmp_path, "--workload", "long", "--settings", "max_model_len=8192", engine="vllm"
    )
    assert result.exit_code == 0
    assert seen["workload"].name == "long"