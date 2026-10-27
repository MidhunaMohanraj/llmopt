from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import llmopt.cli as cli_module
from llmopt.cli import app
from llmopt.profile.factory import make_config, make_engine
from llmopt.schemas import Config, Engine, Status, TrialResult, Workload

runner = CliRunner()


def test_vllm_rocm_config_uses_the_vllm_engine() -> None:
    config = make_config("vllm-rocm", Path("/m"), {})
    assert config.engine is Engine.VLLM


def test_pytorch_rocm_config_uses_the_pytorch_engine() -> None:
    config = make_config("pytorch-rocm", Path("/m"), {})
    assert config.engine is Engine.PYTORCH


def test_make_engine_returns_the_rocm_adapters() -> None:
    assert make_engine("vllm-rocm").name == "vllm-rocm"
    assert make_engine("pytorch-rocm").name == "pytorch-rocm"


def test_unknown_engine_error_lists_the_rocm_names() -> None:
    with pytest.raises(ValueError, match="vllm-rocm"):
        make_config("nope", Path("/m"), {})


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


def run_bench_cmd(tmp_path: Path, *extra: str) -> Any:
    args = ["bench", "--engine", "vllm-rocm", "--model", "/m"]
    args += ["--results", str(tmp_path / "res"), *extra]
    return runner.invoke(app, args)


def test_vllm_rocm_long_is_rejected_when_context_is_too_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = capture(monkeypatch)
    result = run_bench_cmd(tmp_path, "--workload", "long")
    assert result.exit_code == 1
    assert seen == {}


def test_vllm_rocm_long_runs_with_a_larger_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = capture(monkeypatch)
    result = run_bench_cmd(tmp_path, "--workload", "long", "--settings", "max_model_len=8192")
    assert result.exit_code == 0
    assert seen["workload"].name == "long"