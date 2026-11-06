from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import llmopt.cli as cli_module
from llmopt.cli import app
from llmopt.validate.baseline import (
    Baseline,
    check_baseline,
    read_baseline,
    write_baseline,
)

runner = CliRunner()


def use_scorers(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def engine_fake(
        engine: str,
        model: Path,
        datasets: Path,
        limit: int | None,
        precision: str,
        kv_cache_dtype: str,
        settings: str,
    ) -> float:
        calls.append((engine, precision, kv_cache_dtype, settings))
        return value

    def plain_fake(
        task: str, model: Path, datasets: Path, limit: int | None, device: str
    ) -> float:
        calls.append(("plain", task))
        return value

    monkeypatch.setattr(cli_module, "_score_engine", engine_fake)
    monkeypatch.setattr(cli_module, "_score", plain_fake)
    return calls


def save_baseline(path: Path, engine: str) -> None:
    write_baseline(
        Baseline(
            task="wikitext2_perplexity", score=30.0, model="/m", engine=engine
        ),
        path,
    )


def test_old_baseline_files_default_to_transformers(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text('{"task": "t", "score": 2.0, "model": "/m", "limit": null}')
    assert read_baseline(path).engine == "transformers"


def test_baseline_from_another_engine_is_not_comparable() -> None:
    saved = Baseline(task="t", score=1.0, model="/m", engine="vllm")
    check_baseline(saved, "t", None, "vllm")
    with pytest.raises(ValueError, match="not comparable"):
        check_baseline(saved, "t", None)


def test_engine_baseline_records_the_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = use_scorers(monkeypatch, 30.0)
    path = tmp_path / "base.json"
    result = runner.invoke(
        app,
        [
            "validate",
            "--model",
            str(tmp_path),
            "--task",
            "ppl",
            "--engine",
            "vllm",
            "--write-baseline",
            str(path),
        ],
    )
    assert result.exit_code == 0
    assert read_baseline(path).engine == "vllm"
    assert calls == [("vllm", "bf16", "auto", "")]


def test_engine_run_passes_precision_kv_and_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = use_scorers(monkeypatch, 30.3)
    path = tmp_path / "base.json"
    save_baseline(path, "vllm")
    result = runner.invoke(
        app,
        [
            "validate",
            "--model",
            str(tmp_path),
            "--task",
            "ppl",
            "--engine",
            "vllm",
            "--baseline",
            str(path),
            "--precision",
            "fp8",
            "--kv-cache-dtype",
            "fp8",
            "--settings",
            "max_model_len=1024",
        ],
    )
    assert result.exit_code == 0
    assert calls == [("vllm", "fp8", "fp8", "max_model_len=1024")]


def test_transformers_baseline_is_refused_for_an_engine_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = use_scorers(monkeypatch, 30.0)
    path = tmp_path / "base.json"
    save_baseline(path, "transformers")
    result = runner.invoke(
        app,
        [
            "validate",
            "--model",
            str(tmp_path),
            "--task",
            "ppl",
            "--engine",
            "vllm",
            "--baseline",
            str(path),
        ],
    )
    assert result.exit_code == 1
    assert calls == []


def test_engine_supports_only_perplexity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = use_scorers(monkeypatch, 0.5)
    result = runner.invoke(
        app,
        [
            "validate",
            "--model",
            str(tmp_path),
            "--task",
            "arc",
            "--engine",
            "vllm",
            "--write-baseline",
            str(tmp_path / "base.json"),
        ],
    )
    assert result.exit_code == 1
    assert calls == []


def test_unknown_engine_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = use_scorers(monkeypatch, 30.0)
    result = runner.invoke(
        app,
        [
            "validate",
            "--model",
            str(tmp_path),
            "--task",
            "ppl",
            "--engine",
            "pytorch",
            "--write-baseline",
            str(tmp_path / "base.json"),
        ],
    )
    assert result.exit_code == 1
    assert calls == []


def test_engine_options_need_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = use_scorers(monkeypatch, 30.0)
    result = runner.invoke(
        app,
        [
            "validate",
            "--model",
            str(tmp_path),
            "--task",
            "ppl",
            "--precision",
            "fp8",
            "--write-baseline",
            str(tmp_path / "base.json"),
        ],
    )
    assert result.exit_code == 1
    assert calls == []


def test_device_does_not_apply_with_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = use_scorers(monkeypatch, 30.0)
    result = runner.invoke(
        app,
        [
            "validate",
            "--model",
            str(tmp_path),
            "--task",
            "ppl",
            "--engine",
            "vllm",
            "--device",
            "cuda",
            "--write-baseline",
            str(tmp_path / "base.json"),
        ],
    )
    assert result.exit_code == 1
    assert calls == []