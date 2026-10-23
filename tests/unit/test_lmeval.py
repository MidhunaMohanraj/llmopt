from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.validate.lmeval import (
    TASK_NAME,
    accuracy_from_results,
    accuracy_result,
    build_command,
    build_task_yaml,
)

# Same shape as a real lm-eval 0.4.13 results file.
SAMPLE = {
    "results": {
        TASK_NAME: {
            "name": TASK_NAME,
            "sample_len": 20,
            "acc,none": 0.6,
            "acc_stderr,none": 0.11,
            "acc_norm,none": 0.4,
            "acc_norm_stderr,none": 0.11,
        }
    }
}


def test_yaml_points_at_local_files() -> None:
    text = build_task_yaml(Path("/data"))
    assert f"task: {TASK_NAME}" in text
    assert "dataset_path: parquet" in text
    assert "/data/ai2_arc/ARC-Easy/test-00000-of-00001.parquet" in text


def test_accuracy_from_results_reads_acc_and_acc_norm() -> None:
    assert accuracy_from_results(SAMPLE) == pytest.approx(0.6)
    assert accuracy_from_results(SAMPLE, metric="acc_norm") == pytest.approx(0.4)


def test_missing_task_raises() -> None:
    with pytest.raises(KeyError):
        accuracy_from_results({"results": {}})


def test_missing_metric_raises() -> None:
    with pytest.raises(KeyError):
        accuracy_from_results(SAMPLE, metric="f1")


def test_accuracy_drop_is_positive_when_worse() -> None:
    result = accuracy_result(0.45, 0.5)
    assert result.drop_pct == pytest.approx(10.0)
    assert result.task == TASK_NAME


def test_accuracy_rejects_zero_baseline() -> None:
    with pytest.raises(ValueError):
        accuracy_result(0.5, 0.0)


def test_command_includes_limit_only_when_given() -> None:
    base = build_command(Path("/m"), Path("/t"), Path("/o"))
    assert "--limit" not in base
    limited = build_command(Path("/m"), Path("/t"), Path("/o"), limit=20)
    assert limited[-2:] == ["--limit", "20"]


def test_command_rejects_bad_limit() -> None:
    with pytest.raises(ValueError):
        build_command(Path("/m"), Path("/t"), Path("/o"), limit=0)