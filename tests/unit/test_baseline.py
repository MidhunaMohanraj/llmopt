from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.profile.factory import make_config
from llmopt.schemas import AccuracyResult, Status, TrialResult
from llmopt.validate.baseline import (
    Baseline,
    attach_accuracy,
    check_baseline,
    read_baseline,
    write_baseline,
)


def make_trial() -> TrialResult:
    return TrialResult(config=make_config("pytorch", Path("/m"), {}), status=Status.OK)


def make_result(task: str, score: float = 1.0) -> AccuracyResult:
    return AccuracyResult(task=task, score=score, baseline_score=1.0, drop_pct=0.0)


def test_baseline_round_trips(tmp_path: Path) -> None:
    baseline = Baseline(task="t", score=12.5, model="/m", limit=None)
    path = tmp_path / "b.json"
    write_baseline(baseline, path)
    assert read_baseline(path) == baseline


def test_write_baseline_creates_parent_folder(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "b.json"
    write_baseline(Baseline(task="t", score=1.0, model="/m"), path)
    assert path.is_file()


def test_read_missing_baseline_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_baseline(tmp_path / "nope.json")


def test_read_invalid_baseline_raises(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text('{"task": "t"}')
    with pytest.raises(ValueError):
        read_baseline(path)


def test_baseline_score_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Baseline(task="t", score=0.0, model="/m")


def test_check_baseline_accepts_matching_run() -> None:
    check_baseline(Baseline(task="t", score=1.0, model="/m", limit=5), "t", 5)


def test_check_baseline_rejects_other_task() -> None:
    with pytest.raises(ValueError, match="task"):
        check_baseline(Baseline(task="a", score=1.0, model="/m"), "b", None)


def test_check_baseline_rejects_other_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        check_baseline(Baseline(task="t", score=1.0, model="/m", limit=5), "t", None)


def test_attach_adds_a_result() -> None:
    trial = make_trial()
    updated = attach_accuracy(trial, make_result("a"))
    assert [item.task for item in updated.accuracy] == ["a"]
    assert trial.accuracy == []


def test_attach_replaces_same_task() -> None:
    first = attach_accuracy(make_trial(), make_result("a", score=1.0))
    second = attach_accuracy(first, make_result("a", score=2.0))
    assert len(second.accuracy) == 1
    assert second.accuracy[0].score == 2.0


def test_attach_keeps_other_tasks() -> None:
    first = attach_accuracy(make_trial(), make_result("a"))
    second = attach_accuracy(first, make_result("b"))
    assert [item.task for item in second.accuracy] == ["a", "b"]