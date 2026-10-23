"""Baseline score files, and attaching accuracy results to a saved trial."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from llmopt.schemas import AccuracyResult, TrialResult


class Baseline(BaseModel):
    """A reference score. limit is None when the full default run was used."""

    task: str
    score: float = Field(gt=0)
    model: str
    limit: int | None = None


def write_baseline(baseline: Baseline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2) + "\n")


def read_baseline(path: Path) -> Baseline:
    """Raises FileNotFoundError if missing and ValueError if the content is not valid."""
    if not path.is_file():
        raise FileNotFoundError(f"baseline file not found: {path}")
    return Baseline.model_validate_json(path.read_text())


def check_baseline(baseline: Baseline, task: str, limit: int | None) -> None:
    """Refuse a baseline that cannot be compared with the run we are about to do."""
    if baseline.task != task:
        raise ValueError(f"baseline is for task {baseline.task!r}, not {task!r}")
    if baseline.limit != limit:
        raise ValueError(
            f"baseline was made with limit {baseline.limit}, this run uses limit {limit}; "
            "the scores are not comparable"
        )


def attach_accuracy(trial: TrialResult, result: AccuracyResult) -> TrialResult:
    """Return a copy of the trial with this result added. An older result for the
    same task is replaced, so re-running validate does not leave duplicates."""
    kept = [item for item in trial.accuracy if item.task != result.task]
    return trial.model_copy(update={"accuracy": [*kept, result]})