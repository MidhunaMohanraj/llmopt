from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from llmopt.schemas import AccuracyResult, TrialResult


class Baseline(BaseModel):
    task: str
    score: float = Field(gt=0)
    model: str
    limit: int | None = None


def write_baseline(baseline: Baseline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2) + "\n")


def read_baseline(path: Path) -> Baseline:
    if not path.is_file():
        raise FileNotFoundError(f"baseline file not found: {path}")
    return Baseline.model_validate_json(path.read_text())


def check_baseline(baseline: Baseline, task: str, limit: int | None) -> None:
    if baseline.task != task:
        raise ValueError(f"baseline is for task {baseline.task!r}, not {task!r}")
    if baseline.limit != limit:
        raise ValueError(
            f"baseline was made with limit {baseline.limit}, this run uses limit {limit}; "
            "the scores are not comparable"
        )


def attach_accuracy(trial: TrialResult, result: AccuracyResult) -> TrialResult:
    kept = [item for item in trial.accuracy if item.task != result.task]
    return trial.model_copy(update={"accuracy": [*kept, result]})