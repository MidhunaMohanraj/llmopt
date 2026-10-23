"""Save and load results as JSON: one file per trial, and one file for a whole report."""

from __future__ import annotations

from pathlib import Path

from llmopt.schemas import Report, TrialResult


def save_report(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n")


def load_report(path: Path) -> Report:
    return Report.model_validate_json(path.read_text())


def trial_path(directory: Path, trial: TrialResult) -> Path:
    """One file per config, named by the config's own short key."""
    return directory / f"{trial.config.key()}.json"


def save_trial(trial: TrialResult, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = trial_path(directory, trial)
    path.write_text(trial.model_dump_json(indent=2) + "\n")
    return path


def load_trial(path: Path) -> TrialResult:
    return TrialResult.model_validate_json(path.read_text())


def load_trials(directory: Path) -> list[TrialResult]:
    """Load every .json file in the folder, in file name order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"results folder not found: {directory}")
    return [load_trial(path) for path in sorted(directory.glob("*.json"))]