from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import llmopt.cli as cli_module
from llmopt.cli import app
from llmopt.profile.factory import make_config
from llmopt.report.store import load_trial, save_trial
from llmopt.schemas import Status, TrialResult
from llmopt.validate.baseline import Baseline, read_baseline, write_baseline

runner = CliRunner()


def use_score(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    def fake(task: str, model: Path, datasets: Path, limit: int | None) -> float:
        return value

    monkeypatch.setattr(cli_module, "_score", fake)


def make_baseline(
    tmp_path: Path, task: str, score: float, limit: int | None = None
) -> Path:
    path = tmp_path / "base.json"
    write_baseline(Baseline(task=task, score=score, model="/m", limit=limit), path)
    return path


def test_write_baseline_saves_the_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_score(monkeypatch, 12.5)
    path = tmp_path / "base.json"
    result = runner.invoke(
        app,
        ["validate", "--model", "/m", "--task", "ppl", "--write-baseline", str(path)],
    )
    assert result.exit_code == 0
    saved = read_baseline(path)
    assert saved.task == "wikitext2_perplexity"
    assert saved.score == 12.5
    assert saved.limit is None


def test_compare_reports_a_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_score(monkeypatch, 0.5)
    base = make_baseline(tmp_path, "arc_easy_local", 0.6)
    result = runner.invoke(
        app, ["validate", "--model", "/m", "--task", "arc", "--baseline", str(base)]
    )
    assert result.exit_code == 0
    assert "drop +16.67%" in result.output


def test_compare_attaches_to_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_score(monkeypatch, 0.5)
    base = make_baseline(tmp_path, "arc_easy_local", 0.6)
    trial_file = tmp_path / "renamed.json"
    saved = save_trial(
        TrialResult(config=make_config("pytorch", Path("/m"), {}), status=Status.OK),
        tmp_path / "res",
    )
    saved.rename(trial_file)
    args = ["validate", "--model", "/m", "--task", "arc", "--baseline", str(base)]
    result = runner.invoke(app, [*args, "--trial", str(trial_file)])
    assert result.exit_code == 0
    assert len(load_trial(trial_file).accuracy) == 1
    assert not list((tmp_path / "res").glob("*.json"))


def test_missing_trial_file_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_score(monkeypatch, 0.5)
    base = make_baseline(tmp_path, "arc_easy_local", 0.6)
    args = ["validate", "--model", "/m", "--task", "arc", "--baseline", str(base)]
    result = runner.invoke(app, [*args, "--trial", str(tmp_path / "no.json")])
    assert result.exit_code == 1


def test_needs_exactly_one_of_baseline_or_write() -> None:
    result = runner.invoke(app, ["validate", "--model", "/m", "--task", "ppl"])
    assert result.exit_code == 1


def test_rejects_unknown_task(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    result = runner.invoke(
        app,
        ["validate", "--model", "/m", "--task", "x", "--write-baseline", str(path)],
    )
    assert result.exit_code == 1


def test_refuses_limited_baseline(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    result = runner.invoke(
        app,
        [
            "validate", "--model", "/m", "--task", "ppl",
            "--write-baseline", str(path), "--limit", "2",
        ],
    )
    assert result.exit_code == 1
    assert not path.exists()


def test_rejects_baseline_for_other_task(tmp_path: Path) -> None:
    base = make_baseline(tmp_path, "wikitext2_perplexity", 10.0)
    result = runner.invoke(
        app, ["validate", "--model", "/m", "--task", "arc", "--baseline", str(base)]
    )
    assert result.exit_code == 1


def test_rejects_baseline_with_other_limit(tmp_path: Path) -> None:
    base = make_baseline(tmp_path, "arc_easy_local", 0.6)
    result = runner.invoke(
        app,
        [
            "validate", "--model", "/m", "--task", "arc",
            "--baseline", str(base), "--limit", "5",
        ],
    )
    assert result.exit_code == 1


def test_missing_baseline_file_exits_1(tmp_path: Path) -> None:
    missing = tmp_path / "no.json"
    result = runner.invoke(
        app,
        ["validate", "--model", "/m", "--task", "arc", "--baseline", str(missing)],
    )
    assert result.exit_code == 1