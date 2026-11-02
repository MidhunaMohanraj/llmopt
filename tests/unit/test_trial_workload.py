from pathlib import Path

from llmopt.report.store import load_trials, save_trial, trial_path
from llmopt.schemas import Config, Engine, Precision, Status, TrialResult


def make_trial(workload: str | None) -> TrialResult:
    config = Config(model="/models/example", engine=Engine.VLLM, precision=Precision.BF16)
    return TrialResult(config=config, status=Status.OK, workload=workload)


def test_two_workloads_do_not_overwrite_each_other(tmp_path: Path) -> None:
    save_trial(make_trial("smoke"), tmp_path)
    save_trial(make_trial("chat"), tmp_path)
    loaded = load_trials(tmp_path)
    assert sorted(t.workload or "" for t in loaded) == ["chat", "smoke"]


def test_same_workload_replaces_its_own_result(tmp_path: Path) -> None:
    save_trial(make_trial("smoke"), tmp_path)
    save_trial(make_trial("smoke"), tmp_path)
    assert len(load_trials(tmp_path)) == 1


def test_file_name_without_a_workload_is_unchanged(tmp_path: Path) -> None:
    trial = make_trial(None)
    assert trial_path(tmp_path, trial) == tmp_path / f"{trial.config.key()}.json"


def test_file_name_includes_the_workload(tmp_path: Path) -> None:
    trial = make_trial("long")
    assert trial_path(tmp_path, trial).name == f"{trial.config.key()}-long.json"


def test_old_trial_files_without_a_workload_still_load() -> None:
    config = Config(model="/models/example", engine=Engine.VLLM, precision=Precision.BF16)
    old = TrialResult(config=config, status=Status.OK).model_dump_json()
    assert "workload" in old  # the field exists, defaulting to null
    assert TrialResult.model_validate_json(old).workload is None