from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.report.store import (
    load_report,
    load_trial,
    load_trials,
    save_report,
    save_trial,
    trial_path,
)
from llmopt.schemas import (
    AccuracyResult,
    Config,
    Engine,
    HardwareProfile,
    Metrics,
    Objective,
    Precision,
    Report,
    Status,
    TrialResult,
    Workload,
)


def make_trial(model: str = "m", ttft: float = 40.0) -> TrialResult:
    return TrialResult(
        config=Config(model=model, engine=Engine.PYTORCH, precision=Precision.BF16),
        status=Status.OK,
        metrics=Metrics(
            ttft_ms_p50=ttft,
            ttft_ms_p95=ttft + 1,
            tpot_ms=30.0,
            throughput_tps=30.0,
            latency_ms_p50=500.0,
            latency_ms_p95=510.0,
            peak_memory_gb=2.0,
        ),
        accuracy=[
            AccuracyResult(task="arc_easy_local", score=0.6, baseline_score=0.6, drop_pct=0.0)
        ],
    )


def make_report() -> Report:
    return Report(
        hardware=HardwareProfile(
            gpu_name="test-gpu",
            gfx_target="gfx000",
            compute_units=1,
            hbm_gb=1.0,
            rocm_version="0",
            driver_version="0",
            cpu_model="test-cpu",
            cpu_cores=1,
        ),
        workload=Workload(name="smoke", prompt_tokens=8, output_tokens=8),
        objective=Objective(),
        trials=[make_trial()],
        versions={"llmopt": "test"},
    )


def test_report_round_trip(tmp_path: Path) -> None:
    report = make_report()
    path = tmp_path / "out" / "report.json"
    save_report(report, path)
    assert load_report(path) == report


def test_trial_file_is_named_by_config_key(tmp_path: Path) -> None:
    trial = make_trial()
    path = save_trial(trial, tmp_path)
    assert path == trial_path(tmp_path, trial)
    assert path.name == f"{trial.config.key()}.json"


def test_trial_round_trip(tmp_path: Path) -> None:
    trial = make_trial()
    assert load_trial(save_trial(trial, tmp_path)) == trial


def test_load_trials_reads_every_file(tmp_path: Path) -> None:
    first = make_trial("a")
    second = make_trial("b")
    save_trial(first, tmp_path)
    save_trial(second, tmp_path)
    loaded = load_trials(tmp_path)
    assert len(loaded) == 2
    assert {trial.config.model for trial in loaded} == {"a", "b"}


def test_same_config_overwrites_instead_of_duplicating(tmp_path: Path) -> None:
    save_trial(make_trial(ttft=40.0), tmp_path)
    save_trial(make_trial(ttft=50.0), tmp_path)
    loaded = load_trials(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].metrics is not None
    assert loaded[0].metrics.ttft_ms_p50 == 50.0


def test_load_trials_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trials(tmp_path / "nope")