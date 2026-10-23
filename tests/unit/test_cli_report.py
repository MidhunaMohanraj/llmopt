from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llmopt.cli import app
from llmopt.report.store import save_report, save_trial
from llmopt.schemas import (
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

runner = CliRunner()


def make_trial() -> TrialResult:
    return TrialResult(
        config=Config(model="m", engine=Engine.PYTORCH, precision=Precision.BF16),
        status=Status.OK,
        metrics=Metrics(
            ttft_ms_p50=40.0,
            ttft_ms_p95=41.0,
            tpot_ms=30.0,
            throughput_tps=30.0,
            latency_ms_p50=500.0,
            latency_ms_p95=510.0,
            peak_memory_gb=2.0,
        ),
    )


def make_run_file(path: Path) -> None:
    report = Report(
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
    )
    save_report(report, path)


def test_report_command_writes_html(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    results = tmp_path / "results"
    out = tmp_path / "report.html"
    make_run_file(run)
    save_trial(make_trial(), results)
    result = runner.invoke(
        app,
        ["report", "--run", str(run), "--results", str(results), "--output", str(out)],
    )
    assert result.exit_code == 0
    assert "1 trial(s)" in result.output
    page = out.read_text()
    assert "<h2>Trials</h2>" in page
    assert "test-gpu" in page


def test_report_command_shows_notice(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    results = tmp_path / "results"
    out = tmp_path / "report.html"
    make_run_file(run)
    save_trial(make_trial(), results)
    result = runner.invoke(
        app,
        [
            "report", "--run", str(run), "--results", str(results),
            "--output", str(out), "--notice", "smoke run",
        ],
    )
    assert result.exit_code == 0
    assert "smoke run" in out.read_text()


def test_report_command_fails_cleanly_on_missing_results(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    make_run_file(run)
    result = runner.invoke(
        app, ["report", "--run", str(run), "--results", str(tmp_path / "nope")]
    )
    assert result.exit_code == 1