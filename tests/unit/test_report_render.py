from __future__ import annotations

from pathlib import Path

from llmopt.report.render import render_report, write_report
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


def make_trial(model: str = "m", spread: float = 1.0) -> TrialResult:
    return TrialResult(
        config=Config(model=model, engine=Engine.PYTORCH, precision=Precision.BF16),
        status=Status.OK,
        metrics=Metrics(
            ttft_ms_p50=40.0,
            ttft_ms_p95=41.0,
            tpot_ms=30.0,
            throughput_tps=30.0,
            latency_ms_p50=500.0,
            latency_ms_p95=510.0,
            peak_memory_gb=2.0,
            run_spread_pct=spread,
        ),
        accuracy=[
            AccuracyResult(task="arc_easy_local", score=0.6, baseline_score=0.6, drop_pct=0.0)
        ],
    )


def make_report(trials: list[TrialResult] | None = None) -> Report:
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
        trials=[make_trial()] if trials is None else trials,
        versions={"llmopt": "test"},
    )


def test_page_has_the_main_sections() -> None:
    page = render_report(make_report())
    assert page.startswith("<!DOCTYPE html>")
    for heading in ("Hardware", "Workload", "Objective", "Trials", "Recommendation", "Versions"):
        assert f"<h2>{heading}</h2>" in page
    assert "test-gpu" in page
    assert "arc_easy_local" in page


def test_values_are_escaped() -> None:
    page = render_report(make_report([make_trial(model="<script>alert(1)</script>")]))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_notice_is_shown_only_when_given() -> None:
    assert 'class="notice"' not in render_report(make_report())
    page = render_report(make_report(), notice="smoke run, not a benchmark")
    assert "smoke run, not a benchmark" in page


def test_unstable_spread_is_marked() -> None:
    assert "(unstable?)" in render_report(make_report([make_trial(spread=7.5)]))
    assert "(unstable?)" not in render_report(make_report([make_trial(spread=3.0)]))


def test_rejected_trial_shows_its_reason() -> None:
    trial = TrialResult(
        config=Config(model="m", engine=Engine.VLLM, precision=Precision.FP8),
        status=Status.REJECTED,
        reason="fp8 not supported here",
    )
    page = render_report(make_report([trial]))
    assert "rejected: fp8 not supported here" in page


def test_no_recommendation_message() -> None:
    assert "No recommendation." in render_report(make_report())


def test_write_report_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "out" / "report.html"
    write_report(make_report(), path)
    assert path.read_text().startswith("<!DOCTYPE html>")