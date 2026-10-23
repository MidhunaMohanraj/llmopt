import pytest
from pydantic import ValidationError

from llmopt.schemas import (
    Config,
    Engine,
    HardwareProfile,
    Precision,
    Report,
    Status,
    TrialResult,
    Workload,
)


def make_config(**settings: int) -> Config:
    return Config(
        model="Qwen/Qwen3-8B",
        engine=Engine.VLLM,
        precision=Precision.BF16,
        settings=dict(settings),
    )


def test_config_key_is_stable_and_ignores_setting_order() -> None:
    a = make_config(max_num_seqs=64, max_num_batched_tokens=8192)
    b = make_config(max_num_batched_tokens=8192, max_num_seqs=64)
    assert a.key() == b.key()


def test_config_key_changes_when_settings_change() -> None:
    assert make_config(max_num_seqs=64).key() != make_config(max_num_seqs=128).key()


def test_workload_rejects_zero_tokens() -> None:
    with pytest.raises(ValidationError):
        Workload(name="bad", prompt_tokens=0, output_tokens=256)


def test_report_round_trips_through_json() -> None:
    report = Report(
        hardware=HardwareProfile(
            gpu_name="AMD Instinct MI300X",
            gfx_target="gfx942",
            compute_units=304,
            hbm_gb=192,
            rocm_version="unknown",
            driver_version="unknown",
            cpu_model="AMD EPYC",
            cpu_cores=96,
        ),
        workload=Workload(name="chat-short", prompt_tokens=1024, output_tokens=256),
        objective={"max_ttft_p95_ms": 500, "max_accuracy_drop_pct": 1},  # type: ignore[arg-type]
        trials=[TrialResult(config=make_config(), status=Status.REJECTED, reason="too slow")],
    )
    again = Report.model_validate_json(report.model_dump_json())
    assert again == report
