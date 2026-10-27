from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.engines.vllm_rocm import VllmRocmEngine, build_llm_kwargs, check_config
from llmopt.schemas import Config, Engine, HardwareProfile, Precision, SettingValue

HARDWARE = HardwareProfile(
    gpu_name="test-gpu",
    gfx_target="gfx000",
    compute_units=1,
    hbm_gb=1.0,
    rocm_version="0",
    driver_version="0",
    cpu_model="test-cpu",
    cpu_cores=1,
)


def make(
    model: Path,
    precision: Precision = Precision.BF16,
    kv: str = "auto",
    settings: dict[str, SettingValue] | None = None,
    engine: Engine = Engine.VLLM,
) -> Config:
    return Config(
        model=str(model),
        engine=engine,
        precision=precision,
        kv_cache_dtype=kv,
        settings=settings or {},
    )


def test_bf16_defaults(tmp_path: Path) -> None:
    kwargs = build_llm_kwargs(make(tmp_path))
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["max_model_len"] == 2048
    assert kwargs["gpu_memory_utilization"] == 0.9
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["kv_cache_dtype"] == "auto"
    assert "quantization" not in kwargs


def test_fp8_sets_quantization(tmp_path: Path) -> None:
    kwargs = build_llm_kwargs(make(tmp_path, Precision.FP8))
    assert kwargs["quantization"] == "fp8"


def test_quantization_can_be_overridden(tmp_path: Path) -> None:
    config = make(tmp_path, Precision.FP8, settings={"quantization": "other"})
    assert build_llm_kwargs(config)["quantization"] == "other"


def test_fp8_kv_cache_is_passed_through(tmp_path: Path) -> None:
    assert build_llm_kwargs(make(tmp_path, kv="fp8"))["kv_cache_dtype"] == "fp8"


def test_gpu_memory_utilization_range(tmp_path: Path) -> None:
    config = make(tmp_path, settings={"gpu_memory_utilization": 1.5})
    with pytest.raises(ValueError):
        build_llm_kwargs(config)


def test_gpu_memory_utilization_type(tmp_path: Path) -> None:
    config = make(tmp_path, settings={"gpu_memory_utilization": True})
    with pytest.raises(TypeError):
        build_llm_kwargs(config)


def test_wrong_engine_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        check_config(make(tmp_path, engine=Engine.PYTORCH))


def test_unsupported_precision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        check_config(make(tmp_path, Precision.AWQ_INT4))


def test_bad_kv_cache_dtype_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        check_config(make(tmp_path, kv="int4"))


def test_missing_model_folder_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_config(make(tmp_path / "nope"))


def test_supports(tmp_path: Path) -> None:
    engine = VllmRocmEngine()
    assert engine.name == "vllm-rocm"
    assert engine.supports(HARDWARE, str(tmp_path), Precision.FP8).ok
    assert not engine.supports(HARDWARE, str(tmp_path), Precision.AWQ_INT4).ok
    assert not engine.supports(HARDWARE, str(tmp_path / "nope"), Precision.BF16).ok


def test_prepare_rejects_a_bad_config_before_loading_vllm(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        VllmRocmEngine().prepare(make(tmp_path, Precision.AWQ_INT4))