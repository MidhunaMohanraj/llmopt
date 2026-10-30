import os
from pathlib import Path

import pytest

from llmopt.engines.vllm_cpu import VllmCpuEngine
from llmopt.engines.vllm_rocm import (
    VllmRocmEngine,
    build_env,
    build_llm_kwargs,
    check_config,
)
from llmopt.schemas import Config, Engine, Precision, SettingValue
from llmopt.settings.apply import set_env


def make_config(
    tmp_path: Path,
    settings: dict[str, SettingValue],
    precision: Precision = Precision.BF16,
) -> Config:
    return Config(
        model=str(tmp_path), engine=Engine.VLLM, precision=precision, settings=settings
    )


def test_tunable_settings_reach_the_llm_arguments(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": False,
            "gpu_memory_utilization": 0.8,
        },
    )
    kwargs = build_llm_kwargs(config)
    assert kwargs["max_num_seqs"] == 256
    assert kwargs["max_num_batched_tokens"] == 8192
    assert kwargs["enable_chunked_prefill"] is False
    assert kwargs["gpu_memory_utilization"] == 0.8
    assert kwargs["enable_prefix_caching"] is False


def test_defaults_without_settings(tmp_path: Path) -> None:
    config = make_config(tmp_path, {})
    kwargs = build_llm_kwargs(config)
    assert kwargs["gpu_memory_utilization"] == 0.9
    assert "max_num_seqs" not in kwargs
    assert build_env(config) == {}


def test_graph_mode_goes_into_the_compilation_config(tmp_path: Path) -> None:
    kwargs = build_llm_kwargs(make_config(tmp_path, {"cudagraph_mode": "PIECEWISE"}))
    assert kwargs["compilation_config"] == {"cudagraph_mode": "PIECEWISE"}


def test_environment_settings(tmp_path: Path) -> None:
    config = make_config(tmp_path, {"aiter": True, "shuffle_kv_cache_layout": True})
    assert build_env(config) == {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
    }


def test_unavailable_setting_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="calculate_kv_scales"):
        build_llm_kwargs(make_config(tmp_path, {"calculate_kv_scales": True}))


def test_quantization_needs_fp8(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fp8"):
        check_config(make_config(tmp_path, {"quantization": "fp8"}))


def test_cpu_adapter_refuses_gpu_only_settings(tmp_path: Path) -> None:
    config = make_config(tmp_path, {"aiter": True})
    with pytest.raises(ValueError, match="does not apply"):
        VllmCpuEngine().prepare(config)


def test_teardown_restores_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
    engine = VllmRocmEngine()
    engine._previous_env = set_env({"VLLM_ROCM_USE_AITER": "1"})
    assert os.environ["VLLM_ROCM_USE_AITER"] == "1"
    engine.teardown()
    assert "VLLM_ROCM_USE_AITER" not in os.environ
