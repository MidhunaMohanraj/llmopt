"""vLLM adapter for AMD GPUs (ROCm).

DRAFT: NOT TESTED ON A GPU. Argument names were checked against vLLM 0.32.0 inside the pinned
image (no GPU needed for that). FP8 and FP8 KV cache behaviour on ROCm (including the FNUZ
variant on gfx942) and the effect of every tunable setting must be confirmed on a real MI300X
before any number from this adapter is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llmopt.engines.base import SupportResult
from llmopt.engines.pytorch_cpu import build_prompt_ids
from llmopt.engines.vllm_cpu import VllmCpuEngine, _int_setting
from llmopt.schemas import Config, Engine, HardwareProfile, Precision
from llmopt.settings.apply import apply_settings, restore_env, set_env

SUPPORTED_PRECISIONS = (Precision.BF16, Precision.FP8)
SUPPORTED_KV_DTYPES = ("auto", "fp8")
# Settings this adapter handles: its own three, plus the tunable vLLM settings.
ROCM_SETTINGS = frozenset(
    {
        "max_model_len",
        "gpu_memory_utilization",
        "quantization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "cudagraph_mode",
        "attention_backend",
        "async_scheduling",
        "aiter",
        "shuffle_kv_cache_layout",
        "hipblaslt_preference",
    }
)


def _float_setting(config: Config, key: str, default: float) -> float:
    value = config.settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"setting '{key}' must be a number, got {value!r}")
    if not 0 < value <= 1:
        raise ValueError(f"setting '{key}' must be above 0 and at most 1, got {value}")
    return float(value)


def check_config(config: Config) -> None:
    """Reject configs this adapter cannot run, before anything is loaded."""
    if config.engine is not Engine.VLLM:
        raise ValueError(f"vllm adapter needs engine=vllm, got {config.engine.value}")
    if config.precision not in SUPPORTED_PRECISIONS:
        names = ", ".join(p.value for p in SUPPORTED_PRECISIONS)
        raise ValueError(f"vllm rocm adapter supports {names}, got {config.precision.value}")
    if config.precision is not Precision.FP8 and "quantization" in config.settings:
        raise ValueError("setting 'quantization' only applies when precision is fp8")
    if config.kv_cache_dtype not in SUPPORTED_KV_DTYPES:
        raise ValueError(
            f"kv_cache_dtype must be one of {', '.join(SUPPORTED_KV_DTYPES)}, "
            f"got {config.kv_cache_dtype!r}"
        )
    if not Path(config.model).is_dir():
        raise FileNotFoundError(f"model folder not found: {config.model}")


def build_llm_kwargs(config: Config) -> dict[str, Any]:
    """Arguments for vllm.LLM. Prefix caching is off so TTFT is not flattered."""
    kwargs: dict[str, Any] = {
        "model": config.model,
        "dtype": "bfloat16",
        "max_model_len": _int_setting(config, "max_model_len", 2048),
        "gpu_memory_utilization": _float_setting(config, "gpu_memory_utilization", 0.9),
        "enable_prefix_caching": False,
        "kv_cache_dtype": config.kv_cache_dtype,
    }
    if config.precision is Precision.FP8:
        kwargs["quantization"] = str(config.settings.get("quantization", "fp8"))
    kwargs.update(apply_settings(config.settings, ROCM_SETTINGS, "vllm-rocm").kwargs)
    return kwargs


def build_env(config: Config) -> dict[str, str]:
    """Environment variables to set before vLLM starts."""
    return apply_settings(config.settings, ROCM_SETTINGS, "vllm-rocm").env


class VllmRocmEngine(VllmCpuEngine):
    name = "vllm-rocm"

    def __init__(self) -> None:
        super().__init__()
        self._previous_env: dict[str, str | None] = {}

    def supports(
        self, hardware: HardwareProfile, model: str, precision: Precision
    ) -> SupportResult:
        if precision not in SUPPORTED_PRECISIONS:
            return SupportResult(
                ok=False, reason=f"vllm rocm adapter does not support {precision.value}"
            )
        if not Path(model).is_dir():
            return SupportResult(ok=False, reason=f"model folder not found: {model}")
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        check_config(config)
        kwargs = build_llm_kwargs(config)
        self._previous_env = set_env(build_env(config))
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self._tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
        llm: Any = LLM(**kwargs)
        warm_ids = build_prompt_ids(self._tokenizer, 32)
        warm = SamplingParams(temperature=0.0, max_tokens=4, min_tokens=4, ignore_eos=True)
        llm.generate([{"prompt_token_ids": warm_ids}], warm)
        self._llm = llm
        self._max_model_len = int(kwargs["max_model_len"])

    def teardown(self) -> None:
        try:
            super().teardown()
        finally:
            restore_env(self._previous_env)
            self._previous_env = {}
