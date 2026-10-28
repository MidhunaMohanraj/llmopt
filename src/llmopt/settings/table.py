"""Tunable vLLM settings from AMD's vLLM V1 optimization guide (single GPU only).

Source: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
Every entry is verified=False: names and defaults come from that guide, not from a run on
vllm 0.32.0 and a GPU. Multi-GPU settings (parallelism, RCCL) are Phase 2 and excluded.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from llmopt.schemas import SettingValue

Kind = Literal["int", "float", "bool", "choice"]
Target = Literal["arg", "env", "compilation_config"]


class Setting(BaseModel):
    name: str
    kind: Kind
    default: bool | int | float | str | None
    target: Target
    flag: str  # CLI flag, environment variable, or compilation-config key
    note: str
    choices: list[str] = []
    low: float | None = None
    high: float | None = None
    negatable: bool = False  # bool arg that has a --no-<flag> form
    verified: bool = False


SETTINGS: list[Setting] = [
    Setting(
        name="gpu_memory_utilization", kind="float", default=0.9, target="arg",
        flag="--gpu-memory-utilization", low=0.1, high=0.95,
        note="Fraction of GPU memory vLLM may use. Guide: default 0.9, up to 0.95.",
    ),
    Setting(
        name="max_num_seqs", kind="int", default=1024, target="arg",
        flag="--max-num-seqs", low=1, high=4096,
        note="Guide: 128 or 256 saves memory; 2048 or 4096 favors throughput.",
    ),
    Setting(
        name="max_num_batched_tokens", kind="int", default=8192, target="arg",
        flag="--max-num-batched-tokens", low=1024, high=65536,
        note="Guide: 8k to 16k or less for low TTFT, 32k or more for offline throughput.",
    ),
    Setting(
        name="enable_chunked_prefill", kind="bool", default=True, target="arg",
        flag="--enable-chunked-prefill", negatable=True,
        note="On by default in V1. Flag form taken from vllm serve --help (unchecked).",
    ),
    Setting(
        name="enable_prefix_caching", kind="bool", default=None, target="arg",
        flag="--enable-prefix-caching", negatable=True,
        note="The guide gives no recipe; engine default is left unset.",
    ),
    Setting(
        name="kv_cache_dtype", kind="choice", default="auto", target="arg",
        flag="--kv-cache-dtype", choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
        note="FP8 KV cache. Models without calibrated scales need calculate_kv_scales.",
    ),
    Setting(
        name="calculate_kv_scales", kind="bool", default=False, target="arg",
        flag="--calculate-kv-scales",
        note="Dynamic KV scales when the model has none. Unchecked on vllm 0.32.0.",
    ),
    Setting(
        name="cudagraph_mode", kind="choice", default="FULL_AND_PIECEWISE",
        target="compilation_config", flag="cudagraph_mode",
        choices=["NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"],
        note="Graph capture mode, passed through --compilation-config.",
    ),
    Setting(
        name="aiter", kind="bool", default=False, target="env",
        flag="VLLM_ROCM_USE_AITER",
        note="Master switch for AITER kernels. Sub-flags default to on when this is on.",
    ),
    Setting(
        name="shuffle_kv_cache_layout", kind="bool", default=False, target="env",
        flag="VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT",
        note="Guide: for high concurrency (about 32 or more) with the ROCM_AITER_FA backend.",
    ),
    Setting(
        name="attention_backend", kind="choice", default=None, target="arg",
        flag="--attention-backend",
        choices=["ROCM_AITER_FA", "ROCM_ATTN", "TRITON_ATTN", "ROCM_AITER_UNIFIED_ATTN"],
        note="Unset means automatic. ROCM_AITER_FA is chosen for Llama and Qwen with AITER on.",
    ),
    Setting(
        name="hipblaslt_preference", kind="bool", default=False, target="env",
        flag="TORCH_BLAS_PREFER_HIPBLASLT",
        note="Prefer hipBLASLt for GEMMs.",
    ),
    Setting(
        name="async_scheduling", kind="bool", default=False, target="arg",
        flag="--async-scheduling",
        note="Guide: trades TTFT for throughput.",
    ),
]

BY_NAME: dict[str, Setting] = {s.name: s for s in SETTINGS}


class Rendered(BaseModel):
    args: list[str]
    env: dict[str, str]


def _check(setting: Setting, value: SettingValue) -> str | None:
    name = setting.name
    if setting.kind == "bool":
        return None if isinstance(value, bool) else f"{name} must be true or false, got {value!r}"
    if setting.kind == "choice":
        if value in setting.choices:
            return None
        return f"{name} must be one of {', '.join(setting.choices)}, got {value!r}"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return f"{name} must be a number, got {value!r}"
    if setting.kind == "int" and not isinstance(value, int):
        return f"{name} must be a whole number, got {value!r}"
    if setting.low is not None and value < setting.low:
        return f"{name} must be at least {setting.low:g}, got {value!r}"
    if setting.high is not None and value > setting.high:
        return f"{name} must be at most {setting.high:g}, got {value!r}"
    return None


def validate_settings(values: dict[str, SettingValue]) -> list[str]:
    """Return a list of problems; empty means every setting is known and in range."""
    errors: list[str] = []
    for name, value in values.items():
        setting = BY_NAME.get(name)
        if setting is None:
            errors.append(f"unknown setting {name!r}; known: {', '.join(BY_NAME)}")
            continue
        problem = _check(setting, value)
        if problem is not None:
            errors.append(problem)
    return errors


def render_settings(values: dict[str, SettingValue]) -> Rendered:
    """Turn validated settings into vllm arguments and environment variables."""
    errors = validate_settings(values)
    if errors:
        raise ValueError("; ".join(errors))
    args: list[str] = []
    env: dict[str, str] = {}
    compilation: dict[str, str] = {}
    for setting in SETTINGS:
        if setting.name not in values:
            continue
        value = values[setting.name]
        if setting.target == "env":
            env[setting.flag] = "1" if value is True else "0" if value is False else str(value)
        elif setting.target == "compilation_config":
            compilation[setting.flag] = str(value)
        elif setting.kind == "bool":
            if value is True:
                args.append(setting.flag)
            elif setting.negatable:
                args.append("--no-" + setting.flag[2:])
        else:
            args.extend([setting.flag, str(value)])
    if compilation:
        args.extend(["--compilation-config", json.dumps(compilation, sort_keys=True)])
    return Rendered(args=args, env=env)