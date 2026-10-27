"""Build an engine adapter and its Config from simple command-line values."""

from __future__ import annotations

from pathlib import Path

from llmopt.engines.base import EngineAdapter
from llmopt.schemas import Config, Engine, Precision, SettingValue

ENGINE_NAMES = ("pytorch", "vllm", "pytorch-rocm", "vllm-rocm")
KV_CACHE_DTYPES = ("auto", "fp8")


def _coerce(text: str) -> SettingValue:
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


def parse_settings(text: str) -> dict[str, SettingValue]:
    """Parse 'a=1,b=true,c=name' into typed values. Empty text gives no settings."""
    settings: dict[str, SettingValue] = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"setting must look like key=value, got {item!r}")
        settings[key.strip()] = _coerce(value.strip())
    return settings


def _unknown_engine(engine: str) -> ValueError:
    return ValueError(f"engine must be one of {', '.join(ENGINE_NAMES)}, got {engine!r}")


def parse_precision(text: str) -> Precision:
    """Turn 'bf16' or 'fp8' into a Precision, with a clear error for anything else."""
    try:
        return Precision(text.strip().lower())
    except ValueError:
        valid = ", ".join(p.value for p in Precision)
        raise ValueError(f"precision must be one of {valid}, got {text!r}") from None


def check_kv_cache_dtype(value: str) -> str:
    if value not in KV_CACHE_DTYPES:
        valid = ", ".join(KV_CACHE_DTYPES)
        raise ValueError(f"kv_cache_dtype must be one of {valid}, got {value!r}")
    return value


def make_config(
    engine: str,
    model: Path,
    settings: dict[str, SettingValue],
    precision: str = "bf16",
    kv_cache_dtype: str = "auto",
) -> Config:
    if engine in ("pytorch", "pytorch-rocm"):
        kind = Engine.PYTORCH
    elif engine in ("vllm", "vllm-rocm"):
        kind = Engine.VLLM
    else:
        raise _unknown_engine(engine)
    return Config(
        model=str(model),
        engine=kind,
        precision=parse_precision(precision),
        kv_cache_dtype=check_kv_cache_dtype(kv_cache_dtype),
        settings=settings,
    )


def make_engine(engine: str) -> EngineAdapter:
    """Create the adapter. Imports are inside so the CLI starts fast."""
    if engine == "pytorch":
        from llmopt.engines.pytorch_cpu import PyTorchCpuEngine

        return PyTorchCpuEngine()
    if engine == "vllm":
        from llmopt.engines.vllm_cpu import VllmCpuEngine

        return VllmCpuEngine()
    if engine == "pytorch-rocm":
        from llmopt.engines.pytorch_rocm import PyTorchRocmEngine

        return PyTorchRocmEngine()
    if engine == "vllm-rocm":
        from llmopt.engines.vllm_rocm import VllmRocmEngine

        return VllmRocmEngine()
    raise _unknown_engine(engine)
