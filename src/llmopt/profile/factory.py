"""Build an engine adapter and its Config from simple command-line values."""

from __future__ import annotations

from pathlib import Path

from llmopt.engines.base import EngineAdapter
from llmopt.schemas import Config, Engine, Precision, SettingValue

ENGINE_NAMES = ("pytorch", "vllm")


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


def make_config(engine: str, model: Path, settings: dict[str, SettingValue]) -> Config:
    if engine == "pytorch":
        kind = Engine.PYTORCH
    elif engine == "vllm":
        kind = Engine.VLLM
    else:
        raise _unknown_engine(engine)
    return Config(
        model=str(model), engine=kind, precision=Precision.BF16, settings=settings
    )


def make_engine(engine: str) -> EngineAdapter:
    """Create the adapter. Imports are inside so the CLI starts fast."""
    if engine == "pytorch":
        from llmopt.engines.pytorch_cpu import PyTorchCpuEngine

        return PyTorchCpuEngine()
    if engine == "vllm":
        from llmopt.engines.vllm_cpu import VllmCpuEngine

        return VllmCpuEngine()
    raise _unknown_engine(engine)