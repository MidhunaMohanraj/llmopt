"""Tests for the PyTorch CPU adapter that need no model files."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.engines.base import GenRequest, SupportResult
from llmopt.engines.pytorch_cpu import PyTorchCpuEngine, build_prompt_ids
from llmopt.schemas import Config, Engine, HardwareProfile, Precision


class _StubTokenizer:
    def encode(self, text: str, **_: object) -> list[int]:
        return [len(text), 22, 33]


def _hardware() -> HardwareProfile:
    # Placeholder values. The adapter ignores the hardware profile.
    return HardwareProfile(
        gpu_name="none",
        gfx_target="none",
        compute_units=1,
        hbm_gb=1.0,
        rocm_version="none",
        driver_version="none",
        cpu_model="test",
        cpu_cores=1,
    )


def _config(**changes: object) -> Config:
    values: dict[str, object] = {
        "model": "/nonexistent",
        "engine": Engine.PYTORCH,
        "precision": Precision.BF16,
    }
    values.update(changes)
    return Config(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("n", [1, 3, 10, 100])
def test_prompt_ids_have_exact_length(n: int) -> None:
    assert len(build_prompt_ids(_StubTokenizer(), n)) == n


def test_supports_accepts_bf16_with_existing_folder(tmp_path: Path) -> None:
    result = PyTorchCpuEngine().supports(_hardware(), str(tmp_path), Precision.BF16)
    assert result == SupportResult(ok=True)


def test_supports_rejects_fp8_and_missing_folder(tmp_path: Path) -> None:
    engine = PyTorchCpuEngine()
    assert not engine.supports(_hardware(), str(tmp_path), Precision.FP8).ok
    assert not engine.supports(_hardware(), "/nonexistent", Precision.BF16).ok


def test_prepare_rejects_wrong_engine_precision_and_kv_dtype() -> None:
    engine = PyTorchCpuEngine()
    with pytest.raises(ValueError, match="engine"):
        engine.prepare(_config(engine=Engine.VLLM))
    with pytest.raises(ValueError, match="bf16"):
        engine.prepare(_config(precision=Precision.FP8))
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        engine.prepare(_config(kv_cache_dtype="fp8"))


def test_generate_before_prepare_fails() -> None:
    request = GenRequest(request_id="r", prompt_tokens=4, max_new_tokens=2)
    with pytest.raises(RuntimeError, match="prepare"):
        PyTorchCpuEngine().generate([request])