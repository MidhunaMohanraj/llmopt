"""Tests for the vLLM CPU adapter that need neither vLLM nor model files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llmopt.engines.base import GenRequest, SupportResult
from llmopt.engines.vllm_cpu import VllmCpuEngine, run_request
from llmopt.schemas import Config, Engine, HardwareProfile, Precision


@dataclass
class _Sample:
    token_ids: list[int]
    text: str


@dataclass
class _Out:
    request_id: str
    outputs: list[_Sample]


class _StubEngine:
    """Gives one more token per step, like the vLLM step loop with CUMULATIVE output."""

    def __init__(self, target: int, with_other_request: bool = False) -> None:
        self.target = target
        self.with_other_request = with_other_request
        self.made = 0
        self.request_id = ""
        self.prompt: Any = None

    def add_request(self, request_id: str, prompt: Any, params: Any) -> None:
        self.request_id = request_id
        self.prompt = prompt

    def has_unfinished_requests(self) -> bool:
        return self.made < self.target

    def step(self) -> list[_Out]:
        self.made += 1
        mine = _Out(self.request_id, [_Sample(list(range(self.made)), "x" * self.made)])
        if self.with_other_request:
            return [_Out("other", [_Sample([0], "o")]), mine]
        return [mine]


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
        "engine": Engine.VLLM,
        "precision": Precision.BF16,
    }
    values.update(changes)
    return Config(**values)  # type: ignore[arg-type]


def test_run_request_gives_one_time_per_token() -> None:
    stub = _StubEngine(target=5)
    result = run_request(stub, object(), [7, 8, 9], "vllm-id", "public-id", 5)
    assert result.request_id == "public-id"
    assert len(result.token_times_ms) == 5
    assert result.token_times_ms == sorted(result.token_times_ms)
    assert result.text == "xxxxx"
    assert stub.request_id == "vllm-id"
    assert stub.prompt == {"prompt_token_ids": [7, 8, 9]}


def test_run_request_rejects_wrong_token_count() -> None:
    with pytest.raises(RuntimeError, match="expected 5"):
        run_request(_StubEngine(target=3), object(), [1], "a", "b", 5)


def test_run_request_ignores_other_requests() -> None:
    stub = _StubEngine(target=4, with_other_request=True)
    result = run_request(stub, object(), [1], "mine", "mine", 4)
    assert len(result.token_times_ms) == 4


def test_supports_accepts_bf16_with_existing_folder(tmp_path: Path) -> None:
    result = VllmCpuEngine().supports(_hardware(), str(tmp_path), Precision.BF16)
    assert result == SupportResult(ok=True)


def test_supports_rejects_fp8_and_missing_folder(tmp_path: Path) -> None:
    engine = VllmCpuEngine()
    assert not engine.supports(_hardware(), str(tmp_path), Precision.FP8).ok
    assert not engine.supports(_hardware(), "/nonexistent", Precision.BF16).ok


def test_prepare_rejects_wrong_engine_precision_and_kv_dtype() -> None:
    engine = VllmCpuEngine()
    with pytest.raises(ValueError, match="engine"):
        engine.prepare(_config(engine=Engine.PYTORCH))
    with pytest.raises(ValueError, match="bf16"):
        engine.prepare(_config(precision=Precision.FP8))
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        engine.prepare(_config(kv_cache_dtype="fp8"))


def test_prepare_rejects_bad_settings() -> None:
    engine = VllmCpuEngine()
    with pytest.raises(TypeError, match="integer"):
        engine.prepare(_config(settings={"kv_cache_space_gib": "four"}))
    with pytest.raises(ValueError, match="greater than 0"):
        engine.prepare(_config(settings={"max_model_len": 0}))


def test_generate_before_prepare_fails() -> None:
    request = GenRequest(request_id="r", prompt_tokens=4, max_new_tokens=2)
    with pytest.raises(RuntimeError, match="prepare"):
        VllmCpuEngine().generate([request])
