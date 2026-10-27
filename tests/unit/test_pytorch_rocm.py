from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from llmopt.engines.pytorch_rocm import PyTorchRocmEngine, check_config
from llmopt.schemas import Config, Engine, Precision

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)


def make(
    model: Path,
    precision: Precision = Precision.BF16,
    kv: str = "auto",
    engine: Engine = Engine.PYTORCH,
) -> Config:
    return Config(model=str(model), engine=engine, precision=precision, kv_cache_dtype=kv)


def test_name() -> None:
    assert PyTorchRocmEngine().name == "pytorch-rocm"


def test_valid_config_passes(tmp_path: Path) -> None:
    check_config(make(tmp_path))


def test_wrong_engine_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        check_config(make(tmp_path, engine=Engine.VLLM))


def test_fp8_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        check_config(make(tmp_path, Precision.FP8))


def test_fp8_kv_cache_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        check_config(make(tmp_path, kv="fp8"))


def test_missing_model_folder_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_config(make(tmp_path / "nope"))


def test_prepare_rejects_a_bad_config_before_loading_anything(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PyTorchRocmEngine().prepare(make(tmp_path, Precision.FP8))


@needs_torch
def test_prepare_fails_clearly_without_a_gpu(tmp_path: Path) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("a GPU is visible here")
    with pytest.raises(RuntimeError, match="no GPU"):
        PyTorchRocmEngine().prepare(make(tmp_path))