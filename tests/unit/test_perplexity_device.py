from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest

from llmopt.validate.perplexity import check_device, evaluate_perplexity

VOCAB = 50

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)


class RecordingModel:
    """Predicts every token with equal probability and records where the input was."""

    def __init__(self) -> None:
        self.devices: list[str] = []

    def __call__(self, input_ids: Any, **kwargs: Any) -> SimpleNamespace:
        import torch

        self.devices.append(input_ids.device.type)
        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], VOCAB))


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % VOCAB for c in text]


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_check_device_accepts(device: str) -> None:
    check_device(device)


@pytest.mark.parametrize("device", ["mps", "", "gpu", "cuda:", "cpu:0"])
def test_check_device_rejects(device: str) -> None:
    with pytest.raises(ValueError):
        check_device(device)


@needs_torch
def test_default_device_is_cpu() -> None:
    model = RecordingModel()
    evaluate_perplexity(
        model, CharTokenizer(), ["a" * 100], window_tokens=10, max_windows=3
    )
    assert model.devices == ["cpu", "cpu", "cpu"]


@needs_torch
def test_explicit_cpu_device() -> None:
    model = RecordingModel()
    result = evaluate_perplexity(
        model, CharTokenizer(), ["a" * 100], window_tokens=10, max_windows=2, device="cpu"
    )
    assert model.devices == ["cpu", "cpu"]
    assert result.perplexity == pytest.approx(VOCAB, rel=1e-4)


@needs_torch
def test_bad_device_fails_before_the_model_is_called() -> None:
    model = RecordingModel()
    with pytest.raises(ValueError):
        evaluate_perplexity(
            model, CharTokenizer(), ["a" * 100], window_tokens=10, device="mps"
        )
    assert model.devices == []