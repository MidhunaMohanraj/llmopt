from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace
from typing import Any

import pytest

from llmopt.validate.perplexity import (
    evaluate_perplexity,
    perplexity_accuracy,
    perplexity_from_nll,
)

VOCAB = 50

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)


class UniformModel:
    """Predicts every token with equal probability, so perplexity must equal VOCAB."""

    def __call__(self, input_ids: Any, **kwargs: Any) -> SimpleNamespace:
        import torch

        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], VOCAB))


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % VOCAB for c in text]


def test_perplexity_from_nll() -> None:
    assert perplexity_from_nll(math.log(4) * 10, 10) == pytest.approx(4.0)


def test_perplexity_from_nll_rejects_zero_tokens() -> None:
    with pytest.raises(ValueError):
        perplexity_from_nll(1.0, 0)


@needs_torch
def test_uniform_model_gives_vocab_size() -> None:
    result = evaluate_perplexity(
        UniformModel(), CharTokenizer(), ["a" * 100], window_tokens=10, max_windows=3
    )
    assert result.perplexity == pytest.approx(VOCAB, rel=1e-4)
    assert result.windows == 3
    assert result.tokens_scored == 27


@needs_torch
def test_max_windows_limits_the_sample() -> None:
    result = evaluate_perplexity(
        UniformModel(), CharTokenizer(), ["b" * 1000], window_tokens=10, max_windows=2
    )
    assert result.windows == 2


@needs_torch
def test_too_little_text_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_perplexity(UniformModel(), CharTokenizer(), ["abc"], window_tokens=10)


def test_accuracy_drop_is_positive_when_worse() -> None:
    result = perplexity_accuracy(11.0, 10.0)
    assert result.drop_pct == pytest.approx(10.0)
    assert result.task == "wikitext2_perplexity"