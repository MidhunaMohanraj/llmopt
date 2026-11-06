import math
from dataclasses import dataclass
from typing import Any

import pytest

from llmopt.engines.vllm_cpu import VllmCpuEngine
from llmopt.validate.perplexity import evaluate_perplexity_engine

TEXTS = ["a b c d e f g h i j"]


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


@dataclass
class Logprob:
    logprob: float


@dataclass
class Output:
    prompt_logprobs: list[Any] | None


class StubLlm:
    def __init__(self, value: float = -1.0, broken: bool = False) -> None:
        self.value = value
        self.broken = broken

    def generate(self, prompts: list[dict[str, list[int]]], params: Any) -> list[Output]:
        if self.broken:
            return [Output(None)]
        ids = prompts[0]["prompt_token_ids"]
        entries: list[Any] = [None] + [{token: Logprob(self.value)} for token in ids[1:]]
        return [Output(entries)]


class StubEngine(VllmCpuEngine):
    def __init__(self, llm: Any, max_len: int) -> None:
        super().__init__()
        self._llm = llm
        self._max_model_len = max_len

    def _make_logprob_params(self) -> Any:
        return None


def test_engine_perplexity_comes_from_the_log_probabilities() -> None:
    lengths: list[int] = []

    def fake(chunk: list[int]) -> list[float]:
        lengths.append(len(chunk))
        return [-2.0] * (len(chunk) - 1)

    result = evaluate_perplexity_engine(
        fake, WordTokenizer(), TEXTS, window_tokens=4, max_windows=2
    )
    assert result.perplexity == pytest.approx(math.exp(2.0))
    assert result.tokens_scored == 6
    assert result.windows == 2
    assert lengths == [4, 4]


def test_engine_perplexity_rejects_a_wrong_number_of_log_probabilities() -> None:
    def fake(chunk: list[int]) -> list[float]:
        return [-1.0]

    with pytest.raises(ValueError, match="expected 3"):
        evaluate_perplexity_engine(
            fake, WordTokenizer(), TEXTS, window_tokens=4, max_windows=2
        )


def test_engine_perplexity_rejects_bad_arguments() -> None:
    def fake(chunk: list[int]) -> list[float]:
        return [-1.0] * (len(chunk) - 1)

    with pytest.raises(ValueError, match="window_tokens"):
        evaluate_perplexity_engine(fake, WordTokenizer(), TEXTS, window_tokens=1)
    with pytest.raises(ValueError, match="max_windows"):
        evaluate_perplexity_engine(fake, WordTokenizer(), TEXTS, max_windows=0)
    with pytest.raises(ValueError, match="fewer than one window"):
        evaluate_perplexity_engine(fake, WordTokenizer(), TEXTS, window_tokens=512)


def test_prompt_logprobs_gives_one_value_per_token_after_the_first() -> None:
    engine = StubEngine(StubLlm(-1.5), 8)
    assert engine.prompt_logprobs([5, 6, 7, 8]) == [-1.5, -1.5, -1.5]


def test_prompt_logprobs_needs_prepare() -> None:
    with pytest.raises(RuntimeError, match="prepare"):
        VllmCpuEngine().prompt_logprobs([1, 2])


def test_prompt_logprobs_rejects_a_prompt_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match="max_model_len"):
        StubEngine(StubLlm(), 4).prompt_logprobs([1, 2, 3, 4])


def test_prompt_logprobs_rejects_missing_log_probabilities() -> None:
    with pytest.raises(RuntimeError, match="every prompt token"):
        StubEngine(StubLlm(broken=True), 8).prompt_logprobs([1, 2, 3])