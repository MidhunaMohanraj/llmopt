"""Tests for concurrent vLLM requests. A stub replaces the vLLM step loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llmopt.engines.base import GenRequest
from llmopt.engines.fake import FakeEngine
from llmopt.engines.vllm_cpu import VllmCpuEngine, run_batch
from llmopt.profile.factory import make_config


@dataclass
class _Sample:
    token_ids: list[int]
    text: str


@dataclass
class _Out:
    request_id: str
    outputs: list[_Sample]


class _BatchStub:
    """Gives every unfinished request one more token per step, like vLLM batching."""

    def __init__(self, targets: dict[str, int]) -> None:
        self.targets = targets
        self.made: dict[str, int] = {}
        self.added: list[str] = []
        self.steps = 0

    def add_request(self, request_id: str, prompt: Any, params: Any) -> None:
        if self.steps:
            raise AssertionError("request added after the first step")
        self.added.append(request_id)
        self.made[request_id] = 0

    def has_unfinished_requests(self) -> bool:
        return any(self.made[i] < self.targets[i] for i in self.made)

    def step(self) -> list[_Out]:
        self.steps += 1
        outs: list[_Out] = []
        for request_id in self.made:
            if self.made[request_id] < self.targets[request_id]:
                self.made[request_id] += 1
                n = self.made[request_id]
                outs.append(_Out(request_id, [_Sample(list(range(n)), "x" * n)]))
        return outs


class _NoisyStub(_BatchStub):
    def step(self) -> list[_Out]:
        return [_Out("other", [_Sample([0], "o")]), *super().step()]


def test_run_batch_times_each_request_separately() -> None:
    stub = _BatchStub({"a": 2, "b": 4})
    items: list[tuple[Any, list[int], str, str, int]] = [
        (object(), [1, 2], "a", "ra", 2),
        (object(), [1, 2], "b", "rb", 4),
    ]
    results = run_batch(stub, items)
    assert stub.added == ["a", "b"]
    assert [r.request_id for r in results] == ["ra", "rb"]
    assert [len(r.token_times_ms) for r in results] == [2, 4]
    assert all(r.token_times_ms == sorted(r.token_times_ms) for r in results)
    assert [r.text for r in results] == ["xx", "xxxx"]


def test_run_batch_rejects_wrong_token_count() -> None:
    items: list[tuple[Any, list[int], str, str, int]] = [(object(), [1], "a", "ra", 5)]
    with pytest.raises(RuntimeError, match="expected 5"):
        run_batch(_BatchStub({"a": 3}), items)


def test_run_batch_ignores_other_requests() -> None:
    items: list[tuple[Any, list[int], str, str, int]] = [(object(), [1], "a", "ra", 3)]
    results = run_batch(_NoisyStub({"a": 3}), items)
    assert len(results[0].token_times_ms) == 3


def test_generate_batch_before_prepare_fails() -> None:
    request = GenRequest(request_id="r", prompt_tokens=4, max_new_tokens=2)
    with pytest.raises(RuntimeError, match="prepare"):
        VllmCpuEngine().generate_batch([request])


def test_vllm_cpu_engine_declares_concurrency() -> None:
    assert VllmCpuEngine.supports_concurrency is True


def test_base_generate_batch_defaults_to_generate() -> None:
    engine = FakeEngine()
    engine.prepare(make_config("pytorch", Path("/m"), {}))
    requests = [GenRequest(request_id="r", prompt_tokens=8, max_new_tokens=3)]
    assert FakeEngine.supports_concurrency is False
    assert engine.generate_batch(requests) == engine.generate(requests)