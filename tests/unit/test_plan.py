from __future__ import annotations

from llmopt.schemas import Engine, Precision
from llmopt.tune.plan import build_plan, format_plan

PAIRS = [
    (Engine.VLLM, Precision.BF16),
    (Engine.VLLM, Precision.FP8),
    (Engine.PYTORCH, Precision.BF16),
    (Engine.LLAMACPP, Precision.GGUF),
]


def test_only_bf16_pytorch_and_vllm_have_adapters() -> None:
    entries = build_plan(PAIRS)
    flags = {(e.engine, e.precision): e.has_adapter for e in entries}
    assert flags[(Engine.VLLM, Precision.BF16)] is True
    assert flags[(Engine.PYTORCH, Precision.BF16)] is True
    assert flags[(Engine.VLLM, Precision.FP8)] is False
    assert flags[(Engine.LLAMACPP, Precision.GGUF)] is False


def test_plan_keeps_the_order_and_count() -> None:
    entries = build_plan(PAIRS)
    assert [(e.engine, e.precision) for e in entries] == PAIRS


def test_empty_plan() -> None:
    text = format_plan("gfx000", build_plan([]))
    assert "Candidates for gfx000: 0" in text
    assert "0 of 0" in text


def test_format_lists_every_pair_and_counts_runnable() -> None:
    text = format_plan("gfx942", build_plan(PAIRS))
    assert "Candidates for gfx942: 4" in text
    assert "2 of 4 can be run" in text
    assert "no adapter yet" in text
    assert "fp8" in text