import pytest

from llmopt.precision.selector import (
    candidate_pairs,
    load_compat,
    load_gpus,
    supported_precisions,
)
from llmopt.schemas import Engine, Precision


def test_mi300x_never_offers_mxfp4() -> None:
    assert Precision.MXFP4 not in supported_precisions("gfx942")


def test_mi300x_pairs_never_include_mxfp4() -> None:
    assert all(p != Precision.MXFP4 for _, p in candidate_pairs("gfx942"))


def test_mi350_series_offers_mxfp4() -> None:
    assert Precision.MXFP4 in supported_precisions("gfx950")


def test_unknown_gpu_raises() -> None:
    with pytest.raises(ValueError):
        supported_precisions("gfx000")


def test_every_offered_precision_runs_on_some_engine() -> None:
    # From the brief: each option runs on at least one engine.
    compat = load_compat()
    runnable = {p for plist in compat.values() for p in plist}
    for gfx, info in load_gpus().items():
        for precision in info.precisions:
            assert precision in runnable, f"{precision.value} on {gfx} has no engine"


def test_known_pairs_on_mi300x() -> None:
    pairs = set(candidate_pairs("gfx942"))
    assert (Engine.VLLM, Precision.BF16) in pairs
    assert (Engine.VLLM, Precision.FP8) in pairs
    assert (Engine.SGLANG, Precision.FP8) in pairs
    assert (Engine.LLAMACPP, Precision.GGUF) in pairs


def test_pytorch_is_bf16_only_and_llamacpp_is_gguf_only() -> None:
    pairs = candidate_pairs("gfx942")
    assert {p for e, p in pairs if e == Engine.PYTORCH} == {Precision.BF16}
    assert {p for e, p in pairs if e == Engine.LLAMACPP} == {Precision.GGUF}
