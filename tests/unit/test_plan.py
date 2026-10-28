from llmopt.engines.vllm_rocm import SUPPORTED_PRECISIONS
from llmopt.profile.factory import ENGINE_NAMES
from llmopt.schemas import Engine, Precision
from llmopt.tune.plan import ADAPTERS, build_plan, format_plan

PAIRS = [
    (Engine.VLLM, Precision.BF16),
    (Engine.VLLM, Precision.FP8),
    (Engine.PYTORCH, Precision.BF16),
    (Engine.LLAMACPP, Precision.GGUF),
]


def test_adapters_per_pair() -> None:
    plan = build_plan(PAIRS)
    assert plan[0].adapters == ["vllm", "vllm-rocm"]
    assert plan[1].adapters == ["vllm-rocm"]
    assert plan[2].adapters == ["pytorch", "pytorch-rocm"]
    assert plan[3].adapters == []


def test_has_adapter_flag() -> None:
    plan = build_plan(PAIRS)
    assert plan[1].has_adapter
    assert not plan[3].has_adapter


def test_order_and_count_are_kept() -> None:
    plan = build_plan(PAIRS)
    assert [(e.engine, e.precision) for e in plan] == PAIRS


def test_empty_plan_text() -> None:
    text = format_plan("gfx000", [])
    assert "Candidates for gfx000: 0" in text
    assert "0 of 0 have an adapter" in text


def test_format_text_for_gfx942() -> None:
    text = format_plan("gfx942", build_plan(PAIRS))
    assert "Candidates for gfx942: 4" in text
    assert "3 of 4 have an adapter" in text
    assert "2 of 4 can run on the bench PC" in text
    assert "vllm-rocm (draft, not tested on a GPU)" in text
    assert "no adapter yet" in text


def test_every_adapter_name_is_a_bench_engine() -> None:
    for names in ADAPTERS.values():
        for name in names:
            assert name in ENGINE_NAMES


def test_vllm_rocm_precisions_match_adapter_table() -> None:
    listed = {
        precision
        for (engine, precision), names in ADAPTERS.items()
        if engine is Engine.VLLM and "vllm-rocm" in names
    }
    assert listed == set(SUPPORTED_PRECISIONS)