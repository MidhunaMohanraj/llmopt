import pytest

from llmopt.engines.base import GenRequest
from llmopt.engines.fake import FakeBehavior, FakeEngine
from llmopt.schemas import Config, Engine, HardwareProfile, Precision


def hardware() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="AMD Instinct MI300X",
        gfx_target="gfx942",
        compute_units=304,
        hbm_gb=192,
        rocm_version="unknown",
        driver_version="unknown",
        cpu_model="AMD EPYC",
        cpu_cores=96,
    )


def config(precision: Precision) -> Config:
    return Config(model="Qwen/Qwen3-8B", engine=Engine.VLLM, precision=precision)


def requests() -> list[GenRequest]:
    return [GenRequest(request_id="r1", prompt_tokens=1000, max_new_tokens=10)]


def test_supports_known_precision() -> None:
    result = FakeEngine().supports(hardware(), "Qwen/Qwen3-8B", Precision.FP8)
    assert result.ok


def test_supports_gives_a_reason_when_it_says_no() -> None:
    result = FakeEngine().supports(hardware(), "Qwen/Qwen3-8B", Precision.MXFP4)
    assert not result.ok
    assert "mxfp4" in result.reason


def test_generate_before_prepare_fails() -> None:
    with pytest.raises(RuntimeError):
        FakeEngine().generate(requests())


def test_prepare_rejects_unsupported_precision() -> None:
    with pytest.raises(ValueError):
        FakeEngine().prepare(config(Precision.MXFP4))


def test_teardown_resets_the_engine() -> None:
    engine = FakeEngine()
    engine.prepare(config(Precision.BF16))
    engine.teardown()
    with pytest.raises(RuntimeError):
        engine.generate(requests())


def test_one_timestamp_per_token_and_increasing() -> None:
    engine = FakeEngine()
    engine.prepare(config(Precision.BF16))
    result = engine.generate(requests())[0]
    assert len(result.token_times_ms) == 10
    assert result.token_times_ms == sorted(result.token_times_ms)
    assert result.ttft_ms == result.token_times_ms[0]


def test_same_input_gives_same_output() -> None:
    engine = FakeEngine()
    engine.prepare(config(Precision.BF16))
    assert engine.generate(requests()) == engine.generate(requests())


def test_fp8_is_faster_than_bf16() -> None:
    engine = FakeEngine()
    engine.prepare(config(Precision.BF16))
    bf16 = engine.generate(requests())[0]
    engine.prepare(config(Precision.FP8))
    fp8 = engine.generate(requests())[0]
    assert fp8.ttft_ms < bf16.ttft_ms
    assert fp8.total_ms < bf16.total_ms


def test_behavior_can_be_changed() -> None:
    engine = FakeEngine(FakeBehavior(ttft_base_ms=500.0, prefill_ms_per_token=0.0))
    engine.prepare(config(Precision.BF16))
    assert engine.generate(requests())[0].ttft_ms == 500.0
