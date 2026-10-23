from __future__ import annotations

import pytest

from llmopt.engines.base import (
    EngineAdapter,
    GenRequest,
    GenResult,
    SupportResult,
)
from llmopt.profile.bench import build_requests, run_bench
from llmopt.schemas import (
    Config,
    Engine,
    HardwareProfile,
    Precision,
    Status,
    Workload,
)

CONFIG = Config(model="m", engine=Engine.PYTORCH, precision=Precision.BF16)
WORKLOAD = Workload(name="t", prompt_tokens=8, output_tokens=4, num_requests=3)


def fake_memory() -> int:
    return 2**30


class ScriptedEngine(EngineAdapter):
    """Test double with made-up timings. Never use its numbers as a benchmark."""

    name = "scripted"

    def __init__(
        self,
        growing: bool = False,
        fail_prepare: bool = False,
        fail_generate: bool = False,
        fail_teardown: bool = False,
    ) -> None:
        self.growing = growing
        self.fail_prepare = fail_prepare
        self.fail_generate = fail_generate
        self.fail_teardown = fail_teardown
        self.calls = 0
        self.prepared = False
        self.torn_down = False
        self.request_counts: list[int] = []

    def supports(
        self, hardware: HardwareProfile, model: str, precision: Precision
    ) -> SupportResult:
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        if self.fail_prepare:
            raise RuntimeError("no model")
        self.prepared = True

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        if self.fail_generate:
            raise RuntimeError("engine crashed")
        self.calls += 1
        self.request_counts.append(len(requests))
        scale = float(self.calls) if self.growing else 1.0
        return [
            GenResult(
                request_id=request.request_id,
                token_times_ms=[10.0 * scale * (k + 1) for k in range(4)],
            )
            for request in requests
        ]

    def teardown(self) -> None:
        self.torn_down = True
        if self.fail_teardown:
            raise RuntimeError("stuck")


def test_build_requests_matches_workload() -> None:
    requests = build_requests(WORKLOAD)
    assert len(requests) == 3
    assert requests[0].prompt_tokens == 8
    assert requests[0].max_new_tokens == 4
    assert len({request.request_id for request in requests}) == 3


def test_success_gives_ok_trial_with_metrics() -> None:
    engine = ScriptedEngine()
    trial = run_bench(engine, CONFIG, WORKLOAD, read_memory_bytes=fake_memory)
    assert trial.status is Status.OK
    assert trial.reason is None
    assert trial.config == CONFIG
    assert trial.metrics is not None
    assert trial.metrics.ttft_ms_p50 == pytest.approx(10.0)
    assert engine.torn_down


def test_engine_runs_warmup_three_runs_and_memory_pass() -> None:
    engine = ScriptedEngine()
    run_bench(engine, CONFIG, WORKLOAD, read_memory_bytes=fake_memory)
    assert engine.request_counts == [3, 3, 3, 3, 3]


def test_prepare_failure_gives_failed_trial_and_still_tears_down() -> None:
    engine = ScriptedEngine(fail_prepare=True)
    trial = run_bench(engine, CONFIG, WORKLOAD, read_memory_bytes=fake_memory)
    assert trial.status is Status.FAILED
    assert trial.metrics is None
    assert trial.reason is not None
    assert "no model" in trial.reason
    assert engine.torn_down


def test_generate_failure_gives_failed_trial() -> None:
    engine = ScriptedEngine(fail_generate=True)
    trial = run_bench(engine, CONFIG, WORKLOAD, read_memory_bytes=fake_memory)
    assert trial.status is Status.FAILED
    assert trial.reason is not None
    assert "engine crashed" in trial.reason
    assert engine.torn_down


def test_unstable_runs_are_flagged_but_keep_their_metrics() -> None:
    engine = ScriptedEngine(growing=True)
    trial = run_bench(engine, CONFIG, WORKLOAD, read_memory_bytes=fake_memory)
    assert trial.status is Status.OK
    assert trial.metrics is not None
    assert trial.reason is not None
    assert trial.reason.startswith("unstable")


def test_teardown_failure_is_recorded_without_losing_the_result() -> None:
    engine = ScriptedEngine(fail_teardown=True)
    trial = run_bench(engine, CONFIG, WORKLOAD, read_memory_bytes=fake_memory)
    assert trial.status is Status.OK
    assert trial.metrics is not None
    assert trial.reason is not None
    assert "teardown failed" in trial.reason


def test_concurrency_above_one_is_rejected_before_starting() -> None:
    engine = ScriptedEngine()
    workload = Workload(name="t", prompt_tokens=8, output_tokens=4, concurrency=2)
    with pytest.raises(ValueError):
        run_bench(engine, CONFIG, workload)
    assert not engine.prepared


def test_too_few_runs_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_bench(ScriptedEngine(), CONFIG, WORKLOAD, runs=1)