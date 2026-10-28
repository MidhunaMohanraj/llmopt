from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.profile.bench import run_bench
from llmopt.profile.factory import make_config
from llmopt.profile.profiler import check_concurrency, profile_engine, run_metrics
from llmopt.schemas import Config, HardwareProfile, Precision, Status, Workload

ONE_GIB = 2**30
TIMES = [250.0, 500.0, 750.0, 1000.0]


class FakeClock:
    """A clock the test engine moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class WaveEngine(EngineAdapter):
    """Pretend engine, not a real one. Every request takes 1000 ms. Requests in one
    call overlap if concurrent is True (the call takes 1 s), otherwise they run in
    turn (the call takes 1 s per request)."""

    name = "wave"

    def __init__(self, clock: FakeClock, concurrent: bool = True) -> None:
        self.clock = clock
        self.supports_concurrency = concurrent
        self.batch_sizes: list[int] = []

    def supports(
        self, hardware: HardwareProfile, model: str, precision: Precision
    ) -> SupportResult:
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        return None

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        self.batch_sizes.append(len(requests))
        self.clock.now += 1.0 if self.supports_concurrency else float(len(requests))
        return [
            GenResult(request_id=r.request_id, token_times_ms=list(TIMES))
            for r in requests
        ]

    def teardown(self) -> None:
        return None


def make_requests(count: int) -> list[GenRequest]:
    return [
        GenRequest(request_id=f"r{i}", prompt_tokens=32, max_new_tokens=4)
        for i in range(count)
    ]


def test_run_metrics_uses_wall_time_when_given() -> None:
    results = [GenResult(request_id=f"r{i}", token_times_ms=list(TIMES)) for i in range(4)]
    assert run_metrics(results).throughput_tps == pytest.approx(4.0)
    assert run_metrics(results, wall_ms=1000.0).throughput_tps == pytest.approx(16.0)


def test_run_metrics_rejects_bad_wall_time() -> None:
    results = [GenResult(request_id="r", token_times_ms=list(TIMES))]
    with pytest.raises(ValueError, match="wall_ms"):
        run_metrics(results, wall_ms=0.0)


def test_profile_engine_sends_waves_and_uses_wall_time() -> None:
    clock = FakeClock()
    engine = WaveEngine(clock)
    result = profile_engine(
        engine,
        make_requests(8),
        runs=2,
        read_memory_bytes=lambda: ONE_GIB,
        concurrency=4,
        clock=clock,
    )
    # warm-up, 2 timed runs and the memory pass: 4 passes of 2 waves each
    assert engine.batch_sizes == [4] * 8
    assert result.metrics.throughput_tps == pytest.approx(16.0)
    assert result.metrics.tpot_ms == pytest.approx(250.0)
    assert result.stable


def test_concurrency_one_keeps_the_sequential_path() -> None:
    clock = FakeClock()
    engine = WaveEngine(clock, concurrent=False)
    result = profile_engine(
        engine, make_requests(2), runs=2, read_memory_bytes=lambda: ONE_GIB, clock=clock
    )
    assert engine.batch_sizes[0] == 2
    assert result.metrics.throughput_tps == pytest.approx(4.0)


def test_check_concurrency_rejects_engine_without_support() -> None:
    engine = WaveEngine(FakeClock(), concurrent=False)
    with pytest.raises(ValueError, match="concurrency must be 1"):
        check_concurrency(engine, 4, 8)


def test_check_concurrency_needs_full_waves() -> None:
    engine = WaveEngine(FakeClock())
    with pytest.raises(ValueError, match="multiple"):
        check_concurrency(engine, 4, 6)


def test_run_bench_with_concurrency() -> None:
    clock = FakeClock()
    workload = Workload(
        name="w", prompt_tokens=32, output_tokens=4, concurrency=4, num_requests=8
    )
    trial = run_bench(
        WaveEngine(clock),
        make_config("pytorch", Path("/m"), {}),
        workload,
        runs=2,
        read_memory_bytes=lambda: ONE_GIB,
        clock=clock,
    )
    assert trial.status is Status.OK
    assert trial.metrics is not None
    assert trial.metrics.throughput_tps == pytest.approx(16.0)


def test_run_bench_rejects_concurrency_for_sequential_engine() -> None:
    workload = Workload(
        name="w", prompt_tokens=32, output_tokens=4, concurrency=4, num_requests=8
    )
    with pytest.raises(ValueError, match="concurrency must be 1"):
        run_bench(
            WaveEngine(FakeClock(), concurrent=False),
            make_config("pytorch", Path("/m"), {}),
            workload,
        )