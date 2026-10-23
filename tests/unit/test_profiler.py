import time
from pathlib import Path

import pytest

from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.profile.memory import PeakMemoryMonitor, process_tree_memory_bytes
from llmopt.profile.profiler import percentile, profile_engine, run_metrics, spread_pct
from llmopt.schemas import Config, HardwareProfile, Precision


class ScriptedEngine(EngineAdapter):
    """Returns the given token timings for each call, in order (not a real engine)."""

    name = "scripted"

    def __init__(self, scripts: list[list[float]]) -> None:
        self._scripts = scripts
        self.calls = 0

    def supports(
        self, hardware: HardwareProfile, model: str, precision: Precision
    ) -> SupportResult:
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        return None

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        times = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1
        return [GenResult(request_id=r.request_id, token_times_ms=list(times)) for r in requests]

    def teardown(self) -> None:
        return None


REQUESTS = [GenRequest(request_id="r1", prompt_tokens=32, max_new_tokens=3)]
THREE_GIB = 3 * 2**30


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert percentile([10], 95) == 10


def test_percentile_errors() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="between"):
        percentile([1], 101)


def test_run_metrics_single_request() -> None:
    result = GenResult(request_id="a", token_times_ms=[100, 130, 160, 190])
    metrics = run_metrics([result])
    assert metrics.ttft_ms_p50 == 100
    assert metrics.tpot_ms == 30
    assert metrics.latency_ms_p50 == 190
    assert metrics.throughput_tps == pytest.approx(4 / 0.19)


def test_run_metrics_two_requests() -> None:
    first = GenResult(request_id="a", token_times_ms=[100, 130])
    second = GenResult(request_id="b", token_times_ms=[200, 260])
    metrics = run_metrics([first, second])
    assert metrics.ttft_ms_p50 == 150
    assert metrics.ttft_ms_p95 == pytest.approx(195)
    assert metrics.tpot_ms == 45
    assert metrics.latency_ms_p95 == pytest.approx(253.5)
    assert metrics.throughput_tps == pytest.approx(4 / 0.39)


def test_run_metrics_needs_two_tokens() -> None:
    with pytest.raises(ValueError, match="fewer than 2"):
        run_metrics([GenResult(request_id="a", token_times_ms=[100])])


def test_spread_pct() -> None:
    assert spread_pct([98, 100, 102]) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="median"):
        spread_pct([0, 0])


def test_profile_discards_warmup_and_takes_medians() -> None:
    engine = ScriptedEngine(
        [
            [900, 1800, 2700],  # warm-up, must not count
            [100, 130, 160],
            [102, 132, 162],
            [98, 128, 158],
            [100, 130, 160],  # memory pass, timings unused
        ]
    )
    result = profile_engine(engine, REQUESTS, read_memory_bytes=lambda: THREE_GIB)
    assert engine.calls == 5
    assert result.metrics.ttft_ms_p50 == 100
    assert result.metrics.tpot_ms == 30
    assert result.metrics.run_spread_pct == pytest.approx(4.0)
    assert result.metrics.peak_memory_gb == pytest.approx(3.0)
    assert result.stable is True
    assert len(result.runs) == 3
    assert result.memory_method == "custom"


def test_profile_flags_unstable_runs() -> None:
    engine = ScriptedEngine([[100, 130], [150, 180], [100, 130], [100, 130]])
    result = profile_engine(engine, REQUESTS, warmup=False, read_memory_bytes=lambda: 1)
    assert result.metrics.run_spread_pct == pytest.approx(50.0)
    assert result.stable is False


def test_profile_input_errors() -> None:
    engine = ScriptedEngine([[100, 130]])
    with pytest.raises(ValueError, match="at least 2"):
        profile_engine(engine, REQUESTS, runs=1)
    with pytest.raises(ValueError, match="no requests"):
        profile_engine(engine, [])


def test_peak_memory_monitor_keeps_the_maximum() -> None:
    values = iter([1, 9, 2])

    def read() -> int:
        return next(values, 2)

    with PeakMemoryMonitor(read, interval_s=0.01) as monitor:
        time.sleep(0.05)
    assert monitor.peak_bytes == 9


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="needs Linux /proc")
def test_process_tree_memory_is_positive() -> None:
    assert process_tree_memory_bytes() > 0