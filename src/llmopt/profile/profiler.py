"""Run an engine adapter several times and summarize latency, throughput and memory."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from llmopt.engines.base import EngineAdapter, GenRequest, GenResult
from llmopt.profile.memory import PeakMemoryMonitor, process_tree_memory_bytes
from llmopt.schemas import Metrics


class RunMetrics(BaseModel):
    ttft_ms_p50: float
    ttft_ms_p95: float
    tpot_ms: float
    throughput_tps: float
    latency_ms_p50: float
    latency_ms_p95: float


class ProfileResult(BaseModel):
    metrics: Metrics
    runs: list[RunMetrics]
    stable: bool
    spread_limit_pct: float
    memory_method: str


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile q (0 to 100) with linear interpolation between neighbours."""
    if not values:
        raise ValueError("percentile of an empty list")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be between 0 and 100, got {q}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def run_metrics(results: list[GenResult]) -> RunMetrics:
    """Summarize one pass over the requests (requests ran one after another)."""
    if not results:
        raise ValueError("no results to summarize")
    ttfts: list[float] = []
    latencies: list[float] = []
    tpots: list[float] = []
    tokens = 0
    busy_ms = 0.0
    for result in results:
        times = result.token_times_ms
        if len(times) < 2:
            raise ValueError(f"request {result.request_id} made fewer than 2 tokens")
        ttfts.append(times[0])
        latencies.append(times[-1])
        tpots.append((times[-1] - times[0]) / (len(times) - 1))
        tokens += len(times)
        busy_ms += times[-1]
    if busy_ms <= 0:
        raise ValueError("total request time must be greater than 0")
    return RunMetrics(
        ttft_ms_p50=percentile(ttfts, 50),
        ttft_ms_p95=percentile(ttfts, 95),
        tpot_ms=statistics.mean(tpots),
        throughput_tps=tokens / (busy_ms / 1000.0),
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
    )


def spread_pct(values: Sequence[float]) -> float:
    """(max - min) / median, in percent."""
    if not values:
        raise ValueError("spread of an empty list")
    middle = statistics.median(values)
    if middle <= 0:
        raise ValueError("cannot compute spread: the median is not positive")
    return (max(values) - min(values)) / middle * 100.0


def aggregate_runs(runs: list[RunMetrics], peak_memory_bytes: int) -> Metrics:
    """Median of every metric across runs; spread is the worst of TTFT, TPOT, throughput."""
    spread = max(
        spread_pct([r.ttft_ms_p50 for r in runs]),
        spread_pct([r.tpot_ms for r in runs]),
        spread_pct([r.throughput_tps for r in runs]),
    )
    return Metrics(
        ttft_ms_p50=statistics.median(r.ttft_ms_p50 for r in runs),
        ttft_ms_p95=statistics.median(r.ttft_ms_p95 for r in runs),
        tpot_ms=statistics.median(r.tpot_ms for r in runs),
        throughput_tps=statistics.median(r.throughput_tps for r in runs),
        latency_ms_p50=statistics.median(r.latency_ms_p50 for r in runs),
        latency_ms_p95=statistics.median(r.latency_ms_p95 for r in runs),
        peak_memory_gb=peak_memory_bytes / 2**30,
        run_spread_pct=spread,
    )


def profile_engine(
    engine: EngineAdapter,
    requests: list[GenRequest],
    *,
    runs: int = 3,
    warmup: bool = True,
    spread_limit_pct: float = 5.0,
    read_memory_bytes: Callable[[], int] = process_tree_memory_bytes,
) -> ProfileResult:
    """Profile a prepared engine. The caller prepares and tears down the engine.

    Order: one discarded warm-up pass, `runs` timed passes, then one extra pass
    during which memory is sampled (its timings are not used).
    peak_memory_gb holds GiB (2**30 bytes).
    """
    if runs < 2:
        raise ValueError("runs must be at least 2 to measure stability")
    if not requests:
        raise ValueError("no requests to run")
    if warmup:
        engine.generate(requests)
    run_list = [run_metrics(engine.generate(requests)) for _ in range(runs)]
    with PeakMemoryMonitor(read_memory_bytes) as monitor:
        engine.generate(requests)
    metrics = aggregate_runs(run_list, monitor.peak_bytes)
    method = (
        "process-tree PSS (RSS fallback)"
        if read_memory_bytes is process_tree_memory_bytes
        else "custom"
    )
    return ProfileResult(
        metrics=metrics,
        runs=run_list,
        stable=metrics.run_spread_pct <= spread_limit_pct,
        spread_limit_pct=spread_limit_pct,
        memory_method=method,
    )