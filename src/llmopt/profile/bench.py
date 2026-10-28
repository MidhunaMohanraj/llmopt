"""Run one config through an engine adapter and the profiler, and return a TrialResult."""

from __future__ import annotations

import time
from collections.abc import Callable

from llmopt.engines.base import EngineAdapter, GenRequest
from llmopt.profile.memory import process_tree_memory_bytes
from llmopt.profile.profiler import check_concurrency, profile_engine
from llmopt.schemas import Config, Status, TrialResult, Workload


def build_requests(workload: Workload) -> list[GenRequest]:
    return [
        GenRequest(
            request_id=f"{workload.name}-{index}",
            prompt_tokens=workload.prompt_tokens,
            max_new_tokens=workload.output_tokens,
        )
        for index in range(workload.num_requests)
    ]


def run_bench(
    engine: EngineAdapter,
    config: Config,
    workload: Workload,
    *,
    runs: int = 3,
    warmup: bool = True,
    spread_limit_pct: float = 5.0,
    read_memory_bytes: Callable[[], int] = process_tree_memory_bytes,
    clock: Callable[[], float] = time.perf_counter,
) -> TrialResult:
    """Prepare the engine, profile it, tear it down. A crash gives a FAILED trial."""
    check_concurrency(engine, workload.concurrency, workload.num_requests)
    if runs < 2:
        raise ValueError(f"runs must be at least 2 to measure stability, got {runs}")
    requests = build_requests(workload)
    try:
        engine.prepare(config)
        result = profile_engine(
            engine,
            requests,
            runs=runs,
            warmup=warmup,
            spread_limit_pct=spread_limit_pct,
            read_memory_bytes=read_memory_bytes,
            concurrency=workload.concurrency,
            clock=clock,
        )
    except Exception as error:  # noqa: BLE001
        trial = TrialResult(
            config=config,
            status=Status.FAILED,
            reason=f"{type(error).__name__}: {error}",
        )
    else:
        reason = None
        if not result.stable:
            reason = (
                f"unstable: run spread {result.metrics.run_spread_pct:.1f}% "
                f"is over the {spread_limit_pct:.1f}% limit"
            )
        trial = TrialResult(
            config=config, status=Status.OK, reason=reason, metrics=result.metrics
        )
    try:
        engine.teardown()
    except Exception as error:  # noqa: BLE001
        note = f"teardown failed: {type(error).__name__}: {error}"
        combined = f"{trial.reason}; {note}" if trial.reason else note
        trial = trial.model_copy(update={"reason": combined})
    return trial