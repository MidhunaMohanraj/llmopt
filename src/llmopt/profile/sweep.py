"""Run one config at several concurrency levels and collect the trials."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from llmopt.engines.base import EngineAdapter
from llmopt.profile.bench import run_bench
from llmopt.profile.memory import process_tree_memory_bytes
from llmopt.schemas import Config, Status, TrialResult, Workload


class SweepRow(BaseModel):
    concurrency: int
    num_requests: int
    trial: TrialResult


class SweepResult(BaseModel):
    config: Config
    workload: str
    rows: list[SweepRow]


def parse_levels(text: str) -> list[int]:
    """Parse '1,2,4,8' into increasing whole numbers."""
    levels: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            raise ValueError(f"concurrency level must be a whole number, got {item!r}") from None
        if value < 1:
            raise ValueError(f"concurrency level must be at least 1, got {value}")
        levels.append(value)
    if not levels:
        raise ValueError("no concurrency levels given")
    if levels != sorted(set(levels)):
        raise ValueError("concurrency levels must be increasing with no repeats")
    return levels


def requests_for_level(base: int, level: int) -> int:
    """At least `base` requests and at least one full wave, in whole waves."""
    wanted = max(base, level)
    return -(-wanted // level) * level


def run_sweep(
    make_engine: Callable[[], EngineAdapter],
    config: Config,
    workload: Workload,
    levels: list[int],
    *,
    runs: int = 3,
    read_memory_bytes: Callable[[], int] = process_tree_memory_bytes,
    clock: Callable[[], float] = time.perf_counter,
) -> SweepResult:
    """Bench the config once per level. Stops after the first failed level."""
    if any(level > 1 for level in levels) and not make_engine().supports_concurrency:
        raise ValueError(
            "this engine runs requests one at a time, so only concurrency 1 can be swept"
        )
    rows: list[SweepRow] = []
    for level in levels:
        sized = workload.model_copy(
            update={
                "concurrency": level,
                "num_requests": requests_for_level(workload.num_requests, level),
            }
        )
        trial = run_bench(
            make_engine(),
            config,
            sized,
            runs=runs,
            read_memory_bytes=read_memory_bytes,
            clock=clock,
        )
        rows.append(SweepRow(concurrency=level, num_requests=sized.num_requests, trial=trial))
        if trial.status is Status.FAILED:
            break
    return SweepResult(config=config, workload=workload.name, rows=rows)


def format_sweep(result: SweepResult) -> str:
    head = (
        f"{'conc':>5} {'reqs':>5} {'TTFT p50 ms':>12} {'TPOT ms':>9} "
        f"{'tok/s':>9} {'GiB':>7} {'spread %':>9}  status"
    )
    lines = [head]
    for row in result.rows:
        status = row.trial.status.value
        if row.trial.reason:
            status += f" ({row.trial.reason})"
        metrics = row.trial.metrics
        if metrics is None:
            cells = f"{'-':>12} {'-':>9} {'-':>9} {'-':>7} {'-':>9}"
        else:
            cells = (
                f"{metrics.ttft_ms_p50:>12.1f} {metrics.tpot_ms:>9.1f} "
                f"{metrics.throughput_tps:>9.1f} {metrics.peak_memory_gb:>7.2f} "
                f"{metrics.run_spread_pct:>9.1f}"
            )
        lines.append(f"{row.concurrency:>5} {row.num_requests:>5} {cells}  {status}")
    return "\n".join(lines)


def save_sweep(result: SweepResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n")