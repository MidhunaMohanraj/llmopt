"""Fixed-grid check for the settings search, and the saved optimize report."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from llmopt.schemas import Config, Objective, SettingValue, TrialResult
from llmopt.tune.search import METRICS, Evaluated, SearchResult, feasibility_problem

GRID_LIMIT_PCT = 5.0


class GridCheck(BaseModel):
    points: int
    grid_best: Evaluated | None
    # How far the tuner's best is behind the grid's best; negative means it is ahead.
    gap_pct: float | None
    within_limit: bool
    limit_pct: float


class OptimizeReport(BaseModel):
    config: Config
    workload: str
    objective: Objective
    search: SearchResult
    grid: GridCheck | None = None


def grid_points(
    base: dict[str, SettingValue],
    candidates: dict[str, list[SettingValue]],
    max_points: int,
) -> list[dict[str, SettingValue]]:
    """Every combination of the candidate values. Each axis starts with the value in
    base (or 'not set'), so the starting config is one of the points."""
    if max_points < 1:
        raise ValueError(f"max_grid must be at least 1, got {max_points}")
    names = list(candidates)
    axes: list[list[SettingValue | None]] = []
    for name in names:
        start = base.get(name)
        axes.append([start, *[value for value in candidates[name] if value != start]])
    size = math.prod(len(axis) for axis in axes)
    if size > max_points:
        raise ValueError(
            f"the grid has {size} points, more than the limit of {max_points}; "
            "tune fewer settings or raise --max-grid"
        )
    points: list[dict[str, SettingValue]] = []
    for combo in itertools.product(*axes):
        settings = dict(base)
        for name, value in zip(names, combo, strict=True):
            if value is None:
                settings.pop(name, None)
            else:
                settings[name] = value
        points.append(settings)
    return points


def run_grid(
    evaluate: Callable[[Config], TrialResult],
    config: Config,
    objective: Objective,
    points: list[dict[str, SettingValue]],
) -> list[Evaluated]:
    """Measure every grid point."""
    results: list[Evaluated] = []
    for settings in points:
        trial = evaluate(config.model_copy(update={"settings": dict(settings)}))
        results.append(
            Evaluated(
                settings=dict(settings),
                trial=trial,
                problem=feasibility_problem(trial, objective),
            )
        )
    return results


def _value(item: Evaluated, field: str) -> float:
    metrics = item.trial.metrics
    if metrics is None:
        raise ValueError("trial has no metrics")
    return float(getattr(metrics, field))


def compare_to_grid(
    search: SearchResult,
    grid: list[Evaluated],
    objective: Objective,
    limit_pct: float = GRID_LIMIT_PCT,
) -> GridCheck:
    """Is the tuner's best within limit_pct of the best usable grid point?"""
    field, higher = METRICS[objective.optimize]
    usable = [item for item in grid if item.feasible]
    if not usable:
        return GridCheck(
            points=len(grid), grid_best=None, gap_pct=None, within_limit=True, limit_pct=limit_pct
        )
    best = (
        max(usable, key=lambda item: _value(item, field))
        if higher
        else min(usable, key=lambda item: _value(item, field))
    )
    if not search.best.feasible:
        return GridCheck(
            points=len(grid), grid_best=best, gap_pct=None, within_limit=False, limit_pct=limit_pct
        )
    grid_value = _value(best, field)
    tuner_value = _value(search.best, field)
    if grid_value <= 0:
        raise ValueError("the best grid value must be positive")
    if higher:
        gap = (grid_value - tuner_value) / grid_value * 100
    else:
        gap = (tuner_value - grid_value) / grid_value * 100
    return GridCheck(
        points=len(grid),
        grid_best=best,
        gap_pct=gap,
        within_limit=gap <= limit_pct,
        limit_pct=limit_pct,
    )


def _show(settings: dict[str, SettingValue]) -> str:
    return json.dumps(settings, sort_keys=True)


def format_report(report: OptimizeReport) -> str:
    name = report.objective.optimize
    field, higher = METRICS[name]
    search = report.search
    base_value = _value(search.baseline, field)
    direction = "higher" if higher else "lower"
    lines = [
        f"objective: {name} ({field}, {direction} is better)",
        f"baseline: {base_value:.2f}  settings {_show(search.baseline.settings)}",
    ]
    best = search.best
    if best.feasible:
        best_value = _value(best, field)
        change = ""
        if base_value > 0:
            gain = (best_value - base_value) if higher else (base_value - best_value)
            change = f"  ({gain / base_value * 100:+.1f}%)"
        lines.append(f"best:     {best_value:.2f}  settings {_show(best.settings)}{change}")
    else:
        lines.append(f"best:     no usable result ({best.problem})")
    stopped = " (stopped at the trial limit)" if search.stopped_early else ""
    lines.append(f"trials measured: {len(search.evaluated)}{stopped}")
    if search.front:
        lines.append("trade-off front (throughput, TTFT p95):")
        for item in search.front:
            metrics = item.trial.metrics
            if metrics is not None:
                lines.append(
                    f"  {metrics.throughput_tps:.1f} tok/s  {metrics.ttft_ms_p95:.1f} ms  "
                    f"{_show(item.settings)}"
                )
    grid = report.grid
    if grid is not None:
        if grid.grid_best is None:
            lines.append(f"grid check: no usable result in {grid.points} grid point(s)")
        elif grid.gap_pct is None:
            lines.append(
                f"grid check: the tuner found no usable result, but the grid did "
                f"({_value(grid.grid_best, field):.2f})"
            )
        else:
            verdict = "OK" if grid.within_limit else "OUTSIDE THE LIMIT"
            lines.append(
                f"grid check: {grid.points} points, best {_value(grid.grid_best, field):.2f}; "
                f"gap {grid.gap_pct:+.1f}% (positive = tuner behind), "
                f"limit {grid.limit_pct:g}%: {verdict}"
            )
    return "\n".join(lines)


def save_optimize(report: OptimizeReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n")