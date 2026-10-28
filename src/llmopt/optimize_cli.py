"""The optimize command: search settings for the best result under the objective."""

from pathlib import Path
from typing import Annotated

import typer

from llmopt.profile.bench import run_bench
from llmopt.profile.factory import make_config, make_engine, parse_settings
from llmopt.profile.workloads import get_workload
from llmopt.schemas import Config, Objective, TrialResult
from llmopt.tune.optimize import (
    GridCheck,
    OptimizeReport,
    compare_to_grid,
    format_report,
    grid_points,
    run_grid,
    save_optimize,
)
from llmopt.tune.search import default_candidates, run_search


def optimize_cmd(
    engine: Annotated[
        str, typer.Option("--engine", help="pytorch, vllm, pytorch-rocm or vllm-rocm.")
    ],
    model: Annotated[Path, typer.Option("--model", help="Model folder.")],
    objective: Annotated[
        str,
        typer.Option("--objective", help="throughput, ttft, tpot, latency or memory."),
    ] = "throughput",
    max_ttft_p95_ms: Annotated[
        float | None, typer.Option("--max-ttft-p95-ms", help="Hard limit on TTFT p95.")
    ] = None,
    tune: Annotated[
        str, typer.Option("--tune", help="Settings to tune, like a,b. Empty means all.")
    ] = "",
    settings: Annotated[
        str, typer.Option("--settings", help="Starting settings, like a=1,b=true.")
    ] = "",
    workload_name: Annotated[
        str, typer.Option("--workload", help="Named workload: smoke, chat or long.")
    ] = "smoke",
    runs: Annotated[int, typer.Option("--runs")] = 3,
    max_trials: Annotated[int, typer.Option("--max-trials")] = 40,
    check_grid: Annotated[
        bool, typer.Option("--check-grid", help="Also try every combination and compare.")
    ] = False,
    max_grid: Annotated[int, typer.Option("--max-grid")] = 64,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to save the report JSON.")
    ] = Path("optimize.json"),
    precision: Annotated[str, typer.Option("--precision")] = "bf16",
    kv_cache_dtype: Annotated[str, typer.Option("--kv-cache-dtype")] = "auto",
) -> None:
    """Search settings for the best result, and optionally check it against a grid."""
    try:
        config = make_config(engine, model, parse_settings(settings), precision, kv_cache_dtype)
        names = [name.strip() for name in tune.split(",") if name.strip()]
        candidates = default_candidates(names or None)
        target = Objective(optimize=objective, max_ttft_p95_ms=max_ttft_p95_ms)
        workload = get_workload(workload_name)
        # Check the grid size first, so a too-big grid fails before any run.
        points = grid_points(config.settings, candidates, max_grid) if check_grid else []
        cache: dict[str, TrialResult] = {}

        def evaluate(trial_config: Config) -> TrialResult:
            key = trial_config.key()
            if key not in cache:
                cache[key] = run_bench(make_engine(engine), trial_config, workload, runs=runs)
            return cache[key]

        search = run_search(evaluate, config, target, candidates, max_trials=max_trials)
        grid: GridCheck | None = None
        if check_grid:
            grid = compare_to_grid(search, run_grid(evaluate, config, target, points), target)
    except (ValueError, RuntimeError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    report = OptimizeReport(
        config=config, workload=workload.name, objective=target, search=search, grid=grid
    )
    save_optimize(report, output)
    typer.echo(format_report(report))
    typer.echo(f"Saved to {output}")
    if grid is not None and not grid.within_limit:
        raise typer.Exit(code=1)