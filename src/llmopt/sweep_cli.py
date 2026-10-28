"""The sweep command: one config at several concurrency levels."""

from pathlib import Path
from typing import Annotated

import typer

from llmopt.profile.factory import make_config, make_engine, parse_settings
from llmopt.profile.sweep import format_sweep, parse_levels, run_sweep, save_sweep
from llmopt.profile.workloads import get_workload
from llmopt.schemas import Status


def sweep_cmd(
    engine: Annotated[
        str, typer.Option("--engine", help="vllm or vllm-rocm (the only ones that can overlap).")
    ],
    model: Annotated[Path, typer.Option("--model", help="Model folder.")],
    levels: Annotated[
        str, typer.Option("--levels", help="Concurrency levels, like 1,2,4,8.")
    ] = "1,2,4,8",
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to save the sweep JSON.")
    ] = Path("sweep.json"),
    settings: Annotated[
        str, typer.Option("--settings", help="Engine settings, like a=1,b=true.")
    ] = "",
    workload_name: Annotated[
        str, typer.Option("--workload", help="Named workload: smoke, chat or long.")
    ] = "smoke",
    runs: Annotated[int, typer.Option("--runs")] = 3,
    precision: Annotated[str, typer.Option("--precision")] = "bf16",
    kv_cache_dtype: Annotated[str, typer.Option("--kv-cache-dtype")] = "auto",
) -> None:
    """Run one config at several concurrency levels and print a table."""
    try:
        config = make_config(engine, model, parse_settings(settings), precision, kv_cache_dtype)
        result = run_sweep(
            lambda: make_engine(engine),
            config,
            get_workload(workload_name),
            parse_levels(levels),
            runs=runs,
        )
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    save_sweep(result, output)
    typer.echo(format_sweep(result))
    typer.echo(f"Saved to {output}")
    if any(row.trial.status is Status.FAILED for row in result.rows):
        raise typer.Exit(code=1)