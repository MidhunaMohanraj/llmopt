from pathlib import Path
from typing import Annotated

import typer

from llmopt.hardware.host import detect_host
from llmopt.precision.selector import candidate_pairs
from llmopt.profile.bench import run_bench
from llmopt.profile.factory import make_config, make_engine, parse_settings
from llmopt.report.render import write_report
from llmopt.report.store import load_report, load_trials, save_trial
from llmopt.schemas import Status, TrialResult, Workload
from llmopt.tune.plan import build_plan, format_plan

app = typer.Typer(help="llmopt: find the best way to serve an LLM on an AMD GPU.")


@app.callback()
def main() -> None:
    """llmopt command line."""


@app.command()
def detect(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to save the JSON.")
    ] = Path("hardware_profile.json"),
) -> None:
    """Print and save the host hardware profile."""
    profile = detect_host()
    text = profile.model_dump_json(indent=2)
    output.write_text(text + "\n")
    typer.echo(text)
    typer.echo(f"Saved to {output}")


@app.command("report")
def report_cmd(
    run: Annotated[
        Path,
        typer.Option(
            "--run", help="Report JSON with hardware, workload and objective."
        ),
    ],
    results: Annotated[
        Path, typer.Option("--results", help="Folder of trial JSON files.")
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to save the HTML.")
    ] = Path("report.html"),
    notice: Annotated[
        str, typer.Option("--notice", help="Banner text, e.g. 'smoke run'.")
    ] = "",
) -> None:
    """Build an HTML report from a run file and a folder of trial results."""
    try:
        base = load_report(run)
        trials = load_trials(results)
    except FileNotFoundError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    full = base.model_copy(update={"trials": trials})
    write_report(full, output, notice or None)
    typer.echo(f"Wrote {output} with {len(trials)} trial(s)")


def _trial_summary(trial: TrialResult) -> str:
    text = f"status: {trial.status.value}"
    if trial.reason:
        text += f" ({trial.reason})"
    metrics = trial.metrics
    if metrics is not None:
        text += (
            f"\nTTFT p50 {metrics.ttft_ms_p50:.1f} ms, TPOT {metrics.tpot_ms:.1f} ms, "
            f"throughput {metrics.throughput_tps:.1f} tok/s, "
            f"peak memory {metrics.peak_memory_gb:.2f} GiB, "
            f"spread {metrics.run_spread_pct:.1f}%"
        )
    return text


@app.command("bench")
def bench_cmd(
    engine: Annotated[str, typer.Option("--engine", help="pytorch or vllm.")],
    model: Annotated[Path, typer.Option("--model", help="Model folder.")],
    results: Annotated[
        Path, typer.Option("--results", help="Folder to save the trial JSON in.")
    ] = Path("results"),
    settings: Annotated[
        str, typer.Option("--settings", help="Engine settings, like a=1,b=true.")
    ] = "",
    prompt_tokens: Annotated[int, typer.Option("--prompt-tokens")] = 32,
    output_tokens: Annotated[int, typer.Option("--output-tokens")] = 16,
    num_requests: Annotated[int, typer.Option("--num-requests")] = 8,
    runs: Annotated[int, typer.Option("--runs")] = 3,
) -> None:
    """Run one config through the profiler and save the trial."""
    try:
        config = make_config(engine, model, parse_settings(settings))
        workload = Workload(
            name="bench",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            num_requests=num_requests,
        )
        trial = run_bench(make_engine(engine), config, workload, runs=runs)
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    path = save_trial(trial, results)
    typer.echo(_trial_summary(trial))
    typer.echo(f"Saved to {path}")
    if trial.status is Status.FAILED:
        raise typer.Exit(code=1)


@app.command("plan")
def plan_cmd(
    gfx: Annotated[str, typer.Option("--gfx", help="GPU target, e.g. gfx942.")],
) -> None:
    """List the engine and precision pairs allowed on a GPU target."""
    try:
        pairs = candidate_pairs(gfx)
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(format_plan(gfx, build_plan(pairs)))


if __name__ == "__main__":
    app()
