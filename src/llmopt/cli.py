import json
from pathlib import Path
from typing import Annotated, Any

import typer

from llmopt.hardware.gpu import detect_gpus
from llmopt.hardware.host import detect_host
from llmopt.precision.selector import candidate_pairs
from llmopt.profile.bench import run_bench
from llmopt.profile.factory import make_config, make_engine, parse_settings
from llmopt.profile.workloads import get_workload
from llmopt.report.render import write_report
from llmopt.report.store import load_report, load_trial, load_trials, save_trial
from llmopt.schemas import AccuracyResult, Status, TrialResult, Workload
from llmopt.tune.plan import build_plan, format_plan
from llmopt.validate.baseline import (
    Baseline,
    attach_accuracy,
    check_baseline,
    read_baseline,
    write_baseline,
)

app = typer.Typer(help="llmopt: find the best way to serve an LLM on an AMD GPU.")


@app.callback()
def main() -> None:
    """llmopt command line."""


@app.command()
def detect(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to save the JSON.")
    ] = Path("hardware_profile.json"),
    gpu: Annotated[
        bool, typer.Option("--gpu", help="Also read GPUs with rocminfo and amd-smi.")
    ] = False,
) -> None:
    """Print and save the host hardware profile."""
    profile = detect_host()
    if gpu:
        data = profile.model_dump(mode="json")
        data["gpus"] = detect_gpus().model_dump(mode="json")
        text = json.dumps(data, indent=2)
    else:
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


def _check_vllm_context(engine: str, settings: dict[str, Any], workload: Workload) -> None:
    """vLLM rejects requests longer than max_model_len; say so before loading the model."""
    if engine not in ("vllm", "vllm-rocm"):
        return
    limit = settings.get("max_model_len", 2048)
    need = workload.prompt_tokens + workload.output_tokens
    if isinstance(limit, int) and not isinstance(limit, bool) and need > limit:
        suggest = -(-need // 1024) * 1024
        raise ValueError(
            f"workload {workload.name!r} needs {need} tokens but max_model_len is {limit}; "
            f"add --settings max_model_len={suggest}"
        )


@app.command("bench")
def bench_cmd(
    engine: Annotated[
        str,
        typer.Option("--engine", help="pytorch, vllm, pytorch-rocm or vllm-rocm."),
    ],
    model: Annotated[Path, typer.Option("--model", help="Model folder.")],
    results: Annotated[
        Path, typer.Option("--results", help="Folder to save the trial JSON in.")
    ] = Path("results"),
    settings: Annotated[
        str, typer.Option("--settings", help="Engine settings, like a=1,b=true.")
    ] = "",
    workload_name: Annotated[
        str, typer.Option("--workload", help="Named workload: smoke, chat or long.")
    ] = "smoke",
    prompt_tokens: Annotated[
        int, typer.Option("--prompt-tokens", help="Override the workload (0 = keep).")
    ] = 0,
    output_tokens: Annotated[
        int, typer.Option("--output-tokens", help="Override the workload (0 = keep).")
    ] = 0,
    num_requests: Annotated[
        int, typer.Option("--num-requests", help="Override the workload (0 = keep).")
    ] = 0,
    runs: Annotated[int, typer.Option("--runs")] = 3,
    precision: Annotated[
        str, typer.Option("--precision", help="Precision, e.g. bf16 or fp8.")
    ] = "bf16",
    kv_cache_dtype: Annotated[
        str, typer.Option("--kv-cache-dtype", help="KV cache type: auto or fp8.")
    ] = "auto",
) -> None:
    """Run one config through the profiler and save the trial."""
    try:
        parsed = parse_settings(settings)
        config = make_config(engine, model, parsed, precision, kv_cache_dtype)
        base = get_workload(workload_name)
        workload = Workload(
            name=base.name,
            prompt_tokens=prompt_tokens or base.prompt_tokens,
            output_tokens=output_tokens or base.output_tokens,
            num_requests=num_requests or base.num_requests,
        )
        _check_vllm_context(engine, parsed, workload)
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


TASKS = {"ppl": "wikitext2_perplexity", "arc": "arc_easy_local"}


def _score(task: str, model: Path, datasets: Path, limit: int | None) -> float:
    """Run the real scorer. Needs torch and transformers (ppl) or lm_eval (arc)."""
    if task == "arc":
        from llmopt.validate.lmeval import run_arc_easy

        return run_arc_easy(model, datasets, limit=limit)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmopt.validate.perplexity import evaluate_perplexity, load_wikitext_texts

    texts = load_wikitext_texts(
        datasets / "wikitext" / "wikitext-2-raw-v1" / "test-00000-of-00001.parquet"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    network: Any = AutoModelForCausalLM.from_pretrained(
        str(model), torch_dtype=torch.bfloat16
    )
    network.eval()
    result = evaluate_perplexity(network, tokenizer, texts, max_windows=limit or 20)
    return result.perplexity


def _compare(task_name: str, score: float, baseline_score: float) -> AccuracyResult:
    if task_name == TASKS["ppl"]:
        from llmopt.validate.perplexity import perplexity_accuracy

        return perplexity_accuracy(score, baseline_score)
    from llmopt.validate.lmeval import accuracy_result

    return accuracy_result(score, baseline_score)


@app.command("validate")
def validate_cmd(
    model: Annotated[Path, typer.Option("--model", help="Model folder.")],
    task: Annotated[str, typer.Option("--task", help="ppl or arc.")],
    baseline: Annotated[
        Path | None, typer.Option("--baseline", help="Baseline file to compare with.")
    ] = None,
    write_baseline_to: Annotated[
        Path | None,
        typer.Option("--write-baseline", help="Score the model and save a baseline."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Smoke tests only: windows or questions."),
    ] = None,
    datasets: Annotated[Path, typer.Option("--datasets")] = Path("~/datasets"),
    trial: Annotated[
        Path | None, typer.Option("--trial", help="Trial JSON to attach the result to.")
    ] = None,
) -> None:
    """Score a model and compare it with a baseline, or save a new baseline."""
    saved: Baseline | None = None
    trial_data: TrialResult | None = None
    try:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; use ppl or arc")
        if (baseline is None) == (write_baseline_to is None):
            raise ValueError("give exactly one of --baseline or --write-baseline")
        if write_baseline_to is not None and limit is not None:
            raise ValueError("a limited score is not a real baseline; drop --limit")
        task_name = TASKS[task]
        if baseline is not None:
            saved = read_baseline(baseline)
            check_baseline(saved, task_name, limit)
        if trial is not None:
            trial_data = load_trial(trial)
        if limit is not None:
            typer.echo(
                f"WARNING: --limit {limit} used. This is a smoke run, not a real score.",
                err=True,
            )
        score = _score(task, model, datasets.expanduser(), limit)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"{task_name}: score {score:.4f}")
    if write_baseline_to is not None:
        write_baseline(
            Baseline(task=task_name, score=score, model=str(model), limit=limit),
            write_baseline_to,
        )
        typer.echo(f"Baseline saved to {write_baseline_to}")
        return
    if saved is None:
        return  # not reachable: the checks above need one of the two options
    result = _compare(task_name, score, saved.score)
    typer.echo(
        f"baseline {result.baseline_score:.4f}, "
        f"drop {result.drop_pct:+.2f}% (positive = worse)"
    )
    if trial is not None and trial_data is not None:
        updated = attach_accuracy(trial_data, result)
        trial.write_text(updated.model_dump_json(indent=2) + "\n")
        typer.echo(f"Attached to {trial}")


if __name__ == "__main__":
    app()
