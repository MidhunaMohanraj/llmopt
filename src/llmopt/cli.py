from pathlib import Path
from typing import Annotated

import typer

from llmopt.hardware.host import detect_host
from llmopt.report.render import write_report
from llmopt.report.store import load_report, load_trials

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


if __name__ == "__main__":
    app()
