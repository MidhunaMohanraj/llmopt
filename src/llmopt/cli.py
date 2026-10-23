from pathlib import Path
from typing import Annotated

import typer

from llmopt.hardware.host import detect_host

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


if __name__ == "__main__":
    app()
