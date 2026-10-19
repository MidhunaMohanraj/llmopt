import typer

app = typer.Typer(help="llmopt: find the best way to serve an LLM on an AMD GPU.")


@app.command()
def detect() -> None:
    """Print and save the hardware profile."""
    typer.echo("Not implemented yet.")


if __name__ == "__main__":
    app()
