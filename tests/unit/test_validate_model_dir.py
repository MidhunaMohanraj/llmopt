from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmopt import cli as cli_module
from llmopt.cli import app

runner = CliRunner()


def test_score_refuses_a_missing_model_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model folder not found"):
        cli_module._score("ppl", tmp_path / "missing", tmp_path, None, "cpu")


def test_validate_exits_cleanly_for_a_missing_model_folder(tmp_path: Path) -> None:
    baseline = tmp_path / "b.json"
    args = [
        "validate",
        "--model",
        str(tmp_path / "missing"),
        "--task",
        "ppl",
        "--write-baseline",
        str(baseline),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert not baseline.exists()