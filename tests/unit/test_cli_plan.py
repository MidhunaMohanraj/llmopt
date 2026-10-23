from __future__ import annotations

import pytest
from typer.testing import CliRunner

import llmopt.cli as cli_module
from llmopt.cli import app
from llmopt.schemas import Engine, Precision

runner = CliRunner()


def test_plan_lists_the_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pairs(gfx_target: str) -> list[tuple[Engine, Precision]]:
        return [(Engine.VLLM, Precision.BF16), (Engine.VLLM, Precision.FP8)]

    monkeypatch.setattr(cli_module, "candidate_pairs", fake_pairs)
    result = runner.invoke(app, ["plan", "--gfx", "gfx942"])
    assert result.exit_code == 0
    assert "Candidates for gfx942: 2" in result.output
    assert "1 of 2 can be run" in result.output


def test_plan_rejects_an_unknown_gpu() -> None:
    result = runner.invoke(app, ["plan", "--gfx", "gfx-does-not-exist"])
    assert result.exit_code == 1