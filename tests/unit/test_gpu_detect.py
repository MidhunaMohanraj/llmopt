import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import llmopt.cli as cli_module
from llmopt.cli import app
from llmopt.hardware.gpu import GpuReport, detect_gpus

runner = CliRunner()
FIXTURES = Path(__file__).parent.parent / "fixtures" / "gpu"
AMD_SMI_STATIC = (FIXTURES / "amd_smi_static_docs.txt").read_text()

# Synthetic sample written for this test, not real rocminfo output.
ROCMINFO = """\
Agent 1
  Name:                    Example CPU
  Device Type:             CPU
Agent 2
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
  Compute Unit:            304
  Max Clock Freq. (MHz):   2100
"""


def _which_all(name: str) -> str | None:
    return f"/usr/bin/{name}"


def _runner(command: list[str]) -> str:
    return ROCMINFO if command[0] == "rocminfo" else AMD_SMI_STATIC


def test_no_tools_gives_notes_not_errors() -> None:
    report = detect_gpus(runner=_runner, which=lambda name: None)
    assert not report.found
    assert report.notes == ["rocminfo not found", "amd-smi not found"]


def test_both_tools_are_parsed() -> None:
    report = detect_gpus(runner=_runner, which=_which_all)
    assert report.found
    assert report.rocminfo[0].gfx_target == "gfx942"
    assert report.rocminfo[0].compute_units == 304
    assert report.amd_smi
    assert report.notes == []


def test_tool_failure_becomes_a_note() -> None:
    def failing(command: list[str]) -> str:
        raise RuntimeError("no GPU devices")

    report = detect_gpus(runner=failing, which=_which_all)
    assert not report.found
    assert any("rocminfo failed: no GPU devices" in n for n in report.notes)
    assert any("amd-smi failed: no GPU devices" in n for n in report.notes)


def test_unparseable_output_becomes_a_note() -> None:
    report = detect_gpus(runner=lambda command: "garbage", which=_which_all)
    assert not report.found
    assert len(report.notes) == 2


def test_detect_without_gpu_flag_is_unchanged(tmp_path: Path) -> None:
    out = tmp_path / "profile.json"
    result = runner.invoke(app, ["detect", "-o", str(out)])
    assert result.exit_code == 0
    assert "gpus" not in json.loads(out.read_text())


def test_detect_with_gpu_flag_adds_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "detect_gpus", lambda: GpuReport(notes=["rocminfo not found"]))
    out = tmp_path / "profile.json"
    result = runner.invoke(app, ["detect", "--gpu", "-o", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text())
    assert data["gpus"]["notes"] == ["rocminfo not found"]
    assert "cpu_model" in data