"""Read the GPUs on this machine with rocminfo and amd-smi.

Problems become notes in the report; nothing here raises. Never run on real hardware yet:
the parsers have only seen AMD documentation samples.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable

from pydantic import BaseModel, Field

from llmopt.hardware.amd_smi import AmdSmiGpu, parse_amd_smi_static
from llmopt.hardware.rocminfo import GpuAgent, parse_rocminfo

Runner = Callable[[list[str]], str]
Which = Callable[[str], str | None]


class GpuReport(BaseModel):
    rocminfo: list[GpuAgent] = Field(default_factory=list)
    amd_smi: list[AmdSmiGpu] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.rocminfo) or bool(self.amd_smi)


def run_command(command: list[str]) -> str:
    done = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()[:200]
        raise RuntimeError(f"{' '.join(command)} exited with code {done.returncode}: {detail}")
    return done.stdout


def detect_gpus(runner: Runner = run_command, which: Which = shutil.which) -> GpuReport:
    report = GpuReport()
    if which("rocminfo") is None:
        report.notes.append("rocminfo not found")
    else:
        try:
            report.rocminfo = parse_rocminfo(runner(["rocminfo"]))
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            report.notes.append(f"rocminfo failed: {error}")
        else:
            if not report.rocminfo:
                report.notes.append("rocminfo reported no GPU agents")
    if which("amd-smi") is None:
        report.notes.append("amd-smi not found")
    else:
        try:
            report.amd_smi = parse_amd_smi_static(runner(["amd-smi", "static"]))
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            report.notes.append(f"amd-smi failed: {error}")
        else:
            if not report.amd_smi:
                report.notes.append("amd-smi reported no GPUs")
    return report