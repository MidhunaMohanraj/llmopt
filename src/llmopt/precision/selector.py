"""Decide which precisions and engine pairs are allowed on a GPU."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from llmopt.schemas import Engine, Precision

# Works when llmopt is installed with "pip install -e". The repo root is three levels up.
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


class GpuInfo(BaseModel):
    name: str
    hbm_gb: float
    precisions: list[Precision]


def load_gpus(path: Path | None = None) -> dict[str, GpuInfo]:
    data = yaml.safe_load((path or CONFIG_DIR / "gpus.yaml").read_text())
    return {gfx: GpuInfo(**info) for gfx, info in data.items()}


def load_compat(path: Path | None = None) -> dict[Engine, list[Precision]]:
    data = yaml.safe_load((path or CONFIG_DIR / "compat_matrix.yaml").read_text())
    return {Engine(name): [Precision(p) for p in plist] for name, plist in data["engines"].items()}


def supported_precisions(gfx_target: str) -> list[Precision]:
    gpus = load_gpus()
    if gfx_target not in gpus:
        raise ValueError(f"unknown GPU target: {gfx_target}")
    return list(gpus[gfx_target].precisions)


def candidate_pairs(gfx_target: str) -> list[tuple[Engine, Precision]]:
    """Every engine and precision pair that is allowed on this GPU."""
    allowed = set(supported_precisions(gfx_target))
    return [
        (engine, precision)
        for engine, precisions in load_compat().items()
        for precision in precisions
        if precision in allowed
    ]
