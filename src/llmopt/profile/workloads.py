"""Named workload shapes, read from configs/workloads.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from llmopt.precision.selector import CONFIG_DIR
from llmopt.schemas import Workload


def load_workloads(path: Path | None = None) -> dict[str, Workload]:
    data = yaml.safe_load((path or CONFIG_DIR / "workloads.yaml").read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("workloads file must be a non-empty mapping")
    workloads: dict[str, Workload] = {}
    for name, fields in data.items():
        if not isinstance(fields, dict):
            raise TypeError(f"workload {name!r} must be a mapping of settings")
        workloads[str(name)] = Workload(name=str(name), **fields)
    return workloads


def get_workload(name: str, path: Path | None = None) -> Workload:
    workloads = load_workloads(path)
    if name not in workloads:
        valid = ", ".join(sorted(workloads))
        raise ValueError(f"unknown workload {name!r}; choose one of: {valid}")
    return workloads[name]