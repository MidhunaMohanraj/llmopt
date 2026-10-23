"""Accuracy on ARC-Easy through lm-eval, reading the local parquet files offline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from llmopt.schemas import AccuracyResult

TASK_NAME = "arc_easy_local"
SPLITS = ("train", "validation", "test")


def build_task_yaml(datasets_dir: Path) -> str:
    """lm-eval task file: the built-in arc_easy prompt, with data read from local parquet."""
    arc_dir = datasets_dir / "ai2_arc" / "ARC-Easy"
    files = "\n".join(
        f"    {split}: {arc_dir / f'{split}-00000-of-00001.parquet'}" for split in SPLITS
    )
    return (
        f"task: {TASK_NAME}\n"
        "dataset_path: parquet\n"
        "dataset_kwargs:\n"
        "  data_files:\n"
        f"{files}\n"
        "output_type: multiple_choice\n"
        "training_split: train\n"
        "validation_split: validation\n"
        "test_split: test\n"
        'doc_to_text: "Question: {{question}}\\nAnswer:"\n'
        'doc_to_target: "{{choices.label.index(answerKey)}}"\n'
        'doc_to_choice: "{{choices.text}}"\n'
        "metric_list:\n"
        "  - metric: acc\n"
        "    aggregation: mean\n"
        "    higher_is_better: true\n"
        "  - metric: acc_norm\n"
        "    aggregation: mean\n"
        "    higher_is_better: true\n"
        "metadata:\n"
        "  version: 1.0\n"
    )


def build_command(
    model_dir: Path,
    task_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    device: str = "cpu",
    dtype: str = "bfloat16",
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_dir},dtype={dtype}",
        "--tasks",
        TASK_NAME,
        "--include_path",
        str(task_dir),
        "--device",
        device,
        "--batch_size",
        "1",
        "--output_path",
        str(output_dir),
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        command += ["--limit", str(limit)]
    return command


def accuracy_from_results(
    results: dict[str, Any], task: str = TASK_NAME, metric: str = "acc"
) -> float:
    """Read one metric from an lm-eval results file (keys look like 'acc,none')."""
    task_results = results.get("results", {}).get(task)
    if task_results is None:
        raise KeyError(f"task {task!r} not found in lm-eval results")
    key = f"{metric},none"
    if key not in task_results:
        raise KeyError(f"metric {key!r} not found for task {task!r}")
    return float(task_results[key])


def accuracy_result(score: float, baseline_score: float) -> AccuracyResult:
    """Higher accuracy is better, so a positive drop_pct means the model got worse."""
    if baseline_score <= 0:
        raise ValueError("baseline_score must be positive")
    drop = (baseline_score - score) / baseline_score * 100
    return AccuracyResult(
        task=TASK_NAME, score=score, baseline_score=baseline_score, drop_pct=drop
    )


def run_arc_easy(
    model_dir: Path,
    datasets_dir: Path,
    limit: int | None = None,
    device: str = "cpu",
) -> float:
    """Run ARC-Easy offline and return accuracy. Needs the 'eval' extras installed.
    Use limit for smoke tests only; a limited score is not a real baseline."""
    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp) / "tasks"
        output_dir = Path(tmp) / "out"
        task_dir.mkdir()
        (task_dir / f"{TASK_NAME}.yaml").write_text(build_task_yaml(datasets_dir))
        env = {**os.environ, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
        command = build_command(model_dir, task_dir, output_dir, limit, device)
        try:
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            tail = "\n".join(error.stderr.splitlines()[-15:])
            raise RuntimeError(f"lm-eval failed:\n{tail}") from error
        files = sorted(output_dir.rglob("results_*.json"))
        if not files:
            raise RuntimeError("lm-eval wrote no results file")
        return accuracy_from_results(json.loads(files[-1].read_text()))