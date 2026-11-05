from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llmopt.schemas import AccuracyResult

TASK_NAME = "wikitext2_perplexity"


class PerplexityResult(BaseModel):
    perplexity: float
    tokens_scored: int
    windows: int
    window_tokens: int


def perplexity_from_nll(total_nll: float, token_count: int) -> float:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return math.exp(total_nll / token_count)


def load_wikitext_texts(path: Path) -> list[str]:
    parquet = importlib.import_module("pyarrow.parquet")
    table = parquet.read_table(path, columns=["text"])
    texts: list[str] = table.column("text").to_pylist()
    return texts


def check_device(device: str) -> None:
    name, colon, index = device.partition(":")
    valid = (name == "cpu" and not colon) or (
        name == "cuda" and (not colon or index.isdigit())
    )
    if not valid:
        raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    window_tokens: int = 512,
    max_windows: int = 20,
    device: str = "cpu",
) -> PerplexityResult:
    import torch

    if window_tokens < 2:
        raise ValueError(f"window_tokens must be at least 2, got {window_tokens}")
    if max_windows < 1:
        raise ValueError(f"max_windows must be at least 1, got {max_windows}")
    check_device(device)
    text = "\n\n".join(texts)[: window_tokens * max_windows * 8]
    ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    windows = min(max_windows, len(ids) // window_tokens)
    if windows < 1:
        raise ValueError(
            f"text gave {len(ids)} tokens, fewer than one window of {window_tokens}"
        )
    total_nll = 0.0
    scored = 0
    with torch.inference_mode():
        for index in range(windows):
            chunk = ids[index * window_tokens : (index + 1) * window_tokens]
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits[0, :-1].float(), input_ids[0, 1:], reduction="sum"
            )
            total_nll += float(loss.item())
            scored += window_tokens - 1
    return PerplexityResult(
        perplexity=perplexity_from_nll(total_nll, scored),
        tokens_scored=scored,
        windows=windows,
        window_tokens=window_tokens,
    )


def perplexity_accuracy(score: float, baseline_score: float) -> AccuracyResult:
    if baseline_score <= 0:
        raise ValueError("baseline_score must be positive")
    drop = (score - baseline_score) / baseline_score * 100
    return AccuracyResult(
        task=TASK_NAME, score=score, baseline_score=baseline_score, drop_pct=drop
    )
