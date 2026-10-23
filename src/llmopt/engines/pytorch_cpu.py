"""PyTorch engine adapter that runs on CPU. This is the Phase 1 baseline."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.schemas import Config, Engine, HardwareProfile, Precision

# Repeated and cut to the exact prompt length, so every run uses the same prompt.
FILLER_TEXT = "The quick brown fox jumps over the lazy dog. "


def build_prompt_ids(tokenizer: Any, prompt_tokens: int) -> list[int]:
    """Token ids of exactly prompt_tokens tokens, made by repeating FILLER_TEXT."""
    ids: list[int] = tokenizer.encode(FILLER_TEXT, add_special_tokens=False)
    if not ids:
        raise ValueError("tokenizer produced no tokens for the filler text")
    repeats = prompt_tokens // len(ids) + 1
    return (ids * repeats)[:prompt_tokens]


class PyTorchCpuEngine(EngineAdapter):
    name = "pytorch-cpu"

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None

    def supports(self, hardware: HardwareProfile, model: str, precision: Precision) -> SupportResult:
        if precision is not Precision.BF16:
            return SupportResult(
                ok=False, reason=f"pytorch cpu adapter supports bf16 only, not {precision.value}"
            )
        if not Path(model).is_dir():
            return SupportResult(ok=False, reason=f"model folder not found: {model}")
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        if config.engine is not Engine.PYTORCH:
            raise ValueError(f"pytorch adapter needs engine=pytorch, got {config.engine.value}")
        if config.precision is not Precision.BF16:
            raise ValueError(f"pytorch cpu adapter supports bf16 only, got {config.precision.value}")
        if config.kv_cache_dtype != "auto":
            raise ValueError(f"kv_cache_dtype must be 'auto', got {config.kv_cache_dtype!r}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        threads = config.settings.get("num_threads")
        if threads is not None:
            if isinstance(threads, bool) or not isinstance(threads, int):
                raise TypeError(f"setting 'num_threads' must be an integer, got {threads!r}")
            torch.set_num_threads(threads)
        path = Path(config.model)
        if not path.is_dir():
            raise FileNotFoundError(f"model folder not found: {path}")
        self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model: Any = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, local_files_only=True
        )
        model.eval()
        self._model = model

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("call prepare() before generate()")
        return [self._generate_one(req) for req in requests]

    def _generate_one(self, req: GenRequest) -> GenResult:
        """Greedy decoding, one request at a time. EOS is ignored so every run makes
        exactly max_new_tokens tokens and timings stay comparable."""
        import torch

        ids = build_prompt_ids(self._tokenizer, req.prompt_tokens)
        input_ids = torch.tensor([ids], dtype=torch.long)
        new_ids: list[int] = []
        times: list[float] = []
        with torch.inference_mode():
            start = time.perf_counter()
            out = self._model(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1)
            new_ids.append(int(next_id.item()))
            times.append((time.perf_counter() - start) * 1000.0)
            for _ in range(req.max_new_tokens - 1):
                out = self._model(
                    input_ids=next_id.unsqueeze(-1), past_key_values=past, use_cache=True
                )
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1)
                new_ids.append(int(next_id.item()))
                times.append((time.perf_counter() - start) * 1000.0)
        return GenResult(
            request_id=req.request_id,
            token_times_ms=times,
            text=self._tokenizer.decode(new_ids),
        )

    def teardown(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()