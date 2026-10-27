"""PyTorch adapter for AMD GPUs (ROCm).

DRAFT: NOT TESTED ON A GPU. Only the config checks and the no-GPU error are tested, on a
CPU machine. Timings, memory and the device move must be confirmed on a real MI300X before
any number from this adapter is used. The num_threads setting is ignored here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from llmopt.engines.base import GenRequest, GenResult
from llmopt.engines.pytorch_cpu import PyTorchCpuEngine, build_prompt_ids
from llmopt.schemas import Config, Engine, Precision

# ROCm builds of torch call AMD GPUs "cuda".
DEVICE = "cuda"


def check_config(config: Config) -> None:
    """Reject configs this adapter cannot run, before anything is loaded."""
    if config.engine is not Engine.PYTORCH:
        raise ValueError(f"pytorch adapter needs engine=pytorch, got {config.engine.value}")
    if config.precision is not Precision.BF16:
        raise ValueError(
            f"pytorch rocm adapter supports bf16 only, got {config.precision.value}"
        )
    if config.kv_cache_dtype != "auto":
        raise ValueError(f"kv_cache_dtype must be 'auto', got {config.kv_cache_dtype!r}")
    if not Path(config.model).is_dir():
        raise FileNotFoundError(f"model folder not found: {config.model}")


class PyTorchRocmEngine(PyTorchCpuEngine):
    name = "pytorch-rocm"

    def prepare(self, config: Config) -> None:
        check_config(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU is visible to torch (torch.cuda.is_available() is False)")
        self._tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
        model: Any = AutoModelForCausalLM.from_pretrained(
            config.model, dtype=torch.bfloat16, local_files_only=True
        )
        model.to(DEVICE)
        model.eval()
        self._model = model

    def _generate_one(self, req: GenRequest) -> GenResult:
        """Greedy decoding on the GPU, one request at a time. Reading the token with
        .item() waits for the GPU, so every timestamp includes the finished step."""
        import torch

        ids = build_prompt_ids(self._tokenizer, req.prompt_tokens)
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        new_ids: list[int] = []
        times: list[float] = []
        with torch.inference_mode():
            torch.cuda.synchronize()
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