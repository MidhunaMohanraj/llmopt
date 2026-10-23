"""vLLM engine adapter that runs on CPU (Phase 1, main engine on the bench PC)."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.engines.pytorch_cpu import build_prompt_ids
from llmopt.schemas import Config, Engine, HardwareProfile, Precision


def _int_setting(config: Config, key: str, default: int) -> int:
    value = config.settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"setting '{key}' must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"setting '{key}' must be greater than 0, got {value}")
    return value


def run_request(
    engine: Any,
    params: Any,
    prompt_ids: list[int],
    vllm_id: str,
    request_id: str,
    new_tokens: int,
) -> GenResult:
    """Run one request through the vLLM step loop and time every output token.

    The clock starts just before add_request. If one step returns several new tokens,
    they share that step's time, because they cannot be told apart.
    """
    times: list[float] = []
    text = ""
    start = time.perf_counter()
    engine.add_request(vllm_id, {"prompt_token_ids": prompt_ids}, params)
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.request_id != vllm_id:
                continue
            now = (time.perf_counter() - start) * 1000.0
            made = len(out.outputs[0].token_ids)
            times.extend([now] * (made - len(times)))
            text = out.outputs[0].text
    if len(times) != new_tokens:
        raise RuntimeError(f"expected {new_tokens} tokens, vLLM returned {len(times)}")
    return GenResult(request_id=request_id, token_times_ms=times, text=text)


class VllmCpuEngine(EngineAdapter):
    name = "vllm-cpu"

    def __init__(self) -> None:
        self._llm: Any = None
        self._tokenizer: Any = None
        self._max_model_len = 0
        self._counter = 0

    def supports(self, hardware: HardwareProfile, model: str, precision: Precision) -> SupportResult:
        if precision is not Precision.BF16:
            return SupportResult(
                ok=False, reason=f"vllm cpu adapter supports bf16 only, not {precision.value}"
            )
        if not Path(model).is_dir():
            return SupportResult(ok=False, reason=f"model folder not found: {model}")
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        if config.engine is not Engine.VLLM:
            raise ValueError(f"vllm adapter needs engine=vllm, got {config.engine.value}")
        if config.precision is not Precision.BF16:
            raise ValueError(f"vllm cpu adapter supports bf16 only, got {config.precision.value}")
        if config.kv_cache_dtype != "auto":
            raise ValueError(f"kv_cache_dtype must be 'auto', got {config.kv_cache_dtype!r}")
        kv_space = _int_setting(config, "kv_cache_space_gib", 4)
        max_len = _int_setting(config, "max_model_len", 2048)
        path = Path(config.model)
        if not path.is_dir():
            raise FileNotFoundError(f"model folder not found: {path}")
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = str(kv_space)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # Prefix caching is off so repeated prompts do not make TTFT look faster than it is.
        llm: Any = LLM(
            model=str(path),
            dtype="bfloat16",
            max_model_len=max_len,
            enable_prefix_caching=False,
        )
        warm_ids = build_prompt_ids(self._tokenizer, 32)
        warm = SamplingParams(temperature=0.0, max_tokens=4, min_tokens=4, ignore_eos=True)
        llm.generate([{"prompt_token_ids": warm_ids}], warm)
        self._llm = llm
        self._max_model_len = max_len

    def _make_sampling_params(self, max_new_tokens: int) -> Any:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        return SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            min_tokens=max_new_tokens,
            ignore_eos=True,
            output_kind=RequestOutputKind.CUMULATIVE,
        )

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        """Run the requests one at a time (concurrency 1). EOS is ignored so every
        request makes exactly max_new_tokens tokens."""
        if self._llm is None or self._tokenizer is None:
            raise RuntimeError("call prepare() before generate()")
        results: list[GenResult] = []
        for req in requests:
            total = req.prompt_tokens + req.max_new_tokens
            if total > self._max_model_len:
                raise ValueError(
                    f"request needs {total} tokens, max_model_len is {self._max_model_len}"
                )
            ids = build_prompt_ids(self._tokenizer, req.prompt_tokens)
            self._counter += 1
            results.append(
                run_request(
                    self._llm.llm_engine,
                    self._make_sampling_params(req.max_new_tokens),
                    ids,
                    f"{req.request_id}#{self._counter}",
                    req.request_id,
                    req.max_new_tokens,
                )
            )
        return results

    def teardown(self) -> None:
        llm, self._llm = self._llm, None
        self._tokenizer = None
        if llm is not None:
            shutdown = getattr(llm.llm_engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        gc.collect()
