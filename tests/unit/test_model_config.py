"""Tests for reading Hugging Face configs and estimating model memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmopt.models.config import (
    ModelConfig,
    count_parameters,
    estimate_weight_bytes,
    kv_cache_bytes,
    load_model_config,
    parse_model_config,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "models"

# Totals from the safetensors index of each model (BF16 weights, in bytes).
LLAMA_8B_BYTES = 16_060_522_496
QWEN3_8B_BYTES = 16_381_470_720


def _raw(name: str) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return raw


def _cfg(name: str) -> ModelConfig:
    return load_model_config(FIXTURES / name)


def test_load_llama_reads_architecture() -> None:
    cfg = _cfg("llama-3.1-8b-instruct.json")
    assert cfg.model_type == "llama"
    assert cfg.num_layers == 32
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.qk_norm is False


def test_qwen3_has_qk_norm() -> None:
    cfg = _cfg("qwen3-8b.json")
    assert cfg.model_type == "qwen3"
    assert cfg.qk_norm is True


def test_llama_weight_bytes_match_safetensors_index() -> None:
    cfg = _cfg("llama-3.1-8b-instruct.json")
    assert estimate_weight_bytes(count_parameters(cfg), "bf16") == LLAMA_8B_BYTES


def test_qwen3_weight_bytes_match_safetensors_index() -> None:
    cfg = _cfg("qwen3-8b.json")
    assert estimate_weight_bytes(count_parameters(cfg), "bf16") == QWEN3_8B_BYTES


def test_fp8_weights_only_shrink_linear_layers() -> None:
    counts = count_parameters(_cfg("llama-3.1-8b-instruct.json"))
    others = counts.embedding + counts.lm_head + counts.norm
    fp8 = estimate_weight_bytes(counts, "fp8")
    assert fp8 == counts.linear + others * 2
    assert fp8 < estimate_weight_bytes(counts, "bf16")


def test_kv_cache_bytes_scales_with_format_tokens_and_batch() -> None:
    cfg = _cfg("llama-3.1-8b-instruct.json")
    per_token_bf16 = 2 * 32 * 8 * 128 * 2  # keys+values x layers x kv heads x head_dim x 2 bytes
    assert kv_cache_bytes(cfg, 1) == per_token_bf16
    assert kv_cache_bytes(cfg, 1, "fp8") == per_token_bf16 // 2
    assert kv_cache_bytes(cfg, 1000, batch=4) == per_token_bf16 * 1000 * 4


def test_unsupported_model_type_is_rejected() -> None:
    raw = _raw("llama-3.1-8b-instruct.json")
    raw["model_type"] = "mistral"
    with pytest.raises(ValueError, match="unsupported model_type"):
        parse_model_config(raw)


def test_missing_key_and_wrong_type_are_rejected() -> None:
    missing = _raw("qwen3-8b.json")
    del missing["vocab_size"]
    with pytest.raises(ValueError, match="no 'vocab_size'"):
        parse_model_config(missing)

    wrong_type = _raw("qwen3-8b.json")
    wrong_type["hidden_size"] = "4096"
    with pytest.raises(TypeError, match="must be an integer"):
        parse_model_config(wrong_type)