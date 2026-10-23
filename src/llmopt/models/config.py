from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SUPPORTED_MODEL_TYPES = ("llama", "qwen3")
ValueFormat = Literal["bf16", "fp8"]
BYTES_PER_VALUE: dict[str, int] = {"bf16": 2, "fp8": 1}


class ModelConfig(BaseModel):

    model_type: str
    num_layers: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    max_position_embeddings: int = Field(gt=0)
    tie_word_embeddings: bool
    qk_norm: bool
    torch_dtype: str
    quant_method: str | None = None


class ParameterCount(BaseModel):
   
    linear: int
    embedding: int
    lm_head: int
    norm: int

    @property
    def total(self) -> int:
        return self.linear + self.embedding + self.lm_head + self.norm


def _need_int(raw: dict[str, Any], key: str) -> int:
    if key not in raw:
        raise ValueError(f"config.json has no '{key}'")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"config.json '{key}' must be an integer, got {value!r}")
    return value


def _need_bool(raw: dict[str, Any], key: str) -> bool:
    if key not in raw:
        raise ValueError(f"config.json has no '{key}'")
    value = raw[key]
    if not isinstance(value, bool):
        raise TypeError(f"config.json '{key}' must be true or false, got {value!r}")
    return value


def parse_model_config(raw: dict[str, Any]) -> ModelConfig:
  
    model_type = raw.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"unsupported model_type {model_type!r} (supported: {supported})")
    heads = _need_int(raw, "num_attention_heads")
    hidden = _need_int(raw, "hidden_size")
    quantization = raw.get("quantization_config") or {}
    return ModelConfig(
        model_type=model_type,
        num_layers=_need_int(raw, "num_hidden_layers"),
        hidden_size=hidden,
        intermediate_size=_need_int(raw, "intermediate_size"),
        num_attention_heads=heads,
        num_kv_heads=raw.get("num_key_value_heads", heads),
        head_dim=raw.get("head_dim") or hidden // heads,
        vocab_size=_need_int(raw, "vocab_size"),
        max_position_embeddings=_need_int(raw, "max_position_embeddings"),
        tie_word_embeddings=_need_bool(raw, "tie_word_embeddings"),
        qk_norm=model_type == "qwen3",
        torch_dtype=str(raw.get("torch_dtype", "bfloat16")),
        quant_method=quantization.get("quant_method"),
    )


def load_model_config(path: Path) -> ModelConfig:
   
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.is_file():
        raise FileNotFoundError(f"no config.json at {config_path}")
    raw = json.loads(config_path.read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path} does not contain a JSON object")
    return parse_model_config(raw)


def count_parameters(cfg: ModelConfig) -> ParameterCount:
 
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_kv_heads * cfg.head_dim
    attention = cfg.hidden_size * (q_out + 2 * kv_out) + q_out * cfg.hidden_size
    mlp = 3 * cfg.hidden_size * cfg.intermediate_size
    norm = 2 * cfg.hidden_size + (2 * cfg.head_dim if cfg.qk_norm else 0)
    embedding = cfg.vocab_size * cfg.hidden_size
    return ParameterCount(
        linear=cfg.num_layers * (attention + mlp),
        embedding=embedding,
        lm_head=0 if cfg.tie_word_embeddings else embedding,
        norm=cfg.num_layers * norm + cfg.hidden_size,
    )


def estimate_weight_bytes(counts: ParameterCount, weight_format: ValueFormat) -> int:
  
    others = counts.embedding + counts.lm_head + counts.norm
    return counts.linear * BYTES_PER_VALUE[weight_format] + others * BYTES_PER_VALUE["bf16"]


def kv_cache_bytes(
    cfg: ModelConfig, tokens: int, kv_format: ValueFormat = "bf16", batch: int = 1
) -> int:
    """KV-cache memory: keys and values for every layer, for tokens x batch."""
    per_token = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * BYTES_PER_VALUE[kv_format]
    return per_token * tokens * batch

class MemoryComparison(BaseModel):

    estimated_bytes: int = Field(gt=0)
    measured_peak_bytes: int = Field(gt=0)

    @property
    def gap_bytes(self) -> int:
        return self.measured_peak_bytes - self.estimated_bytes

    @property
    def gap_pct(self) -> float:
        return 100.0 * self.gap_bytes / self.estimated_bytes

    @property
    def estimate_is_lower_bound(self) -> bool:
        """True when the measured peak is at least the estimate, as expected."""
        return self.measured_peak_bytes >= self.estimated_bytes


def compare_with_measured(estimated_bytes: int, measured_peak_bytes: int) -> MemoryComparison:
    """Compare estimated weights + KV cache with a measured peak. Overheads sit in the gap."""
    return MemoryComparison(
        estimated_bytes=estimated_bytes, measured_peak_bytes=measured_peak_bytes
    )
