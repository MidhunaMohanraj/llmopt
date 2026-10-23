"""Data models shared by every part of llmopt."""

from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, Field

SettingValue = str | int | float | bool


class Engine(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    PYTORCH = "pytorch"
    LLAMACPP = "llamacpp"


class Precision(str, Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    AWQ_INT4 = "awq_int4"
    GPTQ_INT4 = "gptq_int4"
    GGUF = "gguf"
    MXFP4 = "mxfp4"


class Status(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    FAILED = "failed"


class HardwareProfile(BaseModel):
    gpu_name: str
    gfx_target: str
    compute_units: int = Field(gt=0)
    hbm_gb: float = Field(gt=0)
    rocm_version: str
    driver_version: str
    partition_mode: str = "SPX"
    cpu_model: str
    cpu_cores: int = Field(gt=0)
    numa_nodes: int = Field(default=1, gt=0)


class Workload(BaseModel):
    name: str
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    concurrency: int = Field(default=1, gt=0)
    num_requests: int = Field(default=32, gt=0)
    seed: int = 0


class Objective(BaseModel):
    """Hard limits plus the one thing to improve."""

    max_ttft_p95_ms: float | None = Field(default=None, gt=0)
    max_accuracy_drop_pct: float | None = Field(default=None, ge=0)
    optimize: str = "throughput"


class Config(BaseModel):
    model: str
    engine: Engine
    precision: Precision
    kv_cache_dtype: str = "auto"
    settings: dict[str, SettingValue] = Field(default_factory=dict)

    def key(self) -> str:
        """Short stable id for this config, used to cache results."""
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class Metrics(BaseModel):
    ttft_ms_p50: float
    ttft_ms_p95: float
    tpot_ms: float
    throughput_tps: float
    latency_ms_p50: float
    latency_ms_p95: float
    peak_memory_gb: float
    energy_per_token_j: float | None = None
    run_spread_pct: float = 0.0


class AccuracyResult(BaseModel):
    task: str
    score: float
    baseline_score: float
    drop_pct: float


class TrialResult(BaseModel):
    config: Config
    status: Status
    reason: str | None = None
    metrics: Metrics | None = None
    accuracy: list[AccuracyResult] = Field(default_factory=list)


class Report(BaseModel):
    hardware: HardwareProfile
    workload: Workload
    objective: Objective
    trials: list[TrialResult] = Field(default_factory=list)
    recommendation: Config | None = None
    versions: dict[str, str] = Field(default_factory=dict)
