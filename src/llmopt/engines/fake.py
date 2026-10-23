"""A pretend engine with adjustable behaviour, for tests that need no GPU."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.schemas import Config, HardwareProfile, Precision


class FakeBehavior(BaseModel):
    ttft_base_ms: float = 80.0
    prefill_ms_per_token: float = 0.2
    tpot_ms: float = 12.0
    # Lower factor means faster.
    speed_factor: dict[Precision, float] = Field(
        default_factory=lambda: {
            Precision.BF16: 1.0,
            Precision.FP8: 0.7,
            Precision.AWQ_INT4: 0.6,
            Precision.GPTQ_INT4: 0.65,
        }
    )
    accuracy_drop_pct: dict[Precision, float] = Field(
        default_factory=lambda: {
            Precision.BF16: 0.0,
            Precision.FP8: 0.5,
            Precision.AWQ_INT4: 1.5,
            Precision.GPTQ_INT4: 1.8,
        }
    )
    peak_memory_gb: dict[Precision, float] = Field(
        default_factory=lambda: {
            Precision.BF16: 20.0,
            Precision.FP8: 12.0,
            Precision.AWQ_INT4: 8.0,
            Precision.GPTQ_INT4: 8.0,
        }
    )


class FakeEngine(EngineAdapter):
    name = "fake"

    def __init__(self, behavior: FakeBehavior | None = None) -> None:
        self.behavior = behavior or FakeBehavior()
        self._config: Config | None = None

    def supports(self, hardware: HardwareProfile, model: str, precision: Precision) -> SupportResult:
        if precision not in self.behavior.speed_factor:
            return SupportResult(ok=False, reason=f"fake engine does not support {precision.value}")
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        if config.precision not in self.behavior.speed_factor:
            raise ValueError(f"unsupported precision: {config.precision.value}")
        self._config = config

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        if self._config is None:
            raise RuntimeError("call prepare() before generate()")
        b = self.behavior
        factor = b.speed_factor[self._config.precision]
        step = b.tpot_ms * factor
        results: list[GenResult] = []
        for req in requests:
            ttft = (b.ttft_base_ms + b.prefill_ms_per_token * req.prompt_tokens) * factor
            times = [round(ttft + i * step, 6) for i in range(req.max_new_tokens)]
            results.append(
                GenResult(
                    request_id=req.request_id,
                    token_times_ms=times,
                    text="token " * req.max_new_tokens,
                )
            )
        return results

    def teardown(self) -> None:
        self._config = None

    # Helpers for tests. Real engines will not have these.
    def accuracy_drop_pct(self, precision: Precision) -> float:
        return self.behavior.accuracy_drop_pct[precision]

    def peak_memory_gb(self, precision: Precision) -> float:
        return self.behavior.peak_memory_gb[precision]
