from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from llmopt.schemas import Config, HardwareProfile, Precision


class SupportResult(BaseModel):
    ok: bool
    reason: str = ""


class GenRequest(BaseModel):
    request_id: str
    prompt_tokens: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)


class GenResult(BaseModel):
    request_id: str
    token_times_ms: list[float]
    text: str = ""

    @property
    def ttft_ms(self) -> float:
        return self.token_times_ms[0]

    @property
    def total_ms(self) -> float:
        return self.token_times_ms[-1]


class EngineAdapter(ABC):
    name: str
    supports_concurrency: bool = False

    @abstractmethod
    def supports(self, hardware: HardwareProfile, model: str, precision: Precision) -> SupportResult:
        ...

    @abstractmethod
    def prepare(self, config: Config) -> None:
        ...

    @abstractmethod
    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        ...

    def generate_batch(self, requests: list[GenRequest]) -> list[GenResult]:
        return self.generate(requests)

    @abstractmethod
    def teardown(self) -> None:
        ...
