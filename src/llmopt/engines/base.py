"""The contract every engine adapter follows."""

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
    # Time of each output token, in ms since the request started.
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

    @abstractmethod
    def supports(self, hardware: HardwareProfile, model: str, precision: Precision) -> SupportResult:
        """Can this engine run this case? If not, say why."""

    @abstractmethod
    def prepare(self, config: Config) -> None:
        """Start the engine with the given settings."""

    @abstractmethod
    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        """Run the requests and return per-token timings and outputs."""

    @abstractmethod
    def teardown(self) -> None:
        """Stop the engine and free GPU memory."""
