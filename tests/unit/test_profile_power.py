import pytest

from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.profile.profiler import profile_engine
from llmopt.schemas import Config, HardwareProfile, Precision


class FixedEngine(EngineAdapter):
    name = "fixed"

    def supports(
        self, hardware: HardwareProfile, model: str, precision: Precision
    ) -> SupportResult:
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        return None

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        return [
            GenResult(request_id=r.request_id, token_times_ms=[10.0, 20.0, 30.0])
            for r in requests
        ]

    def teardown(self) -> None:
        return None


REQUESTS = [GenRequest(request_id="r1", prompt_tokens=32, max_new_tokens=3)]


def _memory() -> int:
    return 3 * 2**30


def _hundred_watts() -> float:
    return 100.0


def _broken() -> float:
    raise RuntimeError("amd-smi not found")


def test_energy_is_filled_when_a_power_reader_is_given() -> None:
    result = profile_engine(
        FixedEngine(),
        REQUESTS,
        runs=2,
        read_memory_bytes=_memory,
        read_power_watts=_hundred_watts,
    )
    energy = result.metrics.energy_per_token_j
    assert energy is not None
    assert energy > 0


def test_energy_stays_empty_without_a_power_reader() -> None:
    result = profile_engine(FixedEngine(), REQUESTS, runs=2, read_memory_bytes=_memory)
    assert result.metrics.energy_per_token_j is None


def test_a_failing_power_reader_raises() -> None:
    with pytest.raises(RuntimeError, match="amd-smi"):
        profile_engine(
            FixedEngine(),
            REQUESTS,
            runs=2,
            read_memory_bytes=_memory,
            read_power_watts=_broken,
        )