from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.engines.base import EngineAdapter, GenRequest, GenResult, SupportResult
from llmopt.profile.factory import make_config
from llmopt.profile.sweep import (
    SweepResult,
    format_sweep,
    parse_levels,
    requests_for_level,
    run_sweep,
    save_sweep,
)
from llmopt.schemas import Config, HardwareProfile, Precision, Status, Workload

ONE_GIB = 2**30
TIMES = [250.0, 500.0, 750.0, 1000.0]
CONFIG = make_config("vllm", Path("/m"), {})
WORKLOAD = Workload(name="w", prompt_tokens=32, output_tokens=4, num_requests=8)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class WaveEngine(EngineAdapter):
    """Pretend engine, not a real one: every request takes 1000 ms, a call takes
    1 s when requests overlap and 1 s per request when they do not."""

    name = "wave"

    def __init__(self, clock: FakeClock, concurrent: bool = True, fail: bool = False) -> None:
        self.clock = clock
        self.supports_concurrency = concurrent
        self.fail = fail

    def supports(
        self, hardware: HardwareProfile, model: str, precision: Precision
    ) -> SupportResult:
        return SupportResult(ok=True)

    def prepare(self, config: Config) -> None:
        return None

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        if self.fail:
            raise RuntimeError("engine crashed")
        self.clock.now += 1.0 if self.supports_concurrency else float(len(requests))
        return [
            GenResult(request_id=r.request_id, token_times_ms=list(TIMES)) for r in requests
        ]

    def teardown(self) -> None:
        return None


def rates(result: SweepResult) -> list[float]:
    values: list[float] = []
    for row in result.rows:
        assert row.trial.metrics is not None
        values.append(row.trial.metrics.throughput_tps)
    return values


def sweep(clock: FakeClock, levels: list[int], **engine_options: bool) -> SweepResult:
    return run_sweep(
        lambda: WaveEngine(clock, **engine_options),
        CONFIG,
        WORKLOAD,
        levels,
        runs=2,
        read_memory_bytes=lambda: ONE_GIB,
        clock=clock,
    )


def test_parse_levels_accepts_increasing_numbers() -> None:
    assert parse_levels("1, 2,4") == [1, 2, 4]


@pytest.mark.parametrize("text", ["", "a,b", "0,1", "2,1", "1,1", "1,-2"])
def test_parse_levels_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_levels(text)


def test_requests_for_level_gives_whole_waves() -> None:
    assert requests_for_level(8, 1) == 8
    assert requests_for_level(8, 4) == 8
    assert requests_for_level(8, 16) == 16
    assert requests_for_level(6, 4) == 8
    assert requests_for_level(8, 128) == 128


def test_run_sweep_measures_each_level() -> None:
    result = sweep(FakeClock(), [1, 2, 4])
    assert [row.concurrency for row in result.rows] == [1, 2, 4]
    assert [row.num_requests for row in result.rows] == [8, 8, 8]
    assert rates(result) == pytest.approx([4.0, 8.0, 16.0])


def test_run_sweep_rejects_sequential_engine_for_levels_above_one() -> None:
    with pytest.raises(ValueError, match="one at a time"):
        sweep(FakeClock(), [1, 2], concurrent=False)


def test_run_sweep_stops_after_a_failed_level() -> None:
    result = sweep(FakeClock(), [1, 2, 4], fail=True)
    assert len(result.rows) == 1
    assert result.rows[0].trial.status is Status.FAILED


def test_format_sweep_has_one_line_per_level() -> None:
    text = format_sweep(sweep(FakeClock(), [1, 2, 4]))
    assert "tok/s" in text
    assert len(text.splitlines()) == 4


def test_save_sweep_round_trips(tmp_path: Path) -> None:
    result = sweep(FakeClock(), [1, 2])
    path = tmp_path / "deep" / "s.json"
    save_sweep(result, path)
    loaded = SweepResult.model_validate_json(path.read_text())
    assert [row.concurrency for row in loaded.rows] == [1, 2]
    assert loaded.workload == "w"