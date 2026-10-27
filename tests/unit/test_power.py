from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llmopt.profile.power import (
    PowerSampler,
    energy_per_token_j,
    integrate_joules,
    parse_power_watts,
    read_gpu_power_watts,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "gpu" / "amd_smi_default_docs.txt"
)


def fake_run(text: str) -> Callable[..., Any]:
    def run(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(stdout=text)

    return run


def failing_after_first() -> Callable[[], float]:
    calls = {"count": 0}

    def read() -> float:
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("gpu gone")
        return 100.0

    return read


def test_parses_every_gpu_row_of_the_docs_sample() -> None:
    assert parse_power_watts(FIXTURE.read_text()) == [49.0, 42.0, 33.0, 40.0]


def test_text_without_gpu_rows_gives_no_readings() -> None:
    assert parse_power_watts("nothing useful here\n| 0 | 5 W |\n") == []


def test_integrate_constant_power() -> None:
    assert integrate_joules([(0.0, 100.0), (2.0, 100.0)]) == pytest.approx(200.0)


def test_integrate_a_ramp_uses_the_trapezoid_rule() -> None:
    assert integrate_joules([(0.0, 0.0), (2.0, 100.0)]) == pytest.approx(100.0)


def test_integrate_needs_two_samples() -> None:
    with pytest.raises(ValueError):
        integrate_joules([(0.0, 100.0)])


def test_integrate_rejects_time_going_backwards() -> None:
    with pytest.raises(ValueError):
        integrate_joules([(2.0, 100.0), (1.0, 100.0)])


def test_energy_per_token() -> None:
    assert energy_per_token_j(200.0, 100) == pytest.approx(2.0)


def test_energy_per_token_rejects_zero_tokens() -> None:
    with pytest.raises(ValueError):
        energy_per_token_j(200.0, 0)


def test_sampler_integrates_what_it_read() -> None:
    with PowerSampler(lambda: 100.0, interval_s=0.01) as sampler:
        time.sleep(0.05)
    assert len(sampler.samples) >= 2
    span = sampler.samples[-1][0] - sampler.samples[0][0]
    assert sampler.energy_joules == pytest.approx(100.0 * span)


def test_sampler_reports_a_reader_failure_in_the_thread() -> None:
    with PowerSampler(failing_after_first(), interval_s=0.01) as sampler:
        time.sleep(0.05)
    with pytest.raises(RuntimeError, match="power reading failed"):
        _ = sampler.energy_joules


def test_sampler_fails_at_once_if_the_first_read_fails() -> None:
    def bad() -> float:
        raise RuntimeError("no amd-smi")

    with pytest.raises(RuntimeError, match="no amd-smi"), PowerSampler(bad):
        pass


def test_reader_sums_all_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llmopt.profile.power.subprocess.run", fake_run(FIXTURE.read_text()))
    assert read_gpu_power_watts() == pytest.approx(164.0)


def test_reader_can_pick_one_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llmopt.profile.power.subprocess.run", fake_run(FIXTURE.read_text()))
    assert read_gpu_power_watts(gpu_index=1) == pytest.approx(42.0)


def test_reader_rejects_a_bad_gpu_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llmopt.profile.power.subprocess.run", fake_run(FIXTURE.read_text()))
    with pytest.raises(ValueError):
        read_gpu_power_watts(gpu_index=4)


def test_reader_fails_when_output_has_no_power(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llmopt.profile.power.subprocess.run", fake_run("nothing\n"))
    with pytest.raises(RuntimeError, match="no power readings"):
        read_gpu_power_watts()


def test_reader_fails_when_amd_smi_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("amd-smi")

    monkeypatch.setattr("llmopt.profile.power.subprocess.run", missing)
    with pytest.raises(RuntimeError, match="amd-smi not found"):
        read_gpu_power_watts()