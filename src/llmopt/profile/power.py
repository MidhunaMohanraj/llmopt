"""Measure GPU energy by sampling power (watts) and integrating over time.

DRAFT: NOT TESTED ON A GPU. The parser follows the AMD docs sample of the default
`amd-smi` table (an MI300A), and the sampling interval has not been measured. Confirm
both on a real MI300X before using any number from this module.

The reading is whole-GPU power, idle draw included. Energy per token from it is
"GPU energy for this run divided by tokens made", not the share of the model alone.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections.abc import Callable

from typing_extensions import Self

# One row per GPU in the default view: starts with the PCI address, ends with "49 W |".
_POWER_ROW = re.compile(
    r"^\|\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d\s.*\s(\d+(?:\.\d+)?)\s*W\s*\|\s*$",
    re.MULTILINE,
)


def parse_power_watts(text: str) -> list[float]:
    """Power in watts for each GPU row of the default `amd-smi` table, in table order."""
    return [float(value) for value in _POWER_ROW.findall(text)]


def integrate_joules(samples: list[tuple[float, float]]) -> float:
    """Trapezoid rule over (time in seconds, power in watts) samples."""
    if len(samples) < 2:
        raise ValueError("need at least 2 samples to integrate")
    total = 0.0
    for index in range(1, len(samples)):
        t0, w0 = samples[index - 1]
        t1, w1 = samples[index]
        if t1 < t0:
            raise ValueError("sample times must not go backwards")
        total += (w0 + w1) / 2 * (t1 - t0)
    return total


def energy_per_token_j(joules: float, tokens: int) -> float:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    return joules / tokens


def read_gpu_power_watts(gpu_index: int | None = None, timeout_s: float = 10.0) -> float:
    """Run `amd-smi` once. Returns the sum over all GPUs, or one GPU if gpu_index is given."""
    try:
        completed = subprocess.run(
            ["amd-smi"], capture_output=True, text=True, check=True, timeout=timeout_s
        )
    except FileNotFoundError as error:
        raise RuntimeError("amd-smi not found") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"amd-smi failed with exit code {error.returncode}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"amd-smi did not answer within {timeout_s} s") from error
    watts = parse_power_watts(completed.stdout)
    if not watts:
        raise RuntimeError("no power readings found in the amd-smi output")
    if gpu_index is None:
        return sum(watts)
    if not 0 <= gpu_index < len(watts):
        raise ValueError(f"gpu_index must be between 0 and {len(watts) - 1}, got {gpu_index}")
    return watts[gpu_index]


class PowerSampler:
    """Context manager that reads power in a thread and integrates it into joules."""

    def __init__(self, read_watts: Callable[[], float], interval_s: float = 0.5) -> None:
        self._read_watts = read_watts
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float]] = []
        self.error: Exception | None = None

    def _sample(self) -> None:
        self.samples.append((time.perf_counter(), self._read_watts()))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._sample()
            except (RuntimeError, OSError) as error:
                self.error = error
                return

    def __enter__(self) -> Self:
        self.samples = []
        self.error = None
        self._sample()  # a failure here is raised at once, on the caller's thread
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.error is None:
            self._sample()

    @property
    def energy_joules(self) -> float:
        if self.error is not None:
            raise RuntimeError(f"power reading failed: {self.error}") from self.error
        return integrate_joules(self.samples)