"""Measure the memory of a process and of the processes it started."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path


def _children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("PPid:"):
                    children.setdefault(int(line.split()[1]), []).append(int(entry.name))
                    break
        except OSError:
            continue
    return children


def _memory_bytes(pid: int) -> int:
    for name, key in (("smaps_rollup", "Pss:"), ("status", "VmRSS:")):
        try:
            for line in Path(f"/proc/{pid}/{name}").read_text().splitlines():
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
        except OSError:
            continue
    return 0


def process_tree_memory_bytes(root_pid: int | None = None) -> int:
    """Memory of a process plus all its descendants, in bytes (Linux only).

    Uses proportional set size (PSS) so memory shared between processes is not
    counted twice. If PSS cannot be read, it falls back to resident size (RSS).
    """
    root = os.getpid() if root_pid is None else root_pid
    children = _children_map()
    stack = [root]
    seen: set[int] = set()
    total = 0
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += _memory_bytes(pid)
        stack.extend(children.get(pid, []))
    return total


class PeakMemoryMonitor:
    """Context manager that samples memory in a thread and keeps the highest value."""

    def __init__(self, read_bytes: Callable[[], int], interval_s: float = 0.2) -> None:
        self._read_bytes = read_bytes
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self.peak_bytes = max(self.peak_bytes, self._read_bytes())

    def __enter__(self) -> PeakMemoryMonitor:
        self.peak_bytes = self._read_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._read_bytes())