
from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field


class HostProfile(BaseModel):


    cpu_model: str
    cpu_cores: int = Field(gt=0)
    cpu_threads: int = Field(gt=0)
    sockets: int = Field(gt=0)
    numa_nodes: int = Field(gt=0)
    memory_bytes: int = Field(gt=0)
    os_name: str
    kernel: str
    avx512: bool
    avx512_bf16: bool
    environment: str = "native"


def _parse_colon_lines(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields.setdefault(key.strip(), value.strip())
    return fields


def _require(fields: dict[str, str], key: str) -> str:
    value = fields.get(key)
    if not value:
        raise ValueError(f"lscpu output has no '{key}' line")
    return value


def _parse_os_name(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    raise ValueError("os-release has no PRETTY_NAME line")


def _parse_mem_total_bytes(text: str) -> int:
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    if match is None:
        raise ValueError("meminfo has no MemTotal line")
    return int(match.group(1)) * 1024


def build_host_profile(
    lscpu_text: str,
    meminfo_text: str,
    os_release_text: str,
    kernel: str,
    environment: str = "native",
) -> HostProfile:
    
    lscpu = _parse_colon_lines(lscpu_text)
    sockets = int(_require(lscpu, "Socket(s)"))
    flags = set(lscpu.get("Flags", "").split())
    return HostProfile(
        cpu_model=_require(lscpu, "Model name"),
        cpu_cores=int(_require(lscpu, "Core(s) per socket")) * sockets,
        cpu_threads=int(_require(lscpu, "CPU(s)")),
        sockets=sockets,
        numa_nodes=int(_require(lscpu, "NUMA node(s)")),
        memory_bytes=_parse_mem_total_bytes(meminfo_text),
        os_name=_parse_os_name(os_release_text),
        kernel=kernel,
        avx512="avx512f" in flags,
        avx512_bf16="avx512_bf16" in flags,
        environment=environment,
    )


def detect_host() -> HostProfile:

    lscpu = subprocess.run(
        ["lscpu"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout
    meminfo = Path("/proc/meminfo").read_text()
    os_release = Path("/etc/os-release").read_text()
    kernel = platform.release()
    environment = "wsl2" if "microsoft" in kernel.lower() else "native"
    return build_host_profile(lscpu, meminfo, os_release, kernel, environment)
