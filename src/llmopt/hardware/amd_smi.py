"""Parse the text printed by `amd-smi static` and by plain `amd-smi`."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_GPU_HEADER = re.compile(r"^GPU: (\d+)\s*$")
_FIELD = re.compile(r"^( +)([A-Za-z0-9_ ]+?):\s*(.*)$")
_VRAM_MB = re.compile(r"^(\d+)\s*MB$")

_AMD_SMI_VERSION = re.compile(r"AMD-SMI\s+(\d[\w.+-]*)")
_AMDGPU_VERSION = re.compile(r"amdgpu version:\s+(\S+)", re.IGNORECASE)
_ROCM_VERSION = re.compile(r"ROCm version:\s+(\S+)", re.IGNORECASE)
_PARTITION_ROW = re.compile(
    r"^\|\s+(\d+)\s+(\d+)\s+(\d+)\s+([A-Z0-9]+)/([A-Z0-9]+)\s+\|", re.MULTILINE
)


class AmdSmiGpu(BaseModel):
    """Static facts about one GPU from `amd-smi static`."""

    gpu_index: int = Field(ge=0)
    market_name: str
    gfx_target: str
    compute_units: int = Field(gt=0)
    driver_version: str
    vram_mb: int = Field(gt=0)


class GpuPartition(BaseModel):
    """Compute and memory partition mode of one GPU (for example SPX and NPS1)."""

    gpu_index: int = Field(ge=0)
    compute_mode: str
    memory_mode: str


class AmdSmiSummary(BaseModel):
    """Versions and partition modes from the default `amd-smi` view."""

    amd_smi_version: str
    amdgpu_version: str
    rocm_version: str
    partitions: list[GpuPartition] = Field(default_factory=list)


def _collect_gpu_blocks(text: str) -> dict[int, dict[tuple[str, str], str]]:
    """Map GPU index to {(section, key): value} for the fields at the usual depth."""
    blocks: dict[int, dict[tuple[str, str], str]] = {}
    current: dict[tuple[str, str], str] | None = None
    section = ""
    for line in text.splitlines():
        header = _GPU_HEADER.match(line)
        if header:
            current = blocks.setdefault(int(header.group(1)), {})
            section = ""
            continue
        if line and not line.startswith(" "):
            current = None
            continue
        if current is None:
            continue
        match = _FIELD.match(line)
        if match is None:
            continue
        indent = len(match.group(1))
        key, value = match.group(2).strip(), match.group(3).strip()
        if indent == 4:
            section = "" if value else key
        elif indent == 8 and value:
            current.setdefault((section, key), value)
    return blocks


def _need(fields: dict[tuple[str, str], str], section: str, key: str, gpu: int) -> str:
    value = fields.get((section, key))
    if not value:
        raise ValueError(f"amd-smi static: GPU {gpu} has no {section}.{key} line")
    return value


def parse_amd_smi_static(text: str) -> list[AmdSmiGpu]:
    """Parse `amd-smi static` output into one record per GPU."""
    blocks = _collect_gpu_blocks(text)
    if not blocks:
        raise ValueError("amd-smi static output has no 'GPU: N' sections")
    gpus: list[AmdSmiGpu] = []
    for index in sorted(blocks):
        fields = blocks[index]
        vram = _VRAM_MB.match(_need(fields, "VRAM", "SIZE", index))
        if vram is None:
            raise ValueError(f"amd-smi static: GPU {index} VRAM size is not given in MB")
        gpus.append(
            AmdSmiGpu(
                gpu_index=index,
                market_name=_need(fields, "ASIC", "MARKET_NAME", index),
                gfx_target=_need(fields, "ASIC", "TARGET_GRAPHICS_VERSION", index),
                compute_units=int(_need(fields, "ASIC", "NUM_COMPUTE_UNITS", index)),
                driver_version=_need(fields, "DRIVER", "VERSION", index),
                vram_mb=int(vram.group(1)),
            )
        )
    return gpus


def _find(pattern: re.Pattern[str], text: str, name: str) -> str:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"amd-smi output has no {name} line")
    return match.group(1)


def parse_amd_smi_default(text: str) -> AmdSmiSummary:
    """Parse the default `amd-smi` view: version header and partition modes."""
    partitions = [
        GpuPartition(
            gpu_index=int(row.group(1)),
            compute_mode=row.group(4),
            memory_mode=row.group(5),
        )
        for row in _PARTITION_ROW.finditer(text)
    ]
    return AmdSmiSummary(
        amd_smi_version=_find(_AMD_SMI_VERSION, text, "AMD-SMI version"),
        amdgpu_version=_find(_AMDGPU_VERSION, text, "amdgpu version"),
        rocm_version=_find(_ROCM_VERSION, text, "ROCm version"),
        partitions=partitions,
    )
