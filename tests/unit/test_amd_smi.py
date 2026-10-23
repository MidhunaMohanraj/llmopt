from pathlib import Path

import pytest

from llmopt.hardware.amd_smi import parse_amd_smi_default, parse_amd_smi_static

FIXTURES = Path(__file__).parent.parent / "fixtures" / "gpu"

# Synthetic line modeled on the one-line header of an older amd-smi release.
ONE_LINE_HEADER = (
    "| AMD-SMI 26.0.0+eaa54ecc      amdgpu version: 6.12.12  ROCm version: 7.0.0    |\n"
)


def test_static_docs_sample() -> None:
    gpus = parse_amd_smi_static((FIXTURES / "amd_smi_static_docs.txt").read_text())
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.gpu_index == 0
    assert gpu.market_name == "AMD Instinct MI300A"
    assert gpu.gfx_target == "gfx942"
    assert gpu.compute_units == 228
    assert gpu.driver_version == "6.19.4"
    assert gpu.vram_mb == 96432


def test_static_text_without_gpus_is_an_error() -> None:
    with pytest.raises(ValueError, match="GPU"):
        parse_amd_smi_static("CPU: 0\n    SMU:\n        FW_VERSION: 1\n")


def test_static_missing_vram_is_an_error() -> None:
    text = "GPU: 0\n    ASIC:\n        MARKET_NAME: Synthetic\n"
    with pytest.raises(ValueError, match="VRAM"):
        parse_amd_smi_static(text)


def test_default_view_docs_sample() -> None:
    summary = parse_amd_smi_default((FIXTURES / "amd_smi_default_docs.txt").read_text())
    assert summary.amd_smi_version == "27.0.0"
    assert summary.amdgpu_version == "6.19.4"
    assert summary.rocm_version == "7.14.0"
    assert [p.gpu_index for p in summary.partitions] == [0, 1, 2, 3]
    assert {(p.compute_mode, p.memory_mode) for p in summary.partitions} == {("SPX", "NPS1")}


def test_default_view_one_line_header() -> None:
    summary = parse_amd_smi_default(ONE_LINE_HEADER)
    assert summary.amd_smi_version == "26.0.0+eaa54ecc"
    assert summary.amdgpu_version == "6.12.12"
    assert summary.rocm_version == "7.0.0"
    assert summary.partitions == []


def test_default_view_without_versions_is_an_error() -> None:
    with pytest.raises(ValueError, match="AMD-SMI"):
        parse_amd_smi_default("nothing useful here")
