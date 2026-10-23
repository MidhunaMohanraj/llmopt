from pathlib import Path

import pytest

from llmopt.hardware.rocminfo import parse_rocminfo

FIXTURES = Path(__file__).parent.parent / "fixtures" / "gpu"

# The strings below are small synthetic examples, not AMD output.
CPU_ONLY = """\
Agent 1
  Name:                    Some CPU
  Device Type:             CPU
  Compute Unit:            8
"""

TWO_GPUS = """\
Agent 1
  Name:                    gfx942
  Marketing Name:          Synthetic GPU A
  Device Type:             GPU
  Max Clock Freq. (MHz):   2100
  Compute Unit:            38
Agent 2
  Name:                    gfx942
  Marketing Name:          Synthetic GPU B
  Device Type:             GPU
  Compute Unit:            38
"""

GPU_WITHOUT_COMPUTE_UNIT = """\
Agent 1
  Name:                    gfx942
  Marketing Name:          Synthetic GPU
  Device Type:             GPU
"""


def test_docs_sample_has_one_mi300x() -> None:
    text = (FIXTURES / "rocminfo_mi300x_docs.txt").read_text()
    agents = parse_rocminfo(text)
    assert len(agents) == 1
    gpu = agents[0]
    assert gpu.agent_index == 2
    assert gpu.marketing_name == "AMD Instinct MI300X"
    assert gpu.gfx_target == "gfx941"
    assert gpu.compute_units == 304
    assert gpu.max_clock_mhz == 1800


def test_cpu_only_machine_has_no_gpus() -> None:
    assert parse_rocminfo(CPU_ONLY) == []


def test_several_gpus_keep_their_order() -> None:
    agents = parse_rocminfo(TWO_GPUS)
    assert [a.marketing_name for a in agents] == ["Synthetic GPU A", "Synthetic GPU B"]
    assert agents[0].max_clock_mhz == 2100
    assert agents[1].max_clock_mhz is None


def test_text_without_agents_is_an_error() -> None:
    with pytest.raises(ValueError, match="Agent"):
        parse_rocminfo("this is not rocminfo output")


def test_gpu_without_compute_units_is_an_error() -> None:
    with pytest.raises(ValueError, match="Compute Unit"):
        parse_rocminfo(GPU_WITHOUT_COMPUTE_UNIT)
