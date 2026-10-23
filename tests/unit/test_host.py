from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmopt.cli import app
from llmopt.hardware.host import build_host_profile

FIXTURES = Path(__file__).parent.parent / "fixtures" / "host"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_ryzen_7950x_profile() -> None:
    profile = build_host_profile(
        lscpu_text=_read("lscpu_ryzen_7950x.txt"),
        meminfo_text=_read("meminfo_head.txt"),
        os_release_text=_read("os_release_ubuntu_2404.txt"),
        kernel="7.0.0-38-generic",
    )
    assert profile.cpu_model == "AMD Ryzen 9 7950X 16-Core Processor"
    assert profile.cpu_cores == 16
    assert profile.cpu_threads == 32
    assert profile.sockets == 1
    assert profile.numa_nodes == 1
    assert profile.memory_bytes == 65017616 * 1024
    assert profile.os_name == "Ubuntu 24.04.4 LTS"
    assert profile.avx512 is True
    assert profile.avx512_bf16 is True
    assert profile.environment == "native"


def test_missing_lscpu_line_is_an_error() -> None:
    with pytest.raises(ValueError, match="lscpu"):
        build_host_profile(
            lscpu_text="CPU(s): 4\n",
            meminfo_text="MemTotal: 1 kB\n",
            os_release_text='PRETTY_NAME="X"\n',
            kernel="k",
        )


def test_detect_command_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "profile.json"
    result = CliRunner().invoke(app, ["detect", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
