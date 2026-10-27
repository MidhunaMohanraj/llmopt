from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.profile.workloads import get_workload, load_workloads


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "workloads.yaml"
    path.write_text(text)
    return path


def test_real_file_defines_smoke_chat_and_long() -> None:
    loaded = load_workloads()
    assert set(loaded) == {"smoke", "chat", "long"}
    smoke = loaded["smoke"]
    assert (smoke.prompt_tokens, smoke.output_tokens, smoke.num_requests) == (32, 16, 8)
    assert loaded["long"].prompt_tokens > loaded["chat"].prompt_tokens


def test_name_comes_from_the_key(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny:\n  prompt_tokens: 4\n  output_tokens: 2\n")
    assert load_workloads(path)["tiny"].name == "tiny"


def test_unknown_name_lists_the_valid_ones(tmp_path: Path) -> None:
    path = write(tmp_path, "a:\n  prompt_tokens: 4\n  output_tokens: 2\n")
    with pytest.raises(ValueError, match="choose one of: a"):
        get_workload("b", path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_workloads(write(tmp_path, ""))


def test_entry_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        load_workloads(write(tmp_path, "a: 5\n"))


def test_invalid_values_are_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "a:\n  prompt_tokens: 0\n  output_tokens: 2\n")
    with pytest.raises(ValueError):
        load_workloads(path)