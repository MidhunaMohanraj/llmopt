from pathlib import Path

import pytest

from llmopt.precision.selector import config_dir


def test_default_is_the_repo_configs_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMOPT_CONFIG_DIR", raising=False)
    assert (config_dir() / "gpus.yaml").is_file()


def test_environment_variable_overrides_the_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLMOPT_CONFIG_DIR", str(tmp_path))
    assert config_dir() == tmp_path