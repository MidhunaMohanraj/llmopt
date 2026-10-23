from __future__ import annotations

from pathlib import Path

import pytest

from llmopt.profile.factory import make_config, make_engine, parse_settings
from llmopt.schemas import Engine, Precision


def test_parse_settings_empty() -> None:
    assert parse_settings("") == {}
    assert parse_settings(" , ") == {}


def test_parse_settings_types() -> None:
    result = parse_settings("max_model_len=2048,ratio=0.5,eager=true,name=abc")
    assert result == {"max_model_len": 2048, "ratio": 0.5, "eager": True, "name": "abc"}
    assert isinstance(result["max_model_len"], int)
    assert result["eager"] is True


@pytest.mark.parametrize("text", ["novalue", "=1"])
def test_parse_settings_rejects_bad_part(text: str) -> None:
    with pytest.raises(ValueError):
        parse_settings(text)


def test_make_config() -> None:
    config = make_config("vllm", Path("/m"), {"a": 1})
    assert config.engine is Engine.VLLM
    assert config.precision is Precision.BF16
    assert config.model == "/m"
    assert config.settings == {"a": 1}


def test_make_config_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        make_config("nope", Path("/m"), {})


def test_make_engine_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        make_engine("nope")