from pathlib import Path

import pytest

from llmopt.profile.factory import make_config, parse_precision
from llmopt.schemas import Precision

MODEL = Path("/models/example")


def test_defaults_are_bf16_and_auto_kv() -> None:
    cfg = make_config("vllm", MODEL, {})
    assert cfg.precision is Precision.BF16
    assert cfg.kv_cache_dtype == "auto"


def test_fp8_with_fp8_kv_cache() -> None:
    cfg = make_config("vllm", MODEL, {}, precision="fp8", kv_cache_dtype="fp8")
    assert cfg.precision is Precision.FP8
    assert cfg.kv_cache_dtype == "fp8"


def test_unknown_precision_names_the_valid_choices() -> None:
    with pytest.raises(ValueError, match="precision must be one of"):
        parse_precision("fp3")


def test_unknown_kv_cache_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        make_config("vllm", MODEL, {}, kv_cache_dtype="int2")