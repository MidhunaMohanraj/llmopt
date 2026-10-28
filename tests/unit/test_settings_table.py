import json

import pytest

from llmopt.settings.table import BY_NAME, SETTINGS, render_settings, validate_settings


def test_names_and_flags_are_unique() -> None:
    assert len({s.name for s in SETTINGS}) == len(SETTINGS)
    assert len({s.flag for s in SETTINGS}) == len(SETTINGS)


def test_defaults_are_inside_their_own_limits() -> None:
    for setting in SETTINGS:
        if setting.default is None:
            continue
        assert validate_settings({setting.name: setting.default}) == [], setting.name


def test_no_multi_gpu_settings() -> None:
    for setting in SETTINGS:
        assert "parallel" not in setting.name
        assert "rccl" not in setting.name.lower()


def test_nothing_is_marked_verified_yet() -> None:
    assert not any(s.verified for s in SETTINGS)


def test_valid_settings_have_no_errors() -> None:
    values = {"gpu_memory_utilization": 0.92, "max_num_seqs": 256, "kv_cache_dtype": "fp8"}
    assert validate_settings(values) == []


@pytest.mark.parametrize(
    "values, fragment",
    [
        ({"nonsense": 1}, "unknown setting"),
        ({"gpu_memory_utilization": 1.5}, "at most"),
        ({"gpu_memory_utilization": 0.0}, "at least"),
        ({"max_num_seqs": 1.5}, "whole number"),
        ({"max_num_seqs": True}, "must be a number"),
        ({"max_num_seqs": 0}, "at least"),
        ({"kv_cache_dtype": "int2"}, "must be one of"),
        ({"aiter": "yes"}, "true or false"),
        ({"cudagraph_mode": "SOMETIMES"}, "must be one of"),
    ],
)
def test_bad_values_are_rejected(values: dict[str, object], fragment: str) -> None:
    errors = validate_settings(values)  # type: ignore[arg-type]
    assert errors
    assert fragment in " ".join(errors)


def test_render_args_and_env() -> None:
    out = render_settings(
        {
            "max_num_seqs": 256,
            "kv_cache_dtype": "fp8",
            "enable_chunked_prefill": False,
            "calculate_kv_scales": False,
            "aiter": True,
            "hipblaslt_preference": False,
        }
    )
    assert out.args == [
        "--max-num-seqs",
        "256",
        "--no-enable-chunked-prefill",
        "--kv-cache-dtype",
        "fp8",
    ]
    assert out.env == {"VLLM_ROCM_USE_AITER": "1", "TORCH_BLAS_PREFER_HIPBLASLT": "0"}


def test_render_compilation_config_is_json() -> None:
    out = render_settings({"cudagraph_mode": "FULL"})
    assert out.args[0] == "--compilation-config"
    assert json.loads(out.args[1]) == {"cudagraph_mode": "FULL"}


def test_render_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="at most"):
        render_settings({"gpu_memory_utilization": 2.0})


def test_table_lookup() -> None:
    assert BY_NAME["aiter"].flag == "VLLM_ROCM_USE_AITER"