import os

import pytest

from llmopt.settings.apply import apply_settings, restore_env, set_env

ALLOWED = frozenset(
    {
        "max_num_seqs",
        "enable_chunked_prefill",
        "aiter",
        "hipblaslt_preference",
        "cudagraph_mode",
        "max_model_len",
        "calculate_kv_scales",
    }
)


def test_arguments_are_applied() -> None:
    applied = apply_settings(
        {"max_num_seqs": 256, "enable_chunked_prefill": False}, ALLOWED, "demo"
    )
    assert applied.kwargs == {"max_num_seqs": 256, "enable_chunked_prefill": False}
    assert applied.env == {}


def test_environment_settings_become_variables() -> None:
    applied = apply_settings({"aiter": True, "hipblaslt_preference": False}, ALLOWED, "demo")
    assert applied.kwargs == {}
    assert applied.env == {"VLLM_ROCM_USE_AITER": "1", "TORCH_BLAS_PREFER_HIPBLASLT": "0"}


def test_graph_mode_goes_into_the_compilation_config() -> None:
    applied = apply_settings({"cudagraph_mode": "FULL"}, ALLOWED, "demo")
    assert applied.kwargs == {"compilation_config": {"cudagraph_mode": "FULL"}}


def test_adapter_specific_names_are_left_to_the_adapter() -> None:
    applied = apply_settings({"max_model_len": 4096}, ALLOWED, "demo")
    assert applied.kwargs == {}
    assert applied.env == {}


def test_a_setting_the_adapter_does_not_apply_is_an_error() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        apply_settings({"aiter": True}, frozenset(), "demo")


def test_a_setting_missing_from_the_pinned_vllm_is_refused() -> None:
    with pytest.raises(ValueError, match="calculate_kv_scales"):
        apply_settings({"calculate_kv_scales": True}, ALLOWED, "demo")


def test_prefix_caching_and_kv_dtype_are_not_tunable_settings() -> None:
    allowed = ALLOWED | {"enable_prefix_caching", "kv_cache_dtype"}
    with pytest.raises(ValueError, match="prefix caching"):
        apply_settings({"enable_prefix_caching": True}, allowed, "demo")
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        apply_settings({"kv_cache_dtype": "fp8"}, allowed, "demo")


def test_out_of_range_value_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least"):
        apply_settings({"max_num_seqs": 0}, ALLOWED, "demo")


def test_set_and_restore_a_new_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMOPT_TEST_VAR", raising=False)
    previous = set_env({"LLMOPT_TEST_VAR": "1"})
    assert os.environ["LLMOPT_TEST_VAR"] == "1"
    restore_env(previous)
    assert "LLMOPT_TEST_VAR" not in os.environ


def test_restore_puts_back_an_existing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOPT_TEST_VAR", "old")
    previous = set_env({"LLMOPT_TEST_VAR": "new"})
    assert os.environ["LLMOPT_TEST_VAR"] == "new"
    restore_env(previous)
    assert os.environ["LLMOPT_TEST_VAR"] == "old"
