from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from llmopt.schemas import SettingValue
from llmopt.settings.table import BY_NAME, validate_settings

NOT_IN_PINNED_VLLM: dict[str, str] = {
    "calculate_kv_scales": "vLLM 0.32.0 has no calculate_kv_scales argument",
}
SET_ELSEWHERE: dict[str, str] = {
    "enable_prefix_caching": (
        "the adapter keeps prefix caching off so repeated prompts do not flatter TTFT"
    ),
    "kv_cache_dtype": "use the kv_cache_dtype option of the config, not a setting",
}


class Applied(BaseModel):
    kwargs: dict[str, Any]
    env: dict[str, str]


def _env_value(value: SettingValue) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return str(value)


def apply_settings(
    values: dict[str, SettingValue], allowed: frozenset[str], engine_name: str
) -> Applied:
    problems = validate_settings({k: v for k, v in values.items() if k in BY_NAME})
    for name in values:
        if name in NOT_IN_PINNED_VLLM:
            problems.append(f"{name}: {NOT_IN_PINNED_VLLM[name]}")
        elif name in SET_ELSEWHERE:
            problems.append(f"{name}: {SET_ELSEWHERE[name]}")
        elif name not in allowed:
            problems.append(f"{name}: the {engine_name} adapter does not apply this setting")
    if problems:
        raise ValueError("; ".join(problems))
    kwargs: dict[str, Any] = {}
    env: dict[str, str] = {}
    compilation: dict[str, str] = {}
    for name, value in values.items():
        setting = BY_NAME.get(name)
        if setting is None:
            continue
        if setting.target == "env":
            env[setting.flag] = _env_value(value)
        elif setting.target == "compilation_config":
            compilation[setting.flag] = str(value)
        else:
            kwargs[name] = value
    if compilation:
        kwargs["compilation_config"] = compilation
    return Applied(kwargs=kwargs, env=env)


def set_env(env: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in env.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
