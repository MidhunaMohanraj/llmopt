"""Search over vLLM settings: one at a time, then combine the winners."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from llmopt.schemas import Config, Objective, SettingValue, Status, TrialResult
from llmopt.settings.table import BY_NAME, Setting, validate_settings

# What each optimize name means: the Metrics field, and whether higher is better.
METRICS: dict[str, tuple[str, bool]] = {
    "throughput": ("throughput_tps", True),
    "ttft": ("ttft_ms_p95", False),
    "tpot": ("tpot_ms", False),
    "latency": ("latency_ms_p95", False),
    "memory": ("peak_memory_gb", False),
}


class Evaluated(BaseModel):
    """One measured config. problem is None when it is usable under the objective."""

    settings: dict[str, SettingValue]
    trial: TrialResult
    problem: str | None = None

    @property
    def feasible(self) -> bool:
        return self.problem is None


class SearchResult(BaseModel):
    baseline: Evaluated
    best: Evaluated
    evaluated: list[Evaluated]
    front: list[Evaluated]
    stopped_early: bool


def feasibility_problem(trial: TrialResult, objective: Objective) -> str | None:
    """Why this trial cannot be used under the objective, or None if it can."""
    if trial.status is not Status.OK or trial.metrics is None:
        return f"trial {trial.status.value}: {trial.reason or 'no metrics'}"
    limit = objective.max_ttft_p95_ms
    if limit is not None and trial.metrics.ttft_ms_p95 > limit:
        return f"TTFT p95 {trial.metrics.ttft_ms_p95:.1f} ms is over the {limit:g} ms limit"
    max_drop = objective.max_accuracy_drop_pct
    if max_drop is not None:
        if not trial.accuracy:
            return "accuracy not measured"
        worst = max(item.drop_pct for item in trial.accuracy)
        if worst > max_drop:
            return f"accuracy drop {worst:.2f}% is over the {max_drop:g}% limit"
    return None


def _metric(name: str) -> tuple[str, bool]:
    if name not in METRICS:
        raise ValueError(
            f"unknown optimize target {name!r}; choose one of: {', '.join(METRICS)}"
        )
    return METRICS[name]


def _value(evaluated: Evaluated, name: str) -> float:
    metrics = evaluated.trial.metrics
    if metrics is None:
        raise ValueError("trial has no metrics")
    return float(getattr(metrics, _metric(name)[0]))


def _signed(evaluated: Evaluated, name: str) -> float:
    """Bigger is always better, whichever way the metric points."""
    value = _value(evaluated, name)
    return value if _metric(name)[1] else -value


def better(
    candidate: Evaluated, incumbent: Evaluated, name: str, min_gain_pct: float
) -> bool:
    """True if candidate is usable and beats incumbent by more than min_gain_pct."""
    if not candidate.feasible:
        return False
    if not incumbent.feasible:
        return True
    new = _value(candidate, name)
    old = _value(incumbent, name)
    if _metric(name)[1]:
        return new > old * (1 + min_gain_pct / 100)
    return new < old * (1 - min_gain_pct / 100)


def candidate_values(setting: Setting) -> list[SettingValue]:
    """Values worth trying for one setting, other than its default."""
    default = setting.default
    options: list[SettingValue]
    if setting.kind == "bool":
        options = [True, False]
    elif setting.kind == "choice":
        options = list(setting.choices)
    else:
        if default is None or isinstance(default, bool | str):
            return []
        options = []
        for factor in (0.5, 2.0):
            value = float(default) * factor
            if setting.low is not None:
                value = max(value, setting.low)
            if setting.high is not None:
                value = min(value, setting.high)
            options.append(round(value) if setting.kind == "int" else round(value, 4))
    found: list[SettingValue] = []
    for option in options:
        if option == default or option in found:
            continue
        if validate_settings({setting.name: option}):
            continue
        found.append(option)
    return found


def default_candidates(names: list[str] | None = None) -> dict[str, list[SettingValue]]:
    """Candidate values for the named settings, or for every known setting."""
    chosen = names if names is not None else list(BY_NAME)
    result: dict[str, list[SettingValue]] = {}
    for name in chosen:
        setting = BY_NAME.get(name)
        if setting is None:
            raise ValueError(f"unknown setting {name!r}; known: {', '.join(BY_NAME)}")
        values = candidate_values(setting)
        if values:
            result[name] = values
    return result


def _pair(evaluated: Evaluated) -> tuple[float, float]:
    metrics = evaluated.trial.metrics
    if metrics is None:
        raise ValueError("trial has no metrics")
    return metrics.throughput_tps, metrics.ttft_ms_p95


def _dominates(a: Evaluated, b: Evaluated) -> bool:
    a_rate, a_ttft = _pair(a)
    b_rate, b_ttft = _pair(b)
    return a_rate >= b_rate and a_ttft <= b_ttft and (a_rate > b_rate or a_ttft < b_ttft)


def pareto_front(evaluated: list[Evaluated]) -> list[Evaluated]:
    """Usable results that no other result beats on both throughput and TTFT p95."""
    points = [item for item in evaluated if item.feasible and item.trial.metrics is not None]
    front = [
        item
        for item in points
        if not any(_dominates(other, item) for other in points if other is not item)
    ]
    return sorted(front, key=lambda item: _pair(item)[0])


def run_search(
    evaluate: Callable[[Config], TrialResult],
    config: Config,
    objective: Objective,
    candidates: dict[str, list[SettingValue]],
    *,
    max_trials: int = 40,
    min_gain_pct: float = 2.0,
) -> SearchResult:
    """One setting at a time, then combine the winners, dropping any that clash."""
    name = objective.optimize
    _metric(name)
    if max_trials < 1:
        raise ValueError(f"max_trials must be at least 1, got {max_trials}")
    cache: dict[str, Evaluated] = {}
    order: list[Evaluated] = []
    stopped = False

    def measure(settings: dict[str, SettingValue]) -> Evaluated | None:
        nonlocal stopped
        trial_config = config.model_copy(update={"settings": dict(settings)})
        key = trial_config.key()
        if key in cache:
            return cache[key]
        if len(order) >= max_trials:
            stopped = True
            return None
        trial = evaluate(trial_config)
        item = Evaluated(
            settings=dict(settings),
            trial=trial,
            problem=feasibility_problem(trial, objective),
        )
        cache[key] = item
        order.append(item)
        return item

    baseline = measure(dict(config.settings))
    if baseline is None:
        raise RuntimeError("the baseline was not measured")
    if baseline.trial.status is not Status.OK or baseline.trial.metrics is None:
        raise ValueError(
            f"the starting config did not run: {baseline.trial.reason or baseline.trial.status.value}"
        )

    winners: list[tuple[str, SettingValue, Evaluated]] = []
    for setting_name, values in candidates.items():
        top: tuple[SettingValue, Evaluated] | None = None
        for value in values:
            item = measure({**config.settings, setting_name: value})
            if item is None:
                break
            if not better(item, baseline, name, min_gain_pct):
                continue
            if top is None or better(item, top[1], name, 0.0):
                top = (value, item)
        if top is not None:
            winners.append((setting_name, top[0], top[1]))
        if stopped:
            break

    current = baseline
    for setting_name, value, _ in sorted(
        winners, key=lambda winner: _signed(winner[2], name), reverse=True
    ):
        combined = measure({**current.settings, setting_name: value})
        if combined is not None and better(combined, current, name, min_gain_pct):
            current = combined

    return SearchResult(
        baseline=baseline,
        best=current,
        evaluated=order,
        front=pareto_front(order),
        stopped_early=stopped,
    )