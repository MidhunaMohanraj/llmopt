"""Turn a Report into one self-contained HTML page."""

from __future__ import annotations

from html import escape
from pathlib import Path

from llmopt.schemas import AccuracyResult, Config, Report, TrialResult

SPREAD_WARN_PCT = 5.0

STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 70rem; padding: 0 1rem; }
table { border-collapse: collapse; margin: 0.5rem 0 1.5rem; }
th, td { border: 1px solid #8884; padding: 0.3rem 0.6rem; text-align: left; }
th { background: #8882; }
.notice { border: 2px solid #c80; padding: 0.6rem 1rem; margin: 1rem 0; }
.scroll { overflow-x: auto; }
"""

TRIAL_HEADERS = (
    "Model",
    "Engine",
    "Precision",
    "KV cache",
    "Status",
    "TTFT p50 (ms)",
    "TTFT p95 (ms)",
    "TPOT (ms)",
    "Throughput (tok/s)",
    "Peak memory (GiB)",
    "Run spread (%)",
    "Accuracy",
)


def _cell(text: str) -> str:
    return f"<td>{escape(text)}</td>"


def _number(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _config_cells(config: Config) -> list[str]:
    return [config.model, config.engine.value, config.precision.value, config.kv_cache_dtype]


def _accuracy_text(items: list[AccuracyResult]) -> str:
    if not items:
        return "-"
    return "; ".join(
        f"{item.task}: {item.score:.3f} (baseline {item.baseline_score:.3f}, "
        f"drop {item.drop_pct:.1f}%)"
        for item in items
    )


def _trial_row(trial: TrialResult) -> str:
    status = trial.status.value
    if trial.reason:
        status = f"{status}: {trial.reason}"
    cells = [*_config_cells(trial.config), status]
    metrics = trial.metrics
    if metrics is None:
        cells += ["-"] * 6
    else:
        spread = f"{metrics.run_spread_pct:.1f}"
        if metrics.run_spread_pct > SPREAD_WARN_PCT:
            spread += " (unstable?)"
        cells += [
            _number(metrics.ttft_ms_p50),
            _number(metrics.ttft_ms_p95),
            _number(metrics.tpot_ms),
            _number(metrics.throughput_tps),
            _number(metrics.peak_memory_gb, 2),
            spread,
        ]
    cells.append(_accuracy_text(trial.accuracy))
    return "<tr>" + "".join(_cell(cell) for cell in cells) + "</tr>"


def _key_value_table(title: str, values: dict[str, object]) -> str:
    rows = "".join(
        f"<tr><th>{escape(key)}</th>{_cell(str(value))}</tr>" for key, value in values.items()
    )
    return f"<h2>{escape(title)}</h2><table>{rows}</table>"


def render_report(report: Report, notice: str | None = None) -> str:
    parts = ["<h1>llmopt report</h1>"]
    if notice:
        parts.append(f'<div class="notice">{escape(notice)}</div>')
    parts.append(_key_value_table("Hardware", dict(report.hardware.model_dump())))
    parts.append(_key_value_table("Workload", dict(report.workload.model_dump())))
    parts.append(_key_value_table("Objective", dict(report.objective.model_dump())))
    header = "".join(f"<th>{escape(name)}</th>" for name in TRIAL_HEADERS)
    rows = "".join(_trial_row(trial) for trial in report.trials)
    parts.append(
        f'<h2>Trials</h2><div class="scroll"><table><tr>{header}</tr>{rows}</table></div>'
    )
    if report.recommendation is None:
        parts.append("<h2>Recommendation</h2><p>No recommendation.</p>")
    else:
        config = report.recommendation
        text = ", ".join(_config_cells(config))
        parts.append(f"<h2>Recommendation</h2><p>{escape(text)}</p>")
    if report.versions:
        parts.append(_key_value_table("Versions", dict(report.versions)))
    body = "\n".join(parts)
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>llmopt report</title><style>{STYLE}</style></head>"
        f"<body>\n{body}\n</body></html>\n"
    )


def write_report(report: Report, path: Path, notice: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(report, notice))