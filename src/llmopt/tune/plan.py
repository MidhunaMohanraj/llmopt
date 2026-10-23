"""List the engine and precision pairs worth trying on a GPU, and which ones we can run."""

from __future__ import annotations

from pydantic import BaseModel

from llmopt.schemas import Engine, Precision

# Adapters that exist today. Both are CPU adapters and support bf16 only.
ADAPTER_PRECISIONS: dict[Engine, frozenset[Precision]] = {
    Engine.PYTORCH: frozenset({Precision.BF16}),
    Engine.VLLM: frozenset({Precision.BF16}),
}


class PlanEntry(BaseModel):
    engine: Engine
    precision: Precision
    has_adapter: bool


def build_plan(pairs: list[tuple[Engine, Precision]]) -> list[PlanEntry]:
    return [
        PlanEntry(
            engine=engine,
            precision=precision,
            has_adapter=precision in ADAPTER_PRECISIONS.get(engine, frozenset()),
        )
        for engine, precision in pairs
    ]


def format_plan(gfx_target: str, entries: list[PlanEntry]) -> str:
    lines = [f"Candidates for {gfx_target}: {len(entries)}"]
    for entry in entries:
        status = (
            "adapter exists (CPU adapter, bf16 only)"
            if entry.has_adapter
            else "no adapter yet"
        )
        lines.append(f"  {entry.engine.value:<9} {entry.precision.value:<10} {status}")
    ready = sum(1 for entry in entries if entry.has_adapter)
    lines.append(f"{ready} of {len(entries)} can be run with the adapters we have")
    return "\n".join(lines)