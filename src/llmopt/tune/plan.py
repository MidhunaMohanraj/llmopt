"""Which adapters can run each engine and precision pair."""

from __future__ import annotations

from pydantic import BaseModel

from llmopt.schemas import Engine, Precision

# Adapter names are the values accepted by `llmopt bench --engine`.
# The CPU adapters run on the bench PC. The ROCm adapters are drafts: they have never run
# on a GPU.
ADAPTERS: dict[tuple[Engine, Precision], tuple[str, ...]] = {
    (Engine.PYTORCH, Precision.BF16): ("pytorch", "pytorch-rocm"),
    (Engine.VLLM, Precision.BF16): ("vllm", "vllm-rocm"),
    (Engine.VLLM, Precision.FP8): ("vllm-rocm",),
}
BENCH_ADAPTERS = frozenset({"pytorch", "vllm"})


class PlanEntry(BaseModel):
    engine: Engine
    precision: Precision
    adapters: list[str]

    @property
    def has_adapter(self) -> bool:
        return bool(self.adapters)

    @property
    def runs_on_bench(self) -> bool:
        return any(name in BENCH_ADAPTERS for name in self.adapters)


def build_plan(pairs: list[tuple[Engine, Precision]]) -> list[PlanEntry]:
    return [
        PlanEntry(
            engine=engine,
            precision=precision,
            adapters=list(ADAPTERS.get((engine, precision), ())),
        )
        for engine, precision in pairs
    ]


def _describe(name: str) -> str:
    if name in BENCH_ADAPTERS:
        return f"{name} (runs on the bench PC)"
    return f"{name} (draft, not tested on a GPU)"


def format_plan(gfx_target: str, entries: list[PlanEntry]) -> str:
    lines = [f"Candidates for {gfx_target}: {len(entries)}"]
    for entry in entries:
        status = ", ".join(_describe(name) for name in entry.adapters) or "no adapter yet"
        lines.append(f"  {entry.engine.value:<9} {entry.precision.value:<10} {status}")
    with_adapter = sum(1 for entry in entries if entry.has_adapter)
    on_bench = sum(1 for entry in entries if entry.runs_on_bench)
    lines.append(
        f"{with_adapter} of {len(entries)} have an adapter; "
        f"{on_bench} of {len(entries)} can run on the bench PC (no GPU)"
    )
    return "\n".join(lines)