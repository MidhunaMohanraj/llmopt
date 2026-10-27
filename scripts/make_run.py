"""Write the run file used by scripts/bench_all.sh: workload, objective, hardware, versions.

The GPU fields of HardwareProfile are placeholders, because the bench PC has no GPU.
"""

import argparse
import importlib.metadata as md
import platform
from pathlib import Path

from llmopt.report.store import save_report
from llmopt.schemas import HardwareProfile, Objective, Report, Workload

PACKAGES = ("torch", "transformers", "vllm", "lm_eval", "pydantic", "typer", "llmopt")


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def cpu_layout() -> tuple[int, int]:
    """Physical cores and NUMA nodes, read from /proc and /sys."""
    cores_per_socket = 1
    sockets: set[str] = set()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "cpu cores":
            cores_per_socket = int(value)
        elif key == "physical id":
            sockets.add(value.strip())
    nodes = len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
    return cores_per_socket * max(len(sockets), 1), max(nodes, 1)


def installed_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "not found"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=8)
    args = parser.parse_args()
    cores, nodes = cpu_layout()
    hardware = HardwareProfile(
        gpu_name="none (CPU-only bench, placeholder)",
        gfx_target="none (placeholder)",
        compute_units=1,
        hbm_gb=0.001,
        rocm_version="none (placeholder)",
        driver_version="none (placeholder)",
        cpu_model=cpu_model(),
        cpu_cores=cores,
        numa_nodes=nodes,
    )
    workload = Workload(
        name=f"bench: {args.prompt_tokens} prompt tokens, {args.output_tokens} output tokens",
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        concurrency=1,
        num_requests=args.num_requests,
    )
    objective = Objective(max_ttft_p95_ms=500.0, max_accuracy_drop_pct=1.0, optimize="throughput")
    report = Report(
        hardware=hardware,
        workload=workload,
        objective=objective,
        versions=installed_versions(),
    )
    save_report(report, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()