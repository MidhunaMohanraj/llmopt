# llmopt

A Python SDK that finds the best way to serve a large language model on an AMD GPU.
You give it a model, an AMD GPU target and an objective (for example, the lowest TTFT p95
with an accuracy drop under 1%). It measures candidate setups and writes JSON and HTML reports.

## Status (Phase 1, bench stage)

- Works and is tested on a CPU-only bench PC: host detection, model memory estimates, PyTorch
  and vLLM CPU adapters, profiler, accuracy checks (WikiText-2 perplexity, ARC-Easy),
  concurrency sweep, settings search and reports.
- Drafts that have never run on a GPU: the ROCm adapters (`pytorch-rocm`, `vllm-rocm`), the GPU
  power sampler, GPU detection on real `amd-smi` and `rocminfo` output, and `scripts/gpu/`.
- Numbers from the bench are single CPU runs on Qwen3 0.6B. They are not benchmarks of an AMD GPU.
- Reviewed only by the author so far.

Phase 1 scope: vLLM and PyTorch, with BF16, FP8 and FP8 KV cache. Other engines and
precisions are Phase 2 and are not offered.

## Install

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

The engines need packages that are not installed by default: torch and transformers for
PyTorch, and vLLM for the vLLM adapters. `docs/IMAGE.md` describes the container image that
has the ROCm stack.

## Commands

    llmopt detect                      # host profile; add --gpu on a GPU machine
    llmopt plan --gfx gfx942           # engine and precision pairs for a GPU target
    llmopt bench --engine vllm --model <model folder> --workload smoke
    llmopt sweep --engine vllm --model <model folder> --levels 1,2,4,8
    llmopt validate --model <model folder> --task ppl --write-baseline baseline.json
    llmopt optimize --engine vllm --model <model folder> --objective throughput
    llmopt report --run run.json --results results -o report.html

`scripts/bench_all.sh` runs bench, accuracy and report in one go on the bench PC.
`scripts/gpu/session1.sh` is for the rented GPU machine only and refuses to run unless
`GPU_VM=yes` is set.

## Checks

    python -m pytest
    python -m ruff check .
    python -m mypy src tests