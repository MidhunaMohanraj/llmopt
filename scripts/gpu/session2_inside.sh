#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

step versions bash -c 'record_versions.sh > "$0/versions.txt" 2>&1; cat "$0/versions.txt"' "$OUT"
capture amd-smi-before "$OUT/amd_smi_before.txt" amd-smi
gpu_guard

for workload in smoke chat; do
  for engine in vllm-rocm pytorch-rocm; do
    name="$engine-bf16-$workload"
    bench_step "$name" --engine "$engine" --model "$MODEL" --results "$OUT/results" --workload "$workload" --runs 3
    stop_unless "$name"
    step "leftovers-after-$name" no_leftovers
    stop_unless "leftovers-after-$name"
    capture "amd-smi-after-$name" "$OUT/amd_smi_after_$name.txt" amd-smi
  done
done

finish