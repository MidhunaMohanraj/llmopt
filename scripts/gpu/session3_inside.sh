#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

AMD_FP8_MODEL="${AMD_FP8_MODEL:-/models/amd__Llama-3.1-8B-Instruct-FP8-KV}"
LLAMA_BF16_MODEL="${LLAMA_BF16_MODEL:-/models/meta-llama__Llama-3.1-8B-Instruct}"

step versions bash -c 'record_versions.sh > "$0/versions.txt" 2>&1; cat "$0/versions.txt"' "$OUT"
capture amd-smi-before "$OUT/amd_smi_before.txt" amd-smi
gpu_guard

run_case() {
  local name="$1" model="$2"; shift 2
  bench_step "$name" --engine vllm-rocm --model "$model" --results "$OUT/results" "$@"
  step "leftovers-after-$name" no_leftovers
  stop_unless "leftovers-after-$name"
  capture "amd-smi-after-$name" "$OUT/amd_smi_after_$name.txt" amd-smi
}

run_optional() {
  local name="$1" model="$2"; shift 2
  if [ "$DRY_RUN" = "1" ] || [ -d "$model" ]; then
    run_case "$name" "$model" --workload chat "$@"
  else
    echo "SKIPPED $name: $model not found" | tee -a "$RES"
  fi
}

run_case vllm-rocm-bf16-chat "$MODEL" --workload chat --precision bf16
stop_unless vllm-rocm-bf16-chat
run_case vllm-rocm-fp8-chat "$MODEL" --workload chat --precision fp8
run_case vllm-rocm-bf16-kvfp8-chat "$MODEL" --workload chat --precision bf16 --kv-cache-dtype fp8
run_case vllm-rocm-fp8-kvfp8-chat "$MODEL" --workload chat --precision fp8 --kv-cache-dtype fp8
run_optional amd-fp8-kvfp8-chat "$AMD_FP8_MODEL" --precision fp8 --kv-cache-dtype fp8
run_optional llama-bf16-chat "$LLAMA_BF16_MODEL" --precision bf16

finish