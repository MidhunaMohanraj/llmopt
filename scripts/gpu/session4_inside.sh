#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

DATASETS="${DATASETS:-/datasets}"
WIKITEXT="$DATASETS/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
BASELINE="$OUT/ppl_baseline_vllm_rocm.json"

validate_step() {
  local name="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then echo; echo "--- $name"; echo "DRY RUN: llmopt validate $*"; return 0; fi
  step "$name" bash -c 'log="$0"; llmopt validate "$@" > "$log" 2>&1; rc=$?; tail -6 "$log"; exit $rc' "$OUT/$name.log" "$@"
}

after_case() {
  local name="$1"
  step "leftovers-after-$name" no_leftovers
  stop_unless "leftovers-after-$name"
  capture "amd-smi-after-$name" "$OUT/amd_smi_after_$name.txt" amd-smi
}

step versions bash -c 'record_versions.sh > "$0/versions.txt" 2>&1; cat "$0/versions.txt"' "$OUT"
capture amd-smi-before "$OUT/amd_smi_before.txt" amd-smi
gpu_guard
step datasets-check test -f "$WIKITEXT"
stop_unless datasets-check

validate_step ppl-bf16-baseline --task ppl --model "$MODEL" --engine vllm-rocm --datasets "$DATASETS" --precision bf16 --write-baseline "$BASELINE"
stop_unless ppl-bf16-baseline
after_case ppl-bf16-baseline

validate_step ppl-fp8 --task ppl --model "$MODEL" --engine vllm-rocm --datasets "$DATASETS" --precision fp8 --baseline "$BASELINE"
after_case ppl-fp8
validate_step ppl-kvfp8 --task ppl --model "$MODEL" --engine vllm-rocm --datasets "$DATASETS" --precision bf16 --kv-cache-dtype fp8 --baseline "$BASELINE"
after_case ppl-kvfp8
validate_step ppl-fp8-kvfp8 --task ppl --model "$MODEL" --engine vllm-rocm --datasets "$DATASETS" --precision fp8 --kv-cache-dtype fp8 --baseline "$BASELINE"
after_case ppl-fp8-kvfp8

bench_step power-bf16-chat --engine vllm-rocm --model "$MODEL" --results "$OUT/results_power" --workload chat --precision bf16 --power
stop_unless power-bf16-chat
after_case power-bf16-chat
bench_step power-fp8-chat --engine vllm-rocm --model "$MODEL" --results "$OUT/results_power" --workload chat --precision fp8 --power
after_case power-fp8-chat

for i in 1 2 3; do
  bench_step "stability-bf16-chat-$i" --engine vllm-rocm --model "$MODEL" --results "$OUT/stability_$i" --workload chat --precision bf16
  stop_unless "stability-bf16-chat-$i"
  after_case "stability-bf16-chat-$i"
done

step summary bash -c 'grep -H -E "^(status|TTFT)|score|drop" "$0"/*.log; true' "$OUT"

finish