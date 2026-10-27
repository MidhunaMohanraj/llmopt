#!/usr/bin/env bash
# One command for the Phase 1 bench flow: bench both engines, score accuracy, build the report.
#   scripts/bench_all.sh                 quick run, perplexity only
#   WITH_ARC=1 scripts/bench_all.sh      also ARC-Easy (about 8 minutes per pass on the bench CPU)
# Settings can be changed with environment variables: MODEL, OUT, DATASETS,
# PROMPT_TOKENS, OUTPUT_TOKENS, NUM_REQUESTS, RUNS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${MODEL:-$HOME/models/Qwen__Qwen3-0.6B}"
OUT="${OUT:-$HOME/llmopt-runs/bench-$(date +%Y%m%d-%H%M%S)}"
DATASETS="${DATASETS:-$HOME/datasets}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-16}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
RUNS="${RUNS:-3}"
WITH_ARC="${WITH_ARC:-0}"

command -v llmopt > /dev/null || { echo "llmopt not found: activate the venv first"; exit 1; }
case "$(realpath -m "$OUT")" in
  "$REPO"/*) echo "refusing to write results inside the repo: $OUT"; exit 1 ;;
esac
load=$(cut -d' ' -f1 /proc/loadavg)
if awk "BEGIN{exit !($load > 2)}"; then
  echo "WARNING: load average is $load; timings may be unreliable"
fi

mkdir -p "$OUT/results"
echo "== llmopt bench run, output in $OUT"

trial_of() {
  grep -E "^Saved to " "$OUT/bench_$1.log" | tail -1 | sed 's/^Saved to //'
}

for engine in pytorch vllm; do
  echo "--- bench $engine"
  if ! llmopt bench --engine "$engine" --model "$MODEL" --results "$OUT/results" \
      --prompt-tokens "$PROMPT_TOKENS" --output-tokens "$OUTPUT_TOKENS" \
      --num-requests "$NUM_REQUESTS" --runs "$RUNS" > "$OUT/bench_$engine.log" 2>&1; then
    echo "bench $engine FAILED. Last lines of $OUT/bench_$engine.log:"
    tail -15 "$OUT/bench_$engine.log"
    exit 1
  fi
  grep -E "^(status|TTFT)" "$OUT/bench_$engine.log"
done

PYTORCH_TRIAL="$(trial_of pytorch)"
[ -f "$PYTORCH_TRIAL" ] || { echo "could not find the PyTorch trial file"; exit 1; }

echo "--- accuracy: WikiText-2 perplexity (attached to the PyTorch trial)"
llmopt validate --model "$MODEL" --task ppl --datasets "$DATASETS" \
  --write-baseline "$OUT/ppl_baseline.json" 2>&1 | grep -E "score|saved"
llmopt validate --model "$MODEL" --task ppl --datasets "$DATASETS" \
  --baseline "$OUT/ppl_baseline.json" --trial "$PYTORCH_TRIAL" 2>&1 | grep -E "score|drop|Attached"

if [ "$WITH_ARC" = "1" ]; then
  echo "--- accuracy: ARC-Easy (two passes, slow)"
  llmopt validate --model "$MODEL" --task arc --datasets "$DATASETS" \
    --write-baseline "$OUT/arc_baseline.json" 2>&1 | grep -E "score|saved"
  llmopt validate --model "$MODEL" --task arc --datasets "$DATASETS" \
    --baseline "$OUT/arc_baseline.json" --trial "$PYTORCH_TRIAL" 2>&1 | grep -E "score|drop|Attached"
fi

echo "--- report"
python "$SCRIPT_DIR/make_run.py" "$OUT/run.json" \
  --prompt-tokens "$PROMPT_TOKENS" --output-tokens "$OUTPUT_TOKENS" --num-requests "$NUM_REQUESTS"
llmopt report --run "$OUT/run.json" --results "$OUT/results" --output "$OUT/report.html" \
  --notice "CPU-only run on the bench PC. GPU fields are placeholders. The accuracy baseline is the same model, so a 0.0% drop only shows the pipeline works. Not GPU benchmarks."

echo "== done. Files in $OUT:"
ls "$OUT"