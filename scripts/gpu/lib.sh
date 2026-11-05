if [ "${DRY_RUN:-0}" = "1" ]; then
  OUT="${OUT:-/tmp/llmopt-dry}"
else
  OUT="${OUT:-/out}"
fi
DRY_RUN="${DRY_RUN:-0}"
MODEL="${MODEL:-/models/Qwen__Qwen3-8B}"
CHECKSUMS="${CHECKSUMS:-/work/docs/checksums/qwen3-8b.sha256}"
FAILS=0
mkdir -p "$OUT"
RES="$OUT/results_inside.txt"
: > "$RES"

step() {
  local name="$1"; shift
  echo; echo "--- $name"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN: $*"
    return 0
  fi
  if "$@"; then
    echo "RESULT $name: OK" | tee -a "$RES"
  else
    echo "RESULT $name: FAILED" | tee -a "$RES"
    FAILS=$((FAILS + 1))
  fi
}

finish() {
  echo; echo "failed steps inside the container: $FAILS"
  [ "$FAILS" = 0 ]
  exit $?
}

passed() { [ "$DRY_RUN" = "1" ] || grep -q "^RESULT $1: OK" "$RES"; }
stop_unless() { passed "$1" || { echo "stopping: $1 failed"; finish; }; }

capture() {
  local name="$1" file="$2"; shift 2
  if [ "$DRY_RUN" = "1" ]; then echo; echo "--- $name"; echo "DRY RUN: $* > $file"; return 0; fi
  step "$name" bash -c '"$@" > "$0" 2>&1; rc=$?; head -40 "$0"; exit $rc' "$file" "$@"
}

bench_step() {
  local name="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then echo; echo "--- $name"; echo "DRY RUN: llmopt bench $*"; return 0; fi
  step "$name" bash -c 'log="$0"; llmopt bench "$@" > "$log" 2>&1; rc=$?; tail -5 "$log"; [ "$rc" = 0 ] && grep -q "^status: ok" "$log"' "$OUT/$name.log" "$@"
}

check_model() {
  if [ -f "$CHECKSUMS" ]; then
    (cd /models && sha256sum -c "$CHECKSUMS")
  else
    echo "no checksum file at $CHECKSUMS: falling back to a size check"
    python3 - "$MODEL" <<'PYEOF'
import glob
import os
import sys

total = sum(os.path.getsize(p) for p in glob.glob(sys.argv[1] + "/*.safetensors"))
expected = 16_381_470_720
print("safetensors bytes:", total, "expected about:", expected)
raise SystemExit(0 if abs(total - expected) / expected < 0.01 else 1)
PYEOF
  fi
}

no_leftovers() {
  for _ in 1 2 3 4 5 6; do
    pgrep -f "[v]llm|[E]ngineCore" > /dev/null || return 0
    sleep 5
  done
  return 1
}

gpu_guard() {
  step torch-sees-gpu python3 -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  passed torch-sees-gpu || { echo "torch cannot see a GPU: stopping"; finish; }
  step model-check check_model
  passed model-check || { echo "model check failed: stopping"; finish; }
}