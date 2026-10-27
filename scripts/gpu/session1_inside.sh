#!/usr/bin/env bash
# Runs INSIDE the container (called by session1.sh). Exit code 0 only if every step passed.
set -uo pipefail
OUT=/out
MODEL=/models/Qwen__Qwen3-8B
PORT=8000
CHECKSUMS=/work/docs/checksums/qwen3-8b.sha256
FAILS=0
RES="$OUT/results_inside.txt"
: > "$RES"

step() {
  local name="$1"; shift
  echo; echo "--- $name"
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

# Run a tool, save ALL its output to a file, show the top, and keep the tool's own exit code.
capture() {
  local name="$1" file="$2"; shift 2
  step "$name" bash -c '"$@" > "$0" 2>&1; rc=$?; head -40 "$0"; exit $rc' "$file" "$@"
}

check_model() {
  if [ -f "$CHECKSUMS" ]; then
    (cd /models && sha256sum -c "$CHECKSUMS")
  else
    echo "no checksum file at $CHECKSUMS: falling back to a size check"
    python3 - <<'PYEOF'
import glob

total = sum(__import__("os").path.getsize(p) for p in glob.glob("/models/Qwen__Qwen3-8B/*.safetensors"))
expected = 16_381_470_720
print("safetensors bytes:", total, "expected about:", expected)
raise SystemExit(0 if abs(total - expected) / expected < 0.01 else 1)
PYEOF
  fi
}

healthy() {
  python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health', timeout=3)" > /dev/null 2>&1
}

test_request() {
  python3 - "$PORT" <<'PYEOF'
import json
import sys
import urllib.request

url = f"http://localhost:{sys.argv[1]}/v1/completions"
body = json.dumps(
    {"model": "qwen3-8b", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}
).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
data = json.load(urllib.request.urlopen(req, timeout=120))
with open("/out/completion.json", "w") as handle:
    handle.write(json.dumps(data, indent=2))
text = data["choices"][0]["text"]
print("completion:", repr(text))
raise SystemExit(0 if text.strip() else 1)
PYEOF
}

no_leftovers() {
  for _ in 1 2 3 4 5 6; do
    pgrep -f "[v]llm|[E]ngineCore" > /dev/null || return 0
    sleep 5
  done
  return 1
}

step versions bash -c 'record_versions.sh > /out/versions.txt 2>&1; cat /out/versions.txt'
capture rocminfo /out/rocminfo.txt rocminfo
capture amd-smi-static /out/amd_smi_static.txt amd-smi static
capture amd-smi-default /out/amd_smi_default.txt amd-smi
step parse-real-output python3 /work/scripts/gpu/check_parsers.py /out
step torch-sees-gpu python3 -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

if [ ! -f "$MODEL/config.json" ]; then
  step fetch-model python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='$MODEL')"
  [ -f "$MODEL/config.json" ] || { echo "model still missing, stopping"; finish; }
fi
step model-check check_model

if [ "${SKIP_SERVE:-0}" = "1" ]; then
  echo "SKIP_SERVE=1: not starting vLLM"
  finish
fi

echo; echo "--- serve Qwen3 8B"
vllm serve "$MODEL" --served-model-name qwen3-8b --dtype bfloat16 \
  --max-model-len 4096 --port "$PORT" > "$OUT/vllm_server.log" 2>&1 &
SERVER=$!
cleanup() { kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if healthy; then ready=1; break; fi
  kill -0 "$SERVER" 2>/dev/null || break
  sleep 5
done
if [ "$ready" = 1 ]; then
  echo "RESULT vllm-ready: OK" | tee -a "$RES"
  step test-request test_request
else
  echo "RESULT vllm-ready: FAILED" | tee -a "$RES"
  FAILS=$((FAILS + 1))
  tail -20 "$OUT/vllm_server.log"
fi

cleanup
step no-leftover-processes no_leftovers
finish