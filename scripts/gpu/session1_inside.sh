#!/usr/bin/env bash
# Runs INSIDE the container (called by session1.sh).
set -uo pipefail
OUT=/out
MODEL=/models/Qwen__Qwen3-8B
PORT=8000

step() {
  local name="$1"; shift
  echo; echo "--- $name"
  if "$@"; then echo "RESULT $name: OK"; else echo "RESULT $name: FAILED"; fi
}

step versions bash -c 'record_versions.sh > /out/versions.txt 2>&1; cat /out/versions.txt'
step rocminfo bash -c 'rocminfo > /out/rocminfo.txt 2>&1; head -30 /out/rocminfo.txt'
step amd-smi-default bash -c 'amd-smi > /out/amd_smi_default.txt 2>&1; cat /out/amd_smi_default.txt'
step parse-real-output python3 /work/scripts/gpu/check_parsers.py /out
step torch-sees-gpu python3 -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

if [ ! -f "$MODEL/config.json" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "model missing at $MODEL and HF_TOKEN is not set"; echo "RESULT model: FAILED"; exit 1
  fi
  step fetch-model python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='$MODEL')"
fi

step model-size python3 - <<'EOF'
import glob, os
total = sum(os.path.getsize(p) for p in glob.glob("/models/Qwen__Qwen3-8B/*.safetensors"))
expected = 16_381_470_720
print("safetensors bytes:", total, "expected about:", expected)
raise SystemExit(0 if abs(total - expected) / expected < 0.01 else 1)
EOF

if [ "${SKIP_SERVE:-0}" = "1" ]; then
  echo "SKIP_SERVE=1: not starting vLLM"; exit 0
fi

echo; echo "--- serve Qwen3 8B"
vllm serve "$MODEL" --served-model-name qwen3-8b --dtype bfloat16 \
  --max-model-len 4096 --port "$PORT" > "$OUT/vllm_server.log" 2>&1 &
SERVER=$!
cleanup() { kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; }
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if curl -sf "localhost:$PORT/health" > /dev/null; then ready=1; break; fi
  kill -0 "$SERVER" 2>/dev/null || break
  sleep 5
done
if [ "$ready" = 1 ]; then echo "RESULT vllm-ready: OK"; else echo "RESULT vllm-ready: FAILED"; tail -20 "$OUT/vllm_server.log"; fi

if [ "$ready" = 1 ]; then
  step test-request bash -c "curl -s localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"qwen3-8b\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}' \
    > /out/completion.json; cat /out/completion.json"
fi

cleanup
step no-leftover-processes bash -c '! pgrep -f "[v]llm|[E]ngineCore"'
step amd-smi-after bash -c 'amd-smi > /out/amd_smi_after.txt 2>&1; cat /out/amd_smi_after.txt'