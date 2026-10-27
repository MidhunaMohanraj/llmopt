#!/usr/bin/env bash
# GPU session 1 (Week 2 Day 1). Run ONLY on the MI300X machine.
#   GPU_VM=yes BASE_IMAGE=vllm/vllm-openai-rocm@sha256:<digest> ./scripts/gpu/session1.sh
# Set HF_TOKEN in your shell first only if the model needs it.
# Never write the token into this file or into any file.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

if [ "${GPU_VM:-}" != "yes" ]; then
  echo "Refusing to run: set GPU_VM=yes to confirm this is the GPU machine."
  echo "(The bench PC also has /dev/kfd, so that check alone cannot tell them apart.)"
  exit 2
fi
BASE_IMAGE="${BASE_IMAGE:?set BASE_IMAGE to the pinned base image (tag or digest)}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OUT="${OUT:-$HOME/llmopt-sessions/session1-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT" "$MODELS_DIR"
RESULTS="$OUT/results.txt"
: > "$RESULTS"

result() { echo "RESULT $1: $2"; echo "$1: $2" >> "$RESULTS"; }

main() {
  local start end rc
  start=$(date +%s)
  echo "== GPU session 1 start $(date -Iseconds)"
  echo "base image: $BASE_IMAGE"
  echo "output dir: $OUT"

  echo "--- check GPU device"
  if [ -e /dev/kfd ]; then result gpu-device OK; else result gpu-device FAILED; return 1; fi

  echo "--- pull base image"
  if docker pull "$BASE_IMAGE"; then result pull OK; else result pull FAILED; return 1; fi
  docker image inspect --format '{{index .RepoDigests 0}}' "$BASE_IMAGE" | tee "$OUT/base_image_digest.txt"

  echo "--- build llmopt image"
  if docker build -f docker/Dockerfile --build-arg BASE_IMAGE="$BASE_IMAGE" -t llmopt:session1 .; then
    result build OK
  else
    result build FAILED
    return 1
  fi

  echo "--- run session inside the container"
  docker run --rm \
    --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device /dev/kfd --device /dev/dri --ipc=host --network=host \
    -v "$PWD":/work:ro -v "$MODELS_DIR":/models -v "$OUT":/out \
    -e HF_TOKEN -e SKIP_SERVE="${SKIP_SERVE:-0}" \
    llmopt:session1 bash /work/scripts/gpu/session1_inside.sh
  rc=$?
  if [ "$rc" = 0 ]; then result container OK; else result container "FAILED (exit $rc)"; fi
  docker run --rm -v "$OUT":/out llmopt:session1 chown -R "$(id -u):$(id -g)" /out \
    || echo "note: could not chown $OUT (cleaning it up may need sudo)"

  end=$(date +%s)
  echo "== session 1 end $(date -Iseconds), elapsed $(( (end - start) / 60 )) min"
  echo "== RESULT summary (this script)"
  cat "$RESULTS"
  echo "== RESULT summary (inside the container)"
  cat "$OUT/results_inside.txt" 2>/dev/null || echo "(none)"
  echo "== REMINDER: the VM keeps billing until you destroy it. Copy $OUT off the machine first."
  if grep -q FAILED "$RESULTS" "$OUT/results_inside.txt" 2>/dev/null; then return 1; fi
  return 0
}

main 2>&1 | tee -a "$OUT/session.log"
exit "${PIPESTATUS[0]}"