#!/usr/bin/env bash
# GPU session 1 (Week 2 Day 1). Run on the MI300X machine from the repo root.
#   BASE_IMAGE=vllm/vllm-openai-rocm@sha256:<digest> ./scripts/gpu/session1.sh
# Set HF_TOKEN in your shell first if the model is not already in $MODELS_DIR.
# Never write the token into this file.
set -uo pipefail

BASE_IMAGE="${BASE_IMAGE:?set BASE_IMAGE to the pinned base image (tag or digest)}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OUT="${OUT:-$HOME/llmopt-sessions/session1-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT" "$MODELS_DIR"
START=$(date +%s)
exec > >(tee -a "$OUT/session.log") 2>&1

echo "== GPU session 1 start $(date -Iseconds)"
echo "base image: $BASE_IMAGE"
echo "output dir: $OUT"

echo "--- pull base image"
docker pull "$BASE_IMAGE" || { echo "RESULT pull: FAILED"; exit 1; }
echo "RESULT pull: OK"
docker image inspect --format '{{index .RepoDigests 0}}' "$BASE_IMAGE" | tee "$OUT/base_image_digest.txt"

echo "--- build llmopt image"
docker build -f docker/Dockerfile --build-arg BASE_IMAGE="$BASE_IMAGE" -t llmopt:session1 . \
  || { echo "RESULT build: FAILED"; exit 1; }
echo "RESULT build: OK"

echo "--- run session inside the container"
docker run --rm \
  --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri --ipc=host --network=host \
  -v "$PWD":/work -v "$MODELS_DIR":/models -v "$OUT":/out \
  -e HF_TOKEN -e SKIP_SERVE="${SKIP_SERVE:-0}" \
  llmopt:session1 bash /work/scripts/gpu/session1_inside.sh
echo "container exit code: $?"

END=$(date +%s)
echo "== session 1 end $(date -Iseconds), elapsed $(( (END - START) / 60 )) min"
echo "== RESULT summary"
grep -h "^RESULT" "$OUT/session.log" || true