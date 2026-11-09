#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

N="${1:-}"
case "$N" in
  2|3|4) ;;
  *) echo "usage: GPU_VM=yes BASE_IMAGE=<image> $0 <2|3|4>   or   $0 <2|3|4> --dry-run"; exit 2 ;;
esac
INSIDE="scripts/gpu/session${N}_inside.sh"
[ -f "$INSIDE" ] || { echo "no script for session $N yet: $INSIDE"; exit 2; }

if [ "${2:-}" = "--dry-run" ]; then
  echo "DRY RUN: would build llmopt:session$N and run $INSIDE in a container. Steps inside:"
  DRY_RUN=1 bash "$INSIDE"
  exit $?
fi

if [ "${GPU_VM:-}" != "yes" ]; then
  echo "Refusing to run: set GPU_VM=yes to confirm this is the GPU machine."
  echo "(The bench PC also has /dev/kfd, so that check alone cannot tell them apart.)"
  exit 2
fi
BASE_IMAGE="${BASE_IMAGE:?set BASE_IMAGE to the pinned base image (tag or digest)}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
OUT="${OUT:-$HOME/llmopt-sessions/session$N-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT" "$MODELS_DIR" "$DATASETS_DIR"
RESULTS="$OUT/results.txt"
: > "$RESULTS"
IMAGE="llmopt:session$N"

result() { echo "RESULT $1: $2"; echo "$1: $2" >> "$RESULTS"; }

main() {
  local start end rc
  start=$(date +%s)
  echo "== GPU session $N start $(date -Iseconds)"
  echo "base image: $BASE_IMAGE"
  echo "output dir: $OUT"

  echo "--- check GPU device"
  if [ -e /dev/kfd ]; then result gpu-device OK; else result gpu-device FAILED; return 1; fi

  echo "--- build llmopt image"
  if docker build -f docker/Dockerfile --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" .; then
    result build OK
  else
    result build FAILED
    return 1
  fi

  echo "--- run session $N inside the container"
  docker run --rm \
    --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device /dev/kfd --device /dev/dri --ipc=host --network=host \
    -v "$PWD":/work:ro -v "$MODELS_DIR":/models:ro -v "$DATASETS_DIR":/datasets:ro -v "$OUT":/out \
    "$IMAGE" bash "/work/$INSIDE"
  rc=$?
  if [ "$rc" = 0 ]; then result container OK; else result container "FAILED (exit $rc)"; fi
  docker run --rm -v "$OUT":/out "$IMAGE" chown -R "$(id -u):$(id -g)" /out \
    || echo "note: could not chown $OUT (cleaning it up may need sudo)"

  end=$(date +%s)
  echo "== session $N end $(date -Iseconds), elapsed $(( (end - start) / 60 )) min"
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
