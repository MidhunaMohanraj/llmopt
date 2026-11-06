
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[ "$#" -gt 0 ] || { echo "usage: trace.sh <llmopt bench arguments>"; exit 2; }
command -v rocprofv3 > /dev/null || [ "$DRY_RUN" = "1" ] || { echo "rocprofv3 not found"; exit 1; }

step rocprofv3-trace rocprofv3 --kernel-trace --output-format csv -d "$OUT/rocprof" -- llmopt bench "$@"
finish