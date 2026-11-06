#!/usr/bin/env bash

set -u
echo "date: $(date -Iseconds)"
echo "image_python: $(python3 --version 2>&1)"
for pkg in torch vllm transformers pydantic typer llmopt; do
  python3 - <<EOF 2>/dev/null || echo "$pkg: not found"
import importlib.metadata as m
print("$pkg:", m.version("$pkg"))
EOF
done
echo "rocm: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'not found')"
tool_version() {
  local tool="$1"; shift
  if command -v "$tool" > /dev/null 2>&1; then
    echo "$tool: $("$tool" "$@" 2>&1 | head -1)"
  else
    echo "$tool: not found"
  fi
}
tool_version amd-smi version
tool_version rocminfo --version
tool_version rocprofv3 --version
