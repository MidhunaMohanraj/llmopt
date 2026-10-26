#!/usr/bin/env bash
# Print the version of every component, so reports can record them.
# Missing tools are reported as "not found", never skipped silently.
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
echo "amd-smi: $(amd-smi version 2>&1 | head -1 || echo 'not found')"
echo "rocminfo: $(rocminfo --version 2>&1 | head -1 || echo 'not found')"
echo "rocprofv3: $(rocprofv3 --version 2>&1 | head -1 || echo 'not found')"