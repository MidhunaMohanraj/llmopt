"""Run the real parsers on the real tool output saved by GPU session 1."""

import sys
from pathlib import Path

from llmopt.hardware.amd_smi import parse_amd_smi_default, parse_amd_smi_static
from llmopt.hardware.rocminfo import parse_rocminfo

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/out")


def show(obj: object) -> str:
    dump = getattr(obj, "model_dump", None)
    return str(dump() if callable(dump) else obj)


def main() -> int:
    failures = 0
    for name, file, parser in (
        ("rocminfo", "rocminfo.txt", parse_rocminfo),
        ("amd-smi static", "amd_smi_static.txt", parse_amd_smi_static),
        ("amd-smi default", "amd_smi_default.txt", parse_amd_smi_default),
    ):
        try:
            result = parser((OUT / file).read_text())
        except (OSError, ValueError, KeyError) as exc:  # keep going, report each parser
            failures += 1
            print(f"{name}: PARSE FAILED: {type(exc).__name__}: {exc}")
            continue
        items = result if isinstance(result, list) else [result]
        if not items:
            failures += 1
            print(f"{name}: PARSE FAILED: no entries found in the output")
            continue
        print(f"{name}: parsed {len(items)} item(s)")
        for item in items:
            print("  ", show(item))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())