import subprocess
from pathlib import Path

PATH = Path("src/llmopt/validate/perplexity.py")

CHECK_FN = '''def check_device(device: str) -> None:
    """Accept 'cpu', 'cuda' or 'cuda:N'. ROCm builds of torch call AMD GPUs 'cuda'."""
    name, colon, index = device.partition(":")
    valid = (name == "cpu" and not colon) or (
        name == "cuda" and (not colon or index.isdigit())
    )
    if not valid:
        raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")'''

EDITS = [
    (
        "def evaluate_perplexity(\n",
        CHECK_FN + "\n\n\ndef evaluate_perplexity(\n",
    ),
    (
        "    max_windows: int = 20,\n) -> PerplexityResult:\n",
        '    max_windows: int = 20,\n    device: str = "cpu",\n) -> PerplexityResult:\n',
    ),
    (
        '    """Score up to max_windows non-overlapping windows of window_tokens tokens."""\n',
        '    """Score up to max_windows non-overlapping windows of window_tokens tokens.\n\n'
        "    The caller moves the model to `device`; the input windows are placed there too.\n"
        "    Only the CPU path is tested. The GPU path is untested.\n"
        '    """\n',
    ),
    (
        '        raise ValueError(f"max_windows must be at least 1, got {max_windows}")\n',
        '        raise ValueError(f"max_windows must be at least 1, got {max_windows}")\n'
        "    check_device(device)\n",
    ),
    (
        "            input_ids = torch.tensor([chunk], dtype=torch.long)\n",
        "            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)\n",
    ),
]

text = PATH.read_text()
for old, _ in EDITS:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly 1 match, found {count} for:\n{old}")
for old, new in EDITS:
    text = text.replace(old, new)
if not text.endswith("\n"):
    text += "\n"
PATH.write_text(text)
print("patched", PATH)
subprocess.run(["git", "--no-pager", "diff", "--stat"], check=False)