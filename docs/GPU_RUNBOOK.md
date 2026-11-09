# GPU day runbook (Session 1)

## Before booking time (on the bench PC, free)

1. `git switch main && git pull --ff-only origin main`; `python -m pytest -q` passes.
2. Image: the session script pulls the pinned base image
   `vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905`
   (ROCm 7.2.3, vLLM 0.32.0) and builds llmopt on top of it from `docker/Dockerfile` on the
   VM. It does not use the image on Docker Hub. The old Docker Hub tag
   `0.1.0-vllm0.32.0-rocm7.2.3` is broken (no configs folder): do not use it.
3. Book a SINGLE-GPU MI300X VM. An 8-GPU VM burns quota 8 times faster.
4. Decide how Qwen3-8B (about 16 GB) gets onto the VM. Downloading it on a billed VM costs
   money. If you must download, set `HF_TOKEN` in the shell on the VM. Never paste it anywhere.
5. Write down the start time. Log every session: what ran, hours used, results produced.
6. Set up an SSH key with the cloud account before booking, so you can log in and copy
   results off.

## On the VM

First run `docker --version`, `id` and `ls /dev/kfd /dev/dri`. If Docker is
missing or your user cannot run it, fix that before anything else. This is not verified
for AMD's VM images.

1. Clone the repo and `cd` to its root.
2. Run (read the header of `scripts/gpu/session1.sh` for the current variables):
   `GPU_VM=yes BASE_IMAGE=vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905 ./scripts/gpu/session1.sh`
3. Never set `GPU_VM=yes` on the bench PC. The bench also has a `/dev/kfd` device.
4. Read the result lines at the end. Every step prints `RESULT <name>: OK` or `FAILED`.

## What each step means if it fails

| Step | If it fails | Do |
|---|---|---|
| pull / build | Image or network problem | Fix and retry once. Do not go on |
| rocminfo, amd-smi | Driver or device flags problem | Check `/dev/kfd` and `/dev/dri` are passed to the container. Save the error text |
| parse-real-output | A parser cannot read real MI300X output | Save the raw files. This is expected to be possible: the parsers have only seen AMD doc samples. Do not stop the session for it |
| torch-sees-gpu | ROCm and torch do not see the GPU | Stop. Nothing else will work. Check `rocm-smi` or `amd-smi` outside the container |
| model-check | Model missing or checksum mismatch | Fix the model files before serving |
| vllm-ready / test-request | vLLM did not start or answer | Read `vllm_server.log`. Note the first error line |
| no-leftover-processes | A vLLM process survived | Kill it by PID before any further run |

## What a good session looks like

Every `RESULT` line ends in OK, except `parse-real-output`, which may fail. A failed parser is information, not a stop: keep `rocminfo.txt`, `amd_smi_static.txt` and `amd_smi_default.txt` and fix the parsers afterwards. The script exits with an error when any step failed, so a non-zero exit caused only by that step is still a good session. All five model shards must check OK before the server starts.

## Stop rules (to protect the budget)

- If `torch-sees-gpu` fails, the script stops by itself before the model and server steps.
  Debug outside the paid loop.
- If `model-check` fails, the script stops before starting the server. Fix the model
  files first.
- If the same step fails twice with the same error, stop and write the error down.
- Do not start sweeps or tuning in session 1.

## Before ending the session

1. Copy the whole results folder off the VM (session log, `rocminfo.txt`, `amd_smi_*.txt`,
   `versions.txt`, `vllm_server.log`, `completion.json`). Example, run on the bench PC
   (replace user and address): `scp -r user@vm-address:~/llmopt-sessions/session1-* ~/gpu-results/`
2. Destroy the VM. Assume a stopped VM still bills.
3. Record the hours used.

## Back on the bench

1. Put the real `rocminfo` and `amd-smi` outputs under `tests/fixtures/gpu/` as new fixtures
   and make the parsers pass on them.
2. Replace every "draft, untested" note that the session proved wrong, and keep the ones it
   did not test.
3. Open a PR per fix. Run `gh auth logout` when finished.

## Unproven until this session

ROCm adapters, power sampler, GPU parsers, `vllm serve` on ROCm, environment variables taking
effect in vLLM's engine process, and teardown freeing GPU memory between trials.

## Sessions 2 to 4 (drafts, never run on a GPU)

Run these on the VM after session 1 has passed. The launcher takes the session number:
`GPU_VM=yes BASE_IMAGE=vllm/vllm-openai-rocm@sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905 ./scripts/gpu/session.sh 2`

To see what a session would do without running anything (safe on the bench PC):
`./scripts/gpu/session.sh 3 --dry-run`

| Session | What it runs | Needs on the VM |
|---|---|---|
| 2 | BF16 on vllm-rocm and pytorch-rocm, workloads smoke and chat | Qwen3-8B |
| 3 | BF16 baseline, then FP8, FP8 KV cache and both, workload chat, on vllm-rocm. Optional: AMD's FP8-KV Llama model and the BF16 Llama reference | Qwen3-8B. Optional Llama folders (8.5 GB and 30 GB) |
| 4 | Perplexity through vllm-rocm (BF16 baseline, then FP8, FP8 KV cache, both), energy with `--power`, three repeats of one BF16 run | Qwen3-8B and the WikiText-2 files in the datasets folder |

Notes:
- Session 3 skips the optional models if their folders are missing. The BF16 Llama reference is 30 GB, so leave it off the VM unless it is really needed.
- Session 4 reads `$HOME/datasets` on the VM (variable `DATASETS_DIR`), mounted at `/datasets`. Copy the `wikitext/` folder there before the session. ARC is not run: the image has no lm_eval and `validate --engine` does perplexity only.
- A baseline written through vllm-rocm only compares with vllm-rocm runs, and neither side may use `--limit`.
- The accuracy drop is read from the logs in the output folder. Nothing in the script enforces a limit.
- A failed BF16 baseline stops the session. FP8 steps may fail, and the script carries on. A leftover vLLM process stops the session.
- Copy the whole output folder off the VM and destroy the VM, as in session 1.
- Unproven: FP8 and FP8 KV cache settings on ROCm, `bench --power`, and `validate --engine vllm-rocm`. Expect fixes after the first run.
