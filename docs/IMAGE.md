# Container image (task 1.14)

- Image: docker.io/midhunamohanraj/llmopt:0.1.0-vllm0.32.0-rocm7.2.3
- Digest: sha256:7c503750d17f6e8adc05b0fee3cdbefd868b31f722f1dabdefda5b06034a361d
- Base image: vllm/vllm-openai-rocm:v0.32.0, pinned by digest
  sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905
- Built from docker/Dockerfile on main, on the bench PC (no GPU), 2026-10-27.

## Contents (read from the image during the bench dry run)

| Component | Version |
|---|---|
| ROCm | 7.2.3 |
| vLLM | 0.32.0+rocm723 |
| torch | 2.13.0+git733fca1 |
| transformers | 5.19.0 |
| pydantic / typer | 2.14.1 / 0.27.3 |
| llmopt | 0.1.0 |
| Python | 3.12.15 |
| amd-smi | 26.2.2 |
| rocprofv3 | 1.3.2 |

The image also records its versions at build time in /opt/llmopt/versions_at_build.txt.

## Status

- Built and pushed from the bench. It has never been run on a GPU.
- The image is about 46.8 GB when unpacked; its largest layer is about 21.5 GB.
- The base image's license was not checked.
- Reviewed only by the author; the reviewer has not looked at it yet.

## Use

    docker pull docker.io/midhunamohanraj/llmopt@sha256:7c503750d17f6e8adc05b0fee3cdbefd868b31f722f1dabdefda5b06034a361d

scripts/gpu/session1.sh does not pull this image. It builds its own copy on the GPU
machine from docker/Dockerfile and the pinned base digest. Both paths use the same base.