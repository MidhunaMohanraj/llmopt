# Published image (task 1.14)

| Item | Value |
|---|---|
| Image | `docker.io/midhunamohanraj/llmopt:0.1.0-vllm0.32.0-rocm7.2.3` |
| Image digest | `sha256:7c503750d17f6e8adc05b0fee3cdbefd868b31f722f1dabdefda5b06034a361d` |
| Base image | `vllm/vllm-openai-rocm:v0.32.0` |
| Base digest | `sha256:14f6a575e1fb0a580b0f633c09e06f648709d60405b85f6a7ff3f80d49f39905` |
| ROCm | 7.2.3 |
| vLLM | 0.32.0+rocm723 |
| PyTorch | 2.13.0+git733fca1 |
| Python | 3.12.15 |

Built from `docker/Dockerfile` on the Phase 1 bench. The image is public on Docker Hub. It contains no secrets (checked environment variables only; the base image license terms were not reviewed).

Not yet run on a GPU. First real run: GPU session 1 (`scripts/gpu/session1.sh`).

The base image reports `org.opencontainers.image.version: 22.04`, which is a leftover label from its Ubuntu layer. The image is Ubuntu 24.04.