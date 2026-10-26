# Task 1.14: llmopt on top of a PINNED ROCm vLLM image.
# Pass the exact image on the command line; there is deliberately no default.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/llmopt
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps . \
 && python3 -m pip install --no-cache-dir "typer>=0.9" "pyyaml>=6.0" "pydantic>=2.0" typing_extensions \
 && (python3 -m pip check > /opt/llmopt/pip_check.txt 2>&1 || true)

COPY docker/record_versions.sh /usr/local/bin/record_versions.sh
RUN chmod +x /usr/local/bin/record_versions.sh

# The base image may start a server by default; reset it for session use.
ENTRYPOINT []
CMD ["bash"]