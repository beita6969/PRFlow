#!/usr/bin/env bash
# Serve wiki-18 E5 retrieval (retrieval_service.py) for one rendered configuration.
#   serve_retrieval.sh <rendered_config>
# Device, port, corpus and encoder come from r2orch.serving / r2orch.paths:
# 4gpu puts the fp16 matrix (~32 GB) on cuda:3, 2gpu keeps it in RAM (bf16, 16 threads).
# POST /retrieve {"queries": [...], "topk": 3}; GET /health.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG="${1:?usage: serve_retrieval.sh <rendered_config>}"
r2orch_activate "$CONFIG"
r2orch_load_env "$CONFIG"
threads=()
if [[ "$R2ORCH_RETRIEVAL_DEVICE" == cpu ]]; then
  threads=(--threads "${R2ORCH_RETRIEVAL_THREADS:-16}")
fi
exec python "$R2ORCH_RETRIEVAL_SCRIPT" serve "$R2ORCH_WIKI18_DIR" "$R2ORCH_E5_MODEL" \
  --host 127.0.0.1 --port "$R2ORCH_RETRIEVAL_PORT" --device "$R2ORCH_RETRIEVAL_DEVICE" \
  ${threads[@]+"${threads[@]}"}
