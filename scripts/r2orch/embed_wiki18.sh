#!/usr/bin/env bash
# Build <paths.wiki18_dir>/vectors_fp16.npy with E5 on a GPU node, one shard per visible GPU.
#   embed_wiki18.sh [config] [num_shards]
# config defaults to configs/r2orch.toml (only paths.* are read). Resumable: each shard keeps its
# own progress file; rerun the same command (same num_shards) after a job boundary. Exits at once
# when the vectors already exist.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG="${1:-$R2ORCH_CODE/configs/r2orch.toml}"
r2orch_activate "$CONFIG"
RENDERED="$(python "$R2ORCH_SCRIPTS/render_config.py" --config "$CONFIG" --run-root "$(mktemp -d)")"
r2orch_load_env "$RENDERED"
SCRIPT="$R2ORCH_RETRIEVAL_SCRIPT"
WIKI="$R2ORCH_WIKI18_DIR"
if [[ -f "$WIKI/vectors_fp16.npy" ]]; then
  printf 'vectors ready: %s\n' "$WIKI/vectors_fp16.npy"
  exit 0
fi
IFS=, read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES:-0}"
SHARDS="${2:-${#gpus[@]}}"
LOGS="$WIKI/embed-logs"
mkdir -p "$LOGS"

python "$SCRIPT" embed-init "$WIKI"
pids=()
for ((shard = 0; shard < SHARDS; shard++)); do
  gpu="${gpus[$((shard % ${#gpus[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" embed "$WIKI" "$R2ORCH_E5_MODEL" \
    --device cuda:0 --batch "${R2ORCH_EMBED_BATCH:-512}" --shard "$shard" --num-shards "$SHARDS" \
    >>"$LOGS/shard$shard-of-$SHARDS.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if [[ "$status" != 0 ]]; then
  printf 'an embed shard failed (exit %s); see %s\n' "$status" "$LOGS" >&2
  exit "$status"
fi
python "$SCRIPT" embed-finalize "$WIKI" --num-shards "$SHARDS"
