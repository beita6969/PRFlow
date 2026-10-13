#!/usr/bin/env bash
# Train R2Orch on one node.
#   scripts/r2orch/train.sh <run_root> [4gpu|3gpu|2gpu] [extra selfplay-experiment args...]
# Renders the configuration for the run, starts the retrieval service and the vLLM service (frozen base model with
# the Director and Proposer LoRA adapters), and runs the training driver.  The same command resumes a stopped run.
# Exit codes: 0 = all cycles done (writes <run_root>/DONE); 75 = stopped by <run_root>/STOP or at the end of the
# Slurm allocation (run the same command again); anything else = failure (logs in <run_root>/launch/).
# Environment:
#   R2ORCH_DATA_ROOT            data root that @DATA_ROOT@ in the configuration refers to (see README.md)
#   R2ORCH_CONFIG               source configuration (default configs/r2orch.toml)
#   R2ORCH_CURRICULUM           curriculum profile (default configs/curriculum/r2orch.toml)
#   R2ORCH_CYCLES               training cycles (default 200)
#   R2ORCH_SET                  "table.key=value ..." overrides applied when rendering
#   R2ORCH_TRAIN_FLAGS          learner flags (default: the settings of the paper, below)
#   R2ORCH_ROLLOUT_POOL_REGISTRY, R2ORCH_ROLLOUT_POOL_SIZE, R2ORCH_ROLLOUT_POOL_WAIT_S
#                               extra frozen-Worker replicas published by serve_rollout_shard.sh
#   R2ORCH_RETRIEVAL_WAIT_S     seconds to wait for the retrieval index to load (default 1800)
#   R2ORCH_PARALLEL_ROLES       set to update both adapters in parallel threads
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_ROOT="${1:?usage: train.sh <run_root> [4gpu|3gpu|2gpu] [extra args...]}"
LAYOUT="${2:-4gpu}"
shift $(($# >= 2 ? 2 : 1))
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

# Settings of the paper.
export R2ORCH_DIRECTOR_CONTEXT_MODE="${R2ORCH_DIRECTOR_CONTEXT_MODE:-append_only}"
export R2ORCH_OUTPUT_CEILING="${R2ORCH_OUTPUT_CEILING:-4096}"
export R2ORCH_RECOVERY_ATTEMPTS="${R2ORCH_RECOVERY_ATTEMPTS:-0}"
TRAIN_FLAGS="${R2ORCH_TRAIN_FLAGS:---lora-dropout-override 0 --fuse-old-logprobs --matmul-precision high \
--kl-reference-stride 4 --solver-learning-rate 1e-5 --proposer-learning-rate 1e-5 \
--kl-term-cap 10}"

# One driver per run directory.
exec 9>"$RUN_ROOT/driver.lock"
if ! flock -n 9; then
  printf 'another driver holds %s/driver.lock\n' "$RUN_ROOT" >&2
  exit 0
fi

SOURCE_CONFIG="${R2ORCH_CONFIG:-$R2ORCH_CODE/configs/r2orch.toml}"
r2orch_activate "$SOURCE_CONFIG"
overrides=()
for item in ${R2ORCH_SET:-}; do
  overrides+=(--set "$item")
done
RENDERED="$(python "$R2ORCH_SCRIPTS/render_config.py" --config "$SOURCE_CONFIG" \
  --run-root "$RUN_ROOT" --layout "$LAYOUT" ${overrides[@]+"${overrides[@]}"})"
if [[ -n "${R2ORCH_ROLLOUT_POOL_REGISTRY:-}" ]]; then
  POOL_CONFIG="$RUN_ROOT/launch/r2orch.rollout-pool.toml"
  python "$R2ORCH_SCRIPTS/configure_rollout_pool.py" \
    --source-config "$RENDERED" --output "$POOL_CONFIG" \
    --registry "$R2ORCH_ROLLOUT_POOL_REGISTRY" \
    --count "${R2ORCH_ROLLOUT_POOL_SIZE:-4}" \
    --timeout-s "${R2ORCH_ROLLOUT_POOL_WAIT_S:-1800}"
  RENDERED="$POOL_CONFIG"
fi
r2orch_load_env "$RENDERED"
LOGS="$RUN_ROOT/launch"
STAMP="$(date +%Y%m%d-%H%M%S)${SLURM_JOB_ID:+-$SLURM_JOB_ID}"
mkdir -p "$LOGS" "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"
export ALFWORLD_DATA="$R2ORCH_ALFWORLD_DATA_ROOT"
printf '[%s] driver start run_root=%s layout=%s gpus=%s\n' "$(date -Is)" "$RUN_ROOT" "$LAYOUT" \
  "${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LOGS/driver.log"

RETRIEVAL_PID=""
cleanup() {
  if [[ -n "$RETRIEVAL_PID" ]]; then
    kill "$RETRIEVAL_PID" 2>/dev/null || true
  fi
  python -m r2orch model-services stop --config "$RENDERED" \
    --state-dir "$RUN_ROOT/services" >>"$LOGS/vllm-$STAMP.log" 2>&1 || true
}
trap cleanup EXIT

retrieval_healthy() {
  python - "$R2ORCH_RETRIEVAL_PORT" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=5).read()
PY
}

# The retrieval index (~32 GB of vectors) and the vLLM service start in parallel.
if ! retrieval_healthy; then
  bash "$R2ORCH_SCRIPTS/serve_retrieval.sh" "$RENDERED" >>"$LOGS/retrieval-$STAMP.log" 2>&1 &
  RETRIEVAL_PID=$!
fi
python -m r2orch model-services start --config "$RENDERED" \
  --state-dir "$RUN_ROOT/services" >>"$LOGS/vllm-$STAMP.log" 2>&1 &
VLLM_START_PID=$!

deadline=$((SECONDS + ${R2ORCH_RETRIEVAL_WAIT_S:-1800}))
until retrieval_healthy; do
  if [[ -n "$RETRIEVAL_PID" ]] && ! kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
    printf 'retrieval service exited; see %s\n' "$LOGS/retrieval-$STAMP.log" >&2
    exit 3
  fi
  if ((SECONDS > deadline)); then
    printf 'retrieval service not healthy in time; see %s\n' "$LOGS/retrieval-$STAMP.log" >&2
    exit 3
  fi
  sleep 5
done
if ! wait "$VLLM_START_PID"; then
  printf 'vLLM failed to start; see %s and %s/services/\n' "$LOGS/vllm-$STAMP.log" "$RUN_ROOT" >&2
  exit 3
fi

# Inside a Slurm allocation, stop one minute before it ends (the driver keeps its own margins).
if [[ -n "${SLURM_JOB_ID:-}" ]] && end_epoch="$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null)"; then
  export "$R2ORCH_STOP_ENV"="$((end_epoch - 60))"
fi

set +e
python -m r2orch selfplay-experiment \
  --config "$RENDERED" \
  --task-pool "$R2ORCH_TRAIN_POOL" \
  --curriculum-profile "${R2ORCH_CURRICULUM:-$R2ORCH_CODE/configs/curriculum/r2orch.toml}" \
  --output "$RUN_ROOT" \
  --cycles "${R2ORCH_CYCLES:-200}" \
  --async-next-cycle-rollouts \
  ${R2ORCH_PARALLEL_ROLES:+--parallel-role-training} --max-sequence-length 35000 \
  --max-micro-batch-tokens 35000 --micro-batch-size 1 \
  --raw-policy-backward-mode timeline \
  --activation-cpu-offload --activation-cpu-offload-min-tokens 4096 \
  $TRAIN_FLAGS \
  --resume "$@" >>"$LOGS/experiment-$STAMP.log" 2>&1
code=$?
set -e
printf '[%s] driver exit code=%s\n' "$(date -Is)" "$code" | tee -a "$LOGS/driver.log"
if [[ "$code" == 0 ]]; then
  date -Is >"$RUN_ROOT/DONE"
elif [[ "$code" != "$R2ORCH_RESUME_EXIT_CODE" ]]; then
  tail -n 40 "$LOGS/experiment-$STAMP.log" >&2 || true
fi
exit "$code"
