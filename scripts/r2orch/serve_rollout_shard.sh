#!/usr/bin/env bash
# Serve the frozen Worker model on one isolated GPU and publish its endpoint.
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
SHARD_INDEX="${SHARD_INDEX:?SHARD_INDEX is required}"
SHARD_ROOT="${SHARD_ROOT:?SHARD_ROOT is required}"
CODE="${R2ORCH_SHARD_CODE:?R2ORCH_SHARD_CODE is required}"
VENV="${R2ORCH_SHARD_VENV:-$CODE/.venv}"
MODEL="${R2ORCH_ROLLOUT_MODEL:?set R2ORCH_ROLLOUT_MODEL to the Qwen3.5-9B model directory}"
REGISTRY="$RUN_ROOT/rollout_shards"
PORT=$(( ${R2ORCH_ROLLOUT_PORT_BASE:-19600} + SHARD_INDEX ))
HOST="$(hostname -s)"
ENDPOINT="$REGISTRY/shard-$SHARD_INDEX.json"
# Keep the probe credentials identical to the OpenAI-compatible server.  vLLM
# protects /v1/models when --api-key is set, so an unauthenticated probe would
# wait until the startup deadline even though the model is already ready.
API_KEY="${R2ORCH_ROLLOUT_API_KEY:-EMPTY}"
mkdir -p "$SHARD_ROOT" "$REGISTRY"

MAX_RESTARTS="${R2ORCH_ROLLOUT_MAX_RESTARTS:-3}"
BACKOFF_S="${R2ORCH_ROLLOUT_RESTART_BACKOFF_S:-15}"
BACKOFF_MAX_S="${R2ORCH_ROLLOUT_RESTART_BACKOFF_MAX_S:-120}"
OWNER_RUN="${R2ORCH_SINGLE_GPU_OWNER_RUN:-}"
OWNER_CHECK_S="${R2ORCH_ROLLOUT_OWNER_CHECK_S:-30}"
NEXT_OWNER_CHECK=0
EXIT_REASON=supervisor_exit
export SHARD_HOST="$HOST" SHARD_PORT="$PORT" SHARD_INDEX
export SHARD_SERVICE_ATTEMPT=0
VLLM_PID=""
LAST_STATE=""

write_state() {
  local state="$1" code="${2:-0}" reason="${3:-}" tmp="$ENDPOINT.tmp.$$"
  SHARD_SERVICE_PID="$VLLM_PID" python - "$tmp" "$state" "$ENDPOINT" "$code" "$reason" "$SHARD_ROOT/service_events.jsonl" <<'PY'
import json, os, sys, time
tmp, state, target, code, reason, events = sys.argv[1:]
attempt = int(os.environ["SHARD_SERVICE_ATTEMPT"])
pid = os.environ.get("SHARD_SERVICE_PID", "")
payload = {"state": state, "host": os.environ["SHARD_HOST"],
           "port": int(os.environ["SHARD_PORT"]),
           "shard_index": int(os.environ["SHARD_INDEX"]),
           "job_id": os.environ.get("SLURM_JOB_ID"),
           "service_pid": int(pid) if pid else None,
           "service_attempt": attempt, "restart_count": max(0, attempt - 1),
           "exit_code": int(code), "reason": reason, "updated_at": time.time()}
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
    handle.write("\n")
os.replace(tmp, target)
with open(events, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload) + "\n")
PY
  LAST_STATE="$state"
}

cleanup() {
  local code=$?
  trap - EXIT TERM INT
  # Invalidate ready before terminating the service; readers must never reuse
  # an endpoint that this supervisor is shutting down.
  if [[ "$code" == 0 ]]; then
    write_state stopped 0 "$EXIT_REASON"
  elif [[ "$LAST_STATE" != failed ]]; then
    write_state failed "$code" supervisor_exit
  fi
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
write_state starting 0 environment_setup

for value in "$MAX_RESTARTS" "$BACKOFF_S" "$BACKOFF_MAX_S" "$OWNER_CHECK_S"; do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "Worker restart limits and backoff must be nonnegative integers" >&2
    exit 2
  }
done

check_owner_done() {
  if (( SECONDS >= NEXT_OWNER_CHECK )); then
    NEXT_OWNER_CHECK=$((SECONDS + OWNER_CHECK_S))
    if [[ -n "$OWNER_RUN" && -f "$OWNER_RUN/DONE" ]]; then
      echo "Worker owner completed at $OWNER_RUN/DONE; stopping this shard's service"
      EXIT_REASON=owner_done
      exit 0
    fi
  fi
}

restart_backoff() {
  local remaining="$1" chunk
  while (( remaining > 0 )); do
    check_owner_done
    chunk="$remaining"
    (( chunk <= 30 )) || chunk=30
    sleep "$chunk"
    remaining=$((remaining - chunk))
  done
}

set +u
if [[ -f "$(dirname "$VENV")/env.sh" ]]; then
  source "$(dirname "$VENV")/env.sh"
else
  source "$VENV/bin/activate"
fi
set -u
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Qwen3.5's FlashInfer sampler can trigger a CUDA-header JIT mismatch.  Match the
# resident service path: Triton GDN prefill and the native sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Decode-latency knobs.  CUDA graphs (vLLM default compilation config) are the default: measured on
# an A100 with LoRA enabled at 32 concurrent requests they decode 2.7x (base) to 3.3x (LoRA)
# faster than --enforce-eager.  R2ORCH_VLLM_ENFORCE_EAGER=1 (or R2ORCH_ROLLOUT_CUDAGRAPH=eager) restores eager.
# Eager execution is CPU/launch bound (~32 ms per decode step here, ~35.6 ms per output token end to end).
# Options:
#   R2ORCH_ROLLOUT_CUDAGRAPH      eager | decode | breakable | default   (see vllm_serving.py)
#   R2ORCH_ROLLOUT_PREFIX_CACHING 1 | 0
#   R2ORCH_ROLLOUT_MAX_NUM_SEQS / R2ORCH_ROLLOUT_MAX_NUM_BATCHED_TOKENS   integers (unset: vLLM default)
#   R2ORCH_ROLLOUT_EXTRA_ARGS     extra vLLM flags, word-split (e.g. "--max-num-seqs 128")
VLLM_MODE_ARGS=()
ROLLOUT_GRAPH_MODE="${R2ORCH_ROLLOUT_CUDAGRAPH:-}"
if [[ -z "$ROLLOUT_GRAPH_MODE" ]]; then
  case "${R2ORCH_VLLM_ENFORCE_EAGER:-0}" in 1|true|yes|on) ROLLOUT_GRAPH_MODE=eager ;; *) ROLLOUT_GRAPH_MODE=default ;; esac
fi
case "$ROLLOUT_GRAPH_MODE" in
  eager) VLLM_MODE_ARGS=(--enforce-eager) ;;
  decode) VLLM_MODE_ARGS=(--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  breakable)
    VLLM_MODE_ARGS=(--compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_AND_PIECEWISE"}')
    export VLLM_USE_BREAKABLE_CUDAGRAPH=1 ;;
  default) ;;
  *) echo "R2ORCH_ROLLOUT_CUDAGRAPH must be eager, decode, breakable or default" >&2; exit 2 ;;
esac
if [[ "${R2ORCH_ROLLOUT_PREFIX_CACHING:-1}" == 0 ]]; then
  VLLM_MODE_ARGS+=(--no-enable-prefix-caching)
else
  VLLM_MODE_ARGS+=(--enable-prefix-caching)
fi
[[ -z "${R2ORCH_ROLLOUT_MAX_NUM_SEQS:-}" ]] || VLLM_MODE_ARGS+=(--max-num-seqs "$R2ORCH_ROLLOUT_MAX_NUM_SEQS")
[[ -z "${R2ORCH_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}" ]] || \
  VLLM_MODE_ARGS+=(--max-num-batched-tokens "$R2ORCH_ROLLOUT_MAX_NUM_BATCHED_TOKENS")
VLLM_EXTRA_ARGS=()
if [[ -n "${R2ORCH_ROLLOUT_EXTRA_ARGS:-}" ]]; then
  read -r -a VLLM_EXTRA_ARGS <<<"$R2ORCH_ROLLOUT_EXTRA_ARGS"
fi

probe_ready() {
  python - "$HOST" "$PORT" "$API_KEY" <<'PY'
import sys, urllib.request
try:
    request = urllib.request.Request(
        f"http://{sys.argv[1]}:{sys.argv[2]}/v1/models",
        headers={"Authorization": f"Bearer {sys.argv[3]}"},
    )
    urllib.request.urlopen(request, timeout=5).read()
except Exception:
    raise SystemExit(1)
PY
}

# The sbatch parent holds this shard's lock for the entire loop. Restarting
# only this child leaves other shards, training state and GPU layout untouched.
while true; do
  check_owner_done
  export SHARD_SERVICE_ATTEMPT=$((SHARD_SERVICE_ATTEMPT + 1))
  VLLM_PID=""
  write_state starting 0 service_launch
  printf '\n--- vLLM service attempt %s ---\n' "$SHARD_SERVICE_ATTEMPT" >>"$SHARD_ROOT/vllm.log"
  python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name r2orch-base \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${R2ORCH_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-model-len "${R2ORCH_ROLLOUT_MAX_MODEL_LEN:-32768}" \
  "${VLLM_MODE_ARGS[@]}" \
  --enable-chunked-prefill \
  --gdn-prefill-backend triton \
  ${VLLM_EXTRA_ARGS[@]+"${VLLM_EXTRA_ARGS[@]}"} \
  --api-key "$API_KEY" >>"$SHARD_ROOT/vllm.log" 2>&1 &
  VLLM_PID=$!
  write_state starting 0 service_loading
  deadline=$((SECONDS + ${R2ORCH_ROLLOUT_START_WAIT_S:-1800}))
  startup_warning=0
  while kill -0 "$VLLM_PID" 2>/dev/null; do
    check_owner_done
    if probe_ready; then
      # A successful probe alone is insufficient if our child already exited.
      kill -0 "$VLLM_PID" 2>/dev/null || break
      write_state ready 0 serving
      echo "rollout shard $SHARD_INDEX ready at http://$HOST:$PORT/v1 attempt=$SHARD_SERVICE_ATTEMPT"
      break
    fi
    kill -0 "$VLLM_PID" 2>/dev/null || break
    if (( SECONDS >= deadline && startup_warning == 0 )); then
      echo "vLLM startup exceeded readiness window but process $VLLM_PID is alive; continuing to wait"
      write_state starting 0 startup_slow_process_alive
      startup_warning=1
    fi
    sleep 5
  done

  while kill -0 "$VLLM_PID" 2>/dev/null; do
    check_owner_done
    sleep 5
  done

  # Only wait's confirmed child exit can authorize a retry. A slow loading
  # process or an unresponsive-but-live service is never restarted here.
  if wait "$VLLM_PID"; then code=0; else code=$?; fi
  write_state failed "$code" service_process_exited
  tail -n 80 "$SHARD_ROOT/vllm.log" || true
  if (( SHARD_SERVICE_ATTEMPT > MAX_RESTARTS )); then
    (( code != 0 )) || code=3
    write_state failed "$code" restart_limit_exhausted
    exit "$code"
  fi
  backoff="$BACKOFF_S"
  for ((n = 1; n < SHARD_SERVICE_ATTEMPT && backoff < BACKOFF_MAX_S; n++)); do
    backoff=$((backoff * 2))
  done
  (( backoff <= BACKOFF_MAX_S )) || backoff="$BACKOFF_MAX_S"
  echo "vLLM process exited code=$code; restart $SHARD_SERVICE_ATTEMPT/$MAX_RESTARTS in ${backoff}s"
  restart_backoff "$backoff"
done
