#!/usr/bin/env bash
# One isolated, long-lived rollout worker on one GPU.
#
# The worker command is read from R2ORCH_SHARD_COMMAND_FILE. It must write
# only SHARD_ROOT. This script never acquires run_root/driver.lock and never
# calls train.sh, so multiple single-GPU shards cannot corrupt the
# training driver's checkpoints or progress file.
#
# Usage: slurm_rollout_shard.sbatch <run_root> <shard_index> <shard_count>
set -euo pipefail

# Slurm compute nodes do not necessarily expose ``python`` until the
# project's environment is loaded.  The shard supervisor writes status with
# Python before it sources the user command file, so load the same env used by
# the main driver first.  Callers may override R2ORCH_ENV_SH; keeping this
# optional preserves portability for local dry-runs.
R2ORCH_ENV_SH="${R2ORCH_ENV_SH:-}"
if [[ -n "$R2ORCH_ENV_SH" && -f "$R2ORCH_ENV_SH" ]]; then
  # shellcheck disable=SC1090
  source "$R2ORCH_ENV_SH"
fi
if ! command -v python >/dev/null 2>&1 && [[ -n "${R2ORCH_SHARD_VENV:-}" && -x "$R2ORCH_SHARD_VENV/bin/python" ]]; then
  export PATH="$R2ORCH_SHARD_VENV/bin:$PATH"
fi
command -v python >/dev/null 2>&1 || {
  echo "python is unavailable; set R2ORCH_ENV_SH or R2ORCH_SHARD_VENV" >&2
  exit 127
}

RUN_ROOT="${1:?run_root}"
SHARD_INDEX="${2:?shard_index}"
SHARD_COUNT="${3:?shard_count}"
COMMAND_FILE="${R2ORCH_SHARD_COMMAND_FILE:?R2ORCH_SHARD_COMMAND_FILE is required}"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
SHARD_ROOT="$RUN_ROOT/rollout_shards/shard-$SHARD_INDEX"
mkdir -p "$SHARD_ROOT"

# A retry of the same shard exits without writing. Distinct shard indices use
# distinct lock files and can run concurrently.
command -v flock >/dev/null 2>&1 || {
  echo "flock is required on the compute node for shard isolation" >&2
  exit 2
}
exec 9>"$RUN_ROOT/rollout_shards/shard-$SHARD_INDEX.lock"
if ! flock -n 9; then
  echo "shard $SHARD_INDEX already has a worker; exiting safely" >&2
  exit 0
fi

export RUN_ROOT SHARD_INDEX SHARD_COUNT SHARD_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-$SHARD_ROOT/tmp}"
mkdir -p "$TMPDIR"
status="$SHARD_ROOT/status.json"
tmp_status="$status.tmp.$$"
write_status() {
  local state="$1" code="${2:-0}"
  python - "$tmp_status" "$state" "$code" "$status" <<'PY'
import json, os, sys, time
path, state, code, target = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"state": state, "exit_code": code,
               "shard_index": int(os.environ["SHARD_INDEX"]),
               "shard_count": int(os.environ["SHARD_COUNT"]),
               "updated_at": time.time()}, handle)
    handle.write("\n")
os.replace(path, target)
PY
}
trap 'code=$?; write_status failed "$code"; exit "$code"' EXIT
write_status running 0

[[ -f "$COMMAND_FILE" ]] || { echo "missing worker command $COMMAND_FILE" >&2; exit 2; }
# The command file is intentionally shell syntax so the owner can select a
# worker implementation. Its contract is that output stays below SHARD_ROOT.
source "$COMMAND_FILE"
if [[ -z "${R2ORCH_SHARD_COMMAND:-}" ]]; then
  echo "worker command file must set R2ORCH_SHARD_COMMAND" >&2
  exit 2
fi

echo "starting shard=$SHARD_INDEX/$SHARD_COUNT gpu=$CUDA_VISIBLE_DEVICES root=$SHARD_ROOT"
bash -o pipefail -c "$R2ORCH_SHARD_COMMAND" 2>&1 | tee -a "$SHARD_ROOT/worker.log"
write_status complete 0
trap - EXIT
