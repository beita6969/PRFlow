#!/usr/bin/env bash
# Offline check of the training loop: two cycles of configs/mock/r2orch_mock.toml on a tiny pool, with mock models,
# mock rewards and mock policy updates (no GPU, service or model weights).
#   scripts/r2orch/mock_cycle.sh [run_dir] [cycles]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
run_dir="${1:-runs/mock}"
cycles="${2:-2}"
if [[ -e "$run_dir" ]]; then
  echo "Choose a new output directory: $run_dir already exists." >&2
  exit 1
fi
mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# The rendered 4gpu layout names GPUs 0-3; nothing runs on them in the mock.
export R2ORCH_ALLOWED_PHYSICAL_GPUS="${R2ORCH_ALLOWED_PHYSICAL_GPUS:-0,1,2,3}"
rendered="$("$PYTHON" scripts/r2orch/render_config.py --config configs/mock/r2orch_mock.toml --run-root "$run_dir")"
"$PYTHON" -m r2orch selfplay-experiment \
  --config "$rendered" \
  --task-pool configs/mock/pool.jsonl \
  --curriculum-profile configs/mock/curriculum.toml \
  --output "$run_dir" --cycles "$cycles" --mock --mock-trainer --resume >"$run_dir/experiment.log"
printf 'mock run finished: %s (cycle directories, dashboard.json, experiment.log)\n' "$run_dir"
