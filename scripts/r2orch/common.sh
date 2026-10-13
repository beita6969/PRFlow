# Shared helpers for scripts/r2orch/*.sh. Source it; do not execute it.
# Every path comes from the configuration (paths.*); nothing cluster-specific is hardcoded here.

R2ORCH_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2ORCH_CODE="$(cd "$R2ORCH_SCRIPTS/../.." && pwd)"

# Activate the environment named by paths.venv when it exists (the config may be unrendered). An env.sh next to the
# venv (written by a cluster build script, e.g. to load modules) wins over <venv>/bin/activate. Without either, the
# current Python environment is used.
r2orch_activate() {
  local venv
  venv="$(sed -n 's/^[[:space:]]*venv[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$1" | head -n 1)"
  venv="${venv//@CODE_ROOT@/$R2ORCH_CODE}"
  venv="${venv//@DATA_ROOT@/${R2ORCH_DATA_ROOT:-}}"
  set +u
  if [[ -n "$venv" && -f "$(dirname "$venv")/env.sh" ]]; then
    source "$(dirname "$venv")/env.sh"
  elif [[ -n "$venv" && -f "$venv/bin/activate" ]]; then
    source "$venv/bin/activate"
  fi
  set -u
  export PYTHONPATH="$R2ORCH_CODE/src${PYTHONPATH:+:$PYTHONPATH}"
  # Compute nodes have no internet; models are loaded from paths.* only.
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
}

# Export R2ORCH_* from a (rendered) config and keep caches out of the small home quota.
r2orch_load_env() {
  eval "$(python "$R2ORCH_SCRIPTS/render_config.py" --env-from "$1")"
  # AdaptiveApplicationConfig.validate() only admits [resources] GPUs listed here
  # (default "0"); inside the allocation these are the logical ids of the layout.
  export R2ORCH_ALLOWED_PHYSICAL_GPUS="${R2ORCH_ALLOWED_PHYSICAL_GPUS:-$R2ORCH_ALLOCATED_GPUS}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$R2ORCH_RUNS_DIR/.cache}"
  export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
}
