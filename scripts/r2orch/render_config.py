#!/usr/bin/env python3
"""Render a run configuration (default configs/r2orch.toml) for one run root and one GPU layout.

  render_config.py --run-root <dir> [--data-root <dir>] [--layout 4gpu|3gpu|2gpu|2gpu_serial|3gpu_dp]
                   [--config <toml>] [--output <toml>] [--set table.key=value ...] [--print-env]
      Replace every @RUN_ROOT@, @DATA_ROOT@ (--data-root or $R2ORCH_DATA_ROOT) and @CODE_ROOT@ (this
      checkout), make relative paths that name files of the code checkout
      absolute (the rendered file no longer sits in configs/), rewrite the GPU fields of
      [resources] and [r2orch.serving] for the layout (3gpu also moves the Worker route to its
      own vLLM), apply --set overrides and write
      <run_root>/r2orch.rendered.toml (comments and layout of the file are kept).
  render_config.py --env-from <rendered toml>
      Print shell assignments (R2ORCH_*) for the launch scripts; use with eval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tomllib
from pathlib import Path

PLACEHOLDER = "@RUN_ROOT@"
DATA_PLACEHOLDER = "@DATA_ROOT@"
CODE_PLACEHOLDER = "@CODE_ROOT@"
ROOT = Path(__file__).resolve().parents[2]
SERVING = "r2orch.serving"

# GPU layouts. The layout lives here only; code reads the rendered fields.
LAYOUTS: dict[str, dict[tuple[str, str], object]] = {
    "4gpu": {
        (SERVING, "layout"): "4gpu",
        (SERVING, "vllm_gpu_ids"): [0, 1],
        (SERVING, "vllm_data_parallel_size"): 2,
        (SERVING, "vllm_gpu_memory_utilization"): 0.90,
        (SERVING, "retrieval_device"): "cuda:3",
        (SERVING, "retrieval_gpu_id"): 3,
        (SERVING, "director_train_gpu_id"): 2,
        (SERVING, "proposer_train_gpu_id"): 3,
        ("resources", "solver_gpu_id"): 2,
        ("resources", "proposer_gpu_id"): 3,
        ("resources", "allocated_gpu_ids"): [0, 1, 2, 3],
    },
    "2gpu": {
        (SERVING, "layout"): "2gpu",
        (SERVING, "vllm_gpu_ids"): [0],
        (SERVING, "vllm_data_parallel_size"): 1,
        (SERVING, "vllm_gpu_memory_utilization"): 0.55,
        (SERVING, "retrieval_device"): "cpu",
        (SERVING, "director_train_gpu_id"): 1,
        (SERVING, "proposer_train_gpu_id"): 0,
        ("resources", "solver_gpu_id"): 1,
        ("resources", "proposer_gpu_id"): 0,
        ("resources", "allocated_gpu_ids"): [0, 1],
    },
    # Three GPUs in one allocation (no external Worker shard jobs, so the 16-request-per-endpoint
    # cap of remote shards does not apply):
    #   GPU0 resident vLLM for the Director/Proposer LoRAs (hot-reloaded after every update),
    #   GPU1 a second vLLM that serves only the frozen base (Workers, PATS reviewer, controlled
    #        reruns) on its own port, so Worker decoding never pays LoRA overhead and an adapter
    #        publish (prefix-cache reset) no longer flushes the Worker cache,
    #   GPU2 both trainers, one after the other (the cycle pipeline overlaps training with the next
    #        collection).  The Worker route of the rendered config points at the Worker vLLM.
    "3gpu": {
        (SERVING, "layout"): "3gpu",
        (SERVING, "vllm_gpu_ids"): [0],
        (SERVING, "vllm_data_parallel_size"): 1,
        (SERVING, "vllm_gpu_memory_utilization"): 0.90,
        (SERVING, "worker_port"): 18601,
        (SERVING, "worker_gpu_ids"): [1],
        (SERVING, "worker_gpu_memory_utilization"): 0.90,
        (SERVING, "retrieval_device"): "cpu",
        (SERVING, "director_train_gpu_id"): 2,
        (SERVING, "proposer_train_gpu_id"): 2,
        ("resources", "solver_gpu_id"): 2,
        ("resources", "proposer_gpu_id"): 2,
        ("resources", "allocated_gpu_ids"): [0, 1, 2],
        ("resources", "allow_policy_gpu_colocation"): True,
        ("runtime", "base_url"): "http://127.0.0.1:18601/v1",
    },
    # Roles train one after the other (train.sh default), so both trainers share GPU1
    # and the resident vLLM gets GPU0 to itself: about 2.5x the KV cache of "2gpu".
    "2gpu_serial": {
        (SERVING, "layout"): "2gpu_serial",
        (SERVING, "vllm_gpu_ids"): [0],
        (SERVING, "vllm_data_parallel_size"): 1,
        (SERVING, "vllm_gpu_memory_utilization"): 0.88,
        (SERVING, "retrieval_device"): "cpu",
        (SERVING, "director_train_gpu_id"): 1,
        (SERVING, "proposer_train_gpu_id"): 1,
        ("resources", "solver_gpu_id"): 1,
        ("resources", "proposer_gpu_id"): 1,
        ("resources", "allocated_gpu_ids"): [0, 1],
        ("resources", "allow_policy_gpu_colocation"): True,
    },
    # 3 GPUs: GPU0 serves vLLM alone; the Director update is sample-level data parallel over
    # GPUs 1+2 (launch with `--solver-data-parallel-gpus 1 2`, e.g. via R2ORCH_TRAIN_FLAGS).
    # Roles train one after the other, so the Proposer trains on GPU1 first and both
    # Director ranks then start from a clean GPU.  The Director's in-process device
    # (solver_gpu_id) is only a placeholder: the two-card worker pins its own ranks.
    "3gpu_dp": {
        (SERVING, "layout"): "3gpu_dp",
        (SERVING, "vllm_gpu_ids"): [0],
        (SERVING, "vllm_data_parallel_size"): 1,
        (SERVING, "vllm_gpu_memory_utilization"): 0.88,
        (SERVING, "retrieval_device"): "cpu",
        (SERVING, "director_train_gpu_id"): 1,
        (SERVING, "proposer_train_gpu_id"): 1,
        ("resources", "solver_gpu_id"): 1,
        ("resources", "proposer_gpu_id"): 1,
        ("resources", "allocated_gpu_ids"): [0, 1, 2],
        ("resources", "allow_policy_gpu_colocation"): True,
    },
}


def toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_literal(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def set_value(text: str, table: str, key: str, value: object) -> str:
    """Set ``key`` inside ``[table]``; add the key or the table when missing."""

    lines = text.splitlines()
    header = re.compile(rf"^\s*\[\s*{re.escape(table)}\s*\]\s*(#.*)?$")
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    assignment = f"{key} = {toml_literal(value)}"
    if start is None:
        return text.rstrip("\n") + f"\n\n[{table}]\n{assignment}\n"
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"^\s*\[", lines[i])), len(lines)
    )
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(start + 1, end):
        if pattern.match(lines[index]):
            stop = index + 1
            depth = lines[index].count("[") - lines[index].count("]")
            while depth > 0 and stop < end:  # a multi-line array value
                depth += lines[stop].count("[") - lines[stop].count("]")
                stop += 1
            lines[index:stop] = [assignment]
            break
    else:
        insert = end
        while insert > start + 1 and not lines[insert - 1].strip():
            insert -= 1
        lines.insert(insert, assignment)
    return "\n".join(lines) + "\n"


def parse_override(item: str) -> tuple[str, str, object]:
    dotted, _, raw = item.partition("=")
    table, _, key = dotted.strip().rpartition(".")
    if not table or not key or not _:
        raise ValueError(f"--set expects table.key=value, got {item!r}")
    try:
        value = tomllib.loads(f"value = {raw.strip()}")["value"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return table, key, value


def config_root(config: Path) -> Path:
    """Directory that load_adaptive_config resolves relative paths against."""
    config = Path(config).resolve()
    return config.parent.parent if config.parent.name == "configs" else config.parent


_STRING_VALUE = re.compile(r'^(\s*[A-Za-z0-9_-]+\s*=\s*)"([^"]+)"(.*)$')


def absolutize_paths(text: str, root: Path) -> str:
    """The rendered file lives in the run root, so a relative value that names an existing
    file of the code checkout (seed cards, templates) is made absolute."""

    lines = []
    for line in text.splitlines():
        match = _STRING_VALUE.match(line)
        if match and "/" in match.group(2) and not Path(match.group(2)).is_absolute():
            candidate = root / match.group(2)
            if candidate.exists():
                line = f'{match.group(1)}{json.dumps(str(candidate.resolve()))}{match.group(3)}'
        lines.append(line)
    return "\n".join(lines) + "\n"


def render(
    source: str,
    *,
    run_root: Path,
    layout: str,
    overrides: list[str],
    source_root: Path = ROOT,
    data_root: Path | None = None,
) -> str:
    if data_root is None:
        data_root = os.environ.get("R2ORCH_DATA_ROOT") or None
    if DATA_PLACEHOLDER in source:
        if data_root is None:
            raise ValueError("the config uses @DATA_ROOT@: pass --data-root or set R2ORCH_DATA_ROOT")
        source = source.replace(DATA_PLACEHOLDER, str(Path(data_root).resolve()))
    source = source.replace(CODE_PLACEHOLDER, str(Path(source_root).resolve()))
    text = absolutize_paths(source.replace(PLACEHOLDER, str(run_root)), source_root)
    for (table, key), value in LAYOUTS[layout].items():
        text = set_value(text, table, key, value)
    for item in overrides:
        text = set_value(text, *parse_override(item))
    tomllib.loads(text)  # the rendered file must stay valid TOML
    return text


def shell_env(rendered: Path) -> dict[str, str]:
    with rendered.open("rb") as handle:
        payload = tomllib.load(handle)
    r2orch = payload["r2orch"]
    paths, serving, launch = r2orch["paths"], r2orch["serving"], r2orch["launch"]
    routing = payload.get("runtime_routing", {})
    routes = routing.get("worker_routes") or [payload.get("runtime", {}).get("name", "qwen")]
    return {
        "R2ORCH_RENDERED_CONFIG": str(rendered.resolve()),
        "R2ORCH_RUN_ROOT": str(paths["run_root"]),
        "R2ORCH_RUNS_DIR": str(paths["runs_dir"]),
        "R2ORCH_CODE_ROOT": str(paths["code_root"]),
        "R2ORCH_VENV": str(paths["venv"]),
        "R2ORCH_BASE_MODEL": str(paths["base_model"]),
        "R2ORCH_E5_MODEL": str(paths["e5_model"]),
        "R2ORCH_WIKI18_DIR": str(paths["wiki18_dir"]),
        "R2ORCH_TRAIN_POOL": str(paths["train_pool"]),
        "R2ORCH_ALFWORLD_DATA_ROOT": str(paths.get("alfworld_data_root", "")),
        "R2ORCH_RETRIEVAL_SCRIPT": str(paths["retrieval_service_script"]),
        "R2ORCH_LAYOUT": str(serving["layout"]),
        "R2ORCH_VLLM_PORT": str(serving["vllm_port"]),
        "R2ORCH_VLLM_GPUS": ",".join(str(gpu) for gpu in serving["vllm_gpu_ids"]),
        "R2ORCH_WORKER_PORT": str(serving.get("worker_port", 0)),
        "R2ORCH_WORKER_GPUS": ",".join(str(gpu) for gpu in serving.get("worker_gpu_ids", [])),
        "R2ORCH_BASE_SERVED_NAME": str(serving["base_served_name"]),
        "R2ORCH_RETRIEVAL_PORT": str(serving["retrieval_port"]),
        "R2ORCH_RETRIEVAL_DEVICE": str(serving["retrieval_device"]),
        "R2ORCH_ALLOCATED_GPUS": ",".join(
            str(gpu) for gpu in payload.get("resources", {}).get("allocated_gpu_ids", [])
        ),
        "R2ORCH_WORKER_ROUTES": json.dumps(list(routes)),
        "R2ORCH_STOP_ENV": str(launch["stop_env"]),
        "R2ORCH_STOP_FILE": str(launch["stop_file"]),
        "R2ORCH_RESUME_EXIT_CODE": str(launch["resume_exit_code"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "r2orch.toml")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("R2ORCH_DATA_ROOT") or None)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="4gpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--print-env", action="store_true")
    parser.add_argument("--env-from", type=Path)
    args = parser.parse_args(argv)
    if args.env_from is not None:
        rendered = args.env_from
    else:
        if args.run_root is None:
            parser.error("--run-root is required unless --env-from is given")
        run_root = args.run_root.resolve()
        rendered = args.output or run_root / "r2orch.rendered.toml"
        text = render(
            args.config.read_text(encoding="utf-8"),
            run_root=run_root,
            layout=args.layout,
            overrides=args.overrides,
            source_root=config_root(args.config),
            data_root=args.data_root,
        )
        rendered.parent.mkdir(parents=True, exist_ok=True)
        rendered.write_text(text, encoding="utf-8")
        if not args.print_env:
            print(rendered)
            return 0
    for key, value in shell_env(rendered).items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
