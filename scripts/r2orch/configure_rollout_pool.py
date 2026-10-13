#!/usr/bin/env python3
"""Build a rendered config that routes frozen Worker calls to GPU shards."""

from __future__ import annotations

import argparse
import json
import re
import time
import tomllib
from pathlib import Path
from urllib.request import Request, urlopen


def _toml(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(
            f"{json.dumps(str(key), ensure_ascii=False)} = {_toml(item)}"
            for key, item in value.items()
        ) + " }"
    return json.dumps(str(value), ensure_ascii=False)


def _table_path(line: str) -> tuple[str, ...] | None:
    """Read a rendered TOML table header, including quoted names and child tables."""
    if not line.lstrip().startswith("["):
        return None
    try:
        table = tomllib.loads(line)
    except tomllib.TOMLDecodeError:
        return None
    path: list[str] = []
    while isinstance(table, dict) and len(table) == 1:
        key, table = next(iter(table.items()))
        path.append(key)
        if isinstance(table, list) and len(table) == 1:
            table = table[0]
    return tuple(path) or None


def _table_blocks(source_text: str) -> list[tuple[tuple[str, ...] | None, list[str]]]:
    blocks: list[tuple[tuple[str, ...] | None, list[str]]] = [(None, [])]
    multiline: str | None = None
    for line in source_text.splitlines():
        path = _table_path(line) if multiline is None else None
        if path is not None:
            blocks.append((path, []))
        blocks[-1][1].append(line)
        multiline = _multiline_string_state(line, multiline)
    return blocks


def _multiline_string_state(line: str, multiline: str | None) -> str | None:
    """Avoid treating headers inside multiline prompt strings as real tables."""
    index = 0
    while index < len(line):
        if multiline is not None:
            if multiline == '"""' and line[index] == "\\":
                index += 2
            elif line.startswith(multiline, index):
                index += 3
                multiline = None
            else:
                index += 1
        elif line[index] == "#":
            break
        elif line[index] in "\"'":
            quote = line[index]
            if line.startswith(quote * 3, index):
                multiline = quote * 3
                index += 3
                continue
            index += 1
            while index < len(line):
                if quote == '"' and line[index] == "\\":
                    index += 2
                elif line[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
        else:
            index += 1
    return multiline


def _owned_shard(name: str) -> bool:
    # This script owns only runtimes.shardN, not other named remote runtimes.
    return re.fullmatch(r"shard[0-9]+", name) is not None


def _endpoint_alive(endpoint: dict[str, object], api_key: str, timeout_s: float) -> bool:
    """A ready record is usable only while its expected Worker model responds."""
    try:
        port = int(endpoint["port"])
        if not 0 < port < 65536:
            return False
        request = Request(
            f"http://{endpoint['host']}:{port}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urlopen(request, timeout=timeout_s) as response:
            models = json.load(response)
        return isinstance(models, dict) and any(
            isinstance(model, dict) and model.get("id") == "r2orch-base"
            for model in models.get("data", [])
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _wait_endpoints(
    registry: Path, count: int, timeout_s: float, *, api_key: str = "EMPTY",
) -> list[dict[str, object]]:
    deadline = time.monotonic() + max(1.0, timeout_s)
    rows: list[dict[str, object]] = []
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only {len(rows)}/{count} rollout shards are ready in {registry}")
        rows = []
        for index in range(count):
            try:
                row = json.loads((registry / f"shard-{index}.json").read_text())
                remaining_s = deadline - time.monotonic()
                if (isinstance(row, dict) and row.get("state") == "ready"
                        and row.get("host") and row.get("port") and remaining_s > 0
                        and _endpoint_alive(row, api_key, min(5.0, remaining_s))):
                    rows.append(row)
            except (OSError, ValueError, TypeError):
                pass
        if len(rows) == count:
            return sorted(rows, key=lambda row: int(row.get("shard_index", 0)))
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError(f"only {len(rows)}/{count} rollout shards are ready in {registry}")
        time.sleep(min(5.0, remaining_s))


def _runtime_tables(runtime: dict[str, object], endpoints: list[dict[str, object]]) -> str:
    fields = (
        "base_url", "api_key", "served_model", "model_path", "request_profile", "temperature", "enable_thinking", "max_tokens",
        "max_concurrency", "timeout_s", "managed_locally", "frozen", "self_hosted",
    )
    blocks: list[str] = []
    for index, endpoint in enumerate(endpoints):
        values = dict(runtime)
        # The source runtime is the in-process local vLLM service and may use
        # a high aggregate concurrency (128 in the R2Orch config).  Each shard
        # is an externally managed endpoint, where AdaptiveApplicationConfig
        # caps a physical gateway at 16 concurrent requests.  The endpoint
        # pool still aggregates capacity across shards; leave each member
        # within the per-endpoint contract so the rendered config validates.
        # A shard we run ourselves (runtime.self_hosted = true, e.g. the second single-card Worker vLLM
        # of the remote layout) has no provider quota: keep the configured concurrency.
        capped = not bool(values.get("self_hosted"))
        if "max_concurrency" in values and capped:
            values["max_concurrency"] = min(16, max(1, int(values["max_concurrency"])))
        if capped and isinstance(values.get("max_concurrency_by_dataset"), dict):
            values["max_concurrency_by_dataset"] = {
                key: min(16, max(1, int(limit)))
                for key, limit in values["max_concurrency_by_dataset"].items()
            }
        values["base_url"] = f"http://{endpoint['host']}:{int(endpoint['port'])}/v1"
        values["served_model"] = "r2orch-base"
        values["managed_locally"] = False
        values["frozen"] = True
        lines = [f"[runtimes.shard{index}]"]
        for field in fields:
            if field in values:
                lines.append(f"{field} = {_toml(values[field])}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n\n"


def build(source: Path, output: Path, registry: Path, count: int, timeout_s: float) -> None:
    source_text = source.read_text(encoding="utf-8")
    payload = tomllib.loads(source_text)
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("rendered config lacks [runtime]")
    routing_config = payload.get("runtime_routing")
    if not isinstance(routing_config, dict):
        raise ValueError("rendered config lacks [runtime_routing]")
    if "worker_routes" not in routing_config:
        raise ValueError("runtime_routing.worker_routes is missing")
    endpoints = _wait_endpoints(
        registry, count, timeout_s, api_key=str(runtime.get("api_key", "EMPTY")),
    )
    members = [f"shard{index}" for index in range(count)]
    pools = dict(routing_config.get("endpoint_pools", {}))
    # Preserve other pools, but do not leave a membership pointing at a removed
    # owned shard when a four-shard registry is replaced by a two-shard one.
    pools = {
        route: [name for name in names if not _owned_shard(name) or name in members]
        for route, names in pools.items()
    }
    pools["qwen"] = members
    routing_config = {**routing_config, "endpoint_pools": pools}
    routing_lines = ["[runtime_routing]"] + [
        f"{json.dumps(key)} = {_toml(value)}" for key, value in routing_config.items()
    ]
    output_lines: list[str] = []
    inserted = False
    for path, lines in _table_blocks(source_text):
        if path and len(path) >= 2 and path[0] == "runtimes" and _owned_shard(path[1]):
            # Remove all previous owned root/child tables before inserting the
            # new registry, so repeated configuration cannot duplicate TOML tables.
            continue
        if path and path[0] == "runtime_routing":
            if path == ("runtime_routing",) and not inserted:
                output_lines.extend(_runtime_tables(runtime, endpoints).splitlines())
                output_lines.extend(routing_lines)
                output_lines.append("")
                inserted = True
            # Child routing tables are already represented by routing_config.
            continue
        output_lines.extend(lines)
    if not inserted:
        raise ValueError("rendered config lacks an explicit [runtime_routing] table")
    rendered = "\n".join(output_lines) + "\n"
    # Validate before writing, including when source == output during a relay.
    tomllib.loads(rendered)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    build(args.source_config, args.output, args.registry, args.count, args.timeout_s)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
