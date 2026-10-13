"""R2Orch resident serving.

One vLLM process serves the frozen Qwen3.5-9B base (Worker, review model) plus the
Proposer and Director LoRA adapters. It starts once per allocation; after every
training commit the new adapters are hot-loaded under fresh versioned names and
the prefix cache is reset, so collection never waits for a restart.
"""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from .qwen_compat import qwen_vllm_server_args, qwen_vllm_server_env
from .services import ModelServiceSpec, VLLMServiceManager, _health

ROLE = "r2orch_rollout"
WORKER_ROLE = "r2orch_worker"
VLLM_START_WAIT_S = 900.0
_STEP_DIR = re.compile(r"step-(\d+)")

# serving.*_cudagraph_mode -> vLLM 0.27.1 flags.  "default" (the default) leaves vLLM's own
# compilation config alone: measured on A100 (LoRA on, 32 concurrent requests) it decodes
# 2.7x (base) to 3.3x (LoRA) faster than "eager" and hot-loads LoRA adapters normally.  "eager" is
# the --enforce-eager service; the two non-compiling modes avoid torch.compile entirely.
_CUDAGRAPH_ARGS: dict[str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = {
    "eager": (("--enforce-eager",), ()),
    "decode": (
        ("--compilation-config", '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'),
        (),
    ),
    "breakable": (
        ("--compilation-config", '{"mode":"NONE","cudagraph_mode":"FULL_AND_PIECEWISE"}'),
        (("VLLM_USE_BREAKABLE_CUDAGRAPH", "1"),),
    ),
    "default": ((), ()),
}


def decode_latency_args(
    *,
    cudagraph_mode: str = "default",
    prefix_caching: bool = True,
    max_num_seqs: int = 0,
    max_num_batched_tokens: int = 0,
    extra_args: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], frozenset[str]]:
    """``(extra args, extra env, flags to drop from the base args)`` for the decode-latency knobs.

    The default mode is vLLM's own CUDA-graph configuration (nothing added, nothing dropped)."""

    args, env = _CUDAGRAPH_ARGS[cudagraph_mode]
    drop: set[str] = set()
    if cudagraph_mode != "eager":
        drop.add("--enforce-eager")
    extra: list[str] = list(args)
    if not prefix_caching:
        drop.add("--enable-prefix-caching")
        extra.append("--no-enable-prefix-caching")
    if int(max_num_seqs) > 0:
        extra += ["--max-num-seqs", str(int(max_num_seqs))]
    if int(max_num_batched_tokens) > 0:
        extra += ["--max-num-batched-tokens", str(int(max_num_batched_tokens))]
    return (*extra, *extra_args), env, frozenset(drop)


_LOCAL_HOSTS = frozenset({"", "127.0.0.1", "localhost", "::1"})


def service_host(configured: str) -> str:
    """Host that clients (and the bind address) use for a serving endpoint."""

    return str(configured or "").strip() or "127.0.0.1"


def is_remote_host(host: str) -> bool:
    """True when ``host`` names another node: that service is started elsewhere, never by this driver."""

    name = str(host or "").strip()
    if name in _LOCAL_HOSTS:
        return False
    local = {socket.gethostname(), socket.gethostname().split(".")[0]}
    return name not in local and name.split(".")[0] not in local


def wait_remote_healthy(spec: ModelServiceSpec, *, wait_s: float, poll_s: float = 10.0) -> None:
    """Block until a remote vLLM answers with ``spec.served_model``; never starts anything."""

    deadline = time.monotonic() + float(wait_s)
    while not _health(spec.base_url, spec.api_key, expected_model=spec.served_model):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"remote vLLM {spec.role} at {spec.base_url} did not become healthy within {wait_s:.0f}s; "
                "start it on its node first"
            )
        time.sleep(poll_s)


def versioned_name(lora_name: str, version: int) -> str:
    return f"{lora_name}-v{int(version):04d}"


def adapter_version(adapter: Path) -> int:
    """Optimizer step encoded in the trainer's ``step-XXXXXXXX`` checkpoint directory."""

    match = _STEP_DIR.fullmatch(Path(adapter).name)
    if match is None:
        raise ValueError(f"adapter directory is not a step checkpoint: {adapter}")
    return int(match.group(1))


def latest_adapter(checkpoint_root: Path) -> tuple[Path | None, int]:
    """``(path, step)`` from the trainer's ``latest.json``; ``(None, 0)`` before training."""

    latest = Path(checkpoint_root) / "latest.json"
    if not latest.exists():
        return None, 0
    payload = json.loads(latest.read_text(encoding="utf-8"))
    path = Path(payload["path"])
    return (path, int(payload["step"])) if path.exists() else (None, 0)


def service_base_url(config) -> str:
    serving = config.r2orch.serving
    return f"http://{service_host(serving.vllm_host)}:{int(serving.vllm_port)}/v1"


def _root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def _http(method: str, url: str, payload: dict | None = None, *, api_key: str = "EMPTY") -> dict:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        method=method,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=300.0) as response:
        body = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        return {"text": body}


def _http_post(url: str, payload: dict) -> dict:
    return _http("POST", url, payload)


def _http_get(url: str) -> dict:
    return _http("GET", url)


def _model_ids(get: Callable[[str], dict], base_url: str) -> set[str]:
    return {str(item.get("id")) for item in get(f"{_root(base_url)}/v1/models").get("data", [])}


def rollout_service_spec(
    config, *, proposer_adapter: Path | None, director_adapter: Path | None
) -> tuple[ModelServiceSpec, Any]:
    """Spec of the resident vLLM and the collection config that points at it."""

    serving, lora = config.r2orch.serving, config.r2orch.lora
    names: dict[str, str] = {}
    aliases = [serving.base_served_name]
    modules: list[tuple[str, Path]] = []
    for role, lora_name, adapter in (
        ("proposer", serving.proposer_lora_name, proposer_adapter),
        ("director", serving.director_lora_name, director_adapter),
    ):
        if adapter is None:
            aliases.append(lora_name)
            names[role] = lora_name
        else:
            names[role] = versioned_name(lora_name, adapter_version(adapter))
            modules.append((names[role], Path(adapter)))
    base_model = Path(config.solver_model.base_model_path)
    tuned_args, tuned_env, dropped = decode_latency_args(
        cudagraph_mode=str(serving.vllm_cudagraph_mode),
        prefix_caching=bool(serving.vllm_prefix_caching),
        max_num_seqs=int(serving.vllm_max_num_seqs),
        max_num_batched_tokens=int(serving.vllm_max_num_batched_tokens),
        extra_args=tuple(serving.vllm_extra_args),
    )
    spec = ModelServiceSpec(
        role=ROLE,
        port=int(serving.vllm_port),
        served_model=serving.base_served_name,
        base_model_path=base_model,
        served_model_aliases=tuple(aliases),
        lora_modules=tuple(modules),
        gpu_memory_utilization=float(serving.vllm_gpu_memory_utilization),
        host=service_host(serving.vllm_host),
        # Later flags win in vLLM's argparse, so these override the Qwen defaults and
        # the per-module LoRA sizing that ModelServiceSpec derives on its own.
        extra_args=(
            *(
                arg
                for arg in qwen_vllm_server_args(base_model.name)
                if arg not in dropped
            ),
            *tuned_args,
            "--data-parallel-size", str(int(serving.vllm_data_parallel_size)),
            "--max-model-len", str(int(serving.vllm_max_model_len)),
            "--generation-config", str(serving.generation_config),
            "--enable-lora",
            "--max-loras", str(int(serving.max_loras)),
            "--max-cpu-loras", str(int(serving.max_loras)),
            "--max-lora-rank", str(int(lora.rank)),
        ),
        extra_env=(
            *qwen_vllm_server_env(base_model.name),
            *tuned_env,
            ("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True"),
            ("VLLM_SERVER_DEV_MODE", "1"),
        ),
        gpu_ids=tuple(int(gpu) for gpu in serving.vllm_gpu_ids),
    )
    base_url = spec.base_url
    # With a dedicated Worker vLLM (3gpu layout) the frozen-base route, the PATS reviewer and the
    # controlled reruns go there; the resident service keeps only the Director/Proposer LoRAs.
    worker = worker_service_spec(config)
    rollout_config = replace(
        config,
        proposer_model=replace(
            config.proposer_model, base_url=base_url, served_model=names["proposer"]
        ),
        solver_model=replace(config.solver_model, base_url=base_url, served_model=names["director"]),
        runtime=replace(
            config.runtime,
            base_url=worker.base_url if worker is not None else base_url,
            served_model=serving.base_served_name,
        ),
    )
    return spec, rollout_config


def worker_service_spec(config) -> ModelServiceSpec | None:
    """Spec of the optional frozen-base Worker vLLM (``serving.worker_port`` > 0), else None.

    Same flags as an external rollout shard (serve_rollout_shard.sh): no LoRA, no chat-template
    override, prefix caching + chunked prefill + Triton GDN prefill; the decode-latency knobs come
    from ``serving.worker_*``."""

    serving = config.r2orch.serving
    port = int(serving.worker_port or 0)
    if port <= 0:
        return None
    tuned_args, tuned_env, dropped = decode_latency_args(
        cudagraph_mode=str(serving.worker_cudagraph_mode),
        prefix_caching=bool(serving.worker_prefix_caching),
        max_num_seqs=int(serving.worker_max_num_seqs),
        max_num_batched_tokens=int(serving.worker_max_num_batched_tokens),
        extra_args=tuple(serving.worker_extra_args),
    )
    base_model = Path(config.solver_model.base_model_path)
    base_args = (
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--gdn-prefill-backend", "triton",
    )
    return ModelServiceSpec(
        role=WORKER_ROLE,
        port=port,
        served_model=serving.base_served_name,
        base_model_path=base_model,
        served_model_aliases=(serving.base_served_name,),
        gpu_memory_utilization=float(serving.worker_gpu_memory_utilization),
        host=service_host(serving.worker_host),
        extra_args=(
            *(arg for arg in base_args if arg not in dropped),
            *tuned_args,
            "--max-model-len", str(int(serving.vllm_max_model_len)),
            "--generation-config", str(serving.generation_config),
        ),
        extra_env=(*qwen_vllm_server_env(base_model.name), *tuned_env),
        gpu_ids=tuple(int(gpu) for gpu in serving.worker_gpu_ids),
    )


def _state_dir(config, state_dir: Path | None) -> Path:
    return Path(state_dir) if state_dir else Path(config.r2orch.paths.run_root) / "services"


def ensure_rollout_service(
    config, *, manage: bool, mock: bool = False, state_dir: Path | None = None
) -> Any:
    """Start the resident vLLM once per allocation (reuse a healthy port) and return the
    collection config pointing at it. ``mock`` or ``manage=False`` never starts a process."""

    proposer, _ = latest_adapter(config.proposer_model.checkpoint_path)
    director, _ = latest_adapter(config.solver_model.checkpoint_path)
    spec, rollout_config = rollout_service_spec(
        config, proposer_adapter=proposer, director_adapter=director
    )
    if mock or not manage:
        return rollout_config
    manager = VLLMServiceManager(_state_dir(config, state_dir))
    remote_wait_s = float(config.r2orch.serving.remote_wait_s)
    # The optional Worker vLLM loads in parallel with the resident one (each takes ~100-250 s).
    worker = worker_service_spec(config)
    if worker is not None and is_remote_host(worker.host):
        wait_remote_healthy(worker, wait_s=remote_wait_s)
        worker = None  # started elsewhere; nothing left for this driver to launch
    if is_remote_host(spec.host):
        wait_remote_healthy(spec, wait_s=remote_wait_s)
        loaded = _model_ids(_http_get, spec.base_url)
        for name, path in spec.lora_modules:
            if name not in loaded:
                load_lora_adapter_reliably(_http_post, _http_get, spec.base_url, name, path)
        return rollout_config
    worker_start: Future | None = None
    pool: ThreadPoolExecutor | None = None
    if worker is not None and not _health(
        worker.base_url, worker.api_key, expected_model=worker.served_model
    ):
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r2orch-worker-vllm")
        worker_start = pool.submit(manager.start, worker, wait_s=VLLM_START_WAIT_S)
    try:
        if _health(spec.base_url, spec.api_key, expected_model=spec.served_model):
            loaded = _model_ids(_http_get, spec.base_url)
            for name, path in spec.lora_modules:
                if name not in loaded:
                    load_lora_adapter_reliably(_http_post, _http_get, spec.base_url, name, path)
        else:
            manager.start(spec, wait_s=VLLM_START_WAIT_S)
        if worker_start is not None:
            worker_start.result()
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return rollout_config


def publish_adapters(
    base_url: str,
    *,
    proposer_adapter: Path | None,
    director_adapter: Path | None,
    version: int | Mapping[str, int],
    previous: Mapping[str, str] | None = None,
    base_served_name: str = "r2orch-base",
    proposer_lora_name: str = "r2orch-proposer",
    director_lora_name: str = "r2orch-director",
    mock: bool = False,
) -> dict[str, str]:
    """Hot-load the newly trained adapters under fresh names and return them.

    ``version`` is the role's optimizer step count (one int, or ``{"proposer": …,
    "director"|"solver": …}``). A role without an adapter is served by the base name.
    An unchanged name (role not updated) is not reloaded. Names older than
    ``previous`` are unloaded; ``previous`` stays loaded for in-flight async requests.
    """

    previous = dict(previous or {})
    versions = (
        {"proposer": int(version), "director": int(version)}
        if isinstance(version, int)
        else {
            "proposer": int(version["proposer"]),
            "director": int(version.get("director", version.get("solver", 0))),
        }
    )
    lora_names = {"proposer": proposer_lora_name, "director": director_lora_name}
    adapters = {"proposer": proposer_adapter, "director": director_adapter}
    names = {
        role: base_served_name
        if adapters[role] is None
        else versioned_name(lora_names[role], versions[role])
        for role in ("proposer", "director")
    }
    if mock:
        return names
    to_load = {
        names[role]: Path(adapters[role])
        for role in ("proposer", "director")
        if adapters[role] is not None and names[role] != previous.get(role)
    }
    root = _root(base_url)
    for name, path in to_load.items():
        load_lora_adapter_reliably(_http_post, _http_get, base_url, name, path)
    if to_load:
        _http_post(f"{root}/reset_prefix_cache", {})
    keep = set(names.values()) | set(previous.values())
    prefixes = tuple(f"{name}-v" for name in lora_names.values())
    for loaded in sorted(_model_ids(_http_get, base_url)):
        if loaded.startswith(prefixes) and loaded not in keep:
            _http_post(f"{root}/v1/unload_lora_adapter", {"lora_name": loaded})
    return names


def load_lora_adapter_reliably(
    post: Callable[..., Any],
    get: Callable[..., Any],
    base_url: str,
    name: str,
    path: Path | str,
    *,
    attempts: int = 4,
    pause_s: float = 15.0,
) -> None:
    """Hot-load one adapter, surviving a slow or stuck HTTP answer.

    The 300 s POST can time out while the rollout vLLM is busy, after the (expensive) training
    update.  The load may have completed anyway, so after a failed POST the served model list
    decides: loaded -> done; otherwise retry (an 'already loaded' answer is also resolved by the
    list).
    """

    root = _root(base_url)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            post(f"{root}/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
            return
        except OSError as exc:  # TimeoutError, URLError, HTTPError, ConnectionError
            last = exc
            try:
                if name in _model_ids(get, base_url):
                    return
            except OSError:
                pass
            if attempt + 1 < attempts:
                time.sleep(pause_s)
    assert last is not None
    raise last


def publish_latest_adapters(
    config,
    *,
    previous: Mapping[str, str] | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Publish the trainers' ``latest.json`` adapters; returns the
    ``served_adapters.json`` payload ``{"proposer", "director", "version"}``."""

    serving = config.r2orch.serving
    proposer, proposer_step = latest_adapter(config.proposer_model.checkpoint_path)
    director, director_step = latest_adapter(config.solver_model.checkpoint_path)

    names = publish_adapters(
        service_base_url(config),
        proposer_adapter=proposer,
        director_adapter=director,
        version={"proposer": proposer_step, "director": director_step},
        previous=previous,
        base_served_name=serving.base_served_name,
        proposer_lora_name=serving.proposer_lora_name,
        director_lora_name=serving.director_lora_name,
        mock=mock,
    )
    return {**names, "version": {"proposer": proposer_step, "solver": director_step}}
