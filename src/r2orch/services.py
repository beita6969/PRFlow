from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Loading the base model and registering a LoRA adapter can legitimately take
# slightly more than two minutes on a busy shared GPU.
DEFAULT_SERVICE_START_WAIT_S = 300.0


@dataclass(frozen=True)
class ModelServiceSpec:
    role: str
    port: int
    served_model: str
    base_model_path: Path
    # A single vLLM base can expose several aliases and several LoRA adapters.
    served_model_aliases: tuple[str, ...] = ()
    lora_modules: tuple[tuple[str, Path], ...] = ()
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    host: str = "127.0.0.1"
    api_key: str = "EMPTY"
    extra_args: tuple[str, ...] = ()
    extra_env: tuple[tuple[str, str], ...] = ()
    gpu_ids: tuple[int, ...] = ()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command(self) -> list[str]:
        model_path = self.base_model_path
        base_aliases = self.served_model_aliases or (self.served_model,)
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_path),
            "--served-model-name",
            *base_aliases,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--api-key",
            self.api_key,
        ]
        modules = list(self.lora_modules)
        if modules:
            names = [name for name, _path in modules]
            if len(names) != len(set(names)):
                raise ValueError("vLLM LoRA module names must be unique")
            max_lora_rank = max(_checkpoint_lora_rank(path) for _name, path in modules)
            command.extend(
                [
                    "--enable-lora",
                    "--max-lora-rank",
                    str(max_lora_rank),
                    "--max-loras",
                    str(len(modules)),
                    "--max-cpu-loras",
                    str(len(modules)),
                    "--lora-modules",
                    *(f"{name}={path}" for name, path in modules),
                ]
            )
        return [*command, *self.extra_args]


@dataclass(frozen=True)
class ServiceStatus:
    role: str
    running: bool
    healthy: bool
    pid: int | None
    base_url: str
    gpu_ids: tuple[int, ...]


def _checkpoint_lora_rank(checkpoint_path: Path) -> int:
    config_path = checkpoint_path / "adapter_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        rank = int(payload.get("r", 64))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        rank = 64
    return max(16, rank)


class VLLMServiceManager:
    """Own only vLLM processes recorded in its explicit state directory."""

    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        spec: ModelServiceSpec,
        *,
        wait_s: float = DEFAULT_SERVICE_START_WAIT_S,
    ) -> ServiceStatus:
        current = self.status(spec)
        if current.running:
            state = self._load_state(spec.role)
            if current.healthy and _state_matches_spec(state, spec):
                return current
            self.stop(spec.role)
        log_path = self.state_dir / f"{spec.role}.log"
        log_handle = log_path.open("ab")
        environment = os.environ.copy()
        environment.update(spec.extra_env)
        if spec.gpu_ids:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, spec.gpu_ids))
        process = subprocess.Popen(
            spec.command(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        self._write_state(spec, process.pid)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            status = self.status(spec)
            if status.healthy:
                return status
            if not status.running:
                raise RuntimeError(f"{spec.role} service exited; inspect {log_path}")
            time.sleep(1.0)
        raise TimeoutError(f"{spec.role} did not become healthy within {wait_s}s")

    def stop(self, role: str, *, wait_s: float = 30.0) -> None:
        state = self._load_state(role)
        if state is None:
            return
        pid = int(state["pid"])
        if _pid_running(pid) and not _pid_matches(pid, state):
            raise RuntimeError(f"refusing to stop reused or unowned pid {pid}")
        if _pid_running(pid):
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline and _pid_running(pid):
                time.sleep(0.25)
            if _pid_running(pid):
                os.killpg(pid, signal.SIGKILL)
        self._state_path(role).unlink(missing_ok=True)

    def status(self, spec: ModelServiceSpec) -> ServiceStatus:
        state = self._load_state(spec.role)
        pid = int(state["pid"]) if state else None
        running = bool(pid and _pid_running(pid))
        healthy = running and _health(
            spec.base_url,
            spec.api_key,
            expected_model=(
                spec.lora_modules[0][0]
                if spec.lora_modules
                else (
                    spec.served_model_aliases[0] if spec.served_model_aliases else spec.served_model
                )
            ),
        )
        return ServiceStatus(spec.role, running, healthy, pid, spec.base_url, spec.gpu_ids)

    def _write_state(self, spec: ModelServiceSpec, pid: int) -> None:
        payload = {
            "pid": pid,
            "role": spec.role,
            "base_url": spec.base_url,
            "served_model_aliases": list(spec.served_model_aliases),
            "lora_modules": [[name, str(path)] for name, path in spec.lora_modules],
            "command": spec.command(),
            "gpu_ids": spec.gpu_ids,
            "spec_fingerprint": _spec_fingerprint(spec),
        }
        path = self._state_path(spec.role)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _load_state(self, role: str) -> dict[str, Any] | None:
        path = self._state_path(role)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _state_path(self, role: str) -> Path:
        return self.state_dir / f"{role}.json"


def _pid_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except OSError:
        return False
    if len(fields) >= 3 and fields[2] == "Z":
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _spec_fingerprint(spec: ModelServiceSpec) -> dict[str, Any]:
    return {
        "command": spec.command(),
        "extra_env": [list(item) for item in spec.extra_env],
        "gpu_ids": list(spec.gpu_ids),
    }


def _state_matches_spec(state: dict[str, Any] | None, spec: ModelServiceSpec) -> bool:
    return bool(state and state.get("spec_fingerprint") == _spec_fingerprint(spec))


def _pid_matches(pid: int, state: dict[str, Any]) -> bool:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return False
    command_line = path.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    expected = " ".join(str(value) for value in state.get("command", []))
    return "vllm.entrypoints.openai.api_server" in command_line and expected in command_line


def _health(base_url: str, api_key: str, *, expected_model: str | None = None) -> bool:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            if not 200 <= response.status < 300:
                return False
            if not expected_model:
                return True
            payload = json.loads(response.read().decode("utf-8"))
            model_ids = {str(item.get("id", "")) for item in payload.get("data", [])}
            return expected_model in model_ids
    except (json.JSONDecodeError, urllib.error.URLError, TimeoutError):
        return False
