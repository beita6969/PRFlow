"""ALFWorld episodes in a separate Python environment (JSON lines over a pipe).

R2Orch runs ALFWorld with the interpreter in ``r2orch.paths.alfworld_python``,
so the training venv does not need TextWorld.
``ALFWorldBridgeClient`` implements the ``alfworld.ALFWorldClient`` protocol; this
file is also the server script and imports only the standard library at top level.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any


class ALFWorldBridgeClient:
    """``alfworld.ALFWorldClient`` over a pipe; one server process per client."""

    def __init__(self, python: str) -> None:
        self.python = python
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._process = subprocess.Popen(
                    [self.python, "-u", str(Path(__file__).resolve())],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                )
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("ALFWorld bridge process exited")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"ALFWorld bridge: {response.get('error')}")
        return response["result"]

    def create_session(self, game_path: str, *, max_steps: int) -> dict[str, Any]:
        return self._request({"op": "create", "game_path": str(game_path), "max_steps": int(max_steps)})

    def step(self, session_id: str, command: str) -> dict[str, Any]:
        return self._request({"op": "step", "session_id": session_id, "command": command})

    def close_session(self, session_id: str) -> None:
        self._request({"op": "close", "session_id": session_id})

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


# --------------------------------------------------------------------------- server


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _scalar(value: Any) -> float:
    value = _first(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _commands(infos: dict[str, Any]) -> list[str]:
    value = infos.get("admissible_commands", [])
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(command) for command in value]


class _Server:
    """One TextWorld env per session, run in the ALFWorld interpreter."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    def create(self, game_path: str, max_steps: int) -> dict[str, Any]:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        path = Path(game_path)
        if not path.is_file():
            raise FileNotFoundError(f"ALFWorld game file does not exist: {path}")
        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, score=True, max_score=True,
            description=True, inventory=True,
        )
        env_id = textworld.gym.register_game(
            str(path), request_infos=request_infos, max_episode_steps=int(max_steps),
            wrappers=[AlfredDemangler()],
        )
        env = textworld.gym.make(env_id)
        try:
            observation, infos = env.reset()
        except Exception:
            env.close()
            raise
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = {"env": env, "done": False}
        return {
            "session_id": session_id,
            "observation": _first(observation),
            "admissible_commands": _commands(infos),
            "reward": 0.0,
            "score": _scalar(infos.get("score", 0.0)),
            "done": False,
            "won": False,
        }

    def step(self, session_id: str, command: str) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        if session is None:
            raise RuntimeError("ALFWorld session is not active")
        if session["done"]:
            raise RuntimeError("ALFWorld episode is already done")
        observation, reward, done, infos = session["env"].step(command)
        done = bool(_first(done))
        session["done"] = done
        return {
            "observation": _first(observation),
            "admissible_commands": _commands(infos),
            "reward": float(_scalar(reward)),
            "score": float(_scalar(infos.get("score", reward))),
            "done": done,
            "won": bool(_first(infos.get("won", False))),
        }

    def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            session["env"].close()


def serve() -> None:
    # TextWorld may print; keep the protocol channel for responses only.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    server = _Server()
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            op = request["op"]
            if op == "create":
                result = server.create(request["game_path"], request["max_steps"])
            elif op == "step":
                result = server.step(request["session_id"], request["command"])
            elif op == "close":
                result = server.close(request["session_id"])
            else:
                raise ValueError(f"unknown op {op!r}")
            response = {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - reported to the client
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        protocol.write(json.dumps(response) + "\n")
        protocol.flush()
    for session_id in list(server.sessions):
        server.close(session_id)


if __name__ == "__main__":
    # Run as a script, this package directory is sys.path[0]; its alfworld.py
    # (and other module names) would shadow the real ALFWorld/TextWorld imports.
    here = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != here]
    serve()
