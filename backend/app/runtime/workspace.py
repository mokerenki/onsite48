"""Persistent per-task computer (shell, filesystem, packages, code).

Never executes on the host process tree for production paths.
Backends:
  - dev: process-isolated workspace under a dedicated root (docker when available;
         otherwise tightly constrained local dir with cleaned env). Marked DEV.
  - sandbox: E2B (production path via existing SandboxManager / PersistentSandboxManager).

Activated via WORKSPACE_BACKEND=dev|sandbox (default: dev).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog

from app.runtime.sandbox_policy import check_command, check_python_code, clamp_timeout

logger = structlog.get_logger("onsite48.runtime.workspace")

# Defaults
DEFAULT_TIMEOUT_S = 60
MAX_OUTPUT_CHARS = 120_000
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MiB
WORKSPACE_ROOT_ENV = "ONSITE48_WORKSPACE_ROOT"
DEFAULT_ROOT = Path(os.environ.get(WORKSPACE_ROOT_ENV, "/tmp/onsite48_workspaces"))


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
        }
        if self.exit_code != 0 or self.timed_out:
            out["error_code"] = "timeout" if self.timed_out else "nonzero_exit"
        if self.exit_code == 126 and "sandbox" in (self.stderr or "").lower():
            out["error_code"] = "policy_denied"
        return out


class Workspace(ABC):
    """Abstract persistent computer for one run_id."""

    run_id: str

    @abstractmethod
    async def create(self) -> None:
        ...

    @abstractmethod
    async def destroy(self) -> None:
        ...

    @abstractmethod
    async def exec(self, cmd: str, *, timeout: int = DEFAULT_TIMEOUT_S, cwd: str | None = None) -> ExecResult:
        ...

    @abstractmethod
    async def read(self, path: str, *, max_bytes: int = MAX_FILE_BYTES) -> dict[str, Any]:
        ...

    @abstractmethod
    async def write(self, path: str, content: str | bytes) -> dict[str, Any]:
        ...

    @abstractmethod
    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def list(self, path: str = ".", *, recursive: bool = False) -> dict[str, Any]:
        ...

    @abstractmethod
    async def grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        max_matches: int = 50,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    async def snapshot(self) -> dict[str, Any]:
        """Return metadata enough to re-attach after worker restart."""
        ...

    async def python_exec(self, code: str, *, timeout: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
        """Run Python code inside the workspace (writes a temp script then exec)."""
        # Phase 4: static python policy before writing/executing.
        py_decision = check_python_code(code)
        if not py_decision.allowed:
            return {
                "ok": False,
                "error_code": py_decision.error_code or "policy_denied",
                "message": py_decision.reason,
                "retryable": False,
                "stdout": "",
                "stderr": py_decision.reason,
                "exit_code": 126,
            }
        timeout = clamp_timeout(timeout)
        script_name = f"_py_{uuid.uuid4().hex[:12]}.py"
        await self.write(script_name, code)
        try:
            # Try both python3 and python for cross-platform compatibility
            python_cmd = "python" if os.name == "nt" else "python3"
            result = await self.exec(f"{python_cmd} {script_name}", timeout=timeout)
            return result.to_dict()
        finally:
            try:
                await self.exec(f"rm -f {script_name}", timeout=10)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dev backend: dedicated directory + cleaned-env subprocess
# ---------------------------------------------------------------------------

class DevWorkspace(Workspace):
    """Dev / local backend.

    When Docker is available and WORKSPACE_USE_DOCKER=1, would use a container.
    In this environment Docker is absent, so we use a process-isolated tree under
    a dedicated root with:
      - no credentials in env
      - path escape guards
      - ulimit / timeout
      - no host mounts of secrets
      - Phase 4 sandbox command policy

    NOT a full container; escape suite still exercises path/env/network guards.
    """

    def __init__(self, run_id: str, root: Path | None = None):
        self.run_id = run_id
        self._root = (root or DEFAULT_ROOT).resolve()
        self._home = self._root / run_id
        self._created = False
        self._meta_path = self._home / ".workspace_meta.json"

    def _resolve(self, path: str) -> Path:
        if path.startswith("/"):
            p = Path(path).resolve()
        else:
            p = (self._home / path).resolve()
        try:
            p.relative_to(self._home)
        except ValueError:
            raise PermissionError(f"path_escape: {path} is outside workspace")
        return p

    async def create(self) -> None:
        self._home.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": self.run_id,
            "backend": "dev",
            "created_at": time.time(),
            "home": str(self._home),
            "isolation": "process+path_guards+sandbox_policy",
        }
        self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (self._home / "tmp").mkdir(exist_ok=True)
        (self._home / "out").mkdir(exist_ok=True)
        self._created = True
        logger.info("workspace.dev.created", run_id=self.run_id, home=str(self._home))

    async def destroy(self) -> None:
        if self._home.exists():
            shutil.rmtree(self._home, ignore_errors=True)
        self._created = False
        logger.info("workspace.dev.destroyed", run_id=self.run_id)

    async def exec(self, cmd: str, *, timeout: int = DEFAULT_TIMEOUT_S, cwd: str | None = None) -> ExecResult:
        # Phase 4: sandbox command policy (deny dangerous patterns, clamp timeout).
        decision = check_command(cmd, timeout=timeout)
        if not decision.allowed:
            return ExecResult(
                stdout="",
                stderr=decision.reason or "denied by sandbox policy",
                exit_code=126,
                timed_out=False,
                duration_ms=0,
            )
        timeout = decision.timeout_s
        if not self._created:
            await self.create()
        workdir = self._resolve(cwd) if cwd else self._home
        clean_env = {
            "HOME": str(self._home),
            "TMPDIR": str(self._home / "tmp"),
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        # Preserve system PATH on Windows for finding python/commands
        if os.name == "nt":
            clean_env["PATH"] = os.environ.get("PATH", "")
        else:
            clean_env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        for k in list(os.environ):
            if any(s in k.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "API_")):
                continue
            if k in ("TERM", "TZ"):
                clean_env[k] = os.environ[k]

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(workdir),
                env=clean_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                code = proc.returncode if proc.returncode is not None else 1
                timed_out = False
            except asyncio.TimeoutError:
                timed_out = True
                code = 124
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                stdout_b, stderr_b = b"", b"timeout"
            duration_ms = int((time.monotonic() - start) * 1000)
            stdout = stdout_b.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
            stderr = stderr_b.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=code,
                timed_out=timed_out,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ExecResult(
                stdout="",
                stderr=str(exc),
                exit_code=1,
                timed_out=False,
                duration_ms=duration_ms,
            )

    async def read(self, path: str, *, max_bytes: int = MAX_FILE_BYTES) -> dict[str, Any]:
        try:
            p = self._resolve(path)
            if not p.exists():
                return {"error_code": "not_found", "message": f"{path} not found", "retryable": False}
            if p.is_dir():
                return {"error_code": "is_directory", "message": f"{path} is a directory", "retryable": False}
            size = p.stat().st_size
            data = p.read_bytes()[:max_bytes]
            try:
                text = data.decode("utf-8")
                return {"path": path, "content": text, "size": size, "truncated": size > max_bytes}
            except UnicodeDecodeError:
                import base64

                return {
                    "path": path,
                    "content_b64": base64.b64encode(data).decode("ascii"),
                    "size": size,
                    "truncated": size > max_bytes,
                    "binary": True,
                }
        except PermissionError as e:
            return {"error_code": "path_escape", "message": str(e), "retryable": False}
        except Exception as e:
            return {"error_code": "read_failed", "message": str(e), "retryable": False}

    async def write(self, path: str, content: str | bytes) -> dict[str, Any]:
        try:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                data = content.encode("utf-8")
            else:
                data = content
            if len(data) > MAX_FILE_BYTES:
                return {"error_code": "too_large", "message": f"max {MAX_FILE_BYTES} bytes", "retryable": False}
            p.write_bytes(data)
            return {"path": path, "bytes": len(data)}
        except PermissionError as e:
            return {"error_code": "path_escape", "message": str(e), "retryable": False}
        except Exception as e:
            return {"error_code": "write_failed", "message": str(e), "retryable": False}

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        try:
            p = self._resolve(path)
            if not p.exists():
                return {"error_code": "not_found", "message": f"{path} not found", "retryable": False}
            text = p.read_text(encoding="utf-8", errors="replace")
            if old_string not in text:
                return {"error_code": "not_found", "message": "old_string not found", "retryable": False}
            if replace_all:
                new_text = text.replace(old_string, new_string)
                count = text.count(old_string)
            else:
                new_text = text.replace(old_string, new_string, 1)
                count = 1
            p.write_text(new_text, encoding="utf-8")
            return {"path": path, "replacements": count}
        except PermissionError as e:
            return {"error_code": "path_escape", "message": str(e), "retryable": False}
        except Exception as e:
            return {"error_code": "edit_failed", "message": str(e), "retryable": False}

    async def list(self, path: str = ".", *, recursive: bool = False) -> dict[str, Any]:
        try:
            p = self._resolve(path)
            if not p.exists():
                return {"error_code": "not_found", "message": f"{path} not found", "retryable": False}
            if not p.is_dir():
                return {"error_code": "not_directory", "message": f"{path} is not a directory", "retryable": False}
            entries = []
            if recursive:
                for root, dirs, files in os.walk(p):
                    for name in files:
                        fp = Path(root) / name
                        rel = str(fp.relative_to(self._home))
                        entries.append({"path": rel, "type": "file", "size": fp.stat().st_size})
                    for name in dirs:
                        fp = Path(root) / name
                        rel = str(fp.relative_to(self._home))
                        entries.append({"path": rel, "type": "dir"})
            else:
                for child in sorted(p.iterdir()):
                    rel = str(child.relative_to(self._home))
                    if child.is_dir():
                        entries.append({"path": rel, "type": "dir"})
                    else:
                        entries.append({"path": rel, "type": "file", "size": child.stat().st_size})
            return {"path": path, "entries": entries[:2000]}
        except PermissionError as e:
            return {"error_code": "path_escape", "message": str(e), "retryable": False}
        except Exception as e:
            return {"error_code": "list_failed", "message": str(e), "retryable": False}

    async def grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        max_matches: int = 50,
    ) -> dict[str, Any]:
        try:
            p = self._resolve(path)
            if not p.exists():
                return {"error_code": "not_found", "message": f"{path} not found", "retryable": False}
            cre = re.compile(pattern)
            matches = []
            files: list[Path] = []
            if p.is_file():
                files = [p]
            else:
                for root, _, names in os.walk(p):
                    for name in names:
                        files.append(Path(root) / name)
            for fp in files:
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if cre.search(line):
                        matches.append(
                            {
                                "path": str(fp.relative_to(self._home)),
                                "line": i,
                                "text": line[:500],
                            }
                        )
                        if len(matches) >= max_matches:
                            return {"pattern": pattern, "matches": matches, "truncated": True}
            return {"pattern": pattern, "matches": matches, "truncated": False}
        except PermissionError as e:
            return {"error_code": "path_escape", "message": str(e), "retryable": False}
        except re.error as e:
            return {"error_code": "bad_pattern", "message": str(e), "retryable": False}
        except Exception as e:
            return {"error_code": "grep_failed", "message": str(e), "retryable": False}

    async def snapshot(self) -> dict[str, Any]:
        if not self._created and not self._home.exists():
            return {"run_id": self.run_id, "backend": "dev", "exists": False}
        file_count = 0
        total_bytes = 0
        if self._home.exists():
            for root, _, files in os.walk(self._home):
                for name in files:
                    file_count += 1
                    try:
                        total_bytes += (Path(root) / name).stat().st_size
                    except OSError:
                        pass
        return {
            "run_id": self.run_id,
            "backend": "dev",
            "exists": self._home.exists(),
            "home": str(self._home),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "isolation": "process+path_guards+sandbox_policy",
        }


# ---------------------------------------------------------------------------
# E2B backend (production path)
# ---------------------------------------------------------------------------

class E2BWorkspace(Workspace):
    """E2B-backed workspace. Requires E2B_API_KEY and SandboxManager."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._sandbox = None
        self._created = False

    async def create(self) -> None:
        try:
            from app.services.agent_computer import PersistentSandboxManager

            mgr = PersistentSandboxManager()
            self._sandbox = await mgr.get_or_create(self.run_id)
            self._created = True
            logger.info("workspace.e2b.created", run_id=self.run_id)
        except Exception as exc:
            logger.error("workspace.e2b.create_failed", run_id=self.run_id, error=str(exc))
            raise

    async def destroy(self) -> None:
        self._created = False
        self._sandbox = None

    async def exec(self, cmd: str, *, timeout: int = DEFAULT_TIMEOUT_S, cwd: str | None = None) -> ExecResult:
        decision = check_command(cmd, timeout=timeout)
        if not decision.allowed:
            return ExecResult(
                stdout="",
                stderr=decision.reason or "denied by sandbox policy",
                exit_code=126,
                timed_out=False,
                duration_ms=0,
            )
        timeout = decision.timeout_s
        if not self._created:
            await self.create()
        # Best-effort delegate; shape varies by manager version.
        start = time.monotonic()
        try:
            if hasattr(self._sandbox, "exec"):
                raw = await self._sandbox.exec(cmd, timeout=timeout)
            elif hasattr(self._sandbox, "run_command"):
                raw = await self._sandbox.run_command(cmd, timeout=timeout)
            else:
                return ExecResult(stderr="e2b_exec_unavailable", exit_code=1)
            duration_ms = int((time.monotonic() - start) * 1000)
            if isinstance(raw, dict):
                return ExecResult(
                    stdout=str(raw.get("stdout") or "")[:MAX_OUTPUT_CHARS],
                    stderr=str(raw.get("stderr") or "")[:MAX_OUTPUT_CHARS],
                    exit_code=int(raw.get("exit_code") or 0),
                    timed_out=bool(raw.get("timed_out")),
                    duration_ms=duration_ms,
                )
            return ExecResult(stdout=str(raw)[:MAX_OUTPUT_CHARS], duration_ms=duration_ms)
        except Exception as exc:
            return ExecResult(stderr=str(exc), exit_code=1, duration_ms=int((time.monotonic() - start) * 1000))

    async def read(self, path: str, *, max_bytes: int = MAX_FILE_BYTES) -> dict[str, Any]:
        if not self._created:
            await self.create()
        try:
            if hasattr(self._sandbox, "read"):
                return await self._sandbox.read(path)
            return {"error_code": "unsupported", "message": "e2b read unavailable", "retryable": False}
        except Exception as e:
            return {"error_code": "read_failed", "message": str(e), "retryable": False}

    async def write(self, path: str, content: str | bytes) -> dict[str, Any]:
        if not self._created:
            await self.create()
        try:
            if hasattr(self._sandbox, "write"):
                await self._sandbox.write(path, content)
                return {"path": path, "bytes": len(content) if isinstance(content, (str, bytes)) else 0}
            return {"error_code": "unsupported", "message": "e2b write unavailable", "retryable": False}
        except Exception as e:
            return {"error_code": "write_failed", "message": str(e), "retryable": False}

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        data = await self.read(path)
        if data.get("error_code"):
            return data
        text = str(data.get("content") or "")
        if old_string not in text:
            return {"error_code": "not_found", "message": "old_string not found", "retryable": False}
        if replace_all:
            new_text = text.replace(old_string, new_string)
            count = text.count(old_string)
        else:
            new_text = text.replace(old_string, new_string, 1)
            count = 1
        out = await self.write(path, new_text)
        if out.get("error_code"):
            return out
        return {"path": path, "replacements": count}

    async def list(self, path: str = ".", *, recursive: bool = False) -> dict[str, Any]:
        if not self._created:
            await self.create()
        try:
            if hasattr(self._sandbox, "list"):
                return await self._sandbox.list(path)
            return {"error_code": "unsupported", "message": "e2b list unavailable", "retryable": False}
        except Exception as e:
            return {"error_code": "list_failed", "message": str(e), "retryable": False}

    async def grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        max_matches: int = 50,
    ) -> dict[str, Any]:
        return {"error_code": "unsupported", "message": "e2b grep via shell", "retryable": True}

    async def snapshot(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "backend": "e2b", "created": self._created}


# ---------------------------------------------------------------------------
# Factory / registry
# ---------------------------------------------------------------------------

_workspaces: dict[str, Workspace] = {}


async def create_workspace(run_id: str, *, backend: str | None = None) -> Workspace:
    backend = (backend or os.environ.get("WORKSPACE_BACKEND") or "dev").lower()
    if run_id in _workspaces:
        return _workspaces[run_id]
    if backend == "sandbox" or backend == "e2b":
        ws: Workspace = E2BWorkspace(run_id)
    else:
        ws = DevWorkspace(run_id)
    await ws.create()
    _workspaces[run_id] = ws
    return ws


async def destroy_workspace(run_id: str) -> None:
    ws = _workspaces.pop(run_id, None)
    if ws is not None:
        await ws.destroy()


def get_workspace(run_id: str) -> Workspace | None:
    return _workspaces.get(run_id)


def list_active_workspaces() -> list[str]:
    return list(_workspaces.keys())
