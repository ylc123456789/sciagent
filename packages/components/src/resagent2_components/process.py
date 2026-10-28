"""Shared argv and explicit Linux shell execution with durable process logs."""

from __future__ import annotations

import os
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

from resagent2_contracts import VerificationResult
from resagent2_runtime.budget import current_budget, remaining_timeout
from resagent2_runtime.cancellation import OperationCancelled, check_cancelled

from .workspace import WorkspaceBoundary
from .text import MAX_WORKSPACE_TEXT_BYTES


_SHELL_ENVIRONMENT = frozenset({
    "BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "CDPATH",
    # Bash rebuilds these for each shell. Do not let the CLI launcher's nesting
    # or last command change either the actual startup environment or approval.
    "SHLVL", "_",
    # SSH endpoints describe the connection, not the managed command. A fresh
    # CLI connection must not change the command's actual launch environment.
    "SSH_CLIENT", "SSH_CONNECTION",
})
_BASH_ARGV = ("/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c")

_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "API_TOKEN",
    "ACCESS_KEY",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "AUTH_SOCK",
    "AGENT_PID",
    "_TOKEN",
)


def _sanitized_environment(base: dict[str, str], *, shell: bool = False) -> dict[str, str]:
    """Drop credential-like variables before launching a child process.

    This is a best-effort guard, not a sandbox: it removes API keys, SSH agent
    sockets, cloud credentials and git tokens so a verification command cannot
    inherit them implicitly.
    """
    env = dict(base)
    for name in list(env):
        upper = name.upper()
        if (any(marker in upper for marker in _SENSITIVE_ENV_MARKERS)
                or shell and (name in _SHELL_ENVIRONMENT or name.startswith("BASH_FUNC_"))):
            env.pop(name, None)
    return env


def output_tail(path: str | Path, *, limit: int = 2000) -> str:
    """Read a bounded UTF-8 log tail without loading the complete log."""
    if limit <= 0:
        return ""
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit * 4))
            return handle.read(limit * 4).decode("utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


DEFAULT_OUTPUT_PREVIEW_CHARS = 12_000


def output_preview(path: str | Path, *, limit: int = DEFAULT_OUTPUT_PREVIEW_CHARS) -> dict:
    """Describe one durable log without treating a lossy preview as source text.

    Format validation is bounded by the existing workspace text processing
    limit. Larger logs only read bounded beginning/end windows; their complete
    encoding and character count remain unknown. Access authorization is a
    separate concern belonging to the caller's actual file boundary.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("output preview limit must be a positive integer")
    log_path = Path(path)
    value = {"path": str(log_path.absolute()), "preview_limit_chars": limit,
             "text_max_bytes": MAX_WORKSPACE_TEXT_BYTES, "encoding": "utf-8"}
    try:
        with log_path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            value["total_bytes"] = size
            if size <= MAX_WORKSPACE_TEXT_BYTES:
                raw = handle.read(MAX_WORKSPACE_TEXT_BYTES + 1)
                # An unmanaged writer may append while the receipt is observed.
                if len(raw) <= MAX_WORKSPACE_TEXT_BYTES:
                    try:
                        decoded = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        decoded = raw.decode("utf-8", errors="replace")
                        text_status = "invalid_utf8"
                        value["decoding_replaced"] = True
                    else:
                        text_status = "contains_nul" if b"\x00" in raw else "supported"
                        value["decoding_replaced"] = False
                    value["text_status"] = text_status
                    value["total_chars"] = len(decoded) if text_status != "invalid_utf8" else None
                    if len(decoded) <= limit:
                        value.update(preview_kind="complete", content=decoded,
                                     preview_chars=len(decoded), truncated=False,
                                     omitted_chars=0)
                    else:
                        head_chars = (limit + 1) // 2
                        tail_chars = limit // 2
                        value.update(preview_kind="head_tail", head=decoded[:head_chars],
                                     tail=decoded[-tail_chars:] if tail_chars else "",
                                     preview_chars=limit, truncated=True,
                                     omitted_chars=(len(decoded) - limit
                                                    if text_status != "invalid_utf8" else None))
                    value["changed_during_read"] = (
                        len(raw) != size or os.fstat(handle.fileno()).st_size != size
                    )
                    return value
                # The log grew past the text limit. Re-observe bounded windows.
                size = os.fstat(handle.fileno()).st_size
                value["total_bytes"] = size
            head_chars = (limit + 1) // 2
            tail_chars = limit // 2
            handle.seek(0)
            head_raw = handle.read(head_chars * 4 + 4)
            head = head_raw.decode("utf-8", errors="replace")[:head_chars]
            if tail_chars:
                handle.seek(max(0, size - tail_chars * 4 - 4))
                tail_raw = handle.read(tail_chars * 4 + 4)
                tail = tail_raw.decode("utf-8", errors="replace")[-tail_chars:]
            else:
                tail_raw, tail = b"", ""
            encoded_head, encoded_tail = head.encode("utf-8"), tail.encode("utf-8")
            replaced = (head_raw[:len(encoded_head)] != encoded_head
                        or bool(encoded_tail) and tail_raw[-len(encoded_tail):] != encoded_tail)
            value.update(
                text_status="too_large", total_chars=None,
                preview_kind="head_tail", head=head, tail=tail,
                preview_chars=len(head) + len(tail), truncated=True,
                omitted_chars=None, decoding_replaced=bool(replaced),
                changed_during_read=os.fstat(handle.fileno()).st_size != size,
            )
    except OSError as error:
        value.update(text_status="unavailable", preview_kind="unavailable",
                     error=f"{type(error).__name__}: {str(error)[:500]}")
    return value


def _shell_environment(extra_env: dict[str, str] | None) -> dict[str, str]:
    environment = _sanitized_environment({**os.environ, **(extra_env or {})}, shell=True)
    environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return environment


def _environment_fingerprint(environment: dict[str, str]) -> str:
    """Compare startup environments without persisting their variable values."""
    encoded = json.dumps(environment, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _descendant_pids(root: int) -> list[int]:
    """Collect descendant PIDs by walking ``/proc/*/stat`` (Linux only)."""
    descendants: list[int] = []
    if not os.path.isdir("/proc"):
        return descendants
    try:
        entries = [entry for entry in os.listdir("/proc") if entry.isdigit()]
    except OSError:
        return descendants
    children: dict[int, list[int]] = {}
    for entry in entries:
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                stat = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        close = stat.rfind(")")
        if close < 0:
            continue
        try:
            pid = int(stat[:close].rsplit("(", 1)[0].strip().split()[-1])
        except (ValueError, IndexError):
            continue
        rest = stat[close + 1 :].split()
        if len(rest) < 2:
            continue
        try:
            ppid = int(rest[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    stack = list(children.get(root, []))
    while stack:
        child = stack.pop()
        descendants.append(child)
        stack.extend(children.get(child, []))
    return descendants


def _process_identity(pid: int) -> str | None:
    """Use Linux start time to prevent an observed PID being reused for cleanup."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def _remember_descendants(process: subprocess.Popen, identities: dict[int, str]) -> None:
    # After reaping, this bare PID can be reused. Only remembered identities and
    # this invocation's output pipes may identify children from that point on.
    if process.returncode is not None:
        return
    for pid in _descendant_pids(process.pid):
        identity = _process_identity(pid)
        if identity is not None:
            identities.setdefault(pid, identity)


def _remember_pipe_owners(process: subprocess.Popen, identities: dict[int, str]) -> None:
    """Find this command's inherited pipes even after its parent has exited.

    These pipe identities are unique to this Popen invocation. A detached child
    can retain them and keep communicate() waiting after being reparented, when
    walking the original parent's current descendants can no longer find it.
    """
    targets = set()
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            target = os.readlink(f"/proc/self/fd/{stream.fileno()}")
        except (OSError, ValueError):
            continue
        if target.startswith("pipe:["):
            targets.add(target)
    if not targets:
        return
    try:
        entries = os.listdir("/proc")
    except OSError:
        return
    for entry in entries:
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        pid = int(entry)
        identity = _process_identity(pid)
        if identity is None:
            continue
        try:
            descriptors = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(f"/proc/{pid}/fd/{descriptor}")
            except OSError:
                continue
            if target in targets:
                try:
                    info = Path(f"/proc/{pid}/fdinfo/{descriptor}").read_text()
                    flags = next(line.split()[1] for line in info.splitlines()
                                 if line.startswith("flags:"))
                    writable = int(flags, 8) & os.O_ACCMODE in {os.O_WRONLY, os.O_RDWR}
                except (OSError, ValueError, IndexError, StopIteration):
                    continue
                # Only inherited writers own the command output; a reader must
                # never be stopped. Recheck identity after scanning descriptors.
                if writable and _process_identity(pid) == identity:
                    identities.setdefault(pid, identity)
                if writable:
                    break


def _owned_process_running(pid: int, identity: str) -> bool:
    return _process_identity(pid) == identity and _running_pid(pid)


def _running_pid(pid: int) -> bool:
    """A zombie cannot execute; the owner/init will reap it separately."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _stop_process(process: subprocess.Popen, identities: dict[int, str] | None = None) -> bool:
    """Stop the session and remembered/pipe-owning descendants, then confirm."""
    identities = identities if identities is not None else {}
    if os.name == "posix":
        _remember_descendants(process, identities)
        _remember_pipe_owners(process, identities)
        # The original group is safe while its unreaped leader is active or an
        # identified owned member still belongs to it. Do not signal a recycled
        # bare PID/group once all original members have disappeared.
        group_owned = process.poll() is None
        if not group_owned:
            for pid, identity in identities.items():
                try:
                    if _process_identity(pid) == identity and os.getpgid(pid) == process.pid:
                        group_owned = True
                        break
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        if group_owned:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        for pid, identity in identities.items():
            if _process_identity(pid) == identity:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    else:
        process.kill()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        return False
    until = monotonic() + 3
    while any(_owned_process_running(pid, identity) for pid, identity in identities.items()):
        if monotonic() >= until:
            return False
        sleep(0.01)
    return True


def _wait_process(process: subprocess.Popen, timeout: float, identities: dict[int, str]) -> int:
    deadline = monotonic() + timeout
    while True:
        _remember_descendants(process, identities)
        # Once the operation completed, retain its real result even when an
        # interrupt arrives concurrently at the next dispatch boundary.
        if process.poll() is not None:
            return process.returncode
        check_cancelled()
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(0.05, remaining))
        except subprocess.TimeoutExpired:
            pass


def run_process(command, *, timeout: float = 3600, cwd=None, env=None,
                capture_output: bool = True, text: bool = True, check: bool = False):
    """Bound fixed framework commands by the shared deadline/cancellation scope."""
    check_cancelled()
    timeout = remaining_timeout(timeout)
    process = subprocess.Popen(
        command, cwd=cwd, env=env, text=text,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        start_new_session=os.name == "posix",
    )
    deadline = monotonic() + timeout
    identities: dict[int, str] = {}
    try:
        try:
            while True:
                _remember_descendants(process, identities)
                process.poll()  # Reap an exited parent even if inherited pipes remain open.
                check_cancelled()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            stopped = _stop_process(process, identities)
            stdout, stderr = None, None
            cleanup_error = None
            if stopped:
                try:
                    stdout, stderr = process.communicate(timeout=3)
                except subprocess.TimeoutExpired as drain_error:
                    # Inherited pipes still being held means cessation was not
                    # confirmed. Preserve the original cancellation and its
                    # truthful failure status instead of masking it with timeout.
                    stopped = False
                    stdout, stderr = drain_error.output, drain_error.stderr
                    cleanup_error = "command output pipes did not close after cleanup"
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            if isinstance(error, OperationCancelled) or timed_out and not stopped:
                def tail(value):
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    return (value or "")[-2000:]

                reason = ("Command timed out; process shutdown could not be confirmed"
                          if timed_out else error.reason)
                raise OperationCancelled(reason, stopped=stopped, details={
                    "operation": "process", "command": command, "pid": process.pid,
                    "exit_code": process.returncode,
                    **({"timed_out": True} if timed_out else {}),
                    "stdout_tail": tail(stdout), "stderr_tail": tail(stderr),
                    "files_retained": True,
                    **({"cleanup_error": cleanup_error} if cleanup_error else {}),
                    "unconfirmed_processes": [pid for pid, identity in identities.items()
                                              if _owned_process_running(pid, identity)],
                }) from (error if timed_out else None)
            if isinstance(error, subprocess.TimeoutExpired) and current_budget() is not None:
                current_budget().remaining_timeout()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
    finally:
        # Popen.__exit__ waits without a bound. If stop confirmation failed,
        # return that fact to the Controller instead of hanging in __exit__.
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


class ProcessRunner:
    """Execute argv in a workspace and terminate its process tree on timeout."""

    def __init__(self, boundary: WorkspaceBoundary) -> None:
        self.boundary = boundary

    def shell_context(self, *, extra_env: dict[str, str] | None = None) -> dict:
        """Bind approval to the real shell flags, cwd and filtered environment."""
        return self._shell_context(_shell_environment(extra_env))

    def _shell_context(self, environment: dict[str, str]) -> dict:
        return {
            "cwd": str(self.boundary.root),
            "shell": _BASH_ARGV[0],
            "shell_args": list(_BASH_ARGV[1:]),
            "pipefail": True,
            "environment_fingerprint": _environment_fingerprint(environment),
        }

    def run_shell(
        self,
        script: str,
        *,
        log_dir: str,
        index: int,
        timeout_seconds: int,
        extra_env: dict[str, str] | None = None,
        expected_context: dict | None = None,
    ) -> VerificationResult:
        """Run an exact script in fixed non-login Bash, with pipeline failures retained."""
        if sys.platform != "linux":
            raise RuntimeError("Shell execution is supported only on Linux")
        if not script.strip() or "\x00" in script:
            raise ValueError("shell script must be nonempty and contain no NUL bytes")
        environment = _shell_environment(extra_env)
        context = self._shell_context(environment)
        if expected_context is not None and expected_context != context:
            raise PermissionError("Shell startup configuration changed after approval; script was not executed")
        return self._run_argv(
            [*_BASH_ARGV, script], command=script,
            log_dir=log_dir, index=index, timeout_seconds=timeout_seconds,
            environment=environment, context=context,
        )

    def _run_argv(
        self,
        argv: list[str],
        *,
        command: str,
        log_dir: str,
        index: int,
        timeout_seconds: int,
        environment: dict[str, str],
        context: dict,
    ) -> VerificationResult:
        check_cancelled()
        timeout_seconds = remaining_timeout(timeout_seconds)
        stdout_path, stderr_path = self._log_paths(log_dir, index)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        started = monotonic()
        timed_out = False
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                argv,
                cwd=self.boundary.root,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=os.name == "posix",
            )
            identities: dict[int, str] = {}
            try:
                exit_code = _wait_process(process, remaining_timeout(timeout_seconds), identities)
            except subprocess.TimeoutExpired as error:
                timed_out = True
                stopped = _stop_process(process, identities)
                if not stopped:
                    raise OperationCancelled(
                        "Command timed out; process shutdown could not be confirmed",
                        stopped=False, details={
                            "operation": "process", "command": command, "pid": process.pid,
                            **context, "timed_out": True, "exit_code": process.returncode,
                            "unconfirmed_processes": [pid for pid, identity in identities.items()
                                                      if _owned_process_running(pid, identity)],
                            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                            "duration_seconds": monotonic() - started, "files_retained": True,
                        },
                    ) from error
                exit_code = process.returncode
            except BaseException as error:
                stopped = _stop_process(process, identities)
                if isinstance(error, OperationCancelled):
                    raise OperationCancelled(error.reason, stopped=stopped, details={
                        "operation": "process", "command": command, "pid": process.pid,
                        **context,
                        "exit_code": process.returncode,
                        "unconfirmed_processes": [pid for pid, identity in identities.items()
                                                  if _owned_process_running(pid, identity)],
                        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                        "duration_seconds": monotonic() - started,
                        "files_retained": True,
                    }) from None
                raise
        return VerificationResult(
            command=command,
            exit_code=exit_code,
            timed_out=timed_out,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            duration_seconds=monotonic() - started,
            **context,
        )

    def _log_paths(self, log_dir: str, index: int) -> tuple[Path, Path]:
        """Resolve stdout/stderr paths.

        An absolute ``log_dir`` writes audit logs outside the workspace (the Run
        data directory); a relative ``log_dir`` stays inside the workspace's
        reserved ``.resagent2`` directory, enforcing the write boundary.
        """
        root = Path(log_dir)
        if root.is_absolute():
            return (
                root / f"command_{index:02d}.stdout",
                root / f"command_{index:02d}.stderr",
            )
        return (
            self.boundary.resolve_system_write(f"{log_dir}/command_{index:02d}.stdout"),
            self.boundary.resolve_system_write(f"{log_dir}/command_{index:02d}.stderr"),
        )
