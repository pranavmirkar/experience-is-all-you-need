"""Run untrusted, model-generated Python in an isolated process.

Isolation stack (all applied together, each checked by :func:`detect_isolation`):

* new network namespace: no interfaces, so sockets cannot connect anywhere
* new PID namespace: the candidate sees only itself
* new mount namespace with private tmpfs over /tmp and /home: nothing is shared
  with the host, and the repository and home directory are hidden
* new IPC and UTS namespaces
* privileges dropped to uid/gid 65534 with ``no_new_privs`` and no supplementary groups
* empty environment, so no credentials are inherited
* ``prlimit`` caps on CPU seconds, address space, file size, open files, processes, core size
* per-call alarm inside the process and a wall-clock timeout outside it; on timeout
  the whole process group is killed

This needs root and util-linux. It is not a defence against kernel exploits (no
seccomp filter, no separate kernel). If the stack is unavailable the sandbox refuses
to run code rather than falling back to the host.

Result integrity: only the *inputs* are sent into the sandbox. The candidate never
sees the expected outputs, and the pass/fail comparison happens outside, so a
candidate cannot forge a pass; it can only return answers it computed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

NOBODY = 65534

# Runs inside the sandbox. Reads {"source", "entry", "inputs", "per_call_s"} from stdin,
# defines the candidate, calls it on each input, and writes one JSON line per call.
_RUNNER = r"""
import sys, os, json, signal
payload = json.loads(sys.stdin.read())
out = os.fdopen(os.dup(1), "w")
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")
class _Timeout(BaseException):
    pass
def _alarm(signum, frame):
    raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)
def emit(obj):
    out.write(json.dumps(obj) + "\n")
    out.flush()
ns = {"__name__": "candidate"}
signal.setitimer(signal.ITIMER_REAL, payload["per_call_s"])
try:
    exec(compile(payload["source"], "<candidate>", "exec"), ns)
    fn = ns[payload["entry"]]
except _Timeout:
    emit({"i": -1, "ok": False, "err": "timeout while defining candidate"})
    fn = None
except BaseException as e:
    emit({"i": -1, "ok": False, "err": ("definition: %s: %s" % (type(e).__name__, e))[:200]})
    fn = None
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
for i, args in enumerate(payload["inputs"]):
    if fn is None:
        emit({"i": i, "ok": False, "err": "no callable"})
        continue
    signal.setitimer(signal.ITIMER_REAL, payload["per_call_s"])
    try:
        v = fn(*args)
        s = json.dumps(v, allow_nan=False)
        emit({"i": i, "ok": True, "v": json.loads(s)})
    except _Timeout:
        emit({"i": i, "ok": False, "err": "timeout", "timeout": True})
    except BaseException as e:
        emit({"i": i, "ok": False, "err": ("%s: %s" % (type(e).__name__, e))[:200]})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
"""

_WRAPPER = (
    "mount -t tmpfs -o size=64m,mode=1777 tmpfs /tmp && "
    "mount -t tmpfs -o size=1m tmpfs /home && "
    "cd /tmp && exec setpriv --reuid=%d --regid=%d --clear-groups --no-new-privs "
    'prlimit --cpu={cpu} --as={as_} --fsize={fsize} --nofile=64 --nproc=64 --core=0 '
    '"$1" -I -c "$2"' % (NOBODY, NOBODY)
)


class SandboxUnavailable(RuntimeError):
    """The required isolation could not be established; no code was run."""


@dataclass(frozen=True)
class Limits:
    per_call_s: float = 2.0  # alarm inside the sandbox, per call
    total_s: float = 20.0  # wall clock for the whole run, enforced outside
    cpu_s: int = 15  # RLIMIT_CPU
    memory_bytes: int = 512 * 1024 * 1024  # RLIMIT_AS
    file_bytes: int = 1024 * 1024  # RLIMIT_FSIZE
    max_output_bytes: int = 4 * 1024 * 1024


@dataclass(frozen=True)
class CallResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None
    timed_out: bool = False


@dataclass(frozen=True)
class RunResult:
    calls: tuple[CallResult, ...]
    wall_s: float
    killed: bool  # the outer wall-clock timeout fired
    definition_error: Optional[str] = None


@dataclass(frozen=True)
class IsolationReport:
    strong: bool
    facts: dict = field(default_factory=dict)
    problems: tuple[str, ...] = ()


def _command(limits: Limits) -> list[str]:
    py = shutil.which("python3") or sys.executable
    wrapper = _WRAPPER.format(cpu=limits.cpu_s, as_=limits.memory_bytes, fsize=limits.file_bytes)
    return [
        "env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "LC_ALL=C.UTF-8",
        "unshare", "--net", "--pid", "--fork", "--kill-child", "--mount", "--mount-proc",
        "--ipc", "--uts", "--",
        "sh", "-c", wrapper, "sandbox", py, _RUNNER,
    ]


def _run_raw(stdin_text: str, limits: Limits) -> tuple[str, bool, float]:
    start = time.monotonic()
    proc = subprocess.Popen(
        _command(limits),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    killed = False
    try:
        out, _ = proc.communicate(stdin_text, timeout=limits.total_s)
    except subprocess.TimeoutExpired:
        killed = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
    return out[: limits.max_output_bytes], killed, time.monotonic() - start


def run_candidate(source: str, entry_point: str, inputs: Sequence[Sequence[Any]], limits: Limits = Limits()) -> RunResult:
    """Call ``entry_point`` from ``source`` on each input inside the sandbox."""
    require_isolation()
    payload = json.dumps(
        {"source": source, "entry": entry_point, "inputs": [list(a) for a in inputs], "per_call_s": limits.per_call_s}
    )
    out, killed, wall = _run_raw(payload, limits)
    by_index: dict[int, CallResult] = {}
    definition_error = None
    for line in out.splitlines():
        try:
            rec = json.loads(line)
            i = rec["i"]
        except (ValueError, KeyError, TypeError):
            continue  # noise a candidate wrote to the result channel
        if i == -1:
            definition_error = str(rec.get("err"))
        elif isinstance(i, int) and 0 <= i < len(inputs) and i not in by_index:
            if rec.get("ok"):
                by_index[i] = CallResult(True, value=rec.get("v"))
            else:
                by_index[i] = CallResult(False, error=str(rec.get("err")), timed_out=bool(rec.get("timeout")))
    calls = tuple(
        by_index.get(i, CallResult(False, error="no result (crash or killed)", timed_out=killed))
        for i in range(len(inputs))
    )
    return RunResult(calls, wall, killed, definition_error)


# ---------------------------------------------------------------- isolation check
_PROBE = r"""
import os, socket, json
facts = {"uid": os.getuid(), "gid": os.getgid(), "env_keys": sorted(os.environ),
         "visible_pids": len([p for p in os.listdir("/proc") if p.isdigit()])}
try:
    socket.create_connection(("1.1.1.1", 80), timeout=2); facts["net_blocked"] = False
except OSError:
    facts["net_blocked"] = True
try:
    open("/tmp/__probe", "w").write("x"); facts["tmp_writable"] = True
except OSError:
    facts["tmp_writable"] = False
try:
    facts["home_entries"] = len(os.listdir("/home"))
except OSError:
    facts["home_entries"] = -1
for p in ("/root", "/etc/shadow"):
    try:
        os.listdir(p) if os.path.isdir(p) else open(p).read(1); facts["reads_" + p] = True
    except OSError:
        facts["reads_" + p] = False
print(json.dumps(facts))
"""

_report: Optional[IsolationReport] = None


def detect_isolation(refresh: bool = False) -> IsolationReport:
    """Run a probe through the real sandbox command and judge what it observed."""
    global _report
    if _report is not None and not refresh:
        return _report
    if os.geteuid() != 0 or not all(shutil.which(t) for t in ("unshare", "setpriv", "prlimit", "env", "sh")):
        _report = IsolationReport(False, {}, ("needs root and unshare/setpriv/prlimit",))
        return _report
    cmd = _command(Limits())
    cmd[-1] = _PROBE
    marker = os.path.join(tempfile.gettempdir(), "__probe")
    try:
        os.unlink(marker)
    except FileNotFoundError:
        pass
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        facts = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception as e:  # any failure to run the stack means it is not usable
        _report = IsolationReport(False, {}, (f"probe failed: {type(e).__name__}: {e}",))
        return _report
    problems = []
    if facts.get("uid") != NOBODY or facts.get("gid") != NOBODY:
        problems.append("not running as the unprivileged user")
    if not facts.get("net_blocked"):
        problems.append("network reachable")
    if facts.get("visible_pids") != 1:
        problems.append("host processes visible")
    if set(facts.get("env_keys", [])) - {"LC_ALL", "LC_CTYPE", "PATH", "PWD", "OLDPWD"}:
        problems.append("environment not scrubbed")
    if facts.get("home_entries") != 0:
        problems.append("/home not hidden")
    if facts.get("reads_/root") or facts.get("reads_/etc/shadow"):
        problems.append("privileged paths readable")
    if os.path.exists(marker):
        problems.append("/tmp shared with the host")
    _report = IsolationReport(not problems, facts, tuple(problems))
    return _report


def require_isolation() -> None:
    rep = detect_isolation()
    if not rep.strong:
        raise SandboxUnavailable("; ".join(rep.problems) or "isolation unavailable")
