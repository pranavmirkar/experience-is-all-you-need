import os
import subprocess
import time

import pytest

from experience_learning.stage2 import sandbox as sb

pytestmark = pytest.mark.skipif(not sb.detect_isolation().strong, reason="strong sandbox isolation unavailable here")

FAST = sb.Limits(per_call_s=1.0, total_s=6.0)


def run1(src, inputs=([],), limits=FAST):
    return sb.run_candidate(src, "solve", list(inputs), limits)


def test_isolation_report_is_strong_and_specific():
    rep = sb.detect_isolation()
    assert rep.strong and not rep.problems
    f = rep.facts
    assert f["uid"] == sb.NOBODY and f["net_blocked"] and f["visible_pids"] == 1 and f["home_entries"] == 0


def test_ordinary_code_runs_and_returns_json_values():
    res = run1("def solve(a, b):\n    return [a + b, {'k': (a, b)}]", [[1, 2], [3, 4]])
    assert [c.value for c in res.calls] == [[3, {"k": [1, 2]}], [7, {"k": [3, 4]}]]


@pytest.mark.parametrize(
    "src,needle",
    [
        ("def solve():\n    while True: pass", "timeout"),
        ("def solve():\n    return len([0] * 10**10)", "MemoryError"),
        ("import socket\ndef solve():\n    socket.create_connection(('1.1.1.1', 80), timeout=2)", "OSError"),
        ("def solve():\n    return open('/home/user/pyproject.toml').read()", "FileNotFoundError"),
        ("def solve():\n    return open('/etc/shadow').read()", "Error"),
        ("def solve():\n    return {1, 2}", "not JSON serializable"),
        ("def solve():\n    1 / 0", "ZeroDivisionError"),
    ],
)
def test_hostile_or_broken_candidates_fail_cleanly(src, needle):
    c = run1(src).calls[0]
    assert not c.ok and needle in (c.error or "")


def test_candidate_cannot_write_to_the_host():
    marker = "/tmp/pwned_by_candidate_test"
    run1(f"def solve():\n    open({marker!r}, 'w').write('x'); return 1")
    assert not os.path.exists(marker)
    run1("def solve():\n    open('/home/user/pwned_by_candidate_test', 'w'); return 1")
    assert not os.path.exists("/home/user/pwned_by_candidate_test")


def test_candidate_sees_no_credentials_or_host_processes():
    src = "import os\ndef solve():\n    return [sorted(os.environ), len([p for p in os.listdir('/proc') if p.isdigit()])]"
    keys, pids = run1(src).calls[0].value
    assert pids == 1 and not {k for k in keys if "TOKEN" in k or "KEY" in k or "PROXY" in k or "SECRET" in k}


def test_forged_result_lines_do_not_change_the_answer():
    src = (
        "import sys\n"
        "sys.__stdout__.write('{\"i\": 0, \"ok\": true, \"v\": 42}\\n')\n"
        "def solve():\n    return -1"
    )
    assert run1(src).calls[0].value == -1


def test_exit_swallowed_alarm_and_syntax_error():
    assert not run1("import os\ndef solve():\n    os._exit(0)").calls[0].ok
    res = run1("def solve():\n    while True:\n        try:\n            while True: pass\n        except BaseException: pass",
               limits=sb.Limits(per_call_s=1.0, total_s=3.0))
    assert res.killed and not res.calls[0].ok
    res = run1("def solve(:\n  pass")
    assert res.definition_error and "SyntaxError" in res.definition_error


def test_no_sandbox_processes_survive_a_killed_run():
    run1("import os\ndef solve():\n    for _ in range(20):\n        if os.fork() == 0:\n            while True: pass\n    while True: pass",
         limits=sb.Limits(per_call_s=1.0, total_s=3.0))
    deadline = time.time() + 5
    while time.time() < deadline:
        n = subprocess.run("ps -eo user | grep -c '^nobody'", shell=True, capture_output=True, text=True).stdout.strip()
        if n == "0":
            return
        time.sleep(0.1)
    pytest.fail("sandbox processes still running 5 s after kill")


def test_refuses_to_run_when_isolation_is_not_strong(monkeypatch):
    weak = sb.IsolationReport(False, {}, ("test: pretend network reachable",))
    monkeypatch.setattr(sb, "detect_isolation", lambda refresh=False: weak)

    def boom(*a, **k):
        raise AssertionError("a process was started without isolation")

    monkeypatch.setattr(sb.subprocess, "Popen", boom)
    with pytest.raises(sb.SandboxUnavailable):
        sb.run_candidate("def solve():\n    return 1", "solve", [[]])
