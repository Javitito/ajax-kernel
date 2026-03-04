from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from agency.ajax_core import AjaxCore
from agency.system_executor import SystemExecutor


def test_ajax_core_does_not_use_subprocess_directly() -> None:
    src = Path("agency/ajax_core.py").read_text(encoding="utf-8")
    assert "subprocess.run" not in src
    assert "subprocess.Popen" not in src


def test_system_executor_smoke() -> None:
    executor = SystemExecutor()
    proc = executor.run(
        [sys.executable, "-c", "print('ok')"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert proc.returncode == 0
    assert (proc.stdout or "").strip() == "ok"


def test_paths_using_executor_work() -> None:
    calls = []

    class _FakeExecutor:
        def run(self, cmd, **kwargs):
            calls.append((cmd, kwargs))
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    core = AjaxCore.__new__(AjaxCore)
    core.system_executor = _FakeExecutor()

    out = AjaxCore._tasklist_snapshot(core, ["notepad.exe"])

    assert "notepad.exe" in out
    assert out["notepad.exe"]["rc"] == 0
    assert len(calls) == 1
    assert calls[0][0][0].lower() == "tasklist.exe"
    assert "IMAGENAME eq notepad.exe" in calls[0][0][-1]


def test_no_behavior_regression_minimal() -> None:
    core = AjaxCore.__new__(AjaxCore)
    core.system_executor = SystemExecutor()

    result = AjaxCore._run_cli_with_timeouts(
        core,
        [sys.executable, "-c", "import sys; sys.stdout.write('hello')"],
        input_text=None,
        timeout_cfg={
            "total_timeout_ms": 5000,
            "first_output_timeout_ms": 2000,
            "stall_timeout_ms": 2000,
        },
    )

    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["timeout_kind"] == "NONE"
    assert result["client_abort"] is False
    assert result["stdout"] == "hello"
