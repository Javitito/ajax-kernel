from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from agency import driver_revive


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "drivers" / "os_driver.py"
LAUNCHER = REPO_ROOT / "Start-AjaxDriver.ps1"


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _http_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=2.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_http_json(url: str, *, timeout_s: float = 6.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return _http_json(url)
        except Exception as exc:  # pragma: no cover - retry loop
            last_error = exc
            time.sleep(0.1)
    if last_error:
        raise last_error
    raise AssertionError(f"timed out waiting for {url}")


def _start_entrypoint(root_dir: Path, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(ENTRYPOINT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--root",
            str(root_dir),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


def _kill_pid(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            pass


def _copy_prod_surface(root: Path) -> None:
    (root / "drivers").mkdir(parents=True, exist_ok=True)
    (root / "Start-AjaxDriver.ps1").write_text(LAUNCHER.read_text(encoding="utf-8"), encoding="utf-8")
    (root / "drivers" / "os_driver.py").write_text(ENTRYPOINT.read_text(encoding="utf-8"), encoding="utf-8")


def _receipt_events(result: dict[str, Any]) -> list[str]:
    events: list[str] = []
    for raw in result.get("receipt_paths") or []:
        payload = json.loads(Path(str(raw)).read_text(encoding="utf-8"))
        events.append(str(payload.get("event")))
    return events


def _launch_payload(result: dict[str, Any]) -> dict[str, Any]:
    detail = dict(((result.get("launch") or {}).get("detail") or {}))
    stdout_path = detail.get("stdout_path")
    if isinstance(stdout_path, str) and stdout_path:
        return json.loads(Path(stdout_path).read_text(encoding="utf-8"))
    raw = str(detail.get("stdout") or "").strip()
    if not raw:
        return {}
    return json.loads(raw)


def _failure_payload(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    raw = str(proc.stderr or proc.stdout or "").strip()
    assert raw, "expected explicit startup failure payload"
    return json.loads(raw)


def test_prod_driver_entrypoint_serves_real_health_capabilities_and_displays(tmp_path: Path) -> None:
    port = _free_port()
    proc = _start_entrypoint(tmp_path, port)
    try:
        health = _wait_http_json(f"http://127.0.0.1:{port}/health")
        caps = _http_json(f"http://127.0.0.1:{port}/capabilities")
        displays = _http_json(f"http://127.0.0.1:{port}/displays")

        assert health["ok"] is True
        assert health["driver"] == "prod_os_driver"
        assert health["simulated"] is False
        assert health["port"] == port
        assert caps["ok"] is True
        assert caps["driver"] == "prod_os_driver"
        assert caps["backends"]["actions"] is False
        assert caps["driver_file"].endswith("drivers\\os_driver.py") or caps["driver_file"].endswith("drivers/os_driver.py")
        assert "health" in caps["capabilities"]
        assert displays["ok"] is True
        assert isinstance(displays["displays"], list)
        meta_path = tmp_path / "artifacts" / "pids" / f"prod_os_driver_{port}.json"
        assert meta_path.exists()
    finally:
        _stop_process(proc)


def test_prod_driver_entrypoint_bind_failure_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    port = _free_port()
    proc = _start_entrypoint(tmp_path, port)
    try:
        _wait_http_json(f"http://127.0.0.1:{port}/health")
        second = subprocess.run(
            [
                sys.executable,
                str(ENTRYPOINT),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--root",
                str(tmp_path),
            ],
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        payload = _failure_payload(second)

        assert second.returncode == 2
        assert payload["ok"] is False
        assert payload["reason"] == "bind_failed"
        assert payload["port"] == port
    finally:
        _stop_process(proc)


def test_run_driver_revive_prod_succeeds_with_real_entrypoint_present(tmp_path: Path, monkeypatch) -> None:
    _copy_prod_surface(tmp_path)
    port = _free_port()
    monkeypatch.setenv("OS_DRIVER_HOST", "127.0.0.1")
    monkeypatch.setenv("OS_DRIVER_PORT", str(port))
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))

    result = driver_revive.run_driver_revive(
        root_dir=tmp_path,
        rail="prod",
        launch_timeout_s=12.0,
        postcheck_timeout_s=2.0,
        postcheck_poll_s=0.2,
    )

    launch_payload = _launch_payload(result)
    pid = launch_payload.get("pid") if isinstance(launch_payload, dict) else None
    try:
        assert result["ok"] is True
        assert result["launch_attempted"] is True
        assert result["post_health"]["healthy"] is True
        assert launch_payload["ok"] is True
        assert launch_payload["status"] == "healthy"
        assert launch_payload["resolved_target"].endswith("drivers\\os_driver.py") or launch_payload["resolved_target"].endswith("drivers/os_driver.py")
        assert _receipt_events(result) == [
            "driver_health_checked",
            "driver_revive_launch_attempted",
            "driver_revive_postcheck_success",
        ]
    finally:
        _kill_pid(pid if isinstance(pid, int) else None)
        time.sleep(0.2)
