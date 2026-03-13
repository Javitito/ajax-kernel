from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agency import driver_revive


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_SOURCE = REPO_ROOT / "Start-AjaxDriver.ps1"


def _copy_launcher(root: Path) -> Path:
    target = root / "Start-AjaxDriver.ps1"
    target.write_text(LAUNCHER_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _powershell() -> str:
    launcher = driver_revive._powershell_path()
    assert launcher, "powershell launcher is required for these tests"
    return launcher


def _parse_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    assert text, "expected JSON payload from launcher"
    return json.loads(text)


def _ensure_fake_prod_driver(root: Path) -> Path:
    script = root / "drivers" / "os_driver.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        """from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"ok": True, "driver": "fake_prod"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )
    return script


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
            )
        except Exception:
            pass


def _kill_listener_on_port(port: int) -> None:
    try:
        subprocess.run(
            [
                _powershell(),
                "-NoProfile",
                "-Command",
                (
                    f"$conn = Get-NetTCPConnection -State Listen -LocalPort {int(port)} -ErrorAction SilentlyContinue | "
                    "Select-Object -First 1; "
                    "if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        pass


def _receipt_payloads(result: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for raw in result.get("receipt_paths") or []:
        payloads.append(json.loads(Path(str(raw)).read_text(encoding="utf-8")))
    return payloads


def _launch_payload_from_detail(detail: dict[str, Any]) -> dict[str, Any]:
    stdout_path = detail.get("stdout_path")
    if isinstance(stdout_path, str) and stdout_path:
        raw = Path(stdout_path).read_text(encoding="utf-8")
        return _parse_json(raw)
    return _parse_json(str(detail.get("stdout") or ""))


def test_resolve_driver_revive_target_prod_points_to_canonical_launcher_with_host_and_port(tmp_path: Path) -> None:
    endpoint = driver_revive.resolve_driver_endpoint(
        "prod",
        environ={"OS_DRIVER_HOST": "127.0.0.9", "OS_DRIVER_PORT": "5099"},
    )

    target = driver_revive.resolve_driver_revive_target(tmp_path, "prod", endpoint=endpoint)

    assert target.resolved_target == str(tmp_path / "Start-AjaxDriver.ps1")
    assert target.command[-4:] == ["-Host", "127.0.0.9", "-Port", "5099"]


def test_start_ajax_driver_missing_prod_entrypoint_fails_explicitly(tmp_path: Path) -> None:
    launcher = _copy_launcher(tmp_path)
    port = _free_port()

    proc = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(launcher),
            "-Root",
            str(tmp_path),
            "-Port",
            str(port),
            "-PythonExe",
            sys.executable,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = _parse_json(proc.stdout)

    assert proc.returncode == 3
    assert payload["ok"] is False
    assert payload["reason"] == "missing_driver_entrypoint"
    assert payload["rail"] == "prod"
    assert payload["launch_attempted"] is False


def test_run_driver_revive_prod_with_launcher_present_reports_real_prereq_failure(tmp_path: Path, monkeypatch) -> None:
    _copy_launcher(tmp_path)
    port = _free_port()
    monkeypatch.setenv("OS_DRIVER_HOST", "127.0.0.1")
    monkeypatch.setenv("OS_DRIVER_PORT", str(port))

    result = driver_revive.run_driver_revive(
        root_dir=tmp_path,
        rail="prod",
        launch_timeout_s=12.0,
        postcheck_timeout_s=1.0,
        postcheck_poll_s=0.2,
    )

    payloads = _receipt_payloads(result)
    events = [str(item.get("event")) for item in payloads]
    launch_receipt = next(item for item in payloads if item.get("event") == "driver_revive_launch_timeout_or_failed")
    stderr_text = str(((launch_receipt.get("launch") or {}).get("detail") or {}).get("stderr") or "")
    stdout_text = str(((launch_receipt.get("launch") or {}).get("detail") or {}).get("stdout") or "")

    assert result["ok"] is False
    assert result["launch_attempted"] is True
    assert result["target"]["target_exists"] is True
    assert result["failure_reason"] != "missing_entrypoint"
    assert events == [
        "driver_health_checked",
        "driver_revive_launch_attempted",
        "driver_revive_launch_timeout_or_failed",
        "driver_revive_postcheck_failed",
    ]
    assert "driver_revive_target_missing" not in events
    assert "missing_driver_entrypoint" in stdout_text or "missing_driver_entrypoint" in stderr_text


def test_run_driver_revive_prod_with_fake_driver_reaches_postcheck_success(tmp_path: Path, monkeypatch) -> None:
    _copy_launcher(tmp_path)
    _ensure_fake_prod_driver(tmp_path)
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

    launch_detail = dict(((result.get("launch") or {}).get("detail") or {}))
    launch_payload = _launch_payload_from_detail(launch_detail)
    pid = launch_payload.get("pid")
    try:
        assert result["ok"] is True
        assert result["launch_attempted"] is True
        assert result["post_health"]["healthy"] is True
        assert launch_payload["ok"] is True
        assert launch_payload["status"] == "healthy"
        assert launch_payload["rail"] == "prod"
        assert launch_payload["resolved_target"].endswith("drivers\\os_driver.py") or launch_payload["resolved_target"].endswith("drivers/os_driver.py")
        assert [item.get("event") for item in _receipt_payloads(result)] == [
            "driver_health_checked",
            "driver_revive_launch_attempted",
            "driver_revive_postcheck_success",
        ]
    finally:
        _kill_pid(pid if isinstance(pid, int) else None)
        _kill_listener_on_port(port)
        time.sleep(0.2)
