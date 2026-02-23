from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from agency.process_utils import pid_running


VERSION = "lab_dummy_driver_v1"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pid_path(root_dir: Path, port: int) -> Path:
    return Path(root_dir) / "artifacts" / "pids" / f"lab_dummy_driver_{int(port)}.pid"


def _meta_path(root_dir: Path, port: int) -> Path:
    return Path(root_dir) / "artifacts" / "pids" / f"lab_dummy_driver_{int(port)}.json"


def _module_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_pid(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_port_listening(host: str, port: int, timeout_s: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            return sock.connect_ex((host, int(port))) == 0
        except Exception:
            return False


def stop_dummy_driver(root_dir: Path, *, port: int = 5012) -> Dict[str, Any]:
    root = Path(root_dir)
    pidfile = _pid_path(root, port)
    metafile = _meta_path(root, port)
    pid = _read_pid(pidfile)
    if pid is None:
        return {"ok": True, "stopped": False, "reason": "pid_missing"}
    if not pid_running(pid):
        try:
            pidfile.unlink()
        except Exception:
            pass
        return {"ok": True, "stopped": False, "reason": "pid_not_running"}
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.2)
        if pid_running(pid):
            os.kill(pid, signal.SIGKILL)
    except Exception as exc:
        return {"ok": False, "stopped": False, "error": str(exc)[:200]}
    try:
        if pidfile.exists():
            pidfile.unlink()
    except Exception:
        pass
    try:
        if metafile.exists():
            metafile.unlink()
    except Exception:
        pass
    return {"ok": True, "stopped": True, "pid": pid}


def is_dummy_driver_simulated(root_dir: Path, *, port: int = 5012, host: str = "127.0.0.1") -> bool:
    root = Path(root_dir)
    pid = _read_pid(_pid_path(root, port))
    if pid is None or not pid_running(pid):
        return False
    if not is_port_listening(host, port, timeout_s=0.2):
        return False
    meta = {}
    try:
        meta = json.loads(_meta_path(root, port).read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(isinstance(meta, dict) and meta.get("simulated"))


class _DummyHandler(BaseHTTPRequestHandler):
    server_version = "AjaxLabDummy/1.0"

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(
                200,
                {
                    "ok": True,
                    "simulated": True,
                    "driver": "lab_dummy",
                    "ts_utc": _utc_now(),
                },
            )
            return
        if self.path == "/version":
            self._write_json(
                200,
                {
                    "ok": True,
                    "simulated": True,
                    "version": VERSION,
                    "ts_utc": _utc_now(),
                },
            )
            return
        if self.path == "/capabilities":
            self._write_json(
                200,
                {
                    "ok": True,
                    "simulated": True,
                    "capabilities": ["health", "version", "displays"],
                    "ts_utc": _utc_now(),
                },
            )
            return
        if self.path == "/displays":
            self._write_json(
                200,
                {
                    "ok": True,
                    "simulated": True,
                    "displays": [
                        {"id": 1, "is_primary": True, "name": "Primary (simulated)"},
                        {"id": 2, "is_primary": False, "name": "LAB Dummy (simulated)"},
                    ],
                },
            )
            return
        self._write_json(404, {"ok": False, "error": "not_found", "simulated": True})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def run_dummy_server(
    *,
    root_dir: Path,
    host: str = "127.0.0.1",
    port: int = 5012,
) -> None:
    root = Path(root_dir)
    pidfile = _pid_path(root, port)
    metafile = _meta_path(root, port)
    pidfile.parent.mkdir(parents=True, exist_ok=True)

    def _cleanup() -> None:
        try:
            if pidfile.exists() and pidfile.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pidfile.unlink()
        except Exception:
            pass

    pidfile.write_text(str(os.getpid()), encoding="utf-8")
    _write_json(
        metafile,
        {
            "schema": "ajax.lab_dummy_driver.v1",
            "simulated": True,
            "host": host,
            "port": int(port),
            "pid": os.getpid(),
            "started_utc": _utc_now(),
        },
    )
    atexit.register(_cleanup)
    server = ThreadingHTTPServer((host, int(port)), _DummyHandler)
    server.serve_forever()


def ensure_dummy_driver(
    root_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 5012,
) -> Dict[str, Any]:
    root = Path(root_dir)
    pidfile = _pid_path(root, port)
    pid = _read_pid(pidfile)
    if pid is not None and pid_running(pid) and is_port_listening(host, port):
        return {
            "ok": True,
            "started": False,
            "already_running": True,
            "simulated": True,
            "pid": pid,
            "port": int(port),
        }
    if is_port_listening(host, port):
        return {
            "ok": True,
            "started": False,
            "already_running": True,
            "simulated": False,
            "pid": None,
            "port": int(port),
        }

    cmd = [
        sys.executable,
        "-m",
        "agency.lab_dummy_driver",
        "--serve",
        "--root",
        str(root),
        "--host",
        str(host),
        "--port",
        str(int(port)),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=_module_root(),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=False,
    )
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if is_port_listening(host, port):
            return {
                "ok": True,
                "started": True,
                "already_running": False,
                "simulated": True,
                "pid": proc.pid,
                "port": int(port),
            }
        time.sleep(0.1)
    return {
        "ok": False,
        "started": False,
        "already_running": False,
        "simulated": True,
        "pid": proc.pid,
        "port": int(port),
        "error": "dummy_driver_start_timeout",
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lab_dummy_driver", description="Dummy LAB driver on 5012.")
    parser.add_argument("--root", default=".", help="Root del repo (AJAX_HOME).")
    parser.add_argument("--host", default="127.0.0.1", help="Host bind (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=5012, help="Port bind (default 5012).")
    parser.add_argument("--serve", action="store_true", help="Run blocking HTTP server.")
    parser.add_argument("--stop", action="store_true", help="Stop dummy server for this port.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if args.stop:
        out = stop_dummy_driver(root, port=int(args.port))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out.get("ok") else 2
    if args.serve:
        run_dummy_server(root_dir=root, host=str(args.host), port=int(args.port))
        return 0
    out = ensure_dummy_driver(root, host=str(args.host), port=int(args.port))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
