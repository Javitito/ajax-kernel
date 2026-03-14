from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ctypes import wintypes
except Exception:  # pragma: no cover
    wintypes = None  # type: ignore


VERSION = "prod_driver_entrypoint_real_v1"
DRIVER_NAME = "prod_os_driver"
SUPPORTED_CAPABILITIES = ["health", "version", "capabilities", "displays"]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_root(explicit_root: Optional[str]) -> Path:
    if explicit_root and str(explicit_root).strip():
        return Path(str(explicit_root)).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _driver_file() -> Path:
    return Path(__file__).resolve()


def _pid_path(root_dir: Path, port: int) -> Path:
    return root_dir / "artifacts" / "pids" / f"prod_os_driver_{int(port)}.pid"


def _meta_path(root_dir: Path, port: int) -> Path:
    return root_dir / "artifacts" / "pids" / f"prod_os_driver_{int(port)}.json"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _username() -> Optional[str]:
    for key in ("USERNAME", "USER"):
        raw = str(os.getenv(key) or "").strip()
        if raw:
            return raw
    return None


def _venv_path() -> Optional[str]:
    raw = str(os.getenv("VIRTUAL_ENV") or "").strip()
    if raw:
        return raw
    exe = Path(sys.executable).resolve()
    scripts = exe.parent
    if scripts.name.lower() == "scripts":
        return str(scripts.parent)
    return None


def _foreground_window() -> Dict[str, Any]:
    if os.name != "nt" or wintypes is None:
        return {}
    try:
        user32 = ctypes.windll.user32
        hwnd = int(user32.GetForegroundWindow() or 0)
        if hwnd <= 0:
            return {}
        title_buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title_buf, len(title_buf))
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return {
            "hwnd": hwnd,
            "title": title_buf.value,
            "pid": int(pid.value),
        }
    except Exception as exc:
        return {"error": str(exc)[:160]}


def _enumerate_displays() -> Dict[str, Any]:
    if os.name != "nt" or wintypes is None:
        return {"displays": [], "partial_inventory": True, "source": "non_windows"}

    displays: list[Dict[str, Any]] = []
    error_text: Optional[str] = None
    try:
        user32 = ctypes.windll.user32

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            ]

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32),
            ]

        monitor_enum_proc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HANDLE,
            wintypes.HDC,
            ctypes.POINTER(RECT),
            wintypes.LPARAM,
        )

        def _callback(hmonitor, hdc, lprc, lparam) -> bool:
            info = MONITORINFOEXW()
            info.cbSize = ctypes.sizeof(info)
            if not user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
                return True
            left = int(info.rcMonitor.left)
            top = int(info.rcMonitor.top)
            right = int(info.rcMonitor.right)
            bottom = int(info.rcMonitor.bottom)
            displays.append(
                {
                    "id": len(displays) + 1,
                    "name": str(info.szDevice),
                    "device": str(info.szDevice),
                    "is_primary": bool(int(info.dwFlags) & 1),
                    "left": left,
                    "top": top,
                    "width": max(0, right - left),
                    "height": max(0, bottom - top),
                }
            )
            return True

        cb = monitor_enum_proc(_callback)
        if not user32.EnumDisplayMonitors(0, 0, cb, 0):
            raise OSError("EnumDisplayMonitors failed")
        if displays:
            return {"displays": displays, "partial_inventory": False, "source": "win32_enum"}
    except Exception as exc:
        error_text = str(exc)[:160]

    try:
        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        fallback = []
        if width > 0 and height > 0:
            fallback.append(
                {
                    "id": 1,
                    "name": "Primary",
                    "device": "PRIMARY",
                    "is_primary": True,
                    "left": 0,
                    "top": 0,
                    "width": width,
                    "height": height,
                }
            )
        payload = {
            "displays": fallback,
            "partial_inventory": True,
            "source": "win32_primary_only",
        }
        if error_text:
            payload["enumeration_error"] = error_text
        return payload
    except Exception as exc:
        return {
            "displays": [],
            "partial_inventory": True,
            "source": "display_unavailable",
            "enumeration_error": error_text or str(exc)[:160],
        }


def _startup_failure(reason: str, *, root_dir: Path, host: str, port: int, error: str, exit_code: int) -> int:
    payload = {
        "schema": "ajax.prod_os_driver.startup.v1",
        "ok": False,
        "reason": reason,
        "error": error[:240],
        "host": host,
        "port": int(port),
        "repo_root": str(root_dir),
        "driver_file": str(_driver_file()),
        "python_exe": sys.executable,
        "ts_utc": _utc_now(),
    }
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
    return int(exit_code)


@dataclass
class DriverRuntime:
    root_dir: Path
    host: str
    port: int
    started_utc: str

    @property
    def driver_file(self) -> str:
        return str(_driver_file())

    def health_payload(self) -> Dict[str, Any]:
        displays = _enumerate_displays()
        return {
            "schema": "ajax.prod_os_driver.health.v1",
            "ok": True,
            "driver": DRIVER_NAME,
            "version": VERSION,
            "mode": "minimal_real",
            "simulated": False,
            "ts_utc": _utc_now(),
            "started_utc": self.started_utc,
            "repo_root": str(self.root_dir),
            "driver_file": self.driver_file,
            "host": self.host,
            "port": int(self.port),
            "pid": os.getpid(),
            "fg_window": _foreground_window(),
            "displays_count": len(displays.get("displays") or []),
        }

    def capabilities_payload(self) -> Dict[str, Any]:
        interactive = os.name == "nt"
        return {
            "schema": "ajax.prod_os_driver.capabilities.v1",
            "ok": True,
            "driver": DRIVER_NAME,
            "version": VERSION,
            "mode": "minimal_real",
            "simulated": False,
            "repo_root": str(self.root_dir),
            "driver_file": self.driver_file,
            "build_id": VERSION,
            "python_exe": sys.executable,
            "venv_path": _venv_path(),
            "user": _username(),
            "capabilities": list(SUPPORTED_CAPABILITIES),
            "backends": {
                "health": True,
                "version": True,
                "capabilities": True,
                "displays": True,
                "actions": False,
            },
            "backends_available": list(SUPPORTED_CAPABILITIES),
            "pyautogui_available": False,
            "pywinauto_available": False,
            "screenshot": {
                "supported": False,
                "backend_preferred": "none",
                "mss_available": False,
                "pillow_available": False,
                "interactive_desktop": interactive,
                "interactive_reason": "minimal_entrypoint_no_ui_actions",
            },
            "notes": [
                "Minimal real PROD entrypoint restored for health/revive truth.",
                "UI actions remain explicitly unavailable in this pass.",
            ],
            "ts_utc": _utc_now(),
        }

    def displays_payload(self) -> Dict[str, Any]:
        inventory = _enumerate_displays()
        return {
            "schema": "ajax.prod_os_driver.displays.v1",
            "ok": True,
            "driver": DRIVER_NAME,
            "version": VERSION,
            "simulated": False,
            "repo_root": str(self.root_dir),
            "driver_file": self.driver_file,
            "ts_utc": _utc_now(),
            **inventory,
        }


class _DriverServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True


class _DriverHandler(BaseHTTPRequestHandler):
    server_version = "AjaxProdDriver/1.0"

    @property
    def runtime(self) -> DriverRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _not_implemented(self, method: str) -> None:
        self._write_json(
            501,
            {
                "schema": "ajax.prod_os_driver.not_implemented.v1",
                "ok": False,
                "error": "not_implemented",
                "driver": DRIVER_NAME,
                "version": VERSION,
                "method": method,
                "path": self.path,
                "supported_capabilities": list(SUPPORTED_CAPABILITIES),
                "ts_utc": _utc_now(),
            },
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(200, self.runtime.health_payload())
            return
        if self.path == "/version":
            self._write_json(
                200,
                {
                    "schema": "ajax.prod_os_driver.version.v1",
                    "ok": True,
                    "driver": DRIVER_NAME,
                    "version": VERSION,
                    "repo_root": str(self.runtime.root_dir),
                    "driver_file": self.runtime.driver_file,
                    "ts_utc": _utc_now(),
                },
            )
            return
        if self.path == "/capabilities":
            self._write_json(200, self.runtime.capabilities_payload())
            return
        if self.path == "/displays":
            self._write_json(200, self.runtime.displays_payload())
            return
        if self.path == "/screenshot":
            self._not_implemented("GET")
            return
        self._write_json(404, {"ok": False, "error": "not_found", "driver": DRIVER_NAME, "path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        self._not_implemented("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._not_implemented("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._not_implemented("DELETE")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="os_driver", description="Minimal real PROD driver entrypoint for AJAX.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=5010, help="Bind port (default 5010).")
    parser.add_argument("--root", default=None, help="Repo root for artifacts and metadata.")
    args = parser.parse_args(argv)

    try:
        root_dir = _repo_root(args.root)
    except Exception as exc:
        return _startup_failure(
            "invalid_root",
            root_dir=Path.cwd(),
            host=str(args.host),
            port=int(args.port),
            error=str(exc),
            exit_code=3,
        )

    if not root_dir.exists() or not root_dir.is_dir():
        return _startup_failure(
            "invalid_root",
            root_dir=root_dir,
            host=str(args.host),
            port=int(args.port),
            error="root_dir_missing",
            exit_code=3,
        )

    runtime = DriverRuntime(
        root_dir=root_dir,
        host=str(args.host),
        port=int(args.port),
        started_utc=_utc_now(),
    )
    pidfile = _pid_path(root_dir, runtime.port)
    metafile = _meta_path(root_dir, runtime.port)

    def _cleanup() -> None:
        try:
            if pidfile.exists() and pidfile.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pidfile.unlink()
        except Exception:
            pass
        try:
            if metafile.exists():
                meta = json.loads(metafile.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and str(meta.get("pid")) == str(os.getpid()):
                    metafile.unlink()
        except Exception:
            pass

    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()), encoding="utf-8")
    _write_json(
        metafile,
        {
            "schema": "ajax.prod_os_driver.meta.v1",
            "driver": DRIVER_NAME,
            "version": VERSION,
            "host": runtime.host,
            "port": int(runtime.port),
            "pid": os.getpid(),
            "repo_root": str(root_dir),
            "driver_file": runtime.driver_file,
            "started_utc": runtime.started_utc,
        },
    )
    atexit.register(_cleanup)

    try:
        server = _DriverServer((runtime.host, runtime.port), _DriverHandler)
    except OSError as exc:
        _cleanup()
        return _startup_failure(
            "bind_failed",
            root_dir=root_dir,
            host=runtime.host,
            port=runtime.port,
            error=str(exc),
            exit_code=2,
        )
    except Exception as exc:
        _cleanup()
        return _startup_failure(
            "startup_exception",
            root_dir=root_dir,
            host=runtime.host,
            port=runtime.port,
            error=str(exc),
            exit_code=4,
        )

    server.runtime = runtime  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    finally:
        try:
            server.server_close()
        except Exception:
            pass
        _cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
