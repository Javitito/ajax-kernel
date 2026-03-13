from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agency.driver_keys import load_ajax_driver_api_key
from agency.lab_dummy_driver import ensure_dummy_driver


DEFAULT_DRIVER_HOST = "127.0.0.1"
DEFAULT_DRIVER_PORT_PROD = 5010
DEFAULT_DRIVER_PORT_LAB = 5012
DEFAULT_LAUNCH_TIMEOUT_S = 12.0
DEFAULT_POSTCHECK_TIMEOUT_S = 6.0
DEFAULT_POSTCHECK_POLL_S = 1.0


@dataclass(frozen=True)
class DriverEndpoint:
    rail: str
    host: str
    port: int
    url: str
    source: str


@dataclass(frozen=True)
class DriverReviveTarget:
    rail: str
    target_kind: str
    resolved_target: str
    launcher: Optional[str]
    target_exists: bool
    launcher_exists: bool
    available: bool
    unavailable_reason: Optional[str]
    command: list[str]


def normalize_rail(raw: Optional[str]) -> str:
    value = str(raw or "").strip().lower()
    if value in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ts_label() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"{int((time.time() % 1) * 1000):03d}Z"


def _parse_host_port(raw_url: str, *, default_port: int) -> tuple[str, int]:
    raw = str(raw_url or "").strip()
    if not raw:
        return DEFAULT_DRIVER_HOST, int(default_port)
    parsed = urllib.parse.urlparse(raw if "://" in raw else f"http://{raw}")
    host = parsed.hostname or DEFAULT_DRIVER_HOST
    try:
        port = int(parsed.port or default_port)
    except Exception:
        port = int(default_port)
    return host, port


def resolve_driver_endpoint(rail: Optional[str], *, environ: Optional[Dict[str, str]] = None) -> DriverEndpoint:
    env = environ if environ is not None else os.environ
    rail_n = normalize_rail(rail)
    if rail_n == "prod":
        env_url = str(env.get("OS_DRIVER_URL") or "").strip()
        if env_url:
            host, port = _parse_host_port(env_url, default_port=DEFAULT_DRIVER_PORT_PROD)
            return DriverEndpoint(
                rail=rail_n,
                host=host,
                port=port,
                url=f"http://{host}:{port}",
                source="env:OS_DRIVER_URL",
            )
        host = str(env.get("OS_DRIVER_HOST") or "").strip() or DEFAULT_DRIVER_HOST
        raw_port = str(env.get("OS_DRIVER_PORT") or DEFAULT_DRIVER_PORT_PROD).strip() or str(DEFAULT_DRIVER_PORT_PROD)
        try:
            port = int(raw_port)
        except Exception:
            port = DEFAULT_DRIVER_PORT_PROD
        return DriverEndpoint(
            rail=rail_n,
            host=host,
            port=port,
            url=f"http://{host}:{port}",
            source="env:OS_DRIVER_HOST+OS_DRIVER_PORT" if str(env.get("OS_DRIVER_HOST") or "").strip() else "default:prod",
        )

    env_url = str(env.get("OS_DRIVER_URL_LAB") or "").strip()
    if env_url:
        host, port = _parse_host_port(env_url, default_port=DEFAULT_DRIVER_PORT_LAB)
        return DriverEndpoint(
            rail=rail_n,
            host=host,
            port=port,
            url=f"http://{host}:{port}",
            source="env:OS_DRIVER_URL_LAB",
        )
    host = str(env.get("OS_DRIVER_HOST_LAB") or "").strip() or DEFAULT_DRIVER_HOST
    raw_port = str(env.get("OS_DRIVER_PORT_LAB") or DEFAULT_DRIVER_PORT_LAB).strip() or str(DEFAULT_DRIVER_PORT_LAB)
    try:
        port = int(raw_port)
    except Exception:
        port = DEFAULT_DRIVER_PORT_LAB
    return DriverEndpoint(
        rail=rail_n,
        host=host,
        port=port,
        url=f"http://{host}:{port}",
        source="env:OS_DRIVER_HOST_LAB+OS_DRIVER_PORT_LAB" if str(env.get("OS_DRIVER_HOST_LAB") or "").strip() else "default:lab",
    )


def _powershell_path() -> Optional[str]:
    launcher = shutil.which("powershell.exe") or shutil.which("powershell")
    if launcher:
        return launcher
    unix = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if unix.exists():
        return str(unix)
    if os.name == "nt":
        win = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
        if win.exists():
            return str(win)
    return None


def _python_launcher() -> Optional[str]:
    candidates = [
        sys.executable,
        shutil.which("python"),
        shutil.which("py"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if Path(candidate).exists() or shutil.which(str(candidate)):
                return str(candidate)
        except Exception:
            continue
    return None


def _wsl_to_windows_path(path: Path) -> str:
    raw = str(path)
    if raw.startswith("/mnt/"):
        parts = raw.split("/")
        if len(parts) >= 4:
            drive = parts[2]
            rest = "\\".join(parts[3:])
            return f"{drive.upper()}:\\{rest}"
    return raw


def resolve_driver_revive_target(
    root_dir: Path,
    rail: Optional[str],
    *,
    endpoint: Optional[DriverEndpoint] = None,
) -> DriverReviveTarget:
    root = Path(root_dir)
    endpoint_n = endpoint or resolve_driver_endpoint(rail)
    rail_n = endpoint_n.rail

    if rail_n == "lab":
        target = root / "agency" / "lab_dummy_driver.py"
        launcher = _python_launcher()
        target_exists = target.exists()
        launcher_exists = bool(launcher)
        reason = None
        if not target_exists:
            reason = "missing_entrypoint"
        elif not launcher_exists:
            reason = "missing_launcher"
        return DriverReviveTarget(
            rail=rail_n,
            target_kind="python_module",
            resolved_target=str(target),
            launcher=launcher,
            target_exists=target_exists,
            launcher_exists=launcher_exists,
            available=target_exists and launcher_exists,
            unavailable_reason=reason,
            command=[
                str(launcher or "python"),
                "-m",
                "agency.lab_dummy_driver",
                "--serve",
                "--root",
                str(root),
                "--host",
                endpoint_n.host,
                "--port",
                str(endpoint_n.port),
            ],
        )

    target = root / "Start-AjaxDriver.ps1"
    launcher = _powershell_path()
    target_exists = target.exists()
    launcher_exists = bool(launcher)
    reason = None
    if not target_exists:
        reason = "missing_entrypoint"
    elif not launcher_exists:
        reason = "missing_launcher"
    return DriverReviveTarget(
        rail=rail_n,
        target_kind="powershell_script",
        resolved_target=str(target),
        launcher=launcher,
        target_exists=target_exists,
        launcher_exists=launcher_exists,
        available=target_exists and launcher_exists,
        unavailable_reason=reason,
        command=[
            str(launcher or "powershell.exe"),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _wsl_to_windows_path(target),
            "-Port",
            str(endpoint_n.port),
        ],
    )


def check_driver_health(endpoint: DriverEndpoint, *, timeout_s: float = 1.5) -> Dict[str, Any]:
    url = f"{endpoint.url.rstrip('/')}/health"
    headers: Dict[str, str] = {}
    try:
        api_key = load_ajax_driver_api_key()
        if api_key:
            headers["X-AJAX-KEY"] = str(api_key)
    except Exception:
        pass
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(raw) if raw else {}
        healthy = bool(isinstance(payload, dict) and payload.get("ok"))
        detail = "health_ok" if healthy else "health_not_ok"
        return {
            "healthy": healthy,
            "reachable": True,
            "http_status": int(getattr(resp, "status", 200)),
            "detail": detail,
            "url": url,
            "payload": payload if isinstance(payload, dict) else {},
            "simulated": bool(isinstance(payload, dict) and payload.get("simulated")),
        }
    except urllib.error.HTTPError as exc:
        detail = f"http_{int(exc.code)}"
        if int(exc.code) in {401, 403}:
            detail = "auth_challenge"
        return {
            "healthy": int(exc.code) in {401, 403},
            "reachable": True,
            "http_status": int(exc.code),
            "detail": detail,
            "url": url,
            "payload": {},
            "simulated": False,
        }
    except Exception as exc:
        return {
            "healthy": False,
            "reachable": False,
            "http_status": None,
            "detail": str(exc)[:240],
            "url": url,
            "payload": {},
            "simulated": False,
        }


def build_driver_client_for_rail(rail: Optional[str], *, timeout_s: Optional[float] = None) -> Any:
    from agency.windows_driver_client import WindowsDriverClient

    endpoint = resolve_driver_endpoint(rail)
    return WindowsDriverClient(
        base_url=endpoint.url,
        timeout_s=timeout_s,
        prefer_env=False,
    )


def _write_receipt(root_dir: Path, event: str, payload: Dict[str, Any]) -> str:
    path = Path(root_dir) / "artifacts" / "receipts" / f"{event}_{_ts_label()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _receipt_payload(
    *,
    event: str,
    endpoint: DriverEndpoint,
    target: DriverReviveTarget,
    timeout_s: float,
    pre_health: Optional[Dict[str, Any]],
    post_health: Optional[Dict[str, Any]],
    launch_attempted: bool,
    failure_reason: Optional[str] = None,
    launch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": "ajax.driver_revive_receipt.v1",
        "ts_utc": _utc_now(),
        "event": event,
        "rail": endpoint.rail,
        "host": endpoint.host,
        "port": endpoint.port,
        "url": endpoint.url,
        "endpoint_source": endpoint.source,
        "resolved_target": target.resolved_target,
        "target_kind": target.target_kind,
        "target_exists": target.target_exists,
        "launcher": target.launcher,
        "launcher_exists": target.launcher_exists,
        "target_available": target.available,
        "timeout_s": timeout_s,
        "launch_attempted": launch_attempted,
        "pre_health": pre_health,
        "post_health": post_health,
    }
    if failure_reason:
        payload["failure_reason"] = failure_reason
    if launch is not None:
        payload["launch"] = launch
    return payload


def _launch_target(
    root_dir: Path,
    target: DriverReviveTarget,
    endpoint: DriverEndpoint,
    *,
    system_executor: Optional[Any],
    timeout_s: float,
) -> Dict[str, Any]:
    if target.target_kind == "python_module":
        try:
            result = ensure_dummy_driver(root_dir, host=endpoint.host, port=endpoint.port)
            ok = bool(isinstance(result, dict) and result.get("ok"))
            return {
                "ok": ok,
                "returncode": 0 if ok else 1,
                "timed_out": False,
                "detail": result,
                "reason": None if ok else str((result or {}).get("error") or "lab_dummy_launch_failed"),
            }
        except Exception as exc:
            return {
                "ok": False,
                "returncode": None,
                "timed_out": False,
                "detail": {"error": str(exc)[:240]},
                "reason": f"launch_exception:{str(exc)[:120]}",
            }

    runner = system_executor.run if system_executor is not None else subprocess.run
    try:
        proc = runner(
            target.command,
            cwd=str(root_dir),
            check=False,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            errors="replace",
        )
        ok = int(getattr(proc, "returncode", 1) or 0) == 0
        return {
            "ok": ok,
            "returncode": int(getattr(proc, "returncode", 1)),
            "timed_out": False,
            "detail": {
                "stdout": str(getattr(proc, "stdout", "") or "")[:400],
                "stderr": str(getattr(proc, "stderr", "") or "")[:400],
            },
            "reason": None if ok else f"launch_returncode_{int(getattr(proc, 'returncode', 1))}",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "timed_out": True,
            "detail": {"stdout": str(exc.stdout or "")[:400], "stderr": str(exc.stderr or "")[:400]},
            "reason": "launch_timeout",
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "timed_out": False,
            "detail": {"error": str(exc)[:240]},
            "reason": f"launch_exception:{str(exc)[:120]}",
        }


def _wait_for_postcheck(
    endpoint: DriverEndpoint,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> Dict[str, Any]:
    deadline = time.time() + max(timeout_s, poll_interval_s)
    last = check_driver_health(endpoint)
    while time.time() < deadline:
        if last.get("healthy") is True:
            return last
        time.sleep(poll_interval_s)
        last = check_driver_health(endpoint)
    return last


def run_driver_revive(
    *,
    root_dir: Path,
    rail: Optional[str],
    system_executor: Optional[Any] = None,
    launch_timeout_s: float = DEFAULT_LAUNCH_TIMEOUT_S,
    postcheck_timeout_s: float = DEFAULT_POSTCHECK_TIMEOUT_S,
    postcheck_poll_s: float = DEFAULT_POSTCHECK_POLL_S,
) -> Dict[str, Any]:
    root = Path(root_dir)
    endpoint = resolve_driver_endpoint(rail)
    target = resolve_driver_revive_target(root, endpoint.rail, endpoint=endpoint)
    receipt_paths: list[str] = []

    pre_health = check_driver_health(endpoint)
    receipt_paths.append(
        _write_receipt(
            root,
            "driver_health_checked",
            _receipt_payload(
                event="driver_health_checked",
                endpoint=endpoint,
                target=target,
                timeout_s=launch_timeout_s,
                pre_health=pre_health,
                post_health=None,
                launch_attempted=False,
            ),
        )
    )

    result: Dict[str, Any] = {
        "schema": "ajax.driver_revive.v1",
        "ok": False,
        "rail": endpoint.rail,
        "endpoint": asdict(endpoint),
        "target": asdict(target),
        "pre_health": pre_health,
        "post_health": None,
        "launch_attempted": False,
        "skipped": False,
        "failure_reason": None,
        "receipt_paths": receipt_paths,
        "status": "UNHEALTHY",
    }

    if pre_health.get("healthy") is True:
        receipt_paths.append(
            _write_receipt(
                root,
                "driver_revive_skipped_healthy",
                _receipt_payload(
                    event="driver_revive_skipped_healthy",
                    endpoint=endpoint,
                    target=target,
                    timeout_s=launch_timeout_s,
                    pre_health=pre_health,
                    post_health=pre_health,
                    launch_attempted=False,
                ),
            )
        )
        result.update(
            {
                "ok": True,
                "post_health": pre_health,
                "skipped": True,
                "status": "HEALTHY_SKIP",
            }
        )
        return result

    if not target.available:
        failure_reason = target.unavailable_reason or "rail_target_unavailable"
        receipt_paths.append(
            _write_receipt(
                root,
                "driver_revive_target_missing",
                _receipt_payload(
                    event="driver_revive_target_missing",
                    endpoint=endpoint,
                    target=target,
                    timeout_s=launch_timeout_s,
                    pre_health=pre_health,
                    post_health=None,
                    launch_attempted=False,
                    failure_reason=failure_reason,
                ),
            )
        )
        result.update({"failure_reason": failure_reason, "status": "TARGET_UNAVAILABLE"})
        return result

    launch = _launch_target(root, target, endpoint, system_executor=system_executor, timeout_s=launch_timeout_s)
    result["launch_attempted"] = True
    result["launch"] = launch
    receipt_paths.append(
        _write_receipt(
            root,
            "driver_revive_launch_attempted",
            _receipt_payload(
                event="driver_revive_launch_attempted",
                endpoint=endpoint,
                target=target,
                timeout_s=launch_timeout_s,
                pre_health=pre_health,
                post_health=None,
                launch_attempted=True,
                launch=launch,
            ),
        )
    )

    if not launch.get("ok"):
        receipt_paths.append(
            _write_receipt(
                root,
                "driver_revive_launch_timeout_or_failed",
                _receipt_payload(
                    event="driver_revive_launch_timeout_or_failed",
                    endpoint=endpoint,
                    target=target,
                    timeout_s=launch_timeout_s,
                    pre_health=pre_health,
                    post_health=None,
                    launch_attempted=True,
                    failure_reason=str(launch.get("reason") or "launch_failed"),
                    launch=launch,
                ),
            )
        )

    post_health = _wait_for_postcheck(
        endpoint,
        timeout_s=postcheck_timeout_s,
        poll_interval_s=postcheck_poll_s,
    )
    result["post_health"] = post_health

    if post_health.get("healthy") is True:
        receipt_paths.append(
            _write_receipt(
                root,
                "driver_revive_postcheck_success",
                _receipt_payload(
                    event="driver_revive_postcheck_success",
                    endpoint=endpoint,
                    target=target,
                    timeout_s=launch_timeout_s,
                    pre_health=pre_health,
                    post_health=post_health,
                    launch_attempted=True,
                    launch=launch,
                ),
            )
        )
        result.update({"ok": True, "status": "POSTCHECK_HEALTHY"})
        return result

    failure_reason = str(launch.get("reason") or "postcheck_failed")
    receipt_paths.append(
        _write_receipt(
            root,
            "driver_revive_postcheck_failed",
            _receipt_payload(
                event="driver_revive_postcheck_failed",
                endpoint=endpoint,
                target=target,
                timeout_s=launch_timeout_s,
                pre_health=pre_health,
                post_health=post_health,
                launch_attempted=True,
                failure_reason=failure_reason,
                launch=launch,
            ),
        )
    )
    result.update({"failure_reason": failure_reason, "status": "POSTCHECK_UNHEALTHY"})
    return result
