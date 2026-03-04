from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agency.health_ttl import provider_status_ttl
from agency.process_utils import pid_running

try:
    from agency.anchor_preflight import run_anchor_preflight
except Exception:  # pragma: no cover
    run_anchor_preflight = None  # type: ignore

try:
    from agency.driver_keys import load_ajax_driver_api_key
except Exception:  # pragma: no cover
    load_ajax_driver_api_key = None  # type: ignore


EXPECTED_TASKS_REL = Path("config") / "expected_tasks.json"
DOCTOR_OUT_REL = Path("artifacts") / "doctor"
RECEIPTS_REL = Path("artifacts") / "receipts"
OPS_OUT_REL = Path("artifacts") / "ops"
LAB_REL = Path("artifacts") / "lab"
DEFAULT_PORTS = (5010, 5012)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ts_label() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def windows_supported() -> bool:
    if os.name == "nt":
        return True
    return Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe").exists()


def _powershell_path() -> str:
    unix = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if unix.exists():
        return str(unix)
    if os.name == "nt":
        return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    return "powershell.exe"


def _public_log_root() -> Path:
    if os.name == "nt":
        return Path(os.getenv("PUBLIC") or r"C:\Users\Public") / "ajax"
    return Path("/mnt/c/Users/Public/ajax")


def _to_windows_path(path: Path) -> str:
    if os.name == "nt":
        return str(path)
    try:
        proc = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            raw = (proc.stdout or "").strip()
            if raw:
                return raw
    except Exception:
        pass
    return str(path)


def _run(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    cmd0 = str(cmd[0] if cmd else "")
    is_windows_exe = bool(os.name != "nt" and cmd0.lower().endswith(".exe"))
    kwargs: Dict[str, Any] = {"errors": "replace"}
    if is_windows_exe:
        kwargs["encoding"] = "cp850"
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        **kwargs,
    )


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _decode_json_from_stdout(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip().lstrip("\ufeff")
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _http_get_json(
    url: str,
    *,
    timeout_s: float,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            status = int(getattr(resp, "status", 200))
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = {
            "http_status": int(getattr(exc, "code", 0)),
            "error": "http_error",
        }
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        payload: Optional[Dict[str, Any]] = None
        if body.strip():
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = None
        if payload is not None:
            detail["error"] = str(payload.get("error") or payload.get("detail") or "http_error")
            return False, payload, detail
        return False, None, detail
    except Exception as exc:
        return False, None, {"error": "request_failed", "detail": str(exc)[:180]}
    try:
        payload = json.loads(raw)
    except Exception:
        return False, None, {"error": "response_not_json", "http_status": status}
    if not isinstance(payload, dict):
        return False, None, {"error": "response_not_object", "http_status": status}
    return True, payload, {"http_status": status}


def _probe_listener(port: int, *, host: str = "127.0.0.1", timeout_s: float = 0.35) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(float(timeout_s))
            return sock.connect_ex((host, int(port))) == 0
    except Exception:
        return False


def _driver_headers() -> Dict[str, str]:
    if load_ajax_driver_api_key is None:
        return {}
    try:
        key = load_ajax_driver_api_key()
    except Exception:
        key = None
    if not key:
        return {}
    return {"X-AJAX-KEY": str(key)}


def probe_port_state(
    *,
    port: int,
    timeout_s: float = 1.5,
    include_displays: bool = False,
) -> Dict[str, Any]:
    host = "127.0.0.1"
    base = f"http://{host}:{int(port)}"
    listener = _probe_listener(port, host=host)
    payload: Dict[str, Any] = {
        "port": int(port),
        "listener": bool(listener),
        "health_ok": False,
        "health_http_status": None,
        "health_error": None,
        "displays_ok": None,
        "displays_count": None,
        "displays_error": None,
    }
    if not listener:
        payload["health_error"] = "listener_down"
        if include_displays:
            payload["displays_error"] = "listener_down"
        return payload

    headers = _driver_headers()
    health_ok, health_doc, health_meta = _http_get_json(
        f"{base}/health",
        timeout_s=timeout_s,
        headers=headers,
    )
    payload["health_http_status"] = health_meta.get("http_status")
    if health_ok and isinstance(health_doc, dict):
        payload["health_ok"] = bool(health_doc.get("ok"))
        if not payload["health_ok"]:
            payload["health_error"] = str(health_doc.get("error") or "health_not_ok")
    else:
        status = int(health_meta.get("http_status") or 0)
        if status in {401, 403}:
            payload["health_ok"] = True
            payload["health_error"] = "health_auth_required"
        else:
            payload["health_ok"] = False
            payload["health_error"] = str(health_meta.get("error") or "health_unreachable")

    if not include_displays:
        return payload

    dis_ok, dis_doc, dis_meta = _http_get_json(
        f"{base}/displays",
        timeout_s=timeout_s,
        headers=headers,
    )
    if dis_ok and isinstance(dis_doc, dict):
        displays = dis_doc.get("displays") if isinstance(dis_doc.get("displays"), list) else []
        payload["displays_count"] = len(displays)
        payload["displays_ok"] = True
    else:
        status = int(dis_meta.get("http_status") or 0)
        if status in {401, 403}:
            payload["displays_ok"] = True
            payload["displays_error"] = "displays_auth_required"
        else:
            payload["displays_ok"] = False
            payload["displays_error"] = str(dis_meta.get("error") or "display_catalog_unavailable")
    return payload


def worker_heartbeat_status(root_dir: Path) -> Dict[str, Any]:
    heartbeat_path = root_dir / LAB_REL / "heartbeat.json"
    pid_path = root_dir / LAB_REL / "worker.pid"
    heartbeat = _read_json(heartbeat_path)
    hb_ts = None
    if isinstance(heartbeat, dict):
        try:
            hb_ts = float(heartbeat.get("ts"))
        except Exception:
            hb_ts = None
    hb_age = None
    if hb_ts is not None:
        hb_age = max(0.0, time.time() - hb_ts)
    pid = None
    running = False
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            running = pid_running(pid)
        except Exception:
            pid = None
            running = False
    try:
        stale_sec = float(os.getenv("AJAX_LAB_WORKER_STALE_SEC") or 30)
    except Exception:
        stale_sec = 30.0
    fresh = bool(running and isinstance(hb_age, (int, float)) and hb_age <= stale_sec)
    status = "fresh" if fresh else ("stale" if running else "down")
    return {
        "schema": "ajax.lab_worker_heartbeat.v1",
        "status": status,
        "fresh": fresh,
        "stale": not fresh,
        "stale_threshold_s": stale_sec,
        "heartbeat_age_s": hb_age,
        "last_heartbeat_ts": hb_ts,
        "pid": pid,
        "running": running,
        "heartbeat_path": str(heartbeat_path),
        "pid_path": str(pid_path),
    }


def load_expected_tasks(root_dir: Path, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    path = manifest_path or (Path(root_dir) / EXPECTED_TASKS_REL)
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid_expected_tasks_manifest:{path}")
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("expected_tasks_manifest_tasks_missing")
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(tasks_raw):
        if not isinstance(item, dict):
            raise ValueError(f"expected_task_invalid_entry:{idx}")
        name = str(item.get("task_name") or "").strip()
        if not name.startswith("\\"):
            raise ValueError(f"expected_task_name_invalid:{idx}")
        action = item.get("action") if isinstance(item.get("action"), dict) else {}
        script_rel = str(action.get("script") or "").strip()
        if not script_rel:
            raise ValueError(f"expected_task_action_script_missing:{name}")
        trigger = item.get("trigger") if isinstance(item.get("trigger"), dict) else {}
        settings = item.get("settings") if isinstance(item.get("settings"), dict) else {}
        normalized.append(
            {
                "task_name": name,
                "description": str(item.get("description") or "").strip(),
                "rail": str(item.get("rail") or "lab").strip().lower(),
                "trigger": {
                    "user": str(trigger.get("user") or raw.get("default_user") or "Javi").strip(),
                    "delay": str(trigger.get("delay") or raw.get("default_delay") or "PT45S").strip(),
                },
                "action": {
                    "script": script_rel.replace("\\", "/"),
                    "args": str(action.get("args") or "").strip(),
                },
                "settings": {
                    "restart_count": int(settings.get("restart_count") or 3),
                    "restart_interval": str(settings.get("restart_interval") or "PT1M").strip(),
                    "start_when_available": bool(settings.get("start_when_available", True)),
                    "run_with_highest": bool(settings.get("run_with_highest", True)),
                    "hidden": bool(settings.get("hidden", True)),
                },
            }
        )
    return {
        "schema": str(raw.get("schema") or "ajax.expected_tasks.v1"),
        "manifest_path": str(path),
        "tasks": normalized,
    }


def expected_task_names(manifest_payload: Dict[str, Any]) -> List[str]:
    tasks = manifest_payload.get("tasks") if isinstance(manifest_payload.get("tasks"), list) else []
    names = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        name = str(task.get("task_name") or "").strip()
        if name:
            names.append(name)
    return names


def watchdog_tick_decision(
    *,
    rail: str,
    listener: bool,
    health_ok: bool,
    displays_ok: Optional[bool],
) -> Dict[str, Any]:
    rail_n = str(rail or "lab").strip().lower()
    reasons: List[str] = []
    if not listener:
        reasons.append("listener_down")
    if not health_ok:
        reasons.append("health_not_ok")
    if rail_n == "lab" and displays_ok is not True:
        reasons.append("display_catalog_unavailable")
    return {
        "would_start": bool(reasons),
        "reasons": reasons,
        "rail": rail_n,
    }


def _powershell_json_script(
    root_dir: Path,
    script_rel: str,
    *,
    args: Optional[List[str]] = None,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    if not windows_supported():
        return {
            "ok": False,
            "status": "UNSUPPORTED_PLATFORM",
            "hint": "Run this command from Windows (PowerShell + Task Scheduler).",
            "script": script_rel,
        }
    script = Path(root_dir) / script_rel
    if not script.exists():
        return {
            "ok": False,
            "status": "MISSING_SCRIPT",
            "error": f"missing_script:{script_rel}",
            "script_path": str(script),
        }
    cmd = [
        _powershell_path(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        _to_windows_path(script),
    ]
    cmd.extend(args or [])
    proc = _run(cmd, cwd=Path(root_dir), timeout=timeout_s)
    payload = _decode_json_from_stdout(proc.stdout or "")
    if payload is None:
        payload = {
            "ok": False,
            "status": "INVALID_SCRIPT_OUTPUT",
            "error": "script_stdout_not_json",
        }
    payload.setdefault("ok", proc.returncode == 0)
    payload["script"] = script_rel
    payload["returncode"] = proc.returncode
    payload["stderr"] = (proc.stderr or "").strip()
    payload["stdout_truncated"] = (proc.stdout or "")[:400]
    return payload


def run_tasks_audit(root_dir: Path) -> Dict[str, Any]:
    args = ["-Root", _to_windows_path(Path(root_dir))]
    return _powershell_json_script(
        root_dir,
        "scripts/ops/ajax_tasks_audit.ps1",
        args=args,
        timeout_s=180,
    )


def run_tasks_ensure(root_dir: Path, *, apply: bool) -> Dict[str, Any]:
    args = ["-Root", _to_windows_path(Path(root_dir))]
    args.append("-Apply" if apply else "-DryRun")
    return _powershell_json_script(
        root_dir,
        "scripts/ops/ajax_tasks_ensure.ps1",
        args=args,
        timeout_s=180,
    )


def run_lab_bootstrap(root_dir: Path, *, rail: str = "lab") -> Dict[str, Any]:
    args = [
        "-Root",
        _to_windows_path(Path(root_dir)),
        "-Rail",
        str(rail or "lab").strip().lower(),
    ]
    return _powershell_json_script(
        root_dir,
        "scripts/ops/ajax_lab_bootstrap.ps1",
        args=args,
        timeout_s=180,
    )


def run_watchdog_tick(root_dir: Path, *, rail: str, port: int, ensure_worker: bool) -> Dict[str, Any]:
    args = [
        "-Root",
        _to_windows_path(Path(root_dir)),
        "-Rail",
        str(rail or "lab").strip().lower(),
        "-Port",
        str(int(port)),
        "-MaxTicks",
        "1",
    ]
    if ensure_worker:
        args.append("-EnsureWorker")
    return _powershell_json_script(
        root_dir,
        "scripts/ops/ajax_driver_watchdog.ps1",
        args=args,
        timeout_s=180,
    )


def _build_probable_causes(
    *,
    rail: str,
    tasks_audit: Dict[str, Any],
    ports: Dict[str, Dict[str, Any]],
    worker: Dict[str, Any],
    providers_ttl: Dict[str, Any],
    anchor: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    causes: List[Dict[str, Any]] = []
    if not bool(tasks_audit.get("ok")):
        causes.append(
            {
                "code": "tasks_audit_unavailable",
                "severity": "WARN",
                "detail": str(tasks_audit.get("status") or tasks_audit.get("error") or "tasks_audit_failed"),
            }
        )
    missing = tasks_audit.get("missing") if isinstance(tasks_audit.get("missing"), list) else []
    if missing:
        causes.append(
            {
                "code": "expected_tasks_missing",
                "severity": "BLOCKED",
                "detail": ", ".join(str(it) for it in missing),
            }
        )
    drifted = tasks_audit.get("drifted") if isinstance(tasks_audit.get("drifted"), list) else []
    if drifted:
        causes.append(
            {
                "code": "tasks_drift_detected",
                "severity": "BLOCKED",
                "detail": ", ".join(str(it) for it in drifted),
            }
        )

    lab_n = str(rail or "lab").strip().lower()
    for port in DEFAULT_PORTS:
        state = ports.get(str(port)) if isinstance(ports.get(str(port)), dict) else ports.get(port, {})
        if not isinstance(state, dict):
            state = {}
        if not bool(state.get("listener")):
            causes.append(
                {
                    "code": "expected_port_missing",
                    "severity": "BLOCKED",
                    "detail": f"listener_down:{port}",
                }
            )
            continue
        if not bool(state.get("health_ok")):
            causes.append(
                {
                    "code": "port_health_not_ok",
                    "severity": "BLOCKED",
                    "detail": f"health_not_ok:{port}",
                }
            )
        if port == 5012 and lab_n == "lab" and state.get("displays_ok") is not True:
            causes.append(
                {
                    "code": "display_catalog_unavailable",
                    "severity": "BLOCKED",
                    "detail": str(state.get("displays_error") or "displays_not_ok"),
                }
            )
    if not bool(worker.get("fresh")):
        causes.append(
            {
                "code": "lab_worker_not_running",
                "severity": "BLOCKED",
                "detail": f"worker_status={worker.get('status')}",
            }
        )
    if bool(providers_ttl.get("stale")):
        causes.append(
            {
                "code": "providers_status_stale",
                "severity": "WARN",
                "detail": str(providers_ttl.get("reason") or "providers_status_stale"),
            }
        )
    if isinstance(anchor, dict):
        mismatches = anchor.get("mismatches") if isinstance(anchor.get("mismatches"), list) else []
        for item in mismatches:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if not code:
                continue
            causes.append(
                {
                    "code": code,
                    "severity": "BLOCKED",
                    "detail": str(item.get("detail") or code),
                }
            )
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for cause in causes:
        key = str(cause.get("code") or "")
        detail = str(cause.get("detail") or "")
        identity = (key, detail)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(cause)
    return deduped


def _next_hints(rail: str) -> List[str]:
    rail_n = str(rail or "lab").strip().lower()
    hints = [
        "python bin/ajaxctl ops tasks audit",
        "python bin/ajaxctl ops tasks ensure --dry-run",
        "python bin/ajaxctl ops tasks ensure --apply",
        f"python bin/ajaxctl doctor anchor --rail {rail_n}",
        "python bin/ajaxctl doctor providers",
    ]
    if rail_n == "lab":
        hints.append("python bin/ajaxctl lab ensure --rail lab")
    else:
        hints.append("python bin/ajaxctl lab ensure --rail prod")
    return hints


def format_doctor_boot_summary(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("AJAX Doctor boot")
    lines.append(f"status: {payload.get('status')}")
    lines.append(f"rail: {payload.get('rail')}")
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), dict) else {}
    lines.append(
        "tasks: "
        f"ok={tasks.get('ok')} "
        f"missing={len(tasks.get('missing') or [])} "
        f"drifted={len(tasks.get('drifted') or [])}"
    )
    ports = payload.get("ports") if isinstance(payload.get("ports"), dict) else {}
    for port in DEFAULT_PORTS:
        state = ports.get(str(port)) if isinstance(ports.get(str(port)), dict) else {}
        lines.append(
            f"port {port}: listener={state.get('listener')} health_ok={state.get('health_ok')} displays_ok={state.get('displays_ok')}"
        )
    worker = payload.get("worker") if isinstance(payload.get("worker"), dict) else {}
    lines.append(
        f"worker: status={worker.get('status')} fresh={worker.get('fresh')} pid={worker.get('pid')}"
    )
    ttl = payload.get("providers_status_ttl") if isinstance(payload.get("providers_status_ttl"), dict) else {}
    lines.append(f"providers_status: stale={ttl.get('stale')} reason={ttl.get('reason')}")
    causes = payload.get("probable_causes") if isinstance(payload.get("probable_causes"), list) else []
    if causes:
        lines.append("probable_causes:")
        for cause in causes:
            if not isinstance(cause, dict):
                continue
            code = str(cause.get("code") or "unknown")
            detail = str(cause.get("detail") or "")
            lines.append(f"- {code}: {detail}")
    hints = payload.get("next_hint") if isinstance(payload.get("next_hint"), list) else []
    if hints:
        lines.append("next_hint:")
        for hint in hints:
            lines.append(f"- {hint}")
    return "\n".join(lines)


def run_doctor_boot(root_dir: Path, *, rail: str) -> Dict[str, Any]:
    root = Path(root_dir)
    rail_n = str(rail or "lab").strip().lower()
    task_audit = run_tasks_audit(root)
    ports = {
        "5010": probe_port_state(port=5010, include_displays=False),
        "5012": probe_port_state(port=5012, include_displays=(rail_n == "lab")),
    }
    worker = worker_heartbeat_status(root)
    try:
        ttl_seconds = int(os.getenv("AJAX_HEALTH_TTL_SEC") or 900)
    except Exception:
        ttl_seconds = 900
    providers_ttl = provider_status_ttl(root, ttl_seconds=ttl_seconds)

    anchor: Optional[Dict[str, Any]] = None
    if run_anchor_preflight is not None:
        try:
            anchor_payload = run_anchor_preflight(root_dir=root, rail=rail_n, write_receipt=True)
            anchor = anchor_payload if isinstance(anchor_payload, dict) else None
        except Exception as exc:
            anchor = {"ok": False, "code": "anchor_preflight_failed", "reason": str(exc)[:180]}

    causes = _build_probable_causes(
        rail=rail_n,
        tasks_audit=task_audit,
        ports=ports,
        worker=worker,
        providers_ttl=providers_ttl,
        anchor=anchor,
    )
    blocked = any(str(item.get("severity") or "").upper() == "BLOCKED" for item in causes if isinstance(item, dict))
    status = "READY" if not blocked else "BLOCKED"

    payload: Dict[str, Any] = {
        "schema": "ajax.doctor.boot.v1",
        "ts_utc": _utc_now(),
        "rail": rail_n,
        "status": status,
        "ok": status == "READY",
        "tasks": task_audit,
        "ports": ports,
        "worker": worker,
        "providers_status_ttl": providers_ttl,
        "anchor": anchor,
        "probable_causes": causes,
        "next_hint": _next_hints(rail_n),
    }

    out_dir = root / DOCTOR_OUT_REL / _ts_label()
    out_dir.mkdir(parents=True, exist_ok=True)
    explain_path = out_dir / "boot_explain.json"
    _write_json(explain_path, payload)

    receipt_dir = root / RECEIPTS_REL
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"doctor_boot_{_ts_label()}.json"
    receipt_payload = {
        "schema": "ajax.doctor.boot.receipt.v1",
        "ts_utc": _utc_now(),
        "ok": payload.get("ok"),
        "status": payload.get("status"),
        "rail": rail_n,
        "artifact_path": str(explain_path),
        "probable_causes_count": len(causes),
    }
    _write_json(receipt_path, receipt_payload)
    payload["artifact_path"] = str(explain_path)
    payload["receipt_path"] = str(receipt_path)
    payload["summary"] = format_doctor_boot_summary(payload)
    return payload


def run_lab_ensure(root_dir: Path, *, rail: str) -> Dict[str, Any]:
    root = Path(root_dir)
    rail_n = str(rail or "lab").strip().lower()
    if rail_n == "lab":
        apply = run_lab_bootstrap(root, rail=rail_n)
    else:
        apply = run_watchdog_tick(root, rail=rail_n, port=5010, ensure_worker=False)
    verify = run_doctor_boot(root, rail=rail_n)
    ok = bool(apply.get("ok")) and bool(verify.get("ok"))
    return {
        "schema": "ajax.lab.ensure.v1",
        "ts_utc": _utc_now(),
        "rail": rail_n,
        "apply": apply,
        "verify": {
            "ok": bool(verify.get("ok")),
            "status": verify.get("status"),
            "artifact_path": verify.get("artifact_path"),
            "receipt_path": verify.get("receipt_path"),
        },
        "ok": ok,
        "next_hint": _next_hints(rail_n),
    }


def unsupported_platform_payload(*, command: str) -> Dict[str, Any]:
    return {
        "schema": "ajax.ops.unsupported_platform.v1",
        "ok": False,
        "status": "UNSUPPORTED_PLATFORM",
        "command": command,
        "hint": "Run this command on Windows (PowerShell + Task Scheduler).",
    }
