from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from agency.driver_keys import load_ajax_driver_api_key
from agency.lab_control import LabStateStore
from agency.process_utils import pid_running

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


DEFAULT_HEARTBEAT_STALE_SEC = 30.0
DEFAULT_CANARY_TIMEOUT_S = 60.0
DEFAULT_CANARY_POLL_S = 1.0
DEFAULT_FIX_TIMEOUT_S = 30.0

ProgressFn = Optional[Callable[[Dict[str, Any]], None]]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _lab_ui_enabled() -> bool:
    raw = (os.getenv("AJAX_LAB_UI") or "").strip().lower()
    if not raw:
        return True
    return raw not in {"0", "false", "no", "off"}


def _resolve_lab_driver_url() -> str:
    env_url = (os.getenv("OS_DRIVER_URL_LAB") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    env_host = (os.getenv("OS_DRIVER_HOST_LAB") or "").strip()
    env_port = (os.getenv("OS_DRIVER_PORT_LAB") or "").strip() or "5012"
    if env_host:
        return f"http://{env_host}:{env_port}"
    return f"http://127.0.0.1:{env_port}"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


def _latest_lab_org_receipt(root_dir: Path) -> Optional[Dict[str, Any]]:
    base = root_dir / "artifacts" / "lab_org"
    if not base.exists():
        return None
    try:
        candidates = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    except Exception:
        return None
    for folder in candidates:
        receipt = _read_json(folder / "receipt.json")
        if receipt:
            receipt.setdefault("out_dir", str(folder))
            return receipt
    return None


def _display_probe_receipt(root_dir: Path) -> Optional[Dict[str, Any]]:
    base = root_dir / "artifacts" / "ops" / "display_probe"
    if not base.exists():
        return None
    candidates = sorted(
        [p for p in base.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for candidate in candidates:
        receipt = _read_json(candidate / "receipt.json")
        if receipt:
            receipt.setdefault("out_dir", str(candidate))
            return receipt
    return None


def _run_display_probe(root_dir: Path) -> Dict[str, Any]:
    script = root_dir / "scripts" / "ops" / "lab_display_probe.ps1"
    if not script.exists():
        return {
            "ok": False,
            "display_dummy_ok": False,
            "lab_zone_ok": False,
            "dummy_id": None,
            "error": "missing_lab_display_probe",
        }
    ps = _powershell_path()
    cmd = [
        ps,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        _to_windows_path(script),
    ]
    proc = _run(cmd, cwd=root_dir, timeout=int(DEFAULT_FIX_TIMEOUT_S))
    receipt = _display_probe_receipt(root_dir)
    if receipt is None:
        return {
            "ok": False,
            "display_dummy_ok": False,
            "lab_zone_ok": False,
            "dummy_id": None,
            "error": "display_probe_no_receipt",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    receipt.setdefault("probe_returncode", proc.returncode)
    if proc.stdout:
        receipt.setdefault("probe_stdout", proc.stdout)
    if proc.stderr:
        receipt.setdefault("probe_stderr", proc.stderr)
    return receipt


def _normalize_repo_path(value: str) -> str:
    raw = (value or "").strip().rstrip("\\/")
    if not raw:
        return ""
    if len(raw) >= 3 and raw[1] == ":":
        drive = raw[0].lower()
        tail = raw[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{tail}".lower()
    return raw.replace("\\", "/").lower()


def _lab_worker_paths(root: Path) -> tuple[Path, Path, Path]:
    lab_dir = root / "artifacts" / "lab"
    return (
        lab_dir / "heartbeat.json",
        lab_dir / "worker.pid",
        lab_dir / "worker_info.json",
    )


def _lab_worker_status(root: Path) -> Dict[str, Any]:
    heartbeat_path, pid_path, info_path = _lab_worker_paths(root)
    heartbeat = _read_json(heartbeat_path) if heartbeat_path.exists() else None
    info = _read_json(info_path) if info_path.exists() else None
    pid = None
    running = False
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            running = pid_running(pid)
        except Exception:
            running = False
    hb_ts = None
    hb_age = None
    if isinstance(heartbeat, dict):
        try:
            hb_ts = float(heartbeat.get("ts"))
        except Exception:
            hb_ts = None
    if hb_ts:
        hb_age = max(0.0, time.time() - hb_ts)
    try:
        stale_sec = float(os.getenv("AJAX_LAB_WORKER_STALE_SEC", str(DEFAULT_HEARTBEAT_STALE_SEC)) or DEFAULT_HEARTBEAT_STALE_SEC)
    except Exception:
        stale_sec = DEFAULT_HEARTBEAT_STALE_SEC
    status = "unknown"
    if isinstance(heartbeat, dict) and heartbeat.get("status"):
        status = str(heartbeat.get("status"))
    derived = status
    if not running:
        derived = "down"
    elif hb_age is not None and hb_age > stale_sec:
        derived = "stale"
    return {
        "status": status,
        "derived_status": derived,
        "heartbeat_age_s": hb_age,
        "last_heartbeat_ts": hb_ts,
        "pid": pid,
        "running": running,
        "heartbeat_path": str(heartbeat_path),
        "pid_path": str(pid_path),
        "worker_info_path": str(info_path),
        "worker_id": heartbeat.get("worker_id") if isinstance(heartbeat, dict) else None,
        "version": heartbeat.get("version") if isinstance(heartbeat, dict) else None,
        "capabilities": heartbeat.get("capabilities") if isinstance(heartbeat, dict) else None,
        "driver_url": heartbeat.get("driver_url") if isinstance(heartbeat, dict) else None,
        "info": info if isinstance(info, dict) else None,
        "stale_threshold_s": stale_sec,
    }


def _request_json(
    session: Any,
    *,
    url: str,
    headers: Dict[str, str],
    timeout_s: float,
) -> tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
    if requests is None:
        return False, None, {"error": "requests_unavailable"}
    try:
        resp = session.get(url, headers=headers, timeout=timeout_s)
    except Exception as exc:
        return False, None, {"error": "request_failed", "detail": str(exc)}
    if resp.status_code >= 400:
        detail = (resp.text or "")[:200]
        payload: Optional[Dict[str, Any]] = None
        try:
            parsed = resp.json()
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
        err_code = None
        if isinstance(payload, dict):
            err_code = payload.get("error") or payload.get("detail")
        return False, None, {"error": err_code or f"http_{resp.status_code}", "detail": detail, "http_status": resp.status_code}
    try:
        payload = resp.json()
    except Exception:
        return False, None, {"error": "response_not_json", "detail": (resp.text or "")[:200], "http_status": resp.status_code}
    if not isinstance(payload, dict):
        return False, None, {"error": "response_not_object"}
    return True, payload, {}


def _powershell_path() -> str:
    unix_path = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if unix_path.exists():
        return str(unix_path)
    if os.name == "nt":
        return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    return "powershell.exe"


def _schtasks_path() -> Optional[str]:
    win_path = Path("/mnt/c/Windows/System32/schtasks.exe")
    if win_path.exists():
        return str(win_path)
    if os.name == "nt":
        return "schtasks.exe"
    return None


def _run_schtasks(task_name: str) -> Dict[str, Any]:
    schtasks = _schtasks_path()
    if not schtasks:
        return {"ok": False, "error": "schtasks_unavailable"}
    proc = _run([schtasks, "/Run", "/TN", task_name])
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "task_name": task_name,
    }

def _to_windows_path(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            val = (proc.stdout or "").strip()
            if val:
                return val
    except Exception:
        pass
    return str(path)


def _run(cmd: list[str], *, cwd: Optional[Path] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return subprocess.CompletedProcess(cmd, 124, stdout=stdout, stderr=stderr)


def _driver_check(root_dir: Path, *, driver_url: Optional[str] = None, timeout_s: float = 5.0) -> Dict[str, Any]:
    base_url = (driver_url or _resolve_lab_driver_url()).rstrip("/")
    api_key = load_ajax_driver_api_key()
    headers = {"X-AJAX-KEY": api_key} if api_key else {}
    sess = requests.Session() if requests is not None else None
    health_ok, health_payload, health_err = _request_json(
        sess,
        url=f"{base_url}/health",
        headers=headers,
        timeout_s=timeout_s,
    )
    cap_ok, cap_payload, cap_err = _request_json(
        sess,
        url=f"{base_url}/capabilities",
        headers=headers,
        timeout_s=timeout_s,
    )
    screenshot_caps = {}
    if isinstance(cap_payload, dict):
        screenshot_caps = cap_payload.get("screenshot") if isinstance(cap_payload.get("screenshot"), dict) else {}

    pyautogui_ok = bool(cap_payload.get("pyautogui_available")) if isinstance(cap_payload, dict) else False
    pywinauto_ok = bool(cap_payload.get("pywinauto_available")) if isinstance(cap_payload, dict) else False

    repo_root = None
    driver_file = None
    build_id = None
    python_exe = None
    venv_path = None
    cap_user = None
    backends = None
    backends_available = None
    repo_match = None
    if isinstance(cap_payload, dict):
        repo_root = cap_payload.get("repo_root") if isinstance(cap_payload.get("repo_root"), str) else None
        driver_file = cap_payload.get("driver_file") if isinstance(cap_payload.get("driver_file"), str) else None
        build_id = cap_payload.get("build_id") if isinstance(cap_payload.get("build_id"), str) else None
        python_exe = cap_payload.get("python_exe") if isinstance(cap_payload.get("python_exe"), str) else None
        venv_path = cap_payload.get("venv_path") if isinstance(cap_payload.get("venv_path"), str) else None
        cap_user = cap_payload.get("user") if isinstance(cap_payload.get("user"), str) else None
        backends = cap_payload.get("backends") if isinstance(cap_payload.get("backends"), dict) else None
        backends_available = cap_payload.get("backends_available") if isinstance(cap_payload.get("backends_available"), list) else None
        if repo_root:
            repo_match = _normalize_repo_path(repo_root) == _normalize_repo_path(str(root_dir))

    missing_packages = []
    if cap_ok and isinstance(screenshot_caps, dict):
        if screenshot_caps.get("mss_available") is False:
            missing_packages.append("mss")
        if screenshot_caps.get("pillow_available") is False:
            missing_packages.append("pillow")
        if not pyautogui_ok:
            missing_packages.append("pyautogui")
        if not pywinauto_ok:
            missing_packages.append("pywinauto")

    backend_preferred = screenshot_caps.get("backend_preferred") if isinstance(screenshot_caps, dict) else None
    interactive_desktop = screenshot_caps.get("interactive_desktop") if isinstance(screenshot_caps, dict) else None
    interactive_reason = screenshot_caps.get("interactive_reason") if isinstance(screenshot_caps, dict) else None

    can_snap = False
    if isinstance(backend_preferred, str) and backend_preferred and backend_preferred != "none":
        if interactive_desktop is False:
            can_snap = False
        else:
            can_snap = True

    can_ui = bool(pyautogui_ok and pywinauto_ok)

    return {
        "url": base_url,
        "health_ok": bool(health_ok),
        "health": health_payload if health_ok else None,
        "health_error": health_err if not health_ok else None,
        "capabilities_ok": bool(cap_ok),
        "capabilities": cap_payload if cap_ok else None,
        "capabilities_error": cap_err if not cap_ok else None,
        "repo_root": repo_root,
        "driver_file": driver_file,
        "repo_match": repo_match,
        "build_id": build_id,
        "python_exe": python_exe,
        "venv_path": venv_path,
        "driver_user": cap_user,
        "backends": backends,
        "backends_available": backends_available,
        "screenshot_caps": screenshot_caps,
        "pyautogui_available": pyautogui_ok if cap_ok else None,
        "pywinauto_available": pywinauto_ok if cap_ok else None,
        "backend_preferred": backend_preferred,
        "interactive_desktop": interactive_desktop if cap_ok else None,
        "interactive_reason": interactive_reason if cap_ok else None,
        "can_snap": can_snap if cap_ok else None,
        "can_ui": can_ui if cap_ok else None,
        "missing_packages": sorted(set(missing_packages)),
    }


def _worker_check(root_dir: Path) -> Dict[str, Any]:
    status = _lab_worker_status(root_dir)
    heartbeat_age = status.get("heartbeat_age_s")
    stale_threshold = status.get("stale_threshold_s") or DEFAULT_HEARTBEAT_STALE_SEC
    running = bool(status.get("running"))
    heartbeat_fresh = bool(
        running
        and isinstance(heartbeat_age, (int, float))
        and heartbeat_age <= float(stale_threshold)
    )

    store = LabStateStore(root_dir)
    now = time.time()
    running_jobs = store.list_jobs(statuses={"RUNNING"})
    stale_jobs = []
    for item in running_jobs:
        job = item.get("job") or {}
        info = store.compute_staleness(job, now_ts=now)
        if info.get("is_stale"):
            stale_jobs.append(
                {
                    "job_id": job.get("job_id"),
                    "staleness": info,
                }
            )

    queue_ok = len(stale_jobs) == 0

    return {
        "worker": status,
        "heartbeat_fresh": heartbeat_fresh,
        "queue_ok": queue_ok,
        "stale_jobs": stale_jobs,
        "running_jobs_count": len(running_jobs),
    }


def _extract_deps_from_probe(meta_path: Path) -> set[str]:
    mapping = {
        "mss_not_available": "mss",
        "pillow_not_available": "pillow",
        "pyautogui_not_available": "pyautogui",
        "pywinauto_not_available": "pywinauto",
    }
    found: set[str] = set()
    payload = _read_json(meta_path)
    if not isinstance(payload, dict):
        return found
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    for step in steps:
        if not isinstance(step, dict):
            continue
        detail = step.get("detail")
        if not isinstance(detail, dict):
            continue
        err = detail.get("error")
        if isinstance(err, str) and err in mapping:
            found.add(mapping[err])
    return found


def _extract_deps_from_summary(summary: Optional[str]) -> set[str]:
    if not isinstance(summary, str):
        return set()
    tokens = {
        "mss_not_available": "mss",
        "pillow_not_available": "pillow",
        "pyautogui_not_available": "pyautogui",
        "pywinauto_not_available": "pywinauto",
    }
    found = set()
    for key, val in tokens.items():
        if key in summary:
            found.add(val)
    return found


def _run_lab_job(
    store: LabStateStore,
    payload: Dict[str, Any],
    *,
    timeout_s: float,
    poll_s: float,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    record = store.enqueue_job(payload)
    job_id = record.get("job_id") if isinstance(record, dict) else None
    job_path = record.get("job_path") if isinstance(record, dict) else None
    if progress:
        progress(
            {
                "event": "lab_job_enqueued",
                "message": f"LAB canary enqueued ({payload.get('job_kind')}): {job_id}",
                "job_id": job_id,
                "job_kind": payload.get("job_kind"),
                "objective": payload.get("objective"),
                "job_path": job_path,
            }
        )
    start = time.time()
    result_payload = None
    result_path = None
    while time.time() - start < timeout_s:
        if not job_id:
            break
        found = store.find_result_for_job(str(job_id))
        if found:
            result_payload, result_path = found
            break
        time.sleep(max(0.1, poll_s))

    job_snapshot = None
    if job_id:
        try:
            job_snapshot, _ = store.load_job(str(job_id))
        except Exception:
            job_snapshot = None

    ok = False
    if isinstance(result_payload, dict):
        ok = bool(result_payload.get("efe_pass")) and str(result_payload.get("outcome")) == "PASS"

    if progress:
        progress(
            {
                "event": "lab_job_result",
                "message": f"LAB canary result ({payload.get('job_kind')}): {'PASS' if ok else 'FAIL'}",
                "job_id": job_id,
                "job_kind": payload.get("job_kind"),
                "ok": ok,
                "timed_out": result_payload is None,
                "result_path": str(result_path) if result_path else None,
            }
        )

    return {
        "job_id": job_id,
        "job_path": job_path,
        "job": job_snapshot,
        "result": result_payload,
        "result_path": str(result_path) if result_path else None,
        "ok": ok,
        "timed_out": result_payload is None,
        "wait_s": time.time() - start,
    }


def _canary_check(
    root_dir: Path,
    *,
    driver_ok: bool,
    worker_ok: bool,
    include_ui: bool,
    timeout_s: float,
    poll_s: float,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    if not driver_ok:
        return {"skipped": True, "skip_reason": "driver_not_ready", "ok": False}
    if not worker_ok:
        return {"skipped": True, "skip_reason": "worker_not_ready", "ok": False}

    store = LabStateStore(root_dir)
    ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    mission_id = "doctor_lab"

    snap_payload = {
        "mission_id": mission_id,
        "status": "QUEUED",
        "objective": f"lab_canary_snap_{ts_label}",
        "planned_steps": [],
        "evidence_expected": [],
        "output_paths": [],
        "priority": 90,
        "priority_reason": "doctor_canary",
        "job_kind": "snap_lab",
    }
    if progress:
        progress({"event": "canary_start", "message": "LAB canary: snap_lab"})
    snap = _run_lab_job(store, snap_payload, timeout_s=timeout_s, poll_s=poll_s, progress=progress)

    probe = {
        "skipped": True,
        "skip_reason": "ui_disabled",
        "ok": True,
        "outcome": "SKIPPED",
        "summary": "LAB UI probe disabled.",
    }
    if include_ui:
        probe_payload = {
            "mission_id": mission_id,
            "status": "QUEUED",
            "objective": f"lab_canary_probe_{ts_label}",
            "planned_steps": [],
            "evidence_expected": [],
            "output_paths": [],
            "priority": 90,
            "priority_reason": "doctor_canary",
            "job_kind": "probe_ui",
            "params": {"message": f"LAB canary {ts_label}"},
        }
        if progress:
            progress({"event": "canary_start", "message": "LAB canary: probe_ui"})
        probe = _run_lab_job(store, probe_payload, timeout_s=timeout_s, poll_s=poll_s, progress=progress)

    ok = bool(snap.get("ok") and (probe.get("ok") if include_ui else True))

    return {
        "skipped": False,
        "ok": ok,
        "snap_lab": snap,
        "probe_ui": probe,
        "probe_notepad": probe,
        "ui_skipped": not include_ui,
    }


def _diagnose(
    *,
    driver: Dict[str, Any],
    worker: Dict[str, Any],
    canary: Dict[str, Any],
    display: Dict[str, Any],
) -> list[str]:
    diagnosis = []
    if not driver.get("health_ok"):
        diagnosis.append("driver_down")
    if driver.get("health_ok") and not driver.get("capabilities_ok"):
        diagnosis.append("driver_old_or_no_capabilities")
    if driver.get("capabilities_ok"):
        missing = driver.get("missing_packages") or []
        if missing:
            diagnosis.append("deps_missing")
        if driver.get("interactive_desktop") is False:
            diagnosis.append("session_not_interactive")
        repo_root = driver.get("repo_root")
        repo_match = driver.get("repo_match")
        if not repo_root:
            diagnosis.append("driver_repo_unknown")
        elif repo_match is False:
            diagnosis.append("driver_repo_mismatch")
    if not worker.get("heartbeat_fresh") or not worker.get("queue_ok"):
        diagnosis.append("worker_down_or_stale")
    monitors = display.get("monitors") if isinstance(display, dict) else None
    if isinstance(monitors, list) and len(monitors) < 2:
        diagnosis.append("display_missing")
    else:
        if display.get("display_dummy_ok") is False:
            diagnosis.append("display_dummy_mismatch")
        if display.get("lab_zone_ok") is False:
            diagnosis.append("lab_zone_not_ready")
    if not canary.get("skipped") and not canary.get("ok"):
        diagnosis.append("canary_failed")
        snap = canary.get("snap_lab") if isinstance(canary.get("snap_lab"), dict) else {}
        probe = canary.get("probe_ui") if isinstance(canary.get("probe_ui"), dict) else {}
        if not probe:
            probe = canary.get("probe_notepad") if isinstance(canary.get("probe_notepad"), dict) else {}
        for ent in (snap, probe):
            try:
                result = ent.get("result") if isinstance(ent.get("result"), dict) else None
                failure_codes = result.get("failure_codes") if isinstance(result, dict) else None
                if isinstance(failure_codes, list) and "lab_scope_shared" in failure_codes:
                    if "lab_scope_shared" not in diagnosis:
                        diagnosis.append("lab_scope_shared")
            except Exception:
                continue
        summary = None
        if isinstance(snap.get("result"), dict):
            summary = snap["result"].get("summary")
        if isinstance(probe.get("result"), dict):
            summary = f"{summary or ''} {probe['result'].get('summary') or ''}".strip()
        extra_deps = set(driver.get("missing_packages") or [])
        extra_deps |= _extract_deps_from_summary(summary)
        probe_result = probe.get("result") if isinstance(probe.get("result"), dict) else None
        if isinstance(probe_result, dict):
            for ref in probe_result.get("evidence_refs") or []:
                try:
                    extra_deps |= _extract_deps_from_probe(Path(str(ref)))
                except Exception:
                    continue
        if extra_deps and "deps_missing" not in diagnosis:
            diagnosis.append("deps_missing")
    return diagnosis


def _invariants(
    driver: Dict[str, Any],
    worker: Dict[str, Any],
    canary: Dict[str, Any],
    display: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "driver_health_ok": bool(driver.get("health_ok")),
        "driver_capabilities_ok": bool(driver.get("capabilities_ok")),
        "driver_repo_match": bool(driver.get("repo_match")) if driver.get("capabilities_ok") else False,
        "driver_can_snap": bool(driver.get("can_snap")) if driver.get("capabilities_ok") else False,
        "driver_can_ui": bool(driver.get("can_ui")) if driver.get("capabilities_ok") else False,
        "worker_heartbeat_fresh": bool(worker.get("heartbeat_fresh")),
        "worker_queue_ok": bool(worker.get("queue_ok")),
        "canary_snap_pass": bool((canary.get("snap_lab") or {}).get("ok")) if not canary.get("skipped") else False,
        "canary_probe_pass": bool((canary.get("probe_ui") or canary.get("probe_notepad") or {}).get("ok")) if not canary.get("skipped") else False,
        "display_dummy_ok": bool(display.get("display_dummy_ok")),
        "dummy_display_id": display.get("dummy_id") or display.get("dummy_display_id"),
        "lab_zone_ok": bool(display.get("lab_zone_ok")),
    }


def _compute_ready(inv: Dict[str, Any]) -> bool:
    required = [
        inv.get("driver_health_ok"),
        inv.get("driver_capabilities_ok"),
        inv.get("driver_repo_match"),
        inv.get("driver_can_snap"),
        inv.get("driver_can_ui"),
        inv.get("worker_heartbeat_fresh"),
        inv.get("worker_queue_ok"),
        inv.get("canary_snap_pass"),
        inv.get("canary_probe_pass"),
        inv.get("display_dummy_ok"),
        inv.get("lab_zone_ok"),
    ]
    return all(bool(item) for item in required)


def _compute_ready_non_ui(inv: Dict[str, Any]) -> bool:
    required = [
        inv.get("driver_health_ok"),
        inv.get("driver_capabilities_ok"),
        inv.get("driver_repo_match"),
        inv.get("driver_can_snap"),
        inv.get("worker_heartbeat_fresh"),
        inv.get("worker_queue_ok"),
    ]
    return all(bool(item) for item in required)


def _restart_worker(root_dir: Path) -> Dict[str, Any]:
    status_before = _lab_worker_status(root_dir)
    running = bool(status_before.get("running"))
    hb_age = status_before.get("heartbeat_age_s")
    stale_threshold = status_before.get("stale_threshold_s") or DEFAULT_HEARTBEAT_STALE_SEC
    is_stale = hb_age is None or (isinstance(hb_age, (int, float)) and hb_age >= float(stale_threshold))

    action = "restart_worker"
    reason = None
    pid_path = root_dir / "artifacts" / "lab" / "worker.pid"
    log_path = root_dir / "artifacts" / "lab" / "worker.log"
    cmd = [
        sys.executable,
        "-m",
        "agency.lab_worker",
        "--root",
        str(root_dir),
        "--daemon",
        "--pidfile",
        str(pid_path),
        "--log-file",
        str(log_path),
    ]
    if running and not is_stale:
        return {
            "action": "restart_worker",
            "skipped": True,
            "reason": "worker_running",
            "status_before": status_before,
        }
    if not running and is_stale:
        action = "start_worker"
        reason = "worker_down_or_stale"
        cmd = [
            sys.executable,
            "-m",
            "agency.lab_worker",
            "--root",
            str(root_dir),
            "--daemon",
            "--pidfile",
            str(pid_path),
            "--log-file",
            str(log_path),
        ]
    elif not running:
        action = "start_worker"
        reason = "worker_down"
        cmd = [
            sys.executable,
            "-m",
            "agency.lab_worker",
            "--root",
            str(root_dir),
            "--daemon",
            "--pidfile",
            str(pid_path),
            "--log-file",
            str(log_path),
        ]
    elif is_stale:
        reason = "worker_stale"

    task_attempted = False
    task_result: Dict[str, Any] = {}
    task_name = (os.getenv("AJAX_LAB_WORKER_TASK") or r"\AJAX\AJAX_Lab_Worker_5012").strip()
    prefer_task = (os.getenv("AJAX_LAB_WORKER_PREFER_TASK") or "1").strip().lower() not in {"0", "false", "no", "off"}

    if action in {"start_worker", "restart_worker"} and prefer_task and task_name:
        task_attempted = True
        task_result = _run_schtasks(task_name)
        time.sleep(2.0)
        status_after = _lab_worker_status(root_dir)
        if status_after.get("running") and status_after.get("heartbeat_age_s") is not None:
            return {
                "action": "start_worker_task",
                "reason": reason,
                "task_attempted": task_attempted,
                "task_result": task_result,
                "status_before": status_before,
                "status_after": status_after,
            }

    if action in {"start_worker", "restart_worker"}:
        driver_url = _resolve_lab_driver_url()
        cmd.extend(["--driver-url", driver_url])
        worker_id = status_before.get("worker_id")
        if isinstance(worker_id, str) and worker_id.strip():
            cmd.extend(["--worker-id", worker_id.strip()])

    proc = _run(cmd, cwd=root_dir, timeout=int(DEFAULT_FIX_TIMEOUT_S))
    status_after = _lab_worker_status(root_dir)
    return {
        "action": action,
        "reason": reason,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "timed_out": proc.returncode == 124,
        "timeout_s": DEFAULT_FIX_TIMEOUT_S,
        "task_attempted": task_attempted,
        "task_result": task_result,
        "status_before": status_before,
        "status_after": status_after,
    }


def _restart_driver(root_dir: Path) -> Dict[str, Any]:
    ps = _powershell_path()
    script = root_dir / "Start-AjaxDriver.ps1"
    if not script.exists():
        return {"action": "restart_driver", "ok": False, "error": "missing_Start-AjaxDriver.ps1"}
    cmd = [
        ps,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        _to_windows_path(script),
        "-Port",
        "5012",
    ]
    timeout_s = 60
    proc = _run(cmd, cwd=root_dir, timeout=timeout_s)
    return {
        "action": "restart_driver",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "timed_out": proc.returncode == 124,
        "timeout_s": timeout_s,
    }


def _bootstrap_deps(root_dir: Path, packages: list[str]) -> Dict[str, Any]:
    if not packages:
        return {"action": "bootstrap_deps", "ok": True, "note": "no_missing_packages"}
    venv_python = root_dir / ".venv_os_driver" / "Scripts" / "python.exe"
    python_bin = str(venv_python) if venv_python.exists() else "python"
    cmd = [python_bin, "-m", "pip", "install", "--disable-pip-version-check"]
    cmd.extend(sorted(set(packages)))
    timeout_s = 300
    proc = _run(cmd, cwd=root_dir, timeout=timeout_s)
    return {
        "action": "bootstrap_deps",
        "packages": sorted(set(packages)),
        "python": python_bin,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "timed_out": proc.returncode == 124,
        "timeout_s": timeout_s,
    }


def assess_lab(
    root_dir: Path,
    *,
    include_canary: bool = True,
    include_ui: bool = False,
    driver_url: Optional[str] = None,
    timeout_s: float = DEFAULT_CANARY_TIMEOUT_S,
    poll_s: float = DEFAULT_CANARY_POLL_S,
    progress: ProgressFn = None,
) -> Dict[str, Any]:
    driver = _driver_check(root_dir, driver_url=driver_url)
    worker = _worker_check(root_dir)
    display = _run_display_probe(root_dir)
    canary = {"skipped": True, "skip_reason": "disabled", "ok": False}
    if include_canary:
        canary = _canary_check(
            root_dir,
            driver_ok=bool(driver.get("health_ok") and driver.get("capabilities_ok")),
            worker_ok=bool(worker.get("heartbeat_fresh") and worker.get("queue_ok")),
            include_ui=include_ui,
            timeout_s=timeout_s,
            poll_s=poll_s,
            progress=progress,
        )
    inv = _invariants(driver, worker, canary, display)
    ready_ui = _compute_ready(inv)
    ready_non_ui = _compute_ready_non_ui(inv)
    ready = ready_ui if include_ui else ready_non_ui
    diagnosis = _diagnose(driver=driver, worker=worker, canary=canary, display=display)
    return {
        "ready": ready,
        "ready_ui": ready_ui,
        "ready_non_ui": ready_non_ui,
        "status": "READY" if ready else "DEGRADED",
        "diagnosis": diagnosis,
        "driver": driver,
        "worker": worker,
        "display": display,
        "canary": canary,
        "invariants": inv,
    }


def run_lab_selfcheck(
    root_dir: Path,
    *,
    fix: bool = False,
    include_ui: bool = False,
    out_dir: Optional[Path] = None,
    timeout_s: float = DEFAULT_CANARY_TIMEOUT_S,
    poll_s: float = DEFAULT_CANARY_POLL_S,
    progress: bool = False,
) -> Dict[str, Any]:
    ui_env = os.getenv("AJAX_LAB_UI")
    ui_enabled = _lab_ui_enabled()
    effective_include_ui = bool(include_ui and ui_enabled)
    ui_forced_off = bool(include_ui and not ui_enabled)
    if out_dir is None:
        ts_label = time.strftime("%Y%m%d-%H%M%S")
        out_dir = root_dir / "artifacts" / "doctor" / f"lab_{ts_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    status_payload: Dict[str, Any] = {
        "schema": "ajax.doctor.lab_status.v0",
        "ts_utc": _utc_now(),
        "out_dir": str(out_dir),
        "stage": "start",
        "ready": None,
        "diagnosis": None,
        "actions": [],
        "events": [],
        "ui_policy": {
            "env": ui_env,
            "requested": bool(include_ui),
            "enabled": ui_enabled,
            "effective": effective_include_ui,
            "forced_off": ui_forced_off,
        },
    }

    def _write_status() -> None:
        try:
            (out_dir / "status.json").write_text(
                json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _emit(event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts_utc", _utc_now())
        status_payload["events"].append(payload)
        _write_status()
        msg = payload.get("message")
        if progress and isinstance(msg, str) and msg.strip():
            print(msg, flush=True)

    if ui_forced_off:
        _emit({"event": "ui_forced_off", "message": "LAB doctor: UI disabled by AJAX_LAB_UI"})
    _emit({"event": "baseline_start", "message": "LAB doctor: baseline selfcheck"})
    baseline = assess_lab(
        root_dir,
        include_canary=True,
        include_ui=effective_include_ui,
        timeout_s=timeout_s,
        poll_s=poll_s,
        progress=_emit,
    )
    status_payload["stage"] = "baseline_done"
    status_payload["ready"] = bool(baseline.get("ready"))
    status_payload["diagnosis"] = baseline.get("diagnosis")
    _write_status()
    _emit({"event": "baseline_done", "message": "LAB doctor: baseline complete"})
    actions = []
    post = None

    if fix and not baseline.get("ready"):
        diagnosis = baseline.get("diagnosis") or []
        missing = baseline.get("driver", {}).get("missing_packages") or []
        def _record_action(action: Dict[str, Any]) -> None:
            actions.append(action)
            status_payload["actions"] = actions
            _write_status()
            summary = {
                "action": action.get("action"),
                "returncode": action.get("returncode"),
                "timed_out": action.get("timed_out"),
            }
            msg = f"LAB doctor: {summary['action']} done (rc={summary['returncode']})"
            if summary.get("timed_out"):
                msg += " timeout"
            _emit({"event": "fix_action_done", "message": msg, "action": summary})

        if "worker_down_or_stale" in diagnosis:
            _emit({"event": "fix_action", "message": "LAB doctor: ensuring LAB worker"})
            _record_action(_restart_worker(root_dir))
        if "driver_down" in diagnosis or "driver_old_or_no_capabilities" in diagnosis:
            _emit({"event": "fix_action", "message": "LAB doctor: restarting LAB driver (5012)"})
            _record_action(_restart_driver(root_dir))
        if "deps_missing" in diagnosis and missing:
            _emit({"event": "fix_action", "message": f"LAB doctor: bootstrapping deps ({', '.join(missing)})"})
            _record_action(_bootstrap_deps(root_dir, list(missing)))
            _emit({"event": "fix_action", "message": "LAB doctor: restarting LAB driver after deps"})
            _record_action(_restart_driver(root_dir))
        if "session_not_interactive" in diagnosis:
            actions.append({"action": "needs_human", "ok": False, "reason": "session_not_interactive"})
        status_payload["actions"] = actions
        status_payload["stage"] = "fix_actions_done"
        _write_status()
        _emit({"event": "post_start", "message": "LAB doctor: post-fix selfcheck"})
        post = assess_lab(
            root_dir,
            include_canary=True,
            include_ui=effective_include_ui,
            timeout_s=timeout_s,
            poll_s=poll_s,
            progress=_emit,
        )
        status_payload["stage"] = "post_done"
        status_payload["ready"] = bool(post.get("ready"))
        status_payload["diagnosis"] = post.get("diagnosis")
        _write_status()
        _emit({"event": "post_done", "message": "LAB doctor: post-fix check complete"})

    final = post or baseline
    final_status = "READY" if final.get("ready") else ("FAILED" if fix else "DEGRADED")
    status_payload["stage"] = "done"
    status_payload["ready"] = bool(final.get("ready"))
    status_payload["diagnosis"] = final.get("diagnosis")
    status_payload["actions"] = actions
    _write_status()
    _emit({"event": "done", "message": f"LAB doctor: done ({final_status})"})

    receipt = {
        "schema": "ajax.doctor.lab.v0",
        "ts_utc": _utc_now(),
        "out_dir": str(out_dir),
        "status": final_status,
        "ready": bool(final.get("ready")),
        "diagnosis": final.get("diagnosis"),
        "invariants": final.get("invariants"),
        "driver": final.get("driver"),
        "worker": final.get("worker"),
        "display": final.get("display"),
        "canary": final.get("canary"),
        "lab_org": {
            "status": (LabStateStore(root_dir).state.get("lab_org") or {}).get("status"),
            "last_receipt": _latest_lab_org_receipt(root_dir),
        },
        "ui_policy": status_payload.get("ui_policy"),
        "actions": actions,
        "baseline": baseline if fix else None,
        "post": post,
    }

    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt
