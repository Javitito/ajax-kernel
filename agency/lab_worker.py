from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agency.explore_policy import (
    compute_human_active,
    dummy_display_ok,
    load_explore_policy,
    read_human_signal,
    unknown_signal_as_human_policy,
)
from agency.lab_control import LabStateStore
from agency.process_utils import pid_running
from agency.human_permission import read_human_permission_status
from agency.provider_ledger import ProviderLedger

try:
    from agency.lab_snap import capture_lab_snapshot
except Exception:
    capture_lab_snapshot = None  # type: ignore

try:
    from agency.windows_driver_client import (
        WindowsDriverClient,
        WindowsDriverError,
        DriverConnectionError,
        DriverTimeout,
    )
except Exception:
    WindowsDriverClient = None  # type: ignore
    WindowsDriverError = RuntimeError  # type: ignore
    DriverConnectionError = RuntimeError  # type: ignore
    DriverTimeout = RuntimeError  # type: ignore


WORKER_VERSION = "lab_worker_v0"
LAB_ACTION_EXECUTOR_REGISTRY_SCHEMA = "ajax.lab_action_executor_registry.v1"
LAB_ACTION_EXECUTOR_REGISTRY_REL = Path("config") / "lab_action_executor_registry.json"
KNOWN_JOB_KINDS = {
    "snap_lab",
    "snap_lab_silent",
    "capabilities_refresh",
    "providers_probe",
    "providers_audit",
    "probe_ui",
    "probe_notepad",
    "probe_notepad_dummy",
    "lab_notepad_smoke",
}
BACKGROUND_SAFE_KINDS: set[str] = set()

class LabJobCancelled(RuntimeError):
    pass


def _episode_tags(job: Dict[str, Any], *, kind: str) -> list[str]:
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    ui_intrusive = bool(params.get("ui_intrusive")) if "ui_intrusive" in params else False
    display_target = str(params.get("display_target") or "").strip().lower()
    tags: list[str] = []
    if ui_intrusive:
        tags.append("ui_intrusive")
    if display_target:
        tags.append(f"display:{display_target}")
        if display_target == "dummy":
            tags.append("dummy")
    if kind in {"providers_probe", "providers_audit"}:
        tags.append("providers")
    if kind == "providers_audit":
        tags.extend(["audit", "read_only"])
    tags.append("safety")
    return tags


def _episode_hypothesis(kind: str) -> str:
    if kind == "snap_lab_silent":
        return "Expected to capture a LAB snapshot without focusing windows (silent)."
    if kind == "capabilities_refresh":
        return "Expected driver /health and /capabilities to respond and be persisted as evidence."
    if kind == "providers_probe":
        return "Expected provider ledger refresh to succeed and produce a receipt path."
    if kind == "providers_audit":
        return "Expected read-only providers audit to emit artifacts and a safe remediation plan (without touching credentials)."
    if kind == "probe_notepad_dummy":
        return "Expected a dummy-display UI rehearsal to run only in AWAY and produce evidence."
    return "Expected LAB job to execute per job_kind contract."


def _read_job_display_target(job: Dict[str, Any]) -> str:
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    if not params and isinstance(job.get("args"), dict):
        params = job.get("args") or {}
    return str(params.get("display_target") or "").strip().lower()


def _compute_explore_state(root_dir: Path) -> Dict[str, Any]:
    cfg = load_explore_policy(root_dir)
    pol = cfg.get("policy") if isinstance(cfg.get("policy"), dict) else {}
    try:
        threshold_s = float(pol.get("human_active_threshold_s") or 90)
    except Exception:
        threshold_s = 90.0
    unknown_as_human = unknown_signal_as_human_policy(cfg, root_dir=root_dir)
    signal = read_human_signal(root_dir, policy=cfg)
    active, reason = compute_human_active(signal, threshold_s=threshold_s, unknown_as_human=unknown_as_human)
    return {
        "state": "HUMAN_DETECTED" if active else "AWAY",
        "human_active": bool(active),
        "human_active_reason": reason,
    }


def _update_json_file(path: Path, patch: Dict[str, Any]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
    except Exception:
        return
    data.update(patch)
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


def _maybe_ingest_episode(root_dir: Path, episode_path: Path) -> Dict[str, Any]:
    receipt = {"leann_ingest_attempted": False, "ok": False, "doc_id": None, "error": None}
    try:
        cmd = [sys.executable, str((Path(root_dir) / "bin" / "leann_ingest_episode.py").resolve()), str(episode_path)]
        receipt["leann_ingest_attempted"] = True
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            receipt["error"] = (proc.stderr or out or f"rc_{proc.returncode}")[:200]
            return receipt
        try:
            payload = json.loads(out) if out else {}
        except Exception:
            receipt["error"] = f"ingest_stdout_not_json:{out[:160]}"
            return receipt
        receipt["ok"] = bool(payload.get("ok"))
        receipt["doc_id"] = payload.get("doc_id")
        if payload.get("error"):
            receipt["error"] = str(payload.get("error"))[:200]
        return receipt
    except Exception as exc:
        receipt["leann_ingest_attempted"] = True
        receipt["error"] = str(exc)[:200]
        return receipt


def _read_human_active_flag(root_dir: Path) -> Dict[str, Any]:
    paths = [
        root_dir / "state" / "human_active.flag",
        root_dir / "artifacts" / "state" / "human_active.flag",
        root_dir / "artifacts" / "policy" / "human_active.flag",
    ]
    for path in paths:
        if not path.exists():
            continue
        raw = ""
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            raw = ""
        if not raw:
            return {"active": True, "path": str(path), "value": raw}
        lowered = raw.lower()
        if lowered in {"true", "1", "yes", "on"} or "true" in lowered:
            return {"active": True, "path": str(path), "value": raw}
    return {"active": False, "path": None, "value": None}


def _read_human_active_signal(root_dir: Path) -> Dict[str, Any]:
    """
    Deterministic human detection:
      human_active = (last_input_age_sec < T) AND session_unlocked
    Reads T from config/explore_policy.yaml (policy.human_active_threshold_s).
    """
    cfg = load_explore_policy(root_dir)
    pol = cfg.get("policy") if isinstance(cfg.get("policy"), dict) else {}
    try:
        threshold_s = float(pol.get("human_active_threshold_s") or 90)
    except Exception:
        threshold_s = 90.0
    unknown_as_human = unknown_signal_as_human_policy(cfg, root_dir=root_dir)
    signal = read_human_signal(root_dir, policy=cfg)
    active, reason = compute_human_active(signal, threshold_s=threshold_s, unknown_as_human=unknown_as_human)
    return {
        "active": bool(active),
        "reason": reason,
        "threshold_s": threshold_s,
        "signal": signal,
    }


def _latest_display_probe_receipt(root_dir: Path) -> Optional[Dict[str, Any]]:
    base = root_dir / "artifacts" / "ops" / "display_probe"
    if not base.exists():
        return None
    try:
        candidates = [p for p in base.iterdir() if p.is_dir()]
    except Exception:
        return None
    for folder in sorted(candidates, key=lambda p: p.name, reverse=True):
        receipt = folder / "receipt.json"
        if not receipt.exists():
            continue
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _dummy_display_bounds(root_dir: Path) -> Optional[Dict[str, int]]:
    receipt = _latest_display_probe_receipt(root_dir)
    if not isinstance(receipt, dict):
        return None
    if not receipt.get("lab_zone_ok"):
        return None
    dummy_id = receipt.get("dummy_id") or receipt.get("dummy_display_id") or 2
    monitors = receipt.get("monitors")
    if not isinstance(monitors, list):
        return None
    for mon in monitors:
        if not isinstance(mon, dict):
            continue
        if mon.get("id") != dummy_id:
            continue
        bounds = mon.get("bounds")
        if isinstance(bounds, dict):
            try:
                return {
                    "x": int(bounds.get("x", 0)),
                    "y": int(bounds.get("y", 0)),
                    "width": int(bounds.get("width", 0)),
                    "height": int(bounds.get("height", 0)),
                }
            except Exception:
                return None
    return None


def _dummy_display_id(root_dir: Path) -> Optional[int]:
    receipt = _latest_display_probe_receipt(root_dir)
    if not isinstance(receipt, dict):
        return None
    if not receipt.get("lab_zone_ok"):
        return None
    raw = receipt.get("dummy_id") or receipt.get("dummy_display_id") or 2
    try:
        return int(raw)
    except Exception:
        return None


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _lab_ui_enabled() -> bool:
    raw = (os.getenv("AJAX_LAB_UI") or "").strip().lower()
    if not raw:
        return True
    return raw not in {"0", "false", "no", "off"}


def _ui_policy_gate(root_dir: Path, *, job: Dict[str, Any], ui_intrusive: bool) -> Optional[Dict[str, Any]]:
    """
    Canonical UI gating:
      - Default: block ui_intrusive when HUMAN_DETECTED.
      - Allow ui_intrusive only when AWAY AND display_target=dummy.
      - AJAX_LAB_UI may disable UI globally, but MUST NOT block dummy-only UI in AWAY
        unless config/policy.force_disable_ui_everywhere=true.
    """
    if not ui_intrusive:
        return None

    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    if not params and isinstance(job.get("args"), dict):
        params = job.get("args") or {}
    target = str(params.get("display_target") or "").strip().lower()

    cfg = load_explore_policy(root_dir)
    pol = cfg.get("policy") if isinstance(cfg.get("policy"), dict) else {}
    force_off = bool(pol.get("force_disable_ui_everywhere", False))

    signal = read_human_signal(root_dir, policy=cfg)
    try:
        threshold_s = float(pol.get("human_active_threshold_s") or 90)
    except Exception:
        threshold_s = 90.0
    unknown_as_human = unknown_signal_as_human_policy(cfg, root_dir=root_dir)
    active, _reason = compute_human_active(signal, threshold_s=threshold_s, unknown_as_human=unknown_as_human)
    state = "HUMAN_DETECTED" if active else "AWAY"

    if state != "AWAY":
        return {
            "outcome": "BLOCKED",
            "efe_pass": False,
            "failure_codes": ["human_detected", "ui_blocked"],
            "evidence_refs": [],
            "next_action": "wait_for_away",
            "summary": "UI intrusive job blocked while human detected.",
        }
    if target != "dummy":
        return {
            "outcome": "BLOCKED",
            "efe_pass": False,
            "failure_codes": ["ui_primary_display_forbidden"],
            "evidence_refs": [],
            "next_action": "set_display_target_dummy",
            "summary": "UI intrusive job requires display_target=dummy in AWAY.",
        }
    require_dummy = bool(pol.get("require_dummy_display_ok", True))
    if require_dummy and not dummy_display_ok(root_dir):
        return {
            "outcome": "BLOCKED",
            "efe_pass": False,
            "failure_codes": ["dummy_display_required"],
            "evidence_refs": [],
            "next_action": "run_display_probe",
            "summary": "Dummy display not verified; UI intrusive job blocked.",
        }
    if force_off:
        return {
            "outcome": "BLOCKED",
            "efe_pass": False,
            "failure_codes": ["ui_globally_disabled"],
            "evidence_refs": [],
            "next_action": "enable_lab_ui",
            "summary": "LAB UI disabled by config.force_disable_ui_everywhere.",
        }
    return None


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return float(default)
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        if raw is None or raw == "":
            return int(default)
        return int(raw)
    except Exception:
        return int(default)


def _resolve_lab_driver_url() -> str:
    env_url = (os.getenv("OS_DRIVER_URL_LAB") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    env_host = (os.getenv("OS_DRIVER_HOST_LAB") or "").strip()
    env_port = (os.getenv("OS_DRIVER_PORT_LAB") or "").strip() or "5012"
    if env_host:
        return f"http://{env_host}:{env_port}"
    return f"http://127.0.0.1:{env_port}"


def _windows_to_wsl_path(path_str: str) -> str:
    if not path_str:
        return path_str
    normalized = path_str.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    return path_str


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _parse_json_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    raw = (stdout or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _providers_audit_texts(payload: Dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("summary", "recommendation", "next_action", "title", "code"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            texts.append(val.strip().lower())
    return texts


def _providers_audit_has_auth_quota_risk(audit_doc: Dict[str, Any]) -> bool:
    keywords = ("auth", "quota", "credential", "login", "401", "403", "429", "token", "apikey")
    findings = audit_doc.get("findings") if isinstance(audit_doc.get("findings"), list) else []
    for item in findings:
        if not isinstance(item, dict):
            continue
        for text in _providers_audit_texts(item):
            if any(k in text for k in keywords):
                return True
    recs = audit_doc.get("recommended_actions") if isinstance(audit_doc.get("recommended_actions"), list) else []
    for item in recs:
        if not isinstance(item, dict):
            continue
        for text in _providers_audit_texts(item):
            if any(k in text for k in keywords):
                return True
    return False


def _providers_audit_needs_refresh(audit_doc: Dict[str, Any]) -> bool:
    refresh_codes = {
        "policy_provider_missing_in_status",
        "council_quorum_risk",
        "timeouts_missing_p95_base",
    }
    findings = audit_doc.get("findings") if isinstance(audit_doc.get("findings"), list) else []
    for item in findings:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip().lower()
        if code in refresh_codes:
            return True
    return False


def _providers_audit_collect_recommended_commands(audit_doc: Dict[str, Any]) -> list[str]:
    cmds: list[str] = []
    recs = audit_doc.get("recommended_actions") if isinstance(audit_doc.get("recommended_actions"), list) else []
    for item in recs:
        if not isinstance(item, dict):
            continue
        cmd = str(item.get("command") or "").strip()
        if cmd:
            cmds.append(cmd)
    return cmds


def _default_lab_action_executor_registry() -> Dict[str, Any]:
    """Builtin dispatch registry for LAB micro-jobs (default behavior)."""
    return {
        "schema": LAB_ACTION_EXECUTOR_REGISTRY_SCHEMA,
        "job_kinds": {
            "snap_lab": {
                "action_executor": "builtin.snap_lab.v1",
                "handler": "_execute_snap_lab",
            },
            "capabilities_refresh": {
                "action_executor": "builtin.capabilities_refresh.v1",
                "handler": "_execute_capabilities_refresh",
            },
            "providers_probe": {
                "action_executor": "builtin.providers_probe.v1",
                "handler": "_execute_providers_probe",
            },
            "providers_audit": {
                "action_executor": "builtin.providers_audit.v1",
                "handler": "_execute_providers_audit",
            },
            "probe_ui": {
                "action_executor": "builtin.probe_ui.v1",
                "handler": "_execute_probe_ui",
            },
            "probe_notepad": {
                "action_executor": "builtin.probe_notepad.v1",
                "handler": "_execute_probe_notepad",
            },
        },
    }


def _load_lab_action_executor_registry(root_dir: Path) -> Dict[str, Any]:
    """
    Load optional LAB action-executor registry.

    If the config file is absent, returns builtin defaults. If the file exists but is invalid,
    returns builtin defaults plus ``registry_error`` so callers can fail closed (no silent degrade).
    """
    builtin = _default_lab_action_executor_registry()
    cfg_path = Path(root_dir) / LAB_ACTION_EXECUTOR_REGISTRY_REL
    if not cfg_path.exists():
        return {
            **builtin,
            "registry_path": str(cfg_path),
            "registry_source": "builtin_default",
            "registry_error": None,
        }
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            **builtin,
            "registry_path": str(cfg_path),
            "registry_source": "builtin_default",
            "registry_error": f"registry_parse_failed:{str(exc)[:160]}",
        }
    if not isinstance(raw, dict):
        return {
            **builtin,
            "registry_path": str(cfg_path),
            "registry_source": "builtin_default",
            "registry_error": "registry_invalid_root",
        }
    merged = {
        "schema": str(raw.get("schema") or builtin.get("schema") or LAB_ACTION_EXECUTOR_REGISTRY_SCHEMA),
        "job_kinds": dict(builtin.get("job_kinds") or {}),
        "registry_path": str(cfg_path),
        "registry_source": "config",
        "registry_error": None,
    }
    job_kinds = raw.get("job_kinds")
    if isinstance(job_kinds, dict):
        for job_kind_raw, entry_raw in job_kinds.items():
            job_kind = str(job_kind_raw or "").strip().lower()
            if not job_kind:
                continue
            if isinstance(entry_raw, str):
                merged["job_kinds"][job_kind] = {
                    "action_executor": entry_raw.strip() or f"config.{job_kind}",
                    "handler": builtin.get("job_kinds", {}).get(job_kind, {}).get("handler"),
                }
                continue
            if not isinstance(entry_raw, dict):
                continue
            action_executor = str(entry_raw.get("action_executor") or f"config.{job_kind}").strip()
            handler = str(entry_raw.get("handler") or "").strip()
            if not handler and isinstance(builtin.get("job_kinds"), dict):
                default_entry = builtin["job_kinds"].get(job_kind)
                if isinstance(default_entry, dict):
                    handler = str(default_entry.get("handler") or "").strip()
            merged["job_kinds"][job_kind] = {
                "action_executor": action_executor or f"config.{job_kind}",
                "handler": handler,
            }
    return merged


class LabWorker:
    def __init__(
        self,
        root_dir: Path,
        *,
        worker_id: Optional[str] = None,
        idle_sleep_s: Optional[float] = None,
        heartbeat_s: Optional[float] = None,
        stale_sweep_s: Optional[float] = None,
        driver_url: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.store = LabStateStore(self.root_dir)
        self.worker_id = worker_id or f"lab_worker_{os.getpid()}"
        self.idle_sleep_s = idle_sleep_s if idle_sleep_s is not None else _env_float("AJAX_LAB_WORKER_IDLE_SEC", 2.0)
        self.heartbeat_s = heartbeat_s if heartbeat_s is not None else _env_float("AJAX_LAB_WORKER_HEARTBEAT_SEC", 10.0)
        self.stale_sweep_s = stale_sweep_s if stale_sweep_s is not None else _env_int("AJAX_LAB_STALE_SWEEP_SEC", 30)
        self.driver_url = driver_url or _resolve_lab_driver_url()
        self.action_executor_registry = _load_lab_action_executor_registry(self.root_dir)
        self.worker_version = WORKER_VERSION
        self.capabilities = sorted(KNOWN_JOB_KINDS)
        self.lab_dir = self.root_dir / "artifacts" / "lab"
        self.heartbeat_path = self.lab_dir / "heartbeat.json"
        self.worker_info_path = self.lab_dir / "worker_info.json"
        self.evidence_root = self.lab_dir / "evidence"
        self._last_heartbeat_ts = 0.0
        self._last_sweep_ts = 0.0
        self._last_lab_org_tick_ts = 0.0
        self._status = "READY"
        self.active_job_id: Optional[str] = None
        self.last_job_id: Optional[str] = None
        self._write_worker_info()
        self._write_worker_heartbeat(status="READY")

    def _resolve_action_executor(self, job_kind: str) -> Dict[str, Any]:
        kind = str(job_kind or "").strip().lower()
        registry = self.action_executor_registry if isinstance(self.action_executor_registry, dict) else {}
        registry_error = str(registry.get("registry_error") or "").strip()
        if registry_error:
            return {
                "ok": False,
                "error": registry_error,
                "source": str(registry.get("registry_source") or "builtin_default"),
                "registry_path": registry.get("registry_path"),
                "job_kind": kind,
                "action_executor": None,
                "handler": None,
            }
        job_kinds = registry.get("job_kinds") if isinstance(registry.get("job_kinds"), dict) else {}
        entry = job_kinds.get(kind)
        if not isinstance(entry, dict):
            return {
                "ok": False,
                "error": "executor_not_configured",
                "source": str(registry.get("registry_source") or "builtin_default"),
                "registry_path": registry.get("registry_path"),
                "job_kind": kind,
                "action_executor": None,
                "handler": None,
            }
        action_executor = str(entry.get("action_executor") or "").strip()
        handler_name = str(entry.get("handler") or "").strip()
        if not handler_name:
            return {
                "ok": False,
                "error": "executor_handler_missing",
                "source": str(registry.get("registry_source") or "builtin_default"),
                "registry_path": registry.get("registry_path"),
                "job_kind": kind,
                "action_executor": action_executor or None,
                "handler": None,
            }
        if not handler_name.startswith("_execute_"):
            return {
                "ok": False,
                "error": "executor_handler_invalid",
                "source": str(registry.get("registry_source") or "builtin_default"),
                "registry_path": registry.get("registry_path"),
                "job_kind": kind,
                "action_executor": action_executor or None,
                "handler": handler_name,
            }
        handler_fn = getattr(self, handler_name, None)
        if not callable(handler_fn):
            return {
                "ok": False,
                "error": "executor_handler_unavailable",
                "source": str(registry.get("registry_source") or "builtin_default"),
                "registry_path": registry.get("registry_path"),
                "job_kind": kind,
                "action_executor": action_executor or None,
                "handler": handler_name,
            }
        return {
            "ok": True,
            "error": None,
            "source": str(registry.get("registry_source") or "builtin_default"),
            "registry_path": registry.get("registry_path"),
            "job_kind": kind,
            "action_executor": action_executor or f"builtin.{kind}",
            "handler": handler_name,
            "handler_fn": handler_fn,
        }

    def _write_worker_info(self) -> None:
        payload = {
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "version": self.worker_version,
            "driver_url": self.driver_url,
            "capabilities": list(self.capabilities),
            "started_ts": time.time(),
            "started_utc": _utc_now(),
        }
        _write_json(self.worker_info_path, payload)

    def _write_worker_heartbeat(
        self,
        *,
        status: str,
        active_job_id: Optional[str] = None,
        last_job_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        payload = {
            "ts": time.time(),
            "ts_utc": _utc_now(),
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "status": status,
            "active_job_id": active_job_id,
            "last_job_id": last_job_id,
            "version": self.worker_version,
            "driver_url": self.driver_url,
            "capabilities": list(self.capabilities),
        }
        if error:
            payload["error"] = str(error)
        _write_json(self.heartbeat_path, payload)
        self._last_heartbeat_ts = payload["ts"]

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat_ts >= float(self.heartbeat_s):
            self._write_worker_heartbeat(
                status=self._status,
                active_job_id=self.active_job_id,
                last_job_id=self.last_job_id,
            )

    def _write_job(self, job: Dict[str, Any], path: Path) -> None:
        _write_json(path, job)

    def _mark_running(self, job: Dict[str, Any], path: Path) -> None:
        now = time.time()
        job["status"] = "RUNNING"
        job["worker_id"] = self.worker_id
        job["worker_version"] = self.worker_version
        job["worker_heartbeat_ts"] = now
        job["last_heartbeat_ts"] = now
        if job.get("started_ts") is None:
            job["started_ts"] = now
        if job.get("created_ts") is None:
            job["created_ts"] = now
        if job.get("queued_since_ts") is None:
            job["queued_since_ts"] = job.get("created_ts") or now
        self._write_job(job, path)

    def _touch_job_heartbeat(self, job: Dict[str, Any], path: Path) -> None:
        try:
            latest = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(latest, dict) and str(latest.get("status") or "").upper() == "CANCELLED":
                raise LabJobCancelled("job_cancelled")
        except LabJobCancelled:
            raise
        except Exception:
            pass
        now = time.time()
        job["worker_heartbeat_ts"] = now
        job["last_heartbeat_ts"] = now
        self._write_job(job, path)
        self._maybe_heartbeat()

    def _append_output_paths(self, job: Dict[str, Any], path: Path, paths: list[str]) -> None:
        if not paths:
            return
        output_paths = job.get("output_paths")
        if not isinstance(output_paths, list):
            output_paths = []
        changed = False
        for item in paths:
            if item and item not in output_paths:
                output_paths.append(item)
                changed = True
        if changed:
            job["output_paths"] = output_paths
            self._write_job(job, path)

    def _ensure_evidence_dir(self, job_id: str) -> Path:
        out_dir = self.evidence_root / str(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _resolve_job_kind(self, job: Dict[str, Any]) -> str:
        for key in ("job_kind", "lab_action", "action", "kind"):
            val = job.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().lower()
        normalized = ""
        for key in ("objective_norm", "objective"):
            val = job.get(key)
            if isinstance(val, str) and val.strip():
                normalized = val.strip().lower()
        if normalized in KNOWN_JOB_KINDS:
            return normalized
        return ""

    def _human_active_gate(self, job: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
        status = _read_human_active_signal(self.root_dir)
        if not status.get("active"):
            return None
        ui_intrusive = True
        try:
            params = job.get("params") if isinstance(job.get("params"), dict) else {}
            if "ui_intrusive" in params:
                ui_intrusive = bool(params.get("ui_intrusive"))
            else:
                ui_intrusive = kind in {"probe_ui", "probe_notepad", "lab_notepad_smoke"}
        except Exception:
            ui_intrusive = True
        if not ui_intrusive:
            return None
        perm = read_human_permission_status(self.root_dir)
        if perm.get("ok"):
            return None
        evidence_refs = []
        signal = status.get("signal") if isinstance(status.get("signal"), dict) else None
        if isinstance(signal, dict) and signal.get("probe"):
            evidence_refs.append("human_signal:" + str(signal.get("probe")))
        if perm.get("path"):
            evidence_refs.append(str(perm.get("path")))
        summary = (
            "human_detected; lease required for UI input. "
            "Run `ajaxctl permit 120` and requeue the LAB job."
        )
        return {
            "outcome": "BLOCKED",
            "efe_pass": False,
            "failure_codes": ["human_active", "lease_required"],
            "evidence_refs": evidence_refs,
            "next_action": "request_human_lease",
            "summary": summary,
        }

    def _driver_client(self) -> WindowsDriverClient:
        if WindowsDriverClient is None:
            raise RuntimeError("windows_driver_client_unavailable")
        return WindowsDriverClient(base_url=self.driver_url, prefer_env=False)

    def _execute_snap_lab(self, job: Dict[str, Any], path: Path) -> Dict[str, Any]:
        if capture_lab_snapshot is None:
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["lab_snap_unavailable"],
                "evidence_refs": [],
                "next_action": "review_lab_snap",
                "summary": "LAB snap unavailable (missing lab_snap import).",
            }
        job_id = str(job.get("job_id") or "job")
        mission_id = str(job.get("mission_id") or "mission")
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        if not params and isinstance(job.get("args"), dict):
            params = job.get("args") or {}
        display_id = params.get("display_id")
        if display_id is not None:
            try:
                display_id = int(display_id)
            except Exception:
                display_id = None
        if display_id is None and str(params.get("display_target") or "").strip().lower() == "dummy":
            display_id = _dummy_display_id(self.root_dir)
        evidence_dir = self._ensure_evidence_dir(job_id)
        ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self._touch_job_heartbeat(job, path)
        try:
            shot = capture_lab_snapshot(
                root_dir=self.root_dir,
                job_id=job_id,
                mission_id=mission_id,
                active_window=False,
                driver_url=self.driver_url,
                context=f"{job_id}_snap",
                display_id=display_id,
            )
        except Exception as exc:
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["lab_snap_failed"],
                "evidence_refs": [],
                "next_action": "check_lab_driver",
                "summary": f"LAB snap failed: {str(exc)[:200]}",
            }
        warnings = shot.get("warnings") or []
        meta_path = evidence_dir / f"snap_{ts_label}.json"
        meta = {
            "ts": time.time(),
            "ts_utc": _utc_now(),
            "job_id": job_id,
            "mission_id": mission_id,
            "driver_url": shot.get("driver_url") or self.driver_url,
            "session_id": shot.get("session_id"),
            "shot": {
                "png_path": shot.get("png_path"),
                "json_path": shot.get("json_path"),
            },
            "warnings": warnings,
        }
        _write_json(meta_path, meta)
        evidence_refs = [str(meta_path)]
        for ref_key in ("png_path", "json_path"):
            ref_val = shot.get(ref_key)
            if ref_val:
                evidence_refs.append(str(ref_val))
        failure_codes: list[str] = []
        outcome = "PASS"
        efe_pass = True
        if warnings:
            failure_codes.append("lab_snap_warning")
            if any(isinstance(w, dict) and w.get("evidence_kind") == "scope_shared" for w in warnings):
                failure_codes.append("lab_scope_shared")
            outcome = "PARTIAL"
            efe_pass = False
        summary = "LAB snap captured."
        if warnings:
            summary = f"LAB snap captured with {len(warnings)} warnings."
        return {
            "outcome": outcome,
            "efe_pass": efe_pass,
            "failure_codes": failure_codes,
            "evidence_refs": evidence_refs,
            "next_action": "review_lab_snap",
            "summary": summary,
        }

    def _execute_capabilities_refresh(self, job: Dict[str, Any], path: Path) -> Dict[str, Any]:
        job_id = str(job.get("job_id") or "job")
        evidence_dir = self._ensure_evidence_dir(job_id)
        ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = evidence_dir / f"capabilities_refresh_{ts_label}.json"
        try:
            client = self._driver_client()
            health = client.health()
            caps = client.capabilities()
            data = {"schema": "ajax.lab.capabilities_refresh.v1", "health": health, "capabilities": caps}
        except Exception as exc:
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["driver_health_failed"],
                "evidence_refs": [],
                "next_action": "review_lab_driver",
                "summary": str(exc)[:200],
            }
        try:
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        return {
            "outcome": "PASS",
            "efe_pass": True,
            "failure_codes": [],
            "evidence_refs": [str(out_path)],
            "next_action": "noop",
            "summary": "Driver health snapshot captured.",
        }

    def _refresh_providers_status(self) -> Dict[str, Any]:
        status_path = self.root_dir / "artifacts" / "health" / "providers_status.json"
        now_ts = float(time.time())

        def _fallback_touch(*, reason: str, error: Optional[str] = None) -> Dict[str, Any]:
            doc: Dict[str, Any] = {}
            try:
                raw = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    doc = raw
            except Exception:
                doc = {}
            providers = doc.get("providers")
            if not isinstance(providers, dict):
                providers = {}
            doc["schema"] = doc.get("schema") or "ajax.providers_status.v1"
            doc["providers"] = providers
            doc["updated_at"] = now_ts
            doc["updated_ts"] = now_ts
            doc["updated_utc"] = _utc_now()
            meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
            meta["last_refresh_source"] = "lab_worker.providers_probe"
            meta["last_refresh_reason"] = reason
            meta["last_refresh_ok"] = False
            if error:
                meta["last_refresh_error"] = str(error)[:240]
            doc["meta"] = meta
            _write_json(status_path, doc)
            return {
                "ok": False,
                "path": str(status_path),
                "updated_utc": doc.get("updated_utc"),
                "error": reason,
            }

        try:
            from agency.provider_breathing import ProviderBreathingLoop, _load_provider_configs
        except Exception as exc:
            return _fallback_touch(reason="provider_breathing_import_failed", error=str(exc))

        try:
            provider_cfg = _load_provider_configs(self.root_dir)
            loop = ProviderBreathingLoop(root_dir=self.root_dir, provider_configs=provider_cfg)
            status_doc = loop.run_once()
        except Exception as exc:
            return _fallback_touch(reason="provider_breathing_run_failed", error=str(exc))

        if not isinstance(status_doc, dict):
            return _fallback_touch(reason="provider_breathing_invalid_status_doc")

        try:
            updated_at = status_doc.get("updated_at")
            try:
                updated_ts = float(updated_at) if updated_at is not None else float(time.time())
            except Exception:
                updated_ts = float(time.time())
            status_doc["updated_ts"] = updated_ts
            status_doc["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(updated_ts))
            meta = status_doc.get("meta") if isinstance(status_doc.get("meta"), dict) else {}
            meta["last_refresh_source"] = "lab_worker.providers_probe"
            meta["last_refresh_reason"] = "provider_breathing_run_once"
            meta["last_refresh_ok"] = True
            status_doc["meta"] = meta
            _write_json(status_path, status_doc)
        except Exception as exc:
            return _fallback_touch(reason="providers_status_write_failed", error=str(exc))

        return {
            "ok": True,
            "path": str(status_path),
            "updated_utc": status_doc.get("updated_utc"),
            "error": None,
        }

    def _execute_providers_probe(self, job: Dict[str, Any], path: Path) -> Dict[str, Any]:
        status_refresh = self._refresh_providers_status()
        evidence_refs = [str(status_refresh.get("path") or (self.root_dir / "artifacts" / "health" / "providers_status.json"))]
        try:
            ledger = ProviderLedger(root_dir=self.root_dir)
            doc = ledger.refresh(timeout_s=2.0)
            ledger_path = str(doc.get("path") or (self.root_dir / "artifacts" / "provider_ledger" / "latest.json"))
            evidence_refs.append(ledger_path)
            failure_codes = []
            summary = "Provider status + ledger refreshed."
            if not bool(status_refresh.get("ok")):
                failure_codes.append("providers_status_refresh_fallback")
                summary = (
                    "Provider ledger refreshed; providers_status used fallback refresh "
                    f"({status_refresh.get('error')})."
                )
            return {
                "outcome": "PASS",
                "efe_pass": True,
                "failure_codes": failure_codes,
                "evidence_refs": evidence_refs,
                "next_action": "noop",
                "summary": summary,
            }
        except Exception as exc:
            failure_codes = ["providers_probe_failed"]
            if not bool(status_refresh.get("ok")):
                failure_codes.append("providers_status_refresh_fallback")
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": failure_codes,
                "evidence_refs": evidence_refs,
                "next_action": "review_providers",
                "summary": str(exc)[:200],
            }

    def _execute_providers_audit(self, job: Dict[str, Any], path: Path) -> Dict[str, Any]:
        explore = _compute_explore_state(self.root_dir)
        if str(explore.get("state") or "HUMAN_DETECTED") != "AWAY":
            return {
                "outcome": "BLOCKED",
                "efe_pass": False,
                "failure_codes": ["providers_audit_away_only", "human_detected"],
                "evidence_refs": [],
                "next_action": "wait_for_away",
                "summary": "providers_audit is AWAY-only and was blocked by human detection.",
            }

        job_id = str(job.get("job_id") or "job")
        evidence_dir = self._ensure_evidence_dir(job_id)
        ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        plan_path = evidence_dir / f"providers_audit_plan_{ts_label}.json"
        outcome_path = evidence_dir / f"providers_audit_outcome_{ts_label}.json"
        ajaxctl_path = str((self.root_dir / "bin" / "ajaxctl").resolve())
        audit_cmd = [
            sys.executable,
            ajaxctl_path,
            "audit",
            "providers",
            "--root",
            str(self.root_dir),
            "--json",
        ]
        try:
            audit_proc = subprocess.run(
                audit_cmd,
                capture_output=True,
                text=True,
                timeout=60.0,
                check=False,
            )
        except Exception as exc:
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["providers_audit_exec_failed"],
                "evidence_refs": [],
                "next_action": "review_providers_audit_job",
                "summary": str(exc)[:200],
            }

        audit_doc = _parse_json_stdout(audit_proc.stdout or "")
        if audit_proc.returncode not in {0, 2} or not isinstance(audit_doc, dict):
            _write_json(
                outcome_path,
                {
                    "schema": "ajax.lab.providers_audit_outcome.v1",
                    "ts_utc": _utc_now(),
                    "job_id": job_id,
                    "audit_command": audit_cmd,
                    "audit_returncode": int(audit_proc.returncode),
                    "audit_stdout": (audit_proc.stdout or "")[:2000],
                    "audit_stderr": (audit_proc.stderr or "")[:2000],
                    "ok": False,
                    "reason": "audit_failed_or_invalid_json",
                },
            )
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["providers_audit_failed"],
                "evidence_refs": [str(outcome_path)],
                "next_action": "review_providers_audit_job",
                "summary": f"providers audit failed (rc={audit_proc.returncode}) or returned non-JSON output.",
            }

        findings = audit_doc.get("findings") if isinstance(audit_doc.get("findings"), list) else []
        auth_quota_sensitive = _providers_audit_has_auth_quota_risk(audit_doc)
        needs_refresh = _providers_audit_needs_refresh(audit_doc)
        recommended_cmds = _providers_audit_collect_recommended_commands(audit_doc)

        planned_actions: list[Dict[str, Any]] = []
        blocked_actions: list[Dict[str, Any]] = []
        checklist: list[str] = []
        if auth_quota_sensitive:
            blocked_actions.append(
                {
                    "kind": "auth_quota_sensitive",
                    "reason": "auth_or_quota_finding_detected",
                    "safe": False,
                }
            )
            checklist.extend(
                [
                    "Review providers audit findings for auth/quota failures.",
                    "Run `python bin/ajaxctl doctor providers` manually after credentials/quota are verified.",
                    "Do not perform login/credential changes from LAB micro-jobs.",
                ]
            )
        else:
            if needs_refresh:
                planned_actions.append(
                    {
                        "kind": "providers_probe_refresh",
                        "safe": True,
                        "executor": "builtin.providers_probe.v1",
                        "command": "internal:_execute_providers_probe",
                    }
                )
            if any(cmd.strip().startswith("python bin/ajaxctl doctor providers") for cmd in recommended_cmds):
                planned_actions.append(
                    {
                        "kind": "doctor_providers",
                        "safe": True,
                        "executor": "cli",
                        "command": "python bin/ajaxctl doctor providers",
                    }
                )

        expected_state = {
            "audit_invoked": True,
            "audit_returncode_in": [0, 2],
            "audit_artifact_or_receipt_present": True,
            "no_credential_or_login_actions_executed": True,
            "safe_actions_only_from_allowlist": True,
            "plan_and_outcome_artifacts_written": True,
        }
        plan_payload = {
            "schema": "ajax.lab.providers_audit_plan.v1",
            "ts_utc": _utc_now(),
            "job_id": job_id,
            "job_kind": "providers_audit",
            "audit": {
                "command": audit_cmd,
                "returncode": int(audit_proc.returncode),
                "ok_field": bool(audit_doc.get("ok")),
                "artifact_path": audit_doc.get("artifact_path"),
                "artifact_md_path": audit_doc.get("artifact_md_path"),
                "receipt_path": audit_doc.get("receipt_path"),
            },
            "findings_summary": {
                "count": len(findings),
                "summary": audit_doc.get("summary"),
            },
            "auth_quota_sensitive": auth_quota_sensitive,
            "needs_refresh": needs_refresh,
            "recommended_commands": recommended_cmds,
            "planned_actions": planned_actions,
            "blocked_actions": blocked_actions,
            "checklist": checklist,
            "expected_state": expected_state,
        }
        _write_json(plan_path, plan_payload)

        evidence_refs: list[str] = [str(plan_path)]
        for key in ("artifact_path", "artifact_md_path", "receipt_path"):
            val = audit_doc.get(key)
            if isinstance(val, str) and val:
                evidence_refs.append(val)

        executed_actions: list[Dict[str, Any]] = []
        safe_action_failed = False
        for action in planned_actions:
            kind = str(action.get("kind") or "")
            if kind == "providers_probe_refresh":
                probe_result = self._execute_providers_probe(job, path)
                ok = bool(probe_result.get("efe_pass"))
                if not ok:
                    safe_action_failed = True
                for ref in list(probe_result.get("evidence_refs") or []):
                    if isinstance(ref, str) and ref:
                        evidence_refs.append(ref)
                executed_actions.append(
                    {
                        "kind": kind,
                        "safe": True,
                        "ok": ok,
                        "outcome": probe_result.get("outcome"),
                        "failure_codes": list(probe_result.get("failure_codes") or []),
                        "summary": probe_result.get("summary"),
                        "evidence_refs": list(probe_result.get("evidence_refs") or []),
                    }
                )
                continue
            if kind == "doctor_providers":
                doctor_cmd = [sys.executable, ajaxctl_path, "doctor", "providers"]
                try:
                    doctor_proc = subprocess.run(
                        doctor_cmd,
                        capture_output=True,
                        text=True,
                        timeout=45.0,
                        check=False,
                    )
                    doctor_doc = _parse_json_stdout(doctor_proc.stdout or "")
                    doctor_artifact = doctor_doc.get("artifact") if isinstance(doctor_doc, dict) else None
                    doctor_ok = doctor_proc.returncode in {0, 1}
                    if not doctor_ok:
                        safe_action_failed = True
                    if isinstance(doctor_artifact, str) and doctor_artifact:
                        evidence_refs.append(doctor_artifact)
                    executed_actions.append(
                        {
                            "kind": kind,
                            "safe": True,
                            "ok": doctor_ok,
                            "returncode": int(doctor_proc.returncode),
                            "artifact": doctor_artifact,
                            "stdout": (doctor_proc.stdout or "")[:1000],
                            "stderr": (doctor_proc.stderr or "")[:1000],
                        }
                    )
                except Exception as exc:
                    safe_action_failed = True
                    executed_actions.append(
                        {
                            "kind": kind,
                            "safe": True,
                            "ok": False,
                            "error": str(exc)[:200],
                        }
                    )
                continue

        # preserve insertion order while de-duplicating evidence refs
        evidence_refs = list(dict.fromkeys([ref for ref in evidence_refs if isinstance(ref, str) and ref]))
        outcome_payload = {
            "schema": "ajax.lab.providers_audit_outcome.v1",
            "ts_utc": _utc_now(),
            "job_id": job_id,
            "job_kind": "providers_audit",
            "audit_returncode": int(audit_proc.returncode),
            "auth_quota_sensitive": auth_quota_sensitive,
            "checklist_only": bool(auth_quota_sensitive and not planned_actions),
            "planned_actions": planned_actions,
            "blocked_actions": blocked_actions,
            "executed_actions": executed_actions,
            "checklist": checklist,
            "evidence_refs": evidence_refs,
            "expected_state": expected_state,
        }
        _write_json(outcome_path, outcome_payload)
        evidence_refs.append(str(outcome_path))
        evidence_refs = list(dict.fromkeys(evidence_refs))

        findings_present = int(audit_proc.returncode) == 2
        if safe_action_failed:
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["providers_audit_safe_action_failed"],
                "evidence_refs": evidence_refs,
                "next_action": "review_providers_audit_plan",
                "summary": "Providers audit ran, but a safe remediation action failed.",
                "recommended_actions": planned_actions,
                "providers_audit_plan_path": str(plan_path),
                "providers_audit_outcome_path": str(outcome_path),
            }

        failure_codes = ["providers_audit_findings_present"] if findings_present else []
        next_action = "noop"
        if findings_present:
            next_action = "review_providers_audit_plan"
        if auth_quota_sensitive:
            next_action = "ask_user_auth_quota"
        summary = "Providers audit completed without findings."
        if findings_present and auth_quota_sensitive:
            summary = "Providers audit found auth/quota-sensitive issues; checklist generated (no credential actions executed)."
        elif findings_present and planned_actions:
            summary = "Providers audit found issues; safe remediation actions were executed and documented."
        elif findings_present:
            summary = "Providers audit found issues; remediation plan/checklist was recorded."
        return {
            "outcome": "PASS",
            "efe_pass": True,
            "failure_codes": failure_codes,
            "evidence_refs": evidence_refs,
            "next_action": next_action,
            "summary": summary,
            "recommended_actions": planned_actions,
            "providers_audit_plan_path": str(plan_path),
            "providers_audit_outcome_path": str(outcome_path),
            "checklist_only": bool(auth_quota_sensitive and not planned_actions),
        }

    def _execute_probe_ui(self, job: Dict[str, Any], path: Path) -> Dict[str, Any]:
        gate = _ui_policy_gate(self.root_dir, job=job, ui_intrusive=True)
        if gate:
            return gate
        job_id = str(job.get("job_id") or "job")
        mission_id = str(job.get("mission_id") or "mission")
        evidence_dir = self._ensure_evidence_dir(job_id)
        ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

        steps: list[Dict[str, Any]] = []
        errors: list[str] = []

        def _record(step: str, ok: bool, detail: Any) -> None:
            steps.append({"step": step, "ok": bool(ok), "detail": detail})

        try:
            client = self._driver_client()
        except Exception as exc:
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["lab_driver_unavailable"],
                "evidence_refs": [],
                "next_action": "check_lab_driver",
                "summary": f"LAB driver unavailable: {str(exc)[:200]}",
            }

        self._touch_job_heartbeat(job, path)
        try:
            health = client.health()
            _record("driver.health", True, health)
        except Exception as exc:
            _record("driver.health", False, {"error": str(exc)})
            errors.append("driver_health_failed")

        self._touch_job_heartbeat(job, path)
        try:
            inspected = client.inspect_window()
            _record("window.inspect", True, inspected)
        except Exception as exc:
            _record("window.inspect", False, {"error": str(exc)})
            errors.append("window_inspect_failed")

        meta_path = evidence_dir / f"probe_ui_{ts_label}.json"
        meta = {
            "ts": time.time(),
            "ts_utc": _utc_now(),
            "job_id": job_id,
            "mission_id": mission_id,
            "driver_url": self.driver_url,
            "steps": steps,
        }
        _write_json(meta_path, meta)

        evidence_refs = [str(meta_path)]
        outcome = "PASS"
        efe_pass = True
        failure_codes = []
        if errors:
            outcome = "FAIL"
            efe_pass = False
            failure_codes.extend(errors)
        summary = "LAB UI probe completed."
        if errors:
            summary = f"LAB UI probe failed ({', '.join(errors)})"
        return {
            "outcome": outcome,
            "efe_pass": efe_pass,
            "failure_codes": failure_codes,
            "evidence_refs": evidence_refs,
            "next_action": "review_lab_probe",
            "summary": summary,
        }

    def _execute_probe_notepad(self, job: Dict[str, Any], path: Path) -> Dict[str, Any]:
        gate = _ui_policy_gate(self.root_dir, job=job, ui_intrusive=True)
        if gate:
            return gate
        job_id = str(job.get("job_id") or "job")
        mission_id = str(job.get("mission_id") or "mission")
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        if not params and isinstance(job.get("args"), dict):
            params = job.get("args") or {}
        evidence_dir = self._ensure_evidence_dir(job_id)
        ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        message = str(params.get("message") or f"LAB probe {ts_label}").strip()
        title_hint = str(params.get("title_hint") or "notepad").strip()
        file_path = params.get("file_path")
        display_target = str(params.get("display_target") or "").strip().lower()
        require_dummy = display_target == "dummy"
        if isinstance(file_path, str) and file_path.strip():
            target_path = Path(_windows_to_wsl_path(file_path.strip()))
        else:
            target_path = evidence_dir / f"notepad_{ts_label}.txt"
        if not target_path.is_absolute():
            target_path = (self.root_dir / target_path).resolve()

        steps: list[Dict[str, Any]] = []
        errors: list[str] = []

        def _record(step: str, ok: bool, detail: Any) -> None:
            steps.append({"step": step, "ok": bool(ok), "detail": detail})

        try:
            client = self._driver_client()
        except Exception as exc:
            return {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["lab_driver_unavailable"],
                "evidence_refs": [],
                "next_action": "check_lab_driver",
                "summary": f"LAB driver unavailable: {str(exc)[:200]}",
            }

        try:
            health = client.health()
            _record("driver.health", True, health)
        except Exception as exc:
            _record("driver.health", False, {"error": str(exc)})

        bounds = None
        if require_dummy:
            bounds = _dummy_display_bounds(self.root_dir)
            if not bounds:
                return {
                    "outcome": "FAIL",
                    "efe_pass": False,
                    "failure_codes": ["dummy_display_missing"],
                    "evidence_refs": [],
                    "next_action": "run_display_probe",
                    "summary": "Dummy display bounds missing; aborting notepad probe.",
                }

        self._touch_job_heartbeat(job, path)
        try:
            launch = client.app_launch(process="notepad.exe")
            _record("app.launch", True, launch)
        except Exception as exc:
            _record("app.launch", False, {"error": str(exc)})
            errors.append("notepad_launch_failed")
        time.sleep(0.4)
        self._touch_job_heartbeat(job, path)

        move_ok = True
        if bounds:
            pad = 40
            target_x = bounds["x"] + pad
            target_y = bounds["y"] + pad
            target_w = max(240, min(900, bounds["width"] - (pad * 2)))
            target_h = max(180, min(700, bounds["height"] - (pad * 2)))
            try:
                moved = client.window_move(
                    process="notepad.exe",
                    x=target_x,
                    y=target_y,
                    width=target_w,
                    height=target_h,
                )
                _record("window.move", True, moved)
            except Exception as exc:
                _record("window.move", False, {"error": str(exc)})
                errors.append("notepad_move_failed")
                move_ok = False
        time.sleep(0.2)
        self._touch_job_heartbeat(job, path)

        abort_after_move = bool(require_dummy and not move_ok)
        focus_ok = False
        save_result: Dict[str, Any] = {}
        file_ok = False
        shot = None
        if not abort_after_move:
            try:
                focus = client.window_focus(
                    process="notepad.exe",
                    title_contains=title_hint if title_hint else None,
                )
                _record("window.focus", True, focus)
                focus_ok = True
            except Exception as exc:
                _record("window.focus", False, {"error": str(exc)})
            if not focus_ok:
                errors.append("notepad_focus_failed")
                try:
                    focus = client.window_focus(process="notepad.exe")
                    _record("window.focus_fallback", True, focus)
                    focus_ok = True
                    errors = [e for e in errors if e != "notepad_focus_failed"]
                except Exception as exc:
                    _record("window.focus_fallback", False, {"error": str(exc)})
            time.sleep(0.2)
            self._touch_job_heartbeat(job, path)

            try:
                typed = client.keyboard_type(text=message, submit=False)
                _record("keyboard.type", True, typed)
            except Exception as exc:
                _record("keyboard.type", False, {"error": str(exc)})
                errors.append("notepad_type_failed")
            time.sleep(0.2)
            self._touch_job_heartbeat(job, path)

            try:
                save_result = client.run_save_notepad_v2_path(str(target_path))
                ok = bool(save_result.get("ok"))
                _record("notepad.save", ok, save_result)
                if not ok:
                    errors.append("notepad_save_failed")
            except Exception as exc:
                _record("notepad.save", False, {"error": str(exc)})
                errors.append("notepad_save_failed")
            time.sleep(0.3)
            self._touch_job_heartbeat(job, path)

            try:
                file_ok = target_path.exists()
            except Exception:
                file_ok = False
            if not file_ok:
                errors.append("notepad_file_missing")
                _record("file.exists", False, {"path": str(target_path)})
            else:
                _record("file.exists", True, {"path": str(target_path)})

            if capture_lab_snapshot is not None:
                try:
                    shot = capture_lab_snapshot(
                        root_dir=self.root_dir,
                        job_id=job_id,
                        mission_id=mission_id,
                        active_window=False,
                        driver_url=self.driver_url,
                        context=f"{job_id}_notepad",
                    )
                    _record("lab.snap", True, {"png_path": shot.get("png_path"), "json_path": shot.get("json_path")})
                except Exception as exc:
                    _record("lab.snap", False, {"error": str(exc)})
                    errors.append("lab_snap_failed")
            else:
                _record("lab.snap", False, {"error": "lab_snap_unavailable"})
                errors.append("lab_snap_unavailable")

        close_res: Dict[str, Any] = {}
        try:
            close_res = client.run_close_notepad()
            _record("notepad.close", bool(close_res.get("ok")), close_res)
        except Exception as exc:
            _record("notepad.close", False, {"error": str(exc)})

        meta_path = evidence_dir / f"probe_notepad_{ts_label}.json"
        meta = {
            "ts": time.time(),
            "ts_utc": _utc_now(),
            "job_id": job_id,
            "mission_id": mission_id,
            "driver_url": self.driver_url,
            "message": message,
            "file_path": str(target_path),
            "steps": steps,
            "save_result": save_result,
            "close_result": close_res,
        }
        if shot:
            meta["screenshot"] = {
                "png_path": shot.get("png_path"),
                "json_path": shot.get("json_path"),
                "warnings": shot.get("warnings") or [],
                "session_id": shot.get("session_id"),
            }
        _write_json(meta_path, meta)

        evidence_refs = [str(meta_path)]
        if file_ok:
            evidence_refs.append(str(target_path))
        if shot:
            for ref_key in ("png_path", "json_path"):
                ref_val = shot.get(ref_key)
                if ref_val:
                    evidence_refs.append(str(ref_val))

        outcome = "PASS"
        efe_pass = True
        failure_codes = []
        if errors:
            outcome = "FAIL"
            efe_pass = False
            failure_codes.extend(errors)
        summary = "LAB notepad probe completed."
        if errors:
            summary = f"LAB notepad probe failed ({', '.join(errors)})"
        return {
            "outcome": outcome,
            "efe_pass": efe_pass,
            "failure_codes": failure_codes,
            "evidence_refs": evidence_refs,
            "next_action": "review_lab_probe",
            "summary": summary,
        }

    def _execute_job(self, job: Dict[str, Any], path: Path) -> Dict[str, Any]:
        kind = self._resolve_job_kind(job)
        if kind == "lab_notepad_smoke":
            kind = "probe_notepad"
        if kind == "probe_notepad_dummy":
            kind = "probe_notepad"
        if kind == "snap_lab_silent":
            kind = "snap_lab"
        if kind in {"snap_lab", "capabilities_refresh", "providers_probe", "providers_audit", "probe_ui", "probe_notepad"}:
            resolved = self._resolve_action_executor(kind)
            if not bool(resolved.get("ok")):
                label = kind or "unknown"
                err = str(resolved.get("error") or "executor_resolution_failed")
                evidence_refs = []
                reg_path = resolved.get("registry_path")
                if reg_path:
                    evidence_refs.append(str(reg_path))
                return {
                    "outcome": "FAIL",
                    "efe_pass": False,
                    "failure_codes": ["lab_action_executor_invalid", err, f"job_kind_{label}"],
                    "evidence_refs": evidence_refs,
                    "next_action": "review_lab_action_executor_registry",
                    "summary": f"LAB action executor invalid for {label}: {err}",
                    "action_executor": resolved.get("action_executor"),
                    "action_executor_source": resolved.get("source"),
                    "action_executor_handler": resolved.get("handler"),
                }
            result = resolved["handler_fn"](job, path)
            if isinstance(result, dict):
                result.setdefault("action_executor", resolved.get("action_executor"))
                result.setdefault("action_executor_source", resolved.get("source"))
                result.setdefault("action_executor_handler", resolved.get("handler"))
            return result
        label = kind or "unknown"
        return {
            "outcome": "FAIL",
            "efe_pass": False,
            "failure_codes": ["lab_job_unsupported", f"job_kind_{label}"],
            "evidence_refs": [],
            "next_action": "manual_lab_required",
            "summary": f"LAB job kind unsupported: {label}",
        }

    def _sweep_stale_jobs(self) -> None:
        now = time.time()
        if now - self._last_sweep_ts < float(self.stale_sweep_s):
            return
        self._last_sweep_ts = now
        for item in self.store.list_jobs(statuses={"QUEUED", "RUNNING"}):
            job = item.get("job") or {}
            path = item.get("path")
            if isinstance(path, Path):
                try:
                    self.store.annotate_job_staleness(job, path, now_ts=now)
                except Exception:
                    continue

    def run_once(self) -> Optional[str]:
        self._sweep_stale_jobs()
        next_item = self.store.pick_next_job()
        if not next_item:
            self._maybe_run_lab_org()
            self._status = "READY"
            self._maybe_heartbeat()
            return None
        job = next_item.get("job") or {}
        path = next_item.get("path")
        job_id = str(job.get("job_id") or "").strip()
        if not job_id or not isinstance(path, Path):
            return None
        try:
            job, path = self.store.load_job(job_id)
        except Exception:
            return None
        if str(job.get("status") or "").upper() != "QUEUED":
            return None
        kind = self._resolve_job_kind(job)
        risk_level = str(job.get("risk_level") or "").strip().lower()
        requires_ack = bool(job.get("requires_ack"))
        self._status = "RUNNING"
        self.active_job_id = job_id
        self._mark_running(job, path)
        self._write_worker_heartbeat(status="RUNNING", active_job_id=job_id, last_job_id=self.last_job_id)
        try:
            gate_result = self._human_active_gate(job, kind)
            if gate_result:
                result = gate_result
            else:
                result = self._execute_job(job, path)
        except LabJobCancelled:
            result = {
                "outcome": "CANCELLED",
                "efe_pass": False,
                "failure_codes": ["preempted", "human_detected"],
                "evidence_refs": [],
                "next_action": "requeue_job",
                "summary": "LAB job cancelled due to human detection preemption.",
            }
        except Exception as exc:
            result = {
                "outcome": "FAIL",
                "efe_pass": False,
                "failure_codes": ["lab_worker_exception"],
                "evidence_refs": [],
                "next_action": "review_lab_worker",
                "summary": str(exc)[:200],
            }
        evidence_refs = list(result.get("evidence_refs") or [])
        self._append_output_paths(job, path, evidence_refs)
        explore = _compute_explore_state(self.root_dir)
        write_meta = self.store.write_result(
            job_id=job_id,
            mission_id=str(job.get("mission_id") or "mission"),
            outcome=str(result.get("outcome") or "FAIL"),
            efe_pass=bool(result.get("efe_pass")),
            failure_codes=list(result.get("failure_codes") or []),
            evidence_refs=evidence_refs,
            next_action=str(result.get("next_action") or "review_lab_worker"),
            summary=str(result.get("summary") or ""),
            risk_level=risk_level,
            job_kind=kind,
            requires_ack=requires_ack,
            episode_fields={
                "explore_state": explore.get("state"),
                "human_active": explore.get("human_active"),
                "hypothesis": _episode_hypothesis(kind),
                "delta": str(result.get("summary") or ""),
                "conclusion": str(result.get("summary") or ""),
                "tags": _episode_tags(job, kind=kind),
            },
        )
        # Best-effort LEANN ingestion of the episode (never blocks).
        episode_path = None
        try:
            ep = write_meta.get("episode_path")
            if isinstance(ep, str) and ep:
                episode_path = Path(ep)
        except Exception:
            episode_path = None
        ingest_receipt = None
        if isinstance(episode_path, Path) and episode_path.exists():
            _update_json_file(
                episode_path,
                {
                    "evidence_paths": {
                        "result": str(write_meta.get("result_path") or ""),
                        "job": str(path),
                        "evidence_refs": evidence_refs,
                        "logs": [str(self.root_dir / "artifacts" / "lab" / "worker.log")],
                    }
                },
            )
            ingest_receipt = _maybe_ingest_episode(self.root_dir, episode_path)
            _update_json_file(episode_path, {"leann_ingest": ingest_receipt})
            # Also attach to result payload for quick diagnosis.
            try:
                # Find the latest result for this job id and patch it.
                found = self.store.find_result_for_job(job_id)
                if found:
                    payload_r, result_path = found
                    if isinstance(result_path, Path):
                        _update_json_file(result_path, {"leann_ingest": ingest_receipt})
            except Exception:
                pass
        try:
            job_after, path_after = self.store.load_job(job_id)
            job_after["completed_ts"] = time.time()
            job_after["last_heartbeat_ts"] = time.time()
            self._write_job(job_after, path_after)
        except Exception:
            pass
        self.last_job_id = job_id
        self.active_job_id = None
        self._status = "READY"
        self._write_worker_heartbeat(status="READY", active_job_id=None, last_job_id=job_id)
        return job_id

    def _maybe_run_lab_org(self) -> None:
        now = time.time()
        try:
            tick_s = float(os.getenv("AJAX_LAB_ORG_TICK_SEC", "30") or 30)
        except Exception:
            tick_s = 30.0
        if now - float(self._last_lab_org_tick_ts) < max(5.0, tick_s):
            return
        self._last_lab_org_tick_ts = now

        waiting_path = self.root_dir / "artifacts" / "state" / "waiting_mission.json"
        if waiting_path.exists():
            return
        lab_org = self.store.state.get("lab_org") or {}
        status = str(lab_org.get("status") or "")
        reason = str(lab_org.get("reason") or "")
        if status != "RUNNING" and reason in {"not_started", "chat_preemption", "prod_mission_start"}:
            try:
                self.store.resume_lab_org("idle_autostart", metadata={"source": "lab_worker"})
            except Exception:
                pass
        try:
            from agency.lab_org import lab_org_tick

            lab_org_tick(self.root_dir)
        except Exception:
            return

    def run(self, *, max_jobs: Optional[int] = None) -> int:
        processed = 0
        while True:
            try:
                job_id = self.run_once()
            except KeyboardInterrupt:
                break
            if job_id:
                processed += 1
                if max_jobs is not None and processed >= max_jobs:
                    break
                continue
            if max_jobs is not None and processed >= max_jobs:
                break
            time.sleep(max(0.1, float(self.idle_sleep_s)))
        return processed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lab_worker", description="Loop minimo de ejecucion LAB.")
    parser.add_argument("--root", default=".", help="Root del repo (AJAX_HOME).")
    parser.add_argument("--worker-id", help="ID del worker (override).")
    parser.add_argument("--once", action="store_true", help="Tick-only: ejecuta un scheduler tick (LAB_ORG) y sale (no daemon, no pid tracking).")
    parser.add_argument("--max-jobs", type=int, default=None, help="Maximo de jobs a procesar antes de salir.")
    parser.add_argument("--idle-sleep", type=float, default=None, help="Segundos de espera cuando no hay jobs.")
    parser.add_argument("--heartbeat", type=float, default=None, help="Segundos entre heartbeats.")
    parser.add_argument("--stale-sweep", type=float, default=None, help="Segundos entre sweeps de stale jobs.")
    parser.add_argument("--driver-url", type=str, default=None, help="Override URL del driver LAB.")
    parser.add_argument("--daemon", action="store_true", help="Ejecutar en background (pidfile).")
    parser.add_argument("--pidfile", type=Path, default=None, help="Ruta del pidfile.")
    parser.add_argument("--log-file", type=Path, default=None, help="Ruta de log (stdout/stderr).")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if args.daemon:
        if args.once:
            print("No se puede usar --daemon junto a --once.", file=sys.stderr)
            return 2
        pidfile = args.pidfile or (root / "artifacts" / "lab" / "worker.pid")
        if pidfile.exists():
            try:
                pid = int(pidfile.read_text(encoding="utf-8").strip())
                if pid_running(pid):
                    print("LAB worker ya esta en ejecucion.")
                    return 0
            except Exception:
                pass
            try:
                pidfile.unlink()
            except Exception:
                pass
        cmd = [
            sys.executable,
            "-m",
            "agency.lab_worker",
            "--root",
            str(root),
        ]
        if args.worker_id:
            cmd.extend(["--worker-id", str(args.worker_id)])
        if args.pidfile:
            cmd.extend(["--pidfile", str(pidfile)])
        if args.max_jobs is not None:
            cmd.extend(["--max-jobs", str(args.max_jobs)])
        if args.idle_sleep is not None:
            cmd.extend(["--idle-sleep", str(args.idle_sleep)])
        if args.heartbeat is not None:
            cmd.extend(["--heartbeat", str(args.heartbeat)])
        if args.stale_sweep is not None:
            cmd.extend(["--stale-sweep", str(args.stale_sweep)])
        if args.driver_url:
            cmd.extend(["--driver-url", str(args.driver_url)])
        log_path = args.log_file or (root / "artifacts" / "lab" / "worker.log")
        cmd.extend(["--log-file", str(log_path)])
        log_handle = None
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
        except Exception:
            log_handle = None
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            start_new_session=True,
            stdout=log_handle,
            stderr=log_handle,
        )
        if log_handle:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(str(proc.pid), encoding="utf-8")
        print(f"LAB worker iniciado (pid={proc.pid}).")
        return 0

    if args.log_file:
        try:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
            sys.stdout = log_handle
            sys.stderr = log_handle
            atexit.register(log_handle.close)
        except Exception:
            pass

    if args.once:
        try:
            from agency.lab_org import lab_org_tick

            receipt = lab_org_tick(root)
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(f"lab_org_tick_failed: {exc}", file=sys.stderr)
            return 2

    if args.pidfile:
        try:
            pidfile = Path(args.pidfile)
            pidfile.parent.mkdir(parents=True, exist_ok=True)
            pidfile.write_text(str(os.getpid()), encoding="utf-8")
            def _cleanup_pid() -> None:
                try:
                    if pidfile.exists() and pidfile.read_text(encoding="utf-8").strip() == str(os.getpid()):
                        pidfile.unlink()
                except Exception:
                    pass
            atexit.register(_cleanup_pid)
        except Exception:
            pass

    worker = LabWorker(
        root,
        worker_id=args.worker_id,
        idle_sleep_s=args.idle_sleep,
        heartbeat_s=args.heartbeat,
        stale_sweep_s=args.stale_sweep,
        driver_url=args.driver_url,
    )
    worker.run(max_jobs=args.max_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
