"""
Lightweight plan runner for Agency jobs that embed step metadata.

It is opt-in: only jobs that carry a metadata["steps"] list will use it.

Cada step se interpreta como una **tarea atomizada** (no “línea ligera”).
Schema mínimo requerido por step (TaskStep v1):
  - id: str
  - intent: str
  - preconditions: {"expected_state": ExpectedState}
  - action: str  (ActionCatalog)
  - args: dict
  - evidence_required: list[str]
  - success_spec: {"expected_state": ExpectedState}  # MUST ser no vacío (windows/files/meta.must_be_active)
  - on_fail: "abort"  # prohibido continuar tras fallo
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import sys
import subprocess
from pathlib import Path, PureWindowsPath
import os
import shutil
import time
import threading
import traceback
from collections import defaultdict
import yaml  # type: ignore
from datetime import datetime

import logging

from agency.actuator import Actuator, Strategy, ActuatorError, create_default_actuator
from agency.close_editor_safe import close_editor_safe, SAVE_DIALOG_TOKENS
from agency.contract import AgencyJob
from agency.expected_state import ExpectedState, StateDelta, verify_efe
from agency.windows_driver_client import (
    WindowsDriverClient,
    WindowsDriverError,
    DriverConnectionError,
    DriverTimeout,
)  # type: ignore
from agency.vision_llm import vision_llm_click, VisionClickResult
from agency.models_registry import discover_models, list_vision_models, ModelInfo
from agency.provider_policy import env_rail
from agency.skills.os_inspection import inspect_ui_tree
from agency.explore_policy import load_explore_policy, read_human_signal, compute_human_active
try:
    from agency.actions_catalog import ActionCatalog
except Exception:  # pragma: no cover
    ActionCatalog = None  # type: ignore
try:
    from agency.human_permission import human_permission_gate_enabled, read_human_permission_status
except Exception:  # pragma: no cover - optional safety gate
    human_permission_gate_enabled = None  # type: ignore
    read_human_permission_status = None  # type: ignore
try:
    from agency.ops_ports_sessions import fix_ports_sessions
except Exception:  # pragma: no cover - optional ops
    fix_ports_sessions = None  # type: ignore

# Fallback paths for common apps when process is given without an absolute path.
APP_PATH_HINTS = {
    "brave.exe": r"C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
    "notepad++.exe": r"C:\\Program Files\\Notepad++\\notepad++.exe",
    "notepad.exe": r"C:\\Windows\\System32\\notepad.exe",
}
SECURITY_POLICY_DEFAULT = {
    "allow_app_launch": True,
    "max_app_launches_per_mission": 3,
    "per_app_limits": {},
    # Per-step consent (pixel-safety): TTL es permiso maestro; este gate es adicional y selectivo.
    # Config opcional via config/security_policy.yaml (step_consent.*) o env vars.
    "step_consent": {
        "enabled": True,
        # Si allowlist_actions está vacío/no definido, se deriva automáticamente desde ActionCatalog (risk=low).
        "allowlist_actions": [],
        # Risk levels que requieren consentimiento explícito por step.
        "risk_levels": ["medium", "high"],
        # Tras replan (plan nuevo dentro de la misma misión), exigir consentimiento antes de actuar físicamente.
        "require_after_replan": True,
        # Si el action no está en allowlist (aunque sea low), exigir consentimiento.
        "require_outside_allowlist": True,
    },
}

log = logging.getLogger(__name__)

_LOG_CONFIGURED = False
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOG_FILE = _LOG_DIR / "agency.log"
_PLAN_PROGRESS_LOCK = threading.Lock()
_PLAN_PROGRESS: Dict[str, Dict[str, Any]] = {}

_TASK_REQUIRED_FIELDS = (
    "id",
    "intent",
    "preconditions",
    "action",
    "args",
    "evidence_required",
    "success_spec",
    "on_fail",
)
_TASK_ALLOWED_ON_FAIL = {"abort"}
_EVIDENCE_KEYS = {"driver.screenshot", "driver.active_window", "driver.ui_inspect"}
_PHYSICAL_ACTIONS_REQUIRING_PERMISSION = {
    "app.launch",
    "keyboard.type",
    "keyboard.hotkey",
    "window.focus",
    "vision.llm_click",
    "desktop.isolate_active_window",
    "mouse.click",
    "mouse.move",
}

_STEP_CONSENT_STATUS = "step_consent_required"
_DEFERENCE_STATUS = "deference_human_active"


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _deference_enabled() -> bool:
    raw = os.getenv("AJAX_DEFERENCE_ENABLED")
    if raw is None:
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _deference_threshold_s() -> float:
    try:
        return max(0.1, float(os.getenv("AJAX_DEFERENCE_THRESHOLD_S", "2.0")))
    except Exception:
        return 2.0
_STEP_CONSENT_RISK_LEVELS_DEFAULT = {"medium", "high"}
_STEP_CONSENT_ALLOWLIST_ENV = "AJAX_STEP_CONSENT_ALLOWLIST"
_STEP_CONSENT_ENABLED_ENV = "AJAX_STEP_CONSENT_ENABLED"


def _expected_state_has_checks(expected: Optional[ExpectedState]) -> bool:
    if not expected or not isinstance(expected, dict):
        return False
    windows = expected.get("windows") or []
    files = expected.get("files") or []
    meta = expected.get("meta") or {}
    if windows:
        return True
    if files:
        return True
    if isinstance(meta, dict) and meta.get("must_be_active"):
        return True
    return False


def _parse_env_csv_set(name: str) -> Optional[set[str]]:
    raw = os.getenv(name)
    if raw is None:
        return None
    items = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            items.append(part)
    return set(items)


def _normalize_risk_level(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    if val in {"low", "medium", "high"}:
        return val
    return "medium"


def _resolve_step_consent_config(policy: Dict[str, Any], *, catalog: Any) -> Dict[str, Any]:
    """
    Resuelve la config de per-step consent a partir de security_policy + env overrides.
    """
    cfg = policy.get("step_consent") if isinstance(policy.get("step_consent"), dict) else {}
    enabled = cfg.get("enabled", True)
    try:
        enabled = bool(enabled)
    except Exception:
        enabled = True

    # Env override: AJAX_STEP_CONSENT_ENABLED=0/1
    env_enabled = os.getenv(_STEP_CONSENT_ENABLED_ENV)
    if env_enabled is not None:
        enabled = str(env_enabled).strip().lower() in {"1", "true", "yes", "on"}

    risk_levels_raw = cfg.get("risk_levels") or cfg.get("require_risk_levels")
    risk_levels: set[str] = set()
    if isinstance(risk_levels_raw, list):
        risk_levels = {_normalize_risk_level(x) for x in risk_levels_raw}
    if not risk_levels:
        risk_levels = set(_STEP_CONSENT_RISK_LEVELS_DEFAULT)

    require_after_replan = bool(cfg.get("require_after_replan", True))
    require_outside_allowlist = bool(cfg.get("require_outside_allowlist", True))

    # Allowlist: env override > config > derived default (risk=low).
    allowlist = _parse_env_csv_set(_STEP_CONSENT_ALLOWLIST_ENV)
    if allowlist is None:
        allowlist_raw = cfg.get("allowlist_actions") if isinstance(cfg, dict) else None
        if isinstance(allowlist_raw, list) and [x for x in allowlist_raw if str(x).strip()]:
            allowlist = {str(x).strip() for x in allowlist_raw if str(x).strip()}
        else:
            # Default: acciones low-risk del ActionCatalog.
            allowlist = set()
            try:
                if catalog is not None and hasattr(catalog, "list_actions"):
                    for spec in catalog.list_actions():
                        try:
                            name = getattr(spec, "name", None)
                            risk = getattr(spec, "risk_level", None)
                            if name and _normalize_risk_level(risk) == "low":
                                allowlist.add(str(name))
                        except Exception:
                            continue
            except Exception:
                allowlist = set()

    return {
        "enabled": enabled,
        "risk_levels": risk_levels,
        "allowlist_actions": allowlist,
        "require_after_replan": require_after_replan,
        "require_outside_allowlist": require_outside_allowlist,
    }


def _extract_expected_state(spec: Any) -> Optional[ExpectedState]:
    if not isinstance(spec, dict):
        return None
    candidate = spec.get("expected_state")
    return candidate if isinstance(candidate, dict) else None


def _validate_task_step(step: Any, *, catalog: Any = None) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Validación estructural: si falta algo, el plan es inválido (sin warnings).
    Devuelve (step_dict, violations). step_dict es el dict original (si era dict).
    """
    if not isinstance(step, dict):
        return None, ["step_not_a_dict"]

    violations: List[str] = []
    for key in _TASK_REQUIRED_FIELDS:
        if key not in step:
            violations.append(f"missing_field:{key}")

    step_id = step.get("id")
    if not isinstance(step_id, str) or not step_id.strip():
        violations.append("invalid_field:id")

    intent = step.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        violations.append("invalid_field:intent")

    on_fail = step.get("on_fail")
    if not isinstance(on_fail, str) or on_fail not in _TASK_ALLOWED_ON_FAIL:
        violations.append("invalid_field:on_fail")

    action = step.get("action")
    if not isinstance(action, str) or not action.strip():
        violations.append("invalid_field:action")
    elif catalog is not None:
        try:
            if hasattr(catalog, "is_allowed") and not catalog.is_allowed(action):
                violations.append(f"action_not_in_catalog:{action}")
        except Exception:
            pass

    args = step.get("args")
    if not isinstance(args, dict):
        violations.append("invalid_field:args")

    pre = step.get("preconditions")
    if not isinstance(pre, dict):
        violations.append("invalid_field:preconditions")
    else:
        pre_es = _extract_expected_state(pre)
        if pre_es is None:
            violations.append("invalid_preconditions:missing_expected_state")
        elif not isinstance(pre_es, dict):
            violations.append("invalid_preconditions:expected_state_not_object")

    succ = step.get("success_spec")
    if not isinstance(succ, dict):
        violations.append("invalid_field:success_spec")
    else:
        succ_es = _extract_expected_state(succ)
        if succ_es is None:
            violations.append("invalid_success_spec:missing_expected_state")
        else:
            action_name = step.get("action")
            if action_name != "await_user_input":
                if not _expected_state_has_checks(succ_es):
                    violations.append("invalid_success_spec:expected_state_empty")

    ev = step.get("evidence_required")
    if not isinstance(ev, list):
        violations.append("invalid_field:evidence_required")
    else:
        for item in ev:
            if not isinstance(item, str) or not item.strip():
                violations.append("invalid_evidence_required:item_not_string")
                break
            if item not in _EVIDENCE_KEYS:
                violations.append(f"invalid_evidence_required:unknown_key:{item}")
                break

    return step, violations


def _capture_evidence(driver: Any, requirements: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    evidence: Dict[str, Any] = {}
    errors: List[str] = []
    if not requirements:
        return evidence, errors
    if not driver:
        return evidence, ["driver_missing_for_evidence"]

    for req in requirements:
        if req == "driver.screenshot":
            try:
                evidence["screenshot_path"] = str(driver.screenshot())
            except Exception as exc:
                errors.append(f"screenshot_failed:{str(exc)[:200]}")
        elif req == "driver.active_window":
            try:
                evidence["active_window"] = driver.get_active_window()
            except Exception as exc:
                errors.append(f"active_window_failed:{str(exc)[:200]}")
        elif req == "driver.ui_inspect":
            try:
                evidence["ui_inspect"] = driver.inspect_window()
            except Exception as exc:
                errors.append(f"ui_inspect_failed:{str(exc)[:200]}")
        else:
            errors.append(f"unknown_evidence:{req}")
    return evidence, errors


def _configure_logging() -> None:
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    level_name = str(os.getenv("PLAN_RUNNER_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", stream=sys.stdout)
    else:
        root.setLevel(level)
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception:
        pass
    _LOG_CONFIGURED = True


def _summarize_detail(detail: Any) -> Any:
    if detail is None:
        return None
    if isinstance(detail, (bool, int, float)):
        return detail
    if isinstance(detail, str):
        return detail if len(detail) <= 200 else detail[:200] + "..."
    if isinstance(detail, dict):
        keys = ["action", "process", "path", "error", "success", "skipped_launch", "confidence"]
        summary = {k: detail.get(k) for k in keys if detail.get(k) is not None}
        if summary:
            return summary
        subset: Dict[str, Any] = {}
        for key in list(detail.keys())[:3]:
            subset[key] = detail[key]
        return subset
    if isinstance(detail, list):
        return f"list(len={len(detail)})"
    return str(detail)[:200]


def _record_plan_progress(
    job_id: str,
    mission_id: Optional[str],
    total_steps: int,
    step_index: int,
    step_id: Optional[str],
    action: Optional[str],
    status: str,
    tries: int,
    error: Optional[str] = None,
    detail: Any = None,
    steps_completed: int = 0,
) -> None:
    snapshot = {
        "mission_id": mission_id,
        "job_id": job_id,
        "total_steps": total_steps,
        "step_index": step_index,
        "steps_completed": steps_completed,
        "step_id": step_id,
        "action": action,
        "status": status,
        "tries": tries,
        "error": error,
        "timestamp": time.time(),
    }
    if detail is not None:
        snapshot["detail"] = detail
    with _PLAN_PROGRESS_LOCK:
        _PLAN_PROGRESS[job_id] = snapshot


def _get_plan_progress_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    with _PLAN_PROGRESS_LOCK:
        snap = _PLAN_PROGRESS.get(job_id)
        return dict(snap) if snap else None


def _clear_plan_progress(job_id: str) -> None:
    with _PLAN_PROGRESS_LOCK:
        _PLAN_PROGRESS.pop(job_id, None)


def _format_thread_stack(thread_ident: Optional[int]) -> Optional[List[str]]:
    if not thread_ident:
        return None
    try:
        frame = sys._current_frames().get(thread_ident)
    except Exception:
        frame = None
    if not frame:
        return None
    try:
        return traceback.format_stack(frame)
    except Exception:
        return None


def _resolve_wrapper_timeout(timeout_override: Optional[float]) -> Optional[float]:
    if timeout_override is not None:
        try:
            resolved = float(timeout_override)
        except Exception:
            return None
        return resolved if resolved > 0 else None
    raw = os.getenv("PLAN_RUNNER_FUTURE_TIMEOUT") or os.getenv("PLAN_RUNNER_WRAPPER_TIMEOUT") or "9"
    try:
        val = float(raw)
    except Exception:
        val = 9.0
    return val if val > 0 else None

@dataclass
class StepResult:
    step_id: str
    ok: bool
    detail: Any
    tries: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "ok": self.ok,
            "detail": self.detail,
            "tries": self.tries,
            "error": self.error,
        }


def _emit_capability_gap(job: AgencyJob, step_results: List[StepResult], final_result: Dict[str, Any]) -> None:
    """
    Persiste un registro de gap de capacidad cuando el resultado no es ok.
    """
    try:
        if final_result.get("ok", True):
            return
        if str(final_result.get("status") or "").strip().lower() == "human_permission_required":
            return
        gap_dir = Path("artifacts") / "capability_gaps"
        gap_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        last_failed: Optional[StepResult] = None
        for sr in reversed(step_results):
            if not sr.ok:
                last_failed = sr
                break
        cap_family = None
        if isinstance(job.metadata, dict):
            cap_family = job.metadata.get("capability_family")
        detail_snapshot: Any = None
        if last_failed:
            detail_snapshot = _summarize_detail(last_failed.detail)
            if not cap_family and isinstance(last_failed.detail, dict):
                cap_family = last_failed.detail.get("capability_family")
        gap = {
            "timestamp": ts,
            "job_id": job.job_id,
            "mission_id": (job.metadata or {}).get("mission_id"),
            "goal": job.goal,
            "status": final_result.get("status"),
            "error": final_result.get("error"),
            "capability_family": cap_family or "unknown",
            "step_id": last_failed.step_id if last_failed else None,
            "step_error": last_failed.error if last_failed else None,
            "step_detail": detail_snapshot,
        }
        # Extra metadata for protocol / task-level failures (best-effort; schema is tolerant)
        if last_failed and isinstance(last_failed.detail, dict):
            for key in (
                "kind",
                "severity",
                "description",
                "fix_hint",
                "violations",
                "task_intent",
                "precondition_delta",
                "evidence_errors",
                "evidence",
            ):
                if key in last_failed.detail:
                    gap[key] = last_failed.detail.get(key)
        if "efe_delta" in final_result:
            gap["efe_delta"] = final_result.get("efe_delta")
        maybe_expected = (job.metadata or {}).get("expected_state")
        if maybe_expected:
            gap["expected_state"] = maybe_expected
        out_path = gap_dir / f"{ts}_{job.job_id}.json"
        out_path.write_text(json.dumps(gap, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logging.getLogger(__name__).warning("capability_gap_emit_failed", exc_info=True)


def _parse_strategy(raw: str | None) -> Strategy:
    if not raw:
        return Strategy.AUTO
    raw = str(raw).lower()
    if raw == "force_ballistic":
        return Strategy.FORCE_BALLISTIC
    if raw == "force_cognitive":
        return Strategy.FORCE_COGNITIVE
    return Strategy.AUTO


def _get_arg(step: Dict[str, Any], name: str, default: Any = None) -> Any:
    """
    Helper to pull args from either top-level keys or the nested args dict.
    """
    if name in step:
        return step.get(name)
    args = step.get("args") or {}
    if isinstance(args, dict):
        return args.get(name, default)
    return default


def _normalize_windows_path(raw: Optional[str]) -> Optional[str]:
    """
    Normaliza rutas recibidas desde hábitos/planes (WSL, /mnt/c, barras invertidas).
    """
    if raw is None:
        return None
    text = str(raw).strip().strip('"')
    if not text:
        return None
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5]
        remainder = text[6:]
        if drive.isalpha():
            return f"{drive.upper()}:\\{remainder.replace('/', '\\')}"
    if text.startswith("~/"):
        try:
            home = Path(os.path.expanduser("~"))
            text = str(home / text[2:])
        except Exception:
            pass
    # Normalizar separadores (mantiene UNC si aplica)
    if "://" not in text:
        text = text.replace("/", "\\")
    return text


def _resolve_userprofile() -> Optional[str]:
    """
    Best-effort para expandir %USERPROFILE% cuando el entorno WSL no lo trae.
    """
    env = os.getenv("USERPROFILE")
    if env:
        return _normalize_windows_path(env)
    hd = os.getenv("HOMEDRIVE")
    hp = os.getenv("HOMEPATH")
    if hd and hp:
        return _normalize_windows_path(f"{hd}{hp}")
    uname = os.getenv("USERNAME")
    if uname:
        return f"C:\\Users\\{uname}"
    try:
        repo_root = Path(__file__).resolve().parents[1]
        win_root = _normalize_windows_path(str(repo_root))
        if win_root and ":" in win_root:
            return str(Path(win_root).parent if os.name == "nt" else Path(PureWindowsPath(win_root)).parent)
    except Exception:
        pass
    return None


def _ensure_foreground(
    driver: WindowsDriverClient,
    last_focus: Optional[Dict[str, Any]],
    *,
    allow_save_dialog: bool = False,
) -> Dict[str, Any]:
    """
    Verifica que la ventana activa coincide con el último foco esperado.
    Abortará la acción si no hay foco previo o si no coincide, salvo que allow_save_dialog
    permita tolerar diálogos de guardado comunes (Save As / Guardar como).
    """
    if not last_focus and not allow_save_dialog:
        raise ActuatorError("active_window_mismatch:missing_last_focus")
    try:
        fg = driver.get_active_window()
    except Exception as exc:
        if allow_save_dialog:
            log.warning("plan_runner: foreground check skipped (cannot read active window: %s)", exc)
            return {}
        raise ActuatorError(f"active_window_mismatch:cannot_read_active:{exc}")

    expected_title = str(last_focus.get("title") or last_focus.get("title_contains") or "").lower()
    expected_proc = str(last_focus.get("process") or "").lower()
    actual_title = str((fg or {}).get("title") or "").lower()
    actual_proc = str((fg or {}).get("process") or "").lower()

    match = False
    if expected_title and expected_title in actual_title:
        match = True
    if expected_proc and expected_proc and expected_proc == actual_proc:
        match = True
    # Si el driver no reporta proceso pero sí título, tolerar coincidencia parcial para no bloquear
    if expected_proc and not actual_proc and actual_title:
        match = True

    if match:
        return fg

    if allow_save_dialog:
        # Tolerar diálogos de guardado aunque el proceso/título no coincida exactamente.
        title_tokens = actual_title.split()
        if any(tok in actual_title for tok in ("save as", "guardar", "guardar como")):
            return fg
        if "notepad" in actual_title:
            return fg
        if actual_proc in ("notepad.exe", "explorer.exe"):
            return fg
        if not actual_title and not actual_proc:
            log.warning("plan_runner: active window unknown during save, proceeding leniently")
            return fg

    raise ActuatorError(f"active_window_mismatch:expected={last_focus},actual={fg}")


def _run_os_probe() -> Dict[str, Any]:
    script = Path(__file__).resolve().parents[1] / "bin" / "os_driver_probe"
    if not script.exists():
        raise ActuatorError("os_driver_probe_not_found")
    cmd = [sys.executable, str(script)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ActuatorError(f"os_driver_probe_rc_{proc.returncode}:{proc.stderr.strip()}")
    raw = proc.stdout.strip()
    data: Dict[str, Any] = {}
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            break
        except Exception:
            continue
    if not data:
        raise ActuatorError("os_driver_probe_no_json_output")
    return data


def _run_driver_probe(
    driver: WindowsDriverClient,
    *,
    out_dir: Optional[str] = None,
    include_openapi: bool = True,
    include_shot: bool = True,
    include_screenshot: bool = True,
    include_window_inspect: bool = True,
) -> Dict[str, Any]:
    """
    Probe local sin LLM:
    - /health
    - /openapi.json (si existe)
    - /shot (si existe) + /screenshot
    - /window/inspect
    Persiste evidencia en artifacts/driver_probe/<run_id>/.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root = Path(__file__).resolve().parents[1]
    base = Path(out_dir) if out_dir else (root / "artifacts" / "driver_probe" / ts)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    steps: List[Dict[str, Any]] = []
    artifacts: Dict[str, Any] = {"out_dir": str(base)}
    ok = True

    def _record(name: str, step_ok: bool, detail: Any) -> None:
        steps.append({"name": name, "ok": bool(step_ok), "detail": detail})

    # 1) health
    try:
        health = driver.health()
        hp = base / "health.json"
        hp.write_text(json.dumps(health, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        artifacts["health_json"] = str(hp)
        _record("health", True, {"path": str(hp)})
    except Exception as exc:
        ok = False
        _record("health", False, {"error": str(exc)[:200]})

    # 2) openapi.json (best-effort; grande, pero útil en degradación)
    if include_openapi:
        try:
            openapi = driver._request_json("GET", "/openapi.json")  # type: ignore[attr-defined]
            op = base / "openapi.json"
            op.write_text(json.dumps(openapi, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            artifacts["openapi_json"] = str(op)
            _record("openapi", True, {"path": str(op)})
        except Exception as exc:
            # openapi puede no existir en builds minimal; no es fatal si health ok.
            _record("openapi", False, {"error": str(exc)[:200]})

    # 3) /shot (si existe): best-effort; si falla (404), no marcar fatal.
    if include_shot:
        try:
            shot = driver._request_json("GET", "/shot")  # type: ignore[attr-defined]
            sp = base / "shot.json"
            sp.write_text(json.dumps(shot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            artifacts["shot_json"] = str(sp)
            _record("shot", True, {"path": str(sp)})
        except Exception as exc:
            msg = str(exc)
            _record("shot", False, {"error": msg[:200]})
            if "driver_http_404" not in msg:
                ok = False

    # 4) screenshot
    if include_screenshot:
        try:
            src = driver.screenshot()
            artifacts["screenshot_path"] = str(src)
            # Copia best-effort al folder del probe (mantiene evidencia aunque el driver limpie temp).
            try:
                if src.exists():
                    dst = base / src.name
                    shutil.copy2(src, dst)
                    artifacts["screenshot_copy"] = str(dst)
            except Exception:
                pass
            _record("screenshot", True, {"path": str(src)})
        except Exception as exc:
            ok = False
            _record("screenshot", False, {"error": str(exc)[:200]})

    # 5) window.inspect
    if include_window_inspect:
        try:
            insp = driver.inspect_window()
            ip = base / "window_inspect.json"
            ip.write_text(json.dumps(insp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            artifacts["window_inspect_json"] = str(ip)
            _record("window.inspect", True, {"path": str(ip)})
        except Exception as exc:
            ok = False
            _record("window.inspect", False, {"error": str(exc)[:200]})

    report = {
        "ok": bool(ok),
        "ts_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": steps,
        "artifacts": artifacts,
    }
    try:
        rp = base / "probe_report.json"
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["report_path"] = str(rp)
    except Exception:
        pass
    return report


def _load_security_policy() -> Dict[str, Any]:
    try:
        root = Path(__file__).resolve().parents[1]
        path = root / "config" / "security_policy.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        data = {}
    pol = dict(SECURITY_POLICY_DEFAULT)
    if isinstance(data, dict):
        pol.update(data)
    pol["allow_app_launch"] = bool(pol.get("allow_app_launch", True))
    try:
        pol["max_app_launches_per_mission"] = int(pol.get("max_app_launches_per_mission", SECURITY_POLICY_DEFAULT["max_app_launches_per_mission"]))
    except Exception:
        pol["max_app_launches_per_mission"] = SECURITY_POLICY_DEFAULT["max_app_launches_per_mission"]
    per_limits = pol.get("per_app_limits") or {}
    pol["per_app_limits"] = per_limits if isinstance(per_limits, dict) else {}
    return pol


def run_job_plan(job: AgencyJob, actuator: Optional[Actuator] = None, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    """
    Ejecuta el plan en un hilo dedicado con timeout externo.
    """
    _configure_logging()
    timeout_value = _resolve_wrapper_timeout(timeout_seconds)
    if timeout_value is None:
        return _execute_job_plan(job, actuator=actuator)

    result_holder: Dict[str, Any] = {}
    error_holder: Dict[str, BaseException] = {}
    done_event = threading.Event()

    def _target() -> None:
        try:
            result_holder["value"] = _execute_job_plan(job, actuator=actuator)
        except BaseException as exc:  # pragma: no cover - we propagate upstream
            error_holder["error"] = exc
        finally:
            done_event.set()

    worker = threading.Thread(target=_target, name=f"plan_runner:{job.job_id}", daemon=True)
    worker.start()
    finished = done_event.wait(timeout_value)
    if finished:
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("value", {})

    mission_id = (job.metadata or {}).get("mission_id")
    snapshot = _get_plan_progress_snapshot(job.job_id)
    stack_summary = _format_thread_stack(worker.ident)
    timeout_detail: Dict[str, Any] = {
        "ok": False,
        "error": "plan_future_timeout",
        "status": "plan_future_timeout",
        "timeout_seconds": timeout_value,
        "job_id": job.job_id,
        "goal": job.goal,
        "mission_id": mission_id,
    }
    if snapshot:
        timeout_detail["last_logged_step"] = snapshot
    if stack_summary:
        timeout_detail["thread_stack"] = stack_summary
    log.error(
        "plan_runner: future timeout mission=%s job=%s timeout=%.1fs",
        mission_id or "-",
        job.job_id,
        timeout_value,
    )
    return timeout_detail


def _execute_job_plan(job: AgencyJob, actuator: Optional[Actuator] = None) -> Dict[str, Any]:
    steps = (job.metadata or {}).get("steps") or []
    if not isinstance(steps, list) or not steps:
        raise ValueError("run_job_plan requiere steps en job.metadata")

    _configure_logging()
    total_steps = len(steps)
    log.info("RUN_PLAN: start job=%s goal=%s steps=%d", job.job_id, job.goal, total_steps)
    root_dir = Path(__file__).resolve().parents[1]
    act = actuator or create_default_actuator()
    driver = WindowsDriverClient()
    catalog = None
    if ActionCatalog is not None:
        try:
            catalog = ActionCatalog(Path(__file__).resolve().parents[1])
        except Exception:
            catalog = None
    policy = _load_security_policy()
    step_consent_cfg = _resolve_step_consent_config(policy, catalog=catalog)
    allow_app_launch = bool(policy.get("allow_app_launch", True))
    max_launches_global = int(policy.get("max_app_launches_per_mission", 0) or 0)
    per_app_limits: Dict[str, Any] = policy.get("per_app_limits") or {}
    app_launch_counts: Dict[str, int] = defaultdict(int)
    mission_id = (job.metadata or {}).get("mission_id")
    attempt = (job.metadata or {}).get("attempt")
    after_replan = bool((job.metadata or {}).get("after_replan", False))
    replan_consent_granted = bool((job.metadata or {}).get("replan_consent_granted", False))
    step_consents_raw = (job.metadata or {}).get("step_consents") or {}
    step_consents: Dict[str, bool] = {}
    if isinstance(step_consents_raw, dict):
        for k, v in step_consents_raw.items():
            try:
                step_consents[str(k)] = bool(v)
            except Exception:
                continue

    last_focus: Optional[Dict[str, Any]] = None
    results: List[StepResult] = []
    resume_from_step_index = 1
    try:
        resume_from_step_index = int((job.metadata or {}).get("resume_from_step_index") or 1)
    except Exception:
        resume_from_step_index = 1
    if resume_from_step_index < 1:
        resume_from_step_index = 1
    resume_state_raw = (job.metadata or {}).get("resume_state")
    if isinstance(resume_state_raw, dict):
        try:
            maybe_focus = resume_state_raw.get("last_focus")
            if isinstance(maybe_focus, dict):
                last_focus = dict(maybe_focus)
        except Exception:
            pass
        try:
            maybe_counts = resume_state_raw.get("app_launch_counts")
            if isinstance(maybe_counts, dict):
                for k, v in maybe_counts.items():
                    try:
                        app_launch_counts[str(k)] = int(v)
                    except Exception:
                        continue
        except Exception:
            pass
        try:
            maybe_results = resume_state_raw.get("step_results")
            if isinstance(maybe_results, list):
                for item in maybe_results:
                    if not isinstance(item, dict):
                        continue
                    sid = str(item.get("step_id") or "").strip()
                    if not sid:
                        continue
                    results.append(
                        StepResult(
                            step_id=sid,
                            ok=bool(item.get("ok", False)),
                            detail=item.get("detail"),
                            tries=int(item.get("tries") or 0),
                            error=item.get("error"),
                        )
                    )
        except Exception:
            pass
    expected_state: Optional[ExpectedState] = None
    try:
        maybe_efe = (job.metadata or {}).get("expected_state")
        if isinstance(maybe_efe, dict):
            expected_state = maybe_efe  # type: ignore[assignment]
    except Exception:
        expected_state = None
    success = True
    runner_timeout = float(os.getenv("PLAN_RUNNER_TIMEOUT", "10"))
    efe_timeout = float(os.getenv("EFE_TIMEOUT_SECONDS", "5.0"))
    precondition_timeout_default = float(os.getenv("TASK_PRECONDITION_TIMEOUT_SECONDS", "1.0"))
    start_ts = time.time()

    _record_plan_progress(job.job_id, mission_id, total_steps, 0, None, None, "init", 0, steps_completed=len(results))

    log.info(
        "plan_runner: mission=%s attempt=%s steps=%d goal=%s",
        mission_id or "-",
        attempt or "-",
        total_steps,
        job.goal,
    )

    log.info("RUN_PLAN: entering loop (timeout=%.1fs)", runner_timeout)
    try:
        if resume_from_step_index > total_steps:
            resume_from_step_index = 1
        steps_iter = steps[resume_from_step_index - 1 :]
        for step_index, step in enumerate(steps_iter, start=resume_from_step_index):
            if time.time() - start_ts > runner_timeout:
                log.warning("plan_runner: timeout global tras %.1fs", runner_timeout)
                _record_plan_progress(
                    job.job_id,
                    mission_id,
                    total_steps,
                    step_index,
                    None,
                    None,
                    "internal_timeout",
                    0,
                    steps_completed=len(results),
                )
                return {
                    "ok": False,
                    "error": "plan_runner_timeout",
                    "status": "plan_runner_timeout",
                    "timeout_seconds": runner_timeout,
                    "steps": [r.to_dict() for r in results],
                    "job_id": job.job_id,
                    "goal": job.goal,
                }
            validated_step, violations = _validate_task_step(step, catalog=catalog)
            if violations:
                raw_step_id = None
                if isinstance(step, dict):
                    raw_step_id = step.get("id") or step.get("goal")
                step_id = str(raw_step_id or f"step{step_index}")
                detail = {
                    "status": "protocol_violation",
                    "capability_family": "atomicity_violation",
                    "kind": "atomicity_violation",
                    "severity": "high",
                    "description": "Task decomposition violated equal-rigor rule",
                    "fix_hint": "enforce per-step preconditions + per-step EFE; forbid continue-on-fail; require explicit evidence",
                    "violations": violations,
                }
                results.append(StepResult(step_id=step_id, ok=False, detail=detail, tries=0, error="protocol_violation"))
                _record_plan_progress(
                    job.job_id,
                    mission_id,
                    total_steps,
                    step_index,
                    step_id,
                    None,
                    "protocol_violation",
                    0,
                    error="protocol_violation",
                    detail=_summarize_detail(detail),
                    steps_completed=len(results),
                )
                out = {
                    "ok": False,
                    "error": "protocol_violation",
                    "status": "protocol_violation",
                    "violations": violations,
                    "steps": [r.to_dict() for r in results],
                    "job_id": job.job_id,
                    "goal": job.goal,
                }
                _emit_capability_gap(job, results, out)
                return out

            assert validated_step is not None  # validated by _validate_task_step
            step_id = str(validated_step.get("id") or f"step{step_index}")
            task_intent = str(validated_step.get("intent") or "")
            goal = task_intent
            strategy = _parse_strategy(step.get("strategy"))
            retries = int(step.get("retry", 0))
            on_fail = str(validated_step.get("on_fail") or "abort")
            action = str(validated_step.get("action") or "").strip()
            preconditions_spec = validated_step.get("preconditions") or {}
            success_spec = validated_step.get("success_spec") or {}
            evidence_required = validated_step.get("evidence_required") or []

            # Gate duro: verificar precondiciones antes de ejecutar la tarea
            pre_es = _extract_expected_state(preconditions_spec)
            pre_timeout = precondition_timeout_default
            try:
                if isinstance(preconditions_spec, dict) and preconditions_spec.get("timeout_s") is not None:
                    pre_timeout = float(preconditions_spec.get("timeout_s") or pre_timeout)
            except Exception:
                pre_timeout = precondition_timeout_default
            if _expected_state_has_checks(pre_es):
                pre_ok, pre_delta = verify_efe(pre_es, driver=driver, timeout_s=pre_timeout)
                if not pre_ok:
                    detail = {
                        "status": "precondition_failed",
                        "capability_family": "task.precondition_failed",
                        "task_intent": task_intent,
                        "action": action,
                        "precondition_delta": pre_delta,
                    }
                    results.append(StepResult(step_id=step_id, ok=False, detail=detail, tries=0, error="precondition_failed"))
                    _record_plan_progress(
                        job.job_id,
                        mission_id,
                        total_steps,
                        step_index,
                        step_id,
                        action,
                        "precondition_failed",
                        0,
                        error="precondition_failed",
                        detail=_summarize_detail(detail),
                        steps_completed=len(results),
                    )
                    out = {
                        "ok": False,
                        "error": "precondition_failed",
                        "status": "precondition_failed",
                        "precondition_delta": pre_delta,
                        "steps": [r.to_dict() for r in results],
                        "job_id": job.job_id,
                        "goal": job.goal,
                    }
                    _emit_capability_gap(job, results, out)
                return out

            tries = 0
            step_ok = False
            error_msg: Optional[str] = None
            detail_out: Any = ""
            error_status: Optional[str] = None

            # Deference (PROD): si hay interacción humana reciente, pausar antes de ejecutar acciones físicas.
            # Fail-closed: si no se puede leer la señal humana, tratamos como humano activo.
            try:
                if _deference_enabled() and env_rail() == "prod":
                    if action in _PHYSICAL_ACTIONS_REQUIRING_PERMISSION:
                        cfg = load_explore_policy(root_dir)
                        signal = read_human_signal(root_dir, policy=cfg)
                        active, reason = compute_human_active(
                            signal,
                            threshold_s=_deference_threshold_s(),
                            unknown_as_human=True,
                        )
                        if active:
                            resume_state = {
                                "resume_from_step_index": step_index,
                                "last_focus": last_focus,
                                "app_launch_counts": dict(app_launch_counts),
                                "step_results": [r.to_dict() for r in results],
                            }
                            question = (
                                "[DEFERENCE] Detectada actividad humana reciente (teclado/ratón). "
                                "Pausado en PROD para no pelear por el control. "
                                "Cuando termines, responde cualquier cosa para reanudar."
                            )
                            out = {
                                "ok": False,
                                "error": _DEFERENCE_STATUS,
                                "status": _DEFERENCE_STATUS,
                                "question": question,
                                "deference": {
                                    "step_index": step_index,
                                    "total_steps": total_steps,
                                    "step_id": step_id,
                                    "action": action,
                                    "reason": reason,
                                    "threshold_s": _deference_threshold_s(),
                                    "human_signal": signal,
                                },
                                "resume": resume_state,
                                "steps": [r.to_dict() for r in results],
                                "job_id": job.job_id,
                                "goal": job.goal,
                            }
                            return out
            except Exception:
                # Si algo crashea aquí, NO bloqueamos por excepción; dejamos que el runner siga.
                pass

            while tries <= retries and not step_ok:
                tries += 1
                error_status = None
                log.info(
                    "plan_runner.step mission=%s attempt=%s idx=%d/%d id=%s action=%s ts=%d status=start try=%d",
                    mission_id or "-",
                    attempt or "-",
                    step_index,
                    total_steps,
                    step_id,
                    action,
                    int(time.time()),
                    tries,
                )
                _record_plan_progress(
                    job.job_id,
                    mission_id,
                    total_steps,
                    step_index,
                    step_id,
                    action,
                    "running",
                    tries,
                    steps_completed=len(results),
                )
                try:
                    if action:
                        if (
                            action in _PHYSICAL_ACTIONS_REQUIRING_PERMISSION
                            and human_permission_gate_enabled is not None
                            and read_human_permission_status is not None
                            and human_permission_gate_enabled()
                        ):
                            perm = read_human_permission_status(root_dir)
                            if not bool(perm.get("ok", False)):
                                error_status = "human_permission_required"
                                detail_out = {
                                    "action": action,
                                    "status": error_status,
                                    "permission": perm,
                                }
                                raise ActuatorError("human_permission_required")

                        # Per-step consent (selectivo): solo aplica a acciones físicas y cuando el gate maestro está activo.
                        if (
                            action in _PHYSICAL_ACTIONS_REQUIRING_PERMISSION
                            and human_permission_gate_enabled is not None
                            and human_permission_gate_enabled()
                            and bool(step_consent_cfg.get("enabled", True))
                        ):
                            already_consented = bool(step_consents.get(step_id, False))
                            reasons: List[str] = []

                            # Risk level (ActionCatalog). Default: medium (fail-closed).
                            risk_level = "medium"
                            try:
                                if catalog is not None and hasattr(catalog, "get"):
                                    spec = catalog.get(action)
                                    if spec is not None:
                                        risk_level = _normalize_risk_level(getattr(spec, "risk_level", None))
                            except Exception:
                                risk_level = "medium"

                            if risk_level in (step_consent_cfg.get("risk_levels") or _STEP_CONSENT_RISK_LEVELS_DEFAULT):
                                reasons.append(f"risk:{risk_level}")

                            allowlist_actions = step_consent_cfg.get("allowlist_actions") or set()
                            allowlisted = action in allowlist_actions if isinstance(allowlist_actions, set) else False
                            if bool(step_consent_cfg.get("require_outside_allowlist", True)) and not allowlisted:
                                reasons.append("outside_allowlist")

                            if (
                                bool(step_consent_cfg.get("require_after_replan", True))
                                and after_replan
                                and not replan_consent_granted
                            ):
                                reasons.append("after_replan")

                            if (not already_consented) and reasons:
                                error_status = _STEP_CONSENT_STATUS
                                consent_req = {
                                    "step_index": step_index,
                                    "total_steps": total_steps,
                                    "step_id": step_id,
                                    "action": action,
                                    "risk_level": risk_level,
                                    "allowlisted": allowlisted,
                                    "after_replan": after_replan,
                                    "reasons": reasons,
                                }
                                resume_state = {
                                    "resume_from_step_index": step_index,
                                    "last_focus": last_focus,
                                    "app_launch_counts": dict(app_launch_counts),
                                    "step_results": [r.to_dict() for r in results],
                                }
                                detail_out = {
                                    "action": action,
                                    "status": error_status,
                                    "consent": consent_req,
                                    "resume": resume_state,
                                }
                                raise ActuatorError(_STEP_CONSENT_STATUS)
                        if action == "app.launch":
                            if not allow_app_launch:
                                raise ActuatorError("app_launch_disabled_by_policy")
                            proc = _get_arg(step, "process")
                            path = _get_arg(step, "path")
                            extra_args = _get_arg(step, "args")
                            if extra_args is not None and not isinstance(extra_args, list):
                                extra_args = None
                            if extra_args:
                                extra_args = [str(arg) for arg in extra_args if arg is not None]
                            if not proc and not path:
                                raise ActuatorError("app.launch requiere process o path")
                            if proc:
                                proc = str(proc).strip()
                                if not path and "." not in proc.lower():
                                    proc = f"{proc}.exe"
                                if any(sep in proc for sep in ("/", "\\")):
                                    proc = _normalize_windows_path(proc)
                            if not path and proc:
                                hint = APP_PATH_HINTS.get(str(proc).lower())
                                if hint:
                                    path = hint
                            if path:
                                path = _normalize_windows_path(path)
                            exe_name = os.path.basename(path or proc or "").lower()
                            total_launches = sum(app_launch_counts.values())
                            next_total = total_launches + 1
                            if max_launches_global and next_total > max_launches_global:
                                raise ActuatorError(f"app_launch_limit_exceeded:total>{max_launches_global}")
                            per_limit = None
                            if exe_name and exe_name in per_app_limits:
                                try:
                                    per_limit = int(per_app_limits.get(exe_name, {}).get("max_launches_per_mission"))
                                except Exception:
                                    per_limit = None
                            if per_limit:
                                next_count = app_launch_counts[exe_name] + 1
                                if next_count > per_limit:
                                    raise ActuatorError(f"app_launch_limit_exceeded:{exe_name}>{per_limit}")
                            # Idempotencia: intentar enfocar antes de lanzar
                            focused_existing = False
                            if proc:
                                try:
                                    res_focus = driver.window_focus(process=proc, title_contains=None)
                                    detail_out = {
                                        "action": "app.launch",
                                        "skipped_launch": True,
                                        "launch_count": 0,
                                        "focus_result": res_focus,
                                    }
                                    last_focus = {
                                        "process": res_focus.get("process") or proc,
                                        "title": res_focus.get("title"),
                                        "title_contains": res_focus.get("title"),
                                    }
                                    step_ok = True
                                    focused_existing = True
                                    log.info("plan_runner: reused existing %s window, no launch", proc)
                                except Exception:
                                    focused_existing = False
                            if step_ok and focused_existing:
                                continue
                            resp = driver.app_launch(process=proc, path=path, args=extra_args)
                            if not resp.get("ok"):
                                raise ActuatorError(resp.get("error") or "app_launch_failed")
                            app_launch_counts[exe_name or "unknown"] += 1
                            detail_out = {
                                "action": "app.launch",
                                "process": proc,
                                "path": path,
                                "args": extra_args,
                                "launch_count": app_launch_counts.get(exe_name or "unknown"),
                                "skipped_launch": focused_existing,
                                "result": resp,
                            }
                            # Después de lanzar, asumimos expectativa de foco en ese proceso
                            last_focus = {
                                "process": proc,
                                "title": None,
                                "title_contains": None,
                            }
                            step_ok = True
                            continue
                        if action == "keyboard.type":
                            _ensure_foreground(driver, last_focus)
                            text = _get_arg(step, "text") or goal
                            submit = bool(_get_arg(step, "submit", False))
                            if not text:
                                raise ActuatorError("keyboard.type requiere text")
                            resp = driver.keyboard_type(text=text, submit=submit)
                            detail_out = {"action": "keyboard.type", "text": text, "submit": submit, "result": resp}
                            step_ok = True
                            continue
                        if action == "window.focus":
                            title_contains = _get_arg(step, "title_contains")
                            proc = _get_arg(step, "process")
                            if not title_contains and not proc and last_focus:
                                title_contains = last_focus.get("title_contains") or last_focus.get("title")
                                proc = last_focus.get("process")
                            if not title_contains and not proc:
                                raise ActuatorError("window.focus requiere title_contains o process")
                            res = driver.window_focus(process=proc, title_contains=title_contains)
                            if not res.get("ok"):
                                raise ActuatorError(res.get("error") or "window_focus_failed")
                            detail_out = {"action": "window.focus", "process": proc, "title_contains": title_contains, "result": res}
                            last_focus = {
                                "process": proc or res.get("process"),
                                "title": res.get("title") or title_contains,
                                "title_contains": title_contains,
                            }
                            step_ok = True
                            continue
                        if action == "ui.inspect":
                            title_contains = _get_arg(step, "title_contains")
                            res = inspect_ui_tree(title_regex=title_contains)
                            if not res.get("ok"):
                                raise ActuatorError(res.get("error") or "ui.inspect_failed")
                            last_focus = {
                                "process": res.get("process"),
                                "title": res.get("title") or title_contains,
                                "title_contains": title_contains,
                            }
                            detail_out = {"action": "ui.inspect", "title_contains": title_contains, "result": res}
                            step_ok = True
                            continue
                        if action == "keyboard.hotkey":
                            keys = _get_arg(step, "keys") or []
                            if isinstance(keys, str):
                                keys = [keys]
                            if not keys:
                                raise ActuatorError("keyboard.hotkey requiere keys")
                            _ensure_foreground(driver, last_focus)
                            resp = driver.keyboard_hotkey(*[str(k) for k in keys])
                            detail_out = {"action": "keyboard.hotkey", "keys": keys, "result": resp}
                            step_ok = True
                            continue
                        if action == "sequence.save_notepad_v2":
                            _ensure_foreground(driver, last_focus, allow_save_dialog=True)
                            raw_target = step.get("target_path")
                            target = None
                            if raw_target:
                                try:
                                    expanded = os.path.expandvars(str(raw_target))
                                except Exception:
                                    expanded = str(raw_target)
                                if "%USERPROFILE%" in str(raw_target) and "%USERPROFILE%" in expanded:
                                    userprofile = _resolve_userprofile()
                                    if userprofile:
                                        expanded = str(raw_target).replace("%USERPROFILE%", userprofile)
                                target = _normalize_windows_path(expanded)

                            sent_ctrl_g = False
                            sent_ctrl_s = False

                            # Forzar guardar: primero Ctrl+G (ES), fallback Ctrl+S
                            try:
                                driver.keyboard_hotkey("ctrl", "g", ignore_safety=True)
                                sent_ctrl_g = True
                            except Exception as exc:
                                log.warning("plan_runner: save_notepad_v2 ctrl+g failed: %s", exc)
                            time.sleep(0.35)
                            try:
                                fg_tmp = driver.get_active_window()
                            except Exception:
                                fg_tmp = {}
                            title_tmp = str((fg_tmp or {}).get("title") or "").lower()
                            proc_tmp = str((fg_tmp or {}).get("process") or "").lower()
                            if not (any(tok in title_tmp for tok in SAVE_DIALOG_TOKENS) and proc_tmp != "notepad.exe"):
                                try:
                                    driver.keyboard_hotkey("ctrl", "s", ignore_safety=True)
                                    sent_ctrl_s = True
                                except Exception as exc:
                                    log.warning("plan_runner: save_notepad_v2 ctrl+s fallback failed: %s", exc)
                                time.sleep(0.35)
                            log.info(
                                "plan_runner: save_notepad_v2 hotkeys sent ctrl_g=%s ctrl_s_fallback=%s",
                                sent_ctrl_g,
                                sent_ctrl_s,
                            )
                            dialog_present = False
                            dlg_deadline = time.time() + float(step.get("dialog_wait_s", 2.0))
                            dlg_fg: Dict[str, Any] = {}
                            while time.time() < dlg_deadline:
                                try:
                                    dlg_fg = driver.get_active_window()
                                except Exception:
                                    dlg_fg = {}
                                title = str((dlg_fg or {}).get("title") or "").lower()
                                proc_name = str((dlg_fg or {}).get("process") or "").lower()
                                if any(tok in title for tok in SAVE_DIALOG_TOKENS) and proc_name != "notepad.exe":
                                    dialog_present = True
                                    break
                                time.sleep(0.1)

                            if not dialog_present:
                                detail_out = {
                                    "action": "sequence.save_notepad_v2",
                                    "status": "step_failed",
                                    "dialog_present": False,
                                    "error": "save_dialog_not_found",
                                }
                                log.warning("plan_runner: save_notepad_v2 dialog not found, skipping write")
                                raise ActuatorError("save_dialog_not_found")

                            try:
                                if target:
                                    driver.keyboard_paste(target, submit=False)
                                    time.sleep(0.2)
                                driver.keyboard_hotkey("enter", ignore_safety=True)
                                time.sleep(0.35)
                                overwrite_dialog = False
                                try:
                                    fg_after = driver.get_active_window()
                                except Exception:
                                    fg_after = {}
                                title_after = str((fg_after or {}).get("title") or "").lower()
                                proc_after = str((fg_after or {}).get("process") or "").lower()
                                overwrite_tokens = (
                                    "confirmar guardado como",
                                    "confirmar guardar como",
                                    "confirmar reemplazo",
                                    "confirm save as",
                                    "confirm save",
                                    "overwrite",
                                    "replace file",
                                )
                                if any(tok in title_after for tok in overwrite_tokens) and proc_after != "notepad.exe":
                                    overwrite_dialog = True
                                    try:
                                        driver.keyboard_hotkey("enter", ignore_safety=True)
                                        time.sleep(0.35)
                                    except Exception as exc:
                                        log.warning("plan_runner: save_notepad_v2 overwrite enter failed: %s", exc)
                                    try:
                                        fg_check = driver.get_active_window()
                                    except Exception:
                                        fg_check = {}
                                    title_check = str((fg_check or {}).get("title") or "").lower()
                                    proc_check = str((fg_check or {}).get("process") or "").lower()
                                    if any(tok in title_check for tok in overwrite_tokens) and proc_check != "notepad.exe":
                                        log.warning("plan_runner: save_notepad_v2 overwrite dialog still present")
                                        raise ActuatorError("overwrite_confirm_failed")

                                log.info("plan_runner: save_notepad_v2 overwrite_dialog=%s", overwrite_dialog)

                                # Validar que estamos de vuelta en el editor
                                try:
                                    fg_final = driver.get_active_window()
                                except Exception:
                                    fg_final = {}
                                proc_final = str((fg_final or {}).get("process") or "").lower()
                                title_final = str((fg_final or {}).get("title") or "").lower()
                                if not (proc_final == "notepad.exe" or "notepad" in title_final or "bloc de notas" in title_final):
                                    raise ActuatorError("editor_not_refocused_after_save")

                                detail_out = {
                                    "action": "sequence.save_notepad_v2",
                                    "dialog_present": True,
                                    "overwrite_dialog": overwrite_dialog,
                                    "path": target,
                                }
                                step_ok = True
                                continue
                            except Exception as exc:
                                raise ActuatorError(f"save_dialog_write_failed:{exc}")
                        if action == "sequence.save_notepad_v3":
                            mode = str(step.get("mode") or "save_or_overwrite").lower()
                            raw_target = step.get("target_path")
                            if not raw_target:
                                raise ActuatorError("target_path_required")
                            try:
                                expanded = os.path.expandvars(str(raw_target))
                            except Exception:
                                expanded = str(raw_target)
                            if "%USERPROFILE%" in str(raw_target) and "%USERPROFILE%" in expanded:
                                userprofile = _resolve_userprofile()
                                if userprofile:
                                    expanded = str(raw_target).replace("%USERPROFILE%", userprofile)
                            target = _normalize_windows_path(expanded) or expanded

                            def _is_notepad_window(win: Dict[str, Any]) -> bool:
                                title = str((win or {}).get("title") or "").lower()
                                proc = str((win or {}).get("process") or "").lower()
                                return ("notepad" in title) or ("notepad.exe" in proc) or ("bloc de notas" in title)

                            try:
                                fg0 = driver.get_active_window()
                            except Exception as exc:
                                raise ActuatorError(f"cannot_read_active_window:{exc}")
                            if not _is_notepad_window(fg0):
                                raise ActuatorError("wrong_active_window:notepad")

                            steps_local: List[str] = []
                            dialog_present = False
                            overwrite_dialog = None

                            def _detect_save_dialog(timeout_s: float = 1.5) -> Tuple[bool, Dict[str, Any]]:
                                deadline = time.time() + timeout_s
                                last_fg: Dict[str, Any] = {}
                                while time.time() < deadline:
                                    try:
                                        last_fg = driver.get_active_window()
                                    except Exception:
                                        last_fg = {}
                                    title = str((last_fg or {}).get("title") or "").lower()
                                    proc = str((last_fg or {}).get("process") or "").lower()
                                    if any(tok in title for tok in SAVE_DIALOG_TOKENS) or ("save as" in title):
                                        # Si el proceso no es notepad o el título marca guardar, aceptamos
                                        if proc != "notepad.exe" or title:
                                            return True, last_fg
                                    time.sleep(0.1)
                                return False, last_fg

                            def _detect_overwrite_dialog(timeout_s: float = 2.0) -> Tuple[bool, Dict[str, Any]]:
                                overwrite_tokens = (
                                    "confirmar guardado como",
                                    "confirmar reemplazo",
                                    "confirmar guardar como",
                                    "confirmar guardado",
                                    "confirm save as",
                                    "confirm save",
                                    "overwrite",
                                    "replace file",
                                )
                                deadline = time.time() + timeout_s
                                last_fg: Dict[str, Any] = {}
                                while time.time() < deadline:
                                    try:
                                        last_fg = driver.get_active_window()
                                    except Exception:
                                        last_fg = {}
                                    title = str((last_fg or {}).get("title") or "").lower()
                                    proc = str((last_fg or {}).get("process") or "").lower()
                                    if any(tok in title for tok in overwrite_tokens) and proc != "notepad.exe":
                                        return True, last_fg
                                    time.sleep(0.1)
                                return False, last_fg

                            def _focus_save_dialog(timeout_s: float = 1.5) -> Tuple[bool, Dict[str, Any]]:
                                save_tokens = tuple(SAVE_DIALOG_TOKENS) + (
                                    "guardar como",
                                    "guardar cambios",
                                    "save as",
                                    "save changes",
                                )
                                deadline = time.time() + timeout_s
                                last_fg: Dict[str, Any] = {}
                                while time.time() < deadline:
                                    for tok in save_tokens:
                                        try:
                                            focus_res = driver.window_focus(title_contains=tok)
                                        except Exception:
                                            focus_res = None
                                        if focus_res:
                                            last_fg = focus_res
                                            try:
                                                active_after = driver.get_active_window()
                                                if active_after:
                                                    last_fg = active_after
                                            except Exception:
                                                pass
                                            return True, last_fg
                                    time.sleep(0.1)
                                return False, last_fg

                            def _focus_overwrite_dialog(timeout_s: float = 1.5) -> Tuple[bool, Dict[str, Any]]:
                                overwrite_tokens = (
                                    "confirmar guardado como",
                                    "confirmar guardar como",
                                    "confirmar guardado",
                                    "confirmar reemplazo",
                                    "confirm save as",
                                    "confirm save",
                                    "overwrite",
                                    "replace file",
                                )
                                deadline = time.time() + timeout_s
                                last_fg: Dict[str, Any] = {}
                                while time.time() < deadline:
                                    for tok in overwrite_tokens:
                                        try:
                                            focus_res = driver.window_focus(title_contains=tok)
                                        except Exception:
                                            focus_res = None
                                        if focus_res:
                                            last_fg = focus_res
                                            try:
                                                active_after = driver.get_active_window()
                                                if active_after:
                                                    last_fg = active_after
                                            except Exception:
                                                pass
                                            return True, last_fg
                                    time.sleep(0.1)
                                return False, last_fg

                            # Abrir diálogo de guardado: Ctrl+G primero, luego Ctrl+S si no aparece
                            _ensure_foreground(driver, last_focus, allow_save_dialog=True)
                            try:
                                driver.keyboard_hotkey("ctrl", "g", ignore_safety=True)
                                steps_local.append("hotkey_ctrl_g")
                            except Exception as exc:
                                log.warning("save_notepad_v3: ctrl+g failed: %s", exc)
                            time.sleep(0.35)
                            dialog_present, dlg_fg = _detect_save_dialog(timeout_s=float(step.get("dialog_wait_s", 1.5)))
                            if not dialog_present:
                                try:
                                    driver.keyboard_hotkey("ctrl", "s", ignore_safety=True)
                                    steps_local.append("hotkey_ctrl_s_fallback")
                                except Exception as exc:
                                    log.warning("save_notepad_v3: ctrl+s fallback failed: %s", exc)
                                time.sleep(0.35)
                                dialog_present, dlg_fg = _detect_save_dialog(timeout_s=float(step.get("dialog_wait_s", 1.5)))
                            if not dialog_present:
                                try:
                                    driver.keyboard_hotkey("ctrl", "shift", "s", ignore_safety=True)
                                    steps_local.append("hotkey_ctrl_shift_s_fallback")
                                except Exception as exc:
                                    log.warning("save_notepad_v3: ctrl+shift+s fallback failed: %s", exc)
                                time.sleep(0.4)
                                dialog_present, dlg_fg = _detect_save_dialog(timeout_s=float(step.get("dialog_wait_s", 1.5)))
                            if not dialog_present:
                                try:
                                    driver.keyboard_hotkey("f12", ignore_safety=True)
                                    steps_local.append("hotkey_f12_fallback")
                                except Exception as exc:
                                    log.warning("save_notepad_v3: f12 fallback failed: %s", exc)
                                time.sleep(0.45)
                                dialog_present, dlg_fg = _detect_save_dialog(timeout_s=float(step.get("dialog_wait_s", 1.5)))
                            if not dialog_present:
                                detail_out = {
                                    "action": "sequence.save_notepad_v3",
                                    "dialog_present": False,
                                    "path": target,
                                    "steps": steps_local,
                                    "error": "save_dialog_not_found",
                                    "evidence": {"last_fg": dlg_fg},
                                }
                                log.warning("save_notepad_v3: save dialog not found")
                                raise ActuatorError("save_dialog_not_found")

                            # Escribir la ruta y confirmar
                            steps_local.append("save_dialog_detected")
                            try:
                                driver.keyboard_paste(target, submit=False)
                                steps_local.append("write_path")
                                time.sleep(0.2)
                                driver.keyboard_hotkey("enter", ignore_safety=True)
                                steps_local.append("confirm_save_enter")
                            except Exception as exc:
                                raise ActuatorError(f"save_dialog_write_failed:{exc}")

                            time.sleep(0.4)
                            overwrite_dialog = False
                            ow_present, ow_fg = _detect_overwrite_dialog(timeout_s=float(step.get("overwrite_wait_s", 1.0)))
                            if ow_present:
                                overwrite_dialog = True
                                if mode == "save_new":
                                    try:
                                        driver.keyboard_hotkey("esc", ignore_safety=True)
                                        steps_local.append("overwrite_cancel")
                                    except Exception:
                                        steps_local.append("overwrite_cancel_failed")
                                    detail_out = {
                                        "action": "sequence.save_notepad_v3",
                                        "dialog_present": True,
                                        "overwrite_dialog": True,
                                        "path": target,
                                        "steps": steps_local,
                                        "error": "overwrite_not_allowed",
                                        "evidence": {"overwrite_window": ow_fg},
                                    }
                                    raise ActuatorError("overwrite_not_allowed")
                                try:
                                    driver.keyboard_hotkey("enter", ignore_safety=True)
                                    steps_local.append("overwrite_confirm_enter")
                                    time.sleep(0.4)
                                except Exception as exc:
                                    raise ActuatorError(f"overwrite_confirm_failed:{exc}")
                                ow_still, ow_fg2 = _detect_overwrite_dialog(timeout_s=float(step.get("overwrite_wait_s", 1.0)))
                                if ow_still:
                                    try:
                                        driver.keyboard_hotkey("enter", ignore_safety=True)
                                        steps_local.append("overwrite_confirm_enter_retry")
                                        time.sleep(0.8)
                                        ow_still, ow_fg2 = _detect_overwrite_dialog(timeout_s=float(step.get("overwrite_wait_s", 1.0)))
                                    except Exception:
                                        pass
                                if ow_still:
                                    detail_out = {
                                        "action": "sequence.save_notepad_v3",
                                        "dialog_present": True,
                                        "overwrite_dialog": True,
                                        "path": target,
                                        "steps": steps_local,
                                        "error": "overwrite_confirm_failed",
                                        "evidence": {"overwrite_window": ow_fg2},
                                    }
                                    raise ActuatorError("overwrite_confirm_failed")

                            # Validar foreground final
                            try:
                                fg_final = driver.get_active_window()
                            except Exception:
                                fg_final = {}

                            title_final = str((fg_final or {}).get("title") or "").lower()
                            proc_final = str((fg_final or {}).get("process") or "").lower()

                            save_tokens_final = tuple(SAVE_DIALOG_TOKENS) + ("guardar como",)
                            if any(tok in title_final for tok in save_tokens_final) and proc_final != "notepad.exe":
                                try:
                                    driver.keyboard_hotkey("alt", "g", ignore_safety=True)
                                    steps_local.append("save_dialog_confirm_alt_g")
                                    time.sleep(0.8)
                                    fg_final = driver.get_active_window()
                                    title_final = str((fg_final or {}).get("title") or "").lower()
                                    proc_final = str((fg_final or {}).get("process") or "").lower()
                                except Exception:
                                    try:
                                        driver.keyboard_hotkey("enter", ignore_safety=True)
                                        steps_local.append("save_dialog_confirm_enter_final")
                                        time.sleep(0.8)
                                        fg_final = driver.get_active_window()
                                        title_final = str((fg_final or {}).get("title") or "").lower()
                                        proc_final = str((fg_final or {}).get("process") or "").lower()
                                    except Exception:
                                        pass

                            overwrite_tokens_final = (
                                "confirmar guardado como",
                                "confirmar guardar como",
                                "confirmar reemplazo",
                                "confirm save as",
                                "confirm save",
                                "overwrite",
                                "replace file",
                            )
                            title_final = str((fg_final or {}).get("title") or "").lower()
                            proc_final = str((fg_final or {}).get("process") or "").lower()
                            over_in_title = any(tok in title_final for tok in overwrite_tokens_final)
                            save_in_title = any(tok in title_final for tok in save_tokens_final)

                            if over_in_title and proc_final != "notepad.exe":
                                overwrite_dialog = True
                                if mode == "save_or_overwrite":
                                    try:
                                        driver.keyboard_hotkey("alt", "s", ignore_safety=True)
                                        steps_local.append("overwrite_confirm_alt_s")
                                        time.sleep(0.5)
                                    except Exception:
                                        pass
                                    try:
                                        driver.keyboard_hotkey("enter", ignore_safety=True)
                                        steps_local.append("overwrite_confirm_enter_final")
                                        time.sleep(1.0)
                                        fg_final = driver.get_active_window()
                                        title_final = str((fg_final or {}).get("title") or "").lower()
                                        proc_final = str((fg_final or {}).get("process") or "").lower()
                                        over_in_title = any(tok in title_final for tok in overwrite_tokens_final)
                                        save_in_title = any(tok in title_final for tok in save_tokens_final)
                                    except Exception as exc:
                                        raise ActuatorError(f"overwrite_confirm_failed:{exc}")
                                else:
                                    detail_out = {
                                        "action": "sequence.save_notepad_v3",
                                        "dialog_present": True,
                                        "overwrite_dialog": True,
                                        "path": target,
                                        "steps": steps_local,
                                        "error": "overwrite_not_allowed",
                                        "evidence": {"fg": fg_final},
                                    }
                                    raise ActuatorError("overwrite_not_allowed")
                                if over_in_title and proc_final != "notepad.exe":
                                    detail_out = {
                                        "action": "sequence.save_notepad_v3",
                                        "dialog_present": True,
                                        "overwrite_dialog": True,
                                        "path": target,
                                        "steps": steps_local,
                                        "error": "overwrite_confirm_failed",
                                        "evidence": {"fg": fg_final},
                                    }
                                    raise ActuatorError("overwrite_confirm_failed")

                            if save_in_title and proc_final != "notepad.exe":
                                try:
                                    driver.keyboard_hotkey("enter", ignore_safety=True)
                                    steps_local.append("save_dialog_enter_retry")
                                    time.sleep(0.8)
                                    fg_final = driver.get_active_window()
                                    title_final = str((fg_final or {}).get("title") or "").lower()
                                    proc_final = str((fg_final or {}).get("process") or "").lower()
                                    save_in_title = any(tok in title_final for tok in save_tokens_final)
                                except Exception:
                                    pass
                                if save_in_title and proc_final != "notepad.exe":
                                    detail_out = {
                                        "action": "sequence.save_notepad_v3",
                                        "dialog_present": True,
                                        "overwrite_dialog": overwrite_dialog,
                                        "path": target,
                                        "steps": steps_local,
                                        "error": "save_dialog_not_closed",
                                        "evidence": {"fg": fg_final},
                                    }
                                    raise ActuatorError("save_dialog_not_closed")

                            if not _is_notepad_window(fg_final):
                                detail_out = {
                                    "action": "sequence.save_notepad_v3",
                                    "dialog_present": True,
                                    "overwrite_dialog": overwrite_dialog,
                                    "path": target,
                                    "steps": steps_local,
                                    "error": "editor_not_refocused_after_save",
                                    "evidence": {"fg": fg_final},
                                }
                                raise ActuatorError("editor_not_refocused_after_save")

                            detail_out = {
                                "action": "sequence.save_notepad_v3",
                                "dialog_present": True,
                                "overwrite_dialog": overwrite_dialog,
                                "path": target,
                                "steps": steps_local,
                                "evidence": {"final_fg": fg_final},
                            }
                            last_focus = {
                                "process": (fg_final or {}).get("process") or "notepad.exe",
                                "title": (fg_final or {}).get("title"),
                                "title_contains": (fg_final or {}).get("title"),
                            }
                            step_ok = True
                            continue
                        if action == "sequence.append_diary_entry":
                            _ensure_foreground(driver, last_focus)
                            text = step.get("text")
                            resp = driver.append_diary_entry(text=text, file_path=step.get("target_path"))
                            detail_out = f"append_diary_entry:{resp}"
                            step_ok = bool((resp or {}).get("ok", False))
                            if step_ok:
                                continue
                            raise ActuatorError(resp.get("error") or "append_diary_entry_failed")
                        if action == "sequence.close_notepad":
                            resp = close_editor_safe(
                                driver,
                                action=step.get("action_mode") or "dont_save",
                                file_path=step.get("target_path"),
                                timeout_s=float(step.get("timeout_s", 6.0)),
                                dialog_wait_s=float(step.get("dialog_wait_s", 1.5)),
                            )
                            detail_out = resp
                            step_ok = bool(resp.get("ok"))
                            if step_ok:
                                continue
                            raise ActuatorError(resp.get("error") or "close_editor_failed")
                        if action == "editor.close_safe":
                            resp = close_editor_safe(
                                driver,
                                action=step.get("action_mode") or step.get("mode") or "dont_save",
                                file_path=step.get("target_path") or step.get("file_path"),
                                timeout_s=float(step.get("timeout_s", 6.0)),
                                dialog_wait_s=float(step.get("dialog_wait_s", 1.5)),
                            )
                            detail_out = resp
                            step_ok = bool(resp.get("ok"))
                            if step_ok:
                                continue
                            raise ActuatorError(resp.get("error") or "close_editor_failed")
                        if action == "vision.llm_click":
                            _ensure_foreground(driver, last_focus)
                            target = step.get("target") or step.get("target_text") or goal
                            role = step.get("role") or "button"
                            min_conf = float(step.get("min_confidence", 0.6))
                            try_ocr_first = bool(step.get("try_ocr_first", True))
                            res: VisionClickResult = vision_llm_click(
                                target_text=target,
                                role=role,
                                min_confidence=min_conf,
                                try_ocr_first=try_ocr_first,
                            )
                            detail_out = {
                                "success": res.success,
                                "chosen_id": res.chosen_id,
                                "confidence": res.confidence,
                                "reason": res.reason,
                                "path": res.path,
                                "error": res.error,
                                "alternatives": res.alternatives,
                            }
                            if not res.success:
                                raise ActuatorError(res.error or "vision_llm_click_failed")
                            step_ok = True
                            continue
                        if action == "desktop.isolate_active_window":
                            try:
                                initial = driver.get_active_window()
                            except Exception:
                                initial = {}
                            driver.keyboard_hotkey("win", "d")
                            title = (initial or {}).get("title")
                            proc_name = (initial or {}).get("process")
                            focus_res = None
                            try:
                                if title:
                                    focus_res = driver.window_focus(title_contains=title)
                                elif proc_name:
                                    focus_res = driver.window_focus(process=proc_name)
                            except Exception:
                                focus_res = None
                            detail_out = {
                                "action": "desktop.isolate_active_window",
                                "initial_window": initial,
                                "refocus": focus_res,
                            }
                            last_focus = {
                                "process": (focus_res or initial or {}).get("process"),
                                "title": (focus_res or initial or {}).get("title"),
                                "title_contains": (focus_res or initial or {}).get("title"),
                            }
                            step_ok = True
                            continue
                        if action == "shell.run":
                            cmd = step.get("cmd") or step.get("command")
                            if not cmd:
                                raise ActuatorError("shell_run_missing_cmd")
                            try:
                                proc = subprocess.run(
                                    cmd,
                                    shell=True,
                                    capture_output=True,
                                    text=True,
                                    timeout=float(step.get("timeout_s", 60)),
                                )
                                detail_out = {
                                    "action": "shell.run",
                                    "cmd": cmd,
                                    "returncode": proc.returncode,
                                    "stdout": proc.stdout,
                                    "stderr": proc.stderr,
                                }
                                step_ok = proc.returncode == 0
                                if not step_ok:
                                    error_msg = f"shell_run_rc_{proc.returncode}"
                                continue
                            except Exception as exc:
                                raise ActuatorError(f"shell_run_failed:{exc}")
                        if action == "ops.fix_ports_sessions":
                            if fix_ports_sessions is None:
                                raise ActuatorError("ops_fix_ports_sessions_unavailable")
                            args = step.get("args") if isinstance(step.get("args"), dict) else {}
                            ports = args.get("ports")
                            verify_only = bool(args.get("verify_only", False))
                            timeout_s = int(args.get("timeout_s", 60)) if args.get("timeout_s") is not None else 60
                            detail_out = fix_ports_sessions(
                                root_dir=root_dir,
                                ports=ports,
                                verify_only=verify_only,
                                timeout_s=timeout_s,
                            )
                            step_ok = bool(detail_out.get("ok"))
                            if not step_ok:
                                raise ActuatorError("ops_fix_ports_sessions_failed")
                            continue
                        if action == "os.probe_standby":
                            data = _run_os_probe()
                            detail_out = {"action": "os.probe_standby", "report": data}
                            step_ok = bool(data.get("ok"))
                            if not step_ok:
                                raise ActuatorError(data.get("error") or "os_probe_failed")
                            continue
                        if action == "probe_driver":
                            args = step.get("args") or {}
                            out_dir = args.get("out_dir") if isinstance(args, dict) else None
                            report = _run_driver_probe(
                                driver,
                                out_dir=str(out_dir) if isinstance(out_dir, str) and out_dir.strip() else None,
                                include_openapi=bool(args.get("include_openapi", True)) if isinstance(args, dict) else True,
                                include_shot=bool(args.get("include_shot", True)) if isinstance(args, dict) else True,
                                include_screenshot=bool(args.get("include_screenshot", True)) if isinstance(args, dict) else True,
                                include_window_inspect=bool(args.get("include_window_inspect", True)) if isinstance(args, dict) else True,
                            )
                            detail_out = {"action": "probe_driver", "report": report}
                            # Success = evidencia persistida (no "driver healthy").
                            rp = report.get("report_path")
                            step_ok = False
                            if isinstance(rp, str) and rp:
                                try:
                                    step_ok = Path(rp).exists()
                                except Exception:
                                    step_ok = True
                            if not step_ok:
                                raise ActuatorError("probe_driver_no_report")
                            continue
                        if action == "models.discover":
                            args = step.get("args") or {}
                            provider_filter = args.get("provider") or step.get("provider")
                            vision_only = bool(args.get("vision_only", step.get("vision_only", False)))
                            mods: List[ModelInfo] = list_vision_models(provider_filter) if vision_only else discover_models(provider_filter)
                            detail_out = {"models": [m.__dict__ for m in mods]}
                            step_ok = True
                            continue
                        if action == "lab_web_discovery":
                            args = step.get("args") or {}
                            topic = args.get("topic") or step.get("topic") or goal
                            days = args.get("days") if isinstance(args, dict) else None
                            strict = bool(args.get("strict", step.get("strict", False)))
                            max_results = args.get("max_results") or step.get("max_results") or 8
                            try:
                                from agency.lab_web import run_lab_web
                            except Exception as exc:
                                raise ActuatorError(f"lab_web_import_failed:{exc}")
                            try:
                                result = run_lab_web(
                                    str(topic),
                                    days=int(days) if days is not None else None,
                                    strict=bool(strict),
                                    max_results=int(max_results),
                                    allow_prod=False,
                                )
                            except Exception as exc:
                                detail_out = {"action": "lab_web_discovery", "error": str(exc)}
                                raise ActuatorError(f"lab_web_failed:{exc}")
                            detail_out = {"action": "lab_web_discovery", "result": result}
                            step_ok = bool(result.get("ok"))
                            if not step_ok:
                                raise ActuatorError("lab_web_no_results")
                            continue
                        if action == "lab_codex_web":
                            args = step.get("args") or {}
                            topic = args.get("topic") or step.get("topic") or goal
                            prompt = args.get("prompt") or step.get("prompt") or topic
                            allowlist = (
                                args.get("allowlist_domains")
                                or step.get("allowlist_domains")
                                or args.get("allowlist")
                                or step.get("allowlist")
                            )
                            if isinstance(allowlist, str):
                                allowlist = [a.strip() for a in allowlist.split(",") if a.strip()]
                            if not isinstance(allowlist, list) or not allowlist:
                                raise ActuatorError("lab_codex_web_allowlist_required")
                            if env_rail() == "prod":
                                raise ActuatorError("lab_codex_web_blocked_in_prod")
                            days = args.get("days") if isinstance(args, dict) else None
                            max_results = args.get("max_results") or step.get("max_results") or 8
                            try:
                                from agency.lab_web import run_lab_web
                            except Exception as exc:
                                raise ActuatorError(f"lab_web_import_failed:{exc}")
                            try:
                                lab_result = run_lab_web(
                                    str(topic),
                                    days=int(days) if days is not None else None,
                                    strict=True,
                                    max_results=int(max_results),
                                    allowlist_domains=allowlist,
                                    strict_sources=True,
                                    allow_prod=False,
                                )
                            except Exception as exc:
                                detail_out = {"action": "lab_codex_web", "error": str(exc)}
                                raise ActuatorError(f"lab_web_failed:{exc}")
                            if lab_result.get("ok"):
                                detail_out = {
                                    "action": "lab_codex_web",
                                    "lab_web_result": lab_result,
                                    "skipped": True,
                                    "skip_reason": "official_sources_found",
                                }
                                step_ok = True
                                continue
                            warnings = lab_result.get("warnings") or []
                            if "no_sources_found" not in warnings:
                                detail_out = {
                                    "action": "lab_codex_web",
                                    "lab_web_result": lab_result,
                                    "skipped": True,
                                    "skip_reason": "lab_web_not_eligible",
                                }
                                step_ok = False
                                raise ActuatorError("lab_codex_web_not_eligible")

                            script = (
                                ROOT / "scripts" / "codex_web_exec.ps1"
                                if os.name == "nt"
                                else ROOT / "scripts" / "codex_web_exec.sh"
                            )
                            if not script.exists():
                                raise ActuatorError("lab_codex_web_wrapper_missing")
                            if os.name == "nt":
                                cmd = [
                                    "powershell.exe",
                                    "-NoProfile",
                                    "-ExecutionPolicy",
                                    "Bypass",
                                    "-File",
                                    str(script),
                                    "-Prompt",
                                    str(prompt),
                                ]
                            else:
                                cmd = ["bash", str(script), str(prompt)]
                            try:
                                proc = subprocess.run(
                                    cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=float(step.get("timeout_s", 180)),
                                )
                            except Exception as exc:
                                raise ActuatorError(f"lab_codex_web_exec_failed:{exc}")
                            stdout = (proc.stdout or "").strip()
                            stderr = (proc.stderr or "").strip()
                            if proc.returncode != 0:
                                detail_out = {
                                    "action": "lab_codex_web",
                                    "returncode": proc.returncode,
                                    "stderr": stderr,
                                    "stdout": stdout,
                                    "lab_web_result": lab_result,
                                }
                                raise ActuatorError(f"lab_codex_web_rc_{proc.returncode}")
                            parsed = None
                            if stdout:
                                try:
                                    parsed = json.loads(stdout)
                                except Exception:
                                    lines = [ln for ln in stdout.splitlines() if ln.strip()]
                                    if lines:
                                        try:
                                            parsed = json.loads(lines[-1])
                                        except Exception:
                                            parsed = None
                            if parsed is None:
                                detail_out = {
                                    "action": "lab_codex_web",
                                    "error": "invalid_json",
                                    "stdout": stdout,
                                    "stderr": stderr,
                                    "lab_web_result": lab_result,
                                }
                                raise ActuatorError("lab_codex_web_invalid_json")

                            run_dir = ROOT / "artifacts" / "lab_codex_web" / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                            run_dir.mkdir(parents=True, exist_ok=True)
                            out_path = run_dir / "result.json"
                            payload = {
                                "schema": "ajax.lab_codex_web.v1",
                                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "topic": topic,
                                "prompt": prompt,
                                "allowlist_domains": allowlist,
                                "lab_web_result": lab_result,
                                "codex_result": parsed,
                                "stdout": stdout,
                                "stderr": stderr,
                                "run_dir": str(run_dir),
                            }
                            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                            detail_out = {"action": "lab_codex_web", "result_path": str(out_path), "result": payload}
                            step_ok = True
                            continue

                    act.execute(goal=goal, strategy=strategy)
                    detail_out = "actuator"
                    step_ok = True
                except (DriverConnectionError, DriverTimeout) as exc:
                    error_msg = str(exc)
                    error_status = "driver_unreachable"
                    detail_out = detail_out or {
                        "action": action or "unknown",
                        "status": error_status,
                        "error": error_msg,
                    }
                    log.warning(
                        "plan_runner: mission=%s attempt=%s step=%s action=%s try=%d/%d driver_error=%s",
                        mission_id or "-",
                        attempt or "-",
                        step_id,
                        action,
                        tries,
                        retries,
                        error_msg,
                    )
                    if tries > retries:
                        break
                except (ActuatorError, WindowsDriverError) as exc:
                    error_msg = str(exc)
                    log.warning(
                        "plan_runner: mission=%s attempt=%s step=%s action=%s try=%d/%d error=%s",
                        mission_id or "-",
                        attempt or "-",
                        step_id,
                        action,
                        tries,
                        retries,
                        error_msg,
                    )
                    if error_status in {"human_permission_required", _STEP_CONSENT_STATUS}:
                        break
                    if tries > retries:
                        break
                    if on_fail == "fallback_cognitive" and strategy != Strategy.FORCE_COGNITIVE:
                        strategy = Strategy.FORCE_COGNITIVE
                        continue
                except Exception as exc:
                    error_msg = str(exc)
                    log.debug(
                        "plan_runner: mission=%s attempt=%s step=%s action=%s try=%d/%d unexpected_error=%s",
                        mission_id or "-",
                        attempt or "-",
                        step_id,
                        action,
                        tries,
                        retries,
                        error_msg,
                    )
                    if tries > retries:
                        break

            if step_ok:
                # Evidencia requerida (post-task)
                evidence, evidence_errors = _capture_evidence(driver, list(evidence_required) if isinstance(evidence_required, list) else [])
                if evidence or evidence_errors:
                    if not isinstance(detail_out, dict):
                        detail_out = {"detail": detail_out}
                    detail_out["evidence"] = evidence
                    if evidence_errors:
                        detail_out["evidence_errors"] = evidence_errors
                if evidence_errors:
                    step_ok = False
                    error_status = "evidence_missing"
                    error_msg = "evidence_missing:" + ";".join(evidence_errors[:4])
                    if not isinstance(detail_out, dict):
                        detail_out = {"detail": detail_out}
                    detail_out.setdefault("capability_family", "task.evidence_missing")

            if step_ok:
                # Gate duro: verificar EFE (success_spec) tras la tarea
                succ_es = _extract_expected_state(success_spec)
                task_efe_timeout = efe_timeout
                try:
                    if isinstance(success_spec, dict) and success_spec.get("timeout_s") is not None:
                        task_efe_timeout = float(success_spec.get("timeout_s") or task_efe_timeout)
                except Exception:
                    task_efe_timeout = efe_timeout
                efe_ok, delta = verify_efe(succ_es, driver=driver, timeout_s=task_efe_timeout) if succ_es else (False, None)
                if not efe_ok:
                    step_ok = False
                    error_status = "task_efe_mismatch"
                    error_msg = "efe_mismatch"
                    if not isinstance(detail_out, dict):
                        detail_out = {"detail": detail_out}
                    detail_out["efe_delta"] = delta
                    detail_out["task_intent"] = task_intent
                    detail_out.setdefault("action", action or "unknown")
                    detail_out.setdefault("capability_family", "task.efe_mismatch")

            if not step_ok:
                success = False
                if not detail_out:
                    detail_out = {"action": action or "unknown", "status": "step_failed"}
                if isinstance(detail_out, dict):
                    detail_out.setdefault("action", action or "unknown")
                step_status = error_status or "step_failed"
                if isinstance(detail_out, dict):
                    detail_out.setdefault("status", step_status)
                results.append(StepResult(step_id=step_id, ok=False, detail=detail_out, tries=tries, error=error_msg))
                summary_detail = _summarize_detail(detail_out)
                _record_plan_progress(
                    job.job_id,
                    mission_id,
                    total_steps,
                    step_index,
                    step_id,
                    action,
                    "failed",
                    tries,
                    error=error_msg,
                    detail=summary_detail,
                    steps_completed=len(results),
                )
                log.info(
                    "plan_runner.step mission=%s attempt=%s idx=%d/%d id=%s action=%s ts=%d status=fail error=%s tries=%d",
                    mission_id or "-",
                    attempt or "-",
                    step_index,
                    total_steps,
                    step_id,
                    action,
                    int(time.time()),
                    error_msg,
                    tries,
                )
                if on_fail == "abort":
                    out: Dict[str, Any] = {
                        "ok": False,
                        "error": error_msg or "step_failed",
                        "status": step_status,
                        "steps": [r.to_dict() for r in results],
                        "job_id": job.job_id,
                        "goal": job.goal,
                    }
                    if step_status == _STEP_CONSENT_STATUS and isinstance(detail_out, dict):
                        if isinstance(detail_out.get("consent"), dict):
                            out["consent"] = detail_out.get("consent")
                        if isinstance(detail_out.get("resume"), dict):
                            out["resume"] = detail_out.get("resume")
                    if isinstance(detail_out, dict) and detail_out.get("efe_delta") is not None:
                        out["efe_delta"] = detail_out.get("efe_delta")
                    if step_status not in {"human_permission_required", _STEP_CONSENT_STATUS}:
                        _emit_capability_gap(job, results, out)
                    return out
                log.warning(
                    "plan_runner: mission=%s attempt=%s step=%s failed but continuing (on_fail=%s): %s",
                    mission_id or "-",
                    attempt or "-",
                    step_id,
                    on_fail,
                    error_msg,
                )
            else:
                if not detail_out:
                    detail_out = "ok"
                results.append(StepResult(step_id=step_id, ok=True, detail=detail_out, tries=tries))
                summary_detail = _summarize_detail(detail_out)
                _record_plan_progress(
                    job.job_id,
                    mission_id,
                    total_steps,
                    step_index,
                    step_id,
                    action,
                    "ok",
                    tries,
                    detail=summary_detail,
                    steps_completed=len(results),
                )
                log.info(
                    "plan_runner.step mission=%s attempt=%s idx=%d/%d id=%s action=%s ts=%d status=ok tries=%d detail=%s",
                    mission_id or "-",
                    attempt or "-",
                    step_index,
                    total_steps,
                    step_id,
                    action,
                    int(time.time()),
                    tries,
                    summary_detail,
                )
    finally:
        _clear_plan_progress(job.job_id)

    result: Dict[str, Any] = {
        "ok": success,
        "steps": [r.to_dict() for r in results],
        "job_id": job.job_id,
        "goal": job.goal,
        "status": "success" if success else "step_failed",
    }

    # Verificación del Estado Final Esperado (EFE) si todo lo demás fue bien
    if success and expected_state:
        efe_ok, delta = verify_efe(expected_state, driver=driver, timeout_s=efe_timeout)
        if not efe_ok:
            result["ok"] = False
            result["status"] = "efe_mismatch"
            result["error"] = result.get("error") or "efe_mismatch"
            if delta:
                result["efe_delta"] = delta

    if not result.get("ok"):
        _emit_capability_gap(job, results, result)

    return result
