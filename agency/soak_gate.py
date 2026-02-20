from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agency.health_ttl import provider_status_ttl
from agency.process_utils import pid_running

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _utc_now(now_ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(now_ts or time.time())))


def _ts_label(now_ts: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(float(now_ts or time.time())))


def _normalize_rail(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    if val in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_iso_utc(raw: Any) -> Optional[float]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return float(datetime.fromisoformat(text).timestamp())
    except Exception:
        return None


def _to_ts(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return float(raw)
        except Exception:
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return _parse_iso_utc(text)
    return None


def _extract_ts(payload: Dict[str, Any], path: Path, *, keys: Optional[List[str]] = None) -> float:
    lookup = keys or []
    for key in lookup:
        ts = _to_ts(payload.get(key))
        if ts is not None:
            return float(ts)
    ts_utc = _parse_iso_utc(payload.get("ts_utc"))
    if ts_utc is not None:
        return float(ts_utc)
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return float(time.time())


def _latest_glob(root_dir: Path, pattern: str) -> Optional[Path]:
    candidates = [p for p in root_dir.glob(pattern) if p.exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _repo_markers_ok(root_dir: Path) -> bool:
    markers = [
        root_dir / "AGENTS.md",
        root_dir / "bin" / "ajaxctl",
        root_dir / "agency",
    ]
    return all(path.exists() for path in markers)


def _canonical_root_ok(root_dir: Path) -> bool:
    return root_dir.name == "ajax-kernel" and _repo_markers_ok(root_dir)


def _load_manifest(root_dir: Path) -> Dict[str, Any]:
    manifest_path = root_dir / "config" / "lab_org_manifest.yaml"
    if not manifest_path.exists():
        return {"micro_challenges": []}
    if yaml is not None:
        try:
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {"micro_challenges": []}


def _enabled_work_items(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = manifest.get("micro_challenges") if isinstance(manifest.get("micro_challenges"), list) else []
    enabled: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if bool(item.get("enabled", True)):
            enabled.append(item)
    return enabled


def _collect_lab_org_activity(root_dir: Path, *, now_ts: float, window_s: float) -> Dict[str, Any]:
    receipts = sorted((root_dir / "artifacts" / "lab_org").glob("*/receipt.json"))
    latest_ts = None
    latest_path = None
    latest_payload: Dict[str, Any] = {}
    latest_enqueued_ts = None
    enqueued_in_window = False
    for path in receipts:
        payload = _safe_read_json(path)
        if not payload:
            continue
        ts = _extract_ts(payload, path, keys=["ts"])
        if latest_ts is None or ts >= latest_ts:
            latest_ts = float(ts)
            latest_path = str(path)
            latest_payload = payload
        if bool(payload.get("enqueued")):
            if latest_enqueued_ts is None or ts > latest_enqueued_ts:
                latest_enqueued_ts = float(ts)
            if now_ts - float(ts) <= float(window_s):
                enqueued_in_window = True
    return {
        "latest_receipt_ts": latest_ts,
        "latest_receipt_path": latest_path,
        "latest_receipt": latest_payload if latest_payload else None,
        "latest_enqueued_ts": latest_enqueued_ts,
        "enqueued_in_window": enqueued_in_window,
    }


def _collect_completed_activity(root_dir: Path, *, now_ts: float, window_s: float) -> Dict[str, Any]:
    results = sorted((root_dir / "artifacts" / "lab" / "results").glob("result_*.json"))
    latest_completed_ts = None
    latest_path = None
    completed_in_window = False
    for path in results:
        payload = _safe_read_json(path)
        ts = _extract_ts(payload, path, keys=["completed_ts", "created_ts", "acknowledged_ts"])
        if latest_completed_ts is None or ts > latest_completed_ts:
            latest_completed_ts = float(ts)
            latest_path = str(path)
        if now_ts - float(ts) <= float(window_s):
            completed_in_window = True
    return {
        "latest_completed_ts": latest_completed_ts,
        "latest_result_path": latest_path,
        "completed_in_window": completed_in_window,
    }


def _worker_scheduler_signal(
    root_dir: Path,
    *,
    rail: str,
    now_ts: float,
    window_s: float,
    lab_activity: Dict[str, Any],
) -> Dict[str, Any]:
    if rail != "lab":
        return {
            "ok": True,
            "code": "ALIVE_SCHEDULER_NA",
            "actionable_hint": "",
            "detail": {"rail": rail},
        }

    heartbeat_path = root_dir / "artifacts" / "lab" / "heartbeat.json"
    pid_path = root_dir / "artifacts" / "lab" / "worker.pid"
    heartbeat = _safe_read_json(heartbeat_path)
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.exists() else None
    except Exception:
        pid = None
    running = bool(pid is not None and pid_running(int(pid)))
    hb_ts = _to_ts(heartbeat.get("ts")) if heartbeat else None
    hb_age = None if hb_ts is None else max(0.0, float(now_ts - float(hb_ts)))
    try:
        stale_threshold_s = float(os.getenv("AJAX_LAB_WORKER_STALE_SEC", "30") or "30")
    except Exception:
        stale_threshold_s = 30.0
    heartbeat_fresh = bool(running and hb_age is not None and hb_age <= stale_threshold_s)

    sched_ts = _to_ts(lab_activity.get("latest_receipt_ts"))
    scheduler_recent = bool(sched_ts is not None and (now_ts - float(sched_ts)) <= float(window_s))
    latest_receipt = lab_activity.get("latest_receipt") if isinstance(lab_activity.get("latest_receipt"), dict) else {}
    scheduler_running = str((latest_receipt or {}).get("lab_org_status") or "").upper() == "RUNNING"

    detail = {
        "running": running,
        "pid": pid,
        "heartbeat_ts": hb_ts,
        "heartbeat_age_s": hb_age,
        "stale_threshold_s": stale_threshold_s,
        "heartbeat_fresh": heartbeat_fresh,
        "scheduler_recent": scheduler_recent,
        "scheduler_running": scheduler_running,
        "latest_lab_org_receipt_path": lab_activity.get("latest_receipt_path"),
    }
    if not running:
        return {
            "ok": False,
            "code": "ALIVE_WORKER_NOT_RUNNING",
            "actionable_hint": "Run: python bin/ajaxctl lab start",
            "detail": detail,
        }
    if not heartbeat_fresh:
        return {
            "ok": False,
            "code": "ALIVE_WORKER_HEARTBEAT_STALE",
            "actionable_hint": "Restart worker: python bin/ajaxctl lab restart",
            "detail": detail,
        }
    if not scheduler_recent:
        return {
            "ok": False,
            "code": "ALIVE_SCHEDULER_STALE",
            "actionable_hint": "Run: python -m agency.lab_worker --root <AJAX_HOME>/ajax-kernel --once",
            "detail": detail,
        }
    if not scheduler_running:
        return {
            "ok": False,
            "code": "ALIVE_SCHEDULER_NOT_RUNNING",
            "actionable_hint": "Run: python bin/ajaxctl lab start",
            "detail": detail,
        }
    return {"ok": True, "code": "ALIVE_WORKER_OK", "actionable_hint": "", "detail": detail}


def _latest_receipt_payload(root_dir: Path, prefix: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    latest = _latest_glob(root_dir / "artifacts" / "receipts", f"{prefix}_*.json")
    if latest is None:
        return None, None, None
    payload = _safe_read_json(latest)
    if not payload:
        return None, str(latest), _to_ts(latest.stat().st_mtime)
    ts = _extract_ts(payload, latest, keys=["ts"])
    return payload, str(latest), ts


def _safe_signal(root_dir: Path) -> Dict[str, Any]:
    microfilm, microfilm_path, microfilm_ts = _latest_receipt_payload(root_dir, "microfilm_check")
    anchor, anchor_path, _anchor_ts = _latest_receipt_payload(root_dir, "anchor_preflight")

    detail = {
        "microfilm_receipt_path": microfilm_path,
        "anchor_receipt_path": anchor_path,
        "microfilm_ts": microfilm_ts,
    }
    if not isinstance(microfilm, dict):
        return {
            "ok": False,
            "code": "BLOCKED_ENV",
            "actionable_hint": "Run: python bin/ajaxctl microfilm check --root <AJAX_HOME>",
            "classification": "BLOCKED_ENV",
            "detail": detail,
            "microfilm_ts": microfilm_ts,
        }

    if bool(microfilm.get("overall_ok")):
        return {
            "ok": True,
            "code": "SAFE_OK",
            "actionable_hint": "",
            "classification": "PASS",
            "detail": detail,
            "microfilm_ts": microfilm_ts,
        }

    checks = microfilm.get("checks") if isinstance(microfilm.get("checks"), list) else []
    failing_checks = [c for c in checks if isinstance(c, dict) and not bool(c.get("ok"))]
    blocked_checks = [
        c
        for c in failing_checks
        if str(c.get("code") or "").strip().upper().startswith("BLOCKED_")
    ]
    chosen = blocked_checks[0] if blocked_checks else (failing_checks[0] if failing_checks else {})
    fail_code = str(chosen.get("code") or "SAFE_CHECK_FAILED").strip()
    hint = str(
        chosen.get("actionable_hint")
        or microfilm.get("actionable_hint")
        or "Review microfilm check report and fix blocked gates."
    ).strip()
    classification = "BLOCKED"
    is_anchor_mismatch = False
    if fail_code == "BLOCKED_RAIL_MISMATCH":
        is_anchor_mismatch = True
    if str(chosen.get("name") or "").strip().lower() == "doctor_anchor":
        is_anchor_mismatch = True
    if isinstance(anchor, dict):
        mismatches = anchor.get("mismatches") if isinstance(anchor.get("mismatches"), list) else []
        if mismatches:
            is_anchor_mismatch = True
    if is_anchor_mismatch:
        classification = "BLOCKED_ENV"
    return {
        "ok": False,
        "code": fail_code,
        "actionable_hint": hint,
        "classification": classification,
        "detail": detail,
        "microfilm_ts": microfilm_ts,
    }


def _max_ts(current: Optional[float], new: Optional[float]) -> Optional[float]:
    if current is None:
        return new
    if new is None:
        return current
    return float(new) if float(new) > float(current) else float(current)


def _update_state_latest(
    root_dir: Path,
    *,
    now_ts: float,
    latest_enqueued_ts: Optional[float],
    latest_completed_ts: Optional[float],
    latest_lab_org_ts: Optional[float],
    safe_signal: Dict[str, Any],
) -> Tuple[Dict[str, Any], Path]:
    path = root_dir / "artifacts" / "soak" / "state_latest.json"
    prev = _safe_read_json(path)
    state: Dict[str, Any] = {
        "schema": "ajax.soak_state.v1",
        "updated_ts": float(now_ts),
        "updated_utc": _utc_now(now_ts),
        "latest_enqueued_ts": _max_ts(_to_ts(prev.get("latest_enqueued_ts")), latest_enqueued_ts),
        "latest_completed_ts": _max_ts(_to_ts(prev.get("latest_completed_ts")), latest_completed_ts),
        "latest_lab_org_ts": _max_ts(_to_ts(prev.get("latest_lab_org_ts")), latest_lab_org_ts),
        "latest_microfilm_ts": _max_ts(
            _to_ts(prev.get("latest_microfilm_ts")),
            _to_ts(safe_signal.get("microfilm_ts")),
        ),
        "latest_blocked_ts": _to_ts(prev.get("latest_blocked_ts")),
        "latest_blocked_code": prev.get("latest_blocked_code"),
        "latest_blocked_hint": prev.get("latest_blocked_hint"),
    }
    if not bool(safe_signal.get("ok")):
        blocked_ts = _to_ts(safe_signal.get("microfilm_ts")) or float(now_ts)
        state["latest_blocked_ts"] = _max_ts(_to_ts(state.get("latest_blocked_ts")), blocked_ts)
        state["latest_blocked_code"] = safe_signal.get("code")
        state["latest_blocked_hint"] = safe_signal.get("actionable_hint")
    _write_json(path, state)
    return state, path


def _render_report(payload: Dict[str, Any]) -> str:
    signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
    alive = signals.get("alive") if isinstance(signals.get("alive"), dict) else {}
    effective = signals.get("effective") if isinstance(signals.get("effective"), dict) else {}
    safe = signals.get("safe") if isinstance(signals.get("safe"), dict) else {}
    lines = [
        "# Soak Status",
        "",
        f"- generated_at: {payload.get('ts_utc')}",
        f"- rail: {payload.get('rail')}",
        f"- result: {payload.get('result')}",
        f"- code: {payload.get('outcome_code')}",
        f"- actionable_hint: {payload.get('actionable_hint') or 'none'}",
        "",
        "## Summary",
        "",
        str(payload.get("summary_paragraph") or ""),
        "",
        "## Signals",
        "",
        "| signal | status | code | hint |",
        "|---|---|---|---|",
        "| ALIVE | {status} | {code} | {hint} |".format(
            status="PASS" if alive.get("ok") else "FAIL",
            code=str(alive.get("code") or "-"),
            hint=str(alive.get("actionable_hint") or "-").replace("|", "/"),
        ),
        "| EFFECTIVE | {status} | {code} | {hint} |".format(
            status="PASS" if effective.get("ok") else "FAIL",
            code=str(effective.get("code") or "-"),
            hint=str(effective.get("actionable_hint") or "-").replace("|", "/"),
        ),
        "| SAFE | {status} | {code} | {hint} |".format(
            status="PASS" if safe.get("ok") else "FAIL",
            code=str(safe.get("code") or "-"),
            hint=str(safe.get("actionable_hint") or "-").replace("|", "/"),
        ),
        "",
        "## Raw",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


def run_soak_check(
    root_dir: Path,
    *,
    rail: str = "lab",
    window_min: int = 60,
    now_ts: Optional[float] = None,
    forced_blocked_env_reason: Optional[str] = None,
    requested_root: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    now = float(now_ts if now_ts is not None else time.time())
    rail_n = _normalize_rail(rail)
    window_minutes = max(1, int(window_min or 60))
    window_s = float(window_minutes * 60)

    receipt_dir = root / "artifacts" / "receipts"
    report_dir = root / "artifacts" / "reports"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    signals: Dict[str, Any] = {}
    root_ok = _canonical_root_ok(root)
    if forced_blocked_env_reason:
        root_ok = False

    if not root_ok:
        reason = forced_blocked_env_reason or (
            "root_mismatch: expected canonical ajax-kernel runtime root."
        )
        signals = {
            "alive": {"ok": False, "code": "BLOCKED_ENV", "actionable_hint": reason},
            "effective": {
                "ok": True,
                "code": "NO_WORK_DECLARED",
                "actionable_hint": "Skipped because runtime root is blocked.",
            },
            "safe": {"ok": False, "code": "BLOCKED_ENV", "actionable_hint": reason},
        }
        payload: Dict[str, Any] = {
            "schema": "ajax.soak_check.v1",
            "ts_utc": _utc_now(now),
            "root": str(root),
            "requested_root": requested_root,
            "rail": rail_n,
            "window_min": window_minutes,
            "window_s": window_s,
            "ok": False,
            "result": "FAIL",
            "outcome_code": "BLOCKED_ENV",
            "actionable_hint": reason,
            "summary_paragraph": (
                "Soak FAIL (BLOCKED_ENV): runtime root no canónico o no resolvible; "
                "no se evalúan señales operativas hasta corregir el root."
            ),
            "signals": signals,
            "sources": {},
        }
        receipt_path = receipt_dir / f"soak_check_{_ts_label(now)}.json"
        report_path = report_dir / "soak_status.md"
        latest_report_path = report_dir / "latest_soak.md"
        _write_json(receipt_path, payload)
        report_text = _render_report(payload)
        _write_text(report_path, report_text)
        _write_text(latest_report_path, report_text)
        payload["receipt_path"] = str(receipt_path)
        payload["report_path"] = str(report_path)
        payload["latest_report_path"] = str(latest_report_path)
        return payload

    ttl_seconds = max(30, int(os.getenv("AJAX_HEALTH_TTL_SEC", "900") or "900"))
    provider_ttl = provider_status_ttl(root, ttl_seconds=ttl_seconds, now_ts=now)
    provider_ok = not bool(provider_ttl.get("stale"))

    manifest = _load_manifest(root)
    enabled_work = _enabled_work_items(manifest)
    lab_activity = _collect_lab_org_activity(root, now_ts=now, window_s=window_s)
    completed_activity = _collect_completed_activity(root, now_ts=now, window_s=window_s)
    worker_signal = _worker_scheduler_signal(
        root,
        rail=rail_n,
        now_ts=now,
        window_s=window_s,
        lab_activity=lab_activity,
    )
    safe_signal = _safe_signal(root)

    if provider_ok and bool(worker_signal.get("ok")):
        alive_signal = {
            "ok": True,
            "code": "ALIVE_OK",
            "actionable_hint": "",
            "detail": {"provider_ttl": provider_ttl, "worker_scheduler": worker_signal},
        }
    elif not provider_ok:
        alive_signal = {
            "ok": False,
            "code": "ALIVE_PROVIDER_STALE",
            "actionable_hint": "Run: python bin/ajaxctl doctor providers",
            "detail": {"provider_ttl": provider_ttl, "worker_scheduler": worker_signal},
        }
    else:
        alive_signal = {
            "ok": False,
            "code": worker_signal.get("code") or "ALIVE_WORKER_FAIL",
            "actionable_hint": worker_signal.get("actionable_hint"),
            "detail": {"provider_ttl": provider_ttl, "worker_scheduler": worker_signal},
        }

    work_declared = len(enabled_work) > 0
    enqueued_in_window = bool(lab_activity.get("enqueued_in_window"))
    completed_in_window = bool(completed_activity.get("completed_in_window"))
    if not work_declared:
        effective_signal = {
            "ok": True,
            "code": "NO_WORK_DECLARED",
            "actionable_hint": "No micro_challenges enabled in lab_org_manifest.",
            "detail": {
                "enabled_work_items": 0,
                "latest_enqueued_ts": lab_activity.get("latest_enqueued_ts"),
                "latest_completed_ts": completed_activity.get("latest_completed_ts"),
            },
        }
    elif enqueued_in_window or completed_in_window:
        effective_signal = {
            "ok": True,
            "code": "EFFECTIVE_OK",
            "actionable_hint": "",
            "detail": {
                "enabled_work_items": len(enabled_work),
                "enqueued_in_window": enqueued_in_window,
                "completed_in_window": completed_in_window,
                "latest_enqueued_ts": lab_activity.get("latest_enqueued_ts"),
                "latest_completed_ts": completed_activity.get("latest_completed_ts"),
            },
        }
    else:
        effective_signal = {
            "ok": False,
            "code": "EFFECTIVE_NO_ACTIVITY",
            "actionable_hint": "No enqueued/completed LAB activity in window; run lab start and worker once.",
            "detail": {
                "enabled_work_items": len(enabled_work),
                "window_min": window_minutes,
                "latest_enqueued_ts": lab_activity.get("latest_enqueued_ts"),
                "latest_completed_ts": completed_activity.get("latest_completed_ts"),
            },
        }

    signals = {
        "alive": alive_signal,
        "effective": effective_signal,
        "safe": safe_signal,
    }
    overall_ok = bool(alive_signal.get("ok")) and bool(effective_signal.get("ok")) and bool(
        safe_signal.get("ok")
    )

    first_failure = None
    for key in ("alive", "effective", "safe"):
        sig = signals.get(key) if isinstance(signals.get(key), dict) else {}
        if sig and not bool(sig.get("ok")):
            first_failure = sig
            break
    outcome_code = "SOAK_PASS" if overall_ok else str(first_failure.get("code") or "SOAK_FAIL")
    actionable_hint = "No action required." if overall_ok else str(
        first_failure.get("actionable_hint") or "Review soak signals and rerun."
    )

    summary = (
        "ALIVE {alive} ({alive_code}); EFFECTIVE {eff} ({eff_code}); SAFE {safe} ({safe_code})."
    ).format(
        alive="OK" if alive_signal.get("ok") else "FAIL",
        alive_code=alive_signal.get("code"),
        eff="OK" if effective_signal.get("ok") else "FAIL",
        eff_code=effective_signal.get("code"),
        safe="OK" if safe_signal.get("ok") else "FAIL",
        safe_code=safe_signal.get("code"),
    )
    if not overall_ok:
        summary = summary + f" Next: {actionable_hint}"

    state_latest, state_path = _update_state_latest(
        root,
        now_ts=now,
        latest_enqueued_ts=_to_ts(lab_activity.get("latest_enqueued_ts")),
        latest_completed_ts=_to_ts(completed_activity.get("latest_completed_ts")),
        latest_lab_org_ts=_to_ts(lab_activity.get("latest_receipt_ts")),
        safe_signal=safe_signal,
    )

    payload = {
        "schema": "ajax.soak_check.v1",
        "ts_utc": _utc_now(now),
        "root": str(root),
        "rail": rail_n,
        "window_min": window_minutes,
        "window_s": window_s,
        "ok": overall_ok,
        "result": "PASS" if overall_ok else "FAIL",
        "outcome_code": outcome_code,
        "actionable_hint": actionable_hint,
        "summary_paragraph": summary,
        "signals": signals,
        "sources": {
            "manifest_path": str(root / "config" / "lab_org_manifest.yaml"),
            "lab_org_latest_receipt_path": lab_activity.get("latest_receipt_path"),
            "lab_latest_result_path": completed_activity.get("latest_result_path"),
            "state_latest_path": str(state_path),
            "provider_status_path": str(root / "artifacts" / "health" / "providers_status.json"),
        },
        "state_latest": state_latest,
    }

    receipt_path = receipt_dir / f"soak_check_{_ts_label(now)}.json"
    report_path = report_dir / "soak_status.md"
    latest_report_path = report_dir / "latest_soak.md"
    _write_json(receipt_path, payload)
    report_text = _render_report(payload)
    _write_text(report_path, report_text)
    _write_text(latest_report_path, report_text)
    payload["receipt_path"] = str(receipt_path)
    payload["report_path"] = str(report_path)
    payload["latest_report_path"] = str(latest_report_path)
    return payload
