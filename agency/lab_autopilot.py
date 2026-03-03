from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agency.lab_control import LabStateStore
from agency.lab_session_anchor import validate_expected_session


def _utc_now(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _ts_label(ts: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts or time.time()))


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
    return raw if isinstance(raw, dict) else None


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _relpath(root_dir: Path, path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _dedupe_strings(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _is_interactive(override: Optional[bool]) -> bool:
    if isinstance(override, bool):
        return override
    try:
        return os.isatty(0) and os.isatty(1)
    except Exception:
        return False


def _normalize_rail(raw: Optional[str]) -> str:
    value = str(raw or os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab")
    value = value.strip().lower()
    if value in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def _read_human_active_flag(root_dir: Path) -> Dict[str, Any]:
    paths = [
        root_dir / "state" / "human_active.flag",
        root_dir / "artifacts" / "state" / "human_active.flag",
        root_dir / "artifacts" / "policy" / "human_active.flag",
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            if raw.startswith("{"):
                doc = json.loads(raw)
                if isinstance(doc, dict) and "human_active" in doc:
                    return {
                        "known": True,
                        "human_active": bool(doc.get("human_active")),
                        "source": str(path),
                        "last_input_age_sec": _to_float(doc.get("last_input_age_sec") or doc.get("idle_seconds")),
                    }
            lowered = raw.lower()
            if "true" in lowered:
                return {"known": True, "human_active": True, "source": str(path), "last_input_age_sec": None}
            if "false" in lowered:
                return {"known": True, "human_active": False, "source": str(path), "last_input_age_sec": None}
        except Exception:
            continue
    return {"known": False, "human_active": None, "source": None, "last_input_age_sec": None}


def _read_human_signal_snapshot(root_dir: Path) -> Dict[str, Any]:
    candidates = [
        root_dir / "artifacts" / "health" / "human_signal.json",
        root_dir / "artifacts" / "state" / "human_signal.json",
        root_dir / "artifacts" / "policy" / "human_signal.json",
    ]
    for path in candidates:
        doc = _safe_read_json(path)
        if not isinstance(doc, dict):
            continue
        if isinstance(doc.get("human_active"), bool):
            return {
                "known": True,
                "human_active": bool(doc.get("human_active")),
                "source": str(path),
                "last_input_age_sec": _to_float(doc.get("last_input_age_sec") or doc.get("idle_seconds")),
                "signal_ok": doc.get("ok"),
            }
    return _read_human_active_flag(root_dir)


def _gate_human_absent(
    root_dir: Path,
    *,
    absence_min: float,
    interactive: Optional[bool],
) -> Dict[str, Any]:
    required_absence_s = max(0.0, float(absence_min) * 60.0)
    signal = _read_human_signal_snapshot(root_dir)
    known = bool(signal.get("known"))
    human_active = signal.get("human_active") if isinstance(signal.get("human_active"), bool) else None
    age_s = _to_float(signal.get("last_input_age_sec"))
    tty_interactive = _is_interactive(interactive)
    remaining_s = None

    if known and human_active is True:
        return {
            "ok": False,
            "reason": "human_active_true",
            "source": signal.get("source"),
            "required_absence_s": required_absence_s,
            "observed_age_s": age_s,
            "remaining_s": remaining_s,
        }
    if known and human_active is False and age_s is not None and age_s < required_absence_s:
        remaining_s = max(0.0, required_absence_s - age_s)
        return {
            "ok": False,
            "reason": "absence_threshold_not_met",
            "source": signal.get("source"),
            "required_absence_s": required_absence_s,
            "observed_age_s": age_s,
            "remaining_s": remaining_s,
        }
    if not known and tty_interactive:
        return {
            "ok": False,
            "reason": "missing_human_signal_interactive_tty",
            "source": None,
            "required_absence_s": required_absence_s,
            "observed_age_s": age_s,
            "remaining_s": remaining_s,
        }
    return {
        "ok": True,
        "reason": "human_absent",
        "source": signal.get("source"),
        "required_absence_s": required_absence_s,
        "observed_age_s": age_s,
        "remaining_s": remaining_s,
    }


def _run_anchor_preflight(root_dir: Path, rail: str) -> Dict[str, Any]:
    try:
        from agency.anchor_preflight import run_anchor_preflight  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"anchor_preflight_unavailable:{str(exc)[:200]}", "receipt_path": None}
    try:
        return run_anchor_preflight(root_dir=root_dir, rail=rail, write_receipt=True)
    except Exception as exc:
        return {"ok": False, "error": f"anchor_preflight_failed:{str(exc)[:200]}", "receipt_path": None}


def _gate_env_safe(root_dir: Path, *, rail: Optional[str]) -> Dict[str, Any]:
    resolved_rail = _normalize_rail(rail)
    if resolved_rail != "lab":
        return {
            "ok": False,
            "reason": "rail_not_lab",
            "rail": resolved_rail,
            "anchor_receipt_path": None,
        }
    session_status = validate_expected_session(
        root_dir,
        required_rail=resolved_rail,
        required_display="dummy",
    )
    if not bool(session_status.get("ok")):
        return {
            "ok": False,
            "reason": str(session_status.get("reason") or "expected_session_invalid"),
            "rail": resolved_rail,
            "anchor_receipt_path": None,
            "session_status": session_status,
        }
    anchor = _run_anchor_preflight(root_dir, "lab")
    observed = anchor.get("observed") if isinstance(anchor.get("observed"), dict) else {}
    dummy_ok = bool(observed.get("display_target_is_dummy"))
    anchor_ok = bool(anchor.get("ok"))
    if not anchor_ok:
        mismatches = anchor.get("mismatches") if isinstance(anchor.get("mismatches"), list) else []
        non_ignored: list[Dict[str, Any]] = []
        for item in mismatches:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "")
            if code == "expected_session_missing":
                continue
            non_ignored.append(item)
        if non_ignored:
            first_code = str(non_ignored[0].get("code") or "anchor_preflight_blocked")
            return {
                "ok": False,
                "reason": first_code,
                "rail": resolved_rail,
                "anchor_receipt_path": anchor.get("receipt_path"),
                "anchor_payload": anchor,
                "session_status": session_status,
            }
    if not dummy_ok:
        return {
            "ok": False,
            "reason": "dummy_display_not_guaranteed",
            "rail": resolved_rail,
            "anchor_receipt_path": anchor.get("receipt_path"),
            "anchor_payload": anchor,
            "session_status": session_status,
        }
    return {
        "ok": True,
        "reason": "lab_anchor_ok",
        "rail": resolved_rail,
        "anchor_receipt_path": anchor.get("receipt_path"),
        "anchor_payload": anchor,
        "session_status": session_status,
    }


def _gate_budget(*, require_premium: bool = False) -> Dict[str, Any]:
    if not require_premium:
        return {"ok": True, "reason": "no_premium_required"}
    return {"ok": False, "reason": "premium_requires_consent_ttl"}


def _providers_status_info(root_dir: Path, *, stale_after_s: float, now_ts: float) -> Dict[str, Any]:
    status_path = root_dir / "artifacts" / "health" / "providers_status.json"
    doc = _safe_read_json(status_path) or {}
    updated_ts = _to_float(doc.get("updated_ts") or doc.get("updated_at"))
    if updated_ts is None and status_path.exists():
        try:
            updated_ts = float(status_path.stat().st_mtime)
        except Exception:
            updated_ts = None
    age_s = max(0.0, now_ts - updated_ts) if updated_ts is not None else None
    stale = True
    if age_s is not None:
        stale = age_s > max(1.0, stale_after_s)
    return {
        "path": str(status_path),
        "exists": status_path.exists(),
        "updated_ts": updated_ts,
        "age_s": age_s,
        "stale_after_s": stale_after_s,
        "stale": stale,
    }


def _job_bucket(now_ts: float) -> str:
    return time.strftime("%Y%m%d%H", time.gmtime(now_ts))


def _job_id(kind: str, now_ts: float) -> str:
    return f"autopilot_{kind}_{_job_bucket(now_ts)}"


def _result_exists_for_job(root_dir: Path, job_id: str) -> bool:
    results_dir = root_dir / "artifacts" / "lab" / "results"
    return any(results_dir.glob(f"result_*_{job_id}.json"))


def _queue_housekeeping_needed(root_dir: Path, *, now_ts: float) -> Dict[str, Any]:
    store = LabStateStore(root_dir)
    try:
        receipt = store.prune_terminal_jobs(mode="archive", dry_run=True, older_than_s=3600.0, now_ts=now_ts)
    except Exception as exc:
        return {"ok": False, "reason": f"prune_dry_run_failed:{str(exc)[:200]}", "actions": 0, "receipt": None}
    counts = receipt.get("counts") if isinstance(receipt.get("counts"), dict) else {}
    actions = int(counts.get("actions") or 0)
    return {"ok": True, "reason": "ok", "actions": actions, "receipt": receipt}


def _select_work_item(
    root_dir: Path,
    *,
    now_ts: float,
    providers_stale_min: float,
    allow_filesystem_basic: bool,
    allow_queue_housekeeping: bool,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}
    stale_after_s = max(60.0, float(providers_stale_min) * 60.0)
    providers_info = _providers_status_info(root_dir, stale_after_s=stale_after_s, now_ts=now_ts)
    diagnostics["providers_status"] = providers_info
    kind = "providers_probe_refresh"
    job_id = _job_id(kind, now_ts)
    if providers_info.get("stale") and not _result_exists_for_job(root_dir, job_id):
        return (
            {"kind": kind, "job_id": job_id, "reason": "providers_status_stale"},
            diagnostics,
        )
    if allow_filesystem_basic:
        kind = "filesystem_basic"
        job_id = _job_id(kind, now_ts)
        if not _result_exists_for_job(root_dir, job_id):
            return (
                {"kind": kind, "job_id": job_id, "reason": "filesystem_healthcheck_safe"},
                diagnostics,
            )
    if allow_queue_housekeeping:
        housekeeping = _queue_housekeeping_needed(root_dir, now_ts=now_ts)
        diagnostics["queue_housekeeping"] = housekeeping
        if housekeeping.get("ok") and int(housekeeping.get("actions") or 0) > 0:
            kind = "lab_queue_housekeeping"
            job_id = _job_id(kind, now_ts)
            if not _result_exists_for_job(root_dir, job_id):
                return (
                    {"kind": kind, "job_id": job_id, "reason": "terminal_jobs_prunable"},
                    diagnostics,
                )
    return None, diagnostics


def _refresh_providers_status(root_dir: Path) -> Dict[str, Any]:
    status_path = root_dir / "artifacts" / "health" / "providers_status.json"
    now_ts = float(time.time())

    def _fallback_touch(*, reason: str, error: Optional[str] = None) -> Dict[str, Any]:
        doc = _safe_read_json(status_path) or {}
        providers = doc.get("providers") if isinstance(doc.get("providers"), dict) else {}
        doc["schema"] = doc.get("schema") or "ajax.providers_status.v1"
        doc["providers"] = providers
        doc["updated_at"] = now_ts
        doc["updated_ts"] = now_ts
        doc["updated_utc"] = _utc_now(now_ts)
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        meta["last_refresh_source"] = "lab_autopilot.providers_probe"
        meta["last_refresh_reason"] = reason
        meta["last_refresh_ok"] = False
        if error:
            meta["last_refresh_error"] = str(error)[:240]
        doc["meta"] = meta
        _safe_write_json(status_path, doc)
        return {"ok": False, "path": str(status_path), "updated_ts": doc.get("updated_ts"), "error": reason}

    try:
        from agency.provider_breathing import ProviderBreathingLoop, _load_provider_configs
    except Exception as exc:
        return _fallback_touch(reason="provider_breathing_import_failed", error=str(exc))
    try:
        provider_cfg = _load_provider_configs(root_dir)
        loop = ProviderBreathingLoop(root_dir=root_dir, provider_configs=provider_cfg)
        status_doc = loop.run_once()
    except Exception as exc:
        return _fallback_touch(reason="provider_breathing_run_failed", error=str(exc))
    if not isinstance(status_doc, dict):
        return _fallback_touch(reason="provider_breathing_invalid_doc")

    updated_at = _to_float(status_doc.get("updated_at")) or float(time.time())
    status_doc["updated_ts"] = updated_at
    status_doc["updated_utc"] = _utc_now(updated_at)
    meta = status_doc.get("meta") if isinstance(status_doc.get("meta"), dict) else {}
    meta["last_refresh_source"] = "lab_autopilot.providers_probe"
    meta["last_refresh_reason"] = "provider_breathing_run_once"
    meta["last_refresh_ok"] = True
    status_doc["meta"] = meta
    _safe_write_json(status_path, status_doc)
    return {"ok": True, "path": str(status_path), "updated_ts": updated_at, "error": None}


def _refresh_providers_probe(root_dir: Path) -> Dict[str, Any]:
    status_refresh = _refresh_providers_status(root_dir)
    evidence_refs = [str(status_refresh.get("path") or (root_dir / "artifacts" / "health" / "providers_status.json"))]
    ledger_error = None
    try:
        from agency.provider_ledger import ProviderLedger

        ledger = ProviderLedger(root_dir=root_dir)
        doc = ledger.refresh(timeout_s=2.0)
        ledger_path = str(doc.get("path") or (root_dir / "artifacts" / "provider_ledger" / "latest.json"))
        evidence_refs.append(ledger_path)
    except Exception as exc:
        ledger_error = str(exc)[:200]
    return {
        "ok": bool(status_refresh.get("ok")) and ledger_error is None,
        "status_refresh": status_refresh,
        "ledger_error": ledger_error,
        "evidence_refs": evidence_refs,
    }


def _execute_providers_probe_refresh(root_dir: Path, *, now_ts: float, job_id: str) -> Dict[str, Any]:
    before = _providers_status_info(root_dir, stale_after_s=1.0, now_ts=now_ts)
    probe = _refresh_providers_probe(root_dir)
    after_now = float(time.time())
    after = _providers_status_info(root_dir, stale_after_s=1.0, now_ts=after_now)
    before_ts = _to_float(before.get("updated_ts"))
    after_ts = _to_float(after.get("updated_ts"))
    changed = False
    if before_ts is None and after_ts is not None:
        changed = True
    elif before_ts is not None and after_ts is not None and after_ts >= before_ts:
        changed = True
    ok = bool(changed) and bool(after.get("exists"))
    if bool(probe.get("ok")) is False and after_ts is None:
        ok = False
    return {
        "ok": ok,
        "job_id": job_id,
        "kind": "providers_probe_refresh",
        "summary": "providers_status refreshed." if ok else "providers_status refresh failed verification.",
        "verify": {"before_updated_ts": before_ts, "after_updated_ts": after_ts, "changed": changed},
        "evidence_refs": list(probe.get("evidence_refs") or []),
    }


def _execute_filesystem_basic(root_dir: Path, *, now_ts: float, job_id: str) -> Dict[str, Any]:
    evidence_dir = root_dir / "artifacts" / "lab" / "evidence" / job_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / f"filesystem_basic_{_ts_label(now_ts)}.txt"
    payload = f"lab_autopilot filesystem basic evidence\njob_id={job_id}\nts_utc={_utc_now(now_ts)}\n"
    evidence_path.write_text(payload, encoding="utf-8")
    ok = evidence_path.exists()
    return {
        "ok": ok,
        "job_id": job_id,
        "kind": "filesystem_basic",
        "summary": "filesystem_basic evidence written." if ok else "filesystem_basic evidence missing.",
        "verify": {"evidence_exists": bool(ok), "evidence_path": str(evidence_path)},
        "evidence_refs": [str(evidence_path)],
    }


def _execute_lab_queue_housekeeping(root_dir: Path, *, now_ts: float, job_id: str) -> Dict[str, Any]:
    store = LabStateStore(root_dir)
    try:
        receipt = store.prune_terminal_jobs(mode="archive", dry_run=False, older_than_s=3600.0, now_ts=now_ts)
    except Exception as exc:
        return {
            "ok": False,
            "job_id": job_id,
            "kind": "lab_queue_housekeeping",
            "summary": f"lab_queue_housekeeping failed: {str(exc)[:160]}",
            "verify": {"error": str(exc)[:160]},
            "evidence_refs": [],
        }
    counts = receipt.get("counts") if isinstance(receipt.get("counts"), dict) else {}
    actions = int(counts.get("actions") or 0)
    evidence = []
    receipt_path = receipt.get("receipt_path")
    if isinstance(receipt_path, str) and receipt_path:
        evidence.append(receipt_path)
    return {
        "ok": True,
        "job_id": job_id,
        "kind": "lab_queue_housekeeping",
        "summary": f"lab_queue_housekeeping completed (actions={actions}).",
        "verify": {"actions": actions, "receipt_path": receipt_path},
        "evidence_refs": evidence,
    }


def _execute_work_item(root_dir: Path, *, work_item: Dict[str, Any], now_ts: float) -> Dict[str, Any]:
    kind = str(work_item.get("kind") or "")
    job_id = str(work_item.get("job_id") or _job_id(kind or "work", now_ts))
    if kind == "providers_probe_refresh":
        return _execute_providers_probe_refresh(root_dir, now_ts=now_ts, job_id=job_id)
    if kind == "filesystem_basic":
        return _execute_filesystem_basic(root_dir, now_ts=now_ts, job_id=job_id)
    if kind == "lab_queue_housekeeping":
        return _execute_lab_queue_housekeeping(root_dir, now_ts=now_ts, job_id=job_id)
    return {
        "ok": False,
        "job_id": job_id,
        "kind": kind,
        "summary": f"unsupported_work_item:{kind}",
        "verify": {"error": "unsupported_work_item"},
        "evidence_refs": [],
    }


def _write_result(root_dir: Path, *, now_ts: float, execution: Dict[str, Any]) -> Path:
    results_dir = root_dir / "artifacts" / "lab" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(execution.get("job_id") or "job")
    result_path = results_dir / f"result_{_ts_label(now_ts)}_{job_id}.json"
    payload = {
        "schema": "ajax.lab.autopilot_result.v0",
        "ts_utc": _utc_now(now_ts),
        "job_id": job_id,
        "kind": execution.get("kind"),
        "ok": bool(execution.get("ok")),
        "summary": execution.get("summary"),
        "verify": execution.get("verify"),
        "evidence_refs": execution.get("evidence_refs") or [],
    }
    _safe_write_json(result_path, payload)
    return result_path


def _write_lab_org_tick_receipt(root_dir: Path, *, now_ts: float, receipt: Dict[str, Any]) -> Path:
    out_dir = root_dir / "artifacts" / "lab_org" / _ts_label(now_ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "receipt.json"
    payload = {
        "schema": "ajax.lab.autopilot.lab_org_tick.v0",
        "ts_utc": _utc_now(now_ts),
        "action": receipt.get("action"),
        "mode": receipt.get("mode"),
        "selected_work_item": receipt.get("selected_work_item"),
        "gates": receipt.get("gates"),
        "evidence_refs": receipt.get("evidence_refs") or [],
    }
    _safe_write_json(out_path, payload)
    return out_path


@dataclass
class AutopilotTickOptions:
    mode: str = "once"
    absence_min: float = 10.0
    providers_stale_min: float = 60.0
    rail: Optional[str] = None
    interactive: Optional[bool] = None
    allow_filesystem_basic: bool = True
    allow_queue_housekeeping: bool = True
    require_premium: bool = False


def run_autopilot_tick(root_dir: Path, *, options: AutopilotTickOptions, now_ts: Optional[float] = None) -> Dict[str, Any]:
    root = Path(root_dir)
    now = float(now_ts or time.time())
    mode = str(options.mode or "once")
    evidence_abs: list[Path] = []

    gate_human = _gate_human_absent(
        root,
        absence_min=float(options.absence_min),
        interactive=options.interactive,
    )
    gate_env = _gate_env_safe(root, rail=options.rail)
    gate_budget = _gate_budget(require_premium=bool(options.require_premium))
    gates = {
        "human_absent": gate_human,
        "env_safe": {k: v for k, v in gate_env.items() if k != "anchor_payload"},
        "budget_ok": gate_budget,
    }
    anchor_receipt = gate_env.get("anchor_receipt_path")
    if isinstance(anchor_receipt, str) and anchor_receipt:
        evidence_abs.append(Path(anchor_receipt))
    if isinstance(gate_human.get("source"), str) and gate_human.get("source"):
        evidence_abs.append(Path(str(gate_human.get("source"))))

    all_gates_ok = bool(gate_human.get("ok")) and bool(gate_env.get("ok")) and bool(gate_budget.get("ok"))
    selected, diagnostics = _select_work_item(
        root,
        now_ts=now,
        providers_stale_min=float(options.providers_stale_min),
        allow_filesystem_basic=bool(options.allow_filesystem_basic),
        allow_queue_housekeeping=bool(options.allow_queue_housekeeping),
    )

    action = "NOOP"
    execution = None
    result_path: Optional[Path] = None
    next_hint = "No work item selected."

    if not all_gates_ok:
        action = "BLOCKED"
        selected = None
        reasons = []
        for gate_name, gate_payload in gates.items():
            if not bool(gate_payload.get("ok")):
                reasons.append(f"{gate_name}:{gate_payload.get('reason')}")
        next_hint = f"Blocked by gates: {', '.join(reasons)}"
    elif mode == "dry-run":
        action = "NOOP"
        next_hint = (
            f"Dry-run selected {selected.get('kind')}" if isinstance(selected, dict) else "Dry-run found no safe work-item."
        )
    elif selected is None:
        action = "NOOP"
        next_hint = "No safe work item available for this tick."
    else:
        execution = _execute_work_item(root, work_item=selected, now_ts=now)
        evidence_abs.extend(Path(p) for p in list(execution.get("evidence_refs") or []) if isinstance(p, str))
        if bool(execution.get("ok")):
            action = "EXECUTED"
            result_path = _write_result(root, now_ts=now, execution=execution)
            evidence_abs.append(result_path)
            next_hint = "Tick executed successfully."
        else:
            action = "BLOCKED"
            next_hint = f"Execution failed verify: {execution.get('summary')}"

    evidence_rel = _dedupe_strings(
        [rel for rel in (_relpath(root, p) for p in evidence_abs) if isinstance(rel, str) and rel]
    )
    receipt_payload: Dict[str, Any] = {
        "schema": "ajax.lab.autopilot_tick.v0",
        "ts_utc": _utc_now(now),
        "mode": mode,
        "gates": gates,
        "selected_work_item": selected,
        "action": action,
        "evidence_refs": evidence_rel,
        "next_hint": next_hint,
        "preflight": diagnostics,
    }
    if execution is not None:
        receipt_payload["execution"] = execution
    if result_path is not None:
        receipt_payload["result_path"] = _relpath(root, result_path)

    receipts_dir = root / "artifacts" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    tick_receipt_path = receipts_dir / f"lab_autopilot_tick_{_ts_label(now)}.json"
    _safe_write_json(tick_receipt_path, receipt_payload)
    lab_org_receipt_path = _write_lab_org_tick_receipt(root, now_ts=now, receipt=receipt_payload)

    receipt_payload["tick_receipt_path"] = _relpath(root, tick_receipt_path)
    receipt_payload["lab_org_receipt_path"] = _relpath(root, lab_org_receipt_path)
    artifacts_written = [
        str(receipt_payload["tick_receipt_path"]),
        str(receipt_payload["lab_org_receipt_path"]),
    ]
    if isinstance(receipt_payload.get("result_path"), str):
        artifacts_written.append(str(receipt_payload["result_path"]))
    receipt_payload["artifacts_written"] = _dedupe_strings(artifacts_written)
    if receipt_payload["tick_receipt_path"] not in receipt_payload["evidence_refs"]:
        receipt_payload["evidence_refs"].append(str(receipt_payload["tick_receipt_path"]))
    if receipt_payload["lab_org_receipt_path"] not in receipt_payload["evidence_refs"]:
        receipt_payload["evidence_refs"].append(str(receipt_payload["lab_org_receipt_path"]))
    receipt_payload["evidence_refs"] = _dedupe_strings(list(receipt_payload["evidence_refs"]))
    _safe_write_json(tick_receipt_path, receipt_payload)
    return receipt_payload


def run_autopilot_daemon(
    root_dir: Path,
    *,
    options: AutopilotTickOptions,
    interval_s: float,
    max_ticks: int = 0,
) -> Dict[str, Any]:
    root = Path(root_dir)
    daemon_dir = root / "artifacts" / "lab" / "autopilot"
    daemon_dir.mkdir(parents=True, exist_ok=True)
    pid_path = daemon_dir / "worker.pid"
    heartbeat_path = daemon_dir / "heartbeat.json"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    ticks = 0
    receipts: list[str] = []
    last_action = None
    try:
        while True:
            tick_now = time.time()
            tick_opts = AutopilotTickOptions(
                mode="daemon",
                absence_min=options.absence_min,
                providers_stale_min=options.providers_stale_min,
                rail=options.rail,
                interactive=options.interactive,
                allow_filesystem_basic=options.allow_filesystem_basic,
                allow_queue_housekeeping=options.allow_queue_housekeeping,
                require_premium=options.require_premium,
            )
            tick_payload = run_autopilot_tick(root, options=tick_opts, now_ts=tick_now)
            ticks += 1
            last_action = tick_payload.get("action")
            if isinstance(tick_payload.get("tick_receipt_path"), str):
                receipts.append(str(tick_payload.get("tick_receipt_path")))
            heartbeat = {
                "schema": "ajax.lab.autopilot.daemon_heartbeat.v0",
                "status": "RUNNING",
                "ts_utc": _utc_now(),
                "ticks": ticks,
                "interval_s": float(interval_s),
                "last_action": last_action,
                "last_tick_receipt": tick_payload.get("tick_receipt_path"),
            }
            _safe_write_json(heartbeat_path, heartbeat)
            if max_ticks > 0 and ticks >= max_ticks:
                break
            time.sleep(max(1.0, float(interval_s)))
    finally:
        try:
            pid_path.unlink(missing_ok=True)
        except Exception:
            pass
        done_hb = {
            "schema": "ajax.lab.autopilot.daemon_heartbeat.v0",
            "status": "STOPPED",
            "ts_utc": _utc_now(),
            "ticks": ticks,
            "interval_s": float(interval_s),
            "last_action": last_action,
        }
        _safe_write_json(heartbeat_path, done_hb)

    return {
        "ok": True,
        "mode": "daemon",
        "ticks": ticks,
        "interval_s": float(interval_s),
        "last_action": last_action,
        "tick_receipts": receipts,
        "pid_path": _relpath(root, pid_path),
        "heartbeat_path": _relpath(root, heartbeat_path),
    }
