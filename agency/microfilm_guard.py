from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional


_BLOCKED = "BLOCKED"
_READY = "READY"

_REVERSIBLE_ACTIONS = {
    "app.launch",
    "window.focus",
    "window.minimize",
    "window.maximize",
    "window.restore",
    "window.move",
    "window.resize",
    "desktop.isolate_active_window",
    "keyboard.type",
    "keyboard.hotkey",
    "mouse.click",
    "mouse.move",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        try:
            raw = asdict(value)
            if isinstance(raw, dict):
                return raw
        except Exception:
            return {}
    if hasattr(value, "__dict__"):
        try:
            raw = dict(value.__dict__)
            if isinstance(raw, dict):
                return raw
        except Exception:
            return {}
    return {}


def _normalize_rail(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in {"prod", "production", "live"}:
        return "prod"
    if value in {"lab", "laboratory"}:
        return "lab"
    return "lab"


def _normalize_display_target(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return ""
    if value in {"dummy", "lab"}:
        return "dummy"
    if value in {"primary", "prod", "production", "live"}:
        return "primary"
    return value


def _blocked(code: str, actionable_hint: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "status": _BLOCKED,
        "terminal_status": _BLOCKED,
        "code": code,
        "actionable_hint": actionable_hint,
    }
    payload.update(extra)
    return payload


def _ready(**extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": True, "status": _READY}
    payload.update(extra)
    return payload


def _plan_steps(plan: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(plan, dict):
        steps = plan.get("steps")
    else:
        steps = getattr(plan, "steps", None)
    if not isinstance(steps, list):
        return []
    out = []
    for step in steps:
        if isinstance(step, dict):
            out.append(step)
    return out


def _plan_metadata(plan: Any) -> Dict[str, Any]:
    if isinstance(plan, dict):
        metadata = plan.get("metadata")
    else:
        metadata = getattr(plan, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def enforce_ssc(ctx: Any) -> Dict[str, Any]:
    context = _as_dict(ctx)
    actuation = bool(
        context.get("actuation")
        or context.get("plan_requires_physical_actions")
        or context.get("requires_actuation")
    )
    snapshot0 = context.get("snapshot0")
    if actuation and (not isinstance(snapshot0, dict) or not snapshot0):
        return _blocked(
            "BLOCKED_SSC_MISSING",
            "Capture snapshot0 (SSC) before actuation and retry.",
            actuation=True,
        )
    return _ready(actuation=actuation, code="SSC_OK")


def enforce_verify_before_done(result: Any, verification: Any) -> Dict[str, Any]:
    payload = _as_dict(result)
    verif = _as_dict(verification)
    if not verif:
        detail = payload.get("detail")
        if isinstance(detail, dict):
            verif = _as_dict(detail.get("verification"))

    status_raw = payload.get("status") or payload.get("mission_status")
    status = str(status_raw or "").strip().upper()
    if not status:
        try:
            status = str(getattr(result, "status", "") or "").strip().upper()
        except Exception:
            status = ""
    is_done = status in {"DONE", "COMPLETED"}
    verify_ok = bool(verif.get("ok"))
    if is_done and not verify_ok:
        return _blocked(
            "BLOCKED_VERIFY_REQUIRED",
            "Run verification and set verification.ok=true before closing DONE.",
            status=status or "DONE",
            verification=verif or None,
        )
    return _ready(status=status or None, verify_ok=verify_ok, code="VERIFY_OK")


def enforce_lab_prod_separation(ctx: Any) -> Dict[str, Any]:
    context = _as_dict(ctx)
    rail = _normalize_rail(context.get("rail"))
    display_target = _normalize_display_target(context.get("display_target"))
    require_display_target = bool(context.get("require_display_target"))

    human_value = context.get("human_active")
    human_known = human_value is not None
    human_active = bool(human_value) if human_known else False

    mismatches = []
    warnings = []
    if rail == "lab":
        if display_target == "primary":
            mismatches.append("lab_requires_dummy_display")
        if require_display_target and not display_target:
            mismatches.append("lab_display_target_missing")
        if human_known and human_active:
            mismatches.append("lab_requires_human_inactive")
    elif rail == "prod":
        if display_target == "dummy":
            mismatches.append("prod_requires_primary_display")

    anchor_codes = []
    raw_anchor = context.get("anchor_mismatches")
    if isinstance(raw_anchor, list):
        for item in raw_anchor:
            if isinstance(item, dict):
                code = str(item.get("code") or "").strip()
            else:
                code = str(item or "").strip()
            if code:
                anchor_codes.append(code)
    for code in anchor_codes:
        if code == "expected_session_missing":
            if rail == "lab" and display_target == "dummy":
                warnings.append(code)
            else:
                mismatches.append(code)
        elif code:
            mismatches.append(code)

    if mismatches:
        return _blocked(
            "BLOCKED_RAIL_MISMATCH",
            "Align rail/display_target/human_active before continuing.",
            rail=rail,
            display_target=display_target or None,
            human_active=human_value,
            mismatches=mismatches,
            warnings=warnings or None,
        )
    return _ready(
        code="RAIL_OK",
        rail=rail,
        display_target=display_target or None,
        human_active=human_value,
        warnings=warnings or None,
    )


def enforce_evidence_tiers(ctx: Any, verification: Any) -> Dict[str, Any]:
    context = _as_dict(ctx)
    out = _as_dict(verification)

    ok = bool(out.get("ok"))
    mode = str(
        out.get("verification_mode") or context.get("verification_mode") or "synthetic"
    ).strip().lower()
    driver_online = bool(
        out.get("driver_online")
        if "driver_online" in out
        else context.get("driver_online")
    )
    driver_simulated = bool(
        out.get("driver_simulated")
        if "driver_simulated" in out
        else context.get("driver_simulated")
    )

    if ok and mode == "real" and driver_online and not driver_simulated:
        evidence_tier = "real_online"
        promote_trust = True
    elif ok and driver_simulated:
        evidence_tier = "simulated_driver"
        promote_trust = False
    elif ok:
        evidence_tier = "synthetic_or_offline"
        promote_trust = False
    else:
        evidence_tier = "unverified"
        promote_trust = False

    out["verification_mode"] = mode
    out["driver_online"] = driver_online
    out["driver_simulated"] = driver_simulated
    out["evidence_tier"] = evidence_tier
    out["promote_trust"] = bool(promote_trust)
    if ok and not promote_trust:
        if driver_simulated:
            out["actionable_hint"] = "Driver is simulated; trust promotion is intentionally disabled."
        elif mode != "real":
            out["actionable_hint"] = "Use verification_mode=real to enable trust promotion."
        elif not driver_online:
            out["actionable_hint"] = "Bring the driver online before trust promotion."
    return out


def enforce_undo_for_reversible(plan: Any) -> Dict[str, Any]:
    metadata = _plan_metadata(plan)
    reversible = bool(metadata.get("reversible") or metadata.get("has_reversible_actions"))

    if not reversible:
        for step in _plan_steps(plan):
            action = str(step.get("action") or "").strip()
            if action in _REVERSIBLE_ACTIONS:
                reversible = True
                break

    if not reversible:
        return _ready(code="UNDO_NOT_REQUIRED", reversible=False)

    has_undo = False
    tx_paths = metadata.get("tx_paths") if isinstance(metadata.get("tx_paths"), dict) else {}
    undo_path = tx_paths.get("undo_plan_path") if isinstance(tx_paths, dict) else None
    if isinstance(undo_path, str) and undo_path.strip():
        has_undo = True
    if not has_undo and isinstance(metadata.get("undo_plan"), dict):
        has_undo = True
    if not has_undo and isinstance(plan, dict):
        undo_raw = plan.get("undo")
        if isinstance(undo_raw, list) and undo_raw:
            has_undo = True

    if not has_undo:
        return _blocked(
            "BLOCKED_UNDO_REQUIRED",
            "Add an undo plan for reversible actions before APPLY.",
            reversible=True,
        )
    return _ready(code="UNDO_OK", reversible=True)
