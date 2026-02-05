from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from agency.council import load_council_state
except Exception:  # pragma: no cover - fallback
    def load_council_state(path=None):  # type: ignore
        return {"mode": "normal"}


CRITICAL_ACTIONS = {
    "powershell.run",
    "powershell.exec",
    "cmd.exec",
    "shell.exec",
    "file.delete",
    "app.kill",
    "process.kill",
    "system.stop",
}

DANGEROUS_HOTKEYS = {
    ("alt", "f4"),
    ("ctrl", "alt", "del"),
    ("ctrl", "shift", "esc"),
    ("win", "r"),
    ("win", "x"),
}


@dataclass
class ActionAudit:
    name: str
    args: Dict[str, Any]
    classification: str  # safe|moderate|critical
    allowed: bool
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_action_name(action: Any) -> str:
    if isinstance(action, str):
        return action.strip()
    if isinstance(action, dict):
        return str(action.get("tool") or action.get("action") or "").strip()
    return ""


def _extract_args(action: Any) -> Dict[str, Any]:
    if isinstance(action, dict):
        if "args" in action and isinstance(action.get("args"), dict):
            return action.get("args") or {}
        return {k: v for k, v in action.items() if k not in {"tool", "action"}}
    return {}


def _normalize_hotkeys(args: Dict[str, Any]) -> Tuple[str, ...]:
    keys = args.get("keys") if isinstance(args, dict) else None
    if keys is None:
        return tuple()
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        return tuple()
    normalized = tuple(str(k).strip().lower() for k in keys if str(k).strip())
    return normalized


def classify_action(action: Any) -> Tuple[str, str]:
    """
    Devuelve (classification, reason).
    classification ∈ {"safe","moderate","critical"}.
    """
    name = _normalize_action_name(action)
    args = _extract_args(action)
    if not name:
        return "moderate", "unknown_action"
    lower = name.lower()

    if lower in CRITICAL_ACTIONS:
        return "critical", "listed_critical_action"
    if lower.startswith("powershell") or lower.startswith("cmd."):
        return "critical", "shell_execution"
    if lower.startswith("file.delete") or "delete" in lower:
        return "critical", "delete_operation"
    if lower.startswith("app.kill") or "kill" in lower:
        return "critical", "process_kill"
    if lower == "keyboard.hotkey":
        hotkeys = _normalize_hotkeys(args)
        if hotkeys in DANGEROUS_HOTKEYS:
            return "critical", "dangerous_hotkey"
        return "moderate", "hotkey"
    return "moderate", "default"


def evaluate_plan_actions(plan_actions: List[Any], council_state: Dict[str, Any], infra_blocked: bool) -> Tuple[bool, str, List[ActionAudit]]:
    audits: List[ActionAudit] = []
    degraded = str((council_state or {}).get("mode") or "normal").lower() == "degraded"
    blocked_reason = ""
    for item in plan_actions:
        name = _normalize_action_name(item)
        args = _extract_args(item)
        classification, cls_reason = classify_action(item)
        allowed = True
        reason = None
        if classification == "critical":
            if degraded:
                allowed = False
                reason = "council_degraded"
            elif infra_blocked:
                allowed = False
                reason = "infra_breaker_tripped"
            else:
                reason = cls_reason
        audits.append(ActionAudit(name=name or "unknown", args=args, classification=classification, allowed=allowed, reason=reason))
        if classification == "critical" and not allowed and not blocked_reason:
            blocked_reason = reason or "critical_guard_blocked"
    allowed_overall = not blocked_reason
    return allowed_overall, blocked_reason, audits


def write_audit_log(
    path: Path,
    *,
    mission_id: str,
    intent: str,
    council_state: Dict[str, Any],
    infra_state: Dict[str, Any],
    actions: List[ActionAudit],
    verification: Optional[Dict[str, Any]],
    final_status: str,
    result_ok: bool,
    errors: Optional[List[str]] = None,
) -> None:
    payload = {
        "mission_id": mission_id,
        "intent": intent,
        "timestamp": time.time(),
        "council_state": council_state,
        "infra_state": infra_state,
        "actions": [a.to_dict() for a in actions],
        "verification": verification,
        "final_status": final_status,
        "result_ok": result_ok,
        "errors": errors or [],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
