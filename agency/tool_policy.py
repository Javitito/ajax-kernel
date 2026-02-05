from __future__ import annotations

from typing import Any, Dict, List, Optional

from agency.tool_inventory import get_tool_status
from agency.tool_inventory import _summarize_notes  # type: ignore
from agency.tool_schema import ToolPlan, ToolSpec


HISTORY_TOKENS = [
    "hist",
    "history",
    "repro",
    "antes",
    "previo",
    "officebot",
    "notepad_soberania",
    "recuerda",
    "resume la evolución",
    "evolución",
]

CRITICAL_TOKENS = [
    "powershell",
    "cmd",
    "cmd.exe",
    "delete",
    "borrar",
    "borra",
    "elim",
    "rm ",
    "del ",
    "kill",
    "taskkill",
    "shutdown",
    "hotkey alt",
    "format",
]

UNCERTAINTY_TOKENS = [
    "cierra",
    "cerrar",
    "sigue abierto",
    "sigue abierta",
    "esta abierto",
    "está abierto",
    "no se si",
    "no sé si",
    "quedo abierto",
    "queda abierto",
    "still open",
]


def _lower(text: str) -> str:
    return (text or "").lower()


def _dedup(seq: List[str]) -> List[str]:
    return list(dict.fromkeys([s for s in seq if s]))


def _intention_contains(text: str, tokens: List[str]) -> bool:
    ltext = _lower(text)
    return any(tok in ltext for tok in tokens)


def _heartbeat_status(heartbeat: Optional[Dict[str, Any]]) -> str:
    if not isinstance(heartbeat, dict):
        return ""
    return _lower(heartbeat.get("status") or "")


def _availability(snapshot: Dict[str, Any], tool_id: str) -> bool:
    status = snapshot.get(tool_id)
    if status is None:
        return False
    state = getattr(status, "state", None)
    if state is None and isinstance(status, dict):
        state = status.get("state")
    if state is None and hasattr(status, "get"):
        try:
            state = status.get("state")
        except Exception:
            state = None
    if isinstance(state, str):
        return state.lower() == "available"
    if isinstance(status, dict):
        return str(status.get("state") or "").lower() == "available"
    return False


def select_tool_plan(
    intent_text: str,
    heartbeat: Optional[Dict[str, Any]],
    risk_flags: Optional[Dict[str, Any]],
    inventory: List[ToolSpec],
    history: Optional[List[str]] = None,  # noqa: ARG001 - hook for futura
    tool_use_notes: Optional[Dict[str, Any]] = None,
) -> ToolPlan:
    """
    Selección determinista de ToolPlan basado en intención + heartbeat.
    No llama a modelos; usa solo heurísticas y snapshot de estado.
    """
    hb_status = _heartbeat_status(heartbeat)
    snapshot = get_tool_status(inventory, heartbeat, tool_use_notes)
    allowed = [spec.id for spec in inventory if snapshot.get(spec.id) and snapshot[spec.id].state == "available"]
    plan = ToolPlan(
        considered=[spec.id for spec in inventory],
        tool_status_snapshot=snapshot,
        allowed=allowed,
        budget={
            "memory.leann_history": {"top_k": 3, "cooldown_s": 30},
            "sensing.vision_delta": {"max_per_turn": 2},
        },
        risk_flags=dict(risk_flags or {}),
        affordances={spec.id: spec.affordances for spec in inventory if spec.affordances},
        tool_use_notes=_summarize_notes(tool_use_notes or {}),
    )

    if hb_status and hb_status not in {"green", "healthy"}:
        plan.selected = ["infra.heartbeat"]
        plan.required = ["infra.heartbeat"]
        plan.reasons.append(f"heartbeat:{hb_status}")
        plan.satisfied["infra.heartbeat"] = _availability(snapshot, "infra.heartbeat")
        if not plan.satisfied.get("infra.heartbeat", False):
            plan.incomplete = True
        return plan

    needs_history = _intention_contains(intent_text, HISTORY_TOKENS)
    critical = _intention_contains(intent_text, CRITICAL_TOKENS) or bool((risk_flags or {}).get("high_risk_intent"))
    uncertainty = _intention_contains(intent_text, UNCERTAINTY_TOKENS)

    if needs_history:
        plan.selected.append("memory.leann_history")
        plan.required.append("memory.leann_history")
        plan.reasons.append("historical_intent")

    if critical:
        if "memory.leann_history" not in plan.required:
            plan.required.append("memory.leann_history")
        plan.selected.append("memory.leann_history")
        plan.reasons.append("high_risk_intent")
        plan.risk_flags["high_risk_intent"] = True

    if uncertainty:
        plan.selected.append("actuation.driver_uia")
        plan.reasons.append("uncertainty_state")
        uia_confident = bool((risk_flags or {}).get("uia_confident"))
        if not uia_confident:
            plan.selected.append("sensing.vision_delta")

    # Marca cumplimiento de requeridos basado en estado actual
    for req in plan.required:
        available = _availability(snapshot, req)
        plan.satisfied.setdefault(req, available)
        if not available:
            plan.incomplete = True
            plan.reasons.append(f"required_unavailable:{req}")

    plan.selected = _dedup(plan.selected)
    plan.required = _dedup(plan.required)
    plan.reasons = _dedup(plan.reasons)
    plan.allowed = _dedup(plan.allowed)

    return plan


__all__ = ["select_tool_plan"]
