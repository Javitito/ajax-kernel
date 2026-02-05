from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Optional

from agency.method_pack import AJAX_METHOD_PACK


P0_CHAT = textwrap.dedent(
    """
    You are AJAX's Planning Brain.
    Turn the user's intent into a SAFE, EXECUTABLE PLAN.

    Output:
    - Return ONLY valid JSON (no extra text).
    - Format:
      {
        "plan_id": "short-id",
        "steps": [
          {
            "id": "task-1",
            "intent": "one sentence describing the complete task",
            "preconditions": { "expected_state": { ... } },
            "action": "...",
            "args": { ... },
            "evidence_required": ["driver.active_window"],
            "success_spec": { "expected_state": { ... } },
            "on_fail": "abort"
          }
        ],
        "success_contract": {
          "primary_source": "uia|vision|fs",
          "primary_check": { ... },
          "fallback_source": "vision|none",
          "fallback_check": { ... },
          "conflict_resolution": "fail_safe|primary_wins|ask_human"
        }
      }

    Constraints:
    - Use ONLY actions from actions_catalog.actions.
    - Every step must include success_spec with at least one check.
    - Focus before typing or hotkeys.
    """
).strip()

P1_OPS_STANDARD = textwrap.dedent(
    """
    You are AJAX's Planning Brain.
    Transform a high-level user intent into a SAFE, EXECUTABLE PLAN.

    Rules:
    - Every step is a complete task: preconditions + action + evidence_required + success_spec + on_fail.
    - Do NOT invent actions or arguments outside the ActionCatalog.
    - Act then verify: [LAUNCH] -> [VERIFY OPEN] -> [FOCUS] -> [ACT] -> [VERIFY RESULT].
    - success_spec.expected_state must contain at least one falsifiable check.
    - If typing or hotkeys are used, you MUST focus the target window first.
    - If the goal is to open/play a specific item among search results (ambiguous selection),
      the plan MUST either:
        (a) use a deterministic selection method AND verify with >=2 independent signals, OR
        (b) insert an explicit step with action "await_user_input" to ask the human to confirm/select.
    - For media playback intents, verification MUST use >=2 independent signals (e.g., title + playing/progress).

    Output: JSON only (same schema as P0_CHAT).
    """
).strip()

P2_OPS_SAFE = textwrap.dedent(
    """
    You are AJAX's Planning Brain.
    Convert the user's intent into a SAFE, EXECUTABLE PLAN.

    Planner contract:
    - Use ONLY actions from actions_catalog.actions.
    - Every step is a complete task: preconditions + action + evidence_required + success_spec + on_fail.
    - success_spec.expected_state must contain at least one falsifiable check.
    - Act then verify: [LAUNCH] -> [VERIFY OPEN] -> [FOCUS] -> [ACT] -> [VERIFY RESULT].
    - Typing or hotkeys require a prior window.focus for the target window.
    - For media/search/navigation intents: use a browser from INSTALLED_APPS and keep provider-agnostic navigation.
    - Always define success_contract (telemetry-first, vision only if needed).
    - If selecting a specific result/item is ambiguous, include "await_user_input" or provide deterministic selection + 2-signal verification.
    - For media playback, verify success with >=2 independent signals (title + playing/progress).

    Output: JSON only (same schema as P0_CHAT).
    """
).strip()

MAPPER_PROMPT = textwrap.dedent(
    """
    You are AJAX's Skill Mapper.
    Select the single best skill_id from the provided candidates and fill slots if needed.

    Output:
    - Return ONLY valid JSON (no extra text).
    - Format:
      {
        "skill_id": "candidate_id",
        "slots": { "slot_name": "value" },
        "confidence": 0.0,
        "notes": "short rationale"
      }

    Rules:
    - Choose ONLY from the candidates list.
    - Do NOT output a plan or steps.
    - If information is missing, leave the slot empty and lower confidence.
    """
).strip()

_PACK_LEVELS = ["P0_CHAT", "P1_OPS_STANDARD", "P2_OPS_SAFE"]

_PACK_SYSTEM = {
    "P0_CHAT": P0_CHAT,
    "P1_OPS_STANDARD": P1_OPS_STANDARD,
    "P2_OPS_SAFE": P2_OPS_SAFE,
}

_PACK_CONTEXT_BUDGET = {
    "P0_CHAT": 2400,
    "P1_OPS_STANDARD": 5200,
    "P2_OPS_SAFE": 9000,
}

_ESCALATION_CONFIDENCE_T = 0.6
_DEESCALATION_CONFIDENCE_T = 0.7
_DEESCALATION_STREAK_REQUIRED = 2


def _compose_system_prompt(pack_id: str) -> str:
    pack = _PACK_SYSTEM.get(pack_id) or P2_OPS_SAFE
    if AJAX_METHOD_PACK:
        return AJAX_METHOD_PACK + "\n\n" + pack
    return pack


def mapper_system_prompt() -> str:
    if AJAX_METHOD_PACK:
        return AJAX_METHOD_PACK + "\n\n" + MAPPER_PROMPT
    return MAPPER_PROMPT


def _collect_action_tags(actions_payload: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    actions = actions_payload.get("actions") if isinstance(actions_payload, dict) else []
    if not isinstance(actions, list):
        return tags
    for action in actions:
        if not isinstance(action, dict):
            continue
        raw = action.get("tags") or []
        if isinstance(raw, list):
            for tag in raw:
                if isinstance(tag, str) and tag.strip():
                    tags.append(tag.strip().lower())
    cleaned: List[str] = []
    seen = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
    return cleaned


def _filter_actions_by_tags(actions_payload: Dict[str, Any], allowed_tags: set[str]) -> Dict[str, Any]:
    actions = actions_payload.get("actions") if isinstance(actions_payload, dict) else []
    if not isinstance(actions, list):
        return {"actions": []}
    filtered = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        tags = action.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tag_set = {str(t).strip().lower() for t in tags if isinstance(t, str) and t.strip()}
        if tag_set & allowed_tags:
            filtered.append(action)
    return {"actions": filtered}


def select_prompt_pack(
    *,
    intent: str,
    intent_class: str,
    action_tags: List[str],
    risk_score: float,
    needs_vision: bool,
    confidence: Optional[float],
    fail_count: int,
) -> Dict[str, Any]:
    intent_class = str(intent_class or "").strip().lower()
    pack_id = "P1_OPS_STANDARD"
    if risk_score >= 0.7 or needs_vision:
        pack_id = "P2_OPS_SAFE"

    escalation_reason = None
    if confidence is not None and confidence < _ESCALATION_CONFIDENCE_T:
        escalation_reason = "low_confidence"
    elif fail_count >= 1:
        escalation_reason = "prior_fail"

    base_reason = "intent_risk"
    return {
        "pack_id": pack_id,
        "context_budget": _PACK_CONTEXT_BUDGET.get(pack_id, 9000),
        "escalation_reason": escalation_reason,
        "base_reason": base_reason,
    }


def _pack_rank(pack_id: str) -> int:
    try:
        return _PACK_LEVELS.index(pack_id)
    except ValueError:
        return len(_PACK_LEVELS) - 1


def _apply_pack_hysteresis(
    *,
    base_pack_id: str,
    last_pack_id: Optional[str],
    confidence: Optional[float],
    deescalate_streak: int,
    escalation_active: bool,
    risk_score: float,
) -> Dict[str, Any]:
    last_id = last_pack_id or base_pack_id
    base_rank = _pack_rank(base_pack_id)
    last_rank = _pack_rank(last_id)
    streak = int(deescalate_streak or 0)
    decision = "keep"
    reason = "hysteresis_hold"

    if not escalation_active:
        return {
            "pack_id": base_pack_id,
            "decision": "keep",
            "reason": "base_pack_fresh",
            "next_state": {"last_pack_id": base_pack_id, "deescalate_streak": 0, "escalation_active": False},
        }

    if confidence is None:
        if risk_score < 0.4 and base_rank < last_rank:
            return {
                "pack_id": base_pack_id,
                "decision": "deescalate",
                "reason": "confidence_missing_low_risk",
                "next_state": {"last_pack_id": base_pack_id, "deescalate_streak": 0, "escalation_active": False},
            }
        return {
            "pack_id": last_id,
            "decision": "keep",
            "reason": "confidence_missing",
            "next_state": {"last_pack_id": last_id, "deescalate_streak": 0, "escalation_active": True},
        }

    if confidence >= _DEESCALATION_CONFIDENCE_T:
        streak += 1
    else:
        streak = 0

    if base_rank > last_rank:
        decision = "escalate"
        reason = "base_pack_higher"
        chosen = base_pack_id
    elif base_rank < last_rank:
        if streak >= _DEESCALATION_STREAK_REQUIRED:
            decision = "deescalate"
            reason = "confidence_stable"
            chosen = base_pack_id
        else:
            decision = "keep"
            reason = "confidence_streak_pending"
            chosen = last_id
    else:
        decision = "keep"
        reason = "base_pack_same"
        chosen = base_pack_id

    next_state = {
        "last_pack_id": chosen,
        "deescalate_streak": streak,
        "escalation_active": decision != "deescalate" and base_rank < last_rank,
    }
    return {"pack_id": chosen, "decision": decision, "reason": reason, "next_state": next_state}


def build_brain_prompts(brain_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye prompts system/user para el Brain planner.
    - system: reglas, formato de salida (JSON plan_id + steps)
    - user: brain_input serializado (intention, capabilities, actions_catalog, etc.) + feedback de errores previos.
    """
    prev_errors = brain_input.get("previous_errors") or []
    fail_count = brain_input.get("fail_count")
    if not isinstance(fail_count, int):
        fail_count = len(prev_errors) if isinstance(prev_errors, list) else 0
    actions_payload = brain_input.get("actions_catalog") if isinstance(brain_input.get("actions_catalog"), dict) else {}
    raw_action_tags = brain_input.get("action_tags")
    if isinstance(raw_action_tags, list):
        action_tags = [str(t).strip().lower() for t in raw_action_tags if isinstance(t, str) and t.strip()]
    else:
        action_tags = _collect_action_tags(actions_payload)
    needs_vision = False
    actions_list = actions_payload.get("actions") if isinstance(actions_payload, dict) else []
    if isinstance(actions_list, list):
        needs_vision = any(bool(a.get("vision_required")) for a in actions_list if isinstance(a, dict))
    risk_score = brain_input.get("risk_score")
    if not isinstance(risk_score, (int, float)):
        risk_score = 0.5
    confidence = brain_input.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None

    intent_class = str(brain_input.get("intent_class") or "").strip().lower()
    selection = select_prompt_pack(
        intent=str(brain_input.get("intention") or ""),
        intent_class=intent_class,
        action_tags=action_tags,
        risk_score=float(risk_score),
        needs_vision=bool(needs_vision),
        confidence=confidence,
        fail_count=fail_count,
    )

    escalation_reason = selection.get("escalation_reason")
    if isinstance(prev_errors, list):
        for err in prev_errors:
            low = str(err).lower()
            if any(
                tok in low
                for tok in [
                    "invalid_json",
                    "invalid json",
                    "json inválido",
                    "json invalido",
                    "parse_error",
                    "invalid_brain_plan",
                ]
            ):
                escalation_reason = "invalid_json"
                break
            if any(tok in low for tok in ["empty_plan", "no_plan", "no plan", "empty_steps"]):
                escalation_reason = "no_plan"
                break

    base_pack_id = str(selection.get("pack_id") or "P2_OPS_SAFE")
    context_budget = int(selection.get("context_budget") or 9000)
    decision = "keep"
    decision_reason = str(selection.get("base_reason") or "default")
    pack_state = brain_input.get("prompt_pack_state") if isinstance(brain_input.get("prompt_pack_state"), dict) else {}
    last_pack_id = pack_state.get("last_pack_id") if isinstance(pack_state, dict) else None
    deescalate_streak = pack_state.get("deescalate_streak") if isinstance(pack_state, dict) else 0
    escalation_active = bool(pack_state.get("escalation_active")) if isinstance(pack_state, dict) else False
    pack_id = base_pack_id
    next_state: Dict[str, Any] = {
        "last_pack_id": pack_id,
        "deescalate_streak": int(deescalate_streak or 0),
        "escalation_active": False,
    }

    if escalation_reason:
        try:
            idx = _pack_rank(base_pack_id)
            if idx < len(_PACK_LEVELS) - 1:
                pack_id = _PACK_LEVELS[idx + 1]
        except Exception:
            pack_id = "P2_OPS_SAFE"
        decision = "escalate"
        decision_reason = escalation_reason
        next_state = {"last_pack_id": pack_id, "deescalate_streak": 0, "escalation_active": True}
    else:
        hysteresis = _apply_pack_hysteresis(
            base_pack_id=base_pack_id,
            last_pack_id=str(last_pack_id) if last_pack_id else None,
            confidence=confidence,
            deescalate_streak=int(deescalate_streak or 0),
            escalation_active=escalation_active,
            risk_score=float(risk_score),
        )
        pack_id = hysteresis.get("pack_id") or base_pack_id
        decision = hysteresis.get("decision") or "keep"
        decision_reason = hysteresis.get("reason") or decision_reason
        next_state = hysteresis.get("next_state") or next_state

    system_prompt = _compose_system_prompt(pack_id)

    filters: Dict[str, Any] = {"mode": "none"}
    if pack_id == "P1_OPS_STANDARD":
        allowed_tags = {"ui_basic", "web_nav", "input"}
        actions_payload = _filter_actions_by_tags(actions_payload, allowed_tags)
        filters = {"mode": "tag_allowlist", "allowed_tags": sorted(allowed_tags)}

    # Planning prompt must stay compact: do NOT serialize full runtime state.
    # Only pass intent (1-2 lines), constraints, and the ActionCatalog (possibly filtered by pack).
    payload = {
        "intention": str(brain_input.get("intention") or "").strip(),
        "intent_class": str(brain_input.get("intent_class") or "").strip().lower(),
        "constraints": brain_input.get("constraints") if isinstance(brain_input.get("constraints"), dict) else {},
        "observation": brain_input.get("observation") if isinstance(brain_input.get("observation"), dict) else {},
        "actions_catalog": actions_payload,
    }
    payload["prompt_pack"] = {
        "id": pack_id,
        "context_budget": context_budget,
        "escalation_reason": escalation_reason,
        "decision": decision,
        "reason": decision_reason,
    }
    user_parts = [json.dumps(payload, ensure_ascii=False)]
    if prev_errors:
        user_parts.append(
            "Previous plans were rejected for these reasons:\n"
            + "\n".join(f"- {e}" for e in prev_errors)
            + "\nRegenerate a new plan fixing ALL these violations."
        )
    if brain_input.get("feedback"):
        user_parts.append(f"Feedback from last execution: {brain_input.get('feedback')}")
    installed_apps = brain_input.get("installed_apps") or []
    if installed_apps:
        # Keep deterministic + bounded.
        apps = [str(a) for a in installed_apps if isinstance(a, str)]
        user_parts.append(f"INSTALLED_APPS: {apps[:30]}")
    user_prompt = "\n\n".join(user_parts)
    return {
        "system": system_prompt,
        "user": user_prompt,
        "pack_id": pack_id,
        "context_budget": context_budget,
        "escalation_reason": escalation_reason,
        "decision": decision,
        "reason": decision_reason,
        "filters": filters,
        "confidence": confidence,
        "pack_state": next_state,
    }
