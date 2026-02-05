from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def _canonicalize_evidence_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False, sort_keys=True)
    return json.dumps({"value": item}, ensure_ascii=False, sort_keys=True)


def _step_looks_legacy(step: Dict[str, Any]) -> bool:
    action = step.get("action")
    if isinstance(action, dict) and isinstance(action.get("name"), str):
        return True
    pre = step.get("preconditions")
    if isinstance(pre, list):
        return True
    ev = step.get("evidence_required")
    if isinstance(ev, list) and any(isinstance(x, dict) for x in ev):
        return True
    succ = step.get("success_spec")
    if isinstance(succ, dict) and isinstance(succ.get("expected_state"), list):
        return True
    return False


def adapt_legacy_plan(plan: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    LEGACY_PLAN_ADAPTER (deterministic) to accept older/looser provider shapes.
    Returns (adapted_plan, meta).
    """
    meta: Dict[str, Any] = {"legacy_adapt_applied": False, "edits": []}
    if not isinstance(plan, dict):
        return plan, meta

    out: Dict[str, Any] = dict(plan)
    edits: List[str] = []

    # Accept top-level alias plan[].
    if "steps" not in out and isinstance(out.get("plan"), list):
        out["steps"] = out.get("plan")
        out.pop("plan", None)
        edits.append("top_level_plan_to_steps")

    steps = out.get("steps")
    if not isinstance(steps, list):
        return out, meta

    any_legacy = False
    norm_steps: List[Any] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            norm_steps.append(step)
            continue
        if not _step_looks_legacy(step):
            norm_steps.append(step)
            continue
        any_legacy = True
        s = dict(step)

        # action: {name,args} -> action:str + args:dict
        action = s.get("action")
        if isinstance(action, dict) and isinstance(action.get("name"), str) and action.get("name").strip():
            s["action"] = action.get("name").strip()
            if isinstance(action.get("args"), dict):
                if not isinstance(s.get("args"), dict):
                    s["args"] = {}
                # Merge: step.args wins on conflicts.
                merged = dict(action.get("args") or {})
                merged.update(s.get("args") or {})
                s["args"] = merged
            edits.append(f"step_{idx+1}_action_object_to_string")

        # preconditions list -> object expected_state
        pre = s.get("preconditions")
        if isinstance(pre, list):
            if len(pre) == 0:
                s["preconditions"] = {"expected_state": {}}
                edits.append(f"step_{idx+1}_preconditions_empty_list_to_object")
            else:
                s["preconditions"] = {"expected_state": {"checks": pre}}
                edits.append(f"step_{idx+1}_preconditions_list_checks")

        # evidence_required list[dict] -> list[str] canonicalized
        ev = s.get("evidence_required")
        if isinstance(ev, list) and any(isinstance(x, dict) for x in ev):
            s["evidence_required"] = [_canonicalize_evidence_item(x) for x in ev]
            edits.append(f"step_{idx+1}_evidence_required_canonicalized")

        # success_spec.expected_state list -> object checks
        succ = s.get("success_spec")
        if isinstance(succ, dict):
            expected = succ.get("expected_state")
            if isinstance(expected, list):
                succ2 = dict(succ)
                succ2["expected_state"] = {"checks": expected}
                s["success_spec"] = succ2
                edits.append(f"step_{idx+1}_success_expected_list_to_checks")
            elif expected is None:
                succ2 = dict(succ)
                succ2["expected_state"] = {}
                s["success_spec"] = succ2
                edits.append(f"step_{idx+1}_success_expected_defaulted")
        elif succ is None:
            # If success_spec missing entirely, make it present (validator will still require checks).
            s["success_spec"] = {"expected_state": {}}
            edits.append(f"step_{idx+1}_success_spec_added")

        norm_steps.append(s)

    out["steps"] = norm_steps
    if any_legacy:
        meta["legacy_adapt_applied"] = True
        meta["edits"] = edits
    return out, meta

