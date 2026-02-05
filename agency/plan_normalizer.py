from __future__ import annotations

from typing import Any, Dict, List, Tuple


_KNOWN_STEP_KEYS = {
    "id",
    "intent",
    "preconditions",
    "action",
    "args",
    "evidence_required",
    "success_spec",
    "on_fail",
}


def normalize_plan(plan: Any, mission_intent: str, *, allow_non_abort: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """
    Deterministic Plan Normalizer (non-ad-hoc).
    - Accepts minor provider variations and produces a schema-compatible draft.
    - Returns (normalized_plan, meta) where meta describes performed edits.
    """
    meta: Dict[str, Any] = {"changed": False, "edits": [], "coerced_fields": []}
    if not isinstance(plan, dict):
        return plan, meta

    out: Dict[str, Any] = dict(plan)
    edits: List[str] = []
    coerced_fields: List[str] = []

    # Accept top-level plan[] alias.
    if "steps" not in out and isinstance(out.get("plan"), list):
        out["steps"] = out.get("plan")
        out.pop("plan", None)
        edits.append("top_level_plan_to_steps")

    steps = out.get("steps")
    if not isinstance(steps, list):
        return out, {"changed": bool(edits), "edits": edits}

    norm_steps: List[Any] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            norm_steps.append(step)
            continue

        step_out = dict(step)

        # Accept action dict shape: {action:{name,args}}.
        if isinstance(step_out.get("action"), dict):
            action_obj = step_out.get("action") or {}
            if isinstance(action_obj, dict) and isinstance(action_obj.get("name"), str) and action_obj.get("name").strip():
                step_out["action"] = action_obj["name"].strip()
                if "args" not in step_out and isinstance(action_obj.get("args"), dict):
                    step_out["args"] = dict(action_obj["args"])
                edits.append(f"step_{idx+1}_action_object_normalized")

        # Accept alternative key: name -> action.
        if "action" not in step_out and isinstance(step_out.get("name"), str) and step_out.get("name").strip():
            step_out["action"] = step_out.pop("name").strip()
            edits.append(f"step_{idx+1}_name_to_action")

        if not isinstance(step_out.get("id"), str) or not str(step_out.get("id") or "").strip():
            step_out["id"] = f"step_{idx+1}"
            edits.append(f"step_{idx+1}_id_filled")

        if not isinstance(step_out.get("intent"), str) or not str(step_out.get("intent") or "").strip():
            action = str(step_out.get("action") or "step").strip()
            step_out["intent"] = f"{mission_intent} / {action}"
            edits.append(f"step_{idx+1}_intent_filled")

        # Safety invariant: schema requires per-step on_fail == "abort".
        # Deterministic coercion unless explicitly allowed.
        if not allow_non_abort:
            if step_out.get("on_fail") != "abort":
                step_out["on_fail"] = "abort"
                edits.append(f"step_{idx+1}_coerce_on_fail_abort")
                coerced_fields.append("on_fail")

        args_missing = "args" not in step_out or not isinstance(step_out.get("args"), dict)
        if args_missing:
            step_out["args"] = {}
            moved = 0
            for key in list(step_out.keys()):
                if key in _KNOWN_STEP_KEYS:
                    continue
                step_out["args"][key] = step_out.pop(key)
                moved += 1
            if moved:
                edits.append(f"step_{idx+1}_moved_extra_keys_into_args:{moved}")
            else:
                edits.append(f"step_{idx+1}_args_defaulted")

        norm_steps.append(step_out)

    # Robustness fallback (deterministic, non-ad-hoc):
    # If a plan submits a search and then "Enter" to choose among results without deterministic selection,
    # insert an explicit await_user_input step.
    try:
        has_await = any(isinstance(s, dict) and str(s.get("action") or "").strip() == "await_user_input" for s in norm_steps)
        if not has_await:
            def _is_search_submit(s: Dict[str, Any]) -> bool:
                if str(s.get("action") or "").strip() != "keyboard.type":
                    return False
                args = s.get("args") if isinstance(s.get("args"), dict) else {}
                return bool(args.get("submit") is True)

            def _is_enter(s: Dict[str, Any]) -> bool:
                action = str(s.get("action") or "").strip()
                if action == "keyboard.hotkey":
                    args = s.get("args") if isinstance(s.get("args"), dict) else {}
                    keys = args.get("keys")
                    if isinstance(keys, list):
                        keys_n = [str(k).strip().lower() for k in keys if isinstance(k, str) and k.strip()]
                        return keys_n == ["enter"] or keys_n == ["return"]
                if action == "keyboard.type":
                    args = s.get("args") if isinstance(s.get("args"), dict) else {}
                    return bool(args.get("submit") is True)
                return False

            def _signal_count(expected_state: Any) -> int:
                if not isinstance(expected_state, dict):
                    return 0
                n = 0
                if isinstance(expected_state.get("windows"), list) and expected_state.get("windows"):
                    n += 1
                if isinstance(expected_state.get("files"), list) and expected_state.get("files"):
                    n += 1
                meta = expected_state.get("meta")
                if isinstance(meta, dict) and meta.get("must_be_active") is True:
                    n += 1
                checks = expected_state.get("checks")
                if isinstance(checks, list):
                    n += min(2, len(checks))
                return n

            search_seen = any(isinstance(s, dict) and _is_search_submit(s) for s in norm_steps)
            for i, s in enumerate(list(norm_steps)):
                if not isinstance(s, dict):
                    continue
                if _is_enter(s) and search_seen:
                    succ = s.get("success_spec") if isinstance(s.get("success_spec"), dict) else {}
                    expected_state = succ.get("expected_state")
                    # If verification is not robust (>=2 signals), require human selection.
                    if _signal_count(expected_state) < 2:
                        insert = {
                            "id": f"step_{i+1}_await_user_input",
                            "intent": f"{mission_intent} / await_user_input",
                            "preconditions": {"expected_state": {}},
                            "action": "await_user_input",
                            "args": {"prompt": "Selección ambigua: confirma el resultado exacto a abrir/reproducir (pega enlace o indica cuál)."},
                            "evidence_required": [],
                            "success_spec": {"expected_state": {}},
                            "on_fail": "abort",
                        }
                        norm_steps.insert(i, insert)
                        edits.append("inserted_await_user_input_for_ambiguous_selection")
                        break
    except Exception:
        pass

    if edits:
        meta["changed"] = True
        meta["edits"] = edits
    if coerced_fields:
        # de-dup while preserving order
        seen = set()
        ordered: List[str] = []
        for f in coerced_fields:
            if f in seen:
                continue
            seen.add(f)
            ordered.append(f)
        meta["coerced_fields"] = ordered
    out["steps"] = norm_steps
    return out, meta
