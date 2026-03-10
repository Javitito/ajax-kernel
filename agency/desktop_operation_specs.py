from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from agency.desktop_verify_contract import (
    build_desktop_mismatch,
    build_desktop_verification_result,
    build_desktop_verify_input,
    normalize_desktop_mismatches,
    normalize_desktop_verification_result,
)


OPERATION_CLASSES: Dict[str, Dict[str, Any]] = {
    "open_app": {
        "class": "open_app",
        "allowed_action_types": ["launch_test_app"],
        "expected_templates": ["app_started", "window_open_and_focused"],
        "verify_spec": "verify.window_open_and_focused.v0",
        "undo_spec": "undo.close_window_safe.v0",
    },
    "focus_window": {
        "class": "focus_window",
        "allowed_action_types": ["focus_window"],
        "expected_templates": ["window_open_and_focused"],
        "verify_spec": "verify.window_open_and_focused.v0",
        "undo_spec": "undo.refocus_previous_window.v0",
    },
    "type_text": {
        "class": "type_text",
        "allowed_action_types": ["type_text"],
        "expected_templates": ["text_present_in_window"],
        "verify_spec": "verify.text_present_in_window.v0",
        "undo_spec": "undo.type_text.v0",
    },
    "close_window_safe": {
        "class": "close_window_safe",
        "allowed_action_types": ["close_test_app"],
        "expected_templates": ["dialog_closed", "expected_process_state"],
        "verify_spec": "verify.process_not_foreground.v0",
        "undo_spec": "undo.reopen_test_app.v0",
    },
    "dismiss_dialog": {
        "class": "dismiss_dialog",
        "allowed_action_types": ["safe_hotkey"],
        "expected_templates": ["dialog_closed"],
        "verify_spec": "verify.dialog_closed.v0",
        "undo_spec": "undo.dismiss_dialog.v0",
    },
}

VERIFY_SPECS: Dict[str, Dict[str, Any]] = {
    "open_app": {
        "name": "verify.window_open_and_focused.v0",
        "evidence_required": ["post_screenshot", "desktop_arbiter_post", "runtime_metadata.active_window"],
        "pass_when": [
            "post_action_verdict=pass",
            "active_window_title_contains or visual_markers_any matches the expected app",
        ],
        "fail_when": [
            "post_action_verdict=fail",
            "expected app markers are absent after launch",
        ],
        "uncertain_when": [
            "post_action_verdict=uncertain",
            "runtime metadata is incomplete for the foreground window",
        ],
    },
    "focus_window": {
        "name": "verify.window_open_and_focused.v0",
        "evidence_required": ["post_screenshot", "desktop_arbiter_post", "runtime_metadata.active_window"],
        "pass_when": [
            "post_action_verdict=pass",
            "foreground window matches the requested process/title",
        ],
        "fail_when": [
            "post_action_verdict=fail",
            "focus target mismatch persists after the focus request",
        ],
        "uncertain_when": [
            "post_action_verdict=uncertain",
            "multiple candidate windows could satisfy the title hint",
        ],
    },
    "type_text": {
        "name": "verify.text_present_in_window.v0",
        "evidence_required": ["post_screenshot", "desktop_arbiter_post", "runtime_metadata.find_text"],
        "pass_when": [
            "post_action_verdict=pass",
            "visible_text_any contains the expected typed marker",
        ],
        "fail_when": [
            "post_action_verdict=fail",
            "typed text is missing from the focused LAB window",
        ],
        "uncertain_when": [
            "post_action_verdict=uncertain",
            "text probe could not be observed from current metadata",
        ],
    },
    "close_window_safe": {
        "name": "verify.process_not_foreground.v0",
        "evidence_required": ["post_screenshot", "desktop_arbiter_post", "runtime_metadata.active_window"],
        "pass_when": [
            "post_action_verdict=pass",
            "visual_markers_absent confirms the test app is no longer foreground",
        ],
        "fail_when": [
            "post_action_verdict=fail",
            "closed window markers remain in the foreground",
        ],
        "uncertain_when": [
            "post_action_verdict=uncertain",
            "the app may still run in background without a visible foreground window",
        ],
    },
    "dismiss_dialog": {
        "name": "verify.dialog_closed.v0",
        "evidence_required": ["post_screenshot", "desktop_arbiter_post"],
        "pass_when": [
            "post_action_verdict=pass",
            "dialogs_absent evaluates true after the dismissal attempt",
        ],
        "fail_when": [
            "post_action_verdict=fail",
            "dialog markers are still present after the dismissal attempt",
        ],
        "uncertain_when": [
            "post_action_verdict=uncertain",
            "dialog state could not be confirmed from the screenshot",
        ],
    },
}

UNDO_SPECS: Dict[str, Dict[str, Any]] = {
    "open_app": {
        "name": "undo.close_window_safe.v0",
        "possible": True,
        "reversible": True,
        "suggested_steps": [
            "Focus the opened LAB test app window.",
            "Close it without saving.",
            "Re-run desktop scout before any further actuation.",
        ],
        "safety_limits": [
            "Do not save changes while unwinding a LAB launch.",
            "Do not reuse the same window without a fresh PREPARE pass.",
        ],
    },
    "focus_window": {
        "name": "undo.refocus_previous_window.v0",
        "possible": True,
        "reversible": True,
        "suggested_steps": [
            "Re-focus the previous LAB window if the focus moved unexpectedly.",
            "Capture a fresh screenshot before attempting another action.",
        ],
        "safety_limits": [
            "Do not chain more focus hops without a new VERIFY result.",
        ],
    },
    "type_text": {
        "name": "undo.type_text.v0",
        "possible": True,
        "reversible": True,
        "suggested_steps": [
            "If the field still supports it, send Ctrl+Z manually.",
            "If the text landed in Notepad, close without saving.",
            "Capture a fresh screenshot before retrying another action.",
        ],
        "safety_limits": [
            "Do not type secrets or credentials into the LAB target.",
            "Do not assume Ctrl+Z worked without a fresh VERIFY pass.",
        ],
    },
    "close_window_safe": {
        "name": "undo.reopen_test_app.v0",
        "possible": True,
        "reversible": True,
        "suggested_steps": [
            "Re-launch the LAB test app if the session must continue.",
            "Rebuild focus and expected EFE before typing again.",
        ],
        "safety_limits": [
            "Only reopen the explicit LAB allowlisted app.",
            "Do not restore prior content without a fresh PREPARE pass.",
        ],
    },
    "dismiss_dialog": {
        "name": "undo.dismiss_dialog.v0",
        "possible": False,
        "reversible": False,
        "suggested_steps": [
            "Recapture the desktop state immediately.",
            "If the dialog remained, re-run scout and arbiter before trying another dismissal.",
        ],
        "safety_limits": [
            "Do not escalate to broader hotkeys without explicit allowlist support.",
        ],
    },
}

GENERIC_VERIFY_SPECS: Dict[str, Dict[str, Any]] = {
    "click_coordinates": {
        "name": "verify.post_arbiter_only.v0",
        "evidence_required": ["post_screenshot", "desktop_arbiter_post"],
        "pass_when": ["post_action_verdict=pass"],
        "fail_when": ["post_action_verdict=fail"],
        "uncertain_when": ["post_action_verdict=uncertain"],
    },
    "click_named_target": {
        "name": "verify.post_arbiter_only.v0",
        "evidence_required": ["post_screenshot", "desktop_arbiter_post"],
        "pass_when": ["post_action_verdict=pass"],
        "fail_when": ["post_action_verdict=fail"],
        "uncertain_when": ["post_action_verdict=uncertain"],
    },
    "safe_hotkey": {
        "name": "verify.post_arbiter_only.v0",
        "evidence_required": ["post_screenshot", "desktop_arbiter_post"],
        "pass_when": ["post_action_verdict=pass"],
        "fail_when": ["post_action_verdict=fail"],
        "uncertain_when": ["post_action_verdict=uncertain"],
    },
}

GENERIC_UNDO_SPECS: Dict[str, Dict[str, Any]] = {
    "click_coordinates": {
        "name": "undo.click_coordinates.v0",
        "possible": False,
        "reversible": False,
        "suggested_steps": [
            "Recapture the desktop state immediately.",
            "If a modal opened, dismiss it only after a fresh scout/arbiter pass.",
        ],
        "safety_limits": [
            "Do not chain more clicks without a fresh VERIFY result.",
        ],
    },
    "click_named_target": {
        "name": "undo.click_named_target.v0",
        "possible": False,
        "reversible": False,
        "suggested_steps": [
            "Recapture the desktop state immediately.",
            "If a modal opened, dismiss it only after a fresh scout/arbiter pass.",
        ],
        "safety_limits": [
            "Do not chain more clicks without a fresh VERIFY result.",
        ],
    },
    "safe_hotkey": {
        "name": "undo.safe_hotkey.v0",
        "possible": True,
        "reversible": True,
        "suggested_steps": [
            "Recapture the desktop state.",
            "If focus moved unexpectedly, re-focus the intended test window.",
            "Do not chain another hotkey until VERIFY is explicit.",
        ],
        "safety_limits": [
            "Stay within the explicit safe-hotkey allowlist.",
        ],
    },
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for item in values:
        if item and item not in out:
            out.append(item)
    return out


def _process_stem(process: Any) -> str:
    text = str(process or "").strip().lower()
    if text.endswith(".exe"):
        text = text[:-4]
    return text


def _normalize_hotkeys(keys: Any) -> tuple[str, ...]:
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        return tuple()
    return tuple(str(item).strip().lower() for item in keys if str(item).strip())


def _merge_expected_efe(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    override_n = _as_dict(override)
    for key, value in override_n.items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = _dedupe([str(item) for item in merged.get(key) or []] + [str(item) for item in value])
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_expected_efe_template(template_name: str, **kwargs: Any) -> Dict[str, Any]:
    name = str(template_name or "").strip()
    process = str(kwargs.get("process") or "").strip().lower()
    title = str(kwargs.get("title_contains") or "").strip().lower()
    focus_target = str(kwargs.get("focus_target") or "").strip().lower()
    text_value = str(kwargs.get("text") or "").strip()
    dialog_name = str(kwargs.get("dialog_name") or "").strip().lower()
    state = str(kwargs.get("state") or "running").strip().lower()

    if name == "window_open_and_focused":
        payload: Dict[str, Any] = {}
        if title:
            payload["active_window_title_contains"] = title
        if process:
            payload["visual_markers_any"] = [process]
        if focus_target:
            payload["focus_target_contains"] = focus_target
        return payload

    if name == "text_present_in_window":
        payload = build_expected_efe_template(
            "window_open_and_focused",
            process=process,
            title_contains=title,
            focus_target=focus_target,
        )
        if text_value:
            payload["visible_text_any"] = [text_value]
        return payload

    if name == "app_started":
        payload = {}
        if process:
            payload["visual_markers_any"] = [process]
        if title:
            payload["active_window_title_contains"] = title
        return payload

    if name == "dialog_visible":
        payload = {}
        if dialog_name:
            payload["dialogs_present"] = [dialog_name]
        if title:
            payload["active_window_title_contains"] = title
        return payload

    if name == "dialog_closed":
        payload = {"dialogs_absent": True}
        if title:
            payload["active_window_title_contains"] = title
        return payload

    if name == "expected_process_state":
        if not process:
            return {}
        if state in {"closed", "stopped", "not_foreground"}:
            return {"visual_markers_absent": [process]}
        return {"visual_markers_any": [process]}

    return {}


def derive_operation_class(
    action_type: str,
    *,
    target: Optional[Dict[str, Any]] = None,
    hotkeys: Any = None,
    app: Optional[str] = None,
) -> Optional[str]:
    action_n = str(action_type or "").strip().lower()
    target_n = _as_dict(target)
    hotkeys_n = _normalize_hotkeys(hotkeys or target_n.get("keys"))
    process = str(target_n.get("process") or app or "").strip().lower()

    if action_n == "launch_test_app" and process:
        return "open_app"
    if action_n == "focus_window":
        return "focus_window"
    if action_n == "type_text":
        return "type_text"
    if action_n == "close_test_app" and process:
        return "close_window_safe"
    if action_n == "safe_hotkey" and hotkeys_n in {("esc",), ("enter",)}:
        return "dismiss_dialog"
    return None


def _context_for_operation_class(
    operation_class: str,
    *,
    action_type: str,
    target: Optional[Dict[str, Any]] = None,
    text: Optional[str] = None,
    hotkeys: Any = None,
    app: Optional[str] = None,
) -> Dict[str, Any]:
    target_n = _as_dict(target)
    process = str(target_n.get("process") or app or "").strip().lower()
    title = str(target_n.get("title_contains") or "").strip().lower() or _process_stem(process)
    focus_target = str(target_n.get("focus_target") or "").strip().lower()
    if operation_class == "type_text" and not focus_target:
        focus_target = "editor"
    dialog_name = str(target_n.get("dialog_name") or target_n.get("name") or "dialog").strip().lower()
    ctx: Dict[str, Any] = {
        "action_type": str(action_type or "").strip().lower(),
        "process": process,
        "title_contains": title,
        "focus_target": focus_target,
        "text": str(text or "").strip(),
        "dialog_name": dialog_name,
        "hotkeys": list(_normalize_hotkeys(hotkeys or target_n.get("keys"))),
    }
    if operation_class == "close_window_safe":
        ctx["state"] = "not_foreground"
    return ctx


def _materialize_expected_efe(operation_class: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
    if not operation_class:
        return {}
    spec = OPERATION_CLASSES.get(operation_class)
    if not isinstance(spec, dict):
        return {}
    merged: Dict[str, Any] = {}
    for template_name in spec.get("expected_templates") or []:
        merged = _merge_expected_efe(merged, build_expected_efe_template(str(template_name), **context))
    return merged


def _materialize_verify_spec(operation_class: Optional[str], *, action_type: str) -> Dict[str, Any]:
    if operation_class and operation_class in VERIFY_SPECS:
        return copy.deepcopy(VERIFY_SPECS[operation_class])
    return copy.deepcopy(GENERIC_VERIFY_SPECS.get(str(action_type or "").strip().lower(), {}))


def _materialize_undo_info(operation_class: Optional[str], *, action_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
    info = copy.deepcopy(UNDO_SPECS.get(operation_class or "", GENERIC_UNDO_SPECS.get(str(action_type or "").strip().lower(), {})))
    if not isinstance(info, dict):
        return {"possible": False, "suggested_steps": []}
    process = str(context.get("process") or "").strip().lower()
    if process and info.get("name") == "undo.reopen_test_app.v0":
        info["suggested_steps"] = [
            f"Re-launch {process} if the LAB task must continue.",
            "Rebuild focus and expected EFE before typing again.",
        ]
    return info


def materialize_expected_efe_desktop(operation_class: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
    return _materialize_expected_efe(operation_class, _as_dict(context))


def _extract_visual_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = _as_dict(state)
    runtime = _as_dict(raw.get("runtime_metadata"))
    metadata = _as_dict(raw.get("metadata"))
    post_arbiter = _as_dict(raw.get("post_arbiter"))
    raw_post_result = post_arbiter.get("verify_result") or post_arbiter.get("verification_result")
    post_result = (
        normalize_desktop_verification_result(raw_post_result)
        if isinstance(raw_post_result, dict) and raw_post_result
        else {}
    )
    dialogs = _as_text_list(raw.get("dialogs") or metadata.get("dialogs"))
    affordances = _as_text_list(raw.get("observed_affordances") or metadata.get("observed_affordances") or metadata.get("ui_affordances"))
    markers = _as_text_list(
        raw.get("visual_markers")
        or runtime.get("visual_markers")
        or metadata.get("visual_markers")
        or metadata.get("markers")
    )
    visible_text = _as_text_list(raw.get("visible_text") or runtime.get("visible_text") or metadata.get("visible_text"))
    return {
        "active_window_title": str(
            raw.get("active_window_title")
            or runtime.get("active_window_title")
            or metadata.get("active_window_title")
            or ""
        ).strip(),
        "focus_target": str(raw.get("focus_target") or metadata.get("focus_target") or "").strip(),
        "dialogs": dialogs,
        "observed_affordances": affordances,
        "markers": markers,
        "metadata": {"visible_text": visible_text},
        "post_action_verdict": str(
            raw.get("post_action_verdict")
            or post_result.get("verdict")
            or post_arbiter.get("post_action_verdict")
            or ""
        ).strip().lower(),
        "screenshot_path": str(raw.get("screenshot_path") or raw.get("png_path") or "").strip(),
    }


def _expected_contains(actual: str, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_expected_contains(actual, item) for item in expected)
    text = str(expected or "").strip().lower()
    if not text:
        return False
    return text in actual.lower()


def _evaluate_expected_efe(expected_efe_desktop: Dict[str, Any], visual: Dict[str, Any]) -> list[Dict[str, Any]]:
    mismatches: list[Dict[str, Any]] = []
    active_window = str(visual.get("active_window_title") or "")
    focus_target = str(visual.get("focus_target") or "")
    dialogs = visual.get("dialogs") if isinstance(visual.get("dialogs"), list) else []
    observed_affordances = visual.get("observed_affordances") if isinstance(visual.get("observed_affordances"), list) else []
    markers = visual.get("markers") if isinstance(visual.get("markers"), list) else []
    visible_text = _as_text_list((visual.get("metadata") or {}).get("visible_text") if isinstance(visual.get("metadata"), dict) else None)

    expected_title = expected_efe_desktop.get("active_window_title_contains")
    if expected_title and not _expected_contains(active_window, expected_title):
        mismatches.append(
            build_desktop_mismatch(
                "active_window_title_contains",
                expected=expected_title,
                observed=active_window,
                severity="high",
                note="Foreground window title did not satisfy the expected title fragment.",
            )
        )

    expected_focus = expected_efe_desktop.get("focus_target_contains")
    if expected_focus and not _expected_contains(focus_target, expected_focus):
        mismatches.append(
            build_desktop_mismatch(
                "focus_target_contains",
                expected=expected_focus,
                observed=focus_target,
                severity="medium",
                note="Focused desktop target did not match the expected focus target hint.",
            )
        )

    if bool(expected_efe_desktop.get("dialogs_absent")) and dialogs:
        mismatches.append(
            build_desktop_mismatch(
                "dialogs_absent",
                expected=True,
                observed=dialogs,
                severity="high",
                note="A blocking dialog is still visible after the operation.",
            )
        )

    expected_dialogs = _as_text_list(expected_efe_desktop.get("dialogs_present"))
    for item in expected_dialogs:
        if not any(item.lower() in seen.lower() for seen in dialogs):
            mismatches.append(
                build_desktop_mismatch(
                    "dialogs_present",
                    expected=item,
                    observed=dialogs,
                    severity="high",
                    note="An expected dialog was not visible in the observed desktop state.",
                )
            )

    expected_affordances = _as_text_list(expected_efe_desktop.get("affordances_any"))
    if expected_affordances:
        lowered = [str(item).lower() for item in observed_affordances]
        if not any(any(expected.lower() in seen for seen in lowered) for expected in expected_affordances):
            mismatches.append(
                build_desktop_mismatch(
                    "affordances_any",
                    expected=expected_affordances,
                    observed=observed_affordances,
                    severity="medium",
                    note="No observed affordance matched the expected affordance set.",
                )
            )

    expected_markers = _as_text_list(expected_efe_desktop.get("visual_markers_any"))
    if expected_markers:
        lowered_markers = [str(item).lower() for item in markers]
        if not any(expected.lower() in seen for expected in expected_markers for seen in lowered_markers):
            mismatches.append(
                build_desktop_mismatch(
                    "visual_markers_any",
                    expected=expected_markers,
                    observed=markers,
                    severity="high",
                    note="Expected visual markers were not found after the operation.",
                )
            )

    expected_markers_absent = _as_text_list(expected_efe_desktop.get("visual_markers_absent"))
    if expected_markers_absent:
        lowered_markers = [str(item).lower() for item in markers]
        if any(expected.lower() in seen for expected in expected_markers_absent for seen in lowered_markers):
            mismatches.append(
                build_desktop_mismatch(
                    "visual_markers_absent",
                    expected=expected_markers_absent,
                    observed=markers,
                    severity="high",
                    note="A marker expected to disappear remained visible in the observed desktop state.",
                )
            )

    expected_text = _as_text_list(expected_efe_desktop.get("visible_text_any"))
    if expected_text:
        lowered_text = [str(item).lower() for item in visible_text]
        if not any(expected.lower() in seen for expected in expected_text for seen in lowered_text):
            mismatches.append(
                build_desktop_mismatch(
                    "visible_text_any",
                    expected=expected_text,
                    observed=visible_text,
                    severity="high",
                    note="Expected text was not visible in the LAB window after typing.",
                )
            )

    return mismatches


def build_verify_input(
    operation_class: Optional[str],
    context: Dict[str, Any],
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    *,
    expected_efe_desktop: Optional[Dict[str, Any]] = None,
    verify_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    operation_class_n = str(operation_class or "").strip().lower() or None
    context_n = _as_dict(context)
    expected = _merge_expected_efe(
        materialize_expected_efe_desktop(operation_class_n, context_n),
        expected_efe_desktop,
    )
    verify_spec_n = copy.deepcopy(verify_spec if isinstance(verify_spec, dict) else _materialize_verify_spec(operation_class_n, action_type=str(context_n.get("action_type") or "")))
    before_visual = _extract_visual_state(before_state)
    after_visual = _extract_visual_state(after_state)
    after_state_n = _as_dict(after_state)
    before_state_n = _as_dict(before_state)
    runtime_metadata = _as_dict(after_state_n.get("runtime_metadata"))
    arbiter_context = {
        "post_arbiter": _as_dict(after_state_n.get("post_arbiter")),
        "required_evidence": _as_text_list(verify_spec_n.get("evidence_required")),
    }
    verify_input = build_desktop_verify_input(
        operation_class=operation_class_n,
        expected_efe_desktop=expected,
        before_state=before_visual,
        after_state=after_visual,
        screenshot_before=str(before_visual.get("screenshot_path") or before_state_n.get("screenshot_path") or ""),
        screenshot_after=str(after_visual.get("screenshot_path") or after_state_n.get("screenshot_path") or ""),
        arbiter_context=arbiter_context,
        runtime_metadata=runtime_metadata,
        verify_spec=verify_spec_n,
    )
    verify_input["required_evidence"] = _as_text_list(verify_spec_n.get("evidence_required"))
    return verify_input


def _missing_required_evidence(verify_input: Dict[str, Any]) -> list[str]:
    after_state = _as_dict(verify_input.get("after_state"))
    runtime = _as_dict(verify_input.get("runtime_metadata"))
    arbiter_context = _as_dict(verify_input.get("arbiter_context"))
    post_arbiter = _as_dict(arbiter_context.get("post_arbiter"))
    post_result = _as_dict(post_arbiter.get("verification_result"))
    required = _as_text_list(verify_input.get("required_evidence"))
    missing: list[str] = []
    for item in required:
        if item == "post_screenshot" and not str(verify_input.get("screenshot_after") or after_state.get("screenshot_path") or "").strip():
            missing.append(item)
        elif item == "desktop_arbiter_post" and not str(
            post_result.get("verdict")
            or post_arbiter.get("post_action_verdict")
            or after_state.get("post_action_verdict")
            or ""
        ).strip():
            missing.append(item)
        elif item == "runtime_metadata.active_window" and not (
            str(after_state.get("active_window_title") or runtime.get("active_window_title") or "").strip()
            or _as_text_list(after_state.get("markers"))
            or _as_text_list(runtime.get("visual_markers"))
        ):
            missing.append(item)
        elif item == "runtime_metadata.find_text" and not (
            _as_text_list((_as_dict(after_state.get("metadata"))).get("visible_text"))
            or _as_text_list(runtime.get("visible_text"))
        ):
            missing.append(item)
    return missing


def _verdict_hint(reason_code: str) -> str:
    code = str(reason_code or "").strip().lower()
    if code == "ok":
        return ""
    if code == "verify_evidence_incomplete":
        return "Capture stronger post-action evidence before deciding success."
    if code == "verify_uncertain":
        return "Capture a fresh screenshot or richer runtime metadata before deciding success."
    if code == "missing_expected_efe_desktop":
        return "Materialize a concrete expected_efe_desktop before VERIFY."
    return "Inspect mismatches and tighten the LAB desktop contract before retrying."


def evaluate_desktop_verify_input(verify_input: Dict[str, Any]) -> Dict[str, Any]:
    verify_input_n = _as_dict(verify_input)
    expected = _as_dict(verify_input_n.get("expected_efe_desktop"))
    after_visual = _as_dict(verify_input_n.get("after_state"))
    arbiter_context = _as_dict(verify_input_n.get("arbiter_context"))
    post_arbiter = _as_dict(arbiter_context.get("post_arbiter"))
    raw_post_result = post_arbiter.get("verify_result") or post_arbiter.get("verification_result")
    post_result = (
        normalize_desktop_verification_result(raw_post_result)
        if isinstance(raw_post_result, dict) and raw_post_result
        else {}
    )
    post_verdict = str(
        after_visual.get("post_action_verdict")
        or post_result.get("verdict")
        or post_arbiter.get("post_action_verdict")
        or ""
    ).strip().lower()
    missing_evidence = _missing_required_evidence(verify_input_n)
    mismatches = _evaluate_expected_efe(expected, after_visual)
    mismatches.extend(
        normalize_desktop_mismatches(
            post_result.get("verify_mismatch")
            or post_result.get("mismatches")
            or post_arbiter.get("verify_mismatch")
            or post_arbiter.get("mismatches")
        )
    )
    mismatches = normalize_desktop_mismatches(mismatches)

    verdict = "uncertain"
    reason_code = "verify_uncertain"
    if not expected:
        reason_code = "missing_expected_efe_desktop"
    elif post_verdict == "fail":
        verdict = "fail"
        reason_code = "verify_fail"
    elif mismatches:
        verdict = "fail"
        reason_code = "verify_fail"
    elif missing_evidence:
        reason_code = "verify_evidence_incomplete"
    elif post_verdict == "pass":
        verdict = "pass"
        reason_code = "ok"
    elif post_verdict == "uncertain":
        reason_code = "verify_uncertain"
    else:
        reason_code = "verify_uncertain"

    confidence = "medium"
    if verdict == "pass":
        confidence = "high" if not missing_evidence else "medium"
    elif verdict == "fail":
        confidence = "high"
    elif missing_evidence:
        confidence = "low"

    return build_desktop_verification_result(
        verdict=verdict,
        mismatches=mismatches,
        reason_code=reason_code,
        next_hint=_verdict_hint(reason_code),
        confidence=confidence,
    )


def evaluate_desktop_efe(
    *,
    operation_class: Optional[str],
    context: Dict[str, Any],
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    expected_efe_desktop: Optional[Dict[str, Any]] = None,
    verify_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    verify_input = build_verify_input(
        operation_class,
        context,
        before_state,
        after_state,
        expected_efe_desktop=expected_efe_desktop,
        verify_spec=verify_spec,
    )
    expected = _as_dict(verify_input.get("expected_efe_desktop"))
    verification_result = evaluate_desktop_verify_input(verify_input)
    return {
        "expected_efe_desktop": expected,
        "verify_input": verify_input,
        "verify_result": verification_result,
        "verify_mismatch": verification_result.get("verify_mismatch") or [],
        "verification_result": verification_result,
    }


def resolve_operation_contract(
    *,
    action_type: str,
    target: Optional[Dict[str, Any]] = None,
    text: Optional[str] = None,
    hotkeys: Any = None,
    app: Optional[str] = None,
    expected_efe_desktop: Optional[Dict[str, Any]] = None,
    operation_class: Optional[str] = None,
) -> Dict[str, Any]:
    action_n = str(action_type or "").strip().lower()
    operation_class_n = str(operation_class or "").strip().lower() or None
    if operation_class_n is None:
        operation_class_n = derive_operation_class(action_n, target=target, hotkeys=hotkeys, app=app)
    context = _context_for_operation_class(
        operation_class_n or "",
        action_type=action_n,
        target=target,
        text=text,
        hotkeys=hotkeys,
        app=app,
    )
    spec_expected = materialize_expected_efe_desktop(operation_class_n, context)
    merged_expected = _merge_expected_efe(spec_expected, expected_efe_desktop)
    verify_spec = _materialize_verify_spec(operation_class_n, action_type=action_n)
    undo_info = _materialize_undo_info(operation_class_n, action_type=action_n, context=context)
    contract_source = "user_only"
    if operation_class_n:
        contract_source = "operation_class"
    elif verify_spec or undo_info:
        contract_source = "generic_action"
    return {
        "operation_class": operation_class_n,
        "context": context,
        "expected_efe_desktop_materialized": spec_expected,
        "expected_efe_desktop": merged_expected,
        "verify_spec": verify_spec,
        "undo_info": undo_info,
        "contract_source": contract_source,
    }
