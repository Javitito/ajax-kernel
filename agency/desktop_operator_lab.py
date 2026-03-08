from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agency.close_editor_safe import close_editor_safe
from agency.desktop_operation_specs import evaluate_desktop_efe, resolve_operation_contract
from agency.desktop_roles import _build_lab_gate as build_desktop_lab_gate
from agency.desktop_roles import run_desktop_arbiter, run_desktop_scout
from agency.lab_snap import capture_lab_snapshot
from agency.windows_driver_client import WindowsDriverClient, WindowsDriverError


OPERATION_SCHEMA = "ajax.desktop.operation.v1"
OPERATION_DEMO_SCHEMA = "ajax.desktop.operation_demo.v1"
ROLE_RECEIPT_SCHEMA = "ajax.desktop.role_receipt.v1"

ALLOWED_ACTIONS = {
    "click_coordinates",
    "click_named_target",
    "type_text",
    "safe_hotkey",
    "launch_test_app",
    "focus_window",
    "close_test_app",
}
ALLOWED_TEST_APPS = {"notepad.exe"}
SAFE_HOTKEYS = {
    ("esc",),
    ("tab",),
    ("enter",),
    ("ctrl", "l"),
}
SENSITIVE_TEXT_MARKERS = ("password", "token", "secret", "api_key", "credential", "login")


def _utc_now(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _ts_label(ts: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts or time.time()))


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_rail(raw: Any) -> str:
    value = str(raw or "lab").strip().lower()
    if value in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def _lab_driver_url() -> str:
    env_url = str(os.getenv("OS_DRIVER_URL_LAB") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    env_host = str(os.getenv("OS_DRIVER_HOST_LAB") or "").strip() or "127.0.0.1"
    env_port = str(os.getenv("OS_DRIVER_PORT_LAB") or "").strip() or "5012"
    return f"http://{env_host}:{env_port}"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _normalize_hotkeys(keys: Any) -> tuple[str, ...]:
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        return tuple()
    return tuple(str(item).strip().lower() for item in keys if str(item).strip())


def _reason_hint(reason_code: str) -> str:
    code = str(reason_code or "").strip().lower()
    if code == "lab_only_command":
        return "Run with --rail lab. Desktop operator remains disabled outside LAB."
    if code.startswith("expected_session") or code.startswith("session_"):
        return "Run `python bin/ajaxctl lab session init --rail lab --display dummy` and retry."
    if "anchor" in code or "display_catalog" in code or "port_" in code:
        return "Run `python bin/ajaxctl doctor anchor --rail lab` and fix the LAB anchor before retrying."
    if code in {"action_not_allowlisted", "hotkey_not_allowlisted", "app_not_allowlisted"}:
        return "Use one of the explicit LAB allowlisted actions only."
    if code in {"missing_expected_efe_desktop", "pre_verdict_not_pass"}:
        return "Provide a stricter expected_efe_desktop or fix the visual pre-check before retrying."
    if code.startswith("screenshot_"):
        return "Ensure the LAB screenshot endpoint is available or provide a current screenshot path."
    if code.startswith("apply_"):
        return "Review the allowlisted action detail and retry only after a fresh PREPARE pass."
    if code.startswith("verify_"):
        return "Inspect the post-action screenshot and tighten expected_efe_desktop before retrying."
    return "Review the operation artifact and retry only after the LAB state is stable."


def _capture_current_lab_screenshot(root_dir: Path, *, context: str) -> Dict[str, Any]:
    return capture_lab_snapshot(
        root_dir=root_dir,
        job_id="desktop_operator_lab",
        mission_id="desktop_operator_lab",
        active_window=False,
        driver_url=_lab_driver_url(),
        context=context,
    )


def _metadata_from_runtime(driver: WindowsDriverClient, *, expected_efe_desktop: Dict[str, Any], text_probe: Optional[str] = None) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        fg = driver.get_active_window()
    except Exception:
        fg = {}
    if isinstance(fg, dict):
        title = str(fg.get("title") or "").strip()
        process = str(fg.get("process") or "").strip()
        if title:
            metadata["active_window_title"] = title
        if process:
            metadata["visual_markers"] = [process]
    probe_text = text_probe
    if not probe_text:
        candidates = _as_text_list(expected_efe_desktop.get("visible_text_any"))
        probe_text = candidates[0] if candidates else None
    if probe_text:
        try:
            result = driver.find_text(probe_text)
            matches = result.get("matches") if isinstance(result.get("matches"), list) else []
            if matches:
                metadata["visible_text"] = [probe_text]
                markers = _as_text_list(metadata.get("visual_markers"))
                markers.append("text_found")
                metadata["visual_markers"] = markers
        except Exception:
            pass
    return metadata


def _resolve_named_target(target: Dict[str, Any], before_snapshot: Dict[str, Any], driver: WindowsDriverClient) -> Dict[str, Any]:
    if "x" in target and "y" in target:
        return {
            "x": int(target["x"]),
            "y": int(target["y"]),
            "button": str(target.get("button") or "left"),
            "resolution_source": "target_payload",
        }
    metadata_path = before_snapshot.get("json_path")
    sidecar = {}
    if metadata_path:
        try:
            sidecar = json.loads(Path(str(metadata_path)).read_text(encoding="utf-8"))
        except Exception:
            sidecar = {}
    named_targets = sidecar.get("named_targets") if isinstance(sidecar.get("named_targets"), dict) else {}
    name = str(target.get("name") or "").strip()
    if name and name in named_targets and isinstance(named_targets.get(name), dict):
        item = named_targets[name]
        if "x" in item and "y" in item:
            return {
                "x": int(item["x"]),
                "y": int(item["y"]),
                "button": str(item.get("button") or target.get("button") or "left"),
                "resolution_source": "snapshot_named_targets",
            }
    if name:
        control = driver.find_control(
            name=name,
            control_type=str(target.get("control_type") or "").strip() or None,
        )
        if isinstance(control, dict) and isinstance(control.get("rect"), dict):
            rect = control["rect"]
            x = int(rect.get("x", 0) + (rect.get("width", 0) / 2))
            y = int(rect.get("y", 0) + (rect.get("height", 0) / 2))
            return {
                "x": x,
                "y": y,
                "button": str(target.get("button") or "left"),
                "resolution_source": "driver_control_lookup",
            }
    raise ValueError("named_target_unresolved")


def _normalize_action_request(
    *,
    action_type: str,
    target: Optional[Dict[str, Any]],
    text: Optional[str],
    hotkeys: Optional[list[str]],
    app: Optional[str],
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    action_n = str(action_type or "").strip().lower()
    target_n = _as_dict(target)
    text_n = str(text or "").strip()
    app_n = str(app or target_n.get("process") or "").strip().lower()
    hotkeys_n = _normalize_hotkeys(hotkeys or target_n.get("keys"))

    if action_n not in ALLOWED_ACTIONS:
        return None, "action_not_allowlisted"

    if action_n in {"launch_test_app", "close_test_app"}:
        if app_n not in ALLOWED_TEST_APPS:
            return None, "app_not_allowlisted"
        return {"action_type": action_n, "target": {"process": app_n}}, None

    if action_n == "focus_window":
        if not (target_n.get("process") or target_n.get("title_contains")):
            return None, "focus_target_missing"
        return {"action_type": action_n, "target": target_n}, None

    if action_n == "type_text":
        if not text_n:
            return None, "text_missing"
        lowered = text_n.lower()
        if len(text_n) > 200 or any(marker in lowered for marker in SENSITIVE_TEXT_MARKERS):
            return None, "text_not_allowlisted"
        return {"action_type": action_n, "target": target_n, "text": text_n}, None

    if action_n == "safe_hotkey":
        if hotkeys_n not in SAFE_HOTKEYS:
            return None, "hotkey_not_allowlisted"
        return {"action_type": action_n, "target": target_n, "keys": list(hotkeys_n)}, None

    if action_n == "click_coordinates":
        if "x" not in target_n or "y" not in target_n:
            return None, "coordinate_target_missing"
        return {
            "action_type": action_n,
            "target": {
                "x": int(target_n["x"]),
                "y": int(target_n["y"]),
                "button": str(target_n.get("button") or "left"),
            },
        }, None

    if action_n == "click_named_target":
        if not (target_n.get("name") or ("x" in target_n and "y" in target_n)):
            return None, "named_target_missing"
        return {"action_type": action_n, "target": target_n}, None

    return None, "action_not_allowlisted"


def _derive_pre_verdict(pre_arbiter: Dict[str, Any]) -> str:
    if not bool(pre_arbiter.get("ok")):
        return "fail"
    if bool(pre_arbiter.get("strategy_ok")) and not _as_text_list(pre_arbiter.get("mismatches")):
        return "pass"
    if bool(pre_arbiter.get("strategy_ok")):
        return "uncertain"
    return "fail"


def _execute_action(driver: WindowsDriverClient, action: Dict[str, Any], before_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    action_type = str(action.get("action_type") or "")
    target = _as_dict(action.get("target"))

    if action_type == "launch_test_app":
        process = str(target.get("process") or "notepad.exe")
        return {"executed": True, "action_detail": driver.app_launch(process=process)}

    if action_type == "focus_window":
        return {
            "executed": True,
            "action_detail": driver.window_focus(
                process=str(target.get("process") or "").strip() or None,
                title_contains=str(target.get("title_contains") or "").strip() or None,
            ),
        }

    if action_type == "type_text":
        process = str(target.get("process") or "").strip() or None
        title_contains = str(target.get("title_contains") or "").strip() or None
        if process or title_contains:
            driver.window_focus(process=process, title_contains=title_contains)
            time.sleep(0.2)
        return {
            "executed": True,
            "action_detail": driver.keyboard_type(text=str(action.get("text") or ""), submit=False),
        }

    if action_type == "safe_hotkey":
        return {
            "executed": True,
            "action_detail": driver.keyboard_hotkey(*[str(item) for item in action.get("keys") or []]),
        }

    if action_type == "click_coordinates":
        x = int(target["x"])
        y = int(target["y"])
        button = str(target.get("button") or "left")
        driver.mouse_click(x=x, y=y, button=button)
        return {"executed": True, "action_detail": {"x": x, "y": y, "button": button}}

    if action_type == "click_named_target":
        resolved = _resolve_named_target(target, before_snapshot, driver)
        driver.mouse_click(x=int(resolved["x"]), y=int(resolved["y"]), button=str(resolved.get("button") or "left"))
        return {"executed": True, "action_detail": resolved}

    if action_type == "close_test_app":
        result = close_editor_safe(driver, action="dont_save")
        if not bool(result.get("ok")):
            raise WindowsDriverError(str(result.get("error") or "close_test_app_failed"))
        return {"executed": True, "action_detail": result}

    raise ValueError("action_not_allowlisted")


def _write_operation_artifact(root_dir: Path, payload: Dict[str, Any], *, ts: Optional[float] = None) -> Dict[str, Any]:
    now = float(ts or time.time())
    label = _ts_label(now)
    artifact_path = root_dir / "artifacts" / "desktop" / f"operation_{label}.json"
    receipt_path = root_dir / "artifacts" / "receipts" / f"desktop_operator_{label}.json"
    payload = dict(payload)
    payload["artifact_path"] = str(artifact_path)
    _safe_write_json(artifact_path, payload)
    receipt = {
        "schema": ROLE_RECEIPT_SCHEMA,
        "ts_utc": _utc_now(now),
        "role": "desktop_operator_lab",
        "rail": payload.get("rail"),
        "ok": str(payload.get("result") or "") == "ok",
        "reason_code": payload.get("reason_code"),
        "artifact_path": str(artifact_path),
        "next_hint": payload.get("next_hint") or "",
        "mode": None,
        "anchor_receipt_path": payload.get("anchor_receipt_path"),
        "source_artifacts": payload.get("source_artifacts") or [],
        "action_type": payload.get("action_type"),
        "operation_class": payload.get("operation_class"),
        "verify_spec_name": _as_dict(_as_dict(payload.get("prepare")).get("verify_spec")).get("name"),
        "undo_spec_name": _as_dict(payload.get("undo_info")).get("name"),
        "verdict": str(_as_dict(payload.get("verify")).get("verdict") or ""),
        "post_verdict": _as_dict(payload.get("verify")).get("post_verdict"),
    }
    _safe_write_json(receipt_path, receipt)
    payload["receipt_path"] = str(receipt_path)
    _safe_write_json(artifact_path, payload)
    return payload


def run_desktop_operate(
    root_dir: Path,
    *,
    rail: str,
    action_type: str,
    objective: str,
    expected_efe_desktop: Dict[str, Any],
    target: Optional[Dict[str, Any]] = None,
    text: Optional[str] = None,
    hotkeys: Optional[list[str]] = None,
    app: Optional[str] = None,
    screenshot_path: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    rail_n = _normalize_rail(rail)
    gate = build_desktop_lab_gate(root, rail=rail_n)
    normalized_action, action_reason = _normalize_action_request(
        action_type=action_type,
        target=target,
        text=text,
        hotkeys=hotkeys,
        app=app,
    )
    normalized_target = _as_dict((normalized_action or {}).get("target")) or _as_dict(target)
    normalized_text = str((normalized_action or {}).get("text") or text or "").strip() or None
    normalized_hotkeys = list((normalized_action or {}).get("keys") or (hotkeys or []))
    resolved_contract = resolve_operation_contract(
        action_type=str(action_type or "").strip().lower(),
        target=normalized_target,
        text=normalized_text,
        hotkeys=normalized_hotkeys,
        app=app or str(normalized_target.get("process") or "").strip() or None,
        expected_efe_desktop=expected_efe_desktop if isinstance(expected_efe_desktop, dict) else None,
    )
    operation_context = _as_dict(resolved_contract.get("context"))
    materialized_expected_efe = _as_dict(resolved_contract.get("expected_efe_desktop_materialized"))
    resolved_expected_efe = _as_dict(resolved_contract.get("expected_efe_desktop"))
    verify_spec = _as_dict(resolved_contract.get("verify_spec"))
    undo_info = _as_dict(resolved_contract.get("undo_info"))
    payload: Dict[str, Any] = {
        "schema": OPERATION_SCHEMA,
        "role": "desktop_operator_lab",
        "rail": rail_n,
        "action_type": str(action_type or "").strip().lower(),
        "operation_class": resolved_contract.get("operation_class"),
        "prepare": {
            "screenshot_before": screenshot_path,
            "pre_verdict": "fail",
            "visual_risks": [],
            "expected_efe_desktop_materialized": materialized_expected_efe,
            "expected_efe_desktop": resolved_expected_efe,
            "verify_spec": verify_spec,
            "contract_source": resolved_contract.get("contract_source"),
        },
        "apply": {"executed": False, "action_detail": {}},
        "verify": {"screenshot_after": None, "post_verdict": "fail", "verdict": "fail", "verify_input": {}, "mismatches": []},
        "undo_info": undo_info or {"possible": False, "suggested_steps": []},
        "verdict": "fail",
        "result": "fail_closed",
        "reason_code": "pending",
        "next_hint": "",
        "source_artifacts": [],
        "anchor_receipt_path": gate.get("anchor_receipt_path"),
    }

    if rail_n != "lab":
        payload["reason_code"] = "lab_only_command"
        payload["next_hint"] = _reason_hint("lab_only_command")
        return _write_operation_artifact(root, payload)

    if not resolved_expected_efe:
        payload["reason_code"] = "missing_expected_efe_desktop"
        payload["next_hint"] = _reason_hint("missing_expected_efe_desktop")
        return _write_operation_artifact(root, payload)

    if not bool(gate.get("ok")):
        payload["reason_code"] = gate.get("reason_code") or "lab_gate_failed"
        payload["next_hint"] = gate.get("next_hint") or _reason_hint(payload["reason_code"])
        return _write_operation_artifact(root, payload)

    if normalized_action is None:
        payload["reason_code"] = action_reason or "action_not_allowlisted"
        payload["next_hint"] = _reason_hint(payload["reason_code"])
        return _write_operation_artifact(root, payload)

    try:
        driver = WindowsDriverClient(base_url=_lab_driver_url(), prefer_env=False)
    except Exception:
        payload["reason_code"] = "driver_client_unavailable"
        payload["next_hint"] = _reason_hint("driver_client_unavailable")
        return _write_operation_artifact(root, payload)

    before_snapshot: Dict[str, Any]
    if screenshot_path:
        before_snapshot = {"png_path": str(screenshot_path), "json_path": str(Path(str(screenshot_path)).with_suffix(".json"))}
    else:
        try:
            before_snapshot = _capture_current_lab_screenshot(root, context="desktop_operator_before")
        except Exception as exc:
            payload["reason_code"] = "screenshot_endpoint_unavailable"
            payload["next_hint"] = f"{_reason_hint('screenshot_endpoint_unavailable')} ({str(exc)[:120]})"
            return _write_operation_artifact(root, payload)

    screenshot_before = str(before_snapshot.get("png_path") or screenshot_path or "")
    payload["prepare"]["screenshot_before"] = screenshot_before
    if not screenshot_before or not Path(screenshot_before).exists():
        payload["reason_code"] = "screenshot_missing"
        payload["next_hint"] = _reason_hint("screenshot_missing")
        return _write_operation_artifact(root, payload)

    strategy = {
        "action_type": normalized_action["action_type"],
        "target": normalized_action.get("target") or {},
        "expected_efe_desktop": resolved_expected_efe,
        "operation_class": resolved_contract.get("operation_class"),
        "verify_spec": verify_spec,
    }
    if "text" in normalized_action:
        strategy["text_preview"] = str(normalized_action["text"])[:80]
    if "keys" in normalized_action:
        strategy["keys"] = list(normalized_action["keys"])

    scout = run_desktop_scout(
        root,
        rail="lab",
        screenshot_path=screenshot_before,
        objective=objective,
        strategy=strategy,
    )
    pre_arbiter = run_desktop_arbiter(
        root,
        rail="lab",
        mode="pre",
        screenshot_path=screenshot_before,
        objective=objective,
        strategy=strategy,
    )
    payload["source_artifacts"] = [
        item
        for item in [scout.get("artifact_path"), pre_arbiter.get("artifact_path")]
        if isinstance(item, str) and item
    ]
    visual_risks = _as_text_list(scout.get("visual_risks")) + _as_text_list(pre_arbiter.get("visual_risks"))
    payload["prepare"]["visual_risks"] = visual_risks
    pre_verdict = _derive_pre_verdict(pre_arbiter)
    payload["prepare"]["pre_verdict"] = pre_verdict
    payload["prepare"]["plan"] = {
        "action_type": normalized_action["action_type"],
        "operation_class": resolved_contract.get("operation_class"),
        "target": normalized_action.get("target") or {},
        "expected_efe_desktop": resolved_expected_efe,
        "verify_spec": verify_spec,
        "visual_risks": visual_risks,
        "undo_info": payload["undo_info"],
    }
    if pre_verdict != "pass":
        payload["reason_code"] = "pre_verdict_not_pass"
        payload["next_hint"] = _reason_hint("pre_verdict_not_pass")
        return _write_operation_artifact(root, payload)

    try:
        apply_out = _execute_action(driver, normalized_action, before_snapshot)
    except Exception as exc:
        payload["reason_code"] = "apply_failed"
        payload["next_hint"] = f"{_reason_hint('apply_failed')} ({str(exc)[:120]})"
        return _write_operation_artifact(root, payload)

    payload["apply"] = apply_out

    try:
        after_snapshot = _capture_current_lab_screenshot(root, context="desktop_operator_after")
    except Exception as exc:
        payload["reason_code"] = "verify_screenshot_unavailable"
        payload["next_hint"] = f"{_reason_hint('verify_screenshot_unavailable')} ({str(exc)[:120]})"
        return _write_operation_artifact(root, payload)

    screenshot_after = str(after_snapshot.get("png_path") or "")
    payload["verify"]["screenshot_after"] = screenshot_after
    runtime_metadata = _metadata_from_runtime(
        driver,
        expected_efe_desktop=resolved_expected_efe,
        text_probe=str(normalized_action.get("text") or "") if normalized_action["action_type"] == "type_text" else None,
    )
    post_arbiter = run_desktop_arbiter(
        root,
        rail="lab",
        mode="post",
        screenshot_path=screenshot_after,
        objective=objective,
        strategy=strategy,
        expected_efe_desktop=resolved_expected_efe,
        metadata=runtime_metadata,
    )
    payload["source_artifacts"].append(post_arbiter.get("artifact_path"))
    payload["verify"]["post_verdict"] = (
        str(post_arbiter.get("post_action_verdict") or "fail")
        if bool(post_arbiter.get("ok"))
        else "fail"
    )
    verify_eval = evaluate_desktop_efe(
        operation_class=str(resolved_contract.get("operation_class") or "") or None,
        context=operation_context,
        before_state={
            "screenshot_path": screenshot_before,
            "metadata": {},
        },
        after_state={
            "screenshot_path": screenshot_after,
            "runtime_metadata": runtime_metadata,
            "post_arbiter": post_arbiter,
        },
        expected_efe_desktop=resolved_expected_efe,
        verify_spec=verify_spec,
    )
    payload["verify"]["verify_input"] = verify_eval.get("verify_input") or {}
    payload["verify"]["verdict"] = str(verify_eval.get("verdict") or "fail")
    payload["verify"]["mismatches"] = _as_text_list(verify_eval.get("mismatches"))
    payload["verify"]["post_arbiter_artifact_path"] = post_arbiter.get("artifact_path")
    payload["prepare"]["scout_artifact_path"] = scout.get("artifact_path")
    payload["prepare"]["pre_arbiter_artifact_path"] = pre_arbiter.get("artifact_path")
    payload["verdict"] = payload["verify"]["verdict"]

    if payload["verify"]["verdict"] == "pass":
        payload["result"] = "ok"
        payload["reason_code"] = "ok"
        payload["next_hint"] = ""
    elif payload["verify"]["verdict"] == "uncertain":
        payload["reason_code"] = str(verify_eval.get("reason_code") or "verify_uncertain")
        payload["next_hint"] = str(verify_eval.get("next_hint") or _reason_hint("verify_uncertain"))
    else:
        payload["reason_code"] = str(verify_eval.get("reason_code") or "verify_fail")
        payload["next_hint"] = str(verify_eval.get("next_hint") or _reason_hint("verify_fail"))

    return _write_operation_artifact(root, payload)


def run_desktop_operate_demo(root_dir: Path, *, rail: str) -> Dict[str, Any]:
    root = Path(root_dir)
    now = time.time()
    rail_n = _normalize_rail(rail)
    steps: list[Dict[str, Any]] = []

    launch = run_desktop_operate(
        root,
        rail=rail_n,
        action_type="launch_test_app",
        objective="Open a benign LAB test app.",
        expected_efe_desktop={
            "active_window_title_contains": "notepad",
            "visual_markers_any": ["notepad.exe"],
        },
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    steps.append(
        {
            "name": "launch_test_app",
            "ok": launch.get("result") == "ok",
            "artifact_path": launch.get("artifact_path"),
            "reason_code": launch.get("reason_code"),
        }
    )

    type_step = None
    if launch.get("result") == "ok":
        type_step = run_desktop_operate(
            root,
            rail=rail_n,
            action_type="type_text",
            objective="Type a harmless LAB marker into the focused test app.",
            expected_efe_desktop={
                "active_window_title_contains": "notepad",
                "visible_text_any": ["AJAX LAB DEMO"],
            },
            text="AJAX LAB DEMO",
            target={"process": "notepad.exe", "title_contains": "notepad"},
        )
        steps.append(
            {
                "name": "type_text",
                "ok": type_step.get("result") == "ok",
                "artifact_path": type_step.get("artifact_path"),
                "reason_code": type_step.get("reason_code"),
            }
        )

    audit = {
        "schema": OPERATION_DEMO_SCHEMA,
        "ts_utc": _utc_now(now),
        "rail": rail_n,
        "ok": all(bool(step.get("ok")) for step in steps) if steps else False,
        "steps": steps,
        "reason_code": "ok",
        "next_hint": "",
    }
    if not audit["ok"]:
        first = next((step for step in steps if not step.get("ok")), None)
        audit["reason_code"] = str((first or {}).get("reason_code") or "operate_demo_failed")
        audit["next_hint"] = _reason_hint(audit["reason_code"])
    audit_path = root / "artifacts" / "audits" / f"desktop_operation_demo_{_ts_label(now)}.json"
    _safe_write_json(audit_path, audit)
    audit["artifact_path"] = str(audit_path)
    _safe_write_json(audit_path, audit)
    return audit
