from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from agency.anchor_preflight import run_anchor_preflight
from agency.desktop_operation_specs import evaluate_desktop_verify_input, resolve_operation_contract
from agency.desktop_verify_contract import (
    DESKTOP_VERIFICATION_CONTRACT_VERSION,
    build_desktop_mismatch,
    build_desktop_verification_result,
    build_desktop_verify_input,
    extract_desktop_verify_mismatch,
    normalize_desktop_mismatches,
    normalize_desktop_verification_result,
)
from agency.lab_session_anchor import validate_expected_session


SCOUT_SCHEMA = "ajax.desktop.scout.v1"
ARBITER_SCHEMA = "ajax.desktop.arbiter.v1"
COMPILER_SCHEMA = "ajax.desktop.compiler.v1"
DEMO_SCHEMA = "ajax.desktop.demo.v1"
VERIFY_DEMO_SCHEMA = "ajax.desktop.verify_demo.v1"
ROLE_RECEIPT_SCHEMA = "ajax.desktop.role_receipt.v1"


def _utc_now(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _ts_label(ts: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts or time.time()))


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _normalize_rail(raw: Any) -> str:
    value = str(raw or "lab").strip().lower()
    if value in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def _normalize_mode(raw: Any) -> str:
    value = str(raw or "pre").strip().lower()
    return "post" if value == "post" else "pre"


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


def _normalize_verify_result_alias(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict) and value:
        return normalize_desktop_verification_result(value)
    return {}


def _normalize_verify_results_alias(values: Any) -> list[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [
        normalize_desktop_verification_result(item)
        for item in values
        if isinstance(item, dict) and item
    ]


def _aggregate_verify_mismatch(values: Any) -> list[Dict[str, Any]]:
    mismatches: list[Dict[str, Any]] = []
    for item in _normalize_verify_results_alias(values):
        mismatches.extend(extract_desktop_verify_mismatch(item))
    return normalize_desktop_mismatches(mismatches)


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _slug(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug[:64] or fallback


def _strategy_text(strategy: Any) -> str:
    if isinstance(strategy, str):
        return strategy.strip()
    if strategy is None:
        return ""
    try:
        return json.dumps(strategy, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(strategy)


def _load_visual_metadata(screenshot_path: Path, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sidecar = _load_json(screenshot_path.with_suffix(".json")) or {}
    if isinstance(metadata, dict):
        merged = dict(sidecar)
        merged.update(metadata)
        return merged
    return sidecar


def _arbiter_verify_spec(verify_spec: Dict[str, Any]) -> Dict[str, Any]:
    spec = dict(_as_dict(verify_spec))
    required = _as_text_list(spec.get("evidence_required"))
    spec["evidence_required"] = [item for item in required if item != "desktop_arbiter_post"]
    return spec


def _resolve_strategy_contract(strategy: Any, expected_efe_desktop: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    strategy_n = _as_dict(strategy)
    action_type = str(strategy_n.get("action_type") or "").strip().lower()
    target = _as_dict(strategy_n.get("target"))
    app = str(strategy_n.get("app") or target.get("process") or "").strip() or None
    text = str(strategy_n.get("text") or strategy_n.get("text_preview") or "").strip() or None
    hotkeys = strategy_n.get("keys")
    explicit_expected = _as_dict(expected_efe_desktop) or _as_dict(strategy_n.get("expected_efe_desktop"))
    operation_class = str(strategy_n.get("operation_class") or "").strip().lower() or None
    if not action_type and not operation_class:
        return {
            "operation_class": None,
            "context": {"action_type": "", "process": app or ""},
            "expected_efe_desktop": explicit_expected,
            "verify_spec": {},
            "contract_source": "strategy_only",
        }
    return resolve_operation_contract(
        action_type=action_type,
        target=target,
        text=text,
        hotkeys=hotkeys,
        app=app,
        expected_efe_desktop=explicit_expected,
        operation_class=operation_class,
    )


def _reason_hint(reason_code: str, *, role: str) -> str:
    code = str(reason_code or "").strip().lower()
    if code == "lab_only_command":
        return "Run with --rail lab. Desktop LAB roles are intentionally disabled outside LAB."
    if code.startswith("session_") or code == "expected_session_missing":
        return "Run `python bin/ajaxctl lab session init --rail lab --display dummy` and retry."
    if "anchor" in code or "display_target" in code or "port_" in code:
        return "Run `python bin/ajaxctl doctor anchor --rail lab` and fix the LAB anchor before retrying."
    if code in {"screenshot_missing", "screenshot_unreadable"}:
        return "Provide a valid screenshot path or capture one with `python bin/ajaxctl desktop snap --rail lab`."
    if code in {"missing_strategy", "missing_expected_efe_desktop"}:
        return "Provide the missing strategy/EFE input and retry."
    if code in {"missing_artifact_input", "invalid_artifact_kind", "invalid_artifact_state"}:
        return "Provide valid scout/arbiter artifacts from a successful LAB desktop run."
    if role == "desktop_compiler":
        return "Re-run scout/arbiter in LAB and pass their artifact paths into desktop compile."
    return "Capture a fresh LAB screenshot and retry with a narrower desktop objective."


def _build_lab_gate(root_dir: Path, *, rail: str) -> Dict[str, Any]:
    rail_n = _normalize_rail(rail)
    if rail_n != "lab":
        return {
            "ok": False,
            "reason_code": "lab_only_command",
            "next_hint": _reason_hint("lab_only_command", role="desktop"),
            "session_status": None,
            "anchor": None,
            "anchor_receipt_path": None,
        }

    session_status = validate_expected_session(
        root_dir,
        required_rail="lab",
        required_display="dummy",
    )
    anchor = run_anchor_preflight(root_dir=root_dir, rail="lab", write_receipt=True)

    if not bool(session_status.get("ok")):
        reason_code = str(session_status.get("reason") or "lab_session_invalid")
        return {
            "ok": False,
            "reason_code": reason_code,
            "next_hint": _reason_hint(reason_code, role="desktop"),
            "session_status": session_status,
            "anchor": anchor,
            "anchor_receipt_path": anchor.get("receipt_path"),
        }

    if not bool(anchor.get("ok")):
        mismatches = anchor.get("mismatches") if isinstance(anchor.get("mismatches"), list) else []
        first_code = None
        for item in mismatches:
            if isinstance(item, dict) and str(item.get("code") or "").strip():
                first_code = str(item.get("code")).strip()
                break
        reason_code = first_code or str(anchor.get("reason") or "lab_anchor_missing")
        return {
            "ok": False,
            "reason_code": reason_code,
            "next_hint": _reason_hint(reason_code, role="desktop"),
            "session_status": session_status,
            "anchor": anchor,
            "anchor_receipt_path": anchor.get("receipt_path"),
        }

    return {
        "ok": True,
        "reason_code": "ok",
        "next_hint": "",
        "session_status": session_status,
        "anchor": anchor,
        "anchor_receipt_path": anchor.get("receipt_path"),
    }


def _build_visual_summary(
    screenshot_path: Path,
    *,
    objective: str,
    strategy: Any,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    active_window = str(metadata.get("active_window_title") or metadata.get("window_title") or "").strip()
    focus_target = str(metadata.get("focus_target") or metadata.get("focus") or "").strip()
    affordances = _as_text_list(metadata.get("ui_affordances") or metadata.get("affordances"))
    dialogs = _as_text_list(metadata.get("dialogs") or metadata.get("visible_dialogs"))
    warnings = _as_text_list(metadata.get("display_warnings") or metadata.get("warnings"))
    markers = _as_text_list(metadata.get("visual_markers") or metadata.get("markers"))
    visible_text = _as_text_list(metadata.get("visible_text"))

    observed_affordances: list[str] = []
    if active_window:
        observed_affordances.append(f"active_window:{active_window}")
    if focus_target:
        observed_affordances.append(f"focus_target:{focus_target}")
    observed_affordances.extend(f"visible_affordance:{item}" for item in affordances[:6])
    observed_affordances.extend(f"visual_marker:{item}" for item in markers[:4])
    observed_affordances.extend(f"visible_text:{item}" for item in visible_text[:4])
    if not observed_affordances:
        observed_affordances.append("screenshot_present")

    visual_risks: list[str] = []
    visual_risks.extend(f"dialog_present:{item}" for item in dialogs[:4])
    visual_risks.extend(f"display_warning:{item}" for item in warnings[:4])
    if bool(metadata.get("requires_confirmation")):
        visual_risks.append("confirmation_gate_visible")
    if bool(metadata.get("sensitive_surface")):
        visual_risks.append("sensitive_surface_visible")

    candidate_paths: list[str] = []
    if dialogs:
        candidate_paths.append("stabilize_or_review_dialog_before_any_mutation")
    if affordances:
        candidate_paths.append(f"inspect:{affordances[0]}")
    if focus_target:
        candidate_paths.append(f"follow_focus:{focus_target}")
    if objective:
        candidate_paths.append(f"align_with_goal:{objective[:80]}")
    if _strategy_text(strategy):
        candidate_paths.append("validate_candidate_strategy_against_visible_state")
    if not candidate_paths:
        candidate_paths.append("capture_more_context_before_acting")

    confidence = "low"
    if affordances or active_window or focus_target:
        confidence = "high" if not dialogs and not warnings else "medium"
    elif screenshot_path.exists():
        confidence = "medium"

    return {
        "observed_affordances": _dedupe(observed_affordances),
        "visual_risks": _dedupe(visual_risks),
        "candidate_paths": _dedupe(candidate_paths),
        "confidence": confidence,
        "dialogs": dialogs,
        "markers": markers,
        "active_window_title": active_window,
        "focus_target": focus_target,
        "visible_affordances": affordances,
        "metadata": metadata,
    }


def _write_role_payload(
    root_dir: Path,
    *,
    role: str,
    payload: Dict[str, Any],
    ts: Optional[float] = None,
) -> Dict[str, Any]:
    now = float(ts or time.time())
    label = _ts_label(now)
    role_slug = role.replace("desktop_", "")
    artifact_path = root_dir / "artifacts" / "desktop" / f"{role_slug}_{label}.json"
    receipt_path = root_dir / "artifacts" / "receipts" / f"desktop_{role_slug}_{label}.json"

    payload = dict(payload)
    verify_result = _normalize_verify_result_alias(
        payload.get("verify_result") or payload.get("verification_result")
    )
    if verify_result:
        payload["verify_result"] = verify_result
        payload["verification_result"] = verify_result
        payload["verify_mismatch"] = verify_result.get("verify_mismatch") or []

    verify_inputs = payload.get("verify_inputs")
    if not isinstance(verify_inputs, list) or (
        not verify_inputs
        and isinstance(payload.get("verification_inputs"), list)
        and payload.get("verification_inputs")
    ):
        verify_inputs = payload.get("verification_inputs")
    if isinstance(verify_inputs, list):
        payload["verify_inputs"] = [item for item in verify_inputs if isinstance(item, dict)]
        payload["verification_inputs"] = payload["verify_inputs"]

    verify_results_source = payload.get("verify_results")
    if not isinstance(verify_results_source, list) or (
        not verify_results_source
        and isinstance(payload.get("verification_results"), list)
        and payload.get("verification_results")
    ):
        verify_results_source = payload.get("verification_results")
    verify_results = _normalize_verify_results_alias(verify_results_source)
    if verify_results:
        payload["verify_results"] = verify_results
        payload["verification_results"] = verify_results
        payload["verify_mismatch"] = _aggregate_verify_mismatch(verify_results)

    payload["artifact_path"] = str(artifact_path)
    _safe_write_json(artifact_path, payload)

    receipt = {
        "schema": ROLE_RECEIPT_SCHEMA,
        "ts_utc": _utc_now(now),
        "role": role,
        "rail": payload.get("rail"),
        "ok": bool(payload.get("ok")),
        "reason_code": payload.get("reason_code"),
        "artifact_path": str(artifact_path),
        "next_hint": payload.get("next_hint") or "",
        "mode": payload.get("mode"),
        "anchor_receipt_path": payload.get("anchor_receipt_path"),
        "source_artifacts": payload.get("source_artifacts") or [],
        "verification_contract_version": payload.get("verification_contract_version"),
    }
    if isinstance(payload.get("verify_input"), dict):
        receipt["verify_input"] = payload.get("verify_input")
    if isinstance(payload.get("verify_result"), dict):
        receipt["verify_result"] = payload.get("verify_result")
        receipt["verification_result"] = payload.get("verify_result")
        receipt["verify_mismatch"] = _as_dict(payload.get("verify_result")).get("verify_mismatch") or []
        receipt["verdict"] = _as_dict(payload.get("verify_result")).get("verdict")
    if isinstance(payload.get("verify_results"), list):
        receipt["verify_results"] = payload.get("verify_results")
        receipt["verification_results"] = payload.get("verify_results")
        receipt["verify_mismatch"] = payload.get("verify_mismatch") or []
    _safe_write_json(receipt_path, receipt)

    payload["receipt_path"] = str(receipt_path)
    _safe_write_json(artifact_path, payload)
    return payload


def run_desktop_scout(
    root_dir: Path,
    *,
    rail: str,
    screenshot_path: str,
    objective: str,
    strategy: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    screenshot = Path(screenshot_path)
    rail_n = _normalize_rail(rail)
    gate = _build_lab_gate(root, rail=rail_n)
    payload: Dict[str, Any] = {
        "schema": SCOUT_SCHEMA,
        "role": "desktop_scout",
        "rail": rail_n,
        "ok": False,
        "reason_code": "pending",
        "objective": str(objective or "").strip(),
        "context": str(context or "").strip() or None,
        "strategy_candidate": strategy,
        "screenshot_path": str(screenshot),
        "observed_affordances": [],
        "visual_risks": [],
        "candidate_paths": [],
        "confidence": "low",
        "next_hint": "",
        "anchor_receipt_path": gate.get("anchor_receipt_path"),
        "lab_gate": gate,
        "metadata_path": str(screenshot.with_suffix(".json")) if screenshot.with_suffix(".json").exists() else None,
    }
    if not bool(gate.get("ok")):
        payload["reason_code"] = gate.get("reason_code") or "lab_gate_failed"
        payload["next_hint"] = gate.get("next_hint") or _reason_hint(payload["reason_code"], role="desktop_scout")
        return _write_role_payload(root, role="desktop_scout", payload=payload)

    if not screenshot.exists():
        payload["reason_code"] = "screenshot_missing"
        payload["next_hint"] = _reason_hint("screenshot_missing", role="desktop_scout")
        return _write_role_payload(root, role="desktop_scout", payload=payload)

    meta = _load_visual_metadata(screenshot, metadata)
    visual = _build_visual_summary(
        screenshot,
        objective=str(objective or ""),
        strategy=strategy,
        metadata=meta,
    )
    payload.update(
        {
            "ok": True,
            "reason_code": "ok",
            "observed_affordances": visual["observed_affordances"],
            "visual_risks": visual["visual_risks"],
            "candidate_paths": visual["candidate_paths"],
            "confidence": visual["confidence"],
            "next_hint": visual["candidate_paths"][0] if visual["candidate_paths"] else "",
            "metadata": meta,
        }
    )
    return _write_role_payload(root, role="desktop_scout", payload=payload)


def _strategy_risks(strategy: Any) -> list[str]:
    text = _strategy_text(strategy).lower()
    if not text:
        return []
    risks: list[str] = []
    destructive_terms = ("delete", "format", "wipe", "registry", "credential", "password", "token", "secret")
    for term in destructive_terms:
        if term in text:
            risks.append(f"strategy_mentions:{term}")
    if "click" in text and "dialog" in text and "review" not in text:
        risks.append("strategy_clicks_without_dialog_review")
    return _dedupe(risks)


def _expected_contains(actual: str, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_expected_contains(actual, item) for item in expected)
    text = str(expected or "").strip().lower()
    if not text:
        return False
    return text in actual.lower()


def _evaluate_expected_efe(expected_efe_desktop: Dict[str, Any], visual: Dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    active_window = str(visual.get("active_window_title") or "")
    focus_target = str(visual.get("focus_target") or "")
    dialogs = visual.get("dialogs") if isinstance(visual.get("dialogs"), list) else []
    observed_affordances = visual.get("observed_affordances") if isinstance(visual.get("observed_affordances"), list) else []
    markers = visual.get("markers") if isinstance(visual.get("markers"), list) else []
    visible_text = _as_text_list((visual.get("metadata") or {}).get("visible_text") if isinstance(visual.get("metadata"), dict) else None)

    expected_title = expected_efe_desktop.get("active_window_title_contains")
    if expected_title and not _expected_contains(active_window, expected_title):
        mismatches.append("active_window_title_mismatch")

    expected_focus = expected_efe_desktop.get("focus_target_contains")
    if expected_focus and not _expected_contains(focus_target, expected_focus):
        mismatches.append("focus_target_mismatch")

    if bool(expected_efe_desktop.get("dialogs_absent")) and dialogs:
        mismatches.append("dialogs_still_present")

    expected_dialogs = _as_text_list(expected_efe_desktop.get("dialogs_present"))
    for item in expected_dialogs:
        if item not in dialogs:
            mismatches.append(f"expected_dialog_missing:{item}")

    expected_affordances = _as_text_list(expected_efe_desktop.get("affordances_any"))
    if expected_affordances:
        lowered = [str(item).lower() for item in observed_affordances]
        if not any(any(expected.lower() in seen for seen in lowered) for expected in expected_affordances):
            mismatches.append("expected_affordance_missing")

    expected_markers = _as_text_list(expected_efe_desktop.get("visual_markers_any"))
    if expected_markers:
        lowered_markers = [str(item).lower() for item in markers]
        if not any(expected.lower() in seen for expected in expected_markers for seen in lowered_markers):
            mismatches.append("expected_visual_marker_missing")

    expected_markers_absent = _as_text_list(expected_efe_desktop.get("visual_markers_absent"))
    if expected_markers_absent:
        lowered_markers = [str(item).lower() for item in markers]
        if any(expected.lower() in seen for expected in expected_markers_absent for seen in lowered_markers):
            mismatches.append("unexpected_visual_marker_present")

    expected_text = _as_text_list(expected_efe_desktop.get("visible_text_any"))
    if expected_text:
        lowered_text = [str(item).lower() for item in visible_text]
        if not any(expected.lower() in seen for expected in expected_text for seen in lowered_text):
            mismatches.append("expected_visible_text_missing")

    return mismatches


def _build_verify_visual_state(
    screenshot: Path,
    visual: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    verdict: str = "",
) -> Dict[str, Any]:
    return {
        "active_window_title": str(visual.get("active_window_title") or "").strip(),
        "focus_target": str(visual.get("focus_target") or "").strip(),
        "dialogs": _as_text_list(visual.get("dialogs")),
        "observed_affordances": _as_text_list(visual.get("observed_affordances")),
        "markers": _as_text_list(visual.get("markers")),
        "metadata": {"visible_text": _as_text_list(_as_dict(metadata).get("visible_text"))},
        "post_action_verdict": str(verdict or "").strip().lower(),
        "screenshot_path": str(screenshot),
    }


def run_desktop_arbiter(
    root_dir: Path,
    *,
    rail: str,
    mode: str,
    screenshot_path: str,
    objective: str,
    strategy: Any = None,
    expected_efe_desktop: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    screenshot = Path(screenshot_path)
    rail_n = _normalize_rail(rail)
    mode_n = _normalize_mode(mode)
    gate = _build_lab_gate(root, rail=rail_n)
    payload: Dict[str, Any] = {
        "schema": ARBITER_SCHEMA,
        "role": "desktop_arbiter",
        "mode": mode_n,
        "rail": rail_n,
        "ok": False,
        "reason_code": "pending",
        "objective": str(objective or "").strip(),
        "strategy_candidate": strategy,
        "expected_efe_desktop": expected_efe_desktop if isinstance(expected_efe_desktop, dict) else {},
        "screenshot_path": str(screenshot),
        "strategy_ok": False,
        "visual_risks": [],
        "verify_input": {},
        "verify_result": {},
        "verify_mismatch": [],
        "verification_result": {},
        "verification_contract_version": DESKTOP_VERIFICATION_CONTRACT_VERSION,
        "next_hint": "",
        "anchor_receipt_path": gate.get("anchor_receipt_path"),
        "lab_gate": gate,
        "metadata_path": str(screenshot.with_suffix(".json")) if screenshot.with_suffix(".json").exists() else None,
    }
    if not bool(gate.get("ok")):
        payload["reason_code"] = gate.get("reason_code") or "lab_gate_failed"
        payload["next_hint"] = gate.get("next_hint") or _reason_hint(payload["reason_code"], role="desktop_arbiter")
        return _write_role_payload(root, role="desktop_arbiter", payload=payload)

    if not screenshot.exists():
        payload["reason_code"] = "screenshot_missing"
        payload["next_hint"] = _reason_hint("screenshot_missing", role="desktop_arbiter")
        return _write_role_payload(root, role="desktop_arbiter", payload=payload)

    if mode_n == "pre" and not _strategy_text(strategy):
        payload["reason_code"] = "missing_strategy"
        payload["next_hint"] = _reason_hint("missing_strategy", role="desktop_arbiter")
        return _write_role_payload(root, role="desktop_arbiter", payload=payload)

    contract = _resolve_strategy_contract(strategy, expected_efe_desktop if isinstance(expected_efe_desktop, dict) else None)
    resolved_expected_efe = _as_dict(contract.get("expected_efe_desktop"))
    if mode_n == "post" and not resolved_expected_efe:
        payload["reason_code"] = "missing_expected_efe_desktop"
        payload["next_hint"] = _reason_hint("missing_expected_efe_desktop", role="desktop_arbiter")
        return _write_role_payload(root, role="desktop_arbiter", payload=payload)

    meta = _load_visual_metadata(screenshot, metadata)
    visual = _build_visual_summary(
        screenshot,
        objective=str(objective or ""),
        strategy=strategy,
        metadata=meta,
    )
    risks = _dedupe([*visual["visual_risks"], *_strategy_risks(strategy)])
    mismatches: list[Dict[str, Any]] = []
    strategy_ok = True
    verify_spec = _arbiter_verify_spec(_as_dict(contract.get("verify_spec")))
    provisional_verdict = "uncertain"

    if any(item.startswith("strategy_mentions:") for item in risks):
        strategy_ok = False
        mismatches.append(
            build_desktop_mismatch(
                "strategy_candidate",
                expected="non-destructive LAB-only strategy",
                observed=_strategy_text(strategy),
                severity="high",
                note="The strategy mentions destructive or secretive terms and cannot be allowed in LAB.",
            )
        )
    elif any(item.startswith("dialog_present:") for item in risks) and "dialog" not in _strategy_text(strategy).lower():
        strategy_ok = False
        mismatches.append(
            build_desktop_mismatch(
                "strategy_candidate",
                expected="dialog-aware strategy",
                observed=_strategy_text(strategy),
                severity="high",
                note="A visible dialog was not accounted for by the proposed strategy.",
            )
        )
    elif visual["observed_affordances"] == ["screenshot_present"]:
        strategy_ok = False
        mismatches.append(
            build_desktop_mismatch(
                "observed_affordances",
                expected="usable desktop affordances",
                observed=visual["observed_affordances"],
                severity="medium",
                note="The screenshot exposed too little visual structure to validate the strategy safely.",
            )
        )

    if mode_n == "pre":
        if mismatches and mismatches[0]["field"] == "observed_affordances":
            verification_result = build_desktop_verification_result(
                verdict="uncertain",
                mismatches=mismatches,
                reason_code="strategy_validation_uncertain",
                next_hint="Capture a richer screenshot before validating the desktop strategy.",
                confidence="low",
            )
        elif mismatches:
            verification_result = build_desktop_verification_result(
                verdict="fail",
                mismatches=mismatches,
                reason_code="strategy_validation_failed",
                next_hint="Revise the desktop strategy before any LAB action.",
                confidence="high",
            )
        else:
            verification_result = build_desktop_verification_result(
                verdict="pass",
                mismatches=[],
                reason_code="ok",
                next_hint=visual["candidate_paths"][0] if visual["candidate_paths"] else "",
                confidence="high",
            )
    else:
        if mismatches:
            verification_result = build_desktop_verification_result(
                verdict="fail",
                mismatches=mismatches,
                reason_code="strategy_validation_failed",
                next_hint="Revise the desktop strategy before any LAB action.",
                confidence="high",
            )
        else:
            if visual["observed_affordances"]:
                provisional_verdict = "pass"
            verify_input = build_desktop_verify_input(
                operation_class=contract.get("operation_class"),
                expected_efe_desktop=resolved_expected_efe,
                before_state=_build_verify_visual_state(screenshot, visual, meta, verdict=provisional_verdict),
                after_state=_build_verify_visual_state(screenshot, visual, meta, verdict=provisional_verdict),
                screenshot_before=str(screenshot),
                screenshot_after=str(screenshot),
                arbiter_context={"mode": mode_n, "strategy_candidate": strategy},
                runtime_metadata=meta,
                verify_spec=verify_spec,
            )
            verify_input["required_evidence"] = _as_text_list(verify_spec.get("evidence_required"))
            verification_result = evaluate_desktop_verify_input(verify_input)
            payload["verify_input"] = verify_input

    payload.update(
        {
            "ok": True,
            "reason_code": str(verification_result.get("reason_code") or "ok"),
            "strategy_ok": bool(strategy_ok),
            "expected_efe_desktop": resolved_expected_efe if mode_n == "post" else _as_dict(contract.get("expected_efe_desktop")),
            "verify_result": verification_result,
            "verify_mismatch": verification_result.get("verify_mismatch") or [],
            "visual_risks": risks,
            "verification_result": verification_result,
            "verify_input": payload.get("verify_input")
            or build_desktop_verify_input(
                operation_class=contract.get("operation_class"),
                expected_efe_desktop=_as_dict(contract.get("expected_efe_desktop")),
                before_state=_build_verify_visual_state(screenshot, visual, meta, verdict=str(verification_result["verdict"])),
                after_state=_build_verify_visual_state(screenshot, visual, meta, verdict=str(verification_result["verdict"])),
                screenshot_before=str(screenshot),
                screenshot_after=str(screenshot),
                arbiter_context={"mode": mode_n, "strategy_candidate": strategy},
                runtime_metadata=meta,
                verify_spec=verify_spec,
            ),
            "next_hint": str(verification_result.get("next_hint") or (visual["candidate_paths"][0] if visual["candidate_paths"] else "")),
            "metadata": meta,
            "observed_affordances": visual["observed_affordances"],
            "markers": visual["markers"],
        }
    )
    return _write_role_payload(root, role="desktop_arbiter", payload=payload)


def _derive_expected_efe_from_scout(scout_artifact: Dict[str, Any]) -> Dict[str, Any]:
    metadata = scout_artifact.get("metadata") if isinstance(scout_artifact.get("metadata"), dict) else {}
    active_window = str(metadata.get("active_window_title") or "").strip()
    focus_target = str(metadata.get("focus_target") or "").strip()
    affordances = _as_text_list(metadata.get("ui_affordances") or metadata.get("affordances"))
    dialogs = _as_text_list(metadata.get("dialogs") or metadata.get("visible_dialogs"))
    expected: Dict[str, Any] = {}
    if active_window:
        expected["active_window_title_contains"] = active_window
    if focus_target:
        expected["focus_target_contains"] = focus_target
    if affordances:
        expected["affordances_any"] = affordances[:3]
    if not dialogs:
        expected["dialogs_absent"] = True
    return expected


def run_desktop_compiler(
    root_dir: Path,
    *,
    rail: str,
    scout_artifact_path: str,
    arbiter_artifact_paths: list[str],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    rail_n = _normalize_rail(rail)
    source_artifacts = [str(scout_artifact_path), *[str(item) for item in arbiter_artifact_paths]]
    payload: Dict[str, Any] = {
        "schema": COMPILER_SCHEMA,
        "role": "desktop_compiler",
        "rail": rail_n,
        "ok": False,
        "reason_code": "pending",
        "recipe_name": "",
        "expected_efe_desktop": {},
        "anti_patterns": [],
        "reusable_guards": [],
        "crystallization_candidate": False,
        "verification_contract_version": DESKTOP_VERIFICATION_CONTRACT_VERSION,
        "verify_inputs": [],
        "verify_results": [],
        "verify_mismatch": [],
        "verification_inputs": [],
        "verification_results": [],
        "next_hint": "",
        "source_artifacts": source_artifacts,
        "context": context if isinstance(context, dict) else {},
    }
    if rail_n != "lab":
        payload["reason_code"] = "lab_only_command"
        payload["next_hint"] = _reason_hint("lab_only_command", role="desktop_compiler")
        return _write_role_payload(root, role="desktop_compiler", payload=payload)

    scout_path = Path(scout_artifact_path)
    arbiter_paths = [Path(item) for item in arbiter_artifact_paths if str(item).strip()]
    if not scout_path.exists() or not arbiter_paths:
        payload["reason_code"] = "missing_artifact_input"
        payload["next_hint"] = _reason_hint("missing_artifact_input", role="desktop_compiler")
        return _write_role_payload(root, role="desktop_compiler", payload=payload)

    scout_artifact = _load_json(scout_path)
    arbiter_artifacts = [_load_json(path) for path in arbiter_paths]
    if not isinstance(scout_artifact, dict) or scout_artifact.get("role") != "desktop_scout":
        payload["reason_code"] = "invalid_artifact_kind"
        payload["next_hint"] = _reason_hint("invalid_artifact_kind", role="desktop_compiler")
        return _write_role_payload(root, role="desktop_compiler", payload=payload)

    loaded_arbiters = [item for item in arbiter_artifacts if isinstance(item, dict)]
    if not loaded_arbiters or any(item.get("role") != "desktop_arbiter" for item in loaded_arbiters):
        payload["reason_code"] = "invalid_artifact_kind"
        payload["next_hint"] = _reason_hint("invalid_artifact_kind", role="desktop_compiler")
        return _write_role_payload(root, role="desktop_compiler", payload=payload)

    if not bool(scout_artifact.get("ok")):
        payload["reason_code"] = "invalid_artifact_state"
        payload["next_hint"] = _reason_hint("invalid_artifact_state", role="desktop_compiler")
        return _write_role_payload(root, role="desktop_compiler", payload=payload)

    if _normalize_rail(scout_artifact.get("rail")) != "lab":
        payload["reason_code"] = "invalid_artifact_state"
        payload["next_hint"] = _reason_hint("invalid_artifact_state", role="desktop_compiler")
        return _write_role_payload(root, role="desktop_compiler", payload=payload)

    if any(not bool(item.get("ok")) or _normalize_rail(item.get("rail")) != "lab" for item in loaded_arbiters):
        payload["reason_code"] = "invalid_artifact_state"
        payload["next_hint"] = _reason_hint("invalid_artifact_state", role="desktop_compiler")
        return _write_role_payload(root, role="desktop_compiler", payload=payload)

    objective = str(scout_artifact.get("objective") or "").strip()
    recipe_name = str((context or {}).get("recipe_name") or "").strip()
    if not recipe_name:
        recipe_name = _slug(objective, fallback="desktop_lab_recipe")

    verification_inputs = [
        _as_dict(item.get("verify_input"))
        for item in loaded_arbiters
        if isinstance(item.get("verify_input"), dict)
    ]
    verification_results = [
        normalize_desktop_verification_result(item.get("verify_result") or item.get("verification_result"))
        for item in loaded_arbiters
        if isinstance(item.get("verify_result") or item.get("verification_result"), dict)
    ]
    post_candidates = [
        item for item in loaded_arbiters
        if _normalize_mode(item.get("mode")) == "post"
        and (
            isinstance(item.get("expected_efe_desktop"), dict)
            or isinstance(_as_dict(item.get("verify_input")).get("expected_efe_desktop"), dict)
        )
    ]
    expected_efe = (
        dict(
            _as_dict(_as_dict(post_candidates[0].get("verify_input")).get("expected_efe_desktop"))
            or _as_dict(post_candidates[0].get("expected_efe_desktop"))
        )
        if post_candidates
        else _derive_expected_efe_from_scout(scout_artifact)
    )

    anti_patterns: list[str] = []
    anti_patterns.extend(_as_text_list(scout_artifact.get("visual_risks")))
    for arbiter in loaded_arbiters:
        anti_patterns.extend(_as_text_list(arbiter.get("visual_risks")))
        verification_result = normalize_desktop_verification_result(
            arbiter.get("verify_result") or arbiter.get("verification_result")
        )
        anti_patterns.extend(
            f"verification_mismatch:{item['field']}"
            for item in verification_result.get("verify_mismatch", [])
            if isinstance(item, dict) and str(item.get("field") or "").strip()
        )

    reusable_guards = [
        "require_lab_anchor_receipt",
        "require_lab_expected_session",
        "require_visual_strategy_review_before_action",
    ]
    if bool(expected_efe.get("dialogs_absent")):
        reusable_guards.append("stop_on_modal_or_confirmation_dialog")
    if expected_efe.get("affordances_any"):
        reusable_guards.append("assert_expected_affordance_before_recipe_step")

    crystallization_candidate = any(
        str(item.get("verdict") or "").strip().lower() in {"pass", "uncertain"}
        for item in verification_results
    )

    payload.update(
        {
            "ok": True,
            "reason_code": "ok",
            "recipe_name": recipe_name,
            "expected_efe_desktop": expected_efe,
            "anti_patterns": _dedupe(anti_patterns),
            "reusable_guards": _dedupe(reusable_guards),
            "crystallization_candidate": bool(crystallization_candidate),
            "verify_inputs": verification_inputs,
            "verify_results": verification_results,
            "verify_mismatch": _aggregate_verify_mismatch(verification_results),
            "verification_inputs": verification_inputs,
            "verification_results": verification_results,
            "next_hint": "Review the candidate before promoting it into a wider LAB recipe.",
        }
    )
    return _write_role_payload(root, role="desktop_compiler", payload=payload)


def _write_demo_fixture_screenshot(root_dir: Path, *, ts: Optional[float] = None) -> Path:
    label = _ts_label(ts)
    path = root_dir / "artifacts" / "desktop" / f"demo_fixture_{label}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(payload)
    sidecar = {
        "active_window_title": "LAB Fixture Window",
        "focus_target": "search_box",
        "ui_affordances": ["search_box", "results_panel", "confirm_button"],
        "dialogs": [],
        "visual_markers": ["results_panel"],
    }
    _safe_write_json(path.with_suffix(".json"), sidecar)
    return path


def run_desktop_demo(
    root_dir: Path,
    *,
    rail: str,
    screenshot_path: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    now = time.time()
    rail_n = _normalize_rail(rail)
    screenshot = Path(screenshot_path) if screenshot_path else _write_demo_fixture_screenshot(root, ts=now)
    strategy = {
        "summary": "Inspect visible affordances and only continue if the results panel is visible.",
        "steps": ["review search_box", "check results_panel", "hold on dialog"],
    }
    expected_efe_desktop = {
        "active_window_title_contains": "LAB Fixture",
        "focus_target_contains": "search",
        "affordances_any": ["results_panel"],
        "dialogs_absent": True,
    }
    scout = run_desktop_scout(
        root,
        rail=rail_n,
        screenshot_path=str(screenshot),
        objective="Inspect LAB fixture desktop state.",
        strategy=strategy,
    )
    arbiter_pre = run_desktop_arbiter(
        root,
        rail=rail_n,
        mode="pre",
        screenshot_path=str(screenshot),
        objective="Validate the fixture strategy before any desktop actuation.",
        strategy=strategy,
    )
    arbiter_post = run_desktop_arbiter(
        root,
        rail=rail_n,
        mode="post",
        screenshot_path=str(screenshot),
        objective="Validate the fixture desktop after the simulated step.",
        strategy=strategy,
        expected_efe_desktop=expected_efe_desktop,
    )
    compiler = run_desktop_compiler(
        root,
        rail=rail_n,
        scout_artifact_path=str(scout.get("artifact_path") or ""),
        arbiter_artifact_paths=[
            str(arbiter_pre.get("artifact_path") or ""),
            str(arbiter_post.get("artifact_path") or ""),
        ],
        context={"recipe_name": "desktop_lab_fixture_recipe"},
    )
    audit = {
        "schema": DEMO_SCHEMA,
        "ts_utc": _utc_now(now),
        "rail": rail_n,
        "ok": all(bool(item.get("ok")) for item in (scout, arbiter_pre, arbiter_post, compiler)),
        "reason_code": "ok",
        "fixture_screenshot_path": str(screenshot),
        "steps": [
            {"role": "desktop_scout", "ok": bool(scout.get("ok")), "artifact_path": scout.get("artifact_path")},
            {"role": "desktop_arbiter_pre", "ok": bool(arbiter_pre.get("ok")), "artifact_path": arbiter_pre.get("artifact_path")},
            {"role": "desktop_arbiter_post", "ok": bool(arbiter_post.get("ok")), "artifact_path": arbiter_post.get("artifact_path")},
            {"role": "desktop_compiler", "ok": bool(compiler.get("ok")), "artifact_path": compiler.get("artifact_path")},
        ],
        "next_hint": "",
    }
    failed = [item for item in (scout, arbiter_pre, arbiter_post, compiler) if not bool(item.get("ok"))]
    if failed:
        first = failed[0]
        audit["reason_code"] = str(first.get("reason_code") or "desktop_demo_failed")
        audit["next_hint"] = str(first.get("next_hint") or "")
    audit_path = root / "artifacts" / "audits" / f"desktop_lab_demo_{_ts_label(now)}.json"
    _safe_write_json(audit_path, audit)
    audit["artifact_path"] = str(audit_path)
    _safe_write_json(audit_path, audit)
    return audit


def run_desktop_verify_demo(
    root_dir: Path,
    *,
    rail: str,
    screenshot_path: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    now = time.time()
    rail_n = _normalize_rail(rail)
    screenshot = Path(screenshot_path) if screenshot_path else _write_demo_fixture_screenshot(root, ts=now)
    meta = _load_visual_metadata(screenshot, None)
    contract = resolve_operation_contract(
        action_type="focus_window",
        target={"title_contains": "fixture window", "focus_target": "search_box"},
        expected_efe_desktop={
            "active_window_title_contains": "fixture",
            "focus_target_contains": "search",
            "affordances_any": ["results_panel"],
            "dialogs_absent": True,
        },
        operation_class="focus_window",
    )
    verify_input = build_desktop_verify_input(
        operation_class=contract.get("operation_class"),
        expected_efe_desktop=_as_dict(contract.get("expected_efe_desktop")),
        before_state=_build_verify_visual_state(
            screenshot,
            {
                "active_window_title": str(meta.get("active_window_title") or ""),
                "focus_target": str(meta.get("focus_target") or ""),
                "dialogs": _as_text_list(meta.get("dialogs")),
                "observed_affordances": _dedupe(
                    [f"visible_affordance:{item}" for item in _as_text_list(meta.get("ui_affordances"))]
                    + [f"visual_marker:{item}" for item in _as_text_list(meta.get("visual_markers"))]
                ),
                "markers": _as_text_list(meta.get("visual_markers")),
            },
            meta,
            verdict="pass",
        ),
        after_state=_build_verify_visual_state(
            screenshot,
            {
                "active_window_title": str(meta.get("active_window_title") or ""),
                "focus_target": str(meta.get("focus_target") or ""),
                "dialogs": _as_text_list(meta.get("dialogs")),
                "observed_affordances": _dedupe(
                    [f"visible_affordance:{item}" for item in _as_text_list(meta.get("ui_affordances"))]
                    + [f"visual_marker:{item}" for item in _as_text_list(meta.get("visual_markers"))]
                ),
                "markers": _as_text_list(meta.get("visual_markers")),
            },
            meta,
            verdict="pass",
        ),
        screenshot_before=str(screenshot),
        screenshot_after=str(screenshot),
        arbiter_context={"mode": "post", "demo": True},
        runtime_metadata=meta,
        verify_spec=_arbiter_verify_spec(_as_dict(contract.get("verify_spec"))),
    )
    verify_input["required_evidence"] = _as_text_list(_as_dict(contract.get("verify_spec")).get("evidence_required"))
    verify_input["required_evidence"] = [item for item in verify_input["required_evidence"] if item != "desktop_arbiter_post"]
    verification_result = evaluate_desktop_verify_input(verify_input)
    audit = {
        "schema": VERIFY_DEMO_SCHEMA,
        "ts_utc": _utc_now(now),
        "rail": rail_n,
        "ok": str(verification_result.get("verdict") or "") == "pass",
        "verification_contract_version": DESKTOP_VERIFICATION_CONTRACT_VERSION,
        "fixture_screenshot_path": str(screenshot),
        "verify_input": verify_input,
        "verify_result": verification_result,
        "verify_mismatch": verification_result.get("verify_mismatch") or [],
        "verification_result": verification_result,
        "reason_code": str(verification_result.get("reason_code") or "verify_demo_failed"),
        "next_hint": str(verification_result.get("next_hint") or ""),
    }
    audit_path = root / "artifacts" / "audits" / f"desktop_verify_demo_{_ts_label(now)}.json"
    _safe_write_json(audit_path, audit)
    audit["artifact_path"] = str(audit_path)
    _safe_write_json(audit_path, audit)
    return audit
