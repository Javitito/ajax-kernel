from __future__ import annotations

from typing import Any, Dict, Literal, Optional, TypedDict


DESKTOP_VERIFICATION_CONTRACT_VERSION = "desktop_v1"

DesktopVerdict = Literal["pass", "fail", "uncertain"]
DesktopMismatchSeverity = Literal["low", "medium", "high"]


class DesktopMismatch(TypedDict):
    field: str
    expected: Any
    observed: Any
    severity: DesktopMismatchSeverity
    note: str


class DesktopVerifyInput(TypedDict):
    operation_class: Optional[str]
    expected_efe_desktop: Dict[str, Any]
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    screenshot_before: Optional[str]
    screenshot_after: Optional[str]
    arbiter_context: Dict[str, Any]
    runtime_metadata: Dict[str, Any]
    verify_spec_ref: Optional[str]
    verify_spec: Dict[str, Any]
    verification_contract_version: str


class DesktopVerificationResult(TypedDict):
    verdict: DesktopVerdict
    mismatches: list[DesktopMismatch]
    reason_code: str
    next_hint: str
    confidence: str
    verification_contract_version: str


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_desktop_verdict(value: Any) -> DesktopVerdict:
    verdict = _as_text(value).lower()
    if verdict == "pass":
        return "pass"
    if verdict == "fail":
        return "fail"
    return "uncertain"


def normalize_desktop_mismatch_severity(value: Any) -> DesktopMismatchSeverity:
    severity = _as_text(value).lower()
    if severity == "low":
        return "low"
    if severity == "high":
        return "high"
    return "medium"


def build_desktop_mismatch(
    field: str,
    *,
    expected: Any = None,
    observed: Any = None,
    severity: Any = "medium",
    note: str = "",
) -> DesktopMismatch:
    return {
        "field": _as_text(field),
        "expected": expected,
        "observed": observed,
        "severity": normalize_desktop_mismatch_severity(severity),
        "note": _as_text(note),
    }


def normalize_desktop_mismatch(value: Any) -> DesktopMismatch:
    if isinstance(value, dict):
        return build_desktop_mismatch(
            str(value.get("field") or "unspecified"),
            expected=value.get("expected"),
            observed=value.get("observed"),
            severity=value.get("severity") or "medium",
            note=str(value.get("note") or ""),
        )
    text = _as_text(value)
    return build_desktop_mismatch(
        text or "unspecified",
        expected=None,
        observed=None,
        severity="medium",
        note=text,
    )


def normalize_desktop_mismatches(values: Any) -> list[DesktopMismatch]:
    if isinstance(values, list):
        raw_items = values
    elif values is None:
        raw_items = []
    else:
        raw_items = [values]
    out: list[DesktopMismatch] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_items:
        mismatch = normalize_desktop_mismatch(item)
        key = (
            mismatch["field"],
            str(mismatch.get("expected")),
            str(mismatch.get("observed")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(mismatch)
    return out


def build_desktop_verify_input(
    *,
    operation_class: Optional[str],
    expected_efe_desktop: Optional[Dict[str, Any]],
    before_state: Optional[Dict[str, Any]],
    after_state: Optional[Dict[str, Any]],
    screenshot_before: Optional[str] = None,
    screenshot_after: Optional[str] = None,
    arbiter_context: Optional[Dict[str, Any]] = None,
    runtime_metadata: Optional[Dict[str, Any]] = None,
    verify_spec: Optional[Dict[str, Any]] = None,
) -> DesktopVerifyInput:
    before_state_n = _as_dict(before_state)
    after_state_n = _as_dict(after_state)
    runtime_n = _as_dict(runtime_metadata)
    verify_spec_n = _as_dict(verify_spec)
    return {
        "operation_class": _as_text(operation_class).lower() or None,
        "expected_efe_desktop": dict(_as_dict(expected_efe_desktop)),
        "before_state": before_state_n,
        "after_state": after_state_n,
        "screenshot_before": _as_text(screenshot_before or before_state_n.get("screenshot_path")) or None,
        "screenshot_after": _as_text(screenshot_after or after_state_n.get("screenshot_path")) or None,
        "arbiter_context": dict(_as_dict(arbiter_context)),
        "runtime_metadata": runtime_n,
        "verify_spec_ref": _as_text(verify_spec_n.get("name")) or None,
        "verify_spec": verify_spec_n,
        "verification_contract_version": DESKTOP_VERIFICATION_CONTRACT_VERSION,
    }


def normalize_desktop_verify_input(value: Any) -> DesktopVerifyInput:
    payload = _as_dict(value)
    return build_desktop_verify_input(
        operation_class=payload.get("operation_class"),
        expected_efe_desktop=_as_dict(payload.get("expected_efe_desktop")),
        before_state=_as_dict(payload.get("before_state")),
        after_state=_as_dict(payload.get("after_state")),
        screenshot_before=payload.get("screenshot_before"),
        screenshot_after=payload.get("screenshot_after"),
        arbiter_context=_as_dict(payload.get("arbiter_context")),
        runtime_metadata=_as_dict(payload.get("runtime_metadata")),
        verify_spec=_as_dict(payload.get("verify_spec")),
    )


def build_desktop_verification_result(
    *,
    verdict: Any,
    mismatches: Any,
    reason_code: str,
    next_hint: str,
    confidence: str,
) -> DesktopVerificationResult:
    confidence_n = _as_text(confidence).lower()
    if confidence_n not in {"low", "medium", "high"}:
        confidence_n = "medium"
    return {
        "verdict": normalize_desktop_verdict(verdict),
        "mismatches": normalize_desktop_mismatches(mismatches),
        "reason_code": _as_text(reason_code),
        "next_hint": _as_text(next_hint),
        "confidence": confidence_n,
        "verification_contract_version": DESKTOP_VERIFICATION_CONTRACT_VERSION,
    }


def normalize_desktop_verification_result(value: Any) -> DesktopVerificationResult:
    payload = _as_dict(value)
    return build_desktop_verification_result(
        verdict=payload.get("verdict"),
        mismatches=payload.get("mismatches"),
        reason_code=str(payload.get("reason_code") or ""),
        next_hint=str(payload.get("next_hint") or ""),
        confidence=str(payload.get("confidence") or ""),
    )
