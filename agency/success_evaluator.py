from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional, Literal

from agency.mission_envelope import SuccessContract  # type: ignore
from agency.windows_driver_client import WindowsDriverClient  # type: ignore
from agency.vision_llm import call_vision_llm  # type: ignore
from agency.visual_auditor import VisualAuditor  # type: ignore

log = logging.getLogger(__name__)


def _build_result(
    ok: bool,
    reason: str,
    *,
    source: str,
    score: float = 1.0,
    error: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": bool(ok),
        "reason": reason,
        "score": score,
        "source": source,
        "error": error,
        "detail": detail or {},
    }


def _normalize_contract(spec: Any) -> SuccessContract:
    if isinstance(spec, SuccessContract):
        return spec
    if isinstance(spec, dict):
        if "primary_source" in spec or "primary_check" in spec:
            return SuccessContract(
                primary_source=str(spec.get("primary_source") or "uia"),
                primary_check=spec.get("primary_check") or {},
                fallback_source=spec.get("fallback_source", "none"),
                fallback_check=spec.get("fallback_check"),
                conflict_resolution=str(spec.get("conflict_resolution") or "fail_safe"),
            )
        # compatibilidad: success_spec simple -> primary_check
        return SuccessContract(primary_source="uia", primary_check=spec)
    return SuccessContract(primary_source="uia", primary_check={"type": "check_last_step_status"})


def _check_window_title(driver: Optional[WindowsDriverClient], check: Dict[str, Any]) -> Dict[str, Any]:
    if driver is None:
        return _build_result(False, "driver_missing", source="uia", error="driver_none", score=0.0)
    raw_text = check.get("text") or check.get("title")
    text = str(raw_text or "").lower()
    must_active = bool(check.get("must_be_active", True))
    if not text:
        return _build_result(True, "no_text_expected", source="uia", score=1.0)
    last_title = ""
    for _ in range(5):
        try:
            fg = driver.get_active_window()
            last_title = str((fg or {}).get("title") or "")
            if text in last_title.lower():
                return _build_result(True, "title_match", source="uia", score=1.0, detail={"title": last_title})
            if not must_active:
                try:
                    res = driver.window_focus(title_contains=text)
                    title2 = str((res or {}).get("title") or "")
                    if text in title2.lower():
                        return _build_result(True, "title_match_after_focus", source="uia", score=1.0, detail={"title": title2})
                except Exception:
                    pass
        except Exception as exc:
            return _build_result(False, f"driver_error:{exc}", source="uia", error="driver_exception", score=0.0)
        time.sleep(1.0)
    return _build_result(False, f"title_missing:{text}", source="uia", error="not_found", score=0.0, detail={"observed": last_title})


def _check_active_window_process(driver: Optional[WindowsDriverClient], check: Dict[str, Any]) -> Dict[str, Any]:
    if driver is None:
        return _build_result(False, "driver_missing", source="uia", error="driver_none", score=0.0)
    raw = check.get("processes")
    processes: list[str] = []
    if isinstance(raw, list):
        processes = [str(p) for p in raw if isinstance(p, (str, int, float)) and str(p).strip()]
    elif isinstance(check.get("process"), str):
        processes = [str(check.get("process")).strip()]
    if not processes:
        return _build_result(False, "no_processes_expected", source="uia", error="invalid_check", score=0.0)
    try:
        fg = driver.get_active_window()
    except Exception as exc:
        return _build_result(False, f"driver_error:{exc}", source="uia", error="driver_exception", score=0.0)
    active_process = str((fg or {}).get("process") or "").lower()
    allowed = [p.lower() for p in processes]
    if active_process and active_process in allowed:
        return _build_result(True, "active_process_match", source="uia", score=1.0, detail={"process": active_process})
    return _build_result(
        False,
        "active_process_mismatch",
        source="uia",
        error="not_found",
        score=0.0,
        detail={"observed": active_process, "expected_any_of": processes},
    )


def _check_visual(driver: Optional[WindowsDriverClient], check: Dict[str, Any], intention: Optional[str]) -> Dict[str, Any]:
    description = check.get("description") or check.get("text") or intention or ""
    if not description:
        return _build_result(True, "no_description", source="vision", score=1.0)
    if driver is None:
        return _build_result(False, "driver_missing_for_vision", source="vision", error="driver_none", score=0.0)
    try:
        auditor = VisualAuditor(driver, backend="local")
        result = auditor.verify(intention or description, description)
        return _build_result(bool(result.success), result.reason if hasattr(result, "reason") else "visual_audit", source="vision", score=1.0 if result.success else 0.0, detail={"audit": getattr(result, "__dict__", {})})
    except Exception as exc:
        # fallback a LLM de visión si existe
        try:
            vision_res = call_vision_llm(
                description,
                instructions="Confirma si la pantalla cumple la descripción.",
                strategy_profile="pc_mini_32gb",
            )
            ok = bool(vision_res.get("ok", False)) if isinstance(vision_res, dict) else False
            reason = vision_res.get("reason") if isinstance(vision_res, dict) else str(vision_res)
            return _build_result(ok, reason or f"vision_llm:{exc}", source="vision", score=1.0 if ok else 0.0, detail={"vision": vision_res})
        except Exception:
            return _build_result(False, f"vision_error:{exc}", source="vision", error="vision_exception", score=0.0)


def _evaluate_check(
    driver: Optional[WindowsDriverClient],
    source: Literal["uia", "fs", "vision", "none"],
    check: Optional[Dict[str, Any]],
    intention: Optional[str],
    last_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if source == "none":
        return _build_result(True, "no_check", source=source, score=1.0)
    if not check:
        return _build_result(True, "empty_check", source=source, score=1.0)

    stype = str(check.get("type") or check.get("kind") or "").lower()
    method = str(check.get("method") or stype).lower()

    if source == "vision" or method == "visual_audit":
        return _check_visual(driver, check, intention)

    if method == "check_last_step_status":
        return _build_result(True, "defer_to_runner", source=source, score=1.0, detail={"last_result": last_result or {}})

    if method == "window_title_contains":
        return _check_window_title(driver, check)
    if method == "desktop_isolated_active_window":
        # reutiliza check de ventana + opcionalmente proceso
        return _check_window_title(driver, {"text": check.get("text") or "", "must_be_active": True})
    if method == "active_window_process_in":
        return _check_active_window_process(driver, check)

    return _build_result(False, f"unknown_check:{method}", source=source, score=0.0, error="unknown_check")


def evaluate_success(
    driver: Optional[WindowsDriverClient],
    spec: Any,
    intention: Optional[str] = None,
    last_result: Optional[Dict[str, Any]] = None,
    mission_id: Optional[str] = None,
    attempt: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evalúa la condición de éxito usando SuccessContract.
    Devuelve dict estructurado: {ok, score, reason, primary, fallback}
    """
    contract = _normalize_contract(spec)
    primary = _evaluate_check(driver, contract.primary_source, contract.primary_check, intention, last_result)
    ok = primary.get("ok", False)
    score = primary.get("score", 0.0)
    reason = primary.get("reason", "")
    fallback_result: Optional[Dict[str, Any]] = None

    if (not ok) and contract.fallback_source and contract.fallback_source != "none":
        fallback_result = _evaluate_check(driver, contract.fallback_source, contract.fallback_check or {}, intention, last_result)
        if contract.conflict_resolution == "primary_wins":
            ok = ok
            score = score
            reason = reason
        elif contract.conflict_resolution == "ask_human":
            ok = ok and fallback_result.get("ok", False)
            if not ok:
                reason = f"Needs human confirmation: primary={reason}; fallback={fallback_result.get('reason')}"
                score = 0.0
        else:  # fail_safe (por defecto)
            ok = ok and fallback_result.get("ok", False)
            score = 1.0 if ok else 0.0
            if not ok:
                reason = fallback_result.get("reason") or reason

    result = {
        "ok": bool(ok),
        "score": score,
        "reason": reason,
        "primary": primary,
        "fallback": fallback_result,
    }
    try:
        log.info(
            "success_eval: mission=%s attempt=%s ok=%s score=%s reason=%s primary=%s fallback=%s",
            mission_id or "-",
            attempt or "-",
            result["ok"],
            result["score"],
            result["reason"],
            primary.get("reason"),
            (fallback_result or {}).get("reason"),
        )
    except Exception:
        pass
    return result
