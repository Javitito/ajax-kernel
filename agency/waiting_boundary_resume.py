from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BOUNDARY_COMPLETION_SCHEMA = "ajax.waiting_boundary_completion.v1"
BOUNDARY_COMPLETION_VERSION = "v1"
BOUNDARY_RESUME_RECEIPT_SCHEMA = "ajax.receipt.waiting_boundary_resume.v1"
EFE_CANDIDATE_SCHEMA = "ajax.verify.efe_candidate.v0"
SUPPORTED_BOUNDARY_KINDS = {"efe_boundary_completion"}
SUPPORTED_CHECK_KINDS = {"fs", "process", "port", "receipt_schema", "structured_output"}


def utc_now(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def utc_stamp(ts: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts or time.time()))


def safe_copy(doc: Any) -> Any:
    try:
        return copy.deepcopy(doc)
    except Exception:
        if isinstance(doc, dict):
            return {k: safe_copy(v) for k, v in doc.items()}
        if isinstance(doc, list):
            return [safe_copy(v) for v in doc]
        return doc


def read_json_doc(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def write_json_doc(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def receipts_dir(root_dir: Path) -> Path:
    path = root_dir / "artifacts" / "receipts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def completions_dir(root_dir: Path, mission_id: Optional[str]) -> Path:
    safe_mission = str(mission_id or "_unknown").strip() or "_unknown"
    path = root_dir / "artifacts" / "waiting_for_user" / "completions" / safe_mission
    path.mkdir(parents=True, exist_ok=True)
    return path


def waiting_payload_path(root_dir: Path, mission_id: str) -> Path:
    return root_dir / "artifacts" / "waiting_for_user" / f"{mission_id}.json"


def waiting_state_path(root_dir: Path) -> Path:
    return root_dir / "artifacts" / "state" / "waiting_mission.json"


def emit_resume_receipt(
    *,
    root_dir: Path,
    event: str,
    mission_id: Optional[str],
    boundary_kind: Optional[str],
    completion_source: Optional[str],
    waiting_payload_path: Optional[str],
    completion_path: Optional[str],
    candidate_path: Optional[str],
    validation_ok: Optional[bool],
    patch_applied: bool,
    resume_attempted: bool,
    outcome: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> str:
    ts = time.time()
    stamp = utc_stamp(ts)
    payload: Dict[str, Any] = {
        "schema": BOUNDARY_RESUME_RECEIPT_SCHEMA,
        "version": "v1",
        "created_at": utc_now(ts),
        "event": event,
        "mission_id": mission_id,
        "boundary_kind": boundary_kind,
        "completion_source": completion_source,
        "waiting_payload_path": waiting_payload_path,
        "completion_path": completion_path,
        "candidate_path": candidate_path,
        "validation_ok": validation_ok,
        "patch_applied": bool(patch_applied),
        "resume_attempted": bool(resume_attempted),
        "outcome": outcome,
        "detail": detail or {},
    }
    path = receipts_dir(root_dir) / f"waiting_boundary_resume_{stamp}_{event}.json"
    return write_json_doc(path, payload)


def normalize_completion_payload(
    completion_payload: Dict[str, Any], *, source_path: Optional[Path] = None
) -> Dict[str, Any]:
    normalized = safe_copy(completion_payload) if isinstance(completion_payload, dict) else {}
    if not isinstance(normalized, dict):
        normalized = {}
    mission_id = str(normalized.get("mission_id") or "").strip()
    if mission_id:
        normalized["mission_id"] = mission_id
    boundary_kind = str(normalized.get("boundary_kind") or "").strip()
    if boundary_kind:
        normalized["boundary_kind"] = boundary_kind
    completion_source = str(normalized.get("completion_source") or "").strip()
    if completion_source:
        normalized["completion_source"] = completion_source
    boundary_fields = normalized.get("boundary_fields")
    if not isinstance(boundary_fields, dict):
        user_supplied = normalized.get("user_supplied_fields")
        if isinstance(user_supplied, dict):
            normalized["boundary_fields"] = safe_copy(user_supplied)
        else:
            normalized["boundary_fields"] = {}
    candidate_path_raw = normalized.get("candidate_path")
    if isinstance(candidate_path_raw, str) and candidate_path_raw.strip():
        candidate_path = Path(candidate_path_raw).expanduser()
        if not candidate_path.is_absolute() and source_path is not None:
            candidate_path = (source_path.parent / candidate_path).resolve()
        normalized["candidate_path"] = str(candidate_path)
    return normalized


def validate_completion_contract(completion_payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(completion_payload.get("schema") or "").strip() != BOUNDARY_COMPLETION_SCHEMA:
        errors.append("completion_schema_invalid")
    if str(completion_payload.get("version") or "").strip() != BOUNDARY_COMPLETION_VERSION:
        errors.append("completion_version_invalid")
    if not str(completion_payload.get("mission_id") or "").strip():
        errors.append("mission_id_required")
    boundary_kind = str(completion_payload.get("boundary_kind") or "").strip()
    if not boundary_kind:
        errors.append("boundary_kind_required")
    elif boundary_kind not in SUPPORTED_BOUNDARY_KINDS:
        errors.append(f"unsupported_boundary_kind:{boundary_kind}")
    if not str(completion_payload.get("completion_source") or "").strip():
        errors.append("completion_source_required")
    if not str(completion_payload.get("completed_utc") or "").strip():
        errors.append("completed_utc_required")
    if not isinstance(completion_payload.get("boundary_fields"), dict):
        errors.append("boundary_fields_required")
    if not completion_payload.get("candidate_path") and not isinstance(
        completion_payload.get("candidate_payload"), dict
    ):
        errors.append("candidate_path_or_payload_required")
    return errors


def extract_expected_boundary(waiting_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ask_payload = waiting_payload.get("ask_user_payload")
    if isinstance(ask_payload, dict):
        ctx = ask_payload.get("context")
        if isinstance(ctx, dict) and isinstance(ctx.get("efe_boundary"), dict):
            return safe_copy(ctx.get("efe_boundary"))
    mission_raw = waiting_payload.get("mission")
    if isinstance(mission_raw, dict):
        for key in ("pending_plan", "last_plan"):
            plan = mission_raw.get(key)
            if not isinstance(plan, dict):
                continue
            meta = plan.get("metadata")
            if isinstance(meta, dict) and isinstance(meta.get("efe_boundary"), dict):
                return safe_copy(meta.get("efe_boundary"))
    return None


def extract_resume_plan(waiting_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ask_payload = waiting_payload.get("ask_user_payload")
    if isinstance(ask_payload, dict):
        ctx = ask_payload.get("context")
        if isinstance(ctx, dict):
            for key in ("efe_resume_plan", "efe_candidate_source_doc"):
                candidate = ctx.get(key)
                if isinstance(candidate, dict) and isinstance(candidate.get("steps"), list):
                    return safe_copy(candidate)
    mission_raw = waiting_payload.get("mission")
    if isinstance(mission_raw, dict):
        for key in ("pending_plan", "last_plan"):
            plan = mission_raw.get(key)
            if not isinstance(plan, dict):
                continue
            meta = plan.get("metadata")
            if isinstance(meta, dict):
                for meta_key in ("efe_resume_plan", "efe_candidate_source_doc"):
                    candidate = meta.get(meta_key)
                    if isinstance(candidate, dict) and isinstance(candidate.get("steps"), list):
                        return safe_copy(candidate)
    return None


def is_expected_state_bounded(expected_state: Any) -> bool:
    if not isinstance(expected_state, dict):
        return False
    allowed_keys = {"windows", "files", "checks", "meta"}
    if any(key not in allowed_keys for key in expected_state.keys()):
        return False
    if expected_state.get("windows"):
        return True
    if expected_state.get("files"):
        return True
    checks = expected_state.get("checks")
    if isinstance(checks, list) and checks:
        for check in checks:
            if not isinstance(check, dict):
                return False
            kind = str(check.get("kind") or "").strip().lower()
            if kind not in SUPPORTED_CHECK_KINDS:
                return False
        return True
    meta = expected_state.get("meta")
    return bool(isinstance(meta, dict) and meta.get("must_be_active"))


def validate_candidate_doc(candidate_doc: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if str(candidate_doc.get("schema") or "").strip() != EFE_CANDIDATE_SCHEMA:
        errors.append("candidate_schema_invalid")
    expected_state = candidate_doc.get("expected_state")
    if not isinstance(expected_state, dict):
        errors.append("candidate_expected_state_missing")
    elif not is_expected_state_bounded(expected_state):
        errors.append("candidate_expected_state_unbounded")
    return errors


def materialize_candidate(
    *,
    root_dir: Path,
    mission_id: str,
    completion_payload: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_doc = completion_payload.get("candidate_payload")
    if isinstance(candidate_doc, dict):
        normalized = safe_copy(candidate_doc)
        stamp = utc_stamp()
        out_path = root_dir / "artifacts" / "efe_candidates" / f"{stamp}_{mission_id}_completed.json"
        write_json_doc(out_path, normalized)
        return {
            "candidate_doc": normalized,
            "candidate_path": str(out_path),
            "candidate_source": "payload",
        }
    candidate_path_raw = str(completion_payload.get("candidate_path") or "").strip()
    if not candidate_path_raw:
        raise ValueError("candidate_path_or_payload_required")
    candidate_path = Path(candidate_path_raw).expanduser()
    candidate_doc = read_json_doc(candidate_path)
    if candidate_doc is None:
        raise ValueError("candidate_path_not_found")
    return {
        "candidate_doc": candidate_doc,
        "candidate_path": str(candidate_path),
        "candidate_source": "path",
    }


def build_completion_summary(
    *,
    completion_payload: Dict[str, Any],
    completion_path: str,
    candidate_path: Optional[str],
) -> Dict[str, Any]:
    summary = {
        "schema": BOUNDARY_COMPLETION_SCHEMA,
        "version": BOUNDARY_COMPLETION_VERSION,
        "mission_id": completion_payload.get("mission_id"),
        "boundary_kind": completion_payload.get("boundary_kind"),
        "completion_source": completion_payload.get("completion_source"),
        "completed_utc": completion_payload.get("completed_utc"),
        "completion_path": completion_path,
        "candidate_path": candidate_path,
        "boundary_fields": safe_copy(completion_payload.get("boundary_fields") or {}),
        "notes": completion_payload.get("notes"),
    }
    return summary


def patch_resume_plan(
    *,
    resume_plan: Dict[str, Any],
    expected_boundary: Dict[str, Any],
    candidate_doc: Dict[str, Any],
    completion_summary: Dict[str, Any],
) -> Dict[str, Any]:
    patched = safe_copy(resume_plan)
    steps = patched.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("resume_plan_missing_steps")
    step_id = str(
        completion_summary.get("boundary_fields", {}).get("step_id")
        or expected_boundary.get("step_id")
        or ""
    ).strip()
    if not step_id:
        raise ValueError("boundary_step_id_missing")
    target_step: Optional[Dict[str, Any]] = None
    filtered_steps: List[Dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("action") or "").strip() == "await_user_input":
            continue
        if str(step.get("id") or "").strip() == step_id:
            target_step = step
        filtered_steps.append(step)
    if target_step is None:
        raise ValueError(f"resume_step_not_found:{step_id}")
    success_spec = target_step.get("success_spec") if isinstance(target_step.get("success_spec"), dict) else {}
    success_spec["expected_state"] = safe_copy(candidate_doc.get("expected_state"))
    target_step["success_spec"] = success_spec
    patched["steps"] = filtered_steps
    meta = patched.get("metadata") if isinstance(patched.get("metadata"), dict) else {}
    meta["boundary_completion_resolved"] = True
    meta["boundary_completion"] = safe_copy(completion_summary)
    meta["efe_candidate_path"] = completion_summary.get("candidate_path")
    meta["efe_boundary"] = safe_copy(expected_boundary)
    patched["metadata"] = meta
    success_contract = patched.get("success_contract")
    if not isinstance(success_contract, dict) or str(success_contract.get("type") or "").strip() == "await_user_input":
        patched["success_contract"] = {"type": "check_last_step_status"}
    return patched


def to_waiting_plan_doc(plan_doc: Dict[str, Any]) -> Dict[str, Any]:
    metadata = plan_doc.get("metadata") if isinstance(plan_doc.get("metadata"), dict) else {}
    success_spec = plan_doc.get("success_spec")
    if not isinstance(success_spec, dict):
        success_contract = plan_doc.get("success_contract")
        success_spec = success_contract if isinstance(success_contract, dict) else {"type": "check_last_step_status"}
    return {
        "id": plan_doc.get("id") or plan_doc.get("plan_id"),
        "summary": plan_doc.get("summary") or plan_doc.get("description") or "Boundary-resumed plan",
        "steps": safe_copy(plan_doc.get("steps") or []),
        "plan_id": plan_doc.get("plan_id") or plan_doc.get("id"),
        "metadata": safe_copy(metadata),
        "success_spec": safe_copy(success_spec),
    }


def apply_waiting_boundary_completion(
    *,
    root_dir: Path,
    completion_payload: Dict[str, Any],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    normalized = normalize_completion_payload(completion_payload, source_path=source_path)
    mission_id = str(normalized.get("mission_id") or "").strip() or None
    boundary_kind = str(normalized.get("boundary_kind") or "").strip() or None
    completion_source = str(normalized.get("completion_source") or "").strip() or None
    completion_dir = completions_dir(root_dir, mission_id)
    completion_path = completion_dir / f"boundary_completion_{utc_stamp()}.json"
    completion_path_str = write_json_doc(completion_path, normalized)
    receipt_paths: List[str] = []
    receipt_paths.append(
        emit_resume_receipt(
            root_dir=root_dir,
            event="completion_received",
            mission_id=mission_id,
            boundary_kind=boundary_kind,
            completion_source=completion_source,
            waiting_payload_path=str(waiting_payload_path(root_dir, mission_id)) if mission_id else None,
            completion_path=completion_path_str,
            candidate_path=str(normalized.get("candidate_path") or "") or None,
            validation_ok=None,
            patch_applied=False,
            resume_attempted=False,
            detail={"source_path": str(source_path) if source_path else None},
        )
    )

    contract_errors = validate_completion_contract(normalized)
    if contract_errors:
        receipt_paths.append(
            emit_resume_receipt(
                root_dir=root_dir,
                event="completion_refused",
                mission_id=mission_id,
                boundary_kind=boundary_kind,
                completion_source=completion_source,
                waiting_payload_path=str(waiting_payload_path(root_dir, mission_id)) if mission_id else None,
                completion_path=completion_path_str,
                candidate_path=str(normalized.get("candidate_path") or "") or None,
                validation_ok=False,
                patch_applied=False,
                resume_attempted=False,
                outcome="WAITING_FOR_USER",
                detail={"reasons": contract_errors},
            )
        )
        return {
            "ok": False,
            "mission_id": mission_id,
            "status": "WAITING_FOR_USER",
            "reason": "boundary_completion_invalid",
            "errors": contract_errors,
            "completion_path": completion_path_str,
            "receipt_paths": receipt_paths,
            "resume_attempted": False,
            "patch_applied": False,
        }

    per_path = waiting_payload_path(root_dir, str(mission_id))
    pointer_path = waiting_state_path(root_dir)
    pointer_doc = read_json_doc(pointer_path)
    waiting_doc = None
    if isinstance(pointer_doc, dict) and str(pointer_doc.get("mission_id") or "").strip() == str(mission_id):
        waiting_doc = safe_copy(pointer_doc)
    if waiting_doc is None:
        refusal = ["waiting_mission_not_found"]
        receipt_paths.append(
            emit_resume_receipt(
                root_dir=root_dir,
                event="completion_refused",
                mission_id=mission_id,
                boundary_kind=boundary_kind,
                completion_source=completion_source,
                waiting_payload_path=str(per_path),
                completion_path=completion_path_str,
                candidate_path=str(normalized.get("candidate_path") or "") or None,
                validation_ok=False,
                patch_applied=False,
                resume_attempted=False,
                outcome="WAITING_FOR_USER",
                detail={"reasons": refusal},
            )
        )
        return {
            "ok": False,
            "mission_id": mission_id,
            "status": "WAITING_FOR_USER",
            "reason": "waiting_mission_not_found",
            "errors": refusal,
            "completion_path": completion_path_str,
            "receipt_paths": receipt_paths,
            "resume_attempted": False,
            "patch_applied": False,
        }

    validation_errors: List[str] = []
    waiting_status = str(waiting_doc.get("status") or "").strip().upper()
    if waiting_status != "WAITING_FOR_USER":
        validation_errors.append(f"mission_not_waiting:{waiting_status or 'unknown'}")
    if str(waiting_doc.get("mission_id") or "").strip() != str(mission_id):
        validation_errors.append("mission_id_mismatch")
    expected_boundary = extract_expected_boundary(waiting_doc)
    if not isinstance(expected_boundary, dict):
        validation_errors.append("expected_boundary_missing")
    else:
        expected_kind = str(expected_boundary.get("kind") or "").strip()
        if boundary_kind != expected_kind:
            validation_errors.append(f"boundary_kind_mismatch:{expected_kind or 'missing'}")

    resume_plan = extract_resume_plan(waiting_doc)
    if not isinstance(resume_plan, dict):
        validation_errors.append("resume_plan_missing")

    candidate_info: Dict[str, Any] = {}
    if not validation_errors:
        try:
            candidate_info = materialize_candidate(
                root_dir=root_dir,
                mission_id=str(mission_id),
                completion_payload=normalized,
            )
        except ValueError as exc:
            validation_errors.append(str(exc))

    if not validation_errors:
        candidate_errors = validate_candidate_doc(candidate_info.get("candidate_doc") or {})
        validation_errors.extend(candidate_errors)

    if validation_errors:
        receipt_paths.append(
            emit_resume_receipt(
                root_dir=root_dir,
                event="completion_refused",
                mission_id=mission_id,
                boundary_kind=boundary_kind,
                completion_source=completion_source,
                waiting_payload_path=str(per_path),
                completion_path=completion_path_str,
                candidate_path=candidate_info.get("candidate_path")
                or (str(normalized.get("candidate_path") or "") or None),
                validation_ok=False,
                patch_applied=False,
                resume_attempted=False,
                outcome="WAITING_FOR_USER",
                detail={"reasons": validation_errors},
            )
        )
        return {
            "ok": False,
            "mission_id": mission_id,
            "status": "WAITING_FOR_USER",
            "reason": "boundary_completion_refused",
            "errors": validation_errors,
            "completion_path": completion_path_str,
            "waiting_payload_path": str(per_path),
            "receipt_paths": receipt_paths,
            "resume_attempted": False,
            "patch_applied": False,
        }

    completion_summary = build_completion_summary(
        completion_payload=normalized,
        completion_path=completion_path_str,
        candidate_path=candidate_info.get("candidate_path"),
    )
    try:
        patched_resume_plan = patch_resume_plan(
            resume_plan=resume_plan or {},
            expected_boundary=expected_boundary or {},
            candidate_doc=candidate_info.get("candidate_doc") or {},
            completion_summary=completion_summary,
        )
    except ValueError as exc:
        reasons = [str(exc)]
        receipt_paths.append(
            emit_resume_receipt(
                root_dir=root_dir,
                event="completion_refused",
                mission_id=mission_id,
                boundary_kind=boundary_kind,
                completion_source=completion_source,
                waiting_payload_path=str(per_path),
                completion_path=completion_path_str,
                candidate_path=candidate_info.get("candidate_path"),
                validation_ok=False,
                patch_applied=False,
                resume_attempted=False,
                outcome="WAITING_FOR_USER",
                detail={"reasons": reasons},
            )
        )
        return {
            "ok": False,
            "mission_id": mission_id,
            "status": "WAITING_FOR_USER",
            "reason": "boundary_completion_refused",
            "errors": reasons,
            "completion_path": completion_path_str,
            "waiting_payload_path": str(per_path),
            "receipt_paths": receipt_paths,
            "resume_attempted": False,
            "patch_applied": False,
        }

    patched_waiting = safe_copy(waiting_doc)
    patched_waiting["updated_utc"] = utc_now()
    patched_waiting["boundary_completion"] = safe_copy(completion_summary)
    history = patched_waiting.get("boundary_completion_history")
    if not isinstance(history, list):
        history = []
    history.append(safe_copy(completion_summary))
    patched_waiting["boundary_completion_history"] = history[-10:]
    ask_payload = patched_waiting.get("ask_user_payload")
    if isinstance(ask_payload, dict):
        ctx = ask_payload.get("context")
        if not isinstance(ctx, dict):
            ctx = {}
        ctx["boundary_completion"] = safe_copy(completion_summary)
        ask_payload["context"] = ctx
        patched_waiting["ask_user_payload"] = ask_payload
    mission_raw = patched_waiting.get("mission")
    if not isinstance(mission_raw, dict):
        mission_raw = {}
    waiting_plan_doc = to_waiting_plan_doc(patched_resume_plan)
    mission_raw["pending_plan"] = safe_copy(waiting_plan_doc)
    mission_raw["last_plan"] = safe_copy(waiting_plan_doc)
    mission_raw["feedback"] = None
    notes = mission_raw.get("notes")
    if not isinstance(notes, dict):
        notes = {}
    notes["boundary_completion"] = safe_copy(completion_summary)
    notes["boundary_completion_kind"] = boundary_kind
    notes["boundary_completion_source"] = completion_source
    notes["boundary_completion_path"] = completion_path_str
    notes["boundary_completion_candidate_path"] = candidate_info.get("candidate_path")
    notes["boundary_completion_validated_utc"] = utc_now()
    notes["_pending_user_options"] = []
    mission_raw["notes"] = notes
    mission_raw["pending_user_options"] = []
    patched_waiting["mission"] = mission_raw

    write_json_doc(pointer_path, patched_waiting)
    per_doc = read_json_doc(per_path)
    if not isinstance(per_doc, dict):
        per_doc = {"mission_id": mission_id, "schema": "ajax.waiting_mission.v1"}
    per_doc["updated_utc"] = patched_waiting.get("updated_utc")
    per_doc["boundary_completion"] = safe_copy(completion_summary)
    per_doc["candidate_path"] = candidate_info.get("candidate_path")
    per_doc["completion_path"] = completion_path_str
    write_json_doc(per_path, per_doc)

    receipt_paths.append(
        emit_resume_receipt(
            root_dir=root_dir,
            event="completion_validated",
            mission_id=mission_id,
            boundary_kind=boundary_kind,
            completion_source=completion_source,
            waiting_payload_path=str(per_path),
            completion_path=completion_path_str,
            candidate_path=candidate_info.get("candidate_path"),
            validation_ok=True,
            patch_applied=False,
            resume_attempted=False,
            detail={"resume_plan_available": True},
        )
    )
    receipt_paths.append(
        emit_resume_receipt(
            root_dir=root_dir,
            event="waiting_patched",
            mission_id=mission_id,
            boundary_kind=boundary_kind,
            completion_source=completion_source,
            waiting_payload_path=str(per_path),
            completion_path=completion_path_str,
            candidate_path=candidate_info.get("candidate_path"),
            validation_ok=True,
            patch_applied=True,
            resume_attempted=False,
            detail={
                "patched_fields": [
                    "boundary_completion",
                    "mission.pending_plan",
                    "mission.last_plan",
                    "mission.notes.boundary_completion",
                ]
            },
        )
    )
    return {
        "ok": True,
        "mission_id": mission_id,
        "status": "WAITING_FOR_USER",
        "boundary_kind": boundary_kind,
        "completion_source": completion_source,
        "completion_path": completion_path_str,
        "waiting_payload_path": str(per_path),
        "waiting_state_path": str(pointer_path),
        "candidate_path": candidate_info.get("candidate_path"),
        "candidate_doc": candidate_info.get("candidate_doc"),
        "patched_payload": patched_waiting,
        "receipt_paths": receipt_paths,
        "resume_attempted": False,
        "patch_applied": True,
        "validation_ok": True,
    }
