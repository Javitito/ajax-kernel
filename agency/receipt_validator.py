from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_WARN_REASON_CODES = {"unsupported_receipt_schema", "missing_schema_field"}


def _utc_now(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _read_receipt_json(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None, "io_error"
    try:
        raw = json.loads(text)
    except Exception:
        return None, "json_parse_error"
    if not isinstance(raw, dict):
        return None, "json_parse_error"
    return raw, None


def _schema_path_for_receipt(root_dir: Path, receipt_schema: str) -> Optional[Path]:
    schemas_dir = Path(root_dir) / "schemas" / "receipts"
    if receipt_schema in {
        "ajax.lab.session.init.v0",
        "ajax.lab.session.status.v0",
        "ajax.lab.session.revoke.v0",
        "ajax.lab.session_status.v0",
    }:
        return schemas_dir / "ajax.lab.session.v0.schema.json"
    if receipt_schema == "ajax.lab.session.migrated.v1":
        return schemas_dir / "ajax.lab.session.migrated.v1.schema.json"
    if receipt_schema in {"ajax.lab.autopilot_tick.v1", "ajax.lab.autopilot_tick.v0"}:
        return schemas_dir / "ajax.lab.autopilot_tick.v1.schema.json"
    if receipt_schema in {"ajax.topology_doctor.v0", "ajax.topology_doctor.v1"}:
        return schemas_dir / "ajax.topology_doctor.v0.schema.json"
    if receipt_schema == "ajax.desktop.role_receipt.v1":
        return schemas_dir / "ajax.desktop.role_receipt.v1.schema.json"
    return None


def _is_type(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return True


def _validate_node(value: Any, schema: Dict[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        nested_errors: List[str] = []
        for candidate in any_of:
            if not isinstance(candidate, dict):
                continue
            trial = _validate_node(value, candidate, path)
            if not trial:
                return []
            nested_errors.extend(trial)
        if nested_errors:
            errors.append(f"{path}: failed anyOf validation")
            return errors

    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _is_type(value, expected_type):
        errors.append(f"{path}: expected type {expected_type}")
        return errors
    if isinstance(expected_type, list):
        if not any(_is_type(value, str(kind)) for kind in expected_type):
            errors.append(f"{path}: expected one of types {expected_type}")
            return errors

    if "enum" in schema and isinstance(schema["enum"], list):
        if value not in schema["enum"]:
            errors.append(f"{path}: value not in enum {schema['enum']}")
            return errors

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: value must be {schema['const']!r}")
        return errors

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    errors.append(f"{path}: missing required field '{key}'")
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, sub_schema in properties.items():
                if key not in value or not isinstance(sub_schema, dict):
                    continue
                errors.extend(_validate_node(value[key], sub_schema, f"{path}.{key}"))
    elif isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(_validate_node(item, items_schema, f"{path}[{idx}]"))

    return errors


def validate_receipt(root_dir: Path, receipt_path: Path) -> Dict[str, Any]:
    root = Path(root_dir)
    path = Path(receipt_path)
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "errors": ["io_error"],
            "reason_codes": ["io_error"],
            "schema_in_receipt": None,
            "schema_path": None,
            "schema_used": None,
        }
    payload, load_error = _read_receipt_json(path)
    if payload is None:
        return {
            "ok": False,
            "path": str(path),
            "errors": [str(load_error or "json_parse_error")],
            "reason_codes": [str(load_error or "json_parse_error")],
            "schema_in_receipt": None,
            "schema_path": None,
            "schema_used": None,
        }
    raw_schema = payload.get("schema")
    if raw_schema is None or not str(raw_schema).strip():
        return {
            "ok": False,
            "path": str(path),
            "errors": ["missing_schema_field"],
            "reason_codes": ["missing_schema_field"],
            "schema_in_receipt": None,
            "schema_path": None,
            "schema_used": None,
        }
    schema_in_receipt = str(raw_schema).strip()
    schema_path = _schema_path_for_receipt(root, schema_in_receipt)
    if schema_path is None:
        return {
            "ok": False,
            "path": str(path),
            "errors": [f"unsupported_receipt_schema:{schema_in_receipt or 'missing'}"],
            "reason_codes": ["unsupported_receipt_schema"],
            "schema_in_receipt": schema_in_receipt or None,
            "schema_path": None,
            "schema_used": None,
        }
    schema_doc = _safe_read_json(schema_path)
    if schema_doc is None:
        return {
            "ok": False,
            "path": str(path),
            "errors": [f"schema_file_invalid:{schema_path}"],
            "reason_codes": ["io_error"],
            "schema_in_receipt": schema_in_receipt or None,
            "schema_path": str(schema_path),
            "schema_used": schema_path.name,
        }
    errors = _validate_node(payload, schema_doc, "$")
    return {
        "ok": len(errors) == 0,
        "path": str(path),
        "errors": errors,
        "reason_codes": [] if len(errors) == 0 else ["invalid_against_schema"],
        "schema_in_receipt": schema_in_receipt or None,
        "schema_path": str(schema_path),
        "schema_used": schema_path.name,
    }


def _normalize_reason_codes(report: Dict[str, Any]) -> List[str]:
    raw_codes = report.get("reason_codes")
    if isinstance(raw_codes, list):
        out = []
        for code in raw_codes:
            if isinstance(code, str) and code.strip():
                out.append(code.strip())
        if out:
            return out
    # Backward-compatible inference from legacy error strings.
    out: List[str] = []
    for err in report.get("errors") or []:
        if not isinstance(err, str):
            continue
        token = err.strip()
        if not token:
            continue
        if token.startswith("unsupported_receipt_schema"):
            out.append("unsupported_receipt_schema")
        elif token.startswith("missing_schema_field"):
            out.append("missing_schema_field")
        elif token.startswith("json_parse_error"):
            out.append("json_parse_error")
        elif token.startswith("io_error") or token.startswith("schema_file_invalid"):
            out.append("io_error")
        elif token.startswith("receipt_not_json_object"):
            out.append("json_parse_error")
        else:
            out.append("invalid_against_schema")
    return sorted(set(out))


def _severity_from_reason_codes(*, ok: bool, reason_codes: List[str], strict: bool) -> str:
    if ok:
        return PASS
    codes = [c for c in reason_codes if isinstance(c, str) and c.strip()]
    if codes and all(code in _WARN_REASON_CODES for code in codes):
        return FAIL if strict else WARN
    return FAIL


def doctor_receipts(
    root_dir: Path,
    *,
    since_min: float,
    strict: bool = False,
    top_k: int = 0,
    summary_only: bool = False,
) -> Dict[str, Any]:
    root = Path(root_dir)
    receipts_dir = root / "artifacts" / "receipts"
    now = time.time()
    cutoff = now - max(0.0, float(since_min)) * 60.0
    rows_all: list[Dict[str, Any]] = []
    if receipts_dir.exists():
        files = sorted(receipts_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            if mtime < cutoff:
                continue
            result = validate_receipt(root, path)
            reason_codes = _normalize_reason_codes(result)
            status = _severity_from_reason_codes(
                ok=bool(result.get("ok")),
                reason_codes=reason_codes,
                strict=bool(strict),
            )
            result["reason_codes"] = reason_codes
            result["status"] = status
            rows_all.append(result)
    pass_count = sum(1 for row in rows_all if str(row.get("status") or "") == PASS)
    warn_count = sum(1 for row in rows_all if str(row.get("status") or "") == WARN)
    fail_count = sum(1 for row in rows_all if str(row.get("status") or "") == FAIL)

    reason_counts: Dict[str, int] = {}
    for row in rows_all:
        for code in row.get("reason_codes") or []:
            key = str(code or "").strip()
            if not key:
                continue
            reason_counts[key] = int(reason_counts.get(key, 0)) + 1

    effective_top_k = max(0, int(top_k or 0))
    rows_view = list(rows_all)
    if summary_only:
        rows_view = []
    elif effective_top_k > 0:
        rows_view = rows_view[:effective_top_k]
    omitted = max(0, len(rows_all) - len(rows_view))

    return {
        "schema": "ajax.doctor.receipts.v0",
        "ts_utc": _utc_now(now),
        "since_min": float(since_min),
        "strict": bool(strict),
        "summary_only": bool(summary_only),
        "top_k": effective_top_k,
        "ok": fail_count == 0,
        "counts": {"total": len(rows_all), "pass": pass_count, "warn": warn_count, "fail": fail_count},
        "reason_counts": reason_counts,
        "legacy_warn_detected": bool(warn_count > 0),
        "omitted": omitted,
        "receipts": rows_view,
    }
