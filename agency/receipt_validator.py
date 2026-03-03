from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _schema_path_for_receipt(root_dir: Path, receipt_schema: str) -> Optional[Path]:
    schemas_dir = Path(root_dir) / "schemas" / "receipts"
    if receipt_schema.startswith("ajax.lab.session.") or receipt_schema == "ajax.lab.session_status.v0":
        return schemas_dir / "ajax.lab.session.v0.schema.json"
    if receipt_schema in {"ajax.lab.autopilot_tick.v1", "ajax.lab.autopilot_tick.v0"}:
        return schemas_dir / "ajax.lab.autopilot_tick.v1.schema.json"
    if receipt_schema in {"ajax.topology_doctor.v0", "ajax.topology_doctor.v1"}:
        return schemas_dir / "ajax.topology_doctor.v0.schema.json"
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
    payload = _safe_read_json(path)
    if payload is None:
        return {
            "ok": False,
            "path": str(path),
            "errors": ["receipt_not_json_object"],
            "schema_in_receipt": None,
            "schema_path": None,
            "schema_used": None,
        }
    schema_in_receipt = str(payload.get("schema") or "")
    schema_path = _schema_path_for_receipt(root, schema_in_receipt)
    if schema_path is None:
        return {
            "ok": False,
            "path": str(path),
            "errors": [f"unsupported_receipt_schema:{schema_in_receipt or 'missing'}"],
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
            "schema_in_receipt": schema_in_receipt or None,
            "schema_path": str(schema_path),
            "schema_used": schema_path.name,
        }
    errors = _validate_node(payload, schema_doc, "$")
    return {
        "ok": len(errors) == 0,
        "path": str(path),
        "errors": errors,
        "schema_in_receipt": schema_in_receipt or None,
        "schema_path": str(schema_path),
        "schema_used": schema_path.name,
    }


def doctor_receipts(root_dir: Path, *, since_min: float) -> Dict[str, Any]:
    root = Path(root_dir)
    receipts_dir = root / "artifacts" / "receipts"
    now = time.time()
    cutoff = now - max(0.0, float(since_min)) * 60.0
    rows: list[Dict[str, Any]] = []
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
            result["status"] = "PASS" if bool(result.get("ok")) else "FAIL"
            rows.append(result)
    pass_count = sum(1 for row in rows if bool(row.get("ok")))
    fail_count = len(rows) - pass_count
    return {
        "schema": "ajax.doctor.receipts.v0",
        "ts_utc": _utc_now(now),
        "since_min": float(since_min),
        "ok": fail_count == 0,
        "counts": {"total": len(rows), "pass": pass_count, "fail": fail_count},
        "receipts": rows,
    }
