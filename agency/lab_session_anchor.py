from __future__ import annotations

import hashlib
import json
import secrets
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

SESSION_SCHEMA = "ajax.lab.expected_session.v0"
SESSION_INIT_RECEIPT_SCHEMA = "ajax.lab.session.init.v0"
SESSION_STATUS_RECEIPT_SCHEMA = "ajax.lab.session.status.v0"
SESSION_REVOKE_RECEIPT_SCHEMA = "ajax.lab.session.revoke.v0"


def _utc_now(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _ts_label(ts: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts or time.time()))


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _parse_iso_utc(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        from datetime import datetime

        return float(datetime.fromisoformat(text).timestamp())
    except Exception:
        return None


def _normalize_rail(value: Any) -> str:
    raw = str(value or "lab").strip().lower()
    if raw in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def _normalize_display(value: Any) -> str:
    raw = str(value or "dummy").strip().lower()
    if raw == "dummy":
        return "dummy"
    return raw


def _token_hash_prefix(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return digest[:12]


def _host_fingerprint(root_dir: Path) -> str:
    material = f"{socket.gethostname()}::{root_dir.resolve()}"
    return "sha1:" + hashlib.sha1(material.encode("utf-8")).hexdigest()


def current_host_fingerprint(root_dir: Path) -> str:
    return _host_fingerprint(Path(root_dir))


def _session_path(root_dir: Path) -> Path:
    return Path(root_dir) / "artifacts" / "lab" / "session" / "expected_session.json"


def _receipt_path(root_dir: Path, prefix: str, now_ts: float) -> Path:
    return Path(root_dir) / "artifacts" / "receipts" / f"{prefix}_{_ts_label(now_ts)}.json"


def _relpath(root_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(root_dir).resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def validate_expected_session(
    root_dir: Path,
    *,
    now_ts: Optional[float] = None,
    required_rail: str = "lab",
    required_display: str = "dummy",
) -> Dict[str, Any]:
    root = Path(root_dir)
    now = float(now_ts or time.time())
    path = _session_path(root)
    doc = _safe_read_json(path)
    invalid_reasons: list[str] = []
    token_hash_prefix = None
    expires_at = None
    rail = None
    display_target = None

    if doc is None:
        invalid_reasons.append("expected_session_missing")
    else:
        if str(doc.get("schema") or "").strip() != SESSION_SCHEMA:
            invalid_reasons.append("session_schema_invalid")

        token = str(doc.get("token") or "")
        if token:
            token_hash_prefix = _token_hash_prefix(token)
        else:
            invalid_reasons.append("session_token_missing")

        expires_at = doc.get("expires_at")
        expires_ts = _parse_iso_utc(expires_at)
        if expires_ts is None:
            invalid_reasons.append("session_expiry_invalid")
        elif now > float(expires_ts):
            invalid_reasons.append("session_expired")

        rail = _normalize_rail(doc.get("rail"))
        if rail != _normalize_rail(required_rail):
            invalid_reasons.append("session_rail_invalid")

        display_target = _normalize_display(doc.get("display_target"))
        if display_target != _normalize_display(required_display):
            invalid_reasons.append("session_display_invalid")

        expected_fp = _host_fingerprint(root)
        if str(doc.get("host_fingerprint") or "") != expected_fp:
            invalid_reasons.append("session_fingerprint_mismatch")

        if bool(doc.get("revoked")):
            invalid_reasons.append("session_revoked")

    ok = len(invalid_reasons) == 0
    return {
        "schema": "ajax.lab.session_status.v0",
        "ts_utc": _utc_now(now),
        "ok": ok,
        "reason": "session_valid" if ok else invalid_reasons[0],
        "exists": path.exists(),
        "path": str(path),
        "rail": rail,
        "display_target": display_target,
        "expires_at": expires_at,
        "token_hash_prefix": token_hash_prefix,
        "invalid_reasons": invalid_reasons,
    }


def init_expected_session(
    root_dir: Path,
    *,
    ttl_min: int,
    display: str,
    rail: str,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    now = float(now_ts or time.time())
    ttl = max(1, int(ttl_min))
    display_n = _normalize_display(display)
    rail_n = _normalize_rail(rail)
    created_at = _utc_now(now)
    expires_at = _utc_now(now + float(ttl) * 60.0)
    token = secrets.token_urlsafe(24)
    token_hash_prefix = _token_hash_prefix(token)
    session_doc = {
        "schema": SESSION_SCHEMA,
        "created_at": created_at,
        "expires_at": expires_at,
        "rail": rail_n,
        "display_target": display_n,
        "consent_ttl": {"minutes": ttl},
        "token": token,
        "host_fingerprint": _host_fingerprint(root),
    }
    session_path = _session_path(root)
    _safe_write_json(session_path, session_doc)
    validation = validate_expected_session(root, now_ts=now)
    receipt = {
        "schema": SESSION_INIT_RECEIPT_SCHEMA,
        "ts_utc": _utc_now(now),
        "ok": bool(validation.get("ok")),
        "session_path": _relpath(root, session_path),
        "created_at": created_at,
        "expires_at": expires_at,
        "rail": rail_n,
        "display_target": display_n,
        "token_hash_prefix": token_hash_prefix,
        "validation": validation,
    }
    receipt_path = _receipt_path(root, "lab_session_init", now)
    _safe_write_json(receipt_path, receipt)
    receipt["receipt_path"] = _relpath(root, receipt_path)
    _safe_write_json(receipt_path, receipt)
    return receipt


def session_status(
    root_dir: Path,
    *,
    now_ts: Optional[float] = None,
    write_receipt: bool = False,
) -> Dict[str, Any]:
    root = Path(root_dir)
    now = float(now_ts or time.time())
    status = validate_expected_session(root, now_ts=now)
    payload = {
        "schema": SESSION_STATUS_RECEIPT_SCHEMA,
        "ts_utc": _utc_now(now),
        "ok": bool(status.get("ok")),
        "status": status,
    }
    if not write_receipt:
        return payload
    receipt_path = _receipt_path(root, "lab_session_status", now)
    _safe_write_json(receipt_path, payload)
    payload["receipt_path"] = _relpath(root, receipt_path)
    _safe_write_json(receipt_path, payload)
    return payload


def revoke_expected_session(root_dir: Path, *, now_ts: Optional[float] = None) -> Dict[str, Any]:
    root = Path(root_dir)
    now = float(now_ts or time.time())
    path = _session_path(root)
    existed = path.exists()
    removed = False
    if existed:
        try:
            path.unlink()
            removed = True
        except Exception:
            removed = False
    status = validate_expected_session(root, now_ts=now)
    payload = {
        "schema": SESSION_REVOKE_RECEIPT_SCHEMA,
        "ts_utc": _utc_now(now),
        "ok": True,
        "existed": existed,
        "removed": removed,
        "session_path": _relpath(root, path),
        "status_after": status,
    }
    receipt_path = _receipt_path(root, "lab_session_revoke", now)
    _safe_write_json(receipt_path, payload)
    payload["receipt_path"] = _relpath(root, receipt_path)
    _safe_write_json(receipt_path, payload)
    return payload
