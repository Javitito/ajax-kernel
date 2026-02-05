from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_FLAG_REL_PATH = Path("artifacts") / "governance" / "human_permission.flag"


def _parse_utc(value: str) -> Optional[datetime]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except Exception:
        return None


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def flag_path(root_dir: Path) -> Path:
    return (root_dir / _FLAG_REL_PATH).resolve()


def human_permission_gate_enabled() -> bool:
    """
    Pixel-safety gate (sin HDMI dummy):
    - enabled by default (fail-closed)
    - disable explicitly by setting AJAX_HDMI_DUMMY_PRESENT=1 (or true/yes/on)
    """
    raw = (os.getenv("AJAX_HDMI_DUMMY_PRESENT") or "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def read_human_permission_status(root_dir: Path, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    path = flag_path(root_dir)
    if now is None:
        now = datetime.now(timezone.utc)
    if not path.exists():
        return {"ok": False, "expires_utc": None, "path": str(path)}
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return {"ok": False, "expires_utc": None, "path": str(path), "error": "read_failed"}
    expires = _parse_utc(raw)
    if not expires:
        return {"ok": False, "expires_utc": None, "path": str(path), "error": "invalid_expires_utc"}
    return {"ok": now < expires, "expires_utc": _format_utc(expires), "path": str(path)}


def grant_human_permission(root_dir: Path, ttl_seconds: int) -> Dict[str, Any]:
    try:
        ttl_seconds = int(ttl_seconds)
    except Exception:
        ttl_seconds = 0
    ttl_seconds = max(1, min(ttl_seconds, 3600))
    path = flag_path(root_dir)
    expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_format_utc(expires) + "\n", encoding="utf-8")
    except Exception:
        return {"ok": False, "expires_utc": None, "path": str(path), "error": "write_failed"}
    return {"ok": True, "expires_utc": _format_utc(expires), "path": str(path)}


def revoke_human_permission(root_dir: Path) -> Dict[str, Any]:
    path = flag_path(root_dir)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        return {"ok": False, "path": str(path), "error": "unlink_failed"}
    return {"ok": True, "path": str(path)}
