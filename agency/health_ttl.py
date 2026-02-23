from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_HEALTH_TTL_SECONDS = 900


def _parse_iso_utc(raw: Any) -> Optional[float]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def provider_status_ttl(
    root_dir: Path,
    *,
    ttl_seconds: int = DEFAULT_HEALTH_TTL_SECONDS,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    ttl = max(30, int(ttl_seconds or DEFAULT_HEALTH_TTL_SECONDS))
    status_path = Path(root_dir) / "artifacts" / "health" / "providers_status.json"

    info: Dict[str, Any] = {
        "schema": "ajax.health_ttl.v1",
        "ttl_seconds": ttl,
        "status_path": str(status_path),
        "exists": status_path.exists(),
        "updated_ts": None,
        "updated_utc": None,
        "age_seconds": None,
        "stale": True,
        "reason": "providers_status_missing",
    }

    if not status_path.exists():
        return info

    updated_ts: Optional[float] = None
    updated_utc: Optional[str] = None
    doc = _safe_read_json(status_path)
    if doc:
        raw_ts = doc.get("updated_ts")
        try:
            if raw_ts is not None:
                updated_ts = float(raw_ts)
        except Exception:
            updated_ts = None
        raw_utc = doc.get("updated_utc")
        if isinstance(raw_utc, str) and raw_utc.strip():
            updated_utc = raw_utc.strip()
            if updated_ts is None:
                updated_ts = _parse_iso_utc(raw_utc)

    if updated_ts is None:
        try:
            updated_ts = float(status_path.stat().st_mtime)
            updated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(updated_ts))
        except Exception:
            updated_ts = None

    info["updated_ts"] = updated_ts
    info["updated_utc"] = updated_utc

    if updated_ts is None:
        info["stale"] = True
        info["reason"] = "providers_status_no_timestamp"
        return info

    age_seconds = max(0.0, now - updated_ts)
    stale = age_seconds > float(ttl)
    info["age_seconds"] = round(age_seconds, 3)
    info["stale"] = stale
    info["reason"] = "providers_status_stale" if stale else "providers_status_fresh"
    return info
