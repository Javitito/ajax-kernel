from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "ajax.doctor.metabolism.v0"


def _iso_utc(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_ts(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            pass
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except Exception:
            return None
    return None


def _is_recent(path: Path, *, now_ts: float, since_seconds: float, payload: Optional[Dict[str, Any]] = None) -> bool:
    ts = None
    if isinstance(payload, dict):
        ts = _parse_ts(payload.get("created_at") or payload.get("created_utc") or payload.get("updated_utc"))
        if ts is None:
            ts = _parse_ts(payload.get("updated_ts") or payload.get("ts") or payload.get("timestamp_utc"))
    if ts is None:
        try:
            ts = float(path.stat().st_mtime)
        except Exception:
            return False
    return (now_ts - float(ts)) <= since_seconds


def _scan_gaps(root_dir: Path, *, now_ts: float, since_seconds: float) -> Dict[str, Any]:
    gap_dir = root_dir / "artifacts" / "capability_gaps"
    missing_efe = 0
    crystallize_failed = 0
    candidate_generated = 0
    candidate_unsupported = 0
    candidate_error = 0

    if not gap_dir.exists():
        return {
            "missing_efe_final": 0,
            "crystallize_failed": 0,
            "efe_candidates": {
                "generated": 0,
                "unsupported": 0,
                "error": 0,
            },
        }

    for path in sorted(gap_dir.glob("*.json")):
        data = _safe_read_json(path)
        if not _is_recent(path, now_ts=now_ts, since_seconds=since_seconds, payload=data):
            continue

        haystack = (json.dumps(data, ensure_ascii=False) + " " + path.name).lower()
        if "missing_efe_final" in haystack:
            missing_efe += 1
        if "crystallize_failed" in haystack:
            crystallize_failed += 1

        status = str(data.get("efe_candidate_status") or "").strip().lower()
        if status == "generated":
            candidate_generated += 1
        elif status == "unsupported":
            candidate_unsupported += 1
        elif status == "error":
            candidate_error += 1

    return {
        "missing_efe_final": missing_efe,
        "crystallize_failed": crystallize_failed,
        "efe_candidates": {
            "generated": candidate_generated,
            "unsupported": candidate_unsupported,
            "error": candidate_error,
        },
    }


def _scan_candidates(root_dir: Path, *, now_ts: float, since_seconds: float) -> Dict[str, int]:
    candidates_dir = root_dir / "artifacts" / "efe_candidates"
    generated = 0
    unsupported = 0

    if not candidates_dir.exists():
        return {"generated": 0, "unsupported": 0}

    for path in sorted(candidates_dir.glob("*.json")):
        data = _safe_read_json(path)
        if not _is_recent(path, now_ts=now_ts, since_seconds=since_seconds, payload=data):
            continue
        ok = bool(data.get("ok"))
        if ok:
            generated += 1
        elif str(data.get("unsupported_action_kind") or "").strip():
            unsupported += 1

    return {"generated": generated, "unsupported": unsupported}


def _scan_provider_and_ladder(root_dir: Path, *, now_ts: float, since_seconds: float) -> Dict[str, Any]:
    ledger_path = root_dir / "artifacts" / "provider_ledger" / "latest.json"
    ledger = _safe_read_json(ledger_path)
    rows = ledger.get("rows") if isinstance(ledger.get("rows"), list) else []
    last_429_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").lower()
        if any(tok in reason for tok in ("429", "quota_exhausted", "429_tpm")):
            last_429_count += 1

    receipt_dir = root_dir / "artifacts" / "receipts"
    ladder_total = 0
    ladder_local = 0
    ladder_blocked = 0
    if receipt_dir.exists():
        for path in sorted(receipt_dir.glob("router_ladder_decision_*.json")):
            data = _safe_read_json(path)
            if not _is_recent(path, now_ts=now_ts, since_seconds=since_seconds, payload=data):
                continue
            ladder_total += 1
            if bool(data.get("local_fallback_used")):
                ladder_local += 1
            if data.get("ok") is False:
                ladder_blocked += 1

    return {
        "last_429_count": last_429_count,
        "ladder_decisions": ladder_total,
        "ladder_local_fallback": ladder_local,
        "ladder_blocked": ladder_blocked,
    }


def _scan_waiting(root_dir: Path) -> Dict[str, Any]:
    waiting_dir = root_dir / "artifacts" / "waiting_for_user"
    if not waiting_dir.exists():
        return {"count": 0, "oldest_age_min": None}

    files = sorted(waiting_dir.glob("*.json"))
    if not files:
        return {"count": 0, "oldest_age_min": None}

    now_ts = time.time()
    oldest_age_s = 0.0
    for path in files:
        try:
            age = max(0.0, now_ts - float(path.stat().st_mtime))
            oldest_age_s = max(oldest_age_s, age)
        except Exception:
            continue
    return {
        "count": len(files),
        "oldest_age_min": round(oldest_age_s / 60.0, 2),
    }


def _build_hints(*, gaps: Dict[str, Any], waiting: Dict[str, Any], provider: Dict[str, Any]) -> List[str]:
    hints: List[str] = []
    if int(gaps.get("missing_efe_final") or 0) > 0:
        hints.append("python bin/ajaxctl verify efe apply-candidate --gap <gap.json> --out artifacts/efe_candidates/efe_final.json")
    if int(waiting.get("count") or 0) > 0:
        hints.append("python bin/ajaxctl ops friction gc --dry-run")
    if int(provider.get("last_429_count") or 0) > 0:
        hints.append("python bin/ajaxctl doctor providers --roles brain --explain")
    if not hints:
        hints.append("python bin/ajaxctl doctor metabolism --since-min 180")
    return hints[:3]


def run_doctor_metabolism(*, root_dir: Path, since_min: float = 180.0) -> Dict[str, Any]:
    root = Path(root_dir)
    now_ts = time.time()
    since_seconds = max(0.0, float(since_min) * 60.0)

    gaps = _scan_gaps(root, now_ts=now_ts, since_seconds=since_seconds)
    candidates = _scan_candidates(root, now_ts=now_ts, since_seconds=since_seconds)
    provider = _scan_provider_and_ladder(root, now_ts=now_ts, since_seconds=since_seconds)
    waiting = _scan_waiting(root)

    # Merge candidate counters from gap metadata + candidate files.
    gap_candidates = gaps.get("efe_candidates") if isinstance(gaps.get("efe_candidates"), dict) else {}
    generated = max(int(gap_candidates.get("generated") or 0), int(candidates.get("generated") or 0))
    unsupported = max(int(gap_candidates.get("unsupported") or 0), int(candidates.get("unsupported") or 0))
    error = int(gap_candidates.get("error") or 0)

    critical_default = 999
    try:
        critical_default = int(os.getenv("AJAX_DOCTOR_METABOLISM_CRITICAL_GAPS", "999") or "999")
    except Exception:
        critical_default = 999
    critical = (int(gaps.get("missing_efe_final") or 0) + int(gaps.get("crystallize_failed") or 0)) >= max(1, critical_default)

    hints = _build_hints(gaps=gaps, waiting=waiting, provider=provider)

    payload = {
        "schema": SCHEMA,
        "created_at": _iso_utc(now_ts),
        "since_min": float(since_min),
        "gaps": {
            "missing_efe_final": int(gaps.get("missing_efe_final") or 0),
            "crystallize_failed": int(gaps.get("crystallize_failed") or 0),
        },
        "efe_candidates": {
            "generated": generated,
            "unsupported": unsupported,
            "error": error,
        },
        "provider": provider,
        "waiting_backlog": waiting,
        "next_hint": hints,
        "critical": bool(critical),
        "exit_code": 1 if critical else 0,
    }
    return payload
