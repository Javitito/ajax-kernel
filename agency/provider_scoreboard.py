from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCOREBOARD_SCHEMA = "ajax.provider_scoreboard.v1"


def _iso_utc(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def load_scoreboard(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def score_for(scoreboard: Dict[str, Any], *, provider: str, model: Optional[str]) -> Optional[float]:
    if not isinstance(scoreboard, dict):
        return None
    providers = scoreboard.get("providers")
    if not isinstance(providers, dict):
        return None
    key = f"{provider}:{model}" if model else provider
    entry = providers.get(key)
    if not isinstance(entry, dict):
        return None
    score = entry.get("score")
    try:
        return float(score)
    except Exception:
        return None


def promotion_state(
    scoreboard: Dict[str, Any],
    *,
    provider: str,
    model: Optional[str],
    min_samples: int,
    cooldown_minutes: int,
) -> Dict[str, Any]:
    providers = scoreboard.get("providers") if isinstance(scoreboard, dict) else None
    if not isinstance(providers, dict):
        return {"eligible": None, "reorder_allowed": False, "reason": "scoreboard_missing"}
    key = f"{provider}:{model}" if model else provider
    entry = providers.get(key)
    if not isinstance(entry, dict):
        return {"eligible": None, "reorder_allowed": False, "reason": "scoreboard_missing"}
    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []
    eligible = entry.get("eligible")
    if eligible is False:
        return {"eligible": False, "reorder_allowed": False, "reason": entry.get("gate_fail_reason") or "ineligible"}
    if len(samples) < max(1, int(min_samples or 1)):
        return {"eligible": True, "reorder_allowed": False, "reason": "min_samples"}
    cooldown_ts = entry.get("cooldown_until_ts")
    try:
        if isinstance(cooldown_ts, (int, float)) and cooldown_ts > time.time():
            return {"eligible": True, "reorder_allowed": False, "reason": "cooldown"}
    except Exception:
        pass
    return {"eligible": True, "reorder_allowed": True, "reason": "ok"}


def update_scoreboard(
    *,
    path: Path,
    entries: List[Dict[str, Any]],
    window: int,
) -> Dict[str, Any]:
    scoreboard = load_scoreboard(path)
    providers = scoreboard.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    window = max(1, int(window or 1))
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = entry.get("model")
        if not provider:
            continue
        key = f"{provider}:{model}" if model else provider
        bucket = providers.get(key)
        if not isinstance(bucket, dict):
            bucket = {"samples": [], "score": None, "updated_utc": None}
        samples = bucket.get("samples")
        if not isinstance(samples, list):
            samples = []
        eligible = entry.get("eligible")
        gate_fail_reason = entry.get("gate_fail_reason")
        samples.append(
            {
                "ts_utc": _iso_utc(),
                "score": entry.get("score"),
                "reliability": entry.get("reliability"),
                "obedience": entry.get("obedience"),
                "latency_p50_ms": entry.get("latency_p50_ms"),
                "success_rate": entry.get("success_rate"),
                "json_parse_rate": entry.get("json_parse_rate"),
                "gates_passed": entry.get("gates_passed"),
                "eligible": eligible,
                "gate_fail_reason": gate_fail_reason,
            }
        )
        samples = samples[-window:]
        score_vals = [s.get("score") for s in samples if isinstance(s, dict) and isinstance(s.get("score"), (int, float))]
        bucket["samples"] = samples
        bucket["score"] = float(sum(score_vals) / len(score_vals)) if score_vals else None
        bucket["updated_utc"] = _iso_utc()
        if eligible is False:
            bucket["eligible"] = False
            bucket["gate_fail_reason"] = gate_fail_reason or "ineligible"
            bucket["last_gate_ts"] = _iso_utc()
        elif eligible is True:
            bucket["eligible"] = True
            bucket["gate_fail_reason"] = None
        cooldown_minutes = int(os.getenv("AJAX_SCOREBOARD_COOLDOWN_MIN", "15") or 15)
        if eligible is False:
            bucket["cooldown_until_ts"] = time.time() + (cooldown_minutes * 60)
            bucket["cooldown_until_utc"] = _iso_utc(bucket["cooldown_until_ts"])
        else:
            try:
                cooldown_ts = bucket.get("cooldown_until_ts")
                if isinstance(cooldown_ts, (int, float)) and cooldown_ts <= time.time():
                    bucket["cooldown_until_ts"] = None
                    bucket["cooldown_until_utc"] = None
            except Exception:
                pass
        providers[key] = bucket
    scoreboard["schema"] = SCOREBOARD_SCHEMA
    scoreboard["window"] = window
    scoreboard["min_samples"] = int(os.getenv("AJAX_SCOREBOARD_MIN_SAMPLES", "3") or 3)
    scoreboard["cooldown_minutes"] = int(os.getenv("AJAX_SCOREBOARD_COOLDOWN_MIN", "15") or 15)
    scoreboard["updated_utc"] = _iso_utc()
    scoreboard["providers"] = providers
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scoreboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    return scoreboard
