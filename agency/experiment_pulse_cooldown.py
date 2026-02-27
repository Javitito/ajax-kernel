from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict


def _to_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def load_state(path: Path) -> Dict[str, int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "cooldown_until_ts": _to_int(data.get("cooldown_until_ts"), 0),
        "skip_streak": max(0, _to_int(data.get("skip_streak"), 0)),
        "next_wake_ts": _to_int(data.get("next_wake_ts"), 0),
    }


def save_state(path: Path, state: Dict[str, int]) -> None:
    payload = {
        "cooldown_until_ts": _to_int(state.get("cooldown_until_ts"), 0),
        "skip_streak": max(0, _to_int(state.get("skip_streak"), 0)),
        "next_wake_ts": _to_int(state.get("next_wake_ts"), 0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decide_pulse_action(
    *,
    now_ts: int,
    state: Dict[str, int],
    cooldown_sec: int = 600,
    mode: str = "backoff_jitter",
    base_backoff_sec: int = 30,
    max_backoff_sec: int = 300,
    jitter_ratio: float = 0.2,
    rng: random.Random | None = None,
) -> Dict[str, Any]:
    """
    Decide if experiment_pulse should run now or skip/sleep.

    - `mode=until_cooldown`: sleep exactly remaining cooldown.
    - `mode=backoff_jitter`: exponential backoff + jitter, capped by cooldown end.
    """
    now_ts = _to_int(now_ts, 0)
    cooldown_until_ts = _to_int(state.get("cooldown_until_ts"), 0)
    skip_streak = max(0, _to_int(state.get("skip_streak"), 0))
    next_wake_ts = _to_int(state.get("next_wake_ts"), 0)
    cooldown_sec = max(1, _to_int(cooldown_sec, 600))

    # No active cooldown: run once and open a new cooldown window.
    if now_ts >= cooldown_until_ts:
        next_state = {
            "cooldown_until_ts": now_ts + cooldown_sec,
            "skip_streak": 0,
            "next_wake_ts": now_ts,
        }
        return {"action": "run", "sleep_sec": 0, "remaining_sec": 0, "state": next_state}

    remaining_sec = max(1, cooldown_until_ts - now_ts)
    mode_n = str(mode or "backoff_jitter").strip().lower()
    if mode_n == "until_cooldown":
        sleep_sec = remaining_sec
        next_wake = cooldown_until_ts
        next_state = {
            "cooldown_until_ts": cooldown_until_ts,
            "skip_streak": skip_streak + 1,
            "next_wake_ts": next_wake,
        }
        return {"action": "skip_cooldown", "sleep_sec": sleep_sec, "remaining_sec": remaining_sec, "state": next_state}

    # Backoff mode: if we already scheduled a wake-up in the future, respect it.
    if next_wake_ts > now_ts:
        sleep_sec = max(1, next_wake_ts - now_ts)
        next_state = {
            "cooldown_until_ts": cooldown_until_ts,
            "skip_streak": skip_streak,
            "next_wake_ts": next_wake_ts,
        }
        return {"action": "skip_cooldown", "sleep_sec": sleep_sec, "remaining_sec": remaining_sec, "state": next_state}

    base_backoff_sec = max(1, _to_int(base_backoff_sec, 30))
    max_backoff_sec = max(base_backoff_sec, _to_int(max_backoff_sec, 300))
    backoff_sec = min(base_backoff_sec * (2**skip_streak), max_backoff_sec, remaining_sec)
    rng = rng or random.Random()
    jitter_max = max(0, int(backoff_sec * max(0.0, float(jitter_ratio))))
    jitter = rng.randint(0, jitter_max) if jitter_max > 0 else 0
    sleep_sec = min(remaining_sec, max(1, backoff_sec + jitter))
    next_wake = now_ts + sleep_sec
    next_state = {
        "cooldown_until_ts": cooldown_until_ts,
        "skip_streak": skip_streak + 1,
        "next_wake_ts": next_wake,
    }
    return {"action": "skip_cooldown", "sleep_sec": sleep_sec, "remaining_sec": remaining_sec, "state": next_state}
