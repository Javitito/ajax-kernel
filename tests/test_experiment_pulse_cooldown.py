from __future__ import annotations

import random
from pathlib import Path

from agency.experiment_pulse_cooldown import decide_pulse_action, load_state, save_state


def test_until_cooldown_mode_sleeps_to_cooldown_end() -> None:
    out = decide_pulse_action(
        now_ts=1_000,
        state={"cooldown_until_ts": 1_120, "skip_streak": 0, "next_wake_ts": 0},
        cooldown_sec=600,
        mode="until_cooldown",
    )
    assert out["action"] == "skip_cooldown"
    assert out["remaining_sec"] == 120
    assert out["sleep_sec"] == 120
    assert out["state"]["next_wake_ts"] == 1_120


def test_backoff_mode_limits_skip_churn_for_ten_invocations() -> None:
    rng = random.Random(7)
    state = {"cooldown_until_ts": 0, "skip_streak": 0, "next_wake_ts": 0}
    now_ts = 10_000
    run_count = 0
    skip_count = 0

    # Simula 10 iteraciones del scheduler obedeciendo sleep recomendado.
    for _ in range(10):
        out = decide_pulse_action(
            now_ts=now_ts,
            state=state,
            cooldown_sec=3600,
            mode="backoff_jitter",
            base_backoff_sec=30,
            max_backoff_sec=300,
            jitter_ratio=0.0,
            rng=rng,
        )
        state = out["state"]
        if out["action"] == "run":
            run_count += 1
            now_ts += 1
        else:
            skip_count += 1
            now_ts += int(out["sleep_sec"])

    assert run_count == 1
    assert skip_count == 9
    # El scheduler avanza con backoff; no martillea con cientos de skips cortos.
    assert now_ts - 10_000 >= 1_800


def test_state_roundtrip(tmp_path: Path) -> None:
    state_path = tmp_path / "cooldown_state.json"
    original = {"cooldown_until_ts": 555, "skip_streak": 2, "next_wake_ts": 444}
    save_state(state_path, original)
    loaded = load_state(state_path)
    assert loaded == original
