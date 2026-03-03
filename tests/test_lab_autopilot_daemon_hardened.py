from __future__ import annotations

import json
from pathlib import Path

import pytest

import agency.lab_autopilot as autopilot_mod
from agency.lab_autopilot import AutopilotTickOptions, run_autopilot_daemon


def _fake_tick_factory(action: str = "EXECUTED"):
    calls = {"n": 0}

    def _fake_tick(root_dir: Path, *, options: AutopilotTickOptions, now_ts: float | None = None) -> dict:
        calls["n"] += 1
        return {
            "schema": "ajax.lab.autopilot_tick.v1",
            "action": action,
            "tick_receipt_path": f"artifacts/receipts/fake_tick_{calls['n']}.json",
        }

    return calls, _fake_tick


def test_daemon_runs_max_ticks_and_exits(monkeypatch, tmp_path: Path) -> None:
    calls, fake_tick = _fake_tick_factory(action="EXECUTED")
    monkeypatch.setattr(autopilot_mod, "run_autopilot_tick", fake_tick, raising=True)
    monkeypatch.setattr(autopilot_mod.time, "sleep", lambda _s: None, raising=True)
    payload = run_autopilot_daemon(
        tmp_path,
        options=AutopilotTickOptions(mode="daemon", interactive=False),
        interval_s=0.01,
        max_ticks=3,
    )
    assert payload.get("ok") is True
    assert payload.get("ticks") == 3
    assert calls["n"] == 3
    assert payload.get("stop_reason") == "max_ticks_reached"


def test_daemon_respects_stopfile(monkeypatch, tmp_path: Path) -> None:
    stopfile = tmp_path / "artifacts" / "lab" / "STOP_AUTOPILOT"
    stopfile.parent.mkdir(parents=True, exist_ok=True)
    stopfile.write_text("1\n", encoding="utf-8")

    def _unexpected_tick(*args, **kwargs):  # pragma: no cover - safety
        raise AssertionError("run_autopilot_tick should not run when stopfile exists")

    monkeypatch.setattr(autopilot_mod, "run_autopilot_tick", _unexpected_tick, raising=True)
    payload = run_autopilot_daemon(
        tmp_path,
        options=AutopilotTickOptions(mode="daemon", interactive=False),
        interval_s=0.01,
        max_ticks=3,
    )
    receipt_path = tmp_path / str(payload.get("daemon_receipt_path"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload.get("ticks") == 0
    assert payload.get("stop_reason") == "stopped_by_stopfile"
    assert receipt.get("stop_reason") == "stopped_by_stopfile"


def test_daemon_writes_heartbeat(monkeypatch, tmp_path: Path) -> None:
    _, fake_tick = _fake_tick_factory(action="EXECUTED")
    monkeypatch.setattr(autopilot_mod, "run_autopilot_tick", fake_tick, raising=True)
    monkeypatch.setattr(autopilot_mod.time, "sleep", lambda _s: None, raising=True)
    payload = run_autopilot_daemon(
        tmp_path,
        options=AutopilotTickOptions(mode="daemon", interactive=False),
        interval_s=0.01,
        max_ticks=1,
    )
    heartbeat_path = tmp_path / str(payload.get("heartbeat_path"))
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat.get("status") == "STOPPED"
    assert heartbeat.get("ticks") == 1


def test_daemon_no_session_blocks_but_exits_cleanly(monkeypatch, tmp_path: Path) -> None:
    _, fake_tick = _fake_tick_factory(action="BLOCKED")
    monkeypatch.setattr(autopilot_mod, "run_autopilot_tick", fake_tick, raising=True)
    monkeypatch.setattr(autopilot_mod.time, "sleep", lambda _s: None, raising=True)
    payload = run_autopilot_daemon(
        tmp_path,
        options=AutopilotTickOptions(mode="daemon", interactive=False),
        interval_s=0.01,
        max_ticks=2,
    )
    assert payload.get("ok") is True
    assert payload.get("ticks") == 2
    assert payload.get("last_action") == "BLOCKED"
    assert payload.get("stop_reason") == "max_ticks_reached"


def test_daemon_interval_respected_mock_sleep(monkeypatch, tmp_path: Path) -> None:
    _, fake_tick = _fake_tick_factory(action="EXECUTED")
    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(float(seconds))

    monkeypatch.setattr(autopilot_mod, "run_autopilot_tick", fake_tick, raising=True)
    monkeypatch.setattr(autopilot_mod.time, "sleep", _fake_sleep, raising=True)
    run_autopilot_daemon(
        tmp_path,
        options=AutopilotTickOptions(mode="daemon", interactive=False),
        interval_s=1.5,
        max_ticks=2,
    )
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(1.5)

