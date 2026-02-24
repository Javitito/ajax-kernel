from pathlib import Path

import agency.explore_policy as explore_policy
from agency.explore_policy import (
    compute_human_active,
    evaluate_explore_state,
    human_signal_failure_mode,
    unknown_signal_as_human_policy,
)


def test_compute_human_active_ignores_fields_when_signal_not_ok() -> None:
    active, reason = compute_human_active(
        {
            "ok": False,
            "last_input_age_sec": 0,
            "session_unlocked": True,
            "error": "stub_fail_closed",
        },
        threshold_s=90,
        unknown_as_human=False,
    )
    assert active is False
    assert reason == "signal_not_ok"


def test_compute_human_active_respects_policy_when_signal_not_ok() -> None:
    active, reason = compute_human_active(
        {"ok": False, "last_input_age_sec": 0, "session_unlocked": True},
        threshold_s=90,
        unknown_as_human=True,
    )
    assert active is True
    assert reason == "signal_not_ok"


def _write_stub_signal_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "scripts" / "ops" / "get_human_signal.ps1"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        (
            "$ErrorActionPreference = 'Stop'\n"
            "function Emit-Active([string]$Err) {\n"
            "  $payload = @{ schema='ajax.human_signal.v1'; ok=$false; last_input_age_sec=0; session_unlocked=$true; error=$Err }\n"
            "  $payload | ConvertTo-Json -Compress\n"
            "}\n"
            "Emit-Active 'stub_fail_closed'\n"
        ),
        encoding="utf-8",
    )
    return script_path


def test_evaluate_explore_state_stub_script_strict_blocks(monkeypatch, tmp_path: Path) -> None:
    _write_stub_signal_script(tmp_path)
    cfg = {
        "policy": {"human_active_threshold_s": 90, "unknown_signal_as_human": False},
        "human_signal": {
            "ps_script": "scripts/ops/get_human_signal.ps1",
            "failure_mode": "strict",
            "timeout_s": 1.0,
        },
    }

    def _should_not_run(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("subprocess.run should not be called for stub-detected script")

    monkeypatch.setattr(explore_policy.subprocess, "run", _should_not_run, raising=True)
    state = evaluate_explore_state(tmp_path, policy=cfg)

    assert state["state"] == "HUMAN_DETECTED"
    assert state["human_active"] is True
    assert state["human_active_reason"] == "signal_not_ok"
    assert state["human_signal_failure_mode"] == "strict"
    assert state["human_signal"]["failure_mode"] == "strict"
    assert state["human_signal"]["trusted"] is False
    assert state["human_signal"]["reliability_reason"] == "stub_script_detected"
    assert state["human_signal"]["probe"]["stub_detected"] is True


def test_evaluate_explore_state_stub_script_relaxed_allows_away(monkeypatch, tmp_path: Path) -> None:
    _write_stub_signal_script(tmp_path)
    cfg = {
        "policy": {"human_active_threshold_s": 90, "unknown_signal_as_human": True},
        "human_signal": {
            "ps_script": "scripts/ops/get_human_signal.ps1",
            "failure_mode": "relaxed",
            "timeout_s": 1.0,
        },
    }

    def _should_not_run(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("subprocess.run should not be called for stub-detected script")

    monkeypatch.setattr(explore_policy.subprocess, "run", _should_not_run, raising=True)
    state = evaluate_explore_state(tmp_path, policy=cfg)

    assert state["state"] == "AWAY"
    assert state["human_active"] is False
    assert state["human_active_reason"] == "signal_not_ok"
    assert state["human_signal_failure_mode"] == "relaxed"
    assert state["human_signal"]["failure_mode"] == "relaxed"
    assert state["human_signal"]["reliability_reason"] == "stub_script_detected"


def test_human_signal_failure_mode_legacy_compat() -> None:
    cfg_relaxed = {"policy": {"unknown_signal_as_human": False}}
    cfg_strict = {"policy": {"unknown_signal_as_human": True}}
    assert human_signal_failure_mode(cfg_relaxed) == "relaxed"
    assert human_signal_failure_mode(cfg_strict) == "strict"
    assert unknown_signal_as_human_policy(cfg_relaxed) is False
    assert unknown_signal_as_human_policy(cfg_strict) is True


def test_background_lab_defaults_to_strict_when_mode_not_explicit(monkeypatch, tmp_path: Path) -> None:
    _write_stub_signal_script(tmp_path)
    lab_dir = tmp_path / "artifacts" / "lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "worker.pid").write_text("1234\n", encoding="utf-8")
    cfg = {
        "policy": {"human_active_threshold_s": 90, "unknown_signal_as_human": False},
        "human_signal": {
            "ps_script": "scripts/ops/get_human_signal.ps1",
            "timeout_s": 1.0,
        },
    }

    def _should_not_run(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("subprocess.run should not be called for stub-detected script")

    monkeypatch.setattr(explore_policy.subprocess, "run", _should_not_run, raising=True)
    state = evaluate_explore_state(tmp_path, policy=cfg)

    assert state["state"] == "HUMAN_DETECTED"
    assert state["human_signal_failure_mode"] == "strict"
    assert state["human_signal_failure_mode_source"] == "background_default"
    assert state["human_signal_failure_mode_reason"] == "lab_background_active"
    assert state["human_signal"]["background_lab_active"] is True
    assert state["human_signal"]["failure_mode"] == "strict"


def test_background_default_does_not_override_explicit_relaxed(monkeypatch, tmp_path: Path) -> None:
    _write_stub_signal_script(tmp_path)
    lab_dir = tmp_path / "artifacts" / "lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "heartbeat.json").write_text('{"status":"RUNNING"}\n', encoding="utf-8")
    cfg = {
        "policy": {"human_active_threshold_s": 90, "unknown_signal_as_human": True},
        "human_signal": {
            "ps_script": "scripts/ops/get_human_signal.ps1",
            "failure_mode": "relaxed",
            "timeout_s": 1.0,
        },
    }

    def _should_not_run(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("subprocess.run should not be called for stub-detected script")

    monkeypatch.setattr(explore_policy.subprocess, "run", _should_not_run, raising=True)
    state = evaluate_explore_state(tmp_path, policy=cfg)

    assert state["state"] == "AWAY"
    assert state["human_signal_failure_mode"] == "relaxed"
    assert state["human_signal_failure_mode_source"] == "explicit_config"
    assert state["human_signal"]["background_lab_active"] is True
