from __future__ import annotations

import json
from pathlib import Path

import agency.lab_autopilot as autopilot_mod
import agency.lab_session_anchor as session_mod
from agency.lab_autopilot import AutopilotTickOptions, run_autopilot_tick


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_session_init_creates_expected_session_file(tmp_path: Path) -> None:
    payload = session_mod.init_expected_session(
        tmp_path,
        ttl_min=120,
        display="dummy",
        rail="lab",
        now_ts=1_800_001_000.0,
    )
    session_path = tmp_path / "artifacts" / "lab" / "session" / "expected_session.json"
    assert payload.get("ok") is True
    assert session_path.exists()
    doc = _read_json(session_path)
    assert doc.get("schema") == "ajax.lab.expected_session.v0"
    assert doc.get("rail") == "lab"
    assert doc.get("display_target") == "dummy"
    assert isinstance(doc.get("token"), str) and doc.get("token")


def test_session_status_reports_valid(tmp_path: Path) -> None:
    session_mod.init_expected_session(
        tmp_path,
        ttl_min=120,
        display="dummy",
        rail="lab",
        now_ts=1_800_001_100.0,
    )
    payload = session_mod.session_status(tmp_path, now_ts=1_800_001_200.0)
    status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    assert status.get("ok") is True
    assert status.get("exists") is True
    assert status.get("rail") == "lab"
    assert status.get("display_target") == "dummy"
    assert isinstance(status.get("token_hash_prefix"), str)
    assert status.get("invalid_reasons") == []


def test_session_revoke_removes_or_marks_revoked(tmp_path: Path) -> None:
    session_mod.init_expected_session(
        tmp_path,
        ttl_min=120,
        display="dummy",
        rail="lab",
        now_ts=1_800_001_300.0,
    )
    payload = session_mod.revoke_expected_session(tmp_path, now_ts=1_800_001_310.0)
    session_path = tmp_path / "artifacts" / "lab" / "session" / "expected_session.json"
    assert payload.get("ok") is True
    assert payload.get("existed") is True
    assert payload.get("removed") is True
    assert session_path.exists() is False


def test_session_expired_is_invalid(tmp_path: Path) -> None:
    session_mod.init_expected_session(
        tmp_path,
        ttl_min=1,
        display="dummy",
        rail="lab",
        now_ts=1_800_001_400.0,
    )
    status = session_mod.validate_expected_session(tmp_path, now_ts=1_800_001_600.0)
    reasons = status.get("invalid_reasons") if isinstance(status.get("invalid_reasons"), list) else []
    assert status.get("ok") is False
    assert "session_expired" in reasons


def test_session_fingerprint_mismatch_invalid(tmp_path: Path) -> None:
    session_mod.init_expected_session(
        tmp_path,
        ttl_min=120,
        display="dummy",
        rail="lab",
        now_ts=1_800_001_700.0,
    )
    session_path = tmp_path / "artifacts" / "lab" / "session" / "expected_session.json"
    doc = _read_json(session_path)
    doc["host_fingerprint"] = "sha1:foreign-host"
    _write_json(session_path, doc)
    status = session_mod.validate_expected_session(tmp_path, now_ts=1_800_001_710.0)
    reasons = status.get("invalid_reasons") if isinstance(status.get("invalid_reasons"), list) else []
    assert status.get("ok") is False
    assert "session_fingerprint_mismatch" in reasons


def test_autopilot_gate_env_safe_passes_when_session_valid(monkeypatch, tmp_path: Path) -> None:
    session_mod.init_expected_session(
        tmp_path,
        ttl_min=120,
        display="dummy",
        rail="lab",
        now_ts=1_800_001_800.0,
    )
    anchor_receipt = tmp_path / "artifacts" / "receipts" / "anchor_preflight_fake.json"
    _write_json(anchor_receipt, {"schema": "ajax.anchor_preflight.v1"})

    def _fake_anchor(root_dir: Path, rail: str) -> dict:
        return {
            "ok": False,
            "rail": rail,
            "receipt_path": str(anchor_receipt),
            "observed": {"display_target_is_dummy": True},
            "mismatches": [{"code": "expected_session_missing", "detail": "legacy session source unavailable"}],
        }

    monkeypatch.setattr(autopilot_mod, "_run_anchor_preflight", _fake_anchor, raising=True)
    options = AutopilotTickOptions(
        mode="dry-run",
        interactive=False,
        allow_filesystem_basic=False,
        allow_queue_housekeeping=False,
    )
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_001_900.0)
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    env_gate = gates.get("env_safe") if isinstance(gates.get("env_safe"), dict) else {}
    assert payload.get("action") == "NOOP"
    assert env_gate.get("ok") is True
    assert env_gate.get("reason") == "lab_anchor_ok"

