from __future__ import annotations

import json
from pathlib import Path

import agency.lab_autopilot as autopilot_mod
import agency.lab_session_anchor as session_mod
from agency.lab_autopilot import AutopilotTickOptions, run_autopilot_tick


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _patch_anchor_ok(monkeypatch, tmp_path: Path) -> None:
    anchor_receipt = tmp_path / "artifacts" / "receipts" / "anchor_preflight_fake.json"
    _write_json(anchor_receipt, {"ok": True})

    def _fake_anchor(root_dir: Path, rail: str) -> dict:
        return {
            "ok": True,
            "rail": rail,
            "receipt_path": str(anchor_receipt),
            "observed": {"display_target_is_dummy": True},
            "mismatches": [],
        }

    monkeypatch.setattr(autopilot_mod, "_run_anchor_preflight", _fake_anchor, raising=True)


def _valid_session(tmp_path: Path, now_ts: float) -> None:
    session_mod.init_expected_session(
        tmp_path,
        ttl_min=120,
        display="dummy",
        rail="lab",
        now_ts=now_ts,
    )


def _run_once(tmp_path: Path, now_ts: float) -> dict:
    return run_autopilot_tick(
        tmp_path,
        options=AutopilotTickOptions(mode="once", interactive=False),
        now_ts=now_ts,
    )


def test_soak_updated_when_file_exists(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_030_000.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    soak_path = tmp_path / "artifacts" / "soak" / "status.json"
    _write_json(soak_path, {"schema": "ajax.soak.status.v0"})
    payload = _run_once(tmp_path, now_ts=1_800_030_010.0)
    soak_doc = json.loads(soak_path.read_text(encoding="utf-8"))
    assert payload.get("action") == "EXECUTED"
    assert soak_doc.get("last_tick_ts")
    assert soak_doc.get("last_action") == "EXECUTED"
    assert isinstance(soak_doc.get("last_gate_summary"), dict)


def test_soak_missing_does_not_fail(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_030_100.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    payload = _run_once(tmp_path, now_ts=1_800_030_110.0)
    soak_update = payload.get("soak_update") if isinstance(payload.get("soak_update"), dict) else {}
    assert payload.get("action") == "EXECUTED"
    assert soak_update.get("updated") is False


def test_receipt_includes_soak_hint_when_missing(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_030_200.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    payload = _run_once(tmp_path, now_ts=1_800_030_210.0)
    hint = str(payload.get("next_hint") or "").lower()
    assert "soak status missing" in hint

