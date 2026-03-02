from __future__ import annotations

import json
from pathlib import Path

import agency.lab_autopilot as autopilot_mod
import agency.lab_session_anchor as session_mod
from agency.lab_autopilot import AutopilotTickOptions, run_autopilot_tick
from agency.receipt_validator import doctor_receipts, validate_receipt


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_schema_tree(root: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "schemas" / "receipts"
    dst = root / "schemas" / "receipts"
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*.json"):
        dst.joinpath(path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def test_validate_receipt_ok(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipt = tmp_path / "artifacts" / "receipts" / "lab_session_init_test.json"
    _write_json(
        receipt,
        {
            "schema": "ajax.lab.session.init.v0",
            "ts_utc": "2026-03-02T00:00:00Z",
            "ok": True,
            "session_path": "artifacts/lab/session/expected_session.json",
        },
    )
    report = validate_receipt(tmp_path, receipt)
    assert report.get("ok") is True


def test_validate_receipt_invalid(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipt = tmp_path / "artifacts" / "receipts" / "lab_autopilot_tick_bad.json"
    _write_json(
        receipt,
        {
            "schema": "ajax.lab.autopilot_tick.v1",
            "ts_utc": "2026-03-02T00:00:00Z",
            "mode": "once",
            "gates": {},
            "action": "EXECUTED",
            "evidence_refs": [],
            "next_hint": "ok"
        },
    )
    report = validate_receipt(tmp_path, receipt)
    assert report.get("ok") is False
    assert any("anyOf" in str(err) for err in report.get("errors") or [])


def test_doctor_receipts_lists_recent(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipt = tmp_path / "artifacts" / "receipts" / "topology_doctor_recent.json"
    _write_json(
        receipt,
        {
            "schema": "ajax.topology_doctor.v1",
            "ts_utc": "2026-03-02T00:00:00Z",
            "ok": True,
            "reason": "topology_ok",
            "location": "ajax-kernel",
        },
    )
    payload = doctor_receipts(tmp_path, since_min=9999)
    rows = payload.get("receipts") if isinstance(payload.get("receipts"), list) else []
    assert rows
    assert any("topology_doctor_recent.json" in str(row.get("path")) for row in rows if isinstance(row, dict))


def test_doctor_receipts_reports_schema_mismatch(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipt = tmp_path / "artifacts" / "receipts" / "unknown_schema.json"
    _write_json(
        receipt,
        {
            "schema": "ajax.unknown.receipt.v0",
            "ts_utc": "2026-03-02T00:00:00Z",
            "ok": True,
        },
    )
    payload = doctor_receipts(tmp_path, since_min=9999)
    rows = payload.get("receipts") if isinstance(payload.get("receipts"), list) else []
    fail_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "FAIL"]
    assert fail_rows
    assert any("unsupported_receipt_schema" in " ".join(row.get("errors") or []) for row in fail_rows)


def test_autopilot_tick_receipt_conforms_schema(monkeypatch, tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    session_mod.init_expected_session(tmp_path, ttl_min=120, display="dummy", rail="lab", now_ts=1_800_020_000.0)

    def _fake_anchor(root_dir: Path, rail: str) -> dict:
        return {
            "ok": True,
            "rail": rail,
            "receipt_path": str(tmp_path / "artifacts" / "receipts" / "anchor_fake.json"),
            "observed": {"display_target_is_dummy": True},
            "mismatches": [],
        }

    monkeypatch.setattr(autopilot_mod, "_run_anchor_preflight", _fake_anchor, raising=True)
    payload = run_autopilot_tick(
        tmp_path,
        options=AutopilotTickOptions(mode="once", interactive=False),
        now_ts=1_800_020_010.0,
    )
    tick_path = tmp_path / str(payload.get("tick_receipt_path"))
    report = validate_receipt(tmp_path, tick_path)
    assert report.get("ok") is True


def test_session_receipt_conforms_schema(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    payload = session_mod.init_expected_session(
        tmp_path,
        ttl_min=120,
        display="dummy",
        rail="lab",
        now_ts=1_800_020_100.0,
    )
    receipt_path = tmp_path / str(payload.get("receipt_path"))
    report = validate_receipt(tmp_path, receipt_path)
    assert report.get("ok") is True
