from __future__ import annotations

import importlib.machinery
import importlib.util
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


def _load_ajaxctl_module():
    loader = importlib.machinery.SourceFileLoader("ajaxctl_mod_receipts", "bin/ajaxctl")
    spec = importlib.util.spec_from_loader("ajaxctl_mod_receipts", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


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
    warn_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "WARN"]
    assert warn_rows
    assert any(
        "unsupported_receipt_schema" in " ".join(row.get("errors") or [])
        for row in warn_rows
    )


def test_doctor_receipts_default_warns_for_unsupported_schema(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )

    payload = doctor_receipts(tmp_path, since_min=9999)

    assert payload["counts"]["warn"] == 1
    assert payload["counts"]["fail"] == 0
    row = payload["receipts"][0]
    assert row["status"] == "WARN"
    assert row["reason_codes"] == ["unsupported_receipt_schema"]


def test_doctor_receipts_default_exit_zero_when_only_warns(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )
    ajaxctl = _load_ajaxctl_module()
    ajaxctl.ROOT = tmp_path

    rc = ajaxctl.main(["doctor", "receipts", "--since-min", "9999", "--summary-only"])
    assert rc == 0


def test_doctor_receipts_strict_fails_on_warns(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )
    ajaxctl = _load_ajaxctl_module()
    ajaxctl.ROOT = tmp_path

    rc = ajaxctl.main(
        ["doctor", "receipts", "--since-min", "9999", "--strict", "--summary-only"]
    )
    assert rc == 1


def test_doctor_receipts_fails_on_invalid_known_schema(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    _write_json(
        tmp_path / "artifacts" / "receipts" / "topology_bad.json",
        {
            "schema": "ajax.topology_doctor.v1",
            "ts_utc": "2026-03-02T00:00:00Z",
            "ok": True,
            "location": "ajax-kernel",
        },
    )

    payload = doctor_receipts(tmp_path, since_min=9999)
    assert payload["counts"]["fail"] == 1
    row = payload["receipts"][0]
    assert row["status"] == "FAIL"
    assert row["reason_codes"] == ["invalid_against_schema"]


def test_doctor_receipts_fails_on_json_parse_error(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    bad = tmp_path / "artifacts" / "receipts" / "broken.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{\"schema\":", encoding="utf-8")

    payload = doctor_receipts(tmp_path, since_min=9999)
    assert payload["counts"]["fail"] == 1
    row = payload["receipts"][0]
    assert row["status"] == "FAIL"
    assert row["reason_codes"] == ["json_parse_error"]


def test_doctor_receipts_reports_counts(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    _write_json(
        tmp_path / "artifacts" / "receipts" / "topology_ok.json",
        {
            "schema": "ajax.topology_doctor.v1",
            "ts_utc": "2026-03-02T00:00:00Z",
            "ok": True,
            "reason": "topology_ok",
            "location": "ajax-kernel",
        },
    )
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )
    broken = tmp_path / "artifacts" / "receipts" / "broken.json"
    broken.write_text("{\"schema\":", encoding="utf-8")

    payload = doctor_receipts(tmp_path, since_min=9999)
    counts = payload["counts"]
    assert counts["total"] == 3
    assert counts["pass"] == 1
    assert counts["warn"] == 1
    assert counts["fail"] == 1


def test_doctor_receipts_summary_only(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown_1.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown_2.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )

    payload = doctor_receipts(tmp_path, since_min=9999, summary_only=True)
    assert payload["summary_only"] is True
    assert payload["receipts"] == []
    assert payload["omitted"] == 2


def test_doctor_receipts_top_k_limits_output(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown_1.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown_2.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )
    _write_json(
        tmp_path / "artifacts" / "receipts" / "legacy_unknown_3.json",
        {"schema": "ajax.legacy.unknown.v0", "ok": True},
    )

    payload = doctor_receipts(tmp_path, since_min=9999, top_k=1)
    assert payload["top_k"] == 1
    assert len(payload["receipts"]) == 1
    assert payload["omitted"] == 2


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
