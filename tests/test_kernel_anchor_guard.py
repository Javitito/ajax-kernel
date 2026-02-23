from __future__ import annotations

from pathlib import Path

from agency.anchor_preflight import evaluate_anchor_snapshot, run_anchor_preflight


def _services_ok_for_lab() -> dict:
    return {
        "ok": True,
        "sessions": {"AJAX": 22, "Javi": 11},
        "ports": {"5012": {"SessionId": 22}, "5010": {"SessionId": 11}},
        "health": {"5012": True, "5010": True},
    }


def test_anchor_snapshot_detects_session_mismatch() -> None:
    services = _services_ok_for_lab()
    services["ports"]["5012"]["SessionId"] = 999

    out = evaluate_anchor_snapshot(
        rail="lab",
        services_report=services,
        display_target_id=2,
        display_catalog={"displays": [{"id": 2}]},
    )

    assert out["ok"] is False
    codes = {item.get("code") for item in out.get("mismatches", []) if isinstance(item, dict)}
    assert "port_session_mismatch" in codes


def test_anchor_snapshot_detects_display_target_missing() -> None:
    out = evaluate_anchor_snapshot(
        rail="lab",
        services_report=_services_ok_for_lab(),
        display_target_id=None,
        display_catalog={"displays": [{"id": 2}]},
    )

    assert out["ok"] is False
    codes = {item.get("code") for item in out.get("mismatches", []) if isinstance(item, dict)}
    assert "display_target_missing" in codes


def test_anchor_snapshot_lab_dummy_degrades_expected_session_missing_to_warn() -> None:
    services = _services_ok_for_lab()
    services["sessions"].pop("AJAX", None)
    out = evaluate_anchor_snapshot(
        rail="lab",
        services_report=services,
        display_target_id=2,
        display_catalog={"displays": [{"id": 1, "is_primary": True}, {"id": 2, "is_primary": False}]},
    )
    assert out["ok"] is True
    warn_codes = {item.get("code") for item in out.get("warnings", []) if isinstance(item, dict)}
    assert "expected_session_missing" in warn_codes


def test_anchor_snapshot_prod_keeps_expected_session_missing_blocking() -> None:
    out = evaluate_anchor_snapshot(
        rail="prod",
        services_report={"ok": True, "sessions": {}, "ports": {"5010": {"SessionId": 11}}, "health": {"5010": True}},
        display_target_id=1,
        display_catalog={"displays": [{"id": 1, "is_primary": True}]},
    )
    assert out["ok"] is False
    block_codes = {item.get("code") for item in out.get("mismatches", []) if isinstance(item, dict)}
    assert "expected_session_missing" in block_codes


def test_run_anchor_preflight_writes_receipt_with_injected_inputs(tmp_path: Path) -> None:
    services = _services_ok_for_lab()
    display_map = {"display_targets": {"lab": 2, "prod": 1}}
    display_catalog = {"displays": [{"id": 2}, {"id": 1}]}

    out = run_anchor_preflight(
        root_dir=tmp_path,
        rail="lab",
        write_receipt=True,
        services_report=services,
        display_map=display_map,
        display_catalog=display_catalog,
    )

    assert out["ok"] is True
    assert out["status"] == "READY"
    assert out.get("receipt_path")
    assert Path(out["receipt_path"]).exists()
