from __future__ import annotations

import json
from pathlib import Path

import agency.lab_autopilot as autopilot_mod
from agency.lab_autopilot import AutopilotTickOptions, run_autopilot_tick


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _patch_anchor(monkeypatch, root: Path, *, ok: bool = True, dummy: bool = True) -> None:
    anchor_receipt = root / "artifacts" / "receipts" / "anchor_preflight_fake.json"
    _write_json(anchor_receipt, {"ok": ok, "dummy": dummy})

    def _fake_anchor(root_dir: Path, rail: str) -> dict:
        payload = {
            "ok": ok,
            "rail": rail,
            "receipt_path": str(anchor_receipt),
            "observed": {"display_target_is_dummy": dummy},
        }
        if not ok:
            payload["mismatches"] = [{"code": "anchor_failed"}]
        return payload

    monkeypatch.setattr(autopilot_mod, "_run_anchor_preflight", _fake_anchor, raising=True)

    session_display = "dummy" if dummy else "primary"

    def _fake_session_status(root_dir: Path, **kwargs) -> dict:
        return {
            "schema": "ajax.lab.session_status.v0",
            "ts_utc": "2026-03-02T00:00:00Z",
            "ok": True,
            "reason": "session_valid",
            "exists": True,
            "path": str(root / "artifacts" / "lab" / "session" / "expected_session.json"),
            "rail": "lab",
            "display_target": session_display,
            "expires_at": "2099-01-01T00:00:00Z",
            "token_hash_prefix": "deadbeef1234",
            "invalid_reasons": [],
        }

    monkeypatch.setattr(autopilot_mod, "validate_expected_session", _fake_session_status, raising=True)


def _fresh_status(root: Path, *, updated_ts: float) -> Path:
    status_path = root / "artifacts" / "health" / "providers_status.json"
    _write_json(
        status_path,
        {
            "schema": "ajax.providers_status.v1",
            "updated_ts": updated_ts,
            "updated_utc": "2026-03-02T00:00:00Z",
            "providers": {},
        },
    )
    return status_path


def test_autopilot_dry_run_writes_receipt_and_no_job(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor(monkeypatch, tmp_path, ok=True, dummy=True)
    options = AutopilotTickOptions(
        mode="dry-run",
        interactive=False,
        allow_filesystem_basic=True,
        allow_queue_housekeeping=False,
    )
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_000_000.0)
    assert payload.get("action") == "NOOP"
    tick_rel = str(payload.get("tick_receipt_path"))
    assert tick_rel
    assert (tmp_path / tick_rel).exists()
    results_dir = tmp_path / "artifacts" / "lab" / "results"
    assert list(results_dir.glob("result_*.json")) == []


def test_autopilot_blocks_when_human_active(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor(monkeypatch, tmp_path, ok=True, dummy=True)
    _write_json(
        tmp_path / "artifacts" / "health" / "human_signal.json",
        {"human_active": True, "last_input_age_sec": 5, "ok": True},
    )
    options = AutopilotTickOptions(mode="once", interactive=False, allow_queue_housekeeping=False)
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_000_100.0)
    assert payload.get("action") == "BLOCKED"
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    human_gate = gates.get("human_absent") if isinstance(gates.get("human_absent"), dict) else {}
    assert human_gate.get("reason") == "human_active_true"


def test_autopilot_blocks_when_rail_not_lab(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor(monkeypatch, tmp_path, ok=True, dummy=True)
    options = AutopilotTickOptions(
        mode="once",
        rail="prod",
        interactive=False,
        allow_queue_housekeeping=False,
    )
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_000_200.0)
    assert payload.get("action") == "BLOCKED"
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    env_gate = gates.get("env_safe") if isinstance(gates.get("env_safe"), dict) else {}
    assert env_gate.get("reason") == "rail_not_lab"


def test_autopilot_blocks_when_dummy_display_not_guaranteed(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor(monkeypatch, tmp_path, ok=True, dummy=False)
    options = AutopilotTickOptions(mode="once", interactive=False, allow_queue_housekeeping=False)
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_000_300.0)
    assert payload.get("action") == "BLOCKED"
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    env_gate = gates.get("env_safe") if isinstance(gates.get("env_safe"), dict) else {}
    assert env_gate.get("reason") == "dummy_display_not_guaranteed"


def test_autopilot_noop_when_no_safe_items(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor(monkeypatch, tmp_path, ok=True, dummy=True)
    _fresh_status(tmp_path, updated_ts=1_800_000_350.0)
    options = AutopilotTickOptions(
        mode="once",
        interactive=False,
        allow_providers_probe_refresh=False,
        providers_stale_min=60.0,
        allow_filesystem_basic=False,
        allow_queue_housekeeping=False,
    )
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_000_400.0)
    assert payload.get("action") == "NOOP"
    assert payload.get("selected_work_item") is None


def test_autopilot_executes_providers_probe_refresh_when_stale(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor(monkeypatch, tmp_path, ok=True, dummy=True)
    _fresh_status(tmp_path, updated_ts=1_000.0)
    options = AutopilotTickOptions(
        mode="once",
        interactive=False,
        providers_stale_min=1.0,
        allow_filesystem_basic=False,
        allow_queue_housekeeping=False,
    )
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_000_600.0)
    selected = payload.get("selected_work_item") if isinstance(payload.get("selected_work_item"), dict) else {}
    assert payload.get("action") == "EXECUTED"
    assert selected.get("kind") == "providers_probe_refresh"


def test_autopilot_executes_filesystem_basic_creates_evidence_file(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor(monkeypatch, tmp_path, ok=True, dummy=True)
    _fresh_status(tmp_path, updated_ts=1_800_000_690.0)
    options = AutopilotTickOptions(
        mode="once",
        interactive=False,
        providers_stale_min=60.0,
        allow_filesystem_basic=True,
        allow_queue_housekeeping=False,
    )
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_000_700.0)
    selected = payload.get("selected_work_item") if isinstance(payload.get("selected_work_item"), dict) else {}
    assert payload.get("action") == "EXECUTED"
    assert selected.get("kind") == "filesystem_basic"
    result_rel = str(payload.get("result_path") or "")
    result_doc = json.loads((tmp_path / result_rel).read_text(encoding="utf-8"))
    evidence = result_doc.get("evidence_refs") if isinstance(result_doc.get("evidence_refs"), list) else []
    assert evidence
    assert Path(evidence[0]).exists()


def test_autopilot_once_writes_result_when_executed(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor(monkeypatch, tmp_path, ok=True, dummy=True)
    _fresh_status(tmp_path, updated_ts=1_800_000_790.0)
    options = AutopilotTickOptions(
        mode="once",
        interactive=False,
        providers_stale_min=60.0,
        allow_filesystem_basic=True,
        allow_queue_housekeeping=False,
    )
    payload = run_autopilot_tick(tmp_path, options=options, now_ts=1_800_000_800.0)
    assert payload.get("action") == "EXECUTED"
    result_rel = str(payload.get("result_path") or "")
    assert result_rel
    assert (tmp_path / result_rel).exists()
