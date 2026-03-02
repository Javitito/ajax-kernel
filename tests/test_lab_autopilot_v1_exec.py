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
    options = AutopilotTickOptions(
        mode="once",
        interactive=False,
        allow_filesystem_basic=True,
        allow_queue_housekeeping=True,
    )
    return run_autopilot_tick(tmp_path, options=options, now_ts=now_ts)


def test_autopilot_once_executes_filesystem_basic_with_valid_session(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_010_000.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    payload = _run_once(tmp_path, now_ts=1_800_010_010.0)
    selected = payload.get("selected_work_item") if isinstance(payload.get("selected_work_item"), dict) else {}
    assert payload.get("action") == "EXECUTED"
    assert selected.get("kind") == "filesystem_basic"


def test_autopilot_writes_evidence_and_result(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_010_100.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    payload = _run_once(tmp_path, now_ts=1_800_010_110.0)
    result_rel = str(payload.get("result_path") or "")
    result_path = tmp_path / result_rel
    assert result_path.exists()
    result_doc = json.loads(result_path.read_text(encoding="utf-8"))
    evidence = result_doc.get("evidence_refs") if isinstance(result_doc.get("evidence_refs"), list) else []
    assert evidence
    assert (tmp_path / "artifacts" / "lab" / "evidence" / str(result_doc.get("job_id")) / "touch.txt").exists()


def test_autopilot_idempotent_same_day_no_duplicate(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_010_200.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    first = _run_once(tmp_path, now_ts=1_800_010_210.0)
    second = _run_once(tmp_path, now_ts=1_800_010_220.0)
    first_job = str((first.get("selected_work_item") or {}).get("job_id"))
    results = sorted((tmp_path / "artifacts" / "lab" / "results").glob(f"result_*_{first_job}.json"))
    assert first.get("action") == "EXECUTED"
    assert second.get("action") == "EXECUTED"
    assert len(results) == 1


def test_autopilot_noop_when_already_done(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_010_300.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    _run_once(tmp_path, now_ts=1_800_010_310.0)  # W1
    _run_once(tmp_path, now_ts=1_800_010_320.0)  # W2
    payload = _run_once(tmp_path, now_ts=1_800_010_330.0)  # no W3 pending
    assert payload.get("action") == "NOOP"
    assert "already done today" in str(payload.get("next_hint") or "").lower()


def test_autopilot_still_blocks_when_session_missing(monkeypatch, tmp_path: Path) -> None:
    _patch_anchor_ok(monkeypatch, tmp_path)
    payload = _run_once(tmp_path, now_ts=1_800_010_400.0)
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    env_gate = gates.get("env_safe") if isinstance(gates.get("env_safe"), dict) else {}
    assert payload.get("action") == "BLOCKED"
    assert env_gate.get("reason") == "expected_session_missing"


def test_autopilot_blocks_when_session_expired(monkeypatch, tmp_path: Path) -> None:
    session_mod.init_expected_session(
        tmp_path,
        ttl_min=1,
        display="dummy",
        rail="lab",
        now_ts=1_800_010_500.0,
    )
    _patch_anchor_ok(monkeypatch, tmp_path)
    payload = _run_once(tmp_path, now_ts=1_800_010_700.0)
    gates = payload.get("gates") if isinstance(payload.get("gates"), dict) else {}
    env_gate = gates.get("env_safe") if isinstance(gates.get("env_safe"), dict) else {}
    assert payload.get("action") == "BLOCKED"
    assert env_gate.get("reason") == "session_expired"


def test_autopilot_work_item_selection_priority(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_010_800.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    first = _run_once(tmp_path, now_ts=1_800_010_810.0)
    second = _run_once(tmp_path, now_ts=1_800_010_820.0)
    _write_json(
        tmp_path / "artifacts" / "lab" / "jobs" / "pending" / "job_a.json",
        {"job_id": "job_a", "status": "pending"},
    )
    third = _run_once(tmp_path, now_ts=1_800_010_830.0)
    first_kind = str((first.get("selected_work_item") or {}).get("kind"))
    second_kind = str((second.get("selected_work_item") or {}).get("kind"))
    third_kind = str((third.get("selected_work_item") or {}).get("kind"))
    assert first_kind == "filesystem_basic"
    assert second_kind == "providers_probe_refresh"
    assert third_kind == "lab_queue_housekeeping"


def test_tick_receipt_includes_artifacts_written(monkeypatch, tmp_path: Path) -> None:
    _valid_session(tmp_path, now_ts=1_800_010_900.0)
    _patch_anchor_ok(monkeypatch, tmp_path)
    payload = _run_once(tmp_path, now_ts=1_800_010_910.0)
    tick_receipt = tmp_path / str(payload.get("tick_receipt_path"))
    receipt = json.loads(tick_receipt.read_text(encoding="utf-8"))
    written = receipt.get("artifacts_written") if isinstance(receipt.get("artifacts_written"), list) else []
    assert written
    assert str(payload.get("tick_receipt_path")) in written
    assert "schema" in receipt and str(receipt["schema"]) == "ajax.lab.autopilot_tick.v1"
