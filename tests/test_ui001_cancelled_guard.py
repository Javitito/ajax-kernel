from __future__ import annotations

import json
from pathlib import Path

import agency.lab_org as lab_org_mod
from agency.contract import AgencyJob
from agency.lab_control import LabStateStore
from agency.lab_org import lab_org_tick
from agency.plan_runner import StepResult, _emit_capability_gap


def _write_explore_policy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "schema: ajax.explore_policy.v1\n"
            "policy:\n"
            "  human_active_threshold_s: 90\n"
            "  unknown_signal_as_human: true\n"
            "  away_ui_allowed_job_kinds: [probe_ui]\n"
            "states:\n"
            "  AWAY:\n"
            "    require_dummy_display_ok: false\n"
            "  HUMAN_DETECTED:\n"
            "    ui_intrusive_allowed: false\n"
            "human_signal:\n"
            "  ps_script: scripts/ops/get_human_signal.ps1\n"
            "  timeout_s: 2.5\n"
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path, *, challenges: list[dict]) -> None:
    payload = {"schema": "ajax.lab_org_manifest.v1", "micro_challenges": challenges}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "schema: ajax.experiments_registry.v1\n"
            "experiments:\n"
            "  UI-001:\n"
            "    id: UI-001\n"
            "    status: CANCELLED\n"
            "    reason: \"No ROI / Alto ruido / No estable\"\n"
        ),
        encoding="utf-8",
    )


def test_ui001_is_never_enqueued_in_lab_org(tmp_path: Path, monkeypatch) -> None:
    store = LabStateStore(tmp_path)
    store.resume_lab_org("boot")
    _write_explore_policy(tmp_path / "config" / "explore_policy.yaml")
    _write_registry(tmp_path / "config" / "experiments_registry.yaml")
    manifest = tmp_path / "config" / "lab_org_manifest.yaml"
    _write_manifest(
        manifest,
        challenges=[
            {"id": "UI-001", "job_kind": "probe_ui", "ui_intrusive": True, "cadence_s": 0, "enabled": True}
        ],
    )

    monkeypatch.setattr(
        lab_org_mod,
        "evaluate_explore_state",
        lambda root, policy, prev_state=None, now_ts=None: {
            "schema": "ajax.explore_state.v1",
            "state": "AWAY",
            "trigger": None,
            "human_active": False,
        },
        raising=True,
    )

    receipt = lab_org_tick(tmp_path, manifest_path=manifest, out_base=tmp_path / "artifacts" / "lab_org_cancelled")
    assert receipt.get("enqueued") is False
    assert receipt.get("skipped_reason") == "experiment_cancelled"
    assert receipt.get("experiment_id") == "UI-001"
    assert receipt.get("reason") == "experiment_cancelled"
    assert store.list_jobs(statuses={"QUEUED", "RUNNING"}) == []


def test_cancelled_experiment_gap_write_is_redirected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_registry(tmp_path / "config" / "experiments_registry.yaml")

    job = AgencyJob(job_id="job_ui001_case", goal="Run UI-001", metadata={"mission_id": "UI-001"})
    step_results = [StepResult(step_id="step_1", ok=False, detail={"kind": "demo"}, tries=1, error="failed")]
    final_result = {"ok": False, "status": "FAIL", "error": "driver_timeout"}

    _emit_capability_gap(job, step_results, final_result)

    direct_gaps = list((tmp_path / "artifacts" / "capability_gaps").glob("*.json"))
    assert direct_gaps == []

    cancelled_dir = tmp_path / "artifacts" / "capability_gaps" / "cancelled" / "UI-001"
    cancelled_files = list(cancelled_dir.glob("*.json"))
    assert len(cancelled_files) == 1
    payload = json.loads(cancelled_files[0].read_text(encoding="utf-8"))
    cancelled = payload.get("cancelled_experiment") or {}
    assert cancelled.get("label") == "CANCELLED"
    assert cancelled.get("status") == "CANCELLED"

    receipts = list((tmp_path / "artifacts" / "receipts").glob("gap_guard_*.json"))
    assert receipts
    receipt_payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt_payload.get("skipped_reason") == "experiment_cancelled"
    assert receipt_payload.get("experiment_id") == "UI-001"
