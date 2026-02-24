from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import agency.lab_worker as lab_worker_mod
from agency.lab_worker import LabWorker


def _proc(*, returncode: int, payload: dict | None = None, stdout: str | None = None, stderr: str = ""):
    return SimpleNamespace(
        returncode=returncode,
        stdout=(json.dumps(payload, ensure_ascii=False) if payload is not None else (stdout or "")),
        stderr=stderr,
    )


def _away(monkeypatch, worker: LabWorker) -> None:
    monkeypatch.setattr(
        lab_worker_mod,
        "_compute_explore_state",
        lambda root_dir: {"state": "AWAY", "human_active": False, "human_active_reason": "test"},
        raising=True,
    )


def test_providers_audit_job_ok_no_findings(monkeypatch, tmp_path: Path) -> None:
    worker = LabWorker(tmp_path, worker_id="worker-test", idle_sleep_s=0)
    _away(monkeypatch, worker)
    audit_artifact = tmp_path / "artifacts" / "audit" / "providers_x" / "providers_survival_audit.json"
    audit_receipt = tmp_path / "artifacts" / "receipts" / "providers_survival_audit_x.json"
    audit_artifact.parent.mkdir(parents=True, exist_ok=True)
    audit_receipt.parent.mkdir(parents=True, exist_ok=True)
    audit_artifact.write_text("{}", encoding="utf-8")
    audit_receipt.write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(cmd, capture_output, text, timeout, check):  # noqa: ANN001
        calls.append(list(cmd))
        return _proc(
            returncode=0,
            payload={
                "ok": True,
                "findings": [],
                "summary": {"error": 0, "warn": 0},
                "artifact_path": str(audit_artifact),
                "receipt_path": str(audit_receipt),
            },
        )

    monkeypatch.setattr(lab_worker_mod.subprocess, "run", _fake_run, raising=True)
    result = worker._execute_providers_audit({"job_kind": "providers_audit", "job_id": "j1"}, tmp_path / "job.json")

    assert result["outcome"] == "PASS"
    assert result["efe_pass"] is True
    assert result.get("next_action") == "noop"
    assert result.get("failure_codes") == []
    assert any("providers_survival_audit_x.json" in x for x in (result.get("evidence_refs") or []))
    assert len(calls) == 1
    assert calls[0][2:4] == ["audit", "providers"]


def test_providers_audit_job_findings_quorum_runs_safe_refresh_and_doctor(monkeypatch, tmp_path: Path) -> None:
    worker = LabWorker(tmp_path, worker_id="worker-test", idle_sleep_s=0)
    _away(monkeypatch, worker)
    audit_artifact = tmp_path / "artifacts" / "audit" / "providers_x" / "providers_survival_audit.json"
    audit_receipt = tmp_path / "artifacts" / "receipts" / "providers_survival_audit_x.json"
    doctor_artifact = tmp_path / "artifacts" / "health" / "providers" / "doctor_x.json"
    for p in (audit_artifact, audit_receipt, doctor_artifact):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []
    providers_probe_calls: list[str] = []

    def _fake_run(cmd, capture_output, text, timeout, check):  # noqa: ANN001
        cmd_list = list(cmd)
        calls.append(cmd_list)
        if cmd_list[2:4] == ["audit", "providers"]:
            return _proc(
                returncode=2,
                payload={
                    "ok": False,
                    "findings": [
                        {"code": "policy_provider_missing_in_status", "title": "missing provider status"},
                        {"code": "council_quorum_risk", "title": "quorum risk"},
                    ],
                    "recommended_actions": [
                        {"code": "policy_provider_missing_in_status", "command": "python bin/ajaxctl doctor providers"}
                    ],
                    "artifact_path": str(audit_artifact),
                    "receipt_path": str(audit_receipt),
                },
            )
        if cmd_list[2:4] == ["doctor", "providers"]:
            return _proc(returncode=1, payload={"ok": True, "artifact": str(doctor_artifact), "providers": {"x": "DOWN"}})
        raise AssertionError(f"Unexpected command: {cmd_list}")

    def _fake_probe(job: dict, path: Path) -> dict:  # noqa: ANN001
        providers_probe_calls.append("providers_probe")
        probe_evidence = tmp_path / "artifacts" / "provider_ledger" / "latest.json"
        probe_evidence.parent.mkdir(parents=True, exist_ok=True)
        probe_evidence.write_text("{}", encoding="utf-8")
        return {
            "outcome": "PASS",
            "efe_pass": True,
            "failure_codes": [],
            "evidence_refs": [str(probe_evidence)],
            "next_action": "noop",
            "summary": "probe ok",
        }

    monkeypatch.setattr(lab_worker_mod.subprocess, "run", _fake_run, raising=True)
    monkeypatch.setattr(worker, "_execute_providers_probe", _fake_probe, raising=True)

    result = worker._execute_providers_audit({"job_kind": "providers_audit", "job_id": "j2"}, tmp_path / "job.json")

    assert result["outcome"] == "PASS"
    assert result["efe_pass"] is True
    assert result.get("next_action") == "review_providers_audit_plan"
    assert "providers_audit_findings_present" in (result.get("failure_codes") or [])
    assert providers_probe_calls == ["providers_probe"]
    assert any(cmd[2:4] == ["doctor", "providers"] for cmd in calls)
    assert Path(str(result["providers_audit_plan_path"])).exists()
    assert Path(str(result["providers_audit_outcome_path"])).exists()

    outcome_doc = json.loads(Path(str(result["providers_audit_outcome_path"])).read_text(encoding="utf-8"))
    executed = outcome_doc.get("executed_actions") or []
    assert any(isinstance(x, dict) and x.get("kind") == "providers_probe_refresh" and x.get("ok") for x in executed)
    assert any(isinstance(x, dict) and x.get("kind") == "doctor_providers" and x.get("ok") for x in executed)


def test_providers_audit_job_auth_quota_generates_checklist_only(monkeypatch, tmp_path: Path) -> None:
    worker = LabWorker(tmp_path, worker_id="worker-test", idle_sleep_s=0)
    _away(monkeypatch, worker)
    audit_artifact = tmp_path / "artifacts" / "audit" / "providers_x" / "providers_survival_audit.json"
    audit_receipt = tmp_path / "artifacts" / "receipts" / "providers_survival_audit_x.json"
    for p in (audit_artifact, audit_receipt):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []
    providers_probe_calls: list[str] = []

    def _fake_run(cmd, capture_output, text, timeout, check):  # noqa: ANN001
        cmd_list = list(cmd)
        calls.append(cmd_list)
        if cmd_list[2:4] == ["audit", "providers"]:
            return _proc(
                returncode=2,
                payload={
                    "ok": False,
                    "findings": [
                        {"code": "quota_exhausted", "title": "quota exhausted"},
                        {"code": "provider_auth_failed", "title": "auth failed"},
                    ],
                    "recommended_actions": [
                        {"code": "quota_exhausted", "command": "python bin/ajaxctl doctor providers"}
                    ],
                    "artifact_path": str(audit_artifact),
                    "receipt_path": str(audit_receipt),
                },
            )
        raise AssertionError(f"Unexpected command: {cmd_list}")

    def _fake_probe(job: dict, path: Path) -> dict:  # noqa: ANN001
        providers_probe_calls.append("providers_probe")
        return {
            "outcome": "PASS",
            "efe_pass": True,
            "failure_codes": [],
            "evidence_refs": [],
            "next_action": "noop",
            "summary": "probe ok",
        }

    monkeypatch.setattr(lab_worker_mod.subprocess, "run", _fake_run, raising=True)
    monkeypatch.setattr(worker, "_execute_providers_probe", _fake_probe, raising=True)

    result = worker._execute_providers_audit({"job_kind": "providers_audit", "job_id": "j3"}, tmp_path / "job.json")

    assert result["outcome"] == "PASS"
    assert result["efe_pass"] is True
    assert result.get("next_action") == "ask_user_auth_quota"
    assert result.get("checklist_only") is True
    assert providers_probe_calls == []
    assert len(calls) == 1  # audit only; no doctor providers execution
    outcome_doc = json.loads(Path(str(result["providers_audit_outcome_path"])).read_text(encoding="utf-8"))
    assert outcome_doc.get("checklist_only") is True
    assert outcome_doc.get("executed_actions") == []
    assert outcome_doc.get("checklist")

