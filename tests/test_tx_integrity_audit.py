from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import agency.audits.tx_integrity_audit as audit_mod


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _base_task_step(action: str = "shell.exec") -> Dict[str, Any]:
    return {
        "id": "step-1",
        "action": action,
        "args": {"cmd": "echo hi"},
        "success_spec": {
            "expected_state": {"files": [{"path": "C:/tmp/example.txt", "exists": True}]}
        },
    }


def _tx_step_complete(action: str = "shell.exec") -> Dict[str, Any]:
    step = _base_task_step(action=action)
    step["transactional"] = {
        "prepare": {"snapshot": True, "undo_script_gen": True},
        "verify": {"doctor_check": True, "efe_check": True},
        "undo": {"rollback_best_effort": True},
    }
    return step


def _plan(*steps: Dict[str, Any]) -> Dict[str, Any]:
    return {"steps": list(steps)}


def _result_with_valid_claim() -> Dict[str, Any]:
    return {
        "claims": [
            {
                "type": "verified",
                "statement": "verificado",
                "evidence_refs": [{"kind": "verify_result", "path": "artifacts/verify.json"}],
            }
        ]
    }


def _result_with_invalid_claim() -> Dict[str, Any]:
    return {
        "claims": [{"type": "fixed", "statement": "resuelto", "evidence_refs": []}],
    }


def _result_with_hypothesis_missing_reason() -> Dict[str, Any]:
    return {
        "hypothesis": "Posible causa sin detalle",
        "verification_commands": ["python bin/ajaxctl doctor providers"],
    }


def _write_run(
    root: Path,
    *,
    run_id: str,
    plan_doc: Dict[str, Any],
    result_doc: Dict[str, Any],
    verification_doc: Dict[str, Any] | None = None,
) -> Path:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "plan.json", plan_doc)
    _write_json(run_dir / "result.json", result_doc)
    _write_json(run_dir / "verification.json", verification_doc or {"ok": True})
    return run_dir


def _codes(out: Dict[str, Any]) -> List[str]:
    findings = out.get("findings") if isinstance(out.get("findings"), list) else []
    return [str(f.get("code")) for f in findings if isinstance(f, dict) and f.get("code")]


def test_tx_audit_ok_schema_and_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_run(
        tmp_path,
        run_id="run-ok",
        plan_doc=_plan(_tx_step_complete()),
        result_doc=_result_with_valid_claim(),
    )
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000000.0)

    out = audit_mod.run_tx_integrity_audit(tmp_path, run_id="run-ok")

    assert out["schema"] == "ajax.audit.tx_integrity.v0"
    assert out["ok"] is True
    assert out["read_only"] is True
    assert out["live_probes_invoked"] is False
    assert Path(str(out["artifact_path"])).exists()
    assert Path(str(out["receipt_path"])).exists()
    summary = out.get("summary", {})
    assert "transactional_wrappers_seen" in summary
    assert "proof_claims_valid" in summary
    assert "proof_hypotheses_valid" in summary
    assert int(summary.get("transactional_wrappers_seen") or 0) >= 1
    assert int(summary.get("proof_claims_valid") or 0) >= 1


def test_tx_audit_code_tx_undo_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _tx_step_complete()
    bad["transactional"]["undo"]["rollback_best_effort"] = False
    _write_run(tmp_path, run_id="run-no-undo", plan_doc=_plan(bad), result_doc=_result_with_valid_claim())
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000001.0)

    out = audit_mod.run_tx_integrity_audit(tmp_path, run_id="run-no-undo")
    assert out["ok"] is False
    assert "tx_undo_missing" in _codes(out)


def test_tx_audit_code_mutating_taskstep_without_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(
        tmp_path,
        run_id="run-no-wrapper",
        plan_doc=_plan(_base_task_step("shell.exec")),
        result_doc=_result_with_valid_claim(),
    )
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000002.0)

    out = audit_mod.run_tx_integrity_audit(tmp_path, run_id="run-no-wrapper")
    assert out["ok"] is False
    assert "mutating_taskstep_without_transactional_wrapper" in _codes(out)


def test_tx_audit_fail_closed_claim_without_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_run(
        tmp_path,
        run_id="run-claim-no-proof",
        plan_doc=_plan(_tx_step_complete()),
        result_doc=_result_with_invalid_claim(),
    )
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000003.0)

    out = audit_mod.run_tx_integrity_audit(tmp_path, run_id="run-claim-no-proof")
    assert "proof_claim_missing_minimum_evidence" in _codes(out)
    assert out["ok"] is False  # fail-closed even when finding is warn


def test_tx_audit_code_proof_hypothesis_missing_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(
        tmp_path,
        run_id="run-hyp-no-reason",
        plan_doc=_plan(_tx_step_complete()),
        result_doc=_result_with_hypothesis_missing_reason(),
    )
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000004.0)

    out = audit_mod.run_tx_integrity_audit(tmp_path, run_id="run-hyp-no-reason")
    assert "proof_hypothesis_missing_reason" in _codes(out)
