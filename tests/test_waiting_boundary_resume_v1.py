from __future__ import annotations

import json
import types
from pathlib import Path

from agency.ajax_core import AjaxConfig, AjaxCore, AjaxExecutionResult, MissionState
from agency.receipt_validator import validate_receipt


def _prepare_schema_tree(root: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "schemas" / "receipts"
    dst = root / "schemas" / "receipts"
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*.json"):
        dst.joinpath(path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _build_core(root: Path) -> AjaxCore:
    core = AjaxCore.__new__(AjaxCore)
    state_dir = root / "artifacts" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    core.config = AjaxConfig(root_dir=root, state_dir=state_dir)
    core.root_dir = root
    core.log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    return core


def _write_candidate(path: Path, *, schema: str = "ajax.verify.efe_candidate.v0") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "version": "v0",
                "created_at": "2026-03-12T19:00:00Z",
                "expected_state": {
                    "files": [{"path": str(path.parent / "out.txt"), "must_exist": True}],
                    "checks": [
                        {
                            "kind": "fs",
                            "path": str(path.parent / "out.txt"),
                            "exists": True,
                            "mtime": {"required": True},
                            "size": {"required": True},
                            "sha256": {"required": True},
                        }
                    ],
                },
                "ok": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _seed_waiting_boundary(core: AjaxCore, root: Path, *, mission_id: str, candidate_path: str) -> str:
    original_plan = {
        "plan_id": "plan-resume-1",
        "description": "Write artifact after boundary completion",
        "steps": [
            {
                "id": "task-1",
                "intent": "Write output artifact",
                "preconditions": {"expected_state": {}},
                "action": "write_file",
                "args": {"path": str(root / "artifacts" / "out.txt")},
                "evidence_required": ["driver.active_window"],
                "success_spec": {"expected_state": {}},
                "on_fail": "abort",
            }
        ],
        "success_contract": {"type": "check_last_step_status"},
        "metadata": {"intention": "write artifact", "rail": "lab", "source": "brain"},
    }
    wait_plan = core._build_missing_efe_plan(
        intention="write artifact",
        source="brain",
        receipt_path=str(root / "artifacts" / "receipts" / "efe_repair_seed.json"),
        terminal="WAITING_FOR_USER",
        waiting_prompt="Completa el boundary EFE antes de continuar.",
        boundary={
            "kind": "efe_boundary_completion",
            "step_id": "task-1",
            "repair_path": "candidate",
            "template_id": "efe.fs_path_materialized.v0",
            "candidate_path": candidate_path,
            "refusal_reasons": ["mission_family_requires_boundary:repo_patch"],
            "exact_boundary": "Completa el expected_state para el paso task-1.",
        },
        repair_path="candidate",
        template_id="efe.fs_path_materialized.v0",
        efe_candidate_path=candidate_path,
        efe_candidate_source_doc=original_plan,
        reason="guarded_auto_materialization_refused",
    )
    mission = MissionState(
        intention="write artifact",
        mode="auto",
        mission_id=mission_id,
    )
    mission.status = "WAITING_FOR_USER"
    mission.await_user_input = True
    mission.pending_plan = wait_plan
    mission.last_plan = wait_plan
    mission.notes = {}
    user_payload = core._build_ask_user_payload(
        mission,
        "Completa el boundary EFE antes de continuar.",
        blocking_reason="await_user_input_step",
        source="plan",
        extra_context=core._waiting_resume_context_from_plan(wait_plan),
    )
    return str(
        core._persist_waiting_mission(
            mission,
            question=user_payload.question,
            user_payload=user_payload.to_dict(),
        )
    )


def _completion_payload(*, mission_id: str, candidate_path: str) -> dict:
    return {
        "schema": "ajax.waiting_boundary_completion.v1",
        "version": "v1",
        "mission_id": mission_id,
        "boundary_kind": "efe_boundary_completion",
        "completion_source": "user_json",
        "candidate_path": candidate_path,
        "boundary_fields": {"step_id": "task-1"},
        "completed_utc": "2026-03-12T19:05:00Z",
        "notes": "Candidate reviewed by operator.",
    }


def test_valid_boundary_completion_patches_and_resumes_same_mission_id(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    core = _build_core(tmp_path)
    candidate_path = _write_candidate(tmp_path / "artifacts" / "efe_candidates" / "candidate.json")
    _seed_waiting_boundary(core, tmp_path, mission_id="mission-boundary-1", candidate_path=candidate_path)

    captured = {}

    def _pursue(self: AjaxCore, mission: MissionState):
        captured["mission_id"] = mission.mission_id
        captured["status_before"] = mission.status
        captured["expected_state"] = mission.pending_plan.steps[0]["success_spec"]["expected_state"]
        mission.status = "DONE"
        return AjaxExecutionResult(
            success=True,
            error=None,
            path="plan_runner",
            detail={"resumed": True},
            mission_id=mission.mission_id,
        )

    core.pursue_intent = types.MethodType(_pursue, core)

    outcome = core.complete_waiting_boundary(
        _completion_payload(mission_id="mission-boundary-1", candidate_path=candidate_path)
    )

    assert outcome["ok"] is True
    assert outcome["same_mission_id"] is True
    assert outcome["mission_id"] == "mission-boundary-1"
    assert outcome["resume_attempted"] is True
    assert captured["mission_id"] == "mission-boundary-1"
    assert captured["status_before"] == "IN_PROGRESS"
    assert captured["expected_state"]["checks"][0]["kind"] == "fs"
    assert len(outcome["receipt_paths"]) >= 5
    per_path = tmp_path / "artifacts" / "waiting_for_user" / "mission-boundary-1.json"
    per_doc = json.loads(per_path.read_text(encoding="utf-8"))
    assert per_doc["consumed_utc"]
    assert per_doc["boundary_completion"]["mission_id"] == "mission-boundary-1"
    for receipt_path in outcome["receipt_paths"]:
        report = validate_receipt(tmp_path, Path(str(receipt_path)))
        assert report["ok"] is True


def test_invalid_completion_missing_candidate_path_stays_waiting(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    core = _build_core(tmp_path)
    candidate_path = _write_candidate(tmp_path / "artifacts" / "efe_candidates" / "candidate.json")
    _seed_waiting_boundary(core, tmp_path, mission_id="mission-boundary-2", candidate_path=candidate_path)

    def _boom(self: AjaxCore, mission: MissionState):  # noqa: ANN001
        raise AssertionError("pursue_intent should not be called")

    core.pursue_intent = types.MethodType(_boom, core)
    payload = _completion_payload(
        mission_id="mission-boundary-2",
        candidate_path=str(tmp_path / "artifacts" / "efe_candidates" / "missing.json"),
    )
    outcome = core.complete_waiting_boundary(payload)

    assert outcome["ok"] is False
    assert outcome["status"] == "WAITING_FOR_USER"
    assert outcome["resume_attempted"] is False
    assert "candidate_path_not_found" in outcome["errors"]
    pointer = tmp_path / "artifacts" / "state" / "waiting_mission.json"
    assert pointer.exists()
    receipt_doc = json.loads(Path(str(outcome["receipt_paths"][-1])).read_text(encoding="utf-8"))
    assert receipt_doc["event"] == "completion_refused"


def test_schema_invalid_candidate_is_refused_cleanly(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    core = _build_core(tmp_path)
    seed_candidate = _write_candidate(tmp_path / "artifacts" / "efe_candidates" / "seed.json")
    _seed_waiting_boundary(core, tmp_path, mission_id="mission-boundary-3", candidate_path=seed_candidate)
    bad_candidate = _write_candidate(
        tmp_path / "artifacts" / "efe_candidates" / "bad.json",
        schema="ajax.verify.efe_candidate.v999",
    )

    def _boom(self: AjaxCore, mission: MissionState):  # noqa: ANN001
        raise AssertionError("pursue_intent should not be called")

    core.pursue_intent = types.MethodType(_boom, core)
    outcome = core.complete_waiting_boundary(
        _completion_payload(mission_id="mission-boundary-3", candidate_path=bad_candidate)
    )

    assert outcome["ok"] is False
    assert outcome["status"] == "WAITING_FOR_USER"
    assert "candidate_schema_invalid" in outcome["errors"]


def test_candidate_payload_is_persisted_and_receipts_validate(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    core = _build_core(tmp_path)
    seed_candidate = _write_candidate(tmp_path / "artifacts" / "efe_candidates" / "seed.json")
    _seed_waiting_boundary(core, tmp_path, mission_id="mission-boundary-4", candidate_path=seed_candidate)

    def _pursue(self: AjaxCore, mission: MissionState):
        mission.status = "DONE"
        return AjaxExecutionResult(
            success=True,
            error=None,
            path="plan_runner",
            detail={"resumed": True},
            mission_id=mission.mission_id,
        )

    core.pursue_intent = types.MethodType(_pursue, core)
    payload = _completion_payload(mission_id="mission-boundary-4", candidate_path="")
    payload.pop("candidate_path", None)
    payload["candidate_payload"] = {
        "schema": "ajax.verify.efe_candidate.v0",
        "version": "v0",
        "created_at": "2026-03-12T19:08:00Z",
        "expected_state": {
            "checks": [
                {
                    "kind": "structured_output",
                    "path": str(tmp_path / "artifacts" / "subcalls" / "scout.json"),
                    "format": "json",
                    "root_type": "object",
                    "required_keys": ["schema", "result"],
                }
            ]
        },
        "ok": True,
    }

    outcome = core.complete_waiting_boundary(payload)

    assert outcome["ok"] is True
    assert str(outcome["candidate_path"]).endswith("_completed.json")
    for receipt_path in outcome["receipt_paths"]:
        report = validate_receipt(tmp_path, Path(str(receipt_path)))
        assert report["ok"] is True


def test_downstream_resume_can_honestly_land_back_in_waiting(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    core = _build_core(tmp_path)
    candidate_path = _write_candidate(tmp_path / "artifacts" / "efe_candidates" / "candidate.json")
    _seed_waiting_boundary(core, tmp_path, mission_id="mission-boundary-5", candidate_path=candidate_path)

    def _pursue(self: AjaxCore, mission: MissionState):
        mission.status = "WAITING_FOR_USER"
        return AjaxExecutionResult(
            success=False,
            error="await_user_input",
            path="waiting_for_user",
            detail={"question": "Need another boundary."},
            mission_id=mission.mission_id,
        )

    core.pursue_intent = types.MethodType(_pursue, core)
    outcome = core.complete_waiting_boundary(
        _completion_payload(mission_id="mission-boundary-5", candidate_path=candidate_path)
    )

    assert outcome["ok"] is False
    assert outcome["status"] == "WAITING_FOR_USER"
    assert outcome["resume_attempted"] is True
    receipt_doc = json.loads(Path(str(outcome["receipt_paths"][-1])).read_text(encoding="utf-8"))
    assert receipt_doc["event"] == "resume_outcome"
    assert receipt_doc["outcome"] == "WAITING_FOR_USER"
