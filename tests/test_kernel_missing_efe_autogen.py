from __future__ import annotations

import json
import types
from pathlib import Path

import agency.ajax_core as ajax_core_mod
from agency.ajax_core import AjaxConfig, AjaxCore, MissionState


def _build_core(tmp_path: Path) -> AjaxCore:
    core = AjaxCore.__new__(AjaxCore)
    state_dir = tmp_path / "artifacts" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    core.config = AjaxConfig(root_dir=tmp_path, state_dir=state_dir)
    core.root_dir = tmp_path
    core.log = types.SimpleNamespace(warning=lambda *a, **k: None)
    return core


def test_gap_missing_efe_generates_candidate_for_fs_descriptor(tmp_path: Path) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="write file", mission_id="m-fs")

    gap_path = core._emit_missing_efe_gap(
        mission,
        reason="missing_efe_final",
        efe_candidate_source_doc={
            "steps": [
                {
                    "id": "task-1",
                    "action": "write_file",
                    "args": {"path": "artifacts/tmp/output.txt"},
                }
            ]
        },
    )

    assert isinstance(gap_path, str)
    gap_payload = json.loads(Path(gap_path).read_text(encoding="utf-8"))
    assert gap_payload.get("efe_candidate_status") == "generated"
    candidate_path = Path(gap_payload["efe_candidate_path"])
    assert candidate_path.exists()


def test_gap_missing_efe_does_not_execute_and_sets_fail_closed_flag(tmp_path: Path) -> None:
    core = _build_core(tmp_path)

    plan = core._build_missing_efe_plan(
        intention="no efe",
        source="brain",
        receipt_path="artifacts/receipts/missing_efe.json",
        reason="missing_efe_final",
        errors=["missing expected_state"],
    )

    assert plan.steps == []
    assert isinstance(plan.metadata, dict)
    assert plan.metadata.get("planning_error") == "missing_efe_final"
    assert plan.metadata.get("fail_closed") is True


def test_gap_unknown_descriptor_marks_unsupported_with_hint(tmp_path: Path) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="unknown", mission_id="m-unknown")

    gap_path = core._emit_missing_efe_gap(
        mission,
        reason="missing_efe_final",
        efe_candidate_source_doc={"steps": [{"action": "mystery_action", "args": {}}]},
    )

    payload = json.loads(Path(gap_path).read_text(encoding="utf-8"))
    assert payload.get("efe_candidate_status") == "unsupported"
    assert "unsupported_action_kind" in str(payload.get("efe_candidate_reason") or "")


def test_candidate_written_to_artifacts_and_path_stored(tmp_path: Path) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="emit gap", mission_id="m-path")

    gap_path = core._emit_missing_efe_gap(
        mission,
        reason="missing_efe_final",
        efe_candidate_source_doc={"steps": [{"action": "write_file", "args": {"path": "x.txt"}}]},
    )

    payload = json.loads(Path(gap_path).read_text(encoding="utf-8"))
    candidate_path = Path(payload["efe_candidate_path"])
    gap_id = str(payload.get("gap_id") or "")
    assert candidate_path.exists()
    assert candidate_path.parent.name == "efe_candidates"
    assert candidate_path.name.startswith(gap_id + "_")


def test_no_exception_if_autogen_throws(tmp_path: Path, monkeypatch) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="throw", mission_id="m-throw")

    def _boom(**_kwargs):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(ajax_core_mod, "autogen_efe_candidate", _boom, raising=True)

    gap_path = core._emit_missing_efe_gap(
        mission,
        reason="missing_efe_final",
        efe_candidate_source_doc={"steps": [{"action": "write_file", "args": {"path": "x.txt"}}]},
    )

    assert isinstance(gap_path, str)
    payload = json.loads(Path(gap_path).read_text(encoding="utf-8"))
    assert payload.get("efe_candidate_status") == "error"
    assert str(payload.get("efe_candidate_reason") or "").startswith("autogen_exception")


def test_candidate_is_deterministic_for_same_input(tmp_path: Path) -> None:
    core = _build_core(tmp_path)
    source_doc = {
        "steps": [
            {
                "id": "task-1",
                "action": "write_file",
                "args": {"path": "same.txt"},
            }
        ]
    }

    first = core._autogen_missing_efe_candidate(gap_id="gap-A", source_doc=source_doc)
    second = core._autogen_missing_efe_candidate(gap_id="gap-B", source_doc=source_doc)

    assert first.get("efe_candidate_status") == "generated"
    assert second.get("efe_candidate_status") == "generated"

    first_payload = json.loads(Path(str(first["efe_candidate_path"])).read_text(encoding="utf-8"))
    second_payload = json.loads(Path(str(second["efe_candidate_path"])).read_text(encoding="utf-8"))
    assert first_payload.get("expected_state") == second_payload.get("expected_state")
