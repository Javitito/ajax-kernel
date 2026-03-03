from __future__ import annotations

import json
import types
from pathlib import Path

from agency.ajax_core import AjaxConfig, AjaxCore


def _build_core(tmp_path: Path) -> AjaxCore:
    core = AjaxCore.__new__(AjaxCore)
    state_dir = tmp_path / "artifacts" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    core.config = AjaxConfig(root_dir=tmp_path, state_dir=state_dir)
    core.root_dir = tmp_path
    core.log = types.SimpleNamespace(warning=lambda *a, **k: None)
    return core


def test_autogen_missing_efe_candidate_creates_path(tmp_path: Path) -> None:
    core = _build_core(tmp_path)

    path = core._autogen_missing_efe_candidate(
        mission_id="m-123",
        source="brain",
        source_doc={
            "steps": [
                {
                    "id": "task-1",
                    "action": "write_file",
                    "args": {"path": "artifact.txt"},
                }
            ]
        },
    )

    assert isinstance(path, str)
    candidate_path = Path(path)
    assert candidate_path.exists()
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    expected = payload.get("expected_state") or {}
    assert isinstance(expected.get("checks"), list)


def test_build_missing_efe_plan_carries_candidate_path(tmp_path: Path) -> None:
    core = _build_core(tmp_path)

    plan = core._build_missing_efe_plan(
        intention="test missing efe",
        source="brain",
        receipt_path="artifacts/receipts/missing_efe_final_1.json",
        efe_candidate_path="artifacts/efe_candidates/candidate_1.json",
        reason="missing_efe_final",
        errors=["e1"],
    )

    assert isinstance(plan.metadata, dict)
    assert plan.metadata.get("planning_error") == "missing_efe_final"
    assert plan.metadata.get("efe_candidate_path") == "artifacts/efe_candidates/candidate_1.json"
