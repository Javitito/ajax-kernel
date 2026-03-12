from __future__ import annotations

import json
import time
from pathlib import Path

from agency.crystallization_runtime import (
    emit_crystallization_considered_receipt,
    load_auto_crystallize_flag,
    maybe_auto_crystallize,
)
from agency.receipt_validator import validate_receipt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_governed_run(
    root: Path,
    mission_id: str,
    *,
    rail: str = "lab",
    governed: bool = True,
    success: bool = True,
    intent_class: str = "boundary_completion",
    actions: list[str] | None = None,
) -> None:
    actions = actions or ["driver.click", "driver.type"]
    ts = time.time()
    exec_receipt = root / "artifacts" / "receipts" / f"exec_{mission_id}.json"
    _write_json(
        exec_receipt,
        {
            "schema": "ajax.exec.canary.v1",
            "mission_id": mission_id,
            "success": success,
            "created_at": ts,
        },
    )
    attempt = {
        "mission_id": mission_id,
        "intent": f"Canary intent for {mission_id}",
        "attempt": 1,
        "governance": {"risk_level": "low", "tier": "T2"} if governed else {},
        "plan": {
            "plan_id": f"plan-{mission_id}",
            "timestamp": ts,
            "metadata": {
                "intent_class": intent_class,
                "slots": {"target": "demo"},
            },
            "steps": [{"action": action, "args": {"value": "demo"}} for action in actions],
            "success_spec": ["execution_result.success"],
        },
        "execution_log": [],
        "execution_result": {
            "success": success,
            "artifacts": {"exec_receipt": str(exec_receipt)},
        },
        "success_evaluation": {"success": success},
        "mission_error": {},
        "timestamp": ts,
    }
    history = {
        "mission_id": mission_id,
        "intent_text": f"Canary intent for {mission_id}",
        "mode": "auto",
        "metadata": {},
    }
    snapshot = {
        "mission_id": mission_id,
        "rail": rail,
        "utc": "2026-03-12T00:00:00Z",
    }
    _write_json(root / "artifacts" / "missions" / f"{mission_id}_attempt1.json", attempt)
    _write_json(root / "artifacts" / "history" / f"mission-{mission_id}.json", history)
    _write_json(root / "artifacts" / "missions" / "snapshots" / f"{mission_id}_snapshot0.json", snapshot)


def _assert_receipts_valid(root: Path, *receipt_paths: str | None) -> None:
    for raw_path in receipt_paths:
        assert raw_path
        report = validate_receipt(REPO_ROOT, Path(raw_path))
        assert report["ok"], report


def test_lab_trigger_defaults_on(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AJAX_AUTO_CRYSTALLIZE", raising=False)
    assert load_auto_crystallize_flag(tmp_path / "artifacts" / "state") is True


def test_prod_trigger_is_guarded_and_receipted(tmp_path: Path) -> None:
    _seed_governed_run(tmp_path, "mission-prod-1", rail="prod")
    considered = emit_crystallization_considered_receipt(
        tmp_path,
        mission_id="mission-prod-1",
        trigger="auto",
        rail="prod",
        auto_crystallize_enabled=True,
        waiting_for_user=False,
        plan_id="plan-mission-prod-1",
        plan_source="test",
    )
    result = maybe_auto_crystallize(
        tmp_path,
        mission_id="mission-prod-1",
        rail="prod",
        waiting_for_user=False,
        auto_crystallize_enabled=True,
    )
    assert result["triggered"] is False
    assert result["skip_reason"] == "rail_not_lab"
    assert not (tmp_path / "artifacts" / "episodes").exists()
    _assert_receipts_valid(tmp_path, considered, *(result.get("receipt_paths") or []))


def test_successful_governed_lab_run_creates_episode_without_recipe(tmp_path: Path) -> None:
    _seed_governed_run(tmp_path, "mission-lab-1", rail="lab")
    considered = emit_crystallization_considered_receipt(
        tmp_path,
        mission_id="mission-lab-1",
        trigger="auto",
        rail="lab",
        auto_crystallize_enabled=True,
        waiting_for_user=False,
        plan_id="plan-mission-lab-1",
        plan_source="test",
    )
    result = maybe_auto_crystallize(
        tmp_path,
        mission_id="mission-lab-1",
        rail="lab",
        waiting_for_user=False,
        auto_crystallize_enabled=True,
    )
    assert result["triggered"] is True
    assert result["episode_created"] is True
    assert result["candidate_recipe_created"] is False
    assert result["validation_status"] == "refused"
    episode_doc = json.loads(Path(result["episode_path"]).read_text(encoding="utf-8"))
    assert episode_doc["mission_id"] == "mission-lab-1"
    assert episode_doc["run_id"] == "mission-lab-1:attempt1"
    assert episode_doc["pattern"]["confidence"] == 1.0
    assert not (tmp_path / "artifacts" / "habits").exists()
    _assert_receipts_valid(tmp_path, considered, *(result.get("receipt_paths") or []))


def test_repeated_pattern_creates_candidate_and_validates_without_habit(tmp_path: Path) -> None:
    _seed_governed_run(tmp_path, "mission-lab-1", rail="lab")
    _seed_governed_run(tmp_path, "mission-lab-2", rail="lab")
    considered_one = emit_crystallization_considered_receipt(
        tmp_path,
        mission_id="mission-lab-1",
        trigger="auto",
        rail="lab",
        auto_crystallize_enabled=True,
        waiting_for_user=False,
        plan_id="plan-mission-lab-1",
        plan_source="test",
    )
    result_one = maybe_auto_crystallize(
        tmp_path,
        mission_id="mission-lab-1",
        rail="lab",
        waiting_for_user=False,
        auto_crystallize_enabled=True,
    )
    considered_two = emit_crystallization_considered_receipt(
        tmp_path,
        mission_id="mission-lab-2",
        trigger="auto",
        rail="lab",
        auto_crystallize_enabled=True,
        waiting_for_user=False,
        plan_id="plan-mission-lab-2",
        plan_source="test",
    )
    result_two = maybe_auto_crystallize(
        tmp_path,
        mission_id="mission-lab-2",
        rail="lab",
        waiting_for_user=False,
        auto_crystallize_enabled=True,
    )
    assert result_one["episode_created"] is True
    assert result_one["candidate_recipe_created"] is False
    assert result_two["episode_created"] is True
    assert result_two["candidate_recipe_created"] is True
    assert result_two["validation_status"] == "validated"
    assert result_two["promotion_status"] == "pending_manual_promotion"
    recipe_doc = json.loads(Path(result_two["recipe_path"]).read_text(encoding="utf-8"))
    assert recipe_doc["pattern"]["supporting_episode_count"] == 2
    assert recipe_doc["status"] == "candidate"
    assert not (tmp_path / "artifacts" / "habits").exists()
    _assert_receipts_valid(
        tmp_path,
        considered_one,
        *(result_one.get("receipt_paths") or []),
        considered_two,
        *(result_two.get("receipt_paths") or []),
    )
