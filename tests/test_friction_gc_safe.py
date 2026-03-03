from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agency.friction import run_friction_gc


def _touch_with_age(path: Path, *, age_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    ts = time.time() - age_seconds
    os.utime(path, (ts, ts))


def _write_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ajax.provider_ledger.v1",
        "updated_ts": time.time(),
        "updated_utc": "2026-03-03T00:00:00Z",
        "rows": [
            {"provider": "groq", "role": "brain", "status": "ok", "details": {}},
            {"provider": "lmstudio", "role": "brain", "status": "ok", "details": {}},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_friction_gc_dry_run_reports_without_moving(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    _touch_with_age(waiting_old, age_seconds=7200)

    result = run_friction_gc(root_dir=tmp_path, apply=False, older_than_hours=1.0)

    assert result["mode"] == "dry_run"
    assert waiting_old.exists()
    assert result["waiting_for_user"]["candidates"] == [str(waiting_old)]
    assert result["waiting_for_user"]["archived"] == []


def test_friction_gc_apply_moves_old_waiting_files(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    waiting_recent = tmp_path / "artifacts" / "waiting_for_user" / "recent.json"
    _touch_with_age(waiting_old, age_seconds=7200)
    _touch_with_age(waiting_recent, age_seconds=60)

    result = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=1.0)

    assert result["mode"] == "apply"
    assert waiting_old.exists() is False
    assert waiting_recent.exists() is True
    archived = result["waiting_for_user"]["archived"]
    assert len(archived) == 1
    assert Path(archived[0]).exists()
    assert Path(result["receipt_path"]).exists()


def test_friction_gc_dry_run_does_not_modify_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "artifacts" / "provider_ledger" / "latest.json"
    _write_ledger(ledger)
    before = ledger.read_text(encoding="utf-8")

    result = run_friction_gc(root_dir=tmp_path, apply=False, older_than_hours=1.0)

    after = ledger.read_text(encoding="utf-8")
    assert before == after
    assert result["provider_ledger"]["applied"] is False
    assert (tmp_path / "artifacts" / "provider_ledger" / "_snapshots").exists() is False


def test_friction_gc_apply_resets_ledger_to_minimum_budget_mode(tmp_path: Path) -> None:
    ledger = tmp_path / "artifacts" / "provider_ledger" / "latest.json"
    _write_ledger(ledger)

    result = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=1.0)

    updated = json.loads(ledger.read_text(encoding="utf-8"))
    rows = updated.get("rows") or []
    groq = next(r for r in rows if r.get("provider") == "groq")
    lmstudio = next(r for r in rows if r.get("provider") == "lmstudio")

    assert updated.get("mode") == "minimum_budget"
    assert groq.get("status") == "soft_fail"
    assert groq.get("reason") == "minimum_budget_mode"
    assert lmstudio.get("status") == "ok"
    assert Path(result["provider_ledger"]["snapshot_path"]).exists()


def test_friction_gc_apply_idempotent_second_run(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    _touch_with_age(waiting_old, age_seconds=7200)

    first = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=1.0)
    second = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=1.0)

    assert len(first["waiting_for_user"]["archived"]) == 1
    assert second["waiting_for_user"]["archived"] == []
    assert second["waiting_for_user"]["candidates"] == []
    assert Path(second["receipt_path"]).exists()


def test_friction_gc_apply_handles_missing_ledger(tmp_path: Path) -> None:
    result = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=1.0)

    assert result["provider_ledger"]["ledger_exists"] is False
    assert result["provider_ledger"]["applied"] is False
    assert Path(result["receipt_path"]).exists()
