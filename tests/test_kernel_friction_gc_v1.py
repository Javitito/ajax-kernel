from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agency.friction import run_friction_gc
from agency.metabolism_doctor import run_doctor_metabolism


def _touch_with_age(path: Path, *, age_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    ts = time.time() - age_seconds
    os.utime(path, (ts, ts))


def test_gc_dry_run_no_changes(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    _touch_with_age(waiting_old, age_seconds=48 * 3600)

    result = run_friction_gc(root_dir=tmp_path, apply=False, older_than_hours=24.0)
    waiting = result["waiting_for_user"]

    assert result["mode"] == "dry_run"
    assert waiting_old.exists()
    assert waiting["count"] == 1
    assert waiting["archived"] == []
    assert float(waiting["oldest_age_hours"]) >= 47.0
    target_archive = Path(str(waiting["target_archive_path"]))
    assert target_archive.name == time.strftime("%Y-%m-%d", time.gmtime())
    assert target_archive.parent.name == "_archived"


def test_gc_apply_archives_old_waiting(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    waiting_recent = tmp_path / "artifacts" / "waiting_for_user" / "recent.json"
    _touch_with_age(waiting_old, age_seconds=48 * 3600)
    _touch_with_age(waiting_recent, age_seconds=60)

    result = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=24.0)
    waiting = result["waiting_for_user"]

    assert waiting_old.exists() is False
    assert waiting_recent.exists() is True
    assert waiting["count"] == 1
    assert len(waiting["archived"]) == 1
    archived_path = Path(waiting["archived"][0])
    assert archived_path.exists()
    assert "_archived" in archived_path.parts


def test_gc_idempotent(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    _touch_with_age(waiting_old, age_seconds=48 * 3600)

    first = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=24.0)
    second = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=24.0)

    assert first["waiting_for_user"]["count"] == 1
    assert len(first["waiting_for_user"]["archived"]) == 1
    assert second["waiting_for_user"]["count"] == 0
    assert second["waiting_for_user"]["archived"] == []
    assert second["waiting_for_user"]["candidates"] == []


def test_gc_default_older_than_24h(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    waiting_recent = tmp_path / "artifacts" / "waiting_for_user" / "recent.json"
    _touch_with_age(waiting_old, age_seconds=25 * 3600)
    _touch_with_age(waiting_recent, age_seconds=23 * 3600)

    result = run_friction_gc(root_dir=tmp_path, apply=False)
    waiting = result["waiting_for_user"]

    assert waiting["threshold_hours"] == 24.0
    assert waiting["count"] == 1
    assert waiting["skipped_recent"] == 1
    assert waiting["candidates"] == [str(waiting_old)]


def test_gc_receipt_written(tmp_path: Path) -> None:
    result = run_friction_gc(root_dir=tmp_path, apply=False, older_than_hours=24.0)
    receipt = Path(result["receipt_path"])

    assert receipt.exists()
    assert receipt.name.startswith("friction_gc_v1_")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload.get("schema") == "ajax.ops.friction_gc.v1"


def test_metabolism_hint_when_zombies_present(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "zombie.json"
    _touch_with_age(waiting_old, age_seconds=48 * 3600)

    payload = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)
    hints = payload.get("next_hint") or []

    assert payload["waiting_backlog"]["count"] == 1
    assert any("ops friction gc --dry-run" in str(h) for h in hints)
