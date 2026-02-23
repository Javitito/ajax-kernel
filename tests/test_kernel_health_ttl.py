from __future__ import annotations

import json
import time
from pathlib import Path

from agency.health_ttl import provider_status_ttl


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_provider_status_ttl_marks_missing_as_stale(tmp_path: Path) -> None:
    out = provider_status_ttl(tmp_path, ttl_seconds=120)
    assert out["stale"] is True
    assert out["reason"] == "providers_status_missing"


def test_provider_status_ttl_marks_old_snapshot_as_stale(tmp_path: Path) -> None:
    old_ts = time.time() - 1000
    status = tmp_path / "artifacts" / "health" / "providers_status.json"
    _write_json(status, {"updated_ts": old_ts, "providers": {}})

    out = provider_status_ttl(tmp_path, ttl_seconds=60, now_ts=old_ts + 120)
    assert out["stale"] is True
    assert out["reason"] == "providers_status_stale"


def test_provider_status_ttl_marks_fresh_snapshot_as_green(tmp_path: Path) -> None:
    now = time.time()
    status = tmp_path / "artifacts" / "health" / "providers_status.json"
    _write_json(status, {"updated_ts": now, "providers": {}})

    out = provider_status_ttl(tmp_path, ttl_seconds=300, now_ts=now + 30)
    assert out["stale"] is False
    assert out["reason"] == "providers_status_fresh"
