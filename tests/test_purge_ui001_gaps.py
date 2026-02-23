from __future__ import annotations

import json
from pathlib import Path

from agency.ui001_purge import purge_ui001_gaps


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_purge_ui001_moves_files_and_generates_index_and_report(tmp_path: Path) -> None:
    _write(tmp_path / "artifacts" / "gaps" / "UI-001" / "a.json", '{"kind":"a"}')
    _write(tmp_path / "artifacts" / "capability_gaps" / "trace_UI-001.json", '{"kind":"b"}')
    _write(tmp_path / "gaps" / "UI-001" / "legacy.txt", "legacy")
    _write(tmp_path / "artifacts" / "capability_gaps" / "other.json", '{"kind":"other"}')

    summary = purge_ui001_gaps(tmp_path, timestamp="20260221T120000Z")
    assert summary.get("ok") is True
    assert summary.get("moved_count") == 3

    attic_dir = tmp_path / "artifacts" / "attic" / "purge_UI-001_20260221T120000Z"
    index_path = attic_dir / "INDEX.json"
    report_path = tmp_path / "artifacts" / "reports" / "purge_UI-001_20260221T120000Z.md"
    assert index_path.exists()
    assert report_path.exists()

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index.get("total_count") == 3
    assert index.get("total_bytes", 0) > 0
    assert len(index.get("files") or []) == 3
    for row in index.get("files") or []:
        assert row.get("source_path")
        assert row.get("attic_path")
        assert row.get("size_bytes", 0) >= 0
        assert row.get("sha256")

    assert not (tmp_path / "artifacts" / "gaps" / "UI-001" / "a.json").exists()
    assert not (tmp_path / "artifacts" / "capability_gaps" / "trace_UI-001.json").exists()
    assert not (tmp_path / "gaps" / "UI-001" / "legacy.txt").exists()
    assert (tmp_path / "artifacts" / "capability_gaps" / "other.json").exists()


def test_purge_ui001_is_idempotent_when_no_files_left(tmp_path: Path) -> None:
    _write(tmp_path / "artifacts" / "capability_gaps" / "once_UI-001.json", '{"kind":"once"}')

    first = purge_ui001_gaps(tmp_path, timestamp="20260221T120100Z")
    second = purge_ui001_gaps(tmp_path, timestamp="20260221T120200Z")

    assert first.get("moved_count") == 1
    assert second.get("moved_count") == 0

    first_attic = tmp_path / "artifacts" / "attic" / "purge_UI-001_20260221T120100Z"
    assert (first_attic / "artifacts" / "capability_gaps" / "once_UI-001.json").exists()
    second_index = tmp_path / "artifacts" / "attic" / "purge_UI-001_20260221T120200Z" / "INDEX.json"
    second_data = json.loads(second_index.read_text(encoding="utf-8"))
    assert second_data.get("total_count") == 0
