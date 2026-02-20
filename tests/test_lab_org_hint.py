from __future__ import annotations

from pathlib import Path

from agency.lab_org import lab_org_tick


def test_lab_org_not_running_includes_actionable_hint(tmp_path: Path) -> None:
    root = tmp_path / "ajax-kernel"
    root.mkdir(parents=True, exist_ok=True)
    receipt = lab_org_tick(root)
    assert receipt.get("skipped_reason") == "lab_org_not_running"
    assert receipt.get("actionable_hint") == "Run: python bin/ajaxctl lab start"
