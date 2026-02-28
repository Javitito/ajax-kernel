from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "check_pseudocode_map_refs.py"


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root)],
        capture_output=True,
        text=True,
    )


def test_pseudocode_map_ref_check_passes_when_refs_exist(tmp_path: Path) -> None:
    (tmp_path / "PSEUDOCODE_MAP").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)

    (tmp_path / "PSEUDOCODE_MAP" / "01_EXAMPLE.pseudo.md").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "docs" / "ref.md").write_text(
        "See PSEUDOCODE_MAP/01_EXAMPLE.pseudo.md for details.\n", encoding="utf-8"
    )

    proc = _run(tmp_path)
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["missing_count"] == 0


def test_pseudocode_map_ref_check_fails_when_ref_missing(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "ref.md").write_text(
        "Broken ref: PSEUDOCODE_MAP/99_MISSING.pseudo.md\n", encoding="utf-8"
    )

    proc = _run(tmp_path)
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["missing_count"] == 1
    assert "PSEUDOCODE_MAP/99_MISSING.pseudo.md" in payload["missing"]
