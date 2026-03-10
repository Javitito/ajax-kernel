from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_git_does_not_track_absolute_windows_paths() -> None:
    git = shutil.which("git")
    if not git:
        pytest.skip("git not available")

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [git, "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    tracked = [item for item in proc.stdout.decode("utf-8", errors="replace").split("\0") if item]
    bad = [item for item in tracked if len(item) >= 3 and item[1:3] in {":\\", ":/"}]
    assert bad == [], f"Absolute Windows drive paths must not be tracked: {bad}"
