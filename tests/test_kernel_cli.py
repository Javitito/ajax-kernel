import subprocess
import sys


def test_ajaxctl_help_exits_zero():
    proc = subprocess.run([sys.executable, "bin/ajaxctl", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0


def test_ajaxctl_whereami_root_override():
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "whereami", "--root", "."], capture_output=True, text=True
    )
    assert proc.returncode == 0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "repo_root:" in out
