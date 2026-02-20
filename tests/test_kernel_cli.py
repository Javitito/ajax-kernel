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


def test_ajaxctl_microfilm_check_generates_report():
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "microfilm", "check", "--root", "."],
        capture_output=True,
        text=True,
    )
    # FAIL (2) is valid in fail-closed environments; PASS (0) is also valid.
    assert proc.returncode in {0, 2}
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "Microfilm Compliance v1:" in out
