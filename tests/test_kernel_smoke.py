import subprocess
import sys


def test_compileall_agency_ajax():
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "."],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
