import subprocess
import sys


def test_compileall_agency_ajax():
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "agency", "ajax"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
