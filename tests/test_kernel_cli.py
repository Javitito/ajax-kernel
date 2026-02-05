import json
import subprocess
import sys


def _extract_first_json_object(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise AssertionError("no json object found in output")
    depth = 0
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise AssertionError("unterminated json object in output")
    return json.loads(text[start:end])


def test_ajaxctl_help_exits_zero():
    proc = subprocess.run([sys.executable, "bin/ajaxctl", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0


def test_ajaxctl_health_json_ok():
    proc = subprocess.run([sys.executable, "bin/ajaxctl", "health"], capture_output=True, text=True)
    assert proc.returncode == 0
    data = _extract_first_json_object(proc.stdout)
    assert data.get("ok") is True
