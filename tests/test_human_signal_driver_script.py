from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import agency.lab_bootstrap as lab_bootstrap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REL = Path("scripts") / "ops" / "get_human_signal.ps1"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _decode_bytes(blob: bytes) -> str:
    if not blob:
        return ""
    for enc in ("utf-8", "utf-16", "utf-16-le", "cp1252", "latin-1"):
        try:
            return blob.decode(enc)
        except Exception:
            continue
    return blob.decode("utf-8", errors="replace")


def _parse_json_from_output(blob: bytes) -> dict:
    text = _decode_bytes(blob).strip()
    start = text.find("{")
    end = text.rfind("}")
    assert start >= 0 and end > start, f"no JSON found in output: {text[:200]!r}"
    return json.loads(text[start : end + 1])


def _wsl_to_windows(path: Path) -> str:
    if os.name == "nt":
        return str(path)
    proc = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, check=False)
    if proc.returncode == 0:
        return _decode_bytes(proc.stdout).strip() or str(path)
    return str(path)


def test_human_signal_driver_script_contains_win32_pinvoke_and_no_stub_markers(tmp_path: Path, monkeypatch) -> None:
    script_path = ROOT / SCRIPT_REL
    text = _read_text(script_path)
    lowered = text.lower()

    assert "getlastinputinfo" in lowered
    assert "dllimport(\"user32.dll\"" in lowered
    assert "add-type" in lowered
    assert "ajax_human_signal_mock" in lowered
    assert "stub_fail_closed" not in lowered
    assert "stub_exception" not in lowered
    assert "stub_" not in lowered

    # Bootstrap should generate the same non-stub driver into a fresh repo root.
    fake_root = tmp_path / "ajax-kernel"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        lab_bootstrap,
        "ensure_lab_display_target",
        lambda *args, **kwargs: {"ok": True, "updated": False, "path": None},
        raising=True,
    )
    res = lab_bootstrap.ensure_lab_bootstrap(fake_root)
    assert res.get("ok") is True
    generated = _read_text(fake_root / SCRIPT_REL).lower()
    assert "getlastinputinfo" in generated
    assert "stub_" not in generated


def test_human_signal_driver_mock_outputs_json_schema() -> None:
    powershell = shutil.which("powershell.exe")
    if not powershell:
        pytest.skip("powershell.exe not available")

    script_path = ROOT / SCRIPT_REL
    env = dict(os.environ)
    env["AJAX_HUMAN_SIGNAL_MOCK"] = "1"
    env["AJAX_HUMAN_SIGNAL_MOCK_IDLE_SECONDS"] = "7"
    env["AJAX_HUMAN_SIGNAL_MOCK_THRESHOLD_SECONDS"] = "90"
    env["AJAX_HUMAN_SIGNAL_MOCK_SESSION_UNLOCKED"] = "1"

    proc = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _wsl_to_windows(script_path),
            "-Mock",
        ],
        capture_output=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, _decode_bytes(proc.stderr)[:300]

    data = _parse_json_from_output(proc.stdout)
    assert data.get("schema") == "ajax.human_signal.v1"
    assert data.get("ok") is True
    assert data.get("source") == "win32:GetLastInputInfo"
    assert data.get("mock") is True
    assert data.get("human_active") is True
    assert isinstance(data.get("idle_seconds"), (int, float))
    assert isinstance(data.get("idle_threshold_seconds"), (int, float))
    assert isinstance(data.get("last_input_age_sec"), (int, float))
    assert data.get("session_unlocked") is True
    assert data.get("stub_detected") is False
    assert isinstance(data.get("ts_utc"), str) and data.get("ts_utc")

