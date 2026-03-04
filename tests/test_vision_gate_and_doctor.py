from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from agency.vision_gate import (
    ensure_local_vision_allowed,
    is_local_vision_allowed,
    select_local_vision_provider,
    set_local_vision_allowed_for_process,
)
from agency.vision_tag_screen import run_doctor_vision, tag_screen_with_delta


class _FakeDriver:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def tag_screen_grid(self, rows: int = 4, cols: int = 4) -> Dict[str, Any]:
        return dict(self._payload)


def _write_two_tiles(path: Path) -> None:
    img = Image.new("RGB", (100, 50), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 49, 49], fill=(0, 0, 0))
    img.save(path)


def _payload(image_path: Path) -> Dict[str, Any]:
    return {
        "image_path": str(image_path),
        "marks": [
            {"id": "A1", "rect": {"x": 0, "y": 0, "width": 50, "height": 50}},
            {"id": "B1", "rect": {"x": 50, "y": 0, "width": 50, "height": 50}},
        ],
    }


def _providers_doc(*, lmstudio_up: bool = True) -> Dict[str, Any]:
    state = "UP" if lmstudio_up else "DOWN"
    reason = "" if lmstudio_up else "transport_down"
    return {
        "schema": "ajax.providers_status.v1",
        "providers": {
            "other_vision": {
                "capabilities": {"vision_local": True},
                "transport": {"status": "UP", "reason": ""},
                "breathing": {"roles": {"vision": {"status": "UP", "reason": ""}}, "status": "UP"},
                "unavailable_reason": "",
            },
            "lmstudio_vision": {
                "capabilities": {"vision_local": True},
                "transport": {"status": state, "reason": reason},
                "breathing": {"roles": {"vision": {"status": state, "reason": reason}}, "status": state},
                "unavailable_reason": reason,
            },
        },
    }


def _write_providers_status(root_dir: Path, payload: Dict[str, Any]) -> None:
    target = root_dir / "artifacts" / "health" / "providers_status.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_gate_blocks_when_local_not_allowed_with_hint(monkeypatch) -> None:
    monkeypatch.delenv("VISION_LOCAL_ALLOWED", raising=False)
    set_local_vision_allowed_for_process(False)
    try:
        ensure_local_vision_allowed("tag_screen_grid")
        assert False, "expected PermissionError"
    except PermissionError as exc:
        message = str(exc)
        assert "set VISION_LOCAL_ALLOWED=true" in message
        assert "vision tag-screen --allow-local" in message


def test_gate_allows_when_env_true(monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    set_local_vision_allowed_for_process(False)
    ensure_local_vision_allowed("tag_screen_grid")
    assert is_local_vision_allowed() is True


def test_cli_allow_local_overrides_env(monkeypatch) -> None:
    monkeypatch.delenv("VISION_LOCAL_ALLOWED", raising=False)
    set_local_vision_allowed_for_process(False)
    assert is_local_vision_allowed() is False
    set_local_vision_allowed_for_process(True)
    ensure_local_vision_allowed("tag_screen_grid")
    assert is_local_vision_allowed() is True
    set_local_vision_allowed_for_process(False)


def test_doctor_vision_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "doctor", "vision", "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload.get("schema") == "ajax.doctor.vision.v1"


def test_doctor_vision_reports_bypass_false(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    _write_providers_status(tmp_path, _providers_doc(lmstudio_up=True))
    payload = run_doctor_vision(root_dir=tmp_path)
    assert payload.get("bypass_detected") is False


def test_provider_selection_prefers_lmstudio_vision_when_up() -> None:
    selected = select_local_vision_provider(providers_status=_providers_doc(lmstudio_up=True))
    assert selected.get("provider") == "lmstudio_vision"
    assert selected.get("up") is True


def test_delta_run_includes_provider_used_when_allowed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    _write_providers_status(tmp_path, _providers_doc(lmstudio_up=True))
    image_path = tmp_path / "snap.png"
    _write_two_tiles(image_path)
    payload = tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=lambda _p, _b: ("ok", 0.9),
        cache_path=tmp_path / "delta_cache.json",
        root_dir=tmp_path,
    )
    assert payload["delta_run"]["provider_used"] == "lmstudio_vision"


def test_no_network_calls_in_tests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    _write_providers_status(tmp_path, _providers_doc(lmstudio_up=True))

    def _forbid(*_args, **_kwargs):
        raise AssertionError("network_call_forbidden_in_test")

    monkeypatch.setattr("requests.get", _forbid, raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _forbid, raising=False)

    image_path = tmp_path / "snap.png"
    _write_two_tiles(image_path)
    _ = run_doctor_vision(root_dir=tmp_path)
    _ = select_local_vision_provider(root_dir=tmp_path)
    _ = tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=lambda _p, _b: ("ok", 0.8),
        cache_path=tmp_path / "delta_cache.json",
        root_dir=tmp_path,
    )
