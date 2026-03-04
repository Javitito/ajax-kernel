from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

import agency.vision_tag_screen as vision_tag_screen


class _FakeDriver:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def tag_screen_grid(self, rows: int = 4, cols: int = 4) -> Dict[str, Any]:
        return dict(self._payload)


def _marks() -> List[Dict[str, Any]]:
    return [
        {"id": "A1", "rect": {"x": 0, "y": 0, "width": 50, "height": 50}},
        {"id": "B1", "rect": {"x": 50, "y": 0, "width": 50, "height": 50}},
    ]


def _write_two_tiles(path: Path, left: Tuple[int, int, int], right: Tuple[int, int, int]) -> None:
    img = Image.new("RGB", (100, 50), right)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 49, 49], fill=left)
    img.save(path)


def _write_two_tiles_gradient(path: Path, *, left_reverse: bool, right_reverse: bool) -> None:
    img = Image.new("RGB", (100, 50))
    for y in range(50):
        for x in range(100):
            local = x if x < 50 else (x - 50)
            value = int((local / 49.0) * 255.0)
            if x < 50 and left_reverse:
                value = 255 - value
            if x >= 50 and right_reverse:
                value = 255 - value
            img.putpixel((x, y), (value, value, value))
    img.save(path)


def _payload(image_path: Path) -> Dict[str, Any]:
    return {"image_path": str(image_path), "marks": _marks()}


def test_no_ocr_when_no_tiles_changed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    image_path = tmp_path / "snap.png"
    _write_two_tiles(image_path, (0, 0, 0), (255, 255, 255))
    cache_path = tmp_path / "delta_cache.json"
    root_dir = tmp_path

    def ocr_seed(_path: Path, bbox: List[int]) -> Tuple[str, float]:
        return f"seed-{bbox[0]}", 0.9

    vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=ocr_seed,
        cache_path=cache_path,
        root_dir=root_dir,
    )

    calls = {"n": 0}

    def ocr_count(_path: Path, _bbox: List[int]) -> Tuple[str, float]:
        calls["n"] += 1
        return "should-not-run", 0.1

    second = vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=ocr_count,
        cache_path=cache_path,
        root_dir=root_dir,
    )
    assert calls["n"] == 0
    assert second["delta_run"]["tiles_changed"] == 0
    assert second["delta_run"]["tiles_skipped"] == 2


def test_ocr_only_for_changed_tiles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_two_tiles_gradient(image_a, left_reverse=False, right_reverse=False)
    _write_two_tiles_gradient(image_b, left_reverse=True, right_reverse=False)
    cache_path = tmp_path / "delta_cache.json"
    root_dir = tmp_path

    vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_a)),
        ocr_fn=lambda _p, _b: ("seed", 0.9),
        cache_path=cache_path,
        root_dir=root_dir,
    )

    calls = {"n": 0}

    def ocr_count(_path: Path, _bbox: List[int]) -> Tuple[str, float]:
        calls["n"] += 1
        return "changed", 0.8

    second = vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_b)),
        ocr_fn=ocr_count,
        cache_path=cache_path,
        root_dir=root_dir,
    )
    assert calls["n"] == 1
    assert second["delta_run"]["tiles_changed"] == 1
    assert second["delta_run"]["tiles_skipped"] == 1


def test_delta_run_artifact_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    image_path = tmp_path / "snap.png"
    _write_two_tiles(image_path, (5, 5, 5), (240, 240, 240))
    payload = vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=lambda _p, _b: ("text", 0.7),
        cache_path=tmp_path / "delta_cache.json",
        root_dir=tmp_path,
    )
    delta_path = Path(payload["delta_run_path"])
    assert delta_path.exists()
    doc = json.loads(delta_path.read_text(encoding="utf-8"))
    assert doc["tiles_total"] == 2
    assert doc["hash_kind"] == "dhash64"


def test_threshold_configurable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_two_tiles_gradient(image_a, left_reverse=False, right_reverse=False)
    _write_two_tiles_gradient(image_b, left_reverse=True, right_reverse=True)

    cache_low = tmp_path / "cache_low.json"
    vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_a)),
        ocr_fn=lambda _p, _b: ("seed", 0.5),
        cache_path=cache_low,
        root_dir=tmp_path,
    )
    low = vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_b)),
        ocr_fn=lambda _p, _b: ("changed", 0.8),
        cache_path=cache_low,
        root_dir=tmp_path,
        threshold=6,
    )

    cache_high = tmp_path / "cache_high.json"
    vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_a)),
        ocr_fn=lambda _p, _b: ("seed", 0.5),
        cache_path=cache_high,
        root_dir=tmp_path,
    )
    high = vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_b)),
        ocr_fn=lambda _p, _b: ("changed", 0.8),
        cache_path=cache_high,
        root_dir=tmp_path,
        threshold=80,
    )

    assert low["delta_run"]["threshold"] == 6
    assert high["delta_run"]["threshold"] == 80
    assert low["delta_run"]["tiles_changed"] > high["delta_run"]["tiles_changed"]


def test_cache_reuse_or_unchanged_marker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    image_path = tmp_path / "snap.png"
    _write_two_tiles(image_path, (10, 10, 10), (240, 240, 240))
    cache_path = tmp_path / "delta_cache.json"
    root_dir = tmp_path

    vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=lambda _p, bbox: (f"cached-{bbox[0]}", 0.9),
        cache_path=cache_path,
        root_dir=root_dir,
    )
    second = vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=lambda _p, _b: ("nope", 0.0),
        cache_path=cache_path,
        root_dir=root_dir,
    )
    states = {m.get("delta_state") for m in second["marks"]}
    assert "UNCHANGED_REUSED" in states
    assert any((m.get("text") or "").startswith("cached-") for m in second["marks"])


def test_unchanged_marker_without_cache_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    image_path = tmp_path / "snap.png"
    _write_two_tiles(image_path, (20, 20, 20), (230, 230, 230))
    cache_path = tmp_path / "delta_cache_empty.json"
    root_dir = tmp_path

    vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=lambda _p, _b: ("", 0.0),
        cache_path=cache_path,
        root_dir=root_dir,
    )
    second = vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=lambda _p, _b: ("", 0.0),
        cache_path=cache_path,
        root_dir=root_dir,
    )
    assert all(m.get("delta_state") == "UNCHANGED" for m in second["marks"])
    assert all(m.get("ocr_status") == "UNCHANGED" for m in second["marks"])


def test_bypass_removed_guard() -> None:
    source = Path(vision_tag_screen.__file__).read_text(encoding="utf-8")
    assert "agency.delta_vision" in source
    assert "tile_changed" in source


def test_provider_used_is_recorded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISION_LOCAL_ALLOWED", "true")
    image_path = tmp_path / "snap.png"
    _write_two_tiles(image_path, (0, 0, 0), (255, 255, 255))
    payload = vision_tag_screen.tag_screen_with_delta(
        driver_client=_FakeDriver(_payload(image_path)),
        ocr_fn=lambda _p, _b: ("x", 0.1),
        cache_path=tmp_path / "delta_cache.json",
        root_dir=tmp_path,
        provider_used="lmstudio_vision",
    )
    assert payload["delta_run"]["provider_used"] == "lmstudio_vision"
