from __future__ import annotations

from pathlib import Path

from PIL import Image

from agency import delta_vision


def _gradient_image(size: int = 64, reverse: bool = False) -> Image.Image:
    img = Image.new("L", (size, size))
    for y in range(size):
        for x in range(size):
            value = x if not reverse else (255 - x)
            img.putpixel((x, y), int(max(0, min(255, value))))
    return img.convert("RGB")


def test_dhash_same_image_same_hash() -> None:
    img = _gradient_image()
    h1 = delta_vision.compute_tile_hash(img)
    h2 = delta_vision.compute_tile_hash(img.copy())
    assert h1 == h2


def test_dhash_small_change_low_hamming() -> None:
    base = _gradient_image()
    mutated = base.copy()
    mutated.putpixel((10, 10), (base.getpixel((10, 10))[0] + 1, 0, 0))
    h_base = delta_vision.compute_tile_hash(base)
    h_mutated = delta_vision.compute_tile_hash(mutated)
    dist = delta_vision.hamming_distance(h_base, h_mutated)
    assert dist <= 6


def test_dhash_big_change_high_hamming() -> None:
    left_to_right = _gradient_image(reverse=False)
    right_to_left = _gradient_image(reverse=True)
    h1 = delta_vision.compute_tile_hash(left_to_right)
    h2 = delta_vision.compute_tile_hash(right_to_left)
    assert delta_vision.hamming_distance(h1, h2) >= 32


def test_tile_changed_threshold() -> None:
    h1 = delta_vision.compute_tile_hash(_gradient_image(reverse=False))
    h2 = delta_vision.compute_tile_hash(_gradient_image(reverse=True))
    assert delta_vision.tile_changed(h1, h2, threshold=6) is True
    assert delta_vision.tile_changed(h1, h1, threshold=6) is False
    assert delta_vision.tile_changed(None, h1, threshold=6) is True


def test_hash_stable_types() -> None:
    rgb = _gradient_image()
    gray = rgb.convert("L")
    assert delta_vision.compute_tile_hash(rgb) == delta_vision.compute_tile_hash(gray)


def test_no_md5_in_delta_vision() -> None:
    source = Path(delta_vision.__file__).read_text(encoding="utf-8").lower()
    assert "md5" not in source
