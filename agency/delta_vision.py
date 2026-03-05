"""
Delta-vision helpers: detect visual changes on the desktop by hashing tiles.

Fase 1 (básica):
- Divide frames en tiles.
- Calcula hashes (pantalla y tile) con phash cuando está disponible.
- Detecta tiles cambiados comparando hashes previos con un umbral de distancia.
- Aplica compresión temporal: un tile se considera "cambiado" sólo si el cambio persiste
  más de `min_persist_ms`.
- Memoria visual: guarda hashes vistos por tile para poder degradar prioridad/ignorar
  en fases posteriores (aquí sólo se actualiza la memoria).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

if Image is not None:  # pragma: no branch
    _RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
else:  # pragma: no cover
    _RESAMPLE = None


def _now() -> float:
    return time.time()


def compute_tile_hash(image: "Image.Image") -> int:
    """dHash 64-bit: grayscale -> 9x8 -> diffs horizontales."""
    if Image is None:  # pragma: no cover
        raise RuntimeError("pillow_not_available")
    gray = image.convert("L").resize((9, 8), resample=_RESAMPLE)
    px = list(gray.getdata())
    value = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            left = int(px[base + col])
            right = int(px[base + col + 1])
            bit = 1 if left > right else 0
            value = (value << 1) | bit
    return int(value)


def hamming_distance(first: int, second: int) -> int:
    return (int(first) ^ int(second)).bit_count()


def tile_changed(previous_hash: Optional[int], current_hash: int, threshold: int = 6) -> bool:
    if previous_hash is None:
        return True
    thr = max(0, int(threshold))
    return hamming_distance(int(previous_hash), int(current_hash)) > thr


@dataclass
class DeltaVisionState:
    screen_size: Tuple[int, int]
    tile_size: int = 256
    tiles_per_row: int = 0
    tiles_per_col: int = 0
    last_frame_hash: Optional[int] = None
    last_tile_hashes: Dict[int, int] = field(default_factory=dict)
    known_tile_hashes: Dict[int, set[int]] = field(default_factory=dict)
    last_change_ts: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def init(cls, screen_size: Tuple[int, int], tile_size: int = 256) -> "DeltaVisionState":
        w, h = screen_size
        tpr = max(1, (w + tile_size - 1) // tile_size)
        tpc = max(1, (h + tile_size - 1) // tile_size)
        return cls(screen_size=screen_size, tile_size=tile_size, tiles_per_row=tpr, tiles_per_col=tpc)


def compute_screen_hash(frame: "Image.Image") -> int:
    return compute_tile_hash(frame)


def divide_into_tiles(frame: "Image.Image", tile_size: int) -> Dict[int, "Image.Image"]:
    w, h = frame.size
    tiles: Dict[int, Image.Image] = {}
    cols = max(1, (w + tile_size - 1) // tile_size)
    rows = max(1, (h + tile_size - 1) // tile_size)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            left = c * tile_size
            upper = r * tile_size
            right = min(left + tile_size, w)
            lower = min(upper + tile_size, h)
            tiles[idx] = frame.crop((left, upper, right, lower))
            idx += 1
    return tiles


def _trim_border(img: "Image.Image", border: int = 1) -> "Image.Image":
    """Recorta un borde fino para evitar que un píxel en el límite contamine tiles vecinos."""
    if img.width <= 2 * border or img.height <= 2 * border:
        return img
    return img.crop((border, border, img.width - border, img.height - border))


def detect_changed_tiles(
    state: DeltaVisionState,
    frame: "Image.Image",
    tiles: Dict[int, "Image.Image"],
    threshold: int = 6,
) -> List[int]:
    changed: List[int] = []
    for idx, img in tiles.items():
        img_for_hash = _trim_border(img)
        h_new = compute_tile_hash(img_for_hash)
        h_prev = state.last_tile_hashes.get(idx)
        if tile_changed(h_prev, h_new, threshold=threshold):
            changed.append(idx)
        state.last_tile_hashes[idx] = h_new
    state.last_frame_hash = compute_screen_hash(frame)
    return changed


def get_changed_tiles(
    state: DeltaVisionState,
    frame: "Image.Image",
    threshold: int = 6,
    min_persist_ms: int = 800,
    now_ts: Optional[float] = None,
) -> List[Tuple[int, "Image.Image"]]:
    """
    Devuelve tiles cambiados que han persistido al menos min_persist_ms.
    Actualiza el estado (hashes y memoria visual) internamente.
    """
    now = now_ts if now_ts is not None else _now()
    tiles = divide_into_tiles(frame, state.tile_size)
    changed_idxs = detect_changed_tiles(state, frame, tiles, threshold=threshold)
    out: List[Tuple[int, Image.Image]] = []
    for idx in changed_idxs:
        h = state.last_tile_hashes.get(idx)
        if h is None:
            continue
        first_seen = state.last_change_ts.get(idx)
        if first_seen is None:
            state.last_change_ts[idx] = now
            continue
        if (now - first_seen) * 1000.0 < min_persist_ms:
            continue
        # Si hash ya está en memoria, se degrada; aquí solo actualizamos memoria
        known = state.known_tile_hashes.setdefault(idx, set())
        if h in known:
            # cambio ya conocido: de momento lo consideramos, se podría bajar prioridad en fases posteriores
            pass
        known.add(h)
        out.append((idx, tiles[idx]))
        state.last_change_ts[idx] = now
    return out
