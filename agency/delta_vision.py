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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hashlib

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover - imagehash es opcional
    import imagehash
except Exception:  # pragma: no cover
    imagehash = None  # type: ignore


def _now() -> float:
    return time.time()


def _phash(img: "Image.Image") -> str:
    """
    Hash simple y estable de la imagen.

    Usamos un md5 de la versión RGB. Para reducir sensibilidad a variaciones mínimas
    (como un píxel en el borde entre tiles), el llamador se encarga de recortar un
    pequeño borde antes de pasar la imagen aquí.
    """
    buf = img.convert("RGB").tobytes()
    return hashlib.md5(buf).hexdigest()


def _hamming_distance(h1: str, h2: str) -> int:
    # imagehash devuelve hex string; convertimos a int y calculamos distancia de bits
    try:
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        return (i1 ^ i2).bit_count()
    except Exception:
        # Fallback: distancia por caracteres
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))


@dataclass
class DeltaVisionState:
    screen_size: Tuple[int, int]
    tile_size: int = 256
    tiles_per_row: int = 0
    tiles_per_col: int = 0
    last_frame_hash: Optional[str] = None
    last_tile_hashes: Dict[int, str] = field(default_factory=dict)
    known_tile_hashes: Dict[int, set] = field(default_factory=dict)
    last_change_ts: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def init(cls, screen_size: Tuple[int, int], tile_size: int = 256) -> "DeltaVisionState":
        w, h = screen_size
        tpr = max(1, (w + tile_size - 1) // tile_size)
        tpc = max(1, (h + tile_size - 1) // tile_size)
        return cls(screen_size=screen_size, tile_size=tile_size, tiles_per_row=tpr, tiles_per_col=tpc)


def compute_screen_hash(frame: "Image.Image") -> str:
    return _phash(frame)


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
        h_new = _phash(img_for_hash)
        h_prev = state.last_tile_hashes.get(idx)
        if h_prev is None:
            changed.append(idx)
        else:
            dist = _hamming_distance(h_new, h_prev)
            if dist >= threshold:
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
