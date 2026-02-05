# agency/vision_windows.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import logging
from pathlib import Path
import re

from PIL import Image

from .vision import Vision, VisionError, ElementNotFoundError, ImageLike, Coords
from .windows_driver_client import (
    WindowsDriverClient,
    WindowsDriverError,
    WindowInspectionError,
)
from .vision_gate import ensure_local_vision_allowed

log = logging.getLogger(__name__)


@dataclass
class VisionWindowsConfig:
    base_url: str = "http://127.0.0.1:5010"


class VisionWindows(Vision):
    """
    Implementación de Visión para Windows apoyada en el driver FastAPI.
    """

    def __init__(self, config: VisionWindowsConfig | None = None, client: WindowsDriverClient | None = None) -> None:
        self._config = config or VisionWindowsConfig()
        self._client = client or WindowsDriverClient(base_url=self._config.base_url)

    def _guard(self, action: str) -> None:
        try:
            ensure_local_vision_allowed(action)
        except PermissionError as exc:
            raise VisionError(str(exc)) from exc

    def capture(self) -> ImageLike:
        self._guard("capture")
        log.debug("VisionWindows.capture")
        snap_path = self._client.screenshot()
        try:
            return Image.open(Path(snap_path)).convert("RGB")
        except Exception as exc:
            raise VisionError(f"cannot open screenshot at {snap_path}: {exc}") from exc

    def find_fast(self, goal: str, screen: ImageLike) -> Coords:
        self._guard("find_fast")
        log.info("VisionWindows.find_fast goal=%r", goal)
        try:
            data = self._client.find_text(goal)
        except WindowsDriverError as exc:
            raise ElementNotFoundError(str(exc)) from exc
        bbox = data.get("bbox") if isinstance(data, dict) else None
        if not bbox:
            raise ElementNotFoundError(f"text {goal!r} not found")
        x = int((bbox["left"] + bbox["right"]) / 2)
        y = int((bbox["top"] + bbox["bottom"]) / 2)
        return (x, y)

    def find_cognitive(self, goal: str, screen: ImageLike) -> Coords:
        self._guard("find_cognitive")
        """
        Por ahora reutiliza el camino local de pywinauto; la integración con LLM visión
        puede implementarse sobre la captura retornada por capture().
        """
        return self.find_fast(goal, screen)

    def find_semantic(self, goal: str) -> Coords | None:
        self._guard("find_semantic")
        """
        Intenta localizar un control accesible cuyo nombre/control_type se parezca al goal.
        """
        keywords = _extract_keywords(goal)
        for kw in keywords:
            try:
                ctrl = self._client.find_control(name=kw)
            except WindowInspectionError as exc:
                log.debug("find_semantic inspection failed: %s", exc)
                return None
            except WindowsDriverError as exc:
                log.debug("find_semantic driver error: %s", exc)
                return None
            if not ctrl:
                continue
            rect = ctrl.get("rect") if isinstance(ctrl, dict) else None
            if not rect:
                continue
            try:
                x = int(rect["x"] + rect["width"] / 2)
                y = int(rect["y"] + rect["height"] / 2)
                return (x, y)
            except Exception:
                continue
        return None

    def get_screen_size(self) -> Tuple[int, int]:
        self._guard("get_screen_size")
        snap = self.capture()
        return (snap.width, snap.height)


def _extract_keywords(goal: str) -> List[str]:
    """
    Extrae palabras clave simples del goal para mapearlas a controles.
    """
    # Prioridad: texto entre comillas
    quoted = re.findall(r"[\"']([^\"']+)[\"']", goal)
    tokens: List[str] = []
    if quoted:
        for q in quoted:
            q = q.strip()
            if q:
                tokens.append(q)
    # Fallback: palabras alfabéticas de longitud >=3
    if not tokens:
        tokens = [t for t in re.findall(r"[\wáéíóúüñÁÉÍÓÚÜÑ]{3,}", goal) if t]
    # Deduplicar preservando orden
    seen = set()
    deduped: List[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        deduped.append(tok)
    return deduped or [goal]
