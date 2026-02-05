# agency/vision.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Protocol

import logging

from PIL import Image

from .safe_pyautogui import safe_screenshot, safe_screen_size, has_real_gui



log = logging.getLogger(__name__)



# Types simples para que Codex luego afine

Coords = tuple[int, int]

ImageLike = Image.Image  # placeholder; luego se sustituye por PIL.Image, np.ndarray, etc.





class VisionError(Exception):

    """Base error for vision subsystem."""





class ElementNotFoundError(VisionError):

    """UI element could not be found."""





@dataclass

class VisionConfig:

    """Config básica; luego se puede cargar de TOML/YAML."""



    screenshot_backend: str = 'pyautogui'  # o "mss", "dxcam", etc.





class Vision(Protocol):

    """

    Protocolo de la interfaz de visión.



    Implementaciones concretas pueden vivir aquí mismo (VisionImpl) o aparte.

    """



    def capture(self) -> ImageLike: ...



    def find_fast(self, goal: str, screen: ImageLike) -> Coords: ...



    def find_cognitive(self, goal: str, screen: ImageLike) -> Coords: ...





class VisionImpl:

    """

    Implementación mínima (MVP).



    Fast Path: aún sin lógica real.

    Cognitive Path: hook a LLM vision (Gemini/Qwen-VL) cuando lo conectemos.

    """



    def __init__(self, config: VisionConfig | None = None) -> None:

        self._config = config or VisionConfig()



    # --------- Captura de pantalla ---------

    def capture(self) -> ImageLike:

        log.debug('Vision.capture using backend=%s', self._config.screenshot_backend)
        # Usar la función segura que maneja entornos headless
        return safe_screenshot()



    # --------- Fast Path (Balístico) ---------

    def find_fast(self, goal: str, screen: ImageLike) -> Coords:

        """

        Modo balístico: locateOnScreen / OCR local / templates.



        Por ahora solo stub; Codex implementará.

        """
        log.info("Vision.find_fast goal=%r", goal)
        raise ElementNotFoundError(f"Fast path not implemented for goal={goal!r}")

    # --------- Cognitive Path (Deliberado) ---------
    def find_cognitive(self, goal: str, screen: ImageLike) -> Coords:
        """
        Modo deliberado: llamadas a Gemini/Qwen-VL vía tus bin/*_task.py.

        Aquí encaja la integración con bin/gemini_task.py, etc.
        """
        log.info("Vision.find_cognitive goal=%r", goal)
        # TODO: guardar screenshot en archivo temporal, pasarlo a bin/gemini_task.py
        raise ElementNotFoundError(f"Cognitive path not implemented for goal={goal!r}")

    def get_screen_size(self) -> Coords:
        """
        Obtiene el tamaño actual de la pantalla de forma segura.

        Returns:
            Una tupla con (ancho, alto) de la pantalla
        """
        return safe_screen_size()
