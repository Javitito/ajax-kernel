# agency/motor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import logging

from .safe_pyautogui import safe_mouse_move, safe_mouse_click, safe_keyboard_write, has_real_gui

log = logging.getLogger(__name__)

Coords = Tuple[int, int]


@dataclass
class MotorConfig:
    backend: str = "pyautogui"  # o "pynput", etc.
    move_duration: float = 0.1


class Motor:
    """
    Subsystema motor: ratón + teclado.
    """

    def __init__(self, config: MotorConfig | None = None) -> None:
        self._config = config or MotorConfig()

    def move(self, coords: Coords) -> None:
        log.info("Motor.move to coords=%s backend=%s", coords, self._config.backend)
        # Usar la función segura que maneja entornos headless
        safe_mouse_move(coords[0], coords[1], duration=self._config.move_duration)

    def click(self, button: str = "left") -> None:
        log.info("Motor.click button=%s backend=%s", button, self._config.backend)
        # Usar la función segura que maneja entornos headless
        safe_mouse_click(button=button)

    def type(self, text: str) -> None:
        log.info("Motor.type text=%r", text)
        # Usar la función segura que maneja entornos headless
        safe_keyboard_write(text)
