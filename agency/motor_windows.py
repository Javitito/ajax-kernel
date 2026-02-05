# agency/motor_windows.py
from __future__ import annotations

from dataclasses import dataclass
import logging

from .windows_driver_client import WindowsDriverClient, WindowsDriverError

log = logging.getLogger(__name__)


@dataclass
class MotorWindowsConfig:
    base_url: str = "http://127.0.0.1:5010"
    move_duration: float = 0.1


class MotorWindows:
    """
    Backend Motor para entornos Windows usando el microservicio drivers/os_driver.py.
    """

    def __init__(self, config: MotorWindowsConfig | None = None, client: WindowsDriverClient | None = None) -> None:
        self._config = config or MotorWindowsConfig()
        self._client = client or WindowsDriverClient(base_url=self._config.base_url)

    def move(self, coords: tuple[int, int]) -> None:
        log.info("MotorWindows.move coords=%s", coords)
        self._client.mouse_move(coords[0], coords[1], duration=self._config.move_duration)

    def click(self, button: str = "left") -> None:
        log.info("MotorWindows.click button=%s", button)
        self._client.mouse_click(button=button)

    def type(self, text: str) -> None:
        log.info("MotorWindows.type text=%r", text)
        self._client.keyboard_type(text)
