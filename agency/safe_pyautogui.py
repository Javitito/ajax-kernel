"""
Wrapper seguro para operaciones GUI que maneja entornos headless
"""
import os
from typing import Tuple, Optional

# Importar funciones desde alt_gui_ops
from .alt_gui_ops import (
    safe_screenshot as alt_safe_screenshot,
    safe_mouse_move as alt_safe_mouse_move,
    safe_mouse_click as alt_safe_mouse_click,
    safe_keyboard_write as alt_safe_keyboard_write,
    safe_screen_size as alt_safe_screen_size,
    has_real_gui
)


# Funciones wrapper para operaciones de GUI
def safe_screenshot(region: Optional[Tuple[int, int, int, int]] = None):
    """
    Toma una captura de pantalla de forma segura en entornos headless
    """
    return alt_safe_screenshot(region)


def safe_mouse_move(x: int, y: int, duration: float = 0.0):
    """
    Mueve el ratón de forma segura en entornos headless
    """
    return alt_safe_mouse_move(x, y, duration)


def safe_mouse_click(button: str = 'left'):
    """
    Hace clic con el ratón de forma segura en entornos headless
    """
    return alt_safe_mouse_click(button)


def safe_keyboard_write(text: str):
    """
    Escribe texto de forma segura en entornos headless
    """
    return alt_safe_keyboard_write(text)


def safe_screen_size():
    """
    Obtiene el tamaño de la pantalla de forma segura en entornos headless
    """
    return alt_safe_screen_size()