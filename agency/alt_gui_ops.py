"""
Implementación alternativa de GUI operations que no depende directamente de pyautogui
"""
import subprocess
import os
import time
from typing import Tuple, Optional
from pathlib import Path
from PIL import Image

from .display_manager import VirtualDisplayManager, has_real_gui


def has_xdotool():
    """Verifica si xdotool está disponible"""
    return subprocess.run(['which', 'xdotool'], capture_output=True).returncode == 0


def has_import_cmd():
    """Verifica si import (de ImageMagick) está disponible"""
    return subprocess.run(['which', 'import'], capture_output=True).returncode == 0


def take_screenshot_cmd(filename: str = '/tmp/screenshot.png', region: Optional[Tuple[int, int, int, int]] = None):
    """Toma una captura de pantalla usando la herramienta import de ImageMagick"""
    if not has_import_cmd():
        raise RuntimeError("La herramienta 'import' de ImageMagick no está disponible")
    
    cmd = ['import', '-window', 'root', filename]
    
    if region:
        x, y, width, height = region
        cmd = ['import', '-crop', f'{width}x{height}+{x}+{y}', '-window', 'root', filename]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error al tomar captura de pantalla: {result.stderr}")
    
    return Image.open(filename)


def move_mouse_cmd(x: int, y: int):
    """Mueve el ratón usando xdotool"""
    if not has_xdotool():
        raise RuntimeError("La herramienta 'xdotool' no está disponible")

    cmd = ['xdotool', 'mousemove', str(x), str(y)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Advertencia: Error al mover el ratón (puede ser permiso X11): {result.stderr}")
        # No lanzar error para evitar interrupciones, pero registrar el problema


def click_mouse_cmd(button: str = 'left'):
    """Hace clic con el ratón usando xdotool"""
    if not has_xdotool():
        raise RuntimeError("La herramienta 'xdotool' no está disponible")

    # Normalizar el botón: 'left' -> '1', 'right' -> '3', etc.
    if button == 'left':
        button_code = '1'
    elif button == 'right':
        button_code = '3'
    elif button == 'middle':
        button_code = '2'
    else:
        button_code = str(button)  # Asumir que es un número

    cmd = ['xdotool', 'click', button_code]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Advertencia: Error al hacer clic (puede ser permiso X11): {result.stderr}")
        # No lanzar error para evitar interrupciones, pero registrar el problema


def type_text_cmd(text: str):
    """Escribe texto usando xdotool"""
    if not has_xdotool():
        raise RuntimeError("La herramienta 'xdotool' no está disponible")

    cmd = ['xdotool', 'type', text]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Advertencia: Error al escribir texto (puede ser permiso X11): {result.stderr}")
        # No lanzar error para evitar interrupciones, pero registrar el problema


# Variables globales para controlar si usar el entorno virtual
_command_available = None
_fallback_to_virtual = True


def command_operations_available():
    """Verifica si las operaciones de comando están disponibles"""
    global _command_available
    if _command_available is None:
        _command_available = has_import_cmd() and has_xdotool()
    return _command_available


def safe_screenshot(region: Optional[Tuple[int, int, int, int]] = None):
    """
    Toma una captura de pantalla de forma segura en entornos headless
    """
    if not has_real_gui() and not command_operations_available():
        # Solo usar display virtual si no hay GUI real y no hay herramientas de comando
        if _fallback_to_virtual:
            display_manager = VirtualDisplayManager(display_num=101)  # Usar un display diferente
            display_manager.start()
            
            try:
                # Intentar tomar captura con import si está disponible
                if has_import_cmd():
                    return take_screenshot_cmd('/tmp/virtual_screenshot.png', region)
                else:
                    # Si no hay import disponible, lanzar un error
                    raise RuntimeError("No hay herramientas disponibles para tomar captura de pantalla")
            finally:
                display_manager.stop()
        else:
            raise RuntimeError("No hay entorno gráfico disponible ni herramientas de comando")
    else:
        # Intentar usar herramientas de comando o fallback
        if command_operations_available():
            return take_screenshot_cmd('/tmp/screenshot.png', region)
        else:
            # Si no hay herramientas de comando, intentar con display virtual
            display_manager = VirtualDisplayManager(display_num=102)
            display_manager.start()
            
            try:
                if has_import_cmd():
                    return take_screenshot_cmd('/tmp/virtual_screenshot2.png', region)
                else:
                    raise RuntimeError("No hay herramientas disponibles para tomar captura de pantalla")
            finally:
                display_manager.stop()


def safe_mouse_move(x: int, y: int, duration: float = 0.0):
    """
    Mueve el ratón de forma segura en entornos headless
    """
    if not has_real_gui() and not command_operations_available():
        if _fallback_to_virtual:
            display_manager = VirtualDisplayManager(display_num=103)
            display_manager.start()
            
            try:
                # No podemos realmente mover el mouse en el virtual display sin herramientas
                # Simplemente esperar el tiempo de duración
                time.sleep(duration)
                # Aquí solo simulamos el movimiento en el entorno virtual
                return
            finally:
                display_manager.stop()
        else:
            raise RuntimeError("No hay entorno gráfico disponible ni herramientas de comando")
    else:
        if command_operations_available():
            time.sleep(duration)  # Simular la duración si se puede
            move_mouse_cmd(x, y)
        else:
            display_manager = VirtualDisplayManager(display_num=104)
            display_manager.start()
            
            try:
                time.sleep(duration)  # Simular la duración
                # Aquí solo simulamos el movimiento en el entorno virtual
            finally:
                display_manager.stop()


def safe_mouse_click(button: str = 'left'):
    """
    Hace clic con el ratón de forma segura en entornos headless
    """
    if not has_real_gui() and not command_operations_available():
        if _fallback_to_virtual:
            display_manager = VirtualDisplayManager(display_num=105)
            display_manager.start()
            
            try:
                # Simular clic en entorno virtual
                return
            finally:
                display_manager.stop()
        else:
            raise RuntimeError("No hay entorno gráfico disponible ni herramientas de comando")
    else:
        if command_operations_available():
            click_mouse_cmd(button)
        else:
            display_manager = VirtualDisplayManager(display_num=106)
            display_manager.start()
            
            try:
                # Simular clic en entorno virtual
                pass
            finally:
                display_manager.stop()


def safe_keyboard_write(text: str):
    """
    Escribe texto de forma segura en entornos headless
    """
    if not has_real_gui() and not command_operations_available():
        if _fallback_to_virtual:
            display_manager = VirtualDisplayManager(display_num=107)
            display_manager.start()
            
            try:
                # Simular escritura en entorno virtual
                return
            finally:
                display_manager.stop()
        else:
            raise RuntimeError("No hay entorno gráfico disponible ni herramientas de comando")
    else:
        if command_operations_available():
            type_text_cmd(text)
        else:
            display_manager = VirtualDisplayManager(display_num=108)
            display_manager.start()
            
            try:
                # Simular escritura en entorno virtual
                pass
            finally:
                display_manager.stop()


def safe_screen_size():
    """
    Obtiene el tamaño de la pantalla de forma segura en entornos headless
    """
    if not has_real_gui() and not command_operations_available():
        if _fallback_to_virtual:
            display_manager = VirtualDisplayManager(display_num=109)
            display_manager.start()

            try:
                # Devolver un tamaño de pantalla estándar para el entorno virtual
                return (1920, 1080)
            finally:
                display_manager.stop()
        else:
            raise RuntimeError("No hay entorno gráfico disponible ni herramientas de comando")
    else:
        if command_operations_available():
            # Obtener tamaño de pantalla usando xrandr o xdpyinfo
            try:
                result = subprocess.run(['xdpyinfo'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'dimensions:' in line:
                            parts = line.split()
                            dimensions = parts[1]  # Ej: "1920x1080"
                            width, height = map(int, dimensions.split('x'))
                            return (width, height)
            except Exception:
                pass
            # Si falla, devolver tamaño por defecto
            return (1920, 1080)
        else:
            display_manager = VirtualDisplayManager(display_num=110)
            display_manager.start()

            try:
                # Devolver un tamaño de pantalla estándar para el entorno virtual
                return (1920, 1080)
            finally:
                display_manager.stop()


def has_real_gui():
    """
    Verifica si hay un entorno gráfico real disponible
    """
    return 'DISPLAY' in os.environ and os.environ['DISPLAY']