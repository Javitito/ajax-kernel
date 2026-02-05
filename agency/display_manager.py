"""
Módulo para gestionar el entorno de display virtual usando Xvfb
"""
import os
import subprocess
import time
from contextlib import contextmanager

class VirtualDisplayManager:
    """
    Maneja un display virtual usando Xvfb para permitir operaciones de GUI
    en entornos headless.
    """

    def __init__(self, width=1920, height=1080, color_depth=24, display_num=99):
        self.width = width
        self.height = height
        self.color_depth = color_depth
        self.display_num = display_num  # Número de display virtual configurable
        self.xvfb_process = None
        self.original_display = os.environ.get('DISPLAY')

    def start(self):
        """
        Inicia el servidor Xvfb virtual
        """
        display_var = f":{self.display_num}"
        cmd = [
            'Xvfb',
            display_var,
            '-screen', '0',
            f'{self.width}x{self.height}x{self.color_depth}'
        ]

        print(f"Iniciando Xvfb en display {display_var}...")
        self.xvfb_process = subprocess.Popen(cmd)

        # Esperar a que Xvfb esté listo
        time.sleep(2)

        # Establecer la variable de entorno DISPLAY
        os.environ['DISPLAY'] = display_var
        print(f"DISPLAY establecido a {display_var}")

        return True

    def stop(self):
        """
        Detiene el servidor Xvfb virtual
        """
        if self.xvfb_process:
            print("Deteniendo Xvfb...")
            self.xvfb_process.terminate()
            self.xvfb_process.wait()

        # Restaurar el display original
        if self.original_display:
            os.environ['DISPLAY'] = self.original_display
        elif 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']

        print("Xvfb detenido y entorno restaurado")

    @contextmanager
    def virtual_display(self):
        """
        Context manager para usar el display virtual temporalmente
        """
        try:
            success = self.start()
            if not success:
                raise RuntimeError("No se pudo iniciar el display virtual correctamente")
            yield self
        finally:
            self.stop()


# Función auxiliar para verificar si estamos en un entorno con GUI real
def has_real_gui():
    """
    Verifica si hay un entorno gráfico real disponible
    """
    return 'DISPLAY' in os.environ and os.environ['DISPLAY']