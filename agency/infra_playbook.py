"""Playbook de infra (determinista, sin IA).

Sintetiza acciones conocidas para web (5002), driver (5010) y RAG.
Usar en FIX_INFRA/PAS como guía de “método sobre modelo”.
"""

from __future__ import annotations

PLAYBOOK = {
    "web_ui": {
        "symptoms": [
            "http 404 en /health",
            "Address already in use en 5002",
        ],
        "checks": [
            "verificar qué proceso ocupa 5002: lsof -i :5002",
            "comprobar endpoint válido: /api/adam/health responde 200; /health puede no existir",
        ],
        "actions": [
            "matar procesos viejos en 5002 (p.ej. python web_interface.py duplicado) con lsof -i :5002 -t | xargs kill -9",
            "reiniciar web con systemctl --user restart leann-web.service; si persiste puerto ocupado, usar start_web_interface.sh tras limpiar el puerto",
            "probar health en /api/adam/health si /health devuelve 404",
        ],
        "health_endpoint": "/api/adam/health",
    },
    "driver": {
        "symptoms": [
            "timeout en http://127.0.0.1:5010/health",
        ],
        "checks": [
            "confirmar .venv_os_driver y permisos",
            "ps aux | grep os_driver.py para ver si está vivo",
        ],
        "actions": [
            "lanzar Start-AjaxDriver.ps1 (PowerShell) o drivers/os_driver.py --host 127.0.0.1 --port 5010",
            "incluir header X-AJAX-KEY (usa AJAX_API_KEY/OS_DRIVER_API_KEY o .secret/.secrets/ajax_api_key.txt) para que /health responda 200",
            "aumentar timeout de health en caller si arranque lento",
        ],
    },
    "rag": {
        "symptoms": [
            "systemctl --user restart leann-rag.service -> failed",
        ],
        "checks": [
            "systemctl --user status leann-rag.service",
            "journalctl --user -xeu leann-rag.service",
        ],
        "actions": [
            "resolver dependencias (puerto 8000 libre, AUTH_TOKEN presente)",
            "reiniciar manualmente server.py si el servicio falla",
        ],
    },
}


def get_playbook(subsystem: str) -> dict:
    return PLAYBOOK.get(subsystem, {})
