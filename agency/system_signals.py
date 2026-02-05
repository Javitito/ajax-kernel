"""
Señales rápidas del sistema para enriquecer knowledge_context.
Fase 1: solo lecturas seguras (driver, perfil de seguridad, placeholders).
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict

import requests

from agency.driver_keys import load_ajax_driver_api_key

def _utc_iso() -> str:
    """Devuelve timestamp UTC en ISO 8601 con sufijo Z."""
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _check_driver_health(timeout_seconds: float = 0.3) -> str:
    """
    Comprueba la salud del driver OS vía /health.

    Usa AJAX_DRIVER_HEALTH_URL si está definida, o 127.0.0.1:5010 por defecto.
    Nunca lanza excepción: siempre devuelve 'healthy', 'degraded' o 'down'.
    Criterios:
      - Sin respuesta / timeout → "down"
      - HTTP 200 y status=="ok" en JSON → "healthy"
      - Cualquier otro HTTP o JSON → "degraded" (no lo tratamos como down)
    """
    url = os.getenv("AJAX_DRIVER_HEALTH_URL", "http://127.0.0.1:5010/health")
    headers = {}
    api_key = load_ajax_driver_api_key()
    if api_key:
        headers["X-AJAX-KEY"] = api_key
    try:
        resp = requests.get(url, timeout=timeout_seconds, headers=headers)
        if resp.status_code == 200:
            # Tratamos cualquier 200 como usable/healthy para evitar falsos degradados
            return "healthy"
        return "degraded"
    except Exception:
        return "down"


def _get_safety_profile() -> str:
    """
    Perfil de seguridad del usuario: tonterias | normal | delicado | paranoico.
    """
    val = (os.getenv("AJAX_SAFETY_PROFILE") or "").strip().lower()
    if val in {"tonterias", "normal", "delicado", "paranoico"}:
        return val
    return "normal"


def collect_signals() -> Dict[str, Any]:
    """
    Devuelve un dict con señales del sistema para adjuntar a knowledge_context.

    Ahora mismo:
      - timestamp (UTC)
      - driver_health (ping /health)
      - active_services (placeholder)
      - user_safety_profile (ENV/normal)
      - recent_patterns, known_risks (placeholders ampliables)
    """
    driver_health = _check_driver_health()
    safety_profile = _get_safety_profile()

    signals: Dict[str, Any] = {
        "timestamp": _utc_iso(),
        "driver_health": driver_health,
        "active_services": ["desktop_driver"] if driver_health != "down" else [],
        "user_safety_profile": safety_profile,
        "recent_patterns": {},
        "known_risks": [],
    }
    return signals
