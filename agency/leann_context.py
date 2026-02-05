"""
Adaptador mínimo para obtener contexto desde LEANN.

Stub seguro: siempre devuelve un dict, aunque LEANN no responda todavía.
Se puede evolucionar para llamar a la API HTTP o a otros backends.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests


def _stub_context(intent: str, mode: str, timeout_seconds: float) -> Dict[str, Any]:
    """Contexto seguro por defecto cuando LEANN no está disponible."""
    return {
        "source": "leann_stub",
        "mode": mode,
        "rag_snippets": [],
        "notes": [],
        "related_projects": [],
        "meta": {
            "intent_preview": intent[:200],
            "timeout_seconds": timeout_seconds,
        },
    }


def get_leann_context(intent: str, mode: str = "persona", timeout_seconds: float = 1.5) -> Dict[str, Any]:
    """
    Devuelve un contexto estructurado para enriquecer la misión de AJAX.

    Args:
        intent: intención original del usuario.
        mode: tipo de contexto deseado ("persona", "system", "persona+system"...).
        timeout_seconds: timeout para llamadas HTTP/IPC no bloqueantes.

    Usa LEANN_RAG_URL si está presente; en error o ausencia, devuelve un stub seguro.
    """
    url = (os.getenv("LEANN_RAG_URL") or "").strip()
    if not url:
        return _stub_context(intent, mode, timeout_seconds)

    payload = {
        "query": intent,
        "mode": mode,
        "limit": 8,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout_seconds)
        if resp.status_code >= 400:
            return _stub_context(intent, mode, timeout_seconds)
        data: Dict[str, Any] = {}
        try:
            data = resp.json() or {}
        except Exception:
            return _stub_context(intent, mode, timeout_seconds)
        snippets: List[Any] = (
            data.get("snippets")
            or data.get("results")
            or data.get("passages")
            or []
        )
        notes: List[Any] = data.get("notes") or []
        related: List[Any] = data.get("projects") or data.get("collections") or []
        latency_ms = data.get("latency_ms") or data.get("latency")
        return {
            "source": "leann_http",
            "mode": mode,
            "rag_snippets": snippets,
            "notes": notes,
            "related_projects": related,
            "meta": {
                "intent_preview": intent[:200],
                "query_latency_ms": latency_ms,
            },
        }
    except Exception:
        return _stub_context(intent, mode, timeout_seconds)
