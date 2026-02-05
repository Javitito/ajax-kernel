"""
Heurísticas simples para convertir la salida de vision_delta_tag_screen en
\"afordancias\" accionables (diálogos, botones, campos).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


# Palabras clave básicas para diálogos y botones de escritorio (ES/EN)
_DIALOG_PATTERNS = [
    "guardar como",
    "save as",
    "confirmar guardar",
    "open",
    "abrir",
    "print",
    "imprimir",
]

_BUTTON_PATTERNS = [
    "guardar",
    "guardar como",
    "save",
    "save as",
    "cancelar",
    "cancel",
    "no guardar",
    "don't save",
    "overwrite",
    "reemplazar",
    "ok",
    "aceptar",
    "yes",
    "no",
]


def _matches(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(pat in t for pat in patterns)


def extract_affordances_from_delta(delta_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Toma la salida de vision_delta_tag_screen y devuelve una lista de affordances
    con tipo/bbox/labels.
    """
    tiles = delta_result.get("tiles_processed") or []
    affordances: List[Dict[str, Any]] = []

    for t in tiles:
        bbox = t.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        text = (t.get("ocr_text") or "") + " " + (t.get("vl_summary") or "")
        text = text.strip()
        if not text:
            continue

        # Dialogs
        if _matches(text, _DIALOG_PATTERNS):
            affordances.append(
                {
                    "type": "dialog",
                    "id": f"dialog_{bbox[0]}_{bbox[1]}",
                    "label": text[:80],
                    "bbox": bbox,
                }
            )

        # Buttons
        if _matches(text, _BUTTON_PATTERNS):
            affordances.append(
                {
                    "type": "button",
                    "id": f"button_{bbox[0]}_{bbox[1]}",
                    "label": text[:80],
                    "bbox": bbox,
                }
            )

    return affordances
