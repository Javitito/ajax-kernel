"""
Helpers de visión (OCR y VL) compartidos por los skills.

run_ocr: OCR ligero y tolerante a fallos.
run_vl:  Llamada segura a modelos de visión-lenguaje con fallback a OCR y stub CLI opcional.
"""
from __future__ import annotations

import logging
import os
import subprocess
import json
import tempfile
from typing import Any, Dict

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None
try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

from agency.vision_gate import ensure_local_vision_allowed

log = logging.getLogger(__name__)


# ---------- OCR ----------
def run_ocr(tile_image: "Image.Image") -> str:
    """
    Ejecuta OCR sobre un tile. Nunca lanza excepción: ante fallo devuelve "".
    """
    if pytesseract is None or Image is None:
        return ""
    try:
        data = pytesseract.image_to_data(tile_image, output_type=pytesseract.Output.DICT)
        texts = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = data["text"][i].strip()
            if txt:
                texts.append(txt)
        return " ".join(texts)
    except Exception as exc:  # pragma: no cover - OCR debe ser tolerante
        log.warning("run_ocr failed: %s", exc)
        return ""


# ---------- VL ----------
def run_vl(tile_image: "Image.Image", model: str = "florence", max_tokens: int = 128) -> Dict[str, Any]:
    """
    Ejecuta un modelo VL sobre la imagen.

    Estrategia:
    1) Si se define la variable de entorno VISION_VL_COMMAND, se invoca el CLI externo con los
       argumentos --image <png> --model <model> --max-tokens <n> y se espera JSON con {"summary": "..."}.
    2) Si el CLI no existe o falla, se usa un fallback ligero basado en OCR como pseudo-summary.
    Nunca lanza excepción: ante fallo devuelve summary vacío.
    """
    ensure_local_vision_allowed("run_vl")
    cli = os.getenv("VISION_VL_COMMAND")
    if cli and Image is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tile_image.save(tmp.name)
                cmd = [
                    cli,
                    "--image",
                    tmp.name,
                    "--model",
                    model,
                    "--max-tokens",
                    str(max_tokens),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40, check=False)
                if proc.returncode == 0 and proc.stdout.strip():
                    try:
                        data = json.loads(proc.stdout)
                        if isinstance(data, dict) and "summary" in data:
                            return {"summary": str(data.get("summary", "")), "raw": data, "model": model}
                    except Exception:
                        pass
        except Exception as exc:  # pragma: no cover - defensivo
            log.warning("run_vl CLI failed model=%s: %s", model, exc)

    # Fallback: usar OCR como pseudo-summary
    try:
        ocr_text = run_ocr(tile_image)
        return {"summary": ocr_text, "raw": None, "model": model}
    except Exception as exc:  # pragma: no cover
        log.warning("run_vl fallback failed model=%s: %s", model, exc)
        return {"summary": "", "raw": None, "model": model}
