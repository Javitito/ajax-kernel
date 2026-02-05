from __future__ import annotations

import base64
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

import requests  # type: ignore

from agency.path_utils import windows_to_wsl_path
from agency.windows_driver_client import WindowsDriverClient
from agency.skills.os_inspection import inspect_ui_tree
from agency.method_pack import AJAX_METHOD_PACK
from agency.vision_gate import ensure_local_vision_allowed


@dataclass
class VisionClickResult:
    success: bool
    chosen_id: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    alternatives: List[str] | None = None
    error: Optional[str] = None
    path: str = "none"  # "ocr" | "multimodal" | "none"


# ---------- Helpers: text matching ----------
def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


def _score_text_match(target: str, candidate: str) -> float:
    t = _normalize_text(target)
    c = _normalize_text(candidate)
    if not t or not c:
        return 0.0
    if t == c:
        return 1.0
    if c.startswith(t) or t.startswith(c):
        return 0.85
    if t in c or c in t:
        return 0.75
    common = len(set(t.split()) & set(c.split()))
    if common:
        return 0.5
    return 0.0


def _ocr_match_marks(
    target_text: str,
    marks: List[Dict[str, Any]],
    min_score: float = 0.7,
    min_ocr_conf: float = 0.5,
) -> Tuple[Optional[Dict[str, Any]], float]:
    best_mark = None
    best_score = 0.0
    for m in marks:
        text = m.get("text") or ""
        ocr_conf = float(m.get("ocr_confidence", 1.0))
        if not text or ocr_conf < min_ocr_conf:
            continue
        score = _score_text_match(target_text, text)
        if score > best_score:
            best_score = score
            best_mark = m
    if best_mark and best_score >= min_score:
        return best_mark, best_score
    return None, best_score


# ---------- Helpers: SoM and click ----------
def _bbox_center(bbox: List[int] | Tuple[int, int, int, int]) -> Tuple[int, int]:
    x_min, y_min, x_max, y_max = bbox
    return (int((x_min + x_max) / 2), int((y_min + y_max) / 2))


def _click_bbox(bbox: List[int]) -> None:
    cx, cy = _bbox_center(bbox)
    driver = WindowsDriverClient()
    driver.mouse_click(x=cx, y=cy)


def _find_mark_by_id(marks: List[Dict[str, Any]], mark_id: str) -> Optional[Dict[str, Any]]:
    for m in marks:
        if m.get("id") == mark_id:
            return m
    return None


def _load_vision_strategy(profile: str = "pc_mini_32gb") -> Dict[str, Any]:
    """
    Carga la estrategia de visión desde config/vision_strategies.yaml.
    """
    if yaml is None:
        return {}
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "vision_strategies.yaml"
    if not cfg_path.exists():
        return {}
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return (data.get("profiles") or {}).get(profile, {}) or {}
    except Exception:
        return {}


def capture_and_tag_screen(rows: int = 4, cols: int = 4) -> Dict[str, Any]:
    """
    Usa el client/skill para obtener SoM: image_path y marks (idealmente con bbox, text, ocr_confidence).
    """
    ensure_local_vision_allowed("tag_screen_grid")
    client = WindowsDriverClient()
    res = client.tag_screen_grid(rows=rows, cols=cols)

    # Si recibimos un meta_path/json_path, leerlo y normalizar rutas internas
    meta_path = res.get("meta_path") or res.get("json_path")
    if meta_path:
        meta_path = windows_to_wsl_path(str(meta_path))
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        img_path = meta.get("image_path")
        if img_path:
            meta["image_path"] = windows_to_wsl_path(str(img_path))
        res = meta

    # Normaliza siempre image_path para WSL
    img_path = res.get("image_path")
    if img_path:
        res["image_path"] = windows_to_wsl_path(str(img_path))

    return _enrich_with_ocr(res)


def perform_click_from_mark(mark: Dict[str, Any]) -> None:
    """
    Calcula el centro del bbox y lanza un click vía driver (pasa por Safety del driver).
    """
    bbox = mark.get("screen_bbox") or mark.get("bbox")
    if not bbox or len(bbox) != 4:
        raise ValueError(f"Mark sin bbox válido: {mark!r}")

    x_min, y_min, x_max, y_max = bbox
    x = int((x_min + x_max) / 2)
    y = int((y_min + y_max) / 2)

    driver = WindowsDriverClient()
    driver.mouse_click(x=x, y=y)


# ---------- OCR enrichment ----------
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None
else:
    try:
        default_path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
        if default_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default_path)
    except Exception:
        pass


def _ocr_for_image(img_path: Path | str, bbox: List[int]) -> Tuple[str, float]:
    if pytesseract is None or Image is None:
        return "", 0.0
    img_path = Path(windows_to_wsl_path(str(img_path)))
    # Asegurar ruta de tesseract si no está configurada
    try:
        if not getattr(pytesseract.pytesseract, "tesseract_cmd", ""):
            default_path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
            if default_path.exists():
                pytesseract.pytesseract.tesseract_cmd = str(default_path)
    except Exception:
        pass
    try:
        with Image.open(img_path) as im:
            x0, y0, x1, y1 = bbox
            region = im.crop((x0, y0, x1, y1))
            data = pytesseract.image_to_data(region, output_type=pytesseract.Output.DICT)
            texts = []
            confs = []
            n = len(data.get("text", []))
            for i in range(n):
                txt = data["text"][i].strip()
                conf_raw = str(data["conf"][i])
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.0
                if txt:
                    texts.append(txt)
                    confs.append(conf)
            if not texts:
                return "", 0.0
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            return " ".join(texts), avg_conf / 100.0
    except Exception:
        return "", 0.0


def _enrich_with_ocr(payload: Dict[str, Any]) -> Dict[str, Any]:
    img_path_str = windows_to_wsl_path(str(payload.get("image_path") or ""))
    payload["image_path"] = img_path_str
    img_path = Path(img_path_str)
    marks: List[Dict[str, Any]] = payload.get("marks") or []
    new_marks: List[Dict[str, Any]] = []
    for m in marks:
        rect = m.get("rect") or {}
        bbox = [
            int(rect.get("x", 0)),
            int(rect.get("y", 0)),
            int(rect.get("x", 0) + rect.get("width", 0)),
            int(rect.get("y", 0) + rect.get("height", 0)),
        ]
        text, conf = _ocr_for_image(img_path, bbox)
        m = dict(m)
        m["bbox"] = bbox
        # Conservar screen_bbox si viene del driver para clicks absolutos
        if "screen_bbox" in m and m.get("screen_bbox"):
            m["screen_bbox"] = m["screen_bbox"]
        if text:
            m["text"] = text
            m["ocr_confidence"] = conf
        new_marks.append(m)
    payload["marks"] = new_marks
    return payload


# ---------- Multimodal client placeholders ----------
SYSTEM_PROMPT = (
    AJAX_METHOD_PACK
    + "\n\n"
    + """
Eres un asistente experto en interfaces gráficas.

RECIBIRÁS:
- Una captura de pantalla.
- Una lista de marcas (marks) con IDs y descripciones de su posición.

TU TAREA:
- Identificar qué mark corresponde mejor al objetivo que te indique el usuario.
- Responder EXCLUSIVAMENTE en JSON válido, sin texto extra.

FORMATO EXACTO DE RESPUESTA:
{
  "chosen_id": "Mx",
  "confidence": 0.0-1.0,
  "reason": "una frase corta",
  "alternatives": ["My", "Mz"]
}

REGLAS:
- Solo puedes usar IDs de la lista de marks.
- Si no estás seguro, elige el mark más probable y baja la confidence (<0.5).
- No inventes IDs.
"""
).strip()


def build_user_prompt(target_text: str, role: str, som_meta: Dict[str, Any]) -> str:
    marks_desc = []
    for m in som_meta.get("marks", []):
        cell = m.get("grid_cell") or m.get("id")
        bbox = m.get("bbox") or m.get("rect")
        marks_desc.append(f"- id: {m.get('id')} cell: {cell} bbox: {bbox}")
    return f"""
Objetivo:
- Texto objetivo: "{target_text}"
- Rol esperado: {role}

Marks disponibles:
{chr(10).join(marks_desc)}

Instrucciones:
- Observa la imagen y elige el ID que mejor corresponde a "{target_text}".
- Devuelve SOLO el JSON en el formato solicitado.
"""


def _get_env(candidates: list[str]) -> Optional[str]:
    import os

    for key in candidates:
        val = os.getenv(key)
        if val:
            return val
    return None


def _parse_llm_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, list):
        content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
    if not isinstance(content, str):
        raise RuntimeError("vision_llm_invalid_content")
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[len("json") :].strip()
    return json.loads(raw)


def _resize_image_b64(image_path: Path, max_width: int) -> str:
    img_path = image_path
    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None  # type: ignore
    if Image is not None and max_width:
        try:
            with Image.open(img_path) as im:
                if im.width > max_width:
                    ratio = max_width / im.width
                    new_size = (max_width, int(im.height * ratio))
                    im = im.resize(new_size)
                    tmp = img_path.parent / f"{img_path.stem}_vl_tmp.png"
                    im.save(tmp)
                    img_path = tmp
        except Exception:
            img_path = image_path
    return base64.b64encode(img_path.read_bytes()).decode("ascii")


def _call_local_qwen(image_path: str, user_prompt: str, model: str, base_url: str, api_key: Optional[str], timeout_s: int, max_width: int) -> Dict[str, Any]:
    ensure_local_vision_allowed("call_local_qwen")
    img_b64 = _resize_image_b64(Path(image_path), max_width)
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"vision_llm_http_{resp.status_code}: {resp.text}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("vision_llm_no_choices")
    content = choices[0].get("message", {}).get("content") or ""
    return _parse_llm_content(content)


def _call_qwen_cli(image_path: str, user_prompt: str, model: str, base_url: Optional[str], api_key: Optional[str], timeout_s: int, max_width: int) -> Dict[str, Any]:
    api_key = api_key or _get_env(["QWEN_API_KEY", "DASHSCOPE_API_KEY"])
    if not api_key:
        raise RuntimeError("qwen_cli_api_key_missing")
    base = (base_url or _get_env(["QWEN_API_BASE"]) or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    img_b64 = _resize_image_b64(Path(image_path), max_width)
    url = base + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"vision_llm_http_{resp.status_code}: {resp.text}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("vision_llm_no_choices")
    content = choices[0].get("message", {}).get("content") or ""
    return _parse_llm_content(content)


def _call_groq(image_path: str, user_prompt: str, model: str, base_url: Optional[str], api_key: Optional[str], timeout_s: int, max_width: int) -> Dict[str, Any]:
    api_key = api_key or _get_env(["GROQ_API_KEY"])
    if not api_key:
        raise RuntimeError("groq_api_key_missing")
    base = (base_url or _get_env(["GROQ_API_BASE"]) or "https://api.groq.com/openai/v1").rstrip("/")
    img_b64 = _resize_image_b64(Path(image_path), max_width)
    url = base + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"vision_llm_http_{resp.status_code}: {resp.text}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("vision_llm_no_choices")
    content = choices[0].get("message", {}).get("content") or ""
    return _parse_llm_content(content)


def call_vision_llm(
    image_path: str,
    user_prompt: str,
    # Default alineado con el VL cargado en LM Studio (evita model_not_found).
    model: str = "qwen/qwen3-vl-4b",
    base_url: str = "http://localhost:1235/v1",
    api_key: Optional[str] = None,
    timeout_s: Optional[int] = None,
    max_width: Optional[int] = None,
    strategy_profile: str = "pc_mini_32gb",
    backend: str = "local",
) -> Dict[str, Any]:
    """
    Cliente OpenAI-compatible para VL. Soporta backends:
    - local (LM Studio / servidor local)
    - qwen_cli (DashScope/compatible)
    - groq (API Groq vision)
    """
    strategy = _load_vision_strategy(strategy_profile)
    if timeout_s is None:
        timeout_s = int(strategy.get("vl_local", {}).get("timeout_seconds") or 25)
    if max_width is None:
        max_width = int(strategy.get("vl_local", {}).get("max_width") or 960)

    backend = (backend or "local").lower()
    if backend == "local":
        return _call_local_qwen(image_path, user_prompt, model, base_url, api_key, timeout_s, max_width)
    if backend == "qwen_cli":
        return _call_qwen_cli(image_path, user_prompt, model, base_url, api_key, timeout_s, max_width)
    if backend == "groq":
        return _call_groq(image_path, user_prompt, model, base_url, api_key, timeout_s, max_width)
    raise RuntimeError(f"backend_not_supported:{backend}")


def warm_qwen(model: str = "qwen/qwen3-vl-4b", base_url: str = "http://localhost:1235/v1", timeout_s: int = 10) -> bool:
    """
    Llamada mínima para precargar el modelo multimodal. No se usa automáticamente.
    """
    try:
        dummy = Path("tmp/_vision_ping.png")
        dummy.parent.mkdir(parents=True, exist_ok=True)
        dummy.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc``\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82")
        _ = call_vision_llm(
            image_path=str(dummy),
            user_prompt="ping",
            model=model,
            base_url=base_url,
            api_key=None,
            timeout_s=timeout_s,
            backend="local",
        )
        return True
    except Exception:
        return False


# ---------- API principal ----------
def vision_llm_click(
    target_text: str,
    role: str = "button",
    app_hint: Optional[str] = None,
    timeout_s: int = 15,
    min_confidence: float = 0.6,
    model: str = "gemini-1.5-pro-vision",
    try_ocr_first: bool = True,
    strategy_profile: str = "pc_mini_32gb",
) -> VisionClickResult:
    try:
        # 0) UIA-first: si encontramos el control por nombre, click inmediato
        try:
            ui = inspect_ui_tree()
        except Exception:
            ui = {"ok": False}
        if isinstance(ui, dict) and ui.get("ok"):
            elems = ui.get("elements") or []
            target_norm = _normalize_text(target_text)
            for el in elems:
                name = _normalize_text(str(el.get("name") or ""))
                rect = el.get("rect") or []
                if target_norm and name and target_norm in name and len(rect) == 4:
                    try:
                        _click_bbox([int(v) for v in rect])
                        return VisionClickResult(
                            success=True,
                            chosen_id=None,
                            confidence=0.99,
                            reason="uia",
                            alternatives=[],
                            path="uia",
                        )
                    except Exception as exc:
                        # si el click falla seguimos con el pipeline normal
                        pass

        som_meta = capture_and_tag_screen()
        image_path = som_meta.get("image_path") or ""
        marks = som_meta.get("marks") or []

        # 1) OCR-first
        if try_ocr_first:
            ocr_mark, ocr_score = _ocr_match_marks(target_text, marks)
            if ocr_mark is not None:
                perform_click_from_mark(ocr_mark)
                return VisionClickResult(
                    success=True,
                    chosen_id=ocr_mark.get("id"),
                    confidence=ocr_score,
                    reason=f"OCR local score={ocr_score:.2f}",
                    alternatives=[],
                    path="ocr",
                )
            # Si OCR está activado y no encuentra nada, no forzamos multimodal salvo petición explícita
            return VisionClickResult(
                success=False,
                error="OCR no encontró el objetivo y multimodal no está habilitado",
                reason="try_ocr_first=True",
                alternatives=[],
                path="ocr",
            )

        # 2) Multimodal fallback
        user_prompt = build_user_prompt(target_text, role, som_meta)
        try:
            llm_resp = call_vision_llm(
                image_path=image_path,
                user_prompt=user_prompt,
                model=model,
                timeout_s=timeout_s,
                strategy_profile=strategy_profile,
            )
        except Exception as exc:
            return VisionClickResult(
                success=False,
                error=str(exc),
                reason="Excepción/timeout en multimodal",
                alternatives=[],
                path="multimodal",
            )

        chosen_id = llm_resp.get("chosen_id")
        confidence = float(llm_resp.get("confidence", 0.0))
        alternatives = llm_resp.get("alternatives") or []
        reason = llm_resp.get("reason", "")

        if not chosen_id:
            return VisionClickResult(
                success=False,
                error="LLM no devolvió chosen_id",
                reason=reason,
                alternatives=alternatives,
                path="multimodal",
            )

        mark = _find_mark_by_id(marks, chosen_id)
        if mark is None:
            return VisionClickResult(
                success=False,
                error=f"chosen_id {chosen_id} no existe en marks",
                reason=reason,
                alternatives=alternatives,
                path="multimodal",
            )

        if confidence < min_confidence:
            return VisionClickResult(
                success=False,
                chosen_id=chosen_id,
                confidence=confidence,
                reason=f"Confianza insuficiente ({confidence:.2f}). {reason}",
                alternatives=alternatives,
                path="multimodal",
            )

        perform_click_from_mark(mark)

        return VisionClickResult(
            success=True,
            chosen_id=chosen_id,
            confidence=confidence,
            reason=reason,
            alternatives=alternatives,
            path="multimodal",
        )
    except Exception as exc:  # pragma: no cover
        return VisionClickResult(
            success=False,
            error=str(exc),
            reason="Excepción en vision_llm_click",
            alternatives=[],
            path="none",
        )
