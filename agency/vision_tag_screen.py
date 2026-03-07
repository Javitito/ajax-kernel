from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import inspect

from agency.delta_vision import compute_tile_hash, tile_changed
from agency.path_utils import windows_to_wsl_path
from agency.vision_gate import (
    ensure_local_vision_allowed,
    is_local_vision_allowed,
    select_local_vision_provider,
)
from agency.windows_driver_client import WindowsDriverClient

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None
try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ts_label() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"{int((time.time() % 1) * 1000):03d}Z"


def _artifact_dir(root_dir: Optional[Path] = None) -> Path:
    base = Path(root_dir) if root_dir is not None else Path.cwd()
    return base / "artifacts" / "vision"


def _cache_path(root_dir: Optional[Path] = None, cache_override: Optional[Path] = None) -> Path:
    if cache_override is not None:
        return Path(cache_override)
    return _artifact_dir(root_dir) / "delta_cache.json"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _load_cache(path: Path) -> Tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
    payload = _read_json(path) or {}
    hashes_raw = payload.get("tile_hashes") if isinstance(payload.get("tile_hashes"), dict) else {}
    tags_raw = payload.get("tile_tags") if isinstance(payload.get("tile_tags"), dict) else {}
    hashes: Dict[str, int] = {}
    for key, value in hashes_raw.items():
        try:
            hashes[str(key)] = int(value)
        except Exception:
            continue
    tags: Dict[str, Dict[str, Any]] = {}
    for key, value in tags_raw.items():
        if isinstance(value, dict):
            tags[str(key)] = dict(value)
    return hashes, tags


def _save_cache(path: Path, hashes: Dict[str, int], tags: Dict[str, Dict[str, Any]]) -> None:
    payload = {
        "schema": "ajax.vision.delta_cache.v1",
        "ts_utc": _utc_now(),
        "tile_hashes": hashes,
        "tile_tags": tags,
    }
    _write_json(path, payload)


def _normalize_grid_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    res = dict(raw)
    meta_path = res.get("meta_path") or res.get("json_path")
    if meta_path:
        p = Path(_resolve_image_path(str(meta_path)))
        meta = _read_json(p)
        if isinstance(meta, dict):
            res = meta
    img_path = res.get("image_path")
    if img_path:
        res["image_path"] = _resolve_image_path(str(img_path))
    return res


def _default_threshold() -> int:
    raw = os.getenv("AJAX_VISION_DELTA_THRESHOLD")
    if raw is None:
        return 6
    try:
        return max(0, int(raw))
    except Exception:
        return 6


def _rect_to_bbox(mark: Dict[str, Any]) -> List[int]:
    rect = mark.get("rect") if isinstance(mark.get("rect"), dict) else {}
    x = int(rect.get("x", 0))
    y = int(rect.get("y", 0))
    w = int(rect.get("width", 0))
    h = int(rect.get("height", 0))
    return [x, y, x + max(0, w), y + max(0, h)]


def _resolve_image_path(raw: str) -> str:
    if not raw:
        return raw
    direct = Path(str(raw))
    if direct.exists():
        return str(direct)
    converted = windows_to_wsl_path(str(raw))
    converted_path = Path(converted)
    if converted_path.exists():
        return str(converted_path)
    return str(raw) if os.name == "nt" else str(converted)


def _ocr_for_bbox(
    image_path: Path,
    bbox: List[int],
    *,
    ocr_fn: Optional[Callable[[Path, List[int]], Tuple[str, float]]] = None,
) -> Tuple[str, float]:
    if ocr_fn is not None:
        return ocr_fn(image_path, bbox)
    if pytesseract is None or Image is None:
        return "", 0.0
    try:
        with Image.open(image_path) as image:
            x0, y0, x1, y1 = bbox
            region = image.crop((x0, y0, x1, y1))
            data = pytesseract.image_to_data(region, output_type=pytesseract.Output.DICT)
            texts: List[str] = []
            confs: List[float] = []
            count = len(data.get("text", []))
            for idx in range(count):
                text = str(data["text"][idx] or "").strip()
                if not text:
                    continue
                conf_raw = str(data["conf"][idx] or "")
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.0
                texts.append(text)
                confs.append(conf)
            if not texts:
                return "", 0.0
            avg = (sum(confs) / len(confs)) if confs else 0.0
            return " ".join(texts), avg / 100.0
    except Exception:
        return "", 0.0


def tag_screen_with_delta(
    *,
    rows: int = 4,
    cols: int = 4,
    threshold: Optional[int] = None,
    provider_used: Optional[str] = None,
    root_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    driver_client: Optional[Any] = None,
    ocr_fn: Optional[Callable[[Path, List[int]], Tuple[str, float]]] = None,
    allow_local: bool = False,
) -> Dict[str, Any]:
    ensure_local_vision_allowed("tag_screen_grid", allow_local=allow_local)
    if rows <= 0 or cols <= 0:
        raise ValueError("tag_screen_grid_invalid_dimensions")
    thr = _default_threshold() if threshold is None else max(0, int(threshold))
    client = driver_client or WindowsDriverClient()
    raw_payload = client.tag_screen_grid(rows=rows, cols=cols)
    payload = _normalize_grid_payload(raw_payload if isinstance(raw_payload, dict) else {})
    marks = payload.get("marks") if isinstance(payload.get("marks"), list) else []
    image_path_raw = payload.get("image_path")
    if not image_path_raw:
        payload["marks"] = marks
        return payload
    image_path = Path(_resolve_image_path(str(image_path_raw)))
    if Image is None:
        raise RuntimeError("pillow_not_available")
    if not image_path.exists():
        raise FileNotFoundError(f"vision_image_missing:{image_path}")

    out_dir = _artifact_dir(root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(root_dir, cache_path)
    prev_hashes, cached_tags = _load_cache(cache_file)
    provider_meta = select_local_vision_provider(root_dir=root_dir)
    if provider_used is None and is_local_vision_allowed(allow_local=allow_local) and bool(provider_meta.get("up")):
        provider_used = str(provider_meta.get("provider") or "")

    changed_count = 0
    skipped_count = 0
    new_marks: List[Dict[str, Any]] = []

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        for idx, mark_any in enumerate(marks):
            mark = dict(mark_any) if isinstance(mark_any, dict) else {}
            mark_id = str(mark.get("id") or idx)
            bbox = _rect_to_bbox(mark)
            x0, y0, x1, y1 = bbox
            x0 = max(0, min(x0, rgb.width))
            y0 = max(0, min(y0, rgb.height))
            x1 = max(x0 + 1, min(x1, rgb.width))
            y1 = max(y0 + 1, min(y1, rgb.height))
            bbox = [x0, y0, x1, y1]
            tile = rgb.crop((x0, y0, x1, y1))
            current_hash = compute_tile_hash(tile)
            previous_hash = prev_hashes.get(mark_id)
            is_changed = tile_changed(previous_hash, current_hash, threshold=thr)

            mark["bbox"] = bbox
            mark["tile_hash"] = f"{int(current_hash):016x}"

            if is_changed:
                changed_count += 1
                text, conf = _ocr_for_bbox(image_path, bbox, ocr_fn=ocr_fn)
                if text:
                    mark["text"] = text
                    mark["ocr_confidence"] = conf
                    cached_tags[mark_id] = {"text": text, "ocr_confidence": conf}
                mark["delta_state"] = "CHANGED"
            else:
                skipped_count += 1
                cached = cached_tags.get(mark_id)
                if isinstance(cached, dict) and cached.get("text"):
                    mark["text"] = str(cached.get("text"))
                    mark["ocr_confidence"] = float(cached.get("ocr_confidence") or 0.0)
                    mark["delta_state"] = "UNCHANGED_REUSED"
                else:
                    mark["delta_state"] = "UNCHANGED"
                    mark["ocr_status"] = "UNCHANGED"

            prev_hashes[mark_id] = int(current_hash)
            new_marks.append(mark)

    _save_cache(cache_file, prev_hashes, cached_tags)

    delta_payload: Dict[str, Any] = {
        "schema": "ajax.vision.delta_run.v1",
        "ts_utc": _utc_now(),
        "tiles_total": len(new_marks),
        "tiles_changed": changed_count,
        "tiles_skipped": skipped_count,
        "threshold": thr,
        "hash_kind": "dhash64",
    }
    if provider_used:
        delta_payload["provider_used"] = str(provider_used)
    delta_payload["provider_local_status"] = provider_meta

    delta_path = out_dir / f"delta_run_{_ts_label()}.json"
    _write_json(delta_path, delta_payload)

    payload["image_path"] = str(image_path)
    payload["marks"] = new_marks
    payload["delta_run_path"] = str(delta_path)
    payload["delta_run"] = delta_payload
    return payload


def _latest_delta_metrics(root_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    out_dir = _artifact_dir(root_dir)
    if not out_dir.exists():
        return None
    candidates = sorted(out_dir.glob("delta_run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    latest = candidates[0]
    doc = _read_json(latest)
    if not isinstance(doc, dict):
        return {"path": str(latest), "error": "delta_run_invalid_json"}
    return {"path": str(latest), **doc}


def _delta_bypass_detected() -> tuple[bool, str]:
    try:
        source = inspect.getsource(tag_screen_with_delta)
    except Exception:
        return True, "source_unavailable"
    if "compute_tile_hash" not in source or "tile_changed" not in source:
        return True, "delta_vision_not_used"
    return False, ""


def run_doctor_vision(root_dir: Optional[Path] = None) -> Dict[str, Any]:
    allowed = is_local_vision_allowed()
    provider_meta = select_local_vision_provider(root_dir=root_dir)
    bypass_detected, bypass_reason = _delta_bypass_detected()
    latest_delta = _latest_delta_metrics(root_dir=root_dir)
    next_hint: List[str] = []
    if not allowed:
        next_hint.append("set VISION_LOCAL_ALLOWED=true")
        next_hint.append("python bin/ajaxctl vision tag-screen --allow-local")
    if not provider_meta.get("up"):
        next_hint.append("python bin/ajaxctl doctor providers --roles vision")
    if bypass_detected:
        next_hint.append("pytest tests/test_vision_delta_wireup.py -v --tb=short")
    payload = {
        "schema": "ajax.doctor.vision.v1",
        "ts_utc": _utc_now(),
        "vision_local_allowed": bool(allowed),
        "provider_local": provider_meta,
        "bypass_detected": bool(bypass_detected),
        "bypass_reason": bypass_reason or "none",
        "latest_delta": latest_delta,
        "next_hint": next_hint,
    }
    payload["summary"] = format_doctor_vision_summary(payload)
    return payload


def format_doctor_vision_summary(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("AJAX Doctor vision")
    lines.append(f"VISION_LOCAL_ALLOWED: {payload.get('vision_local_allowed')}")
    provider = payload.get("provider_local") if isinstance(payload.get("provider_local"), dict) else {}
    lines.append(
        f"provider_local: {provider.get('provider') or 'none'} status={provider.get('status')} reason={provider.get('reason') or ''}"
    )
    lines.append(f"bypass_detected: {bool(payload.get('bypass_detected'))}")
    latest = payload.get("latest_delta") if isinstance(payload.get("latest_delta"), dict) else {}
    if latest:
        lines.append(
            f"latest_delta: changed={latest.get('tiles_changed')} skipped={latest.get('tiles_skipped')} threshold={latest.get('threshold')}"
        )
    hints = payload.get("next_hint") if isinstance(payload.get("next_hint"), list) else []
    if hints:
        lines.append("next_hint:")
        for hint in hints:
            lines.append(f"- {hint}")
    return "\n".join(lines)
