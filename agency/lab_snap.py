from __future__ import annotations

import json
import os
import socket
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from agency.driver_keys import load_ajax_driver_api_key
from agency.windows_driver_client import _normalize_driver_path
from agency.display_targets import fetch_driver_displays, load_display_map, resolve_display_selection


ROOT = Path(__file__).resolve().parents[1]


def _resolve_lab_driver_url() -> str:
    env_url = (os.getenv("OS_DRIVER_URL_LAB") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    env_host = (os.getenv("OS_DRIVER_HOST_LAB") or "").strip()
    env_port = (os.getenv("OS_DRIVER_PORT_LAB") or "").strip() or "5012"
    if env_host:
        return f"http://{env_host}:{env_port}"
    return f"http://127.0.0.1:{env_port}"


def _resolve_prod_driver_url() -> str:
    env_url = (os.getenv("OS_DRIVER_URL") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    env_host = (os.getenv("OS_DRIVER_HOST") or "").strip()
    env_port = (os.getenv("OS_DRIVER_PORT") or "").strip() or "5010"
    if env_host:
        return f"http://{env_host}:{env_port}"
    return f"http://127.0.0.1:{env_port}"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _latest_display_probe_receipt(root: Path) -> Optional[Dict[str, Any]]:
    base = root / "artifacts" / "ops" / "display_probe"
    if not base.exists():
        return None
    try:
        candidates = [p for p in base.iterdir() if p.is_dir()]
    except Exception:
        return None
    for folder in sorted(candidates, key=lambda p: p.name, reverse=True):
        receipt = folder / "receipt.json"
        if not receipt.exists():
            continue
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _request_json(
    session: requests.Session,
    *,
    url: str,
    headers: Dict[str, str],
    timeout_s: float,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resp = session.get(url, headers=headers, params=params, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"driver_http_{resp.status_code}: {resp.text[:200]}")
    try:
        payload = resp.json()
    except Exception:
        raise RuntimeError(f"driver_response_not_json: {resp.text[:200]}")
    if not isinstance(payload, dict):
        raise RuntimeError("driver_response_not_object")
    return payload


def fetch_lab_driver_capabilities(
    *,
    driver_url: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    base_url = (driver_url or _resolve_lab_driver_url()).rstrip("/")
    api_key = load_ajax_driver_api_key()
    headers = {"X-AJAX-KEY": api_key} if api_key else {}
    if timeout_s is None:
        try:
            timeout_s = float(os.getenv("OS_DRIVER_TIMEOUT", "15") or 15)
        except Exception:
            timeout_s = 15.0
    sess = session or requests.Session()
    payload = _request_json(
        sess,
        url=f"{base_url}/capabilities",
        headers=headers,
        timeout_s=float(timeout_s or 15.0),
    )
    return {"driver_url": base_url, "capabilities": payload}


def capture_lab_snapshot(
    *,
    root_dir: Optional[Path] = None,
    job_id: str,
    mission_id: Optional[str] = None,
    active_window: bool = False,
    driver_url: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout_s: Optional[float] = None,
    context: Optional[str] = None,
    display_id: Optional[int] = None,
    display_override: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root_dir) if root_dir else ROOT
    out_dir = root / "artifacts" / "lab" / "observability" / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ctx = str(context or job_id or mission_id or "manual").strip() or "manual"
    stem = f"shot_{ts}_{ctx}"
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"

    base_url = (driver_url or _resolve_lab_driver_url()).rstrip("/")
    api_key = load_ajax_driver_api_key()
    headers = {"X-AJAX-KEY": api_key} if api_key else {}
    if timeout_s is None:
        try:
            timeout_s = float(os.getenv("OS_DRIVER_TIMEOUT", "15") or 15)
        except Exception:
            timeout_s = 15.0
    sess = session or requests.Session()

    display_probe = _latest_display_probe_receipt(root)
    probe_display_id = None
    if isinstance(display_probe, dict):
        try:
            probe_display_id = int(display_probe.get("dummy_id") or display_probe.get("dummy_display_id"))
        except Exception:
            probe_display_id = None

    display_map = load_display_map(root)
    displays_payload = None
    displays_list: Optional[list[dict]] = None
    try:
        if session is not None:
            displays_payload = _request_json(
                session,
                url=f"{base_url}/displays",
                headers=headers,
                timeout_s=float(timeout_s or 15.0),
            )
        elif fetch_driver_displays is not None:
            displays_payload = fetch_driver_displays(base_url)
        if isinstance(displays_payload, dict):
            displays_list = displays_payload.get("displays")
        if not isinstance(displays_list, list):
            displays_list = None
    except Exception:
        displays_payload = None
        displays_list = None

    selection = resolve_display_selection(
        rail="lab",
        displays=displays_list,
        override=display_override or display_id,
        display_map=display_map,
        fallback_display_id=probe_display_id,
    )
    if selection.get("display_id") is not None:
        display_id = int(selection.get("display_id"))
    params: Optional[Dict[str, Any]] = None
    if display_id is not None:
        params = {"display_id": display_id}
    shot_payload = _request_json(
        sess,
        url=f"{base_url}/screenshot",
        headers=headers,
        timeout_s=float(timeout_s or 15.0),
        params=params,
    )
    if not shot_payload.get("ok"):
        detail = shot_payload.get("error_detail")
        reason = shot_payload.get("error") or "screenshot_failed"
        if detail:
            raise RuntimeError(f"{reason} detail={detail}")
        raise RuntimeError(reason)
    raw_path = shot_payload.get("path") or ""
    if not raw_path:
        raise RuntimeError("driver_returned_no_path")
    snap_path = Path(_normalize_driver_path(str(raw_path)))
    if not snap_path.exists():
        raise RuntimeError(f"screenshot_missing:{snap_path}")
    shutil.copy2(snap_path, png_path)

    window_title = None
    window_rect = None
    lab_session_id = None
    lab_session_user = None
    lab_session_type = None
    lab_session_name = None
    try:
        health_payload = _request_json(
            sess,
            url=f"{base_url}/health",
            headers=headers,
            timeout_s=float(timeout_s or 15.0),
        )
        fg = health_payload.get("fg_window") if isinstance(health_payload.get("fg_window"), dict) else {}
        window_title = fg.get("title") or None
        window_rect = fg.get("rect") if isinstance(fg.get("rect"), dict) else None
        lab_session_id = health_payload.get("session_id")
        lab_session_user = health_payload.get("user")
        lab_session_type = health_payload.get("session_type")
        lab_session_name = health_payload.get("session_name")
    except Exception:
        window_title = None
        window_rect = None

    warnings: list[Dict[str, Any]] = []
    scope_shared = False
    scope_shared_allowed = False
    prod_session_id = None
    prod_session_user = None
    prod_session_type = None
    prod_session_name = None
    prod_driver_url = _resolve_prod_driver_url()
    if prod_driver_url:
        try:
            prod_health = _request_json(
                sess,
                url=f"{prod_driver_url.rstrip('/')}/health",
                headers=headers,
                timeout_s=float(timeout_s or 15.0),
            )
            prod_session_id = prod_health.get("session_id")
            prod_session_user = prod_health.get("user")
            prod_session_type = prod_health.get("session_type")
            prod_session_name = prod_health.get("session_name")
        except Exception:
            prod_session_id = None
    if lab_session_id is not None and prod_session_id is not None:
        if str(lab_session_id) == str(prod_session_id):
            scope_shared = True
            display_probe = _latest_display_probe_receipt(root)
            if isinstance(display_probe, dict):
                scope_shared_allowed = bool(display_probe.get("lab_zone_ok"))
            if not scope_shared_allowed:
                warnings.append(
                    {
                        "evidence_kind": "scope_shared",
                        "detail": "LAB and PROD share the same session_id; LAB snap may be capturing the caller desktop.",
                        "lab_driver_url": base_url,
                        "prod_driver_url": prod_driver_url,
                        "lab_session": {
                            "session_id": lab_session_id,
                            "user": lab_session_user,
                            "session_type": lab_session_type,
                            "session_name": lab_session_name,
                        },
                        "prod_session": {
                            "session_id": prod_session_id,
                            "user": prod_session_user,
                            "session_type": prod_session_type,
                            "session_name": prod_session_name,
                        },
                    }
                )

    cropped = False
    if active_window and isinstance(window_rect, dict):
        try:
            from PIL import Image

            x = int(window_rect.get("x", 0) or 0)
            y = int(window_rect.get("y", 0) or 0)
            w = int(window_rect.get("width", 0) or 0)
            h = int(window_rect.get("height", 0) or 0)
            if w > 0 and h > 0:
                with Image.open(png_path) as img:
                    right = max(0, x + w)
                    bottom = max(0, y + h)
                    cropped_img = img.crop((max(0, x), max(0, y), right, bottom))
                    cropped_img.save(png_path)
                cropped = True
        except Exception:
            cropped = False

    meta = {
        "ts": time.time(),
        "ts_utc": _utc_now(),
        "host": socket.gethostname(),
        "job_id": job_id,
        "mission_id": mission_id,
        "driver_url": base_url,
        "session_id": lab_session_id,
        "session_user": lab_session_user,
        "session_type": lab_session_type,
        "session_name": lab_session_name,
        "active_window": bool(active_window),
        "window_title": window_title,
        "window_rect": window_rect,
        "active_window_cropped": cropped if active_window else False,
        "context": ctx,
        "screenshot_path": str(png_path),
        "display_id": display_id,
        "selected_display_id": selection.get("display_id"),
        "selected_display_label": selection.get("display_label"),
        "selected_by": selection.get("selected_by"),
        "display": selection.get("display"),
        "display_warnings": selection.get("warnings") or [],
        "display_list": displays_list,
        "scope_shared": scope_shared,
        "scope_shared_allowed": scope_shared_allowed,
        "prod_driver_url": prod_driver_url,
        "prod_session_id": prod_session_id,
        "prod_session_user": prod_session_user,
        "prod_session_type": prod_session_type,
        "prod_session_name": prod_session_name,
        "warnings": warnings,
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "png_path": str(png_path),
        "json_path": str(json_path),
        "meta": meta,
        "warnings": warnings,
        "job_id": job_id,
        "mission_id": mission_id,
        "driver_url": base_url,
        "session_id": lab_session_id,
        "display_id": display_id,
        "selected_display_id": selection.get("display_id"),
        "selected_display_label": selection.get("display_label"),
        "selected_by": selection.get("selected_by"),
        "display_warnings": selection.get("warnings") or [],
    }
