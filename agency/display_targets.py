from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from agency.driver_keys import load_ajax_driver_api_key


DisplayEntry = Dict[str, Any]
DisplaySelection = Dict[str, Any]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _norm_rail(rail: Optional[str]) -> str:
    val = (rail or "").strip().lower()
    if val in {"prod", "production"}:
        return "prod"
    if val in {"lab", "laboratory"}:
        return "lab"
    return "lab"


def _display_map_path(root_dir: Path) -> Path:
    return root_dir / "config" / "display_map.json"


def load_display_map(root_dir: Path) -> Dict[str, Any]:
    path = _display_map_path(root_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def save_display_map(root_dir: Path, targets: Dict[str, Any], source: str = "calibrate") -> Path:
    path = _display_map_path(root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ajax.display_map.v1",
        "updated_at": _utc_now(),
        "source": source,
        "display_targets": targets,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return path


def _display_targets_from_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("display_targets") if isinstance(payload, dict) else None
    if isinstance(raw, dict):
        return raw
    return {}


def _primary_display_id(displays: List[DisplayEntry]) -> Optional[int]:
    for disp in displays:
        if disp.get("is_primary"):
            try:
                return int(disp.get("id"))
            except Exception:
                continue
    for disp in displays:
        try:
            return int(disp.get("id"))
        except Exception:
            continue
    return None


def _non_primary_ids(displays: List[DisplayEntry]) -> List[int]:
    ids: List[int] = []
    for disp in displays:
        if disp.get("is_primary"):
            continue
        try:
            ids.append(int(disp.get("id")))
        except Exception:
            continue
    return ids


def _find_display(displays: List[DisplayEntry], display_id: Optional[int]) -> Optional[DisplayEntry]:
    if display_id is None:
        return None
    for disp in displays:
        try:
            if int(disp.get("id")) == int(display_id):
                return disp
        except Exception:
            continue
    return None


def _parse_display_override(val: Optional[Union[str, int]]) -> Tuple[Optional[int], Optional[str]]:
    if val is None:
        return None, None
    if isinstance(val, int):
        return int(val), None
    raw = str(val).strip().lower()
    if not raw:
        return None, None
    if raw.isdigit():
        try:
            return int(raw), None
        except Exception:
            return None, None
    if raw in {"dummy", "lab"}:
        return None, "dummy"
    if raw in {"primary", "prod"}:
        return None, "primary"
    return None, raw


def resolve_display_selection(
    *,
    rail: Optional[str],
    displays: Optional[List[DisplayEntry]],
    override: Optional[Union[str, int]] = None,
    display_map: Optional[Dict[str, Any]] = None,
    fallback_display_id: Optional[int] = None,
) -> DisplaySelection:
    rail_n = _norm_rail(rail)
    displays_list = displays or []
    warnings: List[str] = []

    map_targets = _display_targets_from_map(display_map or {})
    map_lab = map_targets.get("lab")
    map_prod = map_targets.get("prod")

    override_id, override_label = _parse_display_override(override)
    selected_id: Optional[int] = None
    selected_by = "heuristic"
    label = "display"

    if override_id is not None:
        selected_id = override_id
        selected_by = "override"
    elif override_label in {"dummy"}:
        if map_lab is not None:
            try:
                selected_id = int(map_lab)
                selected_by = "override"
            except Exception:
                selected_id = None
        if selected_id is None:
            non_primary = _non_primary_ids(displays_list)
            if non_primary:
                selected_id = non_primary[0]
                selected_by = "override"
            elif fallback_display_id is not None:
                selected_id = fallback_display_id
                selected_by = "override"
    elif override_label in {"primary"}:
        if map_prod is not None:
            try:
                selected_id = int(map_prod)
                selected_by = "override"
            except Exception:
                selected_id = None
        if selected_id is None:
            selected_id = _primary_display_id(displays_list)
            selected_by = "override"
    elif rail_n == "lab":
        if map_lab is not None:
            try:
                selected_id = int(map_lab)
                selected_by = "config"
            except Exception:
                selected_id = None
        if selected_id is None:
            non_primary = _non_primary_ids(displays_list)
            if non_primary:
                selected_id = non_primary[0]
                selected_by = "heuristic"
            elif fallback_display_id is not None:
                selected_id = fallback_display_id
                selected_by = "probe"
    else:
        if map_prod is not None:
            try:
                selected_id = int(map_prod)
                selected_by = "config"
            except Exception:
                selected_id = None
        if selected_id is None:
            selected_id = _primary_display_id(displays_list)
            selected_by = "heuristic"

    if not displays_list:
        warnings.append("display_list_unavailable")
    display_entry = _find_display(displays_list, selected_id)
    if selected_id is not None and display_entry is None and displays_list:
        warnings.append("display_id_not_found")
        fallback = _primary_display_id(displays_list)
        if fallback is None and displays_list:
            try:
                fallback = int(displays_list[0].get("id"))
            except Exception:
                fallback = None
        if fallback is not None:
            selected_id = fallback
            selected_by = "fallback"
            display_entry = _find_display(displays_list, selected_id)
        else:
            warnings.append("display_fallback_failed")

    if rail_n == "lab":
        if display_entry and not display_entry.get("is_primary"):
            label = "dummy"
        elif selected_by in {"config", "override"}:
            label = "dummy"
    if label != "dummy":
        primary_id = _primary_display_id(displays_list)
        if selected_id is not None and primary_id is not None and selected_id == primary_id:
            label = "primary"

    return {
        "display_id": selected_id,
        "display_label": label,
        "selected_by": selected_by,
        "warnings": warnings,
        "display": display_entry,
        "rail": rail_n,
    }


def fetch_driver_displays(driver_url: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests_not_available")
    api_key = load_ajax_driver_api_key()
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-AJAX-KEY"] = api_key
    url = driver_url.rstrip("/") + "/displays"
    resp = requests.get(url, headers=headers, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"driver_http_{resp.status_code}: {resp.text[:200]}")
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("driver_response_not_object")
    return payload
