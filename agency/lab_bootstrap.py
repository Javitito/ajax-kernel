from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agency.display_targets import fetch_driver_displays, load_display_map, save_display_map


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_canonical_root(root_dir: Path) -> bool:
    return root_dir.name == "ajax-kernel" and (root_dir / "agency").exists() and (
        root_dir / "bin" / "ajaxctl"
    ).exists()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_if_missing(path: Path, content: str, *, created: List[str]) -> None:
    if path.exists():
        return
    _ensure_parent(path)
    path.write_text(content, encoding="utf-8")
    created.append(str(path))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _pick_ids(display_catalog: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    displays = (
        display_catalog.get("displays")
        if isinstance(display_catalog, dict) and isinstance(display_catalog.get("displays"), list)
        else []
    )
    primary_id: Optional[int] = None
    dummy_id: Optional[int] = None
    for item in displays:
        if not isinstance(item, dict):
            continue
        try:
            disp_id = int(item.get("id"))
        except Exception:
            continue
        if bool(item.get("is_primary")) and primary_id is None:
            primary_id = disp_id
        if not bool(item.get("is_primary")) and dummy_id is None:
            dummy_id = disp_id
    if primary_id is None:
        primary_id = 1
    if dummy_id is None:
        dummy_id = 2 if primary_id != 2 else 1
    return {"primary_id": primary_id, "dummy_id": dummy_id}


def ensure_lab_display_target(
    root_dir: Path,
    *,
    display_catalog: Optional[Dict[str, Any]] = None,
    allow_fetch_catalog: bool = True,
) -> Dict[str, Any]:
    root = Path(root_dir)
    map_payload = load_display_map(root)
    targets = (
        dict(map_payload.get("display_targets"))
        if isinstance(map_payload.get("display_targets"), dict)
        else {}
    )

    catalog = display_catalog
    if catalog is None and allow_fetch_catalog:
        try:
            catalog = fetch_driver_displays("http://127.0.0.1:5012")
        except Exception:
            catalog = None

    ids = _pick_ids(catalog)
    dummy_id = int(ids["dummy_id"] or 2)
    primary_id = int(ids["primary_id"] or 1)

    updated = False
    lab_target = targets.get("lab")
    try:
        lab_target_int = int(lab_target)
    except Exception:
        lab_target_int = None
    if lab_target_int is None:
        targets["lab"] = dummy_id
        lab_target_int = dummy_id
        updated = True

    prod_target = targets.get("prod")
    try:
        prod_target_int = int(prod_target)
    except Exception:
        prod_target_int = None
    if prod_target_int is None:
        targets["prod"] = primary_id
        prod_target_int = primary_id
        updated = True

    path = root / "config" / "display_map.json"
    if updated or not path.exists():
        save_display_map(root, targets, source="lab_init")

    return {
        "ok": True,
        "updated": bool(updated or not map_payload),
        "path": str(path),
        "lab_target": int(lab_target_int),
        "prod_target": int(prod_target_int),
        "dummy_id": dummy_id,
        "primary_id": primary_id,
        "display_catalog_available": bool(isinstance(catalog, dict)),
    }


def ensure_lab_bootstrap(root_dir: Path) -> Dict[str, Any]:
    root = Path(root_dir)
    created: List[str] = []
    updated: List[str] = []

    _write_if_missing(
        root / "config" / "lab_org_manifest.yaml",
        (
            "schema: ajax.lab_org_manifest.v1\n"
            "micro_challenges:\n"
            "  - id: capabilities_refresh\n"
            "    job_kind: capabilities_refresh\n"
            "    enabled: true\n"
            "    ui_intrusive: false\n"
            "    cadence_s: 300\n"
            "    budget_s: 20\n"
            "    tags: [safe, health]\n"
        ),
        created=created,
    )
    _write_if_missing(
        root / "config" / "explore_policy.yaml",
        (
            "policy:\n"
            "  human_active_threshold_s: 90\n"
            "  unknown_signal_as_human: true\n"
            "  require_dummy_display_ok: true\n"
            "states:\n"
            "  AWAY:\n"
            "    require_dummy_display_ok: true\n"
            "    force_non_ui_due: true\n"
            "  HUMAN_DETECTED:\n"
            "    allow_ui_intrusive_with_lease: false\n"
            "human_signal:\n"
            "  ps_script: scripts/ops/get_human_signal.ps1\n"
            "  failure_mode: strict\n"
            "  timeout_s: 2.5\n"
        ),
        created=created,
    )
    _write_if_missing(
        root / "scripts" / "ops" / "get_human_signal.ps1",
        (
            "$ErrorActionPreference = 'Stop'\n"
            "function Emit-Active([string]$Err) {\n"
            "  $payload = @{\n"
            "    schema='ajax.human_signal.v1';\n"
            "    ok=$false;\n"
            "    last_input_age_sec=0;\n"
            "    session_unlocked=$true;\n"
            "    error=$Err;\n"
            "  }\n"
            "  $payload | ConvertTo-Json -Compress\n"
            "}\n"
            "try {\n"
            "  Emit-Active 'stub_fail_closed'\n"
            "} catch {\n"
            "  Emit-Active ('stub_exception:' + $_.Exception.Message)\n"
            "}\n"
        ),
        created=created,
    )

    display_res = ensure_lab_display_target(root, allow_fetch_catalog=True)
    if display_res.get("updated"):
        updated.append(display_res.get("path"))

    return {
        "schema": "ajax.lab_bootstrap.v1",
        "ok": True,
        "ts_utc": _utc_now(),
        "root": str(root),
        "canonical_root": bool(_is_canonical_root(root)),
        "created": [p for p in created if p],
        "updated": [str(p) for p in updated if p],
        "display_map": display_res,
        "next_steps": [
            "python bin/ajaxctl lab start",
            "python bin/ajaxctl doctor anchor --rail lab",
            "python bin/ajaxctl microfilm check --root <AJAX_HOME>",
        ],
    }
