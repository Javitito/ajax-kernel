from __future__ import annotations

import json
import importlib.util
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agency.tool_schema import ToolPlan, ToolSpec, ToolStatus


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "config" / "tools_inventory.json"
HEARTBEAT_PATH = ROOT / "artifacts" / "health" / "ajax_heartbeat.json"
TOOL_NOTES_PATH = ROOT / "artifacts" / "tools" / "tool_use_notes.json"


def load_inventory(path: Optional[Path] = None) -> List[ToolSpec]:
    """
    Carga el inventario canónico desde config/tools_inventory.json.
    Devuelve lista vacía en fallo.
    """
    target = Path(path) if path else INVENTORY_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        specs: List[ToolSpec] = []
        for item in raw or []:
            try:
                spec = ToolSpec.from_dict(item)
                if spec.id:
                    specs.append(spec)
            except Exception:
                continue
        return specs
    except Exception:
        return []


def load_heartbeat_snapshot(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Lee artifacts/health/ajax_heartbeat.json si existe. Devuelve dict vacío en fallo.
    """
    target = Path(path) if path else HEARTBEAT_PATH
    try:
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def load_tool_use_notes(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Carga notas de uso acumuladas (fuera de LEANN). No borra ni falla si no existe.
    """
    target = Path(path) if path else TOOL_NOTES_PATH
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def append_tool_use_note(
    tool_id: str,
    note: str,
    outcome: str = "unknown",
    meta: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> None:
    """
    Acumula notas de uso (no destructivo). outcome: success|fail|unknown.
    """
    target = Path(path) if path else TOOL_NOTES_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = load_tool_use_notes(target)
        entries = data.get(tool_id) or []
        entries.append(
            {
                "ts": time.time(),
                "note": note,
                "outcome": outcome,
                "meta": meta or {},
            }
        )
        data[tool_id] = entries
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _can_import(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _command_available(cmd: str) -> bool:
    try:
        return shutil.which(cmd) is not None
    except Exception:
        return False


def _normalize_state(val: Any) -> str:
    state = str(val or "").lower()
    if state in {"green", "healthy"}:
        return "green"
    if state in {"yellow", "warn", "warning", "degraded"}:
        return "yellow"
    if state in {"red", "down", "missing"}:
        return "red"
    return "unknown"


def _extra_checks(tool_id: str) -> Optional[str]:
    if tool_id == "memory.leann_history":
        idx_candidates = [
            ROOT / "ajax_history_v1.leann.index",
            ROOT / "ajax_history_v1.leann.meta.json",
            ROOT / "ajax_history_v1.leann",
        ]
        if not any(p.exists() for p in idx_candidates):
            return "missing_index"
        if not _can_import("agency.leann_query_client"):
            return "leann_query_client_not_importable"
    elif tool_id == "sensing.vision_delta":
        if not _command_available("python3") and not _command_available("python"):
            return "python_not_found"
    elif tool_id == "actuation.driver_uia":
        driver_stub = ROOT / "drivers" / "os_driver.py"
        if not driver_stub.exists():
            return "driver_client_missing"
    elif tool_id == "infra.heartbeat":
        if not HEARTBEAT_PATH.exists():
            return "heartbeat_file_missing"
    elif tool_id == "governance.council":
        providers = ROOT / "config" / "model_providers.yaml"
        if not providers.exists():
            return "model_providers_missing"
    return None


def _summarize_notes(notes: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for tool_id, entries in (notes or {}).items():
        if not isinstance(entries, list):
            continue
        total = len(entries)
        success = len([e for e in entries if isinstance(e, dict) and str(e.get("outcome")).lower() == "success"])
        fail = len([e for e in entries if isinstance(e, dict) and str(e.get("outcome")).lower() == "fail"])
        summary[tool_id] = {
            "uses": total,
            "success": success,
            "fail": fail,
            "observed_confidence": round(success / total, 2) if total else None,
        }
    return summary


def get_tool_status(
    inventory: List[ToolSpec],
    heartbeat: Optional[Dict[str, Any]] = None,
    tool_use_notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, ToolStatus]:
    """
    Devuelve snapshot de estado de cada tool (available/degraded/missing).
    No llama a modelos; solo heartbeat, archivos y presencia de binarios.
    """
    hb = heartbeat or {}
    subsystems = hb.get("subsystems") if isinstance(hb, dict) else {}
    notes_summary = _summarize_notes(tool_use_notes or {})
    status_map: Dict[str, ToolStatus] = {}

    for spec in inventory:
        req_states: Dict[str, str] = {}
        missing: List[str] = []
        degraded: List[str] = []
        for sub in spec.requires_subsystems or []:
            raw_state = None
            if isinstance(subsystems, dict):
                entry = subsystems.get(sub)
                if isinstance(entry, dict):
                    raw_state = entry.get("status")
                else:
                    raw_state = entry
            norm = _normalize_state(raw_state)
            if norm in {"", "unknown"}:
                missing.append(sub)
            elif norm in {"red"}:
                missing.append(sub)
                req_states[sub] = norm
            elif norm in {"yellow"}:
                degraded.append(sub)
                req_states[sub] = norm
            else:
                req_states[sub] = norm

        state = "available"
        reasons: List[str] = []
        if missing:
            state = "missing"
            reasons.append(f"missing:{','.join(missing)}")
        elif degraded:
            state = "degraded"
            reasons.append(f"degraded:{','.join(degraded)}")

        extra_reason = _extra_checks(spec.id)
        if extra_reason:
            reasons.append(extra_reason)
            if state == "available":
                state = "degraded"

        status_map[spec.id] = ToolStatus(
            id=spec.id,
            state=state,
            reason=";".join(reasons) or None,
            subsystems=req_states,
            detail={"notes": notes_summary.get(spec.id)} if spec.id in notes_summary else None,
        )
    return status_map


def build_tool_plan_skeleton(inventory: List[ToolSpec], heartbeat: Dict[str, Any]) -> ToolPlan:
    """
    Conveniencia para obtener ToolPlan vacío con snapshot.
    """
    snapshot = get_tool_status(inventory, heartbeat)
    plan = ToolPlan(
        considered=[spec.id for spec in inventory],
        tool_status_snapshot=snapshot,
    )
    return plan


__all__ = [
    "load_inventory",
    "load_heartbeat_snapshot",
    "load_tool_use_notes",
    "append_tool_use_note",
    "get_tool_status",
    "build_tool_plan_skeleton",
]
