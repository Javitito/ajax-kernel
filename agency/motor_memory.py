"""
Motor Memory: almacenamiento y consulta de rutas motoras (Fast Path) aprendidas.
MVP: guarda secuencias simples (mouse_move/click/type) con etiqueta y contexto.
"""
from __future__ import annotations

import json
time_fmt = "%Y-%m-%dT%H:%M:%SZ"
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import logging

log = logging.getLogger(__name__)

MEMORY_ROOT = Path("artifacts") / "motor_memory"


@dataclass
class MotorStep:
    op: str  # "mouse_move", "mouse_click", "keyboard_type"
    args: Dict[str, Any]


@dataclass
class MotorRoute:
    name: str
    goal: str
    steps: List[MotorStep]
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None


def _now() -> str:
    return time.strftime(time_fmt, time.gmtime())


def save_route(route: MotorRoute) -> Path:
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    ts = route.created_at.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
    path = MEMORY_ROOT / f"{ts}_{route.name}.json"
    data = asdict(route)
    data["path"] = str(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info("MotorRoute saved to %s", path)
    return path


def record_route(name: str, goal: str, steps: List[MotorStep], metadata: Optional[Dict[str, Any]] = None) -> Path:
    route = MotorRoute(
        name=name,
        goal=goal,
        steps=steps,
        created_at=_now(),
        metadata=_with_default_metadata(metadata or {}),
    )
    return save_route(route)


def load_routes() -> List[MotorRoute]:
    routes: List[MotorRoute] = []
    if not MEMORY_ROOT.exists():
        return routes
    for p in MEMORY_ROOT.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            steps = [MotorStep(**s) for s in data.get("steps", [])]
            routes.append(
                MotorRoute(
                    name=data.get("name", p.stem),
                    goal=data.get("goal", ""),
                    steps=steps,
                    created_at=data.get("created_at", ""),
                    metadata=_with_default_metadata(data.get("metadata", {})),
                    path=str(p),
                )
            )
        except Exception:
            continue
    return routes


def find_route_for_goal(goal: str) -> Optional[MotorRoute]:
    for r in load_routes():
        if goal.lower() in r.goal.lower():
            return r
    return None


def _with_default_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(meta or {})
    meta.setdefault("success_count", 0)
    meta.setdefault("fail_count", 0)
    meta.setdefault("score", 0)
    meta.setdefault("last_ok_at", None)
    meta.setdefault("last_fail_at", None)
    return meta


def mark_route_result(route: MotorRoute, success: bool) -> None:
    """
    Actualiza contadores/score y persiste el resultado en disco.
    """
    path_str = route.path or ""
    path = Path(path_str) if path_str else None
    if path is None or not path.exists():
        # Mejor no fallar en pipeline de ejecución
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = _with_default_metadata(data.get("metadata", {}))
        if success:
            meta["success_count"] = int(meta.get("success_count", 0)) + 1
            meta["score"] = int(meta.get("score", 0)) + 1
            meta["last_ok_at"] = _now()
        else:
            meta["fail_count"] = int(meta.get("fail_count", 0)) + 1
            meta["score"] = max(int(meta.get("score", 0)) - 1, 0)
            meta["last_fail_at"] = _now()
        data["metadata"] = meta
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        return
