"""
Skill Fabricator: fabrica nuevas habilidades motoras basadas en primitivas del driver Windows.

Flujo (simplificado):
1) Carga catálogo de primitivas (policy/os_primitives_schema.json).
2) Genera un SkillSpec (via LLM; aquí se stub y se apoya en contexto).
3) Ejecuta la secuencia con WindowsDriverClient en modo ensayo.
4) Si no falla, persiste el SkillSpec y registra en heartbeat.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

from agency.windows_driver_client import WindowsDriverClient, WindowsDriverError

PRIMITIVES_PATH = Path("policy/os_primitives_schema.json")
SKILL_SCHEMA_PATH = Path("policy/skill_spec.schema.json")
SKILL_STORE = Path("skills/generated")
RUN_LOG = Path("artifacts/heartbeat/skill_fabricator.log")


def _load_primitives() -> Dict[str, Any]:
    if not PRIMITIVES_PATH.exists():
        return {}
    try:
        return json.loads(PRIMITIVES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_dirs() -> None:
    SKILL_STORE.mkdir(parents=True, exist_ok=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log_run(record: Dict[str, Any]) -> None:
    _ensure_dirs()
    record.setdefault("ts", int(time.time()))
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stub_generate_skill(goal: str) -> Dict[str, Any]:
    """
    Stub de generación de SkillSpec: crea una secuencia básica usando hotkeys.
    En producción se debería llamar a Qwen/Codex con policy/skill_spec.schema.json como guía.
    """
    return {
        "name": f"skill_auto_{int(time.time())}",
        "goal": goal,
        "steps": [
            {"primitive": "keyboard.hotkey", "args": {"keys": ["ctrl", "s"]}, "wait_ms": 500},
            {"primitive": "keyboard.type", "args": {"text": "v2.txt", "submit": True}, "wait_ms": 0},
        ],
        "metadata": {"generated_by": "skill_fabricator_stub"}
    }


def _execute_step(wdc: WindowsDriverClient, step: Dict[str, Any]) -> bool:
    prim = step.get("primitive") or ""
    args = step.get("args") or {}
    wait_ms = int(step.get("wait_ms", 0) or 0)
    try:
        if prim == "app.launch":
            res = wdc.app_launch(process=args.get("process"), path=args.get("path"), args=args.get("args"))
        elif prim == "keyboard.type":
            res = wdc.keyboard_type(text=args.get("text", ""), submit=bool(args.get("submit", False)))
        elif prim == "keyboard.hotkey":
            keys = args.get("keys") or []
            res = wdc.keyboard_hotkey(*keys)
        elif prim == "mouse.click":
            res = wdc.mouse_click(x=args.get("x"), y=args.get("y"), button=args.get("button", "left"))
        elif prim == "vision.find_text":
            res = wdc.find_text(text=args.get("text", ""), app=args.get("app"))
        else:
            return False
        ok = bool(res.get("ok", True)) if isinstance(res, dict) else True
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)
        return ok
    except Exception:
        return False


def _execute_skill(wdc: WindowsDriverClient, spec: Dict[str, Any]) -> bool:
    for step in spec.get("steps") or []:
        if not _execute_step(wdc, step):
            return False
    return True


def _persist_skill_spec(spec: Dict[str, Any]) -> Path:
    _ensure_dirs()
    name = spec.get("name") or f"skill_{int(time.time())}"
    path = SKILL_STORE / f"{name}.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def fabricate_skill(goal: str, failure_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fabrica una skill nueva para el goal dado. Devuelve dict con ok y paths.
    """
    wdc = WindowsDriverClient()
    if not wdc.is_available():
        return {"ok": False, "error": "windows_driver_unavailable"}

    _ = _load_primitives()  # hoy no se usa, pero queda para validación futura

    # TODO: sustituir stub por llamada real a LLM usando skill_spec.schema.json como guía
    spec = _stub_generate_skill(goal)

    ok_exec = _execute_skill(wdc, spec)
    spec_path = _persist_skill_spec(spec)
    record = {"goal": goal, "spec_path": str(spec_path), "ok_exec": ok_exec, "context": failure_context or {}}
    _log_run(record)

    return {"ok": ok_exec, "spec_path": str(spec_path)}
