"""
Módulo de hábitos (versión mínima).

Objetivo: servir planes habituales antes de invocar al Brain.
Evolución esperada:
- Mejor fingerprinting (normalizar intents, embeddings, clusterización).
- GC/actualización de hábitos según éxito/uso.
- Persistencia en LEANN u otra capa de memoria.
"""

from __future__ import annotations

import json
import logging
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class Habit:
    id: str
    intent_pattern: str
    intent_fingerprint: str
    steps: List[Dict[str, Any]]
    conditions: Dict[str, Any]
    evidence_requirements: List[str]
    success_rate: float
    usage_count: int
    last_used_at: str
    meta: Optional[Dict[str, Any]] = None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 1, "habits": []}
    except Exception as exc:  # pragma: no cover
        log.warning("No se pudo leer %s: %s", path, exc)
        return {"version": 1, "habits": []}


def load_habits(path: Path | str = Path("data/habits.json")) -> List[Habit]:
    """
    Carga hábitos desde disco. Si falta el fichero o está corrupto, devuelve lista vacía.
    """
    p = Path(path)
    data = _load_json(p)
    habits_raw = data.get("habits") or []
    habits: List[Habit] = []
    for h in habits_raw:
        try:
            habits.append(
                Habit(
                    id=h.get("id", ""),
                    intent_pattern=h.get("intent_pattern", ""),
                    intent_fingerprint=h.get("intent_fingerprint", ""),
                    steps=h.get("steps") or [],
                    conditions=h.get("conditions") or {},
                    evidence_requirements=h.get("evidence_requirements") or [],
                    success_rate=float(h.get("success_rate", 0.0)),
                    usage_count=int(h.get("usage_count", 0)),
                    last_used_at=h.get("last_used_at", ""),
                    meta=h.get("meta"),
                )
            )
        except Exception as exc:  # pragma: no cover
            log.warning("Habit mal formado en %s: %s", p, exc)
    return habits


def save_habits(habits: List[Habit], path: Path | str = Path("data/habits.json")) -> None:
    """
    Persiste hábitos. Sobrescribe el fichero destino.
    """
    p = Path(path)
    payload = {
        "version": 1,
        "habits": [h.__dict__ for h in habits],
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        log.warning("No se pudo guardar hábitos en %s: %s", p, exc)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _matches(intent: str, habit: Habit) -> bool:
    """
    Matching simplón: lower-case y contiene tokens clave.
    TODO: mejorar con fingerprints/embeddings/normalización avanzada.
    """
    text = intent.lower().strip()
    patt = (habit.intent_pattern or "").lower()
    if patt and patt in text:
        return True
    # fallback: palabras clave en fingerprint
    fp = (habit.intent_fingerprint or "").lower()
    return all(tok in text for tok in fp.split() if tok)


def find_habits_for_intent(
    intent: str,
    safety_profile: str,
    os_name: str,
    path: Path | str = Path("data/habits.json"),
) -> List[Habit]:
    """
    Devuelve todos los hábitos compatibles con la intención y condiciones básicas.
    """
    items = load_habits(path)
    out: List[Habit] = []
    for h in items:
        if not _matches(intent, h):
            continue
        cond = h.conditions or {}
        os_ok = cond.get("os") is None or cond.get("os") == os_name
        safety_ok = cond.get("safety_profile") is None or safety_profile in cond.get("safety_profile", [])
        if os_ok and safety_ok:
            out.append(h)
    return out


def find_habit_for_intent(
    intent: str,
    safety_profile: str,
    os_name: str,
    path: Path | str = Path("data/habits.json"),
) -> Optional[Habit]:
    """
    Devuelve un hábito compatible con la intención y condiciones básicas.
    """
    matches = find_habits_for_intent(intent, safety_profile, os_name, path)
    return matches[0] if matches else None


def update_habit_usage(habit_id: str, success: bool, path: Path | str = Path("data/habits.json")) -> None:
    """
    Actualiza contadores de uso y tasa de éxito de un hábito.
    """
    items = load_habits(path)
    changed = False
    for h in items:
        if h.id != habit_id:
            continue
        old_count = h.usage_count
        old_rate = h.success_rate
        new_count = old_count + 1
        h.usage_count = new_count
        h.last_used_at = _utc_iso()
        try:
            h.success_rate = ((old_rate * old_count) + (1.0 if success else 0.0)) / max(new_count, 1)
        except Exception:
            h.success_rate = old_rate
        changed = True
        break
    if changed:
        save_habits(items, path)
