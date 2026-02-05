"""
Árbitro de planes: puntúa y selecciona el mejor candidato.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class PlanCandidate:
    """
    Representa un plan propuesto por un origen (habit o explorer).
    - confidence: probabilidad subjetiva de cumplir la intención (0..1)
    - safety: qué tan seguro es (0..1)
    - cost: tokens/latencia normalizada (0..1, 1 = muy caro)
    - speed: estimación inversa de duración (0..1, 1 = muy rápido)
    - notes: texto breve opcional
    """
    origin: Literal["habit", "explorer"]
    plan_json: Dict[str, any]
    confidence: float
    safety: float
    cost: float
    speed: float
    notes: str = ""


def score_candidate(candidate: PlanCandidate, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calcula score ponderado. Por defecto primamos confianza y seguridad,
    penalizamos coste y añadimos velocidad.
    """
    w = weights or {"confidence": 0.4, "safety": 0.3, "cost": 0.2, "speed": 0.1}
    return (
        candidate.confidence * w.get("confidence", 0.0)
        + candidate.safety * w.get("safety", 0.0)
        + candidate.speed * w.get("speed", 0.0)
        + (1 - candidate.cost) * w.get("cost", 0.0)
    )


def choose_best(
    candidates: List[PlanCandidate],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[PlanCandidate], Dict[str, float]]:
    """
    Devuelve (mejor_candidato, scores_por_origen).
    Si lista vacía -> (None, {}).
    Si solo uno -> lo devuelve con su score.
    """
    if not candidates:
        return None, {}
    if len(candidates) == 1:
        s = score_candidate(candidates[0], weights)
        return candidates[0], {candidates[0].origin: s}
    best: Optional[PlanCandidate] = None
    best_score = float("-inf")
    scores: Dict[str, float] = {}
    for c in candidates:
        s = score_candidate(c, weights)
        scores[c.origin] = s if c.origin not in scores else max(scores[c.origin], s)
        if s > best_score:
            best = c
            best_score = s
    return best, scores


if __name__ == "__main__":
    dummy_habit = PlanCandidate(
        origin="habit",
        plan_json={"id": "habit", "steps": []},
        confidence=0.8,
        safety=0.8,
        cost=0.1,
        speed=0.9,
        notes="dummy habit",
    )
    dummy_explorer = PlanCandidate(
        origin="explorer",
        plan_json={"id": "explorer", "steps": []},
        confidence=0.7,
        safety=0.7,
        cost=0.6,
        speed=0.5,
        notes="dummy explorer",
    )
    best, sc = choose_best([dummy_habit, dummy_explorer])
    print("best:", best.origin if best else None, "scores:", sc)
