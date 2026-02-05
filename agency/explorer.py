"""
Explorer: envuelve la planificación Brain+Council y devuelve un PlanCandidate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agency.arbiter import PlanCandidate


class Explorer:
    """
    Wrapper ligero alrededor del camino Brain+Council ya existente en AjaxCore.
    """

    def __init__(self, core: "AjaxCore"):
        self.core = core

    def propose_plan(self, mission: "MissionState") -> Optional[PlanCandidate]:
        """
        Ejecuta la planificación completa (Brain+Council) y devuelve PlanCandidate.
        Si no hay plan viable, devuelve None.
        """
        try:
            obs = self.core.perceive()
            plan = self.core.plan(
                mission.intention,
                obs,
                feedback=mission.feedback,
                envelope=mission.envelope,
                brain_exclude=mission.brain_exclude,
                mission=mission,
            )
            plan_json: Dict[str, Any] = {
                "plan_id": plan.plan_id or plan.id,
                "description": plan.summary,
                "steps": plan.steps,
                "success_contract": plan.success_spec or plan.metadata.get("success_contract") if plan.metadata else None,
                "metadata": plan.metadata or {},
            }
            # Heurísticas simples para scoring (se pueden mejorar con señales reales)
            candidate = PlanCandidate(
                origin="explorer",
                plan_json=plan_json,
                confidence=0.75,
                safety=0.7,
                cost=0.6,
                speed=0.4,
                notes="brain+council",
            )
            return candidate
        except Exception:
            return None


# Evitar import circular en tiempo de carga
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from agency.ajax_core import AjaxCore, MissionState
