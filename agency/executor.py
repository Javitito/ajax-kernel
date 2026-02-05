"""
Executor (hábitos) — propone planes predecibles antes de llamar al Brain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agency import habits


@dataclass
class HabitPlanCandidate:
    plan_json: Dict[str, Any]
    habit_id: str
    confidence: float
    cost: float
    safety: float
    speed: float


@dataclass
class HabitValidationIssue:
    habit_id: str
    reason: str
    detail: Optional[str] = None


def _validate_habit_steps(habit: habits.Habit) -> Optional[str]:
    # Campos mínimos para que un hábito sea considerado "bien formado" antes de enriquecimiento
    required = {
        "id",
        "action",
        "args",
    }

    for step in habit.steps or []:
        if not isinstance(step, dict):
            return "invalid_step_shape"
        if any(k not in step for k in required):
            return "invalid_step_schema"
        if not isinstance(step.get("args"), dict):
            return "invalid_args"
        if not isinstance(step.get("action"), str) or not str(step.get("action")).strip():
            return "invalid_action"
    return None


def propose_habit_plans(
    intent: str,
    safety_profile: str,
    os_name: str,
    path: Optional[str] = None,
) -> Tuple[List[HabitPlanCandidate], List[HabitValidationIssue]]:
    """
    Devuelve candidatos válidos y lista de hábitos inválidos (con razón).
    """
    habits_path = path or str("data/habits.json")
    matches = habits.find_habits_for_intent(intent, safety_profile, os_name, habits_path)
    candidates: List[HabitPlanCandidate] = []
    issues: List[HabitValidationIssue] = []
    for habit in matches:
        reason = _validate_habit_steps(habit)
        if reason:
            issues.append(HabitValidationIssue(habit_id=habit.id, reason=reason))
            continue
        plan_json = {
            "plan_id": f"habit:{habit.id}",
            "description": f"Plan habitual para: {habit.intent_pattern}",
            "steps": habit.steps,
            "success_contract": {"type": "check_last_step_status"},
            "metadata": {
                "intention": habit.intent_pattern,
                "source": "habit",
                "habit_id": habit.id,
            },
        }
        candidates.append(
            HabitPlanCandidate(
                plan_json=plan_json,
                habit_id=habit.id,
                confidence=0.9,
                cost=0.1,
                safety=0.8,
                speed=0.9,
            )
        )
    return candidates, issues


def propose_habit_plan(intent: str, safety_profile: str, os_name: str) -> Optional[HabitPlanCandidate]:
    """
    Devuelve un plan basado en hábitos si hay uno compatible.
    Valores de scoring fijos por ahora; TODO: hacerlos dinámicos.
    """
    candidates, _issues = propose_habit_plans(intent, safety_profile, os_name)
    return candidates[0] if candidates else None
