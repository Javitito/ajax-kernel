from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class HabitLiteStep:
    id: str
    action: str
    args: Dict[str, Any]

@dataclass
class HabitLite:
    """Minimal schema allowed in data/habits.json"""
    id: str
    intent_pattern: str
    intent_fingerprint: str
    steps: List[Dict[str, Any]]  # List of HabitLiteStep-like dicts
    conditions: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None

@dataclass
class TaskStep:
    """Canonical schema required by PlanRunner"""
    id: str
    intent: str
    preconditions: Dict[str, Any]  # {expected_state: {}}
    action: str
    args: Dict[str, Any]
    evidence_required: List[str]
    success_spec: Dict[str, Any]   # {expected_state: {}}
    on_fail: str = "abort"

@dataclass
class CouncilInput:
    """Explicit contract for Council review context"""
    intention: str
    plan: Dict[str, Any]
    context: Dict[str, Any]  # Must be JSON-serializable (signals, knowledge_context, etc.)
    actions_catalog: Optional[Dict[str, Any]] = None
