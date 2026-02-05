from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _as_list_str(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if str(v)]
    if isinstance(val, str):
        return [val] if val else []
    return []


@dataclass
class ToolSpec:
    id: str
    kind: str
    requires_subsystems: List[str] = field(default_factory=list)
    cost: Any = "medium"
    risk: str = "medium"
    cooldown: Dict[str, Any] = field(default_factory=dict)
    max_per_turn: Optional[int] = None
    outputs: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    affordances: List[str] = field(default_factory=list)
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolSpec":
        requires = _as_list_str(data.get("requires_subsystems"))
        outputs = _as_list_str(data.get("outputs"))
        hints = _as_list_str(data.get("hints"))
        affordances = _as_list_str(data.get("affordances"))
        known_keys = {
            "id",
            "kind",
            "requires_subsystems",
            "cost",
            "risk",
            "cooldown",
            "max_per_turn",
            "outputs",
            "hints",
            "affordances",
            "notes",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            id=str(data.get("id") or ""),
            kind=str(data.get("kind") or ""),
            requires_subsystems=requires,
            cost=data.get("cost", "medium"),
            risk=str(data.get("risk") or "medium"),
            cooldown=data.get("cooldown") or {},
            max_per_turn=data.get("max_per_turn"),
            outputs=outputs,
            hints=hints,
            affordances=affordances,
            notes=str(data.get("notes") or ""),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not payload.get("extra"):
            payload.pop("extra", None)
        return payload


@dataclass
class ToolStatus:
    id: str
    state: str  # available | degraded | missing
    reason: Optional[str] = None
    subsystems: Dict[str, str] = field(default_factory=dict)
    detail: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if payload.get("detail") is None:
            payload.pop("detail", None)
        if payload.get("reason") is None:
            payload.pop("reason", None)
        return payload


@dataclass
class ToolPlan:
    considered: List[str] = field(default_factory=list)
    selected: List[str] = field(default_factory=list)
    required: List[str] = field(default_factory=list)
    allowed: List[str] = field(default_factory=list)
    satisfied: Dict[str, bool] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    tool_status_snapshot: Dict[str, ToolStatus] = field(default_factory=dict)
    incomplete: bool = False
    risk_flags: Dict[str, Any] = field(default_factory=dict)
    affordances: Dict[str, List[str]] = field(default_factory=dict)
    tool_use_notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for key, status in (self.tool_status_snapshot or {}).items():
            if isinstance(status, ToolStatus):
                snapshot[key] = status.to_dict()
            elif hasattr(status, "to_dict"):
                snapshot[key] = status.to_dict()
            elif hasattr(status, "__dict__"):
                snapshot[key] = asdict(status)
            else:
                snapshot[key] = status
        return {
            "considered": list(self.considered),
            "selected": list(dict.fromkeys(self.selected)),
            "required": list(dict.fromkeys(self.required)),
            "allowed": list(dict.fromkeys(self.allowed)),
            "satisfied": dict(self.satisfied or {}),
            "budget": self.budget or {},
            "reasons": list(self.reasons),
            "tool_status_snapshot": snapshot,
            "incomplete": bool(self.incomplete),
            "risk_flags": dict(self.risk_flags or {}),
            "affordances": {k: list(v) for k, v in (self.affordances or {}).items()},
            "tool_use_notes": self.tool_use_notes or {},
        }
