from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal


@dataclass
class GovernanceSpec:
    budget_tokens: int = 0
    budget_seconds: int = 0
    risk_level: Literal["low", "medium", "high"] = "medium"
    autonomy: Literal["L1", "L2", "L3"] = "L1"
    notes: Optional[str] = None


@dataclass
class SuccessContract:
    primary_source: Literal["uia", "fs", "vision", "none"] = "uia"
    primary_check: Dict[str, Any] = field(default_factory=dict)
    fallback_source: Optional[Literal["uia", "fs", "vision", "none"]] = "none"
    fallback_check: Optional[Dict[str, Any]] = None
    conflict_resolution: Literal["primary_wins", "fail_safe", "ask_human"] = "fail_safe"


@dataclass
class ExecutionEvent:
    ts: float
    step_id: str
    action: str
    ok: bool
    detail: Any = None
    error: Optional[str] = None


@dataclass
class MissionError:
    kind: Literal["plan_error", "world_error", "sensor_error", "governance_error", "unknown"]
    step_id: Optional[str] = None
    reason: str = ""


@dataclass
class Hypothesis:
    plan: Any
    success_contract: SuccessContract
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionEnvelope:
    mission_id: str
    original_intent: str
    governance: GovernanceSpec
    hypothesis: Optional[Hypothesis] = None
    execution_log: List[ExecutionEvent] = field(default_factory=list)
    last_error: Optional[MissionError] = None
    world_model: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.hypothesis and self.hypothesis.plan is not None:
            plan = self.hypothesis.plan
            if hasattr(plan, "__dataclass_fields__"):
                plan_payload = asdict(plan)
            elif isinstance(plan, dict):
                plan_payload = plan
            elif hasattr(plan, "__dict__"):
                plan_payload = dict(plan.__dict__)
            else:
                plan_payload = str(plan)
            data["hypothesis"]["plan"] = plan_payload
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionEnvelope":
        """
        Best-effort rehydration from `to_dict()` payload (used for mission persistence/resume).
        """
        if not isinstance(data, dict):
            raise TypeError("MissionEnvelope.from_dict expects dict")

        mid = str(data.get("mission_id") or "")
        if not mid:
            raise ValueError("MissionEnvelope.from_dict missing mission_id")
        original_intent = str(data.get("original_intent") or "")
        if not original_intent:
            original_intent = str(data.get("intent") or data.get("intent_text") or "")

        gov_raw = data.get("governance") or {}
        governance = GovernanceSpec(**gov_raw) if isinstance(gov_raw, dict) else GovernanceSpec()

        hypothesis = None
        hyp_raw = data.get("hypothesis")
        if isinstance(hyp_raw, dict):
            sc_raw = hyp_raw.get("success_contract") or {}
            success_contract = (
                SuccessContract(**sc_raw) if isinstance(sc_raw, dict) else SuccessContract()
            )
            hypothesis = Hypothesis(
                plan=hyp_raw.get("plan"),
                success_contract=success_contract,
                notes=hyp_raw.get("notes") or {},
            )

        exec_log: List[ExecutionEvent] = []
        raw_events = data.get("execution_log") or []
        if isinstance(raw_events, list):
            for ev in raw_events:
                if isinstance(ev, dict):
                    try:
                        exec_log.append(ExecutionEvent(**ev))
                    except Exception:
                        continue

        last_error = None
        raw_err = data.get("last_error")
        if isinstance(raw_err, dict):
            try:
                last_error = MissionError(**raw_err)
            except Exception:
                last_error = None

        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        created_at = float(data.get("created_at") or time.time())

        return cls(
            mission_id=mid,
            original_intent=original_intent,
            governance=governance,
            hypothesis=hypothesis,
            execution_log=exec_log,
            last_error=last_error,
            world_model=data.get("world_model") if isinstance(data.get("world_model"), dict) else None,
            metadata=metadata,
            created_at=created_at,
        )
