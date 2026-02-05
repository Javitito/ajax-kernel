"""Agency contract definitions shared by all CLI agents.

This module defines the JSON contracts described in the LEANN agency proposal.
They are intentionally typed and validated for quick fail-fast behaviour before
handing jobs to external CLIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class AgencyBudget:
    """Execution limits for an agent."""

    steps: int = 1
    seconds: int = 60
    tokens: int = 4000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencyBudget":
        return cls(
            steps=int(data.get("steps", 1)),
            seconds=int(data.get("seconds", 60)),
            tokens=int(data.get("tokens", 4000)),
        )

    def to_dict(self) -> Dict[str, int]:
        return {"steps": self.steps, "seconds": self.seconds, "tokens": self.tokens}


@dataclass
class AgencyPolicy:
    """Policy toggles passed to agents, preserving extra keys for flexibility."""

    safety: str = "strict"
    confirm_sensitive: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencyPolicy":
        data = data or {}
        safety = str(data.get("safety", "strict"))
        confirm_sensitive = bool(data.get("confirm_sensitive", True))
        extras = {
            key: value
            for key, value in data.items()
            if key not in {"safety", "confirm_sensitive"}
        }
        return cls(safety=safety, confirm_sensitive=confirm_sensitive, extras=extras)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"safety": self.safety, "confirm_sensitive": self.confirm_sensitive}
        payload.update(self.extras)
        return payload


@dataclass
class AgencyContext:
    """Context bundle containing RAG passages and note snippets."""

    rag: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgencyContext":
        if not data:
            return cls()
        rag_items: List[Dict[str, Any]] = []
        for passage in data.get("rag", []):
            if isinstance(passage, dict):
                rag_items.append({"text": passage.get("text", ""), "source": passage.get("source")})
            else:
                rag_items.append({"text": str(passage)})

        notes = [str(note) for note in data.get("notes", []) if note]
        return cls(rag=rag_items, notes=notes)

    def to_dict(self) -> Dict[str, Any]:
        return {"rag": self.rag, "notes": self.notes}


@dataclass
class AgencyJob:
    """Canonical job description consumed by any agent CLI."""

    job_id: str
    goal: str
    context: AgencyContext = field(default_factory=AgencyContext)
    capabilities: List[str] = field(default_factory=list)
    budget: AgencyBudget = field(default_factory=AgencyBudget)
    policy: AgencyPolicy = field(default_factory=AgencyPolicy)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencyJob":
        missing = [key for key in ("id", "goal") if not data.get(key)]
        if missing:
            raise ValueError(f"Job missing required fields: {missing}")

        capabilities = [str(item) for item in data.get("capabilities", [])]

        metadata = dict(data.get("metadata", {})) if isinstance(data.get("metadata"), dict) else {}
        extras = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "id",
                "goal",
                "context",
                "capabilities",
                "budget",
                "policy",
                "metadata",
            }
        }
        metadata.update(extras)

        return cls(
            job_id=str(data["id"]),
            goal=str(data["goal"]),
            context=AgencyContext.from_dict(data.get("context")),
            capabilities=capabilities,
            budget=AgencyBudget.from_dict(data.get("budget", {})),
            policy=AgencyPolicy.from_dict(data.get("policy", {})),
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: Path) -> "AgencyJob":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.job_id,
            "goal": self.goal,
            "context": self.context.to_dict(),
            "capabilities": self.capabilities,
            "budget": self.budget.to_dict(),
            "policy": self.policy.to_dict(),
        }
        if self.metadata:
            data.update(self.metadata)
        return data

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, payload: Any) -> "ToolCall":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict) and "tool" in payload:
            tool = str(payload["tool"])
            args = payload.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError("tool args must be an object")
            return cls(tool=tool, args=args)
        raise ValueError(f"Invalid tool payload: {payload!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "args": self.args}


@dataclass
class AgentArtifact:
    type: str
    path: str
    description: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, payload: Any) -> "AgentArtifact":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict) and payload.get("type") and payload.get("path"):
            meta = {
                key: value
                for key, value in payload.items()
                if key not in {"type", "path", "description"}
            }
            return cls(
                type=str(payload["type"]),
                path=str(payload["path"]),
                description=payload.get("description"),
                meta=meta,
            )
        raise ValueError(f"Invalid artifact payload: {payload!r}")

    def to_dict(self) -> Dict[str, Any]:
        data = {"type": self.type, "path": self.path}
        if self.description:
            data["description"] = self.description
        if self.meta:
            data.update(self.meta)
        return data


@dataclass
class AgentMetrics:
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentMetrics":
        if not data:
            return cls()
        return cls(
            tokens_in=int(data.get("tokens_in", 0)),
            tokens_out=int(data.get("tokens_out", 0)),
            latency_ms=int(data.get("lat_ms") or data.get("latency_ms", 0)),
            extra={k: v for k, v in data.items() if k not in {"tokens_in", "tokens_out", "lat_ms", "latency_ms"}},
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
        }
        payload.update(self.extra)
        return payload


@dataclass
class AgencyResult:
    ok: bool
    answer: str
    actions: List[ToolCall] = field(default_factory=list)
    artifacts: List[AgentArtifact] = field(default_factory=list)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    confidence: float = 0.0
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgencyResult":
        actions = [ToolCall.from_any(item) for item in data.get("actions", [])]
        artifacts = [AgentArtifact.from_any(item) for item in data.get("artifacts", [])]
        return cls(
            ok=bool(data.get("ok", False)),
            answer=str(data.get("answer", "")),
            actions=actions,
            artifacts=artifacts,
            metrics=AgentMetrics.from_dict(data.get("metrics")),
            confidence=float(data.get("confidence", 0.0)),
            errors=[str(err) for err in data.get("errors", [])],
        )

    @classmethod
    def load(cls, path: Path) -> "AgencyResult":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "answer": self.answer,
            "actions": [action.to_dict() for action in self.actions],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metrics": self.metrics.to_dict(),
            "confidence": self.confidence,
            "errors": self.errors,
        }

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_capabilities(capabilities: Iterable[str]) -> List[str]:
    """Deduplicate and normalise capability identifiers."""

    seen = set()
    normalised: List[str] = []
    for capability in capabilities:
        slug = str(capability).strip()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        normalised.append(slug)
    return normalised


__all__ = [
    "AgencyJob",
    "AgencyResult",
    "AgencyBudget",
    "AgencyPolicy",
    "AgencyContext",
    "ToolCall",
    "AgentArtifact",
    "AgentMetrics",
    "ensure_capabilities",
]
