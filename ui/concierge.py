from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ConciergeState:
    has_pending_mission: bool
    mission_id: Optional[str] = None
    intention: Optional[str] = None
    status: Optional[str] = None
    expects: Optional[str] = None
    question: Optional[str] = None
    budget_exhausted: bool = False
    lab_job_id: Optional[str] = None
    parked_by_user: bool = False
    repeat_count: int = 0
    loop_guard: bool = False


class Concierge:
    def __init__(
        self,
        tone: str = "neutral",
        mode: str = "heuristic",
        local_llm: Optional[Callable[[str, ConciergeState], Dict[str, Any]]] = None,
    ) -> None:
        self.tone = (tone or "neutral").strip().lower()
        self.mode = (mode or "heuristic").strip().lower()
        self._local_llm = local_llm

    def build_banner(self, state: ConciergeState, *, mode: str = "chat") -> Dict[str, Any]:
        lines: List[str] = []
        suggestions: List[str] = []
        if not state.has_pending_mission:
            return {"message": "", "suggestions": []}
        mission_id = state.mission_id or "desconocida"
        status = (state.status or "WAITING_FOR_USER").strip().upper()
        lines.append(f"Tienes 1 misión pendiente: {mission_id} ({status})")
        if mode == "chat":
            suggestions.extend([":mission", ":park", ":close_manual_done", ":open_incident"])
        else:
            suggestions.extend([":chat", ":park"])
        if state.parked_by_user and ":resume" not in suggestions:
            suggestions.insert(0, ":resume")
        if state.loop_guard and mode == "chat":
            suggestions = [":close_manual_done", ":open_incident", ":retry"]
        if state.budget_exhausted:
            suggestions = [s for s in suggestions if s not in {":retry", ":use_deterministic_recipe"}]
            if ":park" not in suggestions:
                suggestions.append(":park")
        if suggestions:
            lines.append("Opciones: " + " | ".join(suggestions))
        return {"message": "\n".join(lines), "suggestions": suggestions}

    def interpret(self, text: str, state: ConciergeState, *, mode: str = "chat") -> Dict[str, Any]:
        if not text:
            return {}
        raw = text.strip()
        if not raw:
            return {}
        lowered = raw.lower()
        if lowered.startswith(("/", ":", "[")):
            return {}
        if any(tok in lowered for tok in {"con quién", "con quien", "quién eres", "quien eres", "who are you"}):
            return {
                "intent": "concierge_message",
                "ui_message": f"Estás en {mode.upper()}. Misión pendiente: {state.mission_id or 'ninguna'}.",
                "ui_suggestions": [":mission", ":park"],
            }
        if any(tok in lowered for tok in {"aparcar", "aparca", "park"}):
            return {"intent": "park"}
        if any(tok in lowered for tok in {"incidente", "incident"}):
            return {"intent": "open_incident"}
        if any(tok in lowered for tok in {"cerrar", "close", "hecho", "manual"}):
            return {"intent": "close_manual_done"}
        if any(tok in lowered for tok in {"mision", "mission", "reanudar", "resume", "continuar"}):
            return {"intent": "mission"}
        if state.expects == "user_answer":
            for prefix in ("respuesta:", "respuesta ", "answer:", "answer "):
                if lowered.startswith(prefix):
                    answer_text = raw[len(prefix) :].strip()
                    if answer_text:
                        return {
                            "intent": "mission_answer",
                            "answer_text": answer_text,
                            "explicit": True,
                            "ui_dispatch": f"[answer] {answer_text}",
                        }
            return {
                "intent": "mission_answer",
                "answer_text": raw,
                "explicit": False,
                "ui_dispatch": f"[answer] {raw}",
            }
        if self.mode == "local_llm" and self._local_llm is not None:
            try:
                resp = self._local_llm(raw, state)
                if isinstance(resp, dict):
                    return resp
            except Exception:
                return {}
        return {}
