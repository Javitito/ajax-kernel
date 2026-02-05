
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os

class RigorStrategy(Enum):
    FAST = "FAST"
    SAFE = "SAFE"
    COUNCIL = "COUNCIL"

@dataclass
class RigorDecision:
    strategy: RigorStrategy
    reason: str
    cost_mode: str
    use_council: bool
    signals: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["strategy"] = self.strategy.value
        return d

def decide_rigor(
    tier: str = "balanced",
    risk: str = "medium",
    fail_count: int = 0,
    cost_mode_override: Optional[str] = None,
    uncertainty_hint: float = 0.0,
    intent_class: Optional[str] = None,
) -> RigorDecision:
    """
    Selector de rigor explícito para AJAX.
    
    Crea una decisión clara: FAST | SAFE | COUNCIL basada en señales.
    - Tier: Calidad del modelo (cheap, balanced, premium).
    - Risk: Nivel de riesgo (low, medium, high).
    - Coste: Modo de coste (save_codex, balanced, premium).
    - Incertidumbre: Basada en fallos previos o hints.
    """
    
    tier = str(tier or "balanced").lower()
    risk = str(risk or "medium").lower()
    cost_mode = cost_mode_override or os.getenv("AJAX_COST_MODE", "premium")
    
    # Mapeo de scores para lógica numérica
    tier_scores = {"cheap": 1, "balanced": 2, "premium": 3}
    risk_scores = {"low": 1, "medium": 2, "high": 3, "critical": 3}
    
    t_score = tier_scores.get(tier, 2)
    r_score = risk_scores.get(risk, 2)
    u_score = min(3, fail_count + (1 if uncertainty_hint > 0.5 else 0))
    
    signals = {
        "tier": tier,
        "risk": risk,
        "fail_count": fail_count,
        "cost_mode": cost_mode,
        "uncertainty_hint": uncertainty_hint,
        "t_score": t_score,
        "r_score": r_score,
        "u_score": u_score
    }
    
    # 0. COUNCIL: intents destructivos siempre requieren Council (antes de cualquier heurística)
    if _is_destructive_intent(intent_class):
        return RigorDecision(
            strategy=RigorStrategy.COUNCIL,
            reason="Destructive intent_class requires Council oversight",
            cost_mode="premium",
            use_council=True,
            signals={**signals, "intent_class": intent_class, "destructive_intent": True},
        )

    # 1. COUNCIL: Máximo rigor
    if r_score >= 3 or u_score >= 2:
        return RigorDecision(
            strategy=RigorStrategy.COUNCIL,
            reason="High risk or high uncertainty requires Council oversight",
            cost_mode="premium",
            use_council=True,
            signals=signals
        )
    
    # 2. SAFE: Rigor intermedio (modelo fuerte, sin necesariamente quorum completo)
    if r_score == 2 or t_score >= 2:
        # Si estamos en modo ahorro, preferimos SAFE sobre COUNCIL pero mantenemos cautela
        effective_council = (cost_mode == "premium")
        return RigorDecision(
            strategy=RigorStrategy.SAFE,
            reason="Medium risk or balanced tier; choosing a safe path",
            cost_mode=cost_mode,
            use_council=effective_council,
            signals=signals
        )
    
    # 3. FAST: Mínimo rigor, máxima velocidad
    return RigorDecision(
        strategy=RigorStrategy.FAST,
        reason="Low risk and low uncertainty; prioritizing speed",
        cost_mode="balanced" if cost_mode != "emergency" else "emergency",
        use_council=False,
        signals=signals
    )


def _is_destructive_intent(intent_class: Optional[str]) -> bool:
    if not intent_class:
        return False
    val = str(intent_class or "").strip().lower()
    destructive_markers = {
        "destructive",
        "delete",
        "remove",
        "wipe",
        "format",
        "drop",
        "purge",
    }
    return val in destructive_markers or val.startswith("delete_") or val.startswith("remove_")
