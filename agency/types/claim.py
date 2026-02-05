"""
Claim - Afirmación con evidencia asociada para Proof-Carrying Output.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from .evidence import EvidenceRef
from .minimums import validate_evidence_minimums, get_missing_evidence


@dataclass(frozen=True)
class Claim:
    """
    Afirmación con evidencia asociada.

    Args:
        type: Tipo de claim (fixed, root_cause, available_green, diagnosed, verified)
        statement: Texto de la afirmación
        evidence_refs: Lista de evidencias que soportan el claim
        efe_ref: Referencia al Expected Final State verificado (opcional)
    """

    type: str
    statement: str
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    efe_ref: Optional[str] = None

    def has_minimum_evidence(self) -> bool:
        """
        Verificar si el claim cumple con los mínimos de evidence para su tipo.

        Returns:
            True si tiene evidence suficiente, False otherwise
        """
        evidence_kinds = {e.kind for e in self.evidence_refs}
        return validate_evidence_minimums(self.type, evidence_kinds)

    def get_missing_evidence(self) -> List[str]:
        """
        Obtener lista de evidence kinds faltantes.

        Returns:
            Lista de kinds requeridos pero no presentes
        """
        evidence_kinds = {e.kind for e in self.evidence_refs}
        return get_missing_evidence(self.type, evidence_kinds)

    def get_evidence_kinds(self) -> Set[str]:
        """Obtener set de kinds de evidence presentes."""
        return {e.kind for e in self.evidence_refs}

    def validate_paths(self) -> List[str]:
        """
        Validar que todos los paths de evidence existen.

        Returns:
            Lista de paths que NO existen (vacía si todo OK)
        """
        missing = []
        for ev in self.evidence_refs:
            if not ev.validate_exists():
                missing.append(ev.path)
        return missing

    def to_dict(self) -> Dict[str, Any]:
        """Serializar a dict para JSON."""
        return {
            "type": self.type,
            "statement": self.statement,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "efe_ref": self.efe_ref,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Deserializar desde dict."""
        return cls(
            type=data["type"],
            statement=data["statement"],
            evidence_refs=[EvidenceRef.from_dict(e) for e in data.get("evidence_refs", [])],
            efe_ref=data.get("efe_ref"),
        )
