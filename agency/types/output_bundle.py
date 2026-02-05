"""
OutputBundle - Bundle de salida con dos formatos válidos para Proof-Carrying Output.

Formato A: claims[] - cuando hay afirmaciones con evidencia
Formato B: hypothesis + verification_commands[] - cuando no hay proof
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .claim import Claim


@dataclass(frozen=True)
class ValidationResult:
    """Resultado de validación de un OutputBundle."""

    valid: bool
    reason: Optional[str] = None
    claim: Optional[Claim] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "claim": self.claim.to_dict() if self.claim else None,
        }


@dataclass
class OutputBundle:
    """
    Bundle de salida estructurado.

    Debe cumplir UNO de estos formatos:
    A) claims: Lista de claims con evidence (para afirmaciones verificables)
    B) hypothesis + verification_commands: Cuando no hay proof suficiente

    Args:
        claims: Lista de claims (Formato A)
        hypothesis: Texto de hipótesis (Formato B)
        verification_commands: 1-3 comandos para verificar (Formato B)
    """

    claims: Optional[List[Claim]] = None
    hypothesis: Optional[str] = None
    verification_commands: Optional[List[str]] = None

    def validate(self) -> ValidationResult:
        """
        Validar que el bundle cumple formato A o B.

        Returns:
            ValidationResult con valid=True si es válido, o valid=False con razón
        """
        # Formato A: claims[]
        if self.claims is not None and len(self.claims) > 0:
            for claim in self.claims:
                if not claim.has_minimum_evidence():
                    missing = claim.get_missing_evidence()
                    return ValidationResult(
                        valid=False,
                        reason=f"Claim tipo '{claim.type}' sin evidence mínima requerida: {missing}",
                        claim=claim,
                    )
            return ValidationResult(valid=True)

        # Formato B: hypothesis + verification_commands
        if self.hypothesis and self.verification_commands:
            if len(self.verification_commands) < 1:
                return ValidationResult(
                    valid=False, reason="Debe haber al menos 1 comando de verificación"
                )
            if len(self.verification_commands) > 3:
                return ValidationResult(
                    valid=False, reason="Máximo 3 comandos de verificación permitidos"
                )
            return ValidationResult(valid=True)

        # Ni A ni B
        if self.hypothesis and not self.verification_commands:
            return ValidationResult(
                valid=False, reason="Hipótesis requiere verification_commands (1-3 comandos)"
            )

        if self.verification_commands and not self.hypothesis:
            return ValidationResult(
                valid=False, reason="Commands de verificación requieren hypothesis"
            )

        return ValidationResult(
            valid=False, reason="Debe proporcionar claims[] O hypothesis + verification_commands[]"
        )

    def is_claims_format(self) -> bool:
        """True si usa formato A (claims)."""
        return self.claims is not None and len(self.claims) > 0

    def is_hypothesis_format(self) -> bool:
        """True si usa formato B (hypothesis)."""
        return self.hypothesis is not None and self.verification_commands is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serializar a dict para JSON."""
        return {
            "claims": [c.to_dict() for c in self.claims] if self.claims else None,
            "hypothesis": self.hypothesis,
            "verification_commands": self.verification_commands,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputBundle":
        """Deserializar desde dict."""
        claims = None
        if data.get("claims"):
            claims = [Claim.from_dict(c) for c in data["claims"]]
        return cls(
            claims=claims,
            hypothesis=data.get("hypothesis"),
            verification_commands=data.get("verification_commands"),
        )

    @classmethod
    def from_claims(cls, claims: List[Claim]) -> "OutputBundle":
        """Factory method para crear bundle desde claims."""
        return cls(claims=claims)

    @classmethod
    def from_hypothesis(cls, hypothesis: str, commands: List[str]) -> "OutputBundle":
        """Factory method para crear bundle desde hipótesis."""
        return cls(hypothesis=hypothesis, verification_commands=commands)
