"""
agency.types - Tipos para Proof-Carrying Output.

Este paquete define los tipos base para implementar la disciplina anti-optimismo
de AGENTS.md: ninguna afirmación "confirmada" sin evidencia verificable.
"""

from .evidence import EvidenceRef
from .claim import Claim
from .output_bundle import OutputBundle, ValidationResult
from .minimums import (
    CLAIM_EVIDENCE_MINIMUMS,
    ALTERNATIVE_EVIDIDENCES,
    VALID_EVIDENCE_KINDS,
    get_minimum_evidence,
    validate_evidence_minimums,
    get_missing_evidence,
    is_valid_evidence_kind,
)

__all__ = [
    # Tipos principales
    "EvidenceRef",
    "Claim",
    "OutputBundle",
    "ValidationResult",
    # Constantes y utilidades
    "CLAIM_EVIDENCE_MINIMUMS",
    "ALTERNATIVE_EVIDIDENCES",
    "VALID_EVIDENCE_KINDS",
    "get_minimum_evidence",
    "validate_evidence_minimums",
    "get_missing_evidence",
    "is_valid_evidence_kind",
]
