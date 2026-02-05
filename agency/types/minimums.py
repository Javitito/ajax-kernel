"""
Tabla de mínimos de evidencia por tipo de Claim.

Esto evita "evidence theatre" donde se incluyen paths irrelevantes.
Cada tipo de claim debe cumplir con evidence específica para ser válido.
"""

from typing import List, Dict, Set


# Mínimos requeridos por tipo de claim
# Cada claim type debe tener al menos UNO de los kinds listados
CLAIM_EVIDENCE_MINIMUMS: Dict[str, List[str]] = {
    "fixed": [
        "verify_result",  # Resultado de verificación (PASS/FAIL)
        "efe",  # Expected Final State cumplido
    ],
    "root_cause": [
        "receipt",  # Receipt de la acción diagnóstica
        "log",  # Log con evidencia del análisis
    ],
    "available_green": [
        "providers_quota.json",  # Estado de cuotas
        "providers_status.json",  # Estado de providers
    ],
    "diagnosed": [
        "snapshot",  # Estado capturado del sistema
        "log",  # Log del diagnóstico
    ],
    "verified": [
        "verify_result"  # Resultado explícito de verificación
    ],
    "confirmed": ["verify_result", "efe"],
}

# Evidencias alternativas (OR lógico interno)
# Para algunos claims, hay múltiples combinaciones válidas
ALTERNATIVE_EVIDIDENCES: Dict[str, List[List[str]]] = {
    "root_cause": [
        # Opción A: receipt + log
        ["receipt", "log"],
        # Opción B: receipt + state_before + state_after
        ["receipt", "state_before", "state_after"],
    ]
}


def get_minimum_evidence(claim_type: str) -> List[str]:
    """Obtener lista de evidence kinds mínimos para un tipo de claim."""
    return CLAIM_EVIDENCE_MINIMUMS.get(claim_type, [])


def validate_evidence_minimums(claim_type: str, evidence_kinds: Set[str]) -> bool:
    """
    Validar si un conjunto de evidence kinds cumple los mínimos.

    Args:
        claim_type: Tipo de claim (fixed, root_cause, etc.)
        evidence_kinds: Set de kinds presentes en el claim

    Returns:
        True si cumple mínimos, False otherwise
    """
    # Verificar si hay alternativas definidas
    if claim_type in ALTERNATIVE_EVIDIDENCES:
        for alternative in ALTERNATIVE_EVIDIDENCES[claim_type]:
            if all(req in evidence_kinds for req in alternative):
                return True
        return False

    # Validación estándar: todos los mínimos deben estar presentes
    required = set(CLAIM_EVIDENCE_MINIMUMS.get(claim_type, []))
    if not required:
        # Si no hay mínimos definidos, cualquier evidence es válida
        return len(evidence_kinds) > 0

    return required.issubset(evidence_kinds)


def get_missing_evidence(claim_type: str, evidence_kinds: Set[str]) -> List[str]:
    """
    Obtener lista de evidence kinds faltantes.

    Args:
        claim_type: Tipo de claim
        evidence_kinds: Set de kinds presentes

    Returns:
        Lista de kinds faltantes
    """
    required = set(CLAIM_EVIDENCE_MINIMUMS.get(claim_type, []))
    return list(required - evidence_kinds)


# Kinds de evidence válidos (para validación)
VALID_EVIDENCE_KINDS: Set[str] = {
    "log",
    "receipt",
    "verify_result",
    "efe",
    "snapshot",
    "state_before",
    "state_after",
    "providers_quota.json",
    "providers_status.json",
    "artifact",
    "screenshot",
    "diff",
}


def is_valid_evidence_kind(kind: str) -> bool:
    """Verificar si un kind de evidence es válido."""
    return kind in VALID_EVIDENCE_KINDS
