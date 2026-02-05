"""
AntiOptimismGuard - Guardia contra optimismo no probado.

Meta: Hacer imposible (o muy caro) que un modelo cierre "confirmado/arreglado/culpable"
sin pruebas verificables.

Según AGENTS.md §X (Proof-Carrying Output):
- Ninguna afirmación "confirmada" sin EvidenceRefs + EFE
- Sin evidence → degradar a HIPÓTESIS + comandos de verificación
- En rail=prod → SOFT_BLOCK (no "entregar feliz")
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from agency.types import OutputBundle, Claim, EvidenceRef, ValidationResult
from agency.types.minimums import get_missing_evidence


@dataclass
class GuardResult:
    """
    Resultado de validación por el AntiOptimismGuard.

    Args:
        approved: True si pasa el guard, False si es bloqueado/degradado
        bundle: El bundle (original o degradado)
        action: Acción tomada (pass, SOFT_BLOCK, DEGRADED_TO_HYPOTHESIS, require_hypothesis)
        reason: Razón de la acción
        receipt_path: Path al receipt generado (si aplica)
    """

    approved: bool
    bundle: Optional[OutputBundle]
    action: str
    reason: Optional[str] = None
    receipt_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "action": self.action,
            "reason": self.reason,
            "receipt_path": self.receipt_path,
            "bundle": self.bundle.to_dict() if self.bundle else None,
        }


class AntiOptimismGuard:
    """
    Guardia que valida OutputBundles y degrada claims sin evidence mínima.

    Uso:
        guard = AntiOptimismGuard(rail="prod")
        result = guard.validate_bundle(bundle, original_text="El problema está resuelto")

        if result.action == "SOFT_BLOCK":
            # En prod: no continuar
            return soft_block_response(result)
        elif result.action == "DEGRADED_TO_HYPOTHESIS":
            # En lab: continuar con bundle degradado
            return result.bundle
    """

    def __init__(self, rail: str = "prod", receipts_dir: Optional[str] = None):
        """
        Inicializar el guard.

        Args:
            rail: "prod" o "lab" - determina política de bloqueo
            receipts_dir: Directorio para receipts (default: artifacts/receipts)
        """
        self.rail = rail
        if receipts_dir:
            base_dir = Path(receipts_dir)
        else:
            override = os.getenv("AJAX_ARTIFACTS_DIR")
            base_dir = Path(override) / "receipts" if override else Path("artifacts/receipts")
        self.receipts_dir = base_dir
        self.receipts_dir.mkdir(parents=True, exist_ok=True)

    def validate_bundle(self, bundle: OutputBundle, original_text: str = "") -> GuardResult:
        """
        Validar un OutputBundle y aplicar degradación si es necesario.

        Args:
            bundle: El bundle a validar
            original_text: Texto original del reporte (para el receipt)

        Returns:
            GuardResult con la acción tomada
        """
        validation = bundle.validate()

        if validation.valid:
            # Bundle válido - pasar
            return GuardResult(approved=True, bundle=bundle, action="pass")

        # Si hay claims sin mínimos → degradar
        if bundle.claims:
            return self._degrade_claims(bundle, original_text, validation)

        # Si no hay formato válido → exigir hipótesis
        return GuardResult(
            approved=False,
            bundle=None,
            action="require_hypothesis",
            reason="Debe proporcionar claims con evidence O hypothesis + verification_commands",
        )

    def _degrade_claims(
        self, bundle: OutputBundle, original_text: str, validation: ValidationResult
    ) -> GuardResult:
        """
        Degradar claims a hipótesis + comandos de verificación.
        """
        # Generar comandos de verificación sugeridos
        suggested_commands = self._generate_verification_commands(bundle)

        # Crear bundle degradado
        missing_evidence = self._extract_missing_evidence(bundle)
        degraded_bundle = OutputBundle(
            hypothesis=f"HIPÓTESIS (evidence incompleta): {validation.reason}",
            verification_commands=suggested_commands,
        )

        # Escribir receipt
        receipt_path = self._write_receipt(
            original_text=original_text,
            missing_evidence=missing_evidence,
            suggested_commands=suggested_commands,
            reason=validation.reason,
        )

        # En prod o rigor alto: SOFT_BLOCK
        if self.rail == "prod":
            return GuardResult(
                approved=False,
                bundle=degraded_bundle,
                action="SOFT_BLOCK",
                reason=f"Claim sin evidence mínima en rail=prod: {validation.reason}",
                receipt_path=receipt_path,
            )

        # En lab: degradar pero permitir continuar
        return GuardResult(
            approved=True,
            bundle=degraded_bundle,
            action="DEGRADED_TO_HYPOTHESIS",
            reason=validation.reason,
            receipt_path=receipt_path,
        )

    def _generate_verification_commands(self, bundle: OutputBundle) -> List[str]:
        """
        Generar comandos de verificación según el contexto del bundle.
        """
        commands = []

        # Comando base: listar receipts
        commands.append(
            "ls -la artifacts/receipts/ 2>/dev/null | head -20 || echo 'No receipts dir'"
        )

        # Si hay claims, sugerir verificación específica
        if bundle.claims:
            for claim in bundle.claims:
                if claim.type == "fixed":
                    commands.append("python -m pytest tests/ -v --tb=short 2>&1 | tail -20")
                elif claim.type == "root_cause":
                    commands.append(
                        "find artifacts -name '*.log' -type f -exec tail -20 {} \\; 2>/dev/null"
                    )
                elif claim.type == "available_green":
                    commands.append(
                        "cat artifacts/providers_status.json 2>/dev/null || echo 'No status file'"
                    )

        return commands[:3]  # Máximo 3 comandos

    def _write_receipt(
        self,
        original_text: str,
        missing_evidence: List[str],
        suggested_commands: List[str],
        reason: str,
    ) -> str:
        """
        Escribir receipt de degradación.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        receipt_file = self.receipts_dir / f"anti_optimism_degraded_{timestamp}.json"

        receipt = {
            "timestamp": timestamp,
            "rail": self.rail,
            "guard_version": "1.0",
            "original_text": original_text[:500] if original_text else "",  # Truncar por seguridad
            "degradation_reason": reason,
            "missing_evidence": missing_evidence,
            "suggested_commands": suggested_commands,
        }

        with open(receipt_file, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, ensure_ascii=False)

        return str(receipt_file)

    def _extract_missing_evidence(self, bundle: OutputBundle) -> List[str]:
        """
        Extraer lista de evidence faltante de todos los claims.
        """
        missing = []
        if bundle.claims:
            for claim in bundle.claims:
                claim_missing = claim.get_missing_evidence()
                for m in claim_missing:
                    missing.append(f"{claim.type}: {m}")
        return missing

    def format_soft_block(self, result: GuardResult) -> str:
        """
        Formatear respuesta de SOFT_BLOCK para el usuario.
        """
        return f"""
[SOFT_BLOCK] Afirmación sin evidencia suficiente

Razón: {result.reason}

El sistema no permite cerrar con "confirmado/arreglado" sin:
1. EvidenceRefs tipadas (log, receipt, verify_result, etc.)
2. EFE (Expected Final State) verificado

Acciones posibles:
- Añadir evidence faltante y reintentar
- Degradar a hipótesis con comandos de verificación
- Revisar receipt: {result.receipt_path}
""".strip()

    def format_degraded(self, result: GuardResult) -> str:
        """
        Formatear respuesta degradada para el usuario.
        """
        if not result.bundle:
            return "Error: Bundle degradado no disponible"

        return f"""
[DEGRADED_TO_HYPOTHESIS] Afirmación convertida a hipótesis

{result.bundle.hypothesis}

Comandos sugeridos para verificar:
{chr(10).join(f"  $ {cmd}" for cmd in result.bundle.verification_commands or [])}

Receipt: {result.receipt_path}
""".strip()


# Singleton para uso global
_default_guard: Optional[AntiOptimismGuard] = None


def get_guard(rail: str = "prod") -> AntiOptimismGuard:
    """Obtener instancia del guard (singleton por rail)."""
    global _default_guard
    if _default_guard is None or _default_guard.rail != rail:
        _default_guard = AntiOptimismGuard(rail=rail)
    return _default_guard


def validate_output(
    bundle: OutputBundle, original_text: str = "", rail: str = "prod"
) -> GuardResult:
    """
    Función de conveniencia para validar un bundle.

    Args:
        bundle: Bundle a validar
        original_text: Texto original del reporte
        rail: "prod" o "lab"

    Returns:
        GuardResult con la acción tomada
    """
    guard = get_guard(rail)
    return guard.validate_bundle(bundle, original_text)
