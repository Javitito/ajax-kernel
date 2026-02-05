"""
EFE-Repair Loop: Reparación constitucional de planes sin expected_state.

Meta: Que missing_efe no termine en frustración; termine en re-draft
constitucional y/o GAP con pasos.

Si el plan viene sin expected_state/success_spec:
1. No ejecutar (principio constitucional)
2. Intentar reparación JSON-only (max 2 intentos, timeout duro)
3. Si falla: GAP_LOGGED:missing_efe_final con receipt
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import json
from pathlib import Path
import time
import os


@dataclass
class RepairResult:
    """Resultado del intento de reparación."""

    success: bool
    plan: Optional[Dict[str, Any]] = None
    attempts: int = 0
    gap_logged: bool = False
    receipt_path: Optional[str] = None
    reason: Optional[str] = None


class EFERepairLoop:
    """
    Loop de reparación para planes incompletos.

    Principio: "No ejecutar sin contrato de verificación",
    pero tampoco "rendirse": intentar completar el contrato.
    """

    MAX_ATTEMPTS = 2
    TIMEOUT_SECONDS = 30
    MAX_TOKENS = 500  # Bajo para evitar inventos

    def __init__(self, receipts_dir: Optional[str] = None):
        if receipts_dir:
            base_dir = Path(receipts_dir)
        else:
            override = os.getenv("AJAX_ARTIFACTS_DIR")
            base_dir = Path(override) / "receipts" if override else Path("artifacts/receipts")
        self.receipts_dir = base_dir
        self.receipts_dir.mkdir(parents=True, exist_ok=True)

    def repair_plan(
        self, plan: Dict[str, Any], drafter_fn: Optional[Callable] = None
    ) -> RepairResult:
        """
        Intentar reparar un plan incompleto.

        Args:
            plan: Plan JSON sin expected_state
            drafter_fn: Función que llama al drafter (inyectable para tests)

        Returns:
            RepairResult con plan reparado o GAP_LOGGED
        """
        if self._has_expected_state(plan):
            return RepairResult(success=True, plan=plan, attempts=0)

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                repaired = self._attempt_repair(plan, attempt, drafter_fn)

                if repaired and self._has_expected_state(repaired):
                    # Éxito: escribir receipt y retornar
                    receipt_path = self._write_success_receipt(plan, repaired, attempt)
                    return RepairResult(
                        success=True, plan=repaired, attempts=attempt, receipt_path=receipt_path
                    )

            except TimeoutError:
                continue
            except Exception as e:
                # Log error pero continuar intentos
                continue

        # Agotados intentos → GAP_LOGGED
        return self._log_gap(plan)

    def _has_expected_state(self, plan: Dict[str, Any]) -> bool:
        """Verificar si el plan tiene expected_state definido (root o por step)."""
        if not isinstance(plan, dict):
            return False
        expected = plan.get("expected_state")
        if expected is not None and isinstance(expected, dict) and len(expected) > 0:
            return True

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return False

        def _expected_state_has_checks(val: Any) -> bool:
            if not isinstance(val, dict):
                return False
            if val.get("windows"):
                return True
            if val.get("files"):
                return True
            if isinstance(val.get("checks"), list) and val.get("checks"):
                return True
            meta = val.get("meta") or {}
            return bool(isinstance(meta, dict) and meta.get("must_be_active"))

        for step in steps:
            if not isinstance(step, dict):
                return False
            action = step.get("action")
            succ = step.get("success_spec")
            if action == "await_user_input":
                continue
            if not isinstance(succ, dict):
                return False
            expected_state = succ.get("expected_state")
            if not _expected_state_has_checks(expected_state):
                return False
        return True

    def _attempt_repair(
        self, plan: Dict[str, Any], attempt: int, drafter_fn: Optional[Callable] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Un intento de reparación JSON-only.

        Prompt quirúrgico: "Tu plan es válido pero le falta contrato de
        verificación (EFE). Completa expected_state sin cambiar pasos."
        """
        # Construir prompt quirúrgico
        prompt = self._build_repair_prompt(plan, attempt)

        # Llamar al drafter (o función mock para tests)
        if drafter_fn:
            try:
                # Aplicar timeout manual si es posible
                start = time.time()
                result = drafter_fn(prompt, plan)
                elapsed = time.time() - start
                if elapsed > self.TIMEOUT_SECONDS:
                    raise TimeoutError(f"Repair timeout: {elapsed}s")
                return result
            except Exception:
                raise

        # Implementación real: llamar al drafter del sistema
        return self._call_drafter(prompt, plan)

    def _build_repair_prompt(self, plan: Dict[str, Any], attempt: int) -> str:
        """Construir prompt quirúrgico para el drafter."""
        plan_json = json.dumps(plan, indent=2, ensure_ascii=False)

        return f"""[EFE-REPAIR attempt {attempt}/{self.MAX_ATTEMPTS}]

Tu plan es válido pero le falta el contrato de verificación (EFE - Expected Final State).

REGLAS:
1. NO cambies los pasos del plan
2. NO inventes hechos ni resultados
3. SÓLO añade expected_state: qué estado debe existir para considerar éxito

Plan actual:
{plan_json}

Responde ÚNICAMENTE con JSON válido:
{{
  "expected_state": {{
    "artifacts": ["path/to/expected/artifact.json"],
    "conditions": ["descripción de condición verificable"],
    "verify_command": "comando para verificar"
  }}
}}
"""

    def _call_drafter(self, prompt: str, original_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Llamar al drafter del sistema.

        En implementación real, esto llamaría a:
        - agency.brain_router para seleccionar modelo
        - agency.drafter para completar el plan

        Por ahora, retorna None para forzar GAP_LOGGED en producción
        hasta que se integre con el brain_router.
        """
        # TODO: Integrar con agency.brain_router
        # from agency.brain_router import BrainRouter
        # router = BrainRouter()
        # response = router.complete(prompt, max_tokens=self.MAX_TOKENS)
        # return self._parse_drafter_response(response, original_plan)
        return None

    def _parse_drafter_response(
        self, response: str, original_plan: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Parsear respuesta del drafter y mergear con plan original."""
        try:
            # Extraer JSON de la respuesta
            data = json.loads(response)

            if "expected_state" in data:
                # Mergear con plan original
                repaired = original_plan.copy()
                repaired["expected_state"] = data["expected_state"]
                return repaired
        except json.JSONDecodeError:
            pass

        return None

    def _write_success_receipt(
        self, original: Dict[str, Any], repaired: Dict[str, Any], attempt: int
    ) -> str:
        """Escribir receipt de repair exitoso."""
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        receipt_file = self.receipts_dir / f"plan_repaired_{timestamp}.json"

        receipt = {
            "timestamp": timestamp,
            "repair_version": "1.0",
            "attempt": attempt,
            "success": True,
            "original_plan_summary": self._summarize_plan(original),
            "repaired_plan_summary": self._summarize_plan(repaired),
            "changes": ["expected_state added"],
            "efe_preview": str(repaired.get("expected_state", {}))[:200],
        }

        with open(receipt_file, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, ensure_ascii=False)

        return str(receipt_file)

    def _log_gap(self, plan: Dict[str, Any]) -> RepairResult:
        """
        Loggear GAP cuando el repair falla.

        GAP_LOGGED:missing_efe_final con receipt.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        receipt_file = self.receipts_dir / f"missing_efe_final_{timestamp}.json"

        receipt = {
            "timestamp": timestamp,
            "gap_type": "missing_efe_final",
            "attempts": self.MAX_ATTEMPTS,
            "plan_summary": self._summarize_plan(plan),
            "failure_reason": "Agotados intentos de repair automático",
            "next_steps": [
                "1. Revisar plan manualmente",
                "2. Añadir expected_state con criterios verificables",
                "3. Re-submit con EFE completo",
            ],
            "repair_version": "1.0",
        }

        with open(receipt_file, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, ensure_ascii=False)

        return RepairResult(
            success=False,
            attempts=self.MAX_ATTEMPTS,
            gap_logged=True,
            receipt_path=str(receipt_file),
            reason="Agotados intentos de repair, EFE no completable automáticamente",
        )

    def _summarize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Crear resumen seguro del plan para el receipt."""
        steps = plan.get("steps", [])
        return {
            "steps_count": len(steps) if isinstance(steps, list) else 0,
            "has_expected_state": self._has_expected_state(plan),
            "goal": str(plan.get("goal", "N/A"))[:100],  # Truncar por seguridad
        }


# Singleton para uso global
_default_repair_loop: Optional[EFERepairLoop] = None


def get_repair_loop(receipts_dir: Optional[str] = None) -> EFERepairLoop:
    """Obtener instancia del repair loop (singleton)."""
    global _default_repair_loop
    if _default_repair_loop is None:
        _default_repair_loop = EFERepairLoop(receipts_dir=receipts_dir)
    return _default_repair_loop


def repair_plan_if_needed(
    plan: Dict[str, Any],
    drafter_fn: Optional[Callable] = None,
    receipts_dir: Optional[str] = None,
) -> RepairResult:
    """
    Función de conveniencia para reparar un plan si es necesario.

    Args:
        plan: Plan a validar/reparar
        drafter_fn: Función opcional para llamar al drafter
        receipts_dir: Directorio para receipts

    Returns:
        RepairResult con plan reparado o GAP_LOGGED
    """
    loop = get_repair_loop(receipts_dir)
    return loop.repair_plan(plan, drafter_fn)


def validate_plan_has_efe(plan: Dict[str, Any]) -> bool:
    """
    Validar rápidamente si un plan tiene EFE.

    Args:
        plan: Plan a validar

    Returns:
        True si tiene expected_state válido
    """
    loop = get_repair_loop()
    return loop._has_expected_state(plan)
