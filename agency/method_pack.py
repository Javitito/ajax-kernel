from __future__ import annotations


AJAX_METHOD_PACK = """
AJAX_METHOD_PACK (canon, short)
- ASK_USER si falta info (no asumir).
- Atomicidad: cada step es una tarea completa con preconditions + evidence_required + success_spec; on_fail=abort.
- Pursuit ante fallos técnicos: reintenta seguro, escala modelo o repara infra; no degradar en silencio.
- EFE > percepción: PASS/FAIL solo por checks falsables; estados dinámicos => >=2 señales independientes.
- Orden/cleanup: si no PASS, revertir/limpiar lo creado por la misión y restaurar el entorno hacia Snapshot-0.
- Anti-overfitting: no hardcodear apps/URLs/proveedores; extraer patrones reutilizables.
""".strip()

