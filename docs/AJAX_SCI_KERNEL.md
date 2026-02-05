# AJAX Scientific Kernel (A–F)

Este documento define el ciclo de vida de ejecución técnica de AJAX. Ninguna misión en PROD puede saltarse este flujo sin evidencia explícita.

## El Protocolo A–F

### A. Normalize (Normalización)
- **Input**: Texto libre del usuario o JSON crudo.
- **Acción**: Clasificar `intent_class`, `task_tier` y `risk_level`.
- **Invariante**: MUST ser agnóstico de dominio (CG-1).

### B. Retrieve (Recuperación)
- **Acción**: Consultar LEANN (Memoria), ActionCatalog (Capacidades) y MotorMemory (Patrones).
- **Output**: Contexto enriquecido para la planificación.

### C. Generate (Generación de Plan)
- **Acción**: BrainRouter produce un plan estructurado (`AjaxPlan`).
- **Invariante**: MUST incluir un contrato de éxito (`ExpectedState`) verificable.

### D. Execute (Ejecución)
- **Acción**: `PlanRunner` despacha las acciones al `Actuator`.
- **Rigor**: `FAST` | `SAFE` | `COUNCIL` según selección explícita.

### E. Verify (Verificación EFE)
- **Acción**: Validar hechos del mundo contra la expectativa (`verify_efe`).
- **Fail-Closed**: Si EFE no se cumple o falta el contrato → `FAIL` o `BLOCKED`.

### F. Consolidate (Consolidación)
- **Acción**: Persistir el episodio, actualizar `MotorMemory` si hubo éxito y emitir `GAP` si hubo fallo.
- **Evidencia**: El `summary.json` y el `receipt` son los árbitros finales de la realidad.

---
*Gobernanza: AGENTS.md*
