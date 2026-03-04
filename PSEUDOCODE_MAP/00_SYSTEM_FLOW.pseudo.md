# 00 - System Flow (As Implemented)

## Scope
Describe el flujo real del kernel para fail-closed por `missing_efe_final` y el loop de recuperacion con EFE candidate.

## As Implemented (main)

```text
run_mission(mission):
  plan = plan_with_router(...)

  if plan.metadata.planning_error == "missing_efe_final":
    gap_path = _emit_missing_efe_gap(
      mission,
      receipt_path=plan.metadata.efe_repair_receipt,
      efe_candidate_path=plan.metadata.efe_candidate_path,
      efe_candidate_status=plan.metadata.efe_candidate_status,
      efe_candidate_reason=plan.metadata.efe_candidate_reason,
      efe_candidate_source_doc=plan.metadata.efe_candidate_source_doc
    )
    # Si no viene candidate path, se intenta autogen:
    # artifacts/efe_candidates/<safe_gap_id>_<UTC>.json
    mission.status = "BLOCKED"
    mission.last_result.error = "missing_efe_final"
    return

  execute_plan_steps(...)
  verify(...)
```

```text
recovery_loop_for_missing_efe(gap_json):
  # Paso de materializacion (sin ejecutar acciones)
  ajaxctl verify efe apply-candidate --gap <gap_json> --out <efe_final_json>
    -> copia expected_state desde efe_candidate_path
    -> escribe payload editable para humano

  human_edits(<efe_final_json>.expected_state)
  next_execution_uses_expected_state_for_verify()
```

## Notes / Planned
- No existe auto-apply del candidate al plan de ejecucion; el flujo actual exige paso humano explicito con `verify efe apply-candidate`.

## Implemented vs Planned

| Item | Status | Evidence pointers |
|---|---|---|
| `missing_efe_final` bloquea la mision (fail-closed) | Implemented | `agency/ajax_core.py` (`_emit_missing_efe_gap`, rama `planning_error == missing_efe_final`) |
| GAP incluye `efe_candidate_path/status/reason` cuando existe | Implemented | `agency/ajax_core.py` (`_emit_missing_efe_gap`) |
| Candidate autogen escribe en `artifacts/efe_candidates/` | Implemented | `agency/ajax_core.py` (`_autogen_missing_efe_candidate`) |
| `verify efe apply-candidate` materializa EFE final sin ejecutar | Implemented | `bin/ajaxctl` (`cmd_verify_efe_apply_candidate`), `agency/verify/efe_apply_candidate.py` |
| Aplicacion automatica del candidate al runtime | Planned | No hook directo en `agency/ajax_core.py` para consumir `efe_apply_candidate.v0` |
