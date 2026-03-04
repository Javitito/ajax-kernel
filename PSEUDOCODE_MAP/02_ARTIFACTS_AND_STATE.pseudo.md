# 02 - Artifacts and State (As Implemented)

## Scope
Mapa descriptivo de paths y contratos observables usados por el loop operativo actual.

## As Implemented (main)

### Core runtime artifacts

```text
artifacts/
  capability_gaps/
    *.json                      # incluye missing_efe_final + candidate metadata cuando aplica
  efe_candidates/
    *.json                      # candidates autogen (schema ajax.verify.efe_candidate.v0)
  receipts/
    friction_gc_v1_<UTC>.json
    efe_autogen_<UTC>.json
    router_ladder_decision_<UTC>.json
    lab_session_init_<UTC>.json
    lab_session_status_<UTC>.json
    lab_session_revoke_<UTC>.json
    lab_autopilot_tick_<UTC>.json
    lab_autopilot_daemon_<UTC>.json
  waiting_for_user/
    *.json
    _archived/<YYYY-MM-DD>/*.json  # creado por friction gc --apply
  provider_ledger/
    latest.json
    _snapshots/latest_<UTC>.json   # creado por friction gc --apply
  lab/
    session/expected_session.json
    STOP_AUTOPILOT
  health/
    autopilot_heartbeat.json
  state/
    fallback_local_model.json      # opcional; existe solo tras lmstudio-bench --select-best
```

### EFE candidate and apply-candidate contracts

- Candidate schema: `ajax.verify.efe_candidate.v0` (`agency/verify/efe_autogen.py`).
- Apply output schema: `ajax.verify.efe_apply_candidate.v0` (`agency/verify/efe_apply_candidate.py`).
- GAP `missing_efe_final` puede incluir:
  - `efe_candidate_path`
  - `efe_candidate_status` (`generated|unsupported|error`)
  - `efe_candidate_reason`

### Receipts validation scope

- Schemas disponibles en `schemas/receipts/`:
  - `ajax.lab.session.v0.schema.json`
  - `ajax.lab.autopilot_tick.v1.schema.json`
  - `ajax.topology_doctor.v0.schema.json`
- Mapeo de alias soportados en `agency/receipt_validator.py` (por ejemplo `ajax.topology_doctor.v1` -> schema v0).
- Si el schema no esta soportado, `doctor receipts` lo marca como fallo en `main` actual.

### Provider ladder and local fallback state

- Ladder runtime:
  - cloud-first
  - si error transitorio/quota/timeout/provider_lock => puede pedir local fallback
  - si no hay local provider: `reason=local_fallback_unavailable`
  - si local tambien falla: `reason=local_fallback_failed`
- Archivo opcional de modelo local preferido:
  - write: `lmstudio-bench --select-best`
  - read: `lmstudio-test`
  - path: `artifacts/state/fallback_local_model.json`

### DSE note: subprocess encapsulation

- `agency/system_executor.py` centraliza `run/popen/check_output`.
- `agency/ajax_core.py` usa `SystemExecutor` (sin `subprocess.run`/`subprocess.Popen` directos).
- Guard test: `tests/test_kernel_system_executor.py::test_ajax_core_does_not_use_subprocess_directly`.

## Notes / Planned
- Politica de severidades `PASS/WARN/FAIL` en `doctor receipts` (legacy como WARN por defecto) no esta en `main` actual; queda como planned.

## Implemented vs Planned

| Item | Status | Evidence pointers |
|---|---|---|
| `artifacts/efe_candidates/` y metadata en GAP | Implemented | `agency/ajax_core.py`, `agency/verify/efe_autogen.py` |
| `verify efe apply-candidate` genera payload final editable | Implemented | `agency/verify/efe_apply_candidate.py` |
| `waiting_for_user/_archived/<date>/` por friction gc apply | Implemented | `agency/friction.py`, `tests/test_kernel_friction_gc_v1.py` |
| `fallback_local_model.json` como estado opcional local-first | Implemented (optional artifact) | `agency/lmstudio_bench.py` |
| `SystemExecutor` y prohibicion de subprocess directo en `ajax_core` | Implemented | `agency/system_executor.py`, `tests/test_kernel_system_executor.py` |
| `doctor receipts` con WARN/strict/top-k/summary-only | Planned next PR | `agency/receipt_validator.py`, parser actual en `bin/ajaxctl` |
