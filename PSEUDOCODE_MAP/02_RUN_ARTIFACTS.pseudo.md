# 02 - Run Artifacts Layout

## Purpose
Definir persistencia minima por corrida para auditoria reproducible.

## Pseudocode

```text
run_broker(mission):
  run_id = create_run_id()
  run_dir = runs/<run_id>/
  mkdir(run_dir)

  write run_dir/plan.json
  execute plan -> write run_dir/result.json
  verify efe/doctors -> write run_dir/verification.json
  append audit events -> write run_dir/audit_log.json

  return run_id
```

## Canonical files under `runs/<run_id>/`
- `plan.json`
- `result.json`
- `verification.json`
- `audit_log.json`

## Invariants
- Si faltan los cuatro artefactos primarios, la corrida es incompleta para auditoria.
- Lectores/audits deben tratar ausencia como finding (fail-closed).
