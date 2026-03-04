# 01 - Interfaces (CLI Surface, As Implemented)

## Scope
Superficie CLI relevante para loop EFE, metabolismo, friccion y session anchor LAB.

## As Implemented (main)

### `doctor metabolism`

```text
python bin/ajaxctl doctor metabolism --since-min <minutes>
```

- Handler: `cmd_doctor_metabolism`.
- Fuente: `agency.metabolism_doctor.run_doctor_metabolism`.
- Output: JSON schema `ajax.doctor.metabolism.v0`.
- Entradas agregadas (read-only): capability gaps, `artifacts/efe_candidates`, provider ledger + `router_ladder_decision_*.json`, waiting backlog.
- Salidas clave: `gaps`, `efe_candidates`, `provider`, `waiting_backlog`, `next_hint`, `critical`, `exit_code`.

### `ops friction gc`

```text
python bin/ajaxctl ops friction gc --dry-run [--older-than-hours 24]
python bin/ajaxctl ops friction gc --apply   [--older-than-hours 24]
```

- Handler: `cmd_ops_friction_gc`.
- Fuente: `agency.friction.run_friction_gc`.
- `--dry-run`: reporta candidatos sin mover archivos.
- `--apply`: archiva waiting antiguos y aplica policy minima al provider ledger (con snapshot).
- Output: JSON schema `ajax.ops.friction_gc.v1`.

### `verify efe apply-candidate`

```text
python bin/ajaxctl verify efe apply-candidate --gap <gap.json> --out <efe_final.json>
```

- Handler: `cmd_verify_efe_apply_candidate`.
- Fuente: `agency.verify.efe_apply_candidate.apply_efe_candidate_from_gap`.
- Comportamiento: lee `efe_candidate_path` del gap, extrae `expected_state`, escribe payload final editable.
- Importante: no ejecuta acciones; solo materializa evidencia/contrato.

### `lab session` (expected session anchor)

```text
python bin/ajaxctl lab session init   --ttl-min 120 --display dummy --rail lab
python bin/ajaxctl lab session status [--write-receipt]
python bin/ajaxctl lab session revoke
```

- Fuente: `agency.lab_session_anchor`.
- Archivo canonico: `artifacts/lab/session/expected_session.json`.
- Receipts: `lab_session_init_*`, `lab_session_status_*`, `lab_session_revoke_*`.

### `doctor receipts` (estado actual)

```text
python bin/ajaxctl doctor receipts --since-min <minutes> [--json]
```

- Fuente: `agency.receipt_validator`.
- En `main`: clasificacion binaria `PASS/FAIL` (`status`), sin canal `WARN`, sin `--strict`, sin `--summary-only`, sin `--top-k`.

## Notes / Planned
- Politica `PASS/WARN/FAIL` con `--strict` esta marcada para siguiente PR (no presente en `main` actual).

## Implemented vs Planned

| Item | Status | Evidence pointers |
|---|---|---|
| `doctor metabolism` registrado y operativo | Implemented | `bin/ajaxctl` (`cmd_doctor_metabolism`, parser `doctor metabolism`), `agency/metabolism_doctor.py` |
| `ops friction gc` con `--dry-run`/`--apply` | Implemented | `bin/ajaxctl` (`cmd_ops_friction_gc`, parser `ops friction gc`), `agency/friction.py` |
| `verify efe apply-candidate` evidence-only | Implemented | `bin/ajaxctl` + `agency/verify/efe_apply_candidate.py` |
| `lab session init/status/revoke` | Implemented | `bin/ajaxctl` + `agency/lab_session_anchor.py` |
| `doctor receipts` con severidad WARN + `--strict` | Planned next PR | `agency/receipt_validator.py` y parser actual sin esos flags |
