# RUNBOOK LAB AUTOPILOT v0

## Estado implementado (v0 real)

- Comando disponible: `python bin/ajaxctl lab autopilot`
- Modos:
  - `--dry-run`: evalúa gates y selecciona plan sin actuar.
  - `--once`: ejecuta 1 tick (SAFE work-item o NOOP).
  - `--daemon`: loop periódico con `worker.pid` + `heartbeat.json`.
- Gates fail-closed:
  - ausencia humana (`human_absent`)
  - entorno seguro (`env_safe`, rail=lab + dummy display)
  - presupuesto (`budget_ok`, sin premium por defecto)
- Receipts siempre:
  - `artifacts/receipts/lab_autopilot_tick_<ts>.json`
  - `artifacts/lab_org/<ts>/receipt.json`
- Resultado de ejecución cuando aplica:
  - `artifacts/lab/results/result_<ts>_<job_id>.json`

## Comandos exactos

```bash
python bin/ajaxctl lab autopilot --dry-run
python bin/ajaxctl lab autopilot --once
python bin/ajaxctl lab autopilot --daemon --interval-s 1800
```

Opciones útiles:

```bash
python bin/ajaxctl lab autopilot --once --absence-min 10 --providers-stale-min 60
python bin/ajaxctl lab autopilot --once --no-filesystem-basic
python bin/ajaxctl lab autopilot --daemon --interval-s 900 --max-ticks 3
```

## Qué NO hace todavía (v0)

- No hace browser real, login, ni acciones irreversibles.
- No consume premium providers por defecto.
- No ejecuta mutaciones fuera del allowlist SAFE v0.

## Troubleshooting

### BLOCKED por `human_active`

- Revisa señal humana:
  - `artifacts/health/human_signal.json`
  - `state/human_active.flag` (o mirrors en `artifacts/state` / `artifacts/policy`)
- Reintenta en modo ausencia real:
  - `python bin/ajaxctl lab autopilot --once`

### BLOCKED por `rail_not_lab` o mismatch de ancla

- Verifica topología y ancla:
  - `python bin/ajaxctl doctor topology`
  - `python bin/ajaxctl doctor anchor --rail lab`
- Si `display_target_is_dummy` no está garantizado, no se ejecutará el tick.

### providers stale

- Fuerza un tick único para refresh SAFE:
  - `python bin/ajaxctl lab autopilot --once`
- Evidencia esperada:
  - `artifacts/health/providers_status.json` actualizado
  - `artifacts/lab/results/result_<ts>_<job_id>.json`
