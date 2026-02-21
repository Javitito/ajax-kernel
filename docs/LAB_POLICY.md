# LAB Policy

## HUMAN_DETECTED -> MAINTENANCE_ONLY

Cuando `human_active=true`, `LAB_ORG` opera en modo `MAINTENANCE_ONLY`.

Reglas:
- No encolar micro-challenges de exploracion.
- Solo encolar jobs allowlisted de mantenimiento (maximo 1 por tick):
  - `providers_probe`
  - `capabilities_refresh`
  - `doctor_*`
  - `health_*`
- Si no hay jobs allowlisted due: `skipped_reason=maintenance_only_no_due`.

Receipts esperados (`ajax.lab_org.receipt.v1`):
- `human_active: true`
- `mode: "MAINTENANCE_ONLY"`
- `allowlist_used: [...]`
- `skipped_reason` explicito cuando no se encola.
