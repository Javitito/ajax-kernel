# RUNBOOK: Loop Closure E2E (GAP -> EFE Candidate)

## 1) Hook automático de EFE candidate
Cuando una misión cae en `missing_efe_final`, AJAX queda en fail-closed (`BLOCKED`) y no ejecuta acciones.

Flujo:
1. Se emite el gap `missing_efe_final`.
2. Se intenta autogenerar un EFE candidato determinista (solo `fs|process|port`).
3. El gap guarda:
- `efe_candidate_path`
- `efe_candidate_status` (`generated|unsupported|error`)
- `efe_candidate_reason` (reason code tipado)

Ruta esperada del candidato:
- `artifacts/efe_candidates/<gap_id>_<UTC>.json`

## 2) Helper apply-candidate (sin ejecución)
Comando:

```bash
python bin/ajaxctl verify efe apply-candidate --gap <gap.json> --out <efe_final.json>
```

Qué hace:
- Lee `efe_candidate_path` desde el GAP.
- Materializa un `efe_final.json` editable por humano.
- No ejecuta acciones ni muta estado operativo más allá de escribir el archivo de salida.

## 3) Doctor metabolism
Comando:

```bash
python bin/ajaxctl doctor metabolism --since-min 180
```

Resume en JSON:
- Gaps recientes (`missing_efe_final`, `crystallize_failed`).
- EFE candidates (`generated`, `unsupported`, `error`).
- Señales de ladder (`last_429_count`, receipts `router_ladder_decision_*`).
- Backlog `waiting_for_user` (conteo y antigüedad).
- `next_hint` con 1-3 comandos concretos.

## 4) Safety boundaries
- El hook de EFE candidate es best-effort: nunca rompe el flujo principal por excepción.
- `missing_efe_final` permanece fail-closed: no hay ejecución sin contrato verificable.
- `apply-candidate` y `doctor metabolism` son read-only/verification helpers; no ejecutan acciones físicas.
- Si el descriptor es no determinista o desconocido: estado `unsupported` + hint/reason code.
