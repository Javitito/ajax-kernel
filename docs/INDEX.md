# AJAX Kernel

Objetivo: repo minimo, legible y estable donde el chasis de AJAX se entiende en 60 segundos.

## Lectura (orden)
- `MICROFILM.md` (o `docs/MICROFILM.md`): mapa mental + dual-rail + transaccion.
- `AGENTS.md` (o `docs/AGENTS.md`): constitucion operativa (EFE fail-closed, PREPARE/APPLY/VERIFY/UNDO).
- `docs/AJAX_SCI_KERNEL.md`: ciclo A-F.
- `docs/AJAX_POLICY_CHALLENGE_LOOP.md`: solve -> LAB -> GAP.

## Directorios
- `agency/`: nucleo (AjaxCore, verify/efe, gobernanza, provider policy).
- `ajax/`: fachada (exports de `AjaxCore`).
- `bin/ajaxctl`: CLI principal.
- `config/`: policy declarativa.
- `schemas/`: contratos JSON.
- `tests/`: tests de kernel (curados).

## Anti-basura (constitucional)
Este repo solo acepta: codigo + contratos + tests + docs.
Evidencia de ejecucion va fuera del repo: `artifacts/`, `runs/`, caches, logs, dumps.
