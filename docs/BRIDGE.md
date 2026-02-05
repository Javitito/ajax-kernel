# Bridge: ajax-legacy -> ajax-kernel

Decision: **A (copia selectiva)**.

Rationale:
- `ajax-legacy` contiene historico, experimentos y ruido (no determinista para subtree).
- El objetivo de `ajax-kernel` es una base estable y legible; preferimos lista explicita de rutas.

## Source
- Legacy: `Javitito/ajax-legacy` (archived, read-only).
- Kernel: `Javitito/ajax-kernel`.

## Included (kernel routes)
- `agency/` (core agent runtime)
- `ajax/` (facade/exports)
- `bin/ajaxctl` (CLI)
- `config/*.yaml`, `config/*.yml`, `config/*.json` (policy declarativa)
- `schemas/` (contratos JSON)
- `ui/concierge.py`, `ui/__init__.py` (minimo para tests/CLI)
- `AGENTS.md`, `MICROFILM.md` (docs canon accesibles desde root)
- `docs/` (INDEX + kernel method docs)
- `tests/` (suite minima `test_kernel_*`)

## Excluded (anti-basura)
- Evidencia de ejecucion: `artifacts/`, `runs/`, `logs/`, caches, dumps.
- Zonas legacy/attic/benchmarks y cualquier output generado.
- Backups `*.bak*`, pointers `*.pid`.

## Update policy
Cualquier migracion adicional debe:
1) Añadir ruta a esta lista.
2) Añadir/ajustar tests en `tests/`.
3) Mantener `.gitignore` anti-basura.
