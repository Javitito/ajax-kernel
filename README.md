# ajax-kernel

Kernel/chasis de AJAX: ejecucion de misiones con EFE (fail-closed) y gobernanza proporcional al riesgo.

## Runtime canonico
- Este repo (`ajax-kernel`) es el unico root de runtime para demo/LAB/PROD.
- Ejecuta siempre desde aqui: `cd /mnt/c/Users/Javi/AJAX_HOME/ajax-kernel`.
- Gate demo local: `bash scripts/ops/demo_gate.sh`.

## 60s
- Lee `MICROFILM.md`.
- Revisa `AGENTS.md`.
- Ejecuta `python bin/ajaxctl health --json`.

## Dev
- `python bin/ajaxctl --help`
- `python -m compileall -q agency ajax`
- `pytest -q`

## Repo hygiene
- Anti-basura fuerte via `.gitignore`.
- No se versiona evidencia de ejecucion.
