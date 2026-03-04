# Boot And Watchdog (PR29)

## Garantía que ofrece
- `\AJAX\AJAX_Start_AJAX`: asegura bootstrap de rail `prod` (puerto `5010`) al logon de `Javi`.
- `\AJAX\AJAX_Lab_Bootstrap`: asegura bootstrap de rail `lab` (`5012` + LAB worker) al logon de `Javi`.
- `\AJAX\AJAX_DriverWatchdog`: watchdog continuo en sesión `Javi` para reponer caídas de `5012` en modo fail-closed.
- Todas las tareas se gestionan desde `config/expected_tasks.json` con `ops tasks ensure` (idempotente).

## Comandos operativos
- Auditoría de tareas y drift:
  - `python bin/ajaxctl ops tasks audit`
- Simulación de reparación:
  - `python bin/ajaxctl ops tasks ensure --dry-run`
- Aplicar reparación:
  - `python bin/ajaxctl ops tasks ensure --apply`
- Diagnóstico consolidado de boot/watchdog:
  - `python bin/ajaxctl doctor boot --rail lab`
- Ensure idempotente por rail:
  - `python bin/ajaxctl lab ensure --rail lab`
  - `python bin/ajaxctl lab ensure --rail prod`

## Evidencia y logs
- `doctor boot` escribe:
  - `artifacts/doctor/<ts>/boot_explain.json`
  - `artifacts/receipts/doctor_boot_<ts>.json`
- `watchdog` escribe:
  - `artifacts/ops/<ts>/watchdog_tick.json`
  - `C:\Users\Public\ajax\watchdog_5012_task.log` (rotación por timestamp al superar tamaño)
- `lab bootstrap` escribe:
  - `artifacts/boot/lab_boot_<ts>.json`
  - `C:\Users\Public\ajax\lab_bootstrap_task.log`

## Break Glass (desactivar watchdog sin borrar tareas)
- Crear archivo stopfile:
  - `C:\Users\Public\ajax\watchdog.stop`
- El watchdog detecta el stopfile, deja evidencia y sale sin ejecutar acciones.
- Para reactivar: borrar `C:\Users\Public\ajax\watchdog.stop`.
