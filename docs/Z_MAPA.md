# Z_MAPA_AJAX_LATEST — Hilo 13 (FULL_CANON)

## 1. Idea central
AJAX = cuerpo local de Adam con Method > Model: THE AJAX PROTOCOL (A–F), P0 epistémico y EFE/Delta como árbitro. Admin-as-a-Service reduce fricción operativa sin abrir agujeros. LAB/PROD es normativo: lo experimental va a LAB; PROD solo stable o break-glass con evidencia.

---

## 2. Capas
- **Infra (Windows + WSL)**: bridge 5011; Scheduled Task \AJAX\AJAX_AdminRunner; cola admin en C:\Users\Public\ajax_admin_queue.
- **LAB zone (display split)**: LAB vive en Display 2 (dummy) con layout extendido; separacion por display + gate human_active/lease (no por usuario Windows).
- **Núcleo**: ajaxctl + broker/plan_runner; gates de madurez; bypass para maintenance.
- **Método científico (A–F)**: ScientificMethodRunner con artefactos persistidos (p0, efe, gate, trace, delta, feedback) y gate LAB/PROD.
- **Evolución (I+D)**: capability_gaps → decisions → backlog; Research Triage prioriza gaps (LAB-only).
- **Memoria**: LEANN + P0 (history/pepitas con budgets y source_resolution).
- **Admin Ops**: policy allowlisted + runner + evidence; break-glass TTL con post-mortem.
- **OfficeBot**: control de escritorio con verificación; promoción a PROD solo tras estabilidad.
- **Repo/Higiene**: zonas CANON/DATA/ATTIC + export público allowlisted.

---

## 3. Hitos/Decisiones
1) Admin-as-a-Service implementado (policy + runner + helpers + docs + examples).
2) THE AJAX PROTOCOL operativo (A–F, PASS/FAIL por EFE, SPL meta-signal, gate LAB/PROD).
3) P0 LEANN integrado (p0_context.json con refs/gaps y budgets).
4) Research Triage (gaps triage) en LAB, determinista.
5) Repo Hygiene PR-A1 + export público allowlisted con skipped_manifest.

---

## 4. Preguntas abiertas
- Registrar task admin y pasar smoke pack (7 EFEs).
- Ruta canónica de Pepitas_Index para P0 (LEANN_ROOT/MIO_ROOT).
- Override de root para ajaxctl experiment run.
- Unificar/documentar break-glass admin vs protocolo científico.
- PR-A2 (migración quirúrgica a attic/data externa).

---

## 5. Próximo paso
- Ejecutar smoke pack admin y archivar evidencias.
- Ejecutar triage real y convertir probes en experimentos baseline.
- Definir ruta pepitas canónica y re-ejecutar P0.

## Actualización 2025-12-25

### Admin-as-a-Service
- Cola allowlisted (C:\Users\Public\ajax_admin_queue) con TTL, evidence y break-glass.
- Helpers para enqueue/break-glass/scheduled task y smoke pack.

### THE AJAX PROTOCOL
- Máquina de estados A–F con contratos JSON; PASS/FAIL por verify_efe().
- Gate LAB/PROD con break-glass; SPL solo meta.

### P0 y Research Triage
- P0 siempre escribe p0_context.json con refs/gaps y source_resolution.
- gaps triage prioriza gaps y sugiere probes (LAB-only).

### Repo Hygiene / Public Export
- Zonas CANON/DATA/ATTIC + repo_health.ps1.
- export_public_repo.ps1 allowlisted con skipped_manifest.
