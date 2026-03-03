# MICROFILM.md — AJAX v5.4 (H25) — ADN mínimo para reconstrucción

## Resumen human friendly
AJAX es un **chasis** para delegar tareas en un PC a agentes LLM **sin perder soberanía**: el modelo “propone”, el método **gobierna**, y la **evidencia** decide.  
Desde H22–H23 reforzamos tres cosas: **(i) tomar nota del punto de partida (SSC) siempre**, **(ii) evidencia con “tiers” (real vs simulado) que no contamine confianza**, y **(iii) cristalización multi‑escala (una subtarea puede convertirse en macro‑habilidad).**

**Fecha de consolidación:** 2026-02-15 21:34:33 CET

---

## §0 — Tesis constitucional (por qué AJAX existe)
**Problema:** automatizar operaciones reales en un PC sin convertir el sistema en un “modelo con poderes”.  
**Solución:** separar *inteligencia* (LLM) de *autoridad* (método + gates + receipts).

**Lema:** MODELO propone · MÉTODO gobierna · EVIDENCIA decide.

**Invariantes (no negociables):**
1) **No “completado” sin VERIFY** contra un EFE observable (delta=0).
2) **No mutación sin UNDO** (Prepare → Apply → Verify → Auto‑Undo si falla) para tareas reversibles.
3) **LAB no contamina PROD** (rail + `display_target=dummy` cuando hay escritorio).
4) **Toda decisión deja RECEIPT durable** (auditabilidad + aprendizaje).
5) **Evidencia válida** = observable + independiente + difícil de falsificar + trazable (receipt).

---

## §1 — SSC (Start Snapshot Capture) — H22/H23 (nuevo pilar)
**Regla:** antes de ejecutar una misión con actuation, AJAX **debe** tomar nota del “punto de partida”.  
**Importante:** SSC es **observación**, no “personalización obligatoria”. El chasis guarda el estado; el aprendizaje decidirá después qué significa (preferencias, hábitos, riesgos, etc.).

**Mínimo SSC recomendado (por rail):**
- Ventana activa / foco, procesos relevantes, instancias de navegador abiertas.
- Estado de audio (si hay objetivo `media_playback`).
- Señales de “humano activo” (para deferencia).
- Driver health (`/health` + auth) y `verification_mode`.

**Salida:** `snapshot0` + receipt asociado.

---

## §2 — Protocolo científico A–F (la física de AJAX)
```text
[A] EPISTEMOLOGÍA  → consulta LEANN (no inventar lo ya sabido)
[B] EFE            → define éxito físico observable
[C] GATES          → rail, permisos, irreversibles, proveedores, riesgos
[D] EJECUTA         → Prepare → Apply → (Undo si aplica)
[E] VERIFY          → compara observables vs EFE (delta=0)
[F] APRENDE         → cristaliza receta / registra GAP con evidencia
```

**Fail‑closed:** si falta EFE verificable, **no se ejecuta** (o se repara EFE con “EFE‑Repair Loop”, con límites estrictos).

---

## §3 — Delegación inteligente (paper 2602.11865) integrado en AJAX
El paper “Intelligent AI Delegation” formaliza que delegar no es “tirar tareas”: es una secuencia de decisiones sobre:
- **Autoridad** (quién puede hacer qué),
- **Responsabilidad / accountability** (quién responde de los efectos),
- **Especificación clara** (roles y límites),
- **Confianza** (evidencia y reputación),
- **Adaptación a cambios y fallos**.

**Traducción a AJAX:**
- *Autoridad* → **Gates + Council + WAITING_FOR_USER** (firebreaks).
- *Accountability* → **Receipts + Mission history + Evidence tiers**.
- *Confianza* → **CompetencyManager** (solo evidencia “real” promueve confianza).
- *Adaptación* → **Provider ledger + breakers + lab discover + explore loop**.

---

## §4 — Subtarea = tarea (pero no “trivial”) — pilar operativo
Cada subtarea tiene entidad de tarea porque:
- genera **evidencia**,
- produce o revela **gaps**,
- y puede cristalizar en un bloque reutilizable.

**Consecuencia:** la “tarea mínima” es contextual (depende del SO/driver/capacidades). No lo decidimos por dogma; lo descubre el sistema.

---

## §5 — Cristalización multi‑escala (deseable)
No asumimos que cristalizar = micro‑acción. Es deseable cristalizar “bloques”:

```text
micro‑acciones → rutina → receta verificada → macro‑habilidad (“poner YouTube”)
```

**Regla:** una cristalización solo asciende cuando hay verificación repetida y estable (p.ej. 3/3 ejecuciones reales).

---

## §6 — Evidencia por tiers (real vs simulado) — H23
Para que LAB sea evolutivo sin auto‑engaño:
- `verification_mode="real"` promueve confianza.
- `verification_mode="simulated"| "manual"` NO promueve confianza (sirve para pruebas/fixtures).
- `driver_status` debe ser “online” para contar como evidencia real.

---

## §7 — LAB autónomo (cuando Javier no está)
Objetivo: “LAB currando” sin sorpresas.
- Solo con **gates de ausencia humana** + `display_target=dummy`.
- Budget & rate‑limits (proveedores) respetados.
- Todo experimento deja receipt; fallos también.

Ver runbook: `docs/RUNBOOK_LAB_AUTOPILOT.md`.

---

## §8 — Checklist de arranque (mínimo)
1) `python bin/ajaxctl doctor drivers providers bridge leann`
2) `python bin/ajaxctl lab discover --targets filesystem_basic window_focus browser_basic`
3) Ejecuta una misión simple con EFE (p.ej. `window_focus`) y verifica receipt “real”.


## §9 — Gate Closure Audit (H25)
Toda clausura de gates constitucionales debe quedar auditada en `artifacts/audits/gate_closure_*.json` con estado y evidencia de verificación.

Comando canónico de auditoría:
`python bin/ajaxctl gate-audit`

Resultado esperado:
- Produce/actualiza artefactos de clausura en `artifacts/audits/`
- Permite verificar rápidamente qué gates están `CLOSED` y con qué evidencia
