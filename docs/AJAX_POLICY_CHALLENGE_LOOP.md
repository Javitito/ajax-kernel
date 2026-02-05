# AJAX Policy Challenge Loop

Este documento formaliza cómo AJAX evoluciona ante lo desconocido y gestiona los fallos de infraestructura.

## Ciclo de Vida del Challenge: solve → LAB → GAP

### 1. El GAP (Nunca vacío)
- Ante cualquier impedimento técnico (429, timeout, missing capability, missing EFE), AJAX **MUST** emitir un `GAP`.
- El `GAP` es la evidencia durable de una necesidad no satisfecha.

### 2. Derivación a LAB
- Tareas con incertidumbre alta o sin receta previa se ejecutan en entorno **LAB** (`rail=lab`).
- LAB permite descubrimiento y fallo seguro sin contaminar el núcleo estable (CANON).

### 3. Decisiones de Gobernanza
- **COUNCIL**: Obligatorio para acciones destructivas o cambios en el CANON.
- **WAITING_FOR_USER**: Solo cuando la ambigüedad es humana o el presupuesto se ha agotado.
- **GAP**: Obligatorio para fallos técnicos de infraestructura.

### 4. Promoción LAB → CANON
- Una solución descubierta en LAB se "promueve" al CANON mediante:
  1. **Evidencia**: 3 ejecuciones exitosas (REPEAT).
  2. **Refactor**: Generalización del script/herramienta (eliminar ad-hoc).
  3. **Docs**: Actualización de `PSEUDOCODE_MAP` o `ActionCatalog`.

---
*Evidencia como Árbitro: AJAX_SCI_KERNEL.md*
