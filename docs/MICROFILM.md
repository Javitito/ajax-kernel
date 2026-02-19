# MICROFILM.md — AJAX v5.1 (H21 + Constitución mínima)

## Resumen human friendly
AJAX se entiende con 4 motores: **(1) dual‑rail en metal**, **(2) método A–F**, **(3) ejecución transaccional**, **(4) gobernanza adaptativa FAST/SAFE/COUNCIL**. En H21 cambiamos la física: **LAB ya no vive en un usuario SO separado**; vive en tu sesión y actúa en **pantalla dummy**.

---

## §0 — Tesis constitucional (por qué AJAX existe)
**Pregunta central:** ¿cómo delegar operación de un PC personal a agentes LLM sin perder soberanía?

**Principio fundacional:**
MODELO propone · MÉTODO gobierna · EVIDENCIA decide.

**Invariantes:**
1) No se acepta “éxito” sin VERIFY contra un EFE observable (delta=0).
2) No se muta sin camino de UNDO (Prepare→Verify→Auto‑Undo si falla).
3) LAB explora sin contaminar PROD (rail + `display_target=dummy`).
4) Toda decisión deja RECEIPT durable (auditabilidad / aprendizaje en LEANN).
5) Evidencia válida = observable + independiente + no trivial de falsificar + receipt.

**Selector de rigor (proporcional al riesgo):**
- **FAST**: cristalizado (receta/hábito). Sin LLM o con intervención mínima. Receipt siempre.
- **SAFE**: LLM + gates + transacción + verify + auto‑undo si falla.
- **COUNCIL**: alto riesgo/incertidumbre → quorum multi‑modelo + veto vinculante antes de mutar.

**Supuestos explícitos (límite del sistema):**
AJAX protege frente a fallos/alucinación/optimismo del modelo y errores operativos; **no** asume adversario con control kernel/driver del host ni cadena de suministro comprometida.

---

## 1) Arquitectura Física (Dual‑Rail)
AJAX vive en el metal. Aislamiento por rail lógico y target de pantalla.

```text
SESIÓN ÚNICA (usuario real)
[ rail=prod | puerto 5010 ]         [ rail=lab | puerto 5012 ]
        │                                    │
        │                                    ├──► display_target = dummy
        │                                    │     (HDMI dummy plug / GPU activa)
        ▼                                    │
 [ MODO DEFERENCIA ] <─── [ BROKER ] ─────────┘
 (se para si humano activo)
        │
        └───────────────────[ LEANN ]
                (memoria epistémica + cristalización)
```

**Invariante:** `rail=lab` nunca toca la pantalla primaria.

---

## 2) Protocolo Científico (Ciclo A–F)
```text
[A] P0/EPISTEMOLOGÍA ──► ¿Ya lo hicimos? Consulta LEANN. No inventar.
[B] EFE/HIPÓTESIS     ──► Define éxito físico: ¿qué observable dirá OK?
[C] GATE/GOBERNANZA   ──► seguridad, rail, permisos, proveedores, irreversibles.
[D] TOTE/EJECUCIÓN    ──► actúa (Prepare→Undo si aplica).
[E] DELTA/OBSERVACIÓN ──► observable vs EFE. Evidencia válida + delta=0.
[F] PROMUEVE/EVOLUCIÓN──► éxito: pepita/receta/hábito; fallo: gap.
```

---

## 3) Ejecución Transaccional (Prepare‑Undo‑Verify)
**Mutación** = cualquier cambio persistente (config, permisos, red, servicios, credenciales, scheduler, repos, archivos con valor).

```text
[PREPARE] snapshot + undo
[APPLY]   mutación
[VERIFY]  doctor + invariantes + EFE + observables independientes
  └─ NO → AUTO‑UNDO (y receipt del fallo)
[RECEIPT] evidencia durable → LEANN
```

### Preflight estricto (nuevo wiring demo-ready)
Antes de `Starting XI` el runtime aplica dos gates fail-closed:
1) `PolicyContractGate`: exige `config/provider_policy.yaml` y `config/provider_failure_policy.yaml`; si faltan o son inválidos => `BLOCKED` con receipt.
2) `PhysicalAnchorGate`: valida `rail + session + port + display`; mismatch => `BLOCKED` con receipt.

**Irreversibles:** dinero, borrados destructivos, envíos, cambios de credenciales/seguridad → **WAITING_FOR_USER** o **COUNCIL** por defecto.

---

## 4) Gobernanza Adaptativa (FAST / SAFE / COUNCIL)
- **FAST**: receta/hábito (lo ya demostrado). Lo rápido también deja rastro.
- **SAFE**: ejecución normal con transacción + verify.
- **COUNCIL**: decisiones delicadas con veto vinculante.

---

## 5) Cristalización (escalera)
TRACES → EPISODIOS → RECETAS → VALIDACIÓN(3/3) → HÁBITO
