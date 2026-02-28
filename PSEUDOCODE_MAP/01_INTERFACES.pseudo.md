# 01 - Interfaces and Proof Contracts

## Purpose
Contratos minimos para TaskStep transaccional y output con evidencia.

## Pseudocode

```text
type EvidenceRef:
  kind: str
  path: str

type Claim:
  type: "fixed" | "root_cause" | "available_green" | "diagnosed" | "verified"
  statement: str
  evidence_refs: EvidenceRef[]    # obligatorio para claims
  efe_ref?: str

type HypothesisOutput:
  hypothesis: str
  verification_commands: str[]    # 1..3 comandos reproducibles

type TransactionalStep:
  prepare:
    snapshot: bool
    undo_script_gen: bool
  apply: TaskStep
  verify:
    doctor_check: bool
    efe_check: bool
  undo:
    rollback_best_effort: bool

task_execution(step):
  if step mutates state:
    require TransactionalStep
    require prepare.snapshot and prepare.undo_script_gen
    require verify.doctor_check
    if expected_state exists:
      require verify.efe_check
    require undo.rollback_best_effort
```

## Invariants
- Claim sin `evidence_refs` no es hecho: degradar a hipotesis + comandos.
- Paso mutante sin wrapper transaccional viola integridad constitucional.
- `verify.doctor_check` debe estar presente en mutaciones T2+.
