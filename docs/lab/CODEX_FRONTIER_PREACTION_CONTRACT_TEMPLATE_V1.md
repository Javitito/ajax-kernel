# CODEX_FRONTIER_PREACTION_CONTRACT_TEMPLATE_V1

Status: LAB template / not runtime enforcement
Scope: reusable pre-action contract for Codex-frontier missions
Spec: docs/lab/CODEX_FRONTIER_BOUNDARY_SPEC_V1.md
Authority: subordinate to repo-truth, `AGENTS.md`, `MICROFILM.md`,
`00_START_HERE_AJAX.md`, current handoff, tests, receipts, and mission-specific
user instructions.

## 1. Purpose

This template helps Codex act as an AJAX harness-frontier without becoming
sovereign.

It forces the worker to declare the task contract, Ladder entry, mutation
permission, forbidden surfaces, evidence expectations, and stop/escalation
conditions before material action.

This is LAB guidance only. It does not enforce runtime behavior, promote Codex
to canon, alter Provider Ladder, or change routing.

## 2. When To Use This Template

Use this template before any Codex-frontier mission that involves:

- synthesis, judgment, design, audit, classification, or non-trivial writing;
- repo mutation, doc mutation, generated artifacts, or receipts;
- worker delegation, subcall, premium/high-capability reasoning, or tool use;
- cost/risk decisions;
- interpretation of repo-truth, receipts, tests, handoff, or Drive Share context.

For purely mechanical read-only commands, a shortened block may be enough, but
the worker must still declare why intelligence is not involved.

## 3. Required Pre-Action Block

Copy this block before acting:

```text
INTELLIGENCE_INVOLVED:
ORIGINAL_TASK:
EXPECTED_STATE:
OBSERVED_STATE:
HYPOTHESES:
L0_EVIDENCE_TO_CHECK:
FISH_EXPECTED:
LADDER_ENTRY:
  starting_rung: L0_ALWAYS
  original_task_preserved:
  l0_probe:
  reason_to_escalate_or_not:
SUBCALL_DECISION:
PREMIUM_TRACE:
MUTATION_DECISION:
FORBIDDEN_SURFACES:
EVIDENCE_CONTRACT:
STOP_CONDITION:
ESCALATION_CONDITION:
```

Rules:

- `INTELLIGENCE_INVOLVED` must be `yes` or `no` with reason.
- `ORIGINAL_TASK` must preserve the user's request, not a softened substitute.
- `EXPECTED_STATE` defines the close condition.
- `OBSERVED_STATE` must come from local evidence, repo-truth, or explicit user
  input.
- `L0_EVIDENCE_TO_CHECK` should list repo files, tests, receipts, artifacts, or
  commands to inspect first.
- `FISH_EXPECTED` names the lower-rung evidence or candidate expected before
  high-rung writing.
- `SUBCALL_DECISION` must be declared before any subcall.
- `PREMIUM_TRACE` must be honest when Codex/premium materially reasons, writes,
  or weighs.
- `MUTATION_DECISION` must state read-only, doc-only, code mutation, artifact
  creation, or blocked.
- `FORBIDDEN_SURFACES` must list mission-specific hard rails.
- `EVIDENCE_CONTRACT` must match the claims expected at closeout.

## 4. Mutation Permission Rules

Default is no mutation.

Mutation is allowed only when the mission explicitly grants it and names the
scope.

Rules:

- read-only means no file writes, receipts, logs, snapshots, bundles, or generated
  reports;
- doc-only means only the named documentation path(s);
- code mutation means only scoped source/tests/config allowed by the mission;
- never touch runtime routing, Provider Ladder, provider/model config, scheduler,
  Heartbeat, Cartero, credentials, remote sync, or destructive cleanup unless
  the user explicitly scopes that surface;
- never normalize, revert, stage, or clean pre-existing dirty files unless they
  are explicitly in scope;
- capture `git status --short` before edits when mutation is allowed;
- after mutation, report final `git status --short` or scoped diff when safe.

## 5. Evidence Contract

The mission must define evidence before action.

Common evidence types:

- file paths and line refs;
- command output summaries;
- tests/checks and pass/fail result;
- git status or scoped diff;
- generated artifact path when artifact creation is allowed;
- receipt path when receipt creation is allowed;
- explicit `no runtime/config/provider/routing touched` confirmation;
- known gaps and missing evidence.

Claims without matching evidence must be labeled as hypothesis, gap, or
uncertainty.

## 6. Closure Bundle

Return at close:

- Mission;
- files touched;
- commands run;
- checks run;
- claims with EvidenceRefs;
- runtime/config/provider/routing touched: YES/NO;
- risks;
- uncertainty;
- next bite;
- verdict.

For intelligent, mutative, delegated, or premium work, also return a Ladder
Capsule:

```text
starting_rung:
closure_rung:
fish_evidence:
weigher_used:
lower_rung_reuse_percent:
premium_read_write_ratio:
tree_decision_followed:
reason_to_escalate_or_not:
premium_trace:
subcall_decision:
```

## 7. Failure / Block Conditions

Block or return GAP when:

- `ORIGINAL_TASK` is missing or altered;
- `EXPECTED_STATE` cannot be stated;
- observed state contradicts the mission contract;
- required evidence cannot be inspected;
- mutation would touch an unscoped or forbidden surface;
- a read-only mission would require writes to proceed;
- subcall/premium use is requested without ROI/evidence reason;
- lower-rung evidence is sufficient but a premium write is still proposed;
- Drive Share copy conflicts with repo-truth and canonical source cannot be
  checked;
- a claim cannot be supported by evidence.

Never close success by narrative when the evidence contract is unmet.

## 8. Minimal Example

```text
INTELLIGENCE_INVOLVED: yes; doc synthesis and boundary judgment required.
ORIGINAL_TASK: Create a LAB pre-action template for Codex-frontier missions.
EXPECTED_STATE: One markdown template exists at the scoped path with required
fields and no runtime changes.
OBSERVED_STATE: Boundary spec exists; worktree has unrelated pre-existing dirty
files.
HYPOTHESES:
- A template can reduce sovereignty drift.
- Runtime enforcement is not needed for this doc-only mission.
L0_EVIDENCE_TO_CHECK:
- docs/lab/CODEX_FRONTIER_BOUNDARY_SPEC_V1.md
- 00_START_HERE_AJAX.md
- AGENTS.md
- docs/prompts/AJAX_DT_CONTEXT_MINIMUM_2026-06.md
FISH_EXPECTED: A scoped markdown template draft grounded in local docs.
LADDER_ENTRY:
  starting_rung: L0_ALWAYS
  original_task_preserved: yes
  l0_probe: local docs and git status
  reason_to_escalate_or_not: no subcall; local evidence sufficient
SUBCALL_DECISION: NO
PREMIUM_TRACE: used as primary Codex brain; no external model call
MUTATION_DECISION: DOC-ONLY; create exactly one named markdown file
FORBIDDEN_SURFACES:
- runtime routing
- Provider Ladder
- provider/model config
- scheduler / Heartbeat / Cartero
- credentials / remote sync / destructive cleanup
EVIDENCE_CONTRACT:
- git status before and after
- required fields found by rg
- scoped diff
STOP_CONDITION: any need to edit runtime, tests, config, or existing dirty files
ESCALATION_CONDITION: missing boundary spec or contradiction in repo-truth
```
