# CODEX_FRONTIER_BOUNDARY_SPEC_V1

Status: LAB boundary spec
Scope: Codex/app/thread as possible AJAX harness-frontier
Repo-truth precedence: this spec is subordinate to `AGENTS.md`, `MICROFILM.md`,
`00_START_HERE_AJAX.md`, repo code, tests, receipts, and current handoff.

## 1. AJAX Non-Delegable Authority

AJAX retains sovereignty over:

- user intent interpretation and expected_state definition;
- repo-truth precedence and conflict resolution;
- Ladder entry, rung choice, cost gate decision, and escalation permission;
- hard rails for runtime, Provider Ladder, provider/model config, scheduler,
  Heartbeat, Cartero, credentials, routing, and destructive cleanup;
- evidence standards, claim classification, and closure verdict;
- LAB-to-canon promotion decisions;
- whether a method becomes reusable habit, recipe, capability, or gap.

Codex may propose, execute scoped tool work, and weigh evidence, but Codex does
not redefine the mission, override repo-truth, promote hypotheses, or bypass AJAX
governance.

## 2. Codex Allowed Harness Capabilities

Codex may serve as harness-frontier for:

- local repo inspection and deterministic command execution;
- scoped file edits when the mission explicitly allows mutation;
- test, lint, diff, and status checks;
- synthesis over repo-truth, receipts, handoffs, and artifacts;
- high-rung weighing of lower-rung fish;
- drafting prompts, specs, checklists, and audit packets;
- read-only audits and risk classification;
- creating explicitly scoped artifacts or receipts when the mission permits.

Codex must report what it did, what it refused to expand, what evidence it used,
and what remains uncertain.

## 3. Pre-Action Requirements For Intelligent Tasks

Any task involving synthesis, judgment, design, non-trivial writing, audit,
classification, delegation, subcall, premium use, or evidence interpretation is
an intelligent task.

Before material action, the worker must declare:

- `INTELLIGENCE_INVOLVED`;
- `ORIGINAL_TASK`;
- `EXPECTED_STATE`;
- `OBSERVED_STATE`;
- `LADDER_ENTRY` with `starting_rung: L0_ALWAYS`;
- `SUBCALL_DECISION`;
- `premium_trace`;
- `MUTATION_DECISION`.

Minimum L0 evidence should be checked first: repo-truth, tests, receipts,
artifacts, handoff, current status, existing skills, and cheap/local evidence
when available. Higher rungs enter only to weigh, patch, block, or escalate
based on that evidence.

## 4. Mutation Permission Rules

Codex may mutate files only when the mission explicitly allows mutation and
names the scope.

Rules:

- read-only missions must not write files, receipts, logs, snapshots, or bundles;
- doc-only missions may edit only the named documentation path(s);
- code missions may edit only the scoped implementation and tests;
- no runtime/provider/routing/scheduler/Heartbeat/Cartero/credential surface may
  be touched without explicit mission authority;
- pre-existing dirty files must not be normalized, reverted, or staged unless
  explicitly included in scope;
- before mutation, capture relevant worktree state;
- after mutation, run the narrowest meaningful checks and report residual risk.

## 5. Required Evidence Bundle

Closeout should include:

- files touched;
- commands run;
- tests/checks run and outcomes;
- git status or scoped diff status when safe;
- claim/evidence/status table for material claims;
- explicit runtime/config/provider/routing touched: YES/NO;
- risks and uncertainty;
- next smallest bite;
- Ladder Capsule for intelligent, mutative, delegated, or premium work.

For generated Drive Share context, the evidence should include
`scripts/update_drive_share.py --check` after any publication write.

## 6. Sovereignty Drift Risks

Risks to monitor:

- Codex starts treating its synthesis as repo-truth;
- Codex bypasses `L0_ALWAYS` and writes from high-rung intuition;
- lower-rung fish is skipped because Codex is capable;
- receipts/tests become decoration rather than claim-specific evidence;
- Drive Share or handoff copies are treated as canonical;
- Codex expands scope into runtime, routing, providers, scheduler, Heartbeat, or
  Cartero without explicit authority;
- migration language hardens into canon before real task evidence supports it;
- premium/high-capability use is described as avoided when Codex actually wrote
  or materially reasoned the answer.

## 7. Promotion Criteria

The Codex harness-frontier hypothesis may move toward canon only after evidence
shows:

- repeated real tasks close with observable user-relevant outcomes;
- Ladder entry and cost gate are consistently present before intelligent action;
- Codex high-rung work reuses lower-rung fish instead of replacing it;
- forbidden surfaces remain untouched unless explicitly authorized;
- independent audits or deterministic checks catch drift;
- Drive Share publication remains synchronized but subordinate;
- migration reduces bespoke harness fragility without weakening AJAX authority.

Until then, this remains LAB guidance and transition hypothesis.

## 8. Next Integration Candidate

Next candidate:

`CODEX_FRONTIER_PREACTION_CONTRACT_TEMPLATE_V1`

Goal: create a reusable mission pre-action block/template for Codex-frontier
tasks that forces `L0_ALWAYS`, mutation decision, subcall decision, premium
trace, expected/observed state, evidence contract, and forbidden-surface
declarations before work begins.

Mode should be doc-only first. Runtime enforcement requires a separate governed
mission.
