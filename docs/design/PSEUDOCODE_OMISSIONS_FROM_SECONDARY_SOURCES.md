# PSEUDOCODE Omissions From Secondary Sources

This file is not canonical pseudocode.
It records possible gaps noticed while contrast-checking `ROOT/PSEUDOCODE_MAP`, `MICROFILM.md`, and `AGENTS.md` against the first code-first regeneration pass.

## Entry 1

- source: `ROOT/PSEUDOCODE_MAP/01_INTERFACES.pseudo.md`
- suspected_gap: The recipe lifecycle (`crystallize`, `validate`, `promote`) may deserve an explicit live block in the exportable map.
- why_it_might_matter: The current code-first pass focuses on mission execution, diagnostics, LAB control, and state. The codebase also exposes post-mission knowledge flow that may be part of the export layer.
- evidence_needed: Inspect `ajaxctl crystallize --help`, `ajaxctl validate --help`, `ajaxctl promote --help`, `agency/crystallization.py`, and current tests.

## Entry 2

- source: `ROOT/PSEUDOCODE_MAP/00_SYSTEM_FLOW.pseudo.md`
- suspected_gap: Provider benchmark and canary tools (`cloud-canary`, `cloud-bench`, `lmstudio-bench`, `lmstudio-test`) may need a dedicated interface note.
- why_it_might_matter: Current maps document provider snapshot and ping, but not whether the benchmark/canary family is considered part of the canonical exported operator surface.
- evidence_needed: Inspect CLI help and runtime modules for those commands, then decide if they are core exportable diagnostics or auxiliary tooling.

## Entry 3

- source: `MICROFILM.md`
- suspected_gap: Trust-promotion rules may need a slightly richer cross-reference between verification tiers, receipts, and claim validation.
- why_it_might_matter: The current code-first pass captures `verification_mode`, `driver_simulated`, and `anti_optimism_guard`, but a future refinement may want a tighter state-to-claim contract block.
- evidence_needed: Inspect `agency/types/output_bundle.py`, `agency/anti_optimism_guard.py`, and related tests for a stable exportable contract.

## Entry 4

- source: `AGENTS.md`
- suspected_gap: Research and capability-gap triage workflow may deserve a compact artifact/state note if it is part of the live kernel surface.
- why_it_might_matter: The runtime clearly writes durable capability gaps, and there are dedicated review utilities in code, but the first pass keeps focus on mission-time state rather than research-time backlog handling.
- evidence_needed: Inspect `agency/review_capability_gaps.py`, `ajaxctl gaps --help`, and any active tests or runbooks for stable operator contracts.
