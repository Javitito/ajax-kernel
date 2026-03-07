# PSEUDOCODE Omissions From Secondary Sources

This file is not canonical pseudocode.
It records possible gaps noticed while contrast-checking `ROOT/PSEUDOCODE_MAP`, `MICROFILM.md`, and `AGENTS.md` against the code-first regeneration passes.

## Integrated In PR-49G

### recipe_lifecycle_crystallize_validate_promote

- source: `ROOT/PSEUDOCODE_MAP/01_INTERFACES.pseudo.md`
- corroborated_by: `code`, `cli`
- decision: `integrate_now`
- target_files: `PSEUDOCODE_MAP/01_INTERFACES.pseudo.md`, `PSEUDOCODE_MAP/02_ARTIFACTS_AND_STATE.pseudo.md`
- why_integrated: `ajaxctl crystallize`, `validate`, and `promote` are live commands, `ajax_core` has `auto_crystallize`, and `agency/crystallization.py` defines stable episode/recipe/validation/habit artifacts.
- outside_canon_after_review: none

### provider_benchmark_and_canary_surface

- source: `ROOT/PSEUDOCODE_MAP/00_SYSTEM_FLOW.pseudo.md`
- corroborated_by: `code`, `cli`
- decision: `integrate_now`
- target_files: `PSEUDOCODE_MAP/01_INTERFACES.pseudo.md`, `PSEUDOCODE_MAP/02_ARTIFACTS_AND_STATE.pseudo.md`
- why_integrated: `cloud-canary`, `cloud-bench`, `lmstudio-bench`, and `lmstudio-test` are live CLI surfaces with stable artifact outputs and fallback-local-model state.
- outside_canon_after_review: heartbeat/lab-internal canaries remain documented by their own runtime modules, not by this operator-facing pseudocode block.

### trust_promotion_claim_contract_detail

- source: `MICROFILM.md`
- corroborated_by: `code`, `tests`
- decision: `partial_integrate`
- target_files: `PSEUDOCODE_MAP/00_SYSTEM_FLOW.pseudo.md`
- why_integrated: `agency/microfilm_guard.py`, `agency/types/output_bundle.py`, and `agency/anti_optimism_guard.py` expose a stable live contract for `promote_trust`, `SOFT_BLOCK`, and the two valid `OutputBundle` shapes.
- outside_canon_after_review: doctrinal wording from secondary sources was not copied; only the live contract entered the map.

### capability_gap_triage_and_research_flow

- source: `AGENTS.md`
- corroborated_by: `code`, `cli`, `docs_live`
- decision: `partial_integrate`
- target_files: `PSEUDOCODE_MAP/01_INTERFACES.pseudo.md`, `PSEUDOCODE_MAP/02_ARTIFACTS_AND_STATE.pseudo.md`
- why_integrated: `ajaxctl gaps triage` is a live LAB-only command with stable outputs under `artifacts/gaps/`, and `ajaxctl research` returns a stable Scout report path under `artifacts/scout_sandbox/research/`.
- outside_canon_after_review: the Director/backlog merge flow in `agency/review_capability_gaps.py` and `LEANN_CAP_GAPS/research_backlog.yaml` stays outside kernel pseudocode canon.

## Pending Or Secondary-Only Remainders

### capability_gap_triage_and_research_flow

- pending_scope: `agency/review_capability_gaps.py` merge logic and `LEANN_CAP_GAPS/research_backlog.yaml`
- why_not_in_canon_now: the backlog target lives outside `ajax-kernel` exportable state and represents a broader research-governance layer, not a kernel-local runtime contract

### trust_promotion_claim_contract_detail

- pending_scope: explanatory doctrine from `MICROFILM.md` / `AGENTS.md`
- why_not_in_canon_now: the canon only imports the live contract already enforced by code, not the surrounding doctrine text
