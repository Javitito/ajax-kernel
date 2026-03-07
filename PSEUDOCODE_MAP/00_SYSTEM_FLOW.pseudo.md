# 00 - System Flow (Code-First, Live)

## Scope
Compact map of the live mission path implemented in `ajax-kernel` main.
Source of truth: runtime code, tests, CLI help, and kernel runbooks.

## Main Mission Path

```text
pursue_intent(intention, mode, rail):
  mission = MissionState(...)
  envelope = MissionEnvelope(...)
  method_pack = AJAX_METHOD_PACK

  providers_snapshot = providers_preflight(requested_tier=AJAX_COST_MODE)
  starting_xi, preflight = _preflight_starting_xi(mission)
  if preflight blocks:
    emit capability_gap / exec receipt
    return BLOCKED or GAP_LOGGED

  plan_json, repair_meta = _validate_brain_plan_with_efe_repair(...)
  if repair_meta.reason == "missing_efe_final" or plan has no usable expected_state:
    gap = _emit_missing_efe_gap(...)
    # optional deterministic candidate:
    # artifacts/efe_candidates/<safe_gap_id>_<utc>.json
    record exec receipt
    mission.status = "BLOCKED"
    return

  if policy or rail requires governance:
    verdict = council.review(...)
    if verdict asks for human input:
      waiting_path = _persist_waiting_mission(...)
      record exec receipt
      return WAITING_FOR_USER
    if verdict rejects:
      record exec receipt
      return GAP_LOGGED or BLOCKED

  if plan implies physical actuation:
    snapshot0 = _capture_snapshot0(...)
    ssc = enforce_ssc(actuation=true, snapshot0=snapshot0)
    rail_gate = enforce_lab_prod_separation(...)
    if not ssc.ok or not rail_gate.ok:
      record exec receipt
      return BLOCKED or WAITING_FOR_USER
    tx_paths = _tx_prepare(...)

  result = execute_plan_steps(...)
  if result requests ASK_USER:
    waiting_path = _persist_waiting_mission(...)
    record exec receipt
    return WAITING_FOR_USER

  verification = verify(...)
  verification.verification_mode = real | synthetic | manual
  verification.driver_simulated = _driver_simulated()
  verification = enforce_evidence_tiers(...)
  done_gate = enforce_verify_before_done(result, verification)

  record exec receipt
  write mission history
  maybe_auto_crystallize(...)
  return DONE | WAITING_FOR_USER | GAP_LOGGED
```

## Provider Snapshot Before Planning

- `providers_preflight()` builds `ajax.providers_preflight.v1` from provider config, health, and ledger state before planning.
- `_preflight_starting_xi()` selects the per-role roster used by plan/council/runtime. Failure happens before actuation, not after.
- `Starting XI` is persisted in mission metadata for retries, council review, and post-failure diagnostics.
- Explicit operator diagnostics exist outside the main loop:
  - `python bin/ajaxctl providers status`
  - `python bin/ajaxctl providers ping --provider <id>`
  - `python bin/ajaxctl provider ping --provider <id>` (alias)

## Fail-Closed EFE Loop

```text
validate_plan_with_efe(plan):
  expected_state = extract success_spec.expected_state
  if expected_state is complete:
    return plan

  repaired_plan = efe_repair(plan)
  if repaired_plan is complete:
    return repaired_plan

  gap = _emit_missing_efe_gap(...)
  # gap may embed:
  # - efe_repair_receipt
  # - efe_candidate_path
  # - efe_candidate_status
  # - efe_candidate_reason
  return BLOCKED
```

- `_autogen_missing_efe_candidate()` writes deterministic candidate payloads under `artifacts/efe_candidates/`.
- `python bin/ajaxctl verify efe apply-candidate --gap <gap.json> --out <efe_final.json>` materializes an editable final EFE payload without executing any action.
- `doctor metabolism` reads both recent gaps and `artifacts/efe_candidates/` to surface this backlog.

## Physical and Rail Gates

- `_capture_snapshot0()` records the pre-actuation state used by SSC and recovery logic.
- `enforce_ssc()` blocks live actuation when `snapshot0` is missing.
- `enforce_lab_prod_separation()` checks rail, driver state, display target, and human presence before live actuation.
- `_tx_prepare()` writes transaction state and undo plan before mutating steps.
- Trust promotion after verification is blocked when the driver is simulated, even if the step passed.

## WAITING_FOR_USER Semantics

```text
if plan or runtime needs human clarification:
  mission.status = WAITING_FOR_USER
  write state_dir/waiting_mission.json
  write artifacts/waiting_for_user/<mission_id>.json
  keep same mission_id for resume
```

- `ASK_USER` is non-terminal. The runtime resumes the same mission instead of creating a new one.
- `ops friction gc --apply` can archive old `waiting_for_user` payloads into `_archived/<date>/`.

## Verification and Claim Discipline

- Verification carries explicit mode and driver context:
  - `verification_mode`
  - `driver_online`
  - `driver_simulated`
- `enforce_evidence_tiers()` promotes trust only for `verification_mode=real` with a live, non-simulated driver.
- `anti_optimism_guard` validates `OutputBundle` claims and degrades unsupported claims to hypothesis plus verification commands.

## Evidence Pointers

| Live item | Evidence pointers |
| --- | --- |
| Mission entrypoint + lifecycle | `agency/ajax_core.py`, `agency/mission_envelope.py`, `agency/method_pack.py` |
| Provider preflight + Starting XI | `agency/ajax_core.py`, `agency/starting_xi.py`, `tests/test_kernel_preflight_fail_closed.py` |
| EFE repair + fail-closed gap | `agency/ajax_core.py`, `agency/efe_repair.py`, `tests/test_kernel_missing_efe_autogen.py` |
| Apply-candidate helper | `agency/verify/efe_apply_candidate.py`, `docs/RUNBOOK_LOOP_CLOSURE_E2E.md` |
| SSC / rail / evidence tiers | `agency/microfilm_guard.py`, `tests/test_microfilm_guard_unit.py`, `tests/test_kernel_anchor_guard.py` |
| WAITING_FOR_USER persistence | `agency/ajax_core.py`, `tests/test_kernel_ask_user_path.py`, `tests/test_deference_waiting.py` |

