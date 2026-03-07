# 02 - Artifacts and State (Code-First, Live)

## Scope
Observable files, receipts, and state slices used by the current runtime.
Only paths verified in code, tests, help output, or kernel runbooks are included.

## Main Layout

```text
artifacts/
  capability_gaps/
    *.json
    open/
    cancelled/
  efe_candidates/
    *.json
  health/
    ajax_heartbeat.json
    providers_status.json
  history/
    mission-<id>.json
  lab/
    session/expected_session.json
  provider_ledger/
    latest.json
    _snapshots/latest_<utc>.json
  receipts/
    exec_<ts>.json
    subcall_<ts>.json
    friction_gc_v1_<ts>.json
    lab_session_init_<ts>.json
    lab_session_status_<ts>.json
    lab_session_revoke_<ts>.json
    lab_autopilot_tick_<ts>.json
    lab_autopilot_daemon_<ts>.json
    anti_optimism_degraded_<ts>.json
    ...
  subcalls/
    subcall_<ts>.json
    subcall_<ts>.txt
  waiting_for_user/
    <mission_id>.json
    _archived/<yyyy-mm-dd>/*.json
  state/
    waiting_mission.json
    fallback_local_model.json
```

## Capability and EFE Artifacts

- `artifacts/capability_gaps/*.json` is the durable fail-closed bucket for execution blockers and degraded outcomes.
- `artifacts/efe_candidates/*.json` stores deterministic candidate EFEs generated from gaps or plan evidence.
- `doctor metabolism` reads both directories to summarize loop debt and next actions.

## Mission and Waiting State

- `artifacts/history/mission-<id>.json` stores structured mission history for inspector/crystallization flows.
- `state_dir/waiting_mission.json` stores the resumable mission payload; shipped LAB tooling reads this as `artifacts/state/waiting_mission.json`.
- `artifacts/waiting_for_user/<mission_id>.json` stores the operator-facing pending payload for the same mission.
- `ops friction gc --apply` archives old `waiting_for_user` payloads into `_archived/<date>/`.

## Health and Provider State

- `artifacts/health/ajax_heartbeat.json` is the lightweight system heartbeat.
- `artifacts/health/providers_status.json` is the live provider health snapshot used by routing, diagnostics, Starting XI, and LAB autopilot.
- `artifacts/provider_ledger/latest.json` is the durable provider availability / cooldown ledger.
- `artifacts/provider_ledger/_snapshots/latest_<utc>.json` is created when friction GC applies a minimum-budget reset.

## Receipts and Validation

### Exec and subcall receipts

- `exec_<ts>.json` is written by `_record_exec_receipt()` with schema `ajax.exec_receipt.v1`.
- `subcall_<ts>.json` is written by `run_subcall()` with schema `ajax.subcall_receipt.v1`.
- `artifacts/subcalls/subcall_<ts>.json|txt` stores the role output payload sidecar for the same subcall.

### Receipt validator contract

```text
doctor_receipts(root_dir, since_min, strict, top_k, summary_only):
  scan artifacts/receipts/*.json
  validate supported schemas
  severity = PASS | WARN | FAIL
  if strict and severity == WARN:
    exit as FAIL
```

- Supported schema mappings in current code:
  - `ajax.lab.session.init.v0`
  - `ajax.lab.session.status.v0`
  - `ajax.lab.session.revoke.v0`
  - `ajax.lab.session_status.v0`
    -> `ajax.lab.session.v0.schema.json`
  - `ajax.lab.session.migrated.v1`
    -> `ajax.lab.session.migrated.v1.schema.json`
  - `ajax.lab.autopilot_tick.v1`
  - `ajax.lab.autopilot_tick.v0`
    -> `ajax.lab.autopilot_tick.v1.schema.json`
  - `ajax.topology_doctor.v0`
  - `ajax.topology_doctor.v1`
    -> `ajax.topology_doctor.v0.schema.json`
- WARN currently covers unsupported or missing schema metadata.
- FAIL covers IO, JSON parse, or schema validation failure.

## LAB and Local Fallback State

- `artifacts/lab/session/expected_session.json` is the canonical LAB session anchor.
- `artifacts/state/fallback_local_model.json` is optional local fallback state written by `lmstudio-bench --select-best` and read by `lmstudio-test`.
- LAB autopilot receipts and queue controls operate against the same artifact root and provider status snapshot.

## Evidence Pointers

| Artifact family | Evidence pointers |
| --- | --- |
| Capability gaps + EFE candidates | `agency/ajax_core.py`, `agency/metabolism_doctor.py`, `docs/RUNBOOK_LOOP_CLOSURE_E2E.md` |
| Waiting state | `agency/ajax_core.py`, `agency/lab_control.py`, `agency/friction.py` |
| History + heartbeat | `agency/history.py`, `agency/ajax_heartbeat.py` |
| Provider status + ledger | `agency/provider_breathing.py`, `agency/provider_ledger.py`, `agency/starting_xi.py` |
| Receipt validation | `agency/receipt_validator.py`, `tests/test_receipts_schema_validator.py`, `bin/ajaxctl doctor receipts --help` |
| Local fallback | `agency/lmstudio_bench.py` |

