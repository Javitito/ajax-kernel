# 02 - Artifacts and State (Code-First, Live)

## Scope
Observable files, receipts, and state slices used by the current runtime.
Only paths verified in code, tests, help output, or kernel runbooks are included.

## Main Layout

```text
artifacts/
  benchmarks/
    cloud/
    lmstudio/
  capability_gaps/
    *.json
    open/
    cancelled/
  episodes/
    episode_<mission_id>_attempt<n>.json
  efe_candidates/
    *.json
  driver/
    launcher/*.log
    revive/*_{stdout,stderr}.log
  gaps/
    triage_<stamp>.json
    triage_<stamp>.md
  health/
    ajax_heartbeat.json
    providers_status.json
  habits/
    habit_<intent_class>_v<n>.json
  history/
    mission-<id>.json
  indexes/
    crystallization_index.json
  lab/
    session/expected_session.json
  provider_ledger/
    latest.json
    _snapshots/latest_<utc>.json
  recipes/
    candidates/recipe_<intent_class>_<pattern>.json
    validated/validation_<ts>_<recipe>.json
  receipts/
    cloud_canary_<ts>.json
    crystallization_seed_<ts>_<event>_<mission>.json
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
  scout_sandbox/
    research/<yyyy-mm-dd>_<topic-slug>/report.md
  subcalls/
    subcall_<ts>.json
    subcall_<ts>.txt
  waiting_for_user/
    <mission_id>.json
    completions/<mission_id>/boundary_completion_<ts>.json
    _archived/<yyyy-mm-dd>/*.json
  state/
    auto_crystallize.flag
    waiting_mission.json
    fallback_local_model.json
  pids/
    prod_os_driver_<port>.pid
    prod_os_driver_<port>.json
```

## Capability and EFE Artifacts

- `artifacts/capability_gaps/*.json` is the durable fail-closed bucket for execution blockers and degraded outcomes.
- `artifacts/efe_candidates/*.json` stores deterministic candidate EFEs generated from gaps or plan evidence.
- `doctor metabolism` reads both directories to summarize loop debt and next actions.

## Mission and Waiting State

- `artifacts/history/mission-<id>.json` stores structured mission history for inspector/crystallization flows.
- `state_dir/waiting_mission.json` stores the resumable mission payload; shipped LAB tooling reads this as `artifacts/state/waiting_mission.json`.
- `artifacts/waiting_for_user/<mission_id>.json` stores the operator-facing pending payload for the same mission.
- `artifacts/waiting_for_user/completions/<mission_id>/boundary_completion_<ts>.json` stores auditable structured boundary-completion payloads tied to the same waiting mission.
- `ops friction gc --apply` archives old `waiting_for_user` payloads into `_archived/<date>/`.

## Knowledge-Lift Artifacts

- `artifacts/episodes/episode_<mission_id>_attempt<n>.json` stores the durable governed episode linked back to mission/run evidence and pattern metadata.
- `artifacts/recipes/candidates/recipe_*.json` stores candidate recipes generated only after repeated pattern evidence crosses the deterministic threshold.
- `artifacts/recipes/validated/validation_<ts>_<recipe>.json` stores eligibility reports produced by `validate recipe`.
- `artifacts/habits/habit_<intent_class>_v<n>.json` stores promoted habits only after eligible validation.
- `artifacts/indexes/crystallization_index.json` is the durable episode/recipe/habit index for this pipeline.
- `artifacts/state/auto_crystallize.flag` is the persisted switch behind `crystallize auto on|off`.
- `artifacts/receipts/crystallization_seed_*.json` records considered / skipped / created / validated / not-promoted decisions for the LAB learning loop.

## Health and Provider State

- `artifacts/health/ajax_heartbeat.json` is the lightweight system heartbeat.
- `artifacts/health/providers_status.json` is the live provider health snapshot used by routing, diagnostics, Starting XI, and LAB autopilot.
- `artifacts/provider_ledger/latest.json` is the durable provider availability / cooldown ledger.
- `artifacts/provider_ledger/_snapshots/latest_<utc>.json` is created when friction GC applies a minimum-budget reset.

## Receipts and Validation

### Exec and subcall receipts

- `exec_<ts>.json` is written by `_record_exec_receipt()` with schema `ajax.exec_receipt.v1`.
- `subcall_<ts>.json` is written by `run_subcall()` with schema `ajax.subcall_receipt.v1`.
- `cloud_canary_<ts>.json` is written by `cloud-canary` with schema `ajax.cloud_canary.v1`.
- `waiting_boundary_resume_<ts>_<event>.json` is written by `complete_waiting_boundary()` with schema `ajax.receipt.waiting_boundary_resume.v1`.
- `crystallization_seed_<ts>_<event>_<mission>.json` is written by the crystallization runtime/engine with schema `ajax.receipt.crystallization_seed.v1`.
- `driver_health_checked_<ts>.json`, `driver_revive_*_<ts>.json` are written by the rail-aware revive path with schema `ajax.driver_revive_receipt.v1`.
- `artifacts/driver/launcher/*.log` is written by `Start-AjaxDriver.ps1`; `artifacts/driver/revive/*_{stdout,stderr}.log` stores the bounded launcher transport logs captured by `_launch_target()`.
- `artifacts/pids/prod_os_driver_<port>.pid|json` is written by `drivers/os_driver.py` while the minimal real PROD entrypoint is alive.
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
  - `ajax.receipt.waiting_boundary_resume.v1`
    -> `ajax.receipt.waiting_boundary_resume.v1.schema.json`
  - `ajax.receipt.crystallization_seed.v1`
    -> `ajax.receipt.crystallization_seed.v1.schema.json`
  - `ajax.topology_doctor.v0`
  - `ajax.topology_doctor.v1`
    -> `ajax.topology_doctor.v0.schema.json`
- WARN currently covers unsupported or missing schema metadata.
- FAIL covers IO, JSON parse, or schema validation failure.

## Gap Triage and Research Artifacts

- `artifacts/gaps/triage_<stamp>.json|md` is written by `gaps triage` from `artifacts/capability_gaps/open` plus `experiments/runs/*/epistemic_feedback.json`.
- `artifacts/scout_sandbox/research/<date>_<topic-slug>/report.md` is the stable research report output returned by `ajaxctl research`.
- External backlog management in `LEANN_CAP_GAPS/research_backlog.yaml` stays outside kernel pseudocode canon.

## LAB, Benchmark, and Local Fallback State

- `artifacts/lab/session/expected_session.json` is the canonical LAB session anchor.
- `artifacts/benchmarks/cloud/` stores cloud benchmark outputs.
- `artifacts/benchmarks/lmstudio/<ts>_bench.{json,md}` plus `LMSTUDIO_BENCH_LATEST.json` store local benchmark outputs.
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
| Knowledge-lift pipeline | `agency/crystallization.py`, `agency/ajax_core.py`, `bin/ajaxctl crystallize --help`, `bin/ajaxctl promote --help` |
| Gaps / research | `agency/gaps_triage.py`, `agency/scout.py`, `bin/ajaxctl gaps --help`, `bin/ajaxctl research --help` |
| Bench / canary | `agency/cloud_bench.py`, `agency/lmstudio_bench.py`, `bin/ajaxctl cloud-canary --help`, `bin/ajaxctl lmstudio-bench --help` |
| Local fallback | `agency/lmstudio_bench.py` |
