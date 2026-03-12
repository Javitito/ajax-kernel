# 01 - Interfaces (Code-First, Live)

## Scope
Live CLI and operator-visible contracts in `ajax-kernel` main.
Only commands verified in current help output or code are listed here.

## Core CLI Surface

| Command | Live contract |
| --- | --- |
| `python bin/ajaxctl doctor {drivers,anchor,boot,vision,auth,council,providers,provider,bridge,leann,topology,receipts,metabolism}` | Read-only diagnostics for runtime, provider health, topology, receipts, and loop state. |
| `python bin/ajaxctl providers status [--json] [--no-refresh]` | Provider snapshot and preflight entrypoint. Refreshes or reuses `providers_status` / ledger state. |
| `python bin/ajaxctl providers ping --provider <id> [--model <id>] [--timeout-ttft ms] [--timeout-total ms] [--timeout-stall ms] [--json]` | Deterministic provider ping with TTFT / stall / total timeout overrides. |
| `python bin/ajaxctl provider ping ...` | Alias for `providers ping`. |
| `python bin/ajaxctl subcall --role <role> --tier <T0|T1|T2> [--json] [--stdin] [--allow-premium-subcall]` | DEV delegation surface. Emits durable subcall receipts and role artifacts. |
| `python bin/ajaxctl council demo` | Minimal scout -> coder -> auditor -> judge flow without touching code. |
| `python bin/ajaxctl verify efe apply-candidate --gap <gap.json> --out <efe_final.json>` | Read-only helper that materializes editable EFE from a gap candidate. |
| `python bin/ajaxctl ops friction gc --dry-run|--apply [--older-than-hours N]` | Safe hygiene for `waiting_for_user` backlog and provider ledger minimum-budget reset. |
| `python bin/ajaxctl crystallize mission <mission_id>` | Materializes a deterministic episode from a recorded mission attempt and only creates a candidate recipe when repeated evidence crosses the current threshold. |
| `python bin/ajaxctl crystallize auto {on,off}` | Toggles post-mission auto-crystallize via a persisted flag. |
| `python bin/ajaxctl validate recipe <recipe> [--runs N --source episodes]` | Validates a candidate recipe against observed episode evidence only. |
| `python bin/ajaxctl validate receipt <file-or-glob>` | Schema validation for one or more receipts outside the doctor summary. |
| `python bin/ajaxctl promote recipe <recipe>` / `promote eligible ...` | Promotes only previously validated eligible recipes into habits. |
| `python bin/ajaxctl inspect [--last N] [--drivers]` | Summarizes recent mission history and optional driver failure context. |
| `python bin/ajaxctl gaps triage [--top N --rail lab]` | LAB-only ranking of open capability gaps with suggested probes. |
| `python bin/ajaxctl research <topic> [--context text]` | Runs Scout and emits a research report path under the artifact sandbox. |
| `python bin/ajaxctl cloud-canary --provider <id> [--model <id>] [--json]` | Deterministic cloud provider canary with receipt output. |
| `python bin/ajaxctl cloud-bench [--suite crawl_v1 --providers ... --runs N --budget label]` | Multi-provider cloud benchmark surface. |
| `python bin/ajaxctl lmstudio-bench [--suite crawl_v1 --models ... --runs N --select-best --json]` | Local benchmark surface that can update fallback local model state. |
| `python bin/ajaxctl lmstudio-test [--json]` | Local canary that reads `fallback_local_model.json`. |
| `python bin/ajaxctl lab ensure --rail {lab,prod}` | Idempotent LAB rail start/recovery with verify fail-closed. |
| `python bin/ajaxctl lab start|stop|restart|status` | Direct LAB worker lifecycle controls. |
| `python bin/ajaxctl lab session {init,status,revoke}` | File-based expected session anchor for LAB. |
| `python bin/ajaxctl lab autopilot [--dry-run|--once|--daemon ...]` | Periodic LAB worker that runs gated background tasks. |
| `python bin/ajaxctl lab {queue,enqueue,acknowledge,inbox,cancel,requeue,snap,prune,web,pause-org,resume-org,probe-complete,probe-apply}` | Queue, evidence, and maintenance controls exposed by current `lab --help`. |

## Diagnostics With Explicit Runtime Contracts

### `doctor auth`

```text
python bin/ajaxctl doctor auth [--json] [--timeout <seconds>]
```

- Real auth, reachability, and quota diagnostic for council providers.
- Produces actionable `next_hint` entries such as `doctor council` or retrying subcalls after auth fixes.

### `doctor council`

```text
python bin/ajaxctl doctor council [--json]
```

- Checks executable council health.
- Includes role strategy, auth/config status, and latest subcall evidence by role.

### `doctor receipts`

```text
python bin/ajaxctl doctor receipts [--since-min <minutes>] [--strict] [--top-k N] [--summary-only] [--json]
```

- Scans recent receipts under `artifacts/receipts/`.
- Current behavior is `PASS / WARN / FAIL`, not binary-only.
- `--strict` upgrades WARN rows to a failing exit code.
- `--top-k` and `--summary-only` are live today.

### `doctor metabolism`

```text
python bin/ajaxctl doctor metabolism --since-min <minutes>
```

- Summarizes recent capability gaps, `efe_candidates`, provider state, and `waiting_for_user` backlog.
- Returns hints such as `verify efe apply-candidate` or `ops friction gc --dry-run`.

## Provider and Governance Surface

- `subcall` routes by role, tier, cost mode, provider policy, and provider ledger.
- `council_subcall_layer` keeps a live role strategy for `scout`, `coder`, `auditor`, and `judge`.
- `doctor auth` and `doctor council` are the primary operator entrypoints when subcalls degrade because of auth, quota, bridge, or timeout problems.
- `providers status` is the explicit health snapshot surface; `providers ping` is the explicit latency surface.

## Knowledge Lift and Research Surface

- `crystallize mission` lifts one recorded mission into an episode and may emit `candidate_recipe_skipped` when repetition or governance evidence is still insufficient.
- `crystallize auto on|off` controls the persisted override; current runtime defaults to auto-on in LAB and guarded refusal in PROD.
- `validate recipe` only uses observed episode evidence; replay validation is still rejected by current code.
- `promote recipe` and `promote eligible` require a prior eligible validation before writing habits; auto-crystallize does not claim habits on its own.
- `inspect` is the compact read-only entrypoint for recent mission history.
- `gaps triage` is LAB-only and writes ranked probes for open capability gaps.
- `research` is the live operator entrypoint for Scout reports; external-search heuristics stay outside canonical pseudocode.

## Benchmark and Canary Surface

- `cloud-canary` is the explicit cloud reachability / parse / validation check with a receipt.
- `cloud-bench` is the explicit multi-provider benchmark surface.
- `lmstudio-bench` writes benchmark results and can update local fallback model state.
- `lmstudio-test` is the matching local canary that consumes `fallback_local_model.json`.

## LAB Control Surface

- `lab ensure` is the safest entrypoint for starting or recovering the LAB rail.
- `lab start|stop|restart|status` are the direct worker controls.
- `lab session init|status|revoke` manages the expected LAB session anchor stored in `artifacts/lab/session/expected_session.json`.
- `lab autopilot` is the periodic, gated background loop with `--dry-run`, `--once`, and `--daemon`.
- Queue and maintenance commands remain part of the live surface exposed by `lab --help`.

## Evidence Pointers

| Interface family | Evidence pointers |
| --- | --- |
| Doctor CLI | `bin/ajaxctl --help`, `bin/ajaxctl doctor --help`, `agency/auth_provider_diagnostics.py`, `agency/council_subcall_layer.py` |
| Provider status / ping | `bin/ajaxctl providers --help`, `bin/ajaxctl providers ping --help`, `agency/ajax_core.py` |
| Subcall / council demo | `bin/ajaxctl subcall --help`, `bin/ajaxctl council --help`, `agency/subcall.py`, `agency/council_subcall_layer.py` |
| EFE helper | `bin/ajaxctl verify efe apply-candidate --help`, `agency/verify/efe_apply_candidate.py` |
| Friction GC | `bin/ajaxctl ops friction gc --help`, `agency/friction.py`, `tests/test_kernel_friction_gc_v1.py` |
| Recipe lifecycle | `bin/ajaxctl crystallize --help`, `bin/ajaxctl validate --help`, `bin/ajaxctl promote --help`, `agency/crystallization.py`, `agency/ajax_core.py` |
| Gaps / research | `bin/ajaxctl gaps --help`, `bin/ajaxctl research --help`, `agency/gaps_triage.py`, `agency/scout.py` |
| Bench / canary | `bin/ajaxctl cloud-canary --help`, `bin/ajaxctl cloud-bench --help`, `bin/ajaxctl lmstudio-bench --help`, `agency/cloud_bench.py`, `agency/lmstudio_bench.py` |
| LAB controls | `bin/ajaxctl lab --help`, `bin/ajaxctl lab session --help`, `agency/lab_session_anchor.py`, `agency/lab_autopilot.py` |
