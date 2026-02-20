# Pseudocode Delta — Microfilm/LAB (2026-02-20)

## 1) Root resolution (CLI)

```text
_detect_repo_root(root_override):
  candidate = root_override or ROOT
  if candidate has markers:
    if candidate.name != 'ajax-kernel' and candidate/ajax-kernel has markers:
      return candidate/ajax-kernel
    return candidate
  if candidate/ajax-kernel has markers:
    return candidate/ajax-kernel
  raise repo_root_not_detected
```

## 2) LAB init bootstrap

```text
lab init --root X:
  root = _detect_repo_root(X)
  require basename(root) == 'ajax-kernel'
  ensure files exist (idempotent):
    config/lab_org_manifest.yaml
    config/explore_policy.yaml
    scripts/ops/get_human_signal.ps1 (fail-closed stub => HUMAN_ACTIVE on failure)
  ensure display_map.display_targets.lab points to dummy target
  print created/updated + next steps
```

## 3) Anchor severity in LAB dummy mode

```text
evaluate_anchor_snapshot(rail='lab'):
  if expected_session_missing and display_target_is_dummy:
    severity = WARN (non-blocking)
  else:
    severity = BLOCKED
```

## 4) Services doctor fallback (when PowerShell doctor missing/fails)

```text
run_services_doctor:
  run tools/ajax_doctor.ps1
  if script missing or command fails:
    fallback_local_probe = true
    ports_map[port] = {SessionId: null, LocalProbe: true} if TCP listener exists
    health[port] = probe http://127.0.0.1:port/health
      - include X-AJAX-KEY if available
      - HTTP 401/403 counts as reachable driver
```

## 5) Microfilm check (when applicable)

```text
microfilm check:
  anchor = run_anchor_preflight
  health_ttl = provider_status_ttl

  actuation_context = has(snapshot0) or has(undo_path)

  ssc = enforce_ssc(actuation=actuation_context)
  if not actuation_context: ssc.code = SSC_NOT_APPLICABLE

  separation = enforce_lab_prod_separation(
    rail, display_target, human_active, anchor_mismatches+warnings
  )

  evidence = enforce_evidence_tiers(
    ok=anchor.ok,
    verification_mode='real',
    driver_simulated=is_dummy_driver_simulated(...)
  )

  verify_before_done = enforce_verify_before_done only if actuation_context
  undo_guard = enforce_undo_for_reversible only if actuation_context

  overall_ok = all(check.ok)
```
