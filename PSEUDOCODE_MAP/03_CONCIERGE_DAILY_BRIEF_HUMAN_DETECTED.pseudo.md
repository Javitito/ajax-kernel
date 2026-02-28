# 03 - Concierge Daily Brief on HUMAN_DETECTED

## Scope
Flujo read-only disparado en `lab_org` cuando hay transicion AWAY -> HUMAN_DETECTED.

## Trigger and cooldown

```text
lab_org_tick(receipt):
  explore = evaluate_explore_state(...)
  receipt.trigger = explore.trigger
  receipt.state = explore.state

  if receipt.trigger == "AWAY->HUMAN_DETECTED":
    preempt ui jobs
    write artifacts/lab/preemption_<ts>.json
    daily = maybe_trigger_human_detected_brief(root, now_ts)
    receipt.daily_brief = summarize(daily)
```

```text
maybe_trigger_human_detected_brief(root, now_ts, cooldown_s=21600):
  state = read artifacts/concierge/daily_brief_state.json
  if state.last_emit_ts exists and (now_ts - last_emit_ts) < cooldown_s:
    write artifacts/receipts/daily_brief_<ts>.json with status="cooldown_skipped"
    return generated=false

  return generate_daily_brief(root, now_ts)
```

## Brief generation

```text
generate_daily_brief(root, now_ts):
  A = run audits/providers + gates + eki(if registered)
  B = scan artifacts/receipts/*.json in 24-72h window
      -> top 3 gap_codes by frequency
  C = query LEANN index antigravity_skills_safe using gap-derived queries
      -> top 3 skills {name, rationale, evidence_refs}

  if LEANN index missing/unavailable:
    status = "capability_missing"
    C.top_skills = []
    include hypothesis + verification_commands

  write:
    artifacts/concierge/daily_brief_<ts>.json
    artifacts/concierge/daily_brief_<ts>.md
    artifacts/receipts/daily_brief_<ts>.json
    artifacts/concierge/daily_brief_state.json
```

## Invariants
- Read-only: no remediate jobs, no mutation outside report artifacts/receipts.
- Anti-spam: max 1 brief per cooldown window (6h).
- Fail-closed for LEANN capability: degrade block C, do not crash.
