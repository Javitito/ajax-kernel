# 00 - Provider Preflight (FAST/SAFE)

## Purpose
Definir el preflight de providers para que use senales observadas cuando existan.

## Pseudocode

```text
provider_preflight(role, rail, now):
  policy = load provider policy/config
  ledger = read artifacts/provider_ledger/latest.json
  status = read artifacts/health/providers_status.json

  candidates = policy.rails[rail].roles[role].preference
  effective = []

  for provider in candidates:
    row = ledger[provider]
    if row.status != "ok":
      continue
    if row.cooldown_until_ts > now:
      continue
    effective.push(provider)

  for provider in effective:
    observed = status[provider]
    if observed has p95/ttft/stall:
      timeout_base = observed metrics
    else:
      timeout_base = config defaults

  if effective is empty:
    return quota_or_unavailable -> escalate per policy

  return effective ordered by policy + observed health
```

## Invariants
- Nunca contar `status != ok` como disponible efectivo.
- Quorum y fallback se gobiernan por config + ledger.
- En modo rapido, usar metrica observada (`providers_status`) cuando exista.
