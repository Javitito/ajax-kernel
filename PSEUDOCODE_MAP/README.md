# PSEUDOCODE_MAP

This directory is the canonical, live, exportable pseudocode map for AJAX.

## Status

- Canonical location: `ajax-kernel/PSEUDOCODE_MAP/`
- Role: compact pseudocode derived from the live kernel code, tests, CLI help, and active runbooks
- Audience: operators, auditors, and export consumers of the live system

## Historical Duality

Two `PSEUDOCODE_MAP/` directories exist because an older ROOT copy predated the kernel-focused code-first regeneration.
That ROOT copy is preserved for lineage and source comparison only.
It is no longer the live default and it must not compete with this directory as canon.

## Root Copy Policy

- `AJAX_HOME/PSEUDOCODE_MAP/` is a deprecated source archive
- if ROOT and KERNEL differ, KERNEL wins
- any future sync from ROOT to KERNEL must be explicit, audited, and manual
- automatic sync or "silent refresh" from ROOT is forbidden

## Drift Guard

If a future change revives ROOT as an active second canon, treat it as documentation drift and stop for review.
Canonical export decisions belong here unless a new audit explicitly changes that policy.
