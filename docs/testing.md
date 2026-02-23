# Testing Notes (Local / WSL)

## `python -m pytest` hang diagnosis (2026-02-23)

### Scope
- Command affected: `python -m pytest`
- In this repo, `pyproject.toml` limits default discovery to `tests/test_kernel_*.py` and sets `addopts = "-q"`.
- In this environment (WSL + repo root with large `artifacts/`), the default suite may appear to "hang".

### Repro (short)
1. `python -m pytest -vv -x --maxfail=1`
2. When it stalls, observe the last printed test name.

Observed blockers in this environment:
- `tests/test_kernel_cli.py::test_ajaxctl_soak_check_generates_report`
- `tests/test_kernel_smoke.py::test_compileall_agency_ajax`

### Root cause (evidence-based)
1. `test_ajaxctl_soak_check_generates_report` runs:
   - `python bin/ajaxctl soak check --root .`
2. In this repo root, `soak check` can spend a long time in `agency.soak_gate._collect_lab_org_activity` scanning and reading many receipts under `artifacts/lab_org/*/receipt.json`.
3. `test_compileall_agency_ajax` runs:
   - `python -m compileall -q .`
4. `compileall -q .` traverses the entire repo (including `artifacts/`), which is slow/timeout-prone in this environment.

### Repro commands (direct)
- `timeout 30s python bin/ajaxctl soak check --root .`
- `timeout 30s python -m compileall -q .`
- (Plugin check) `timeout 20s python -m pytest -vv -p no:timeout tests/test_kernel_cli.py::test_ajaxctl_soak_check_generates_report`
  - If this still times out, `pytest-timeout` is not the root cause.

### Local mitigation (recommended)
Run the default kernel suite excluding the two environment-heavy tests:

```bash
python -m pytest -vv -k "not test_ajaxctl_soak_check_generates_report and not test_compileall_agency_ajax"
```

Expected result in this environment:
- `21 passed, 2 deselected`

### Optional follow-ups (separate PRs)
- Make `test_kernel_cli.py::test_ajaxctl_soak_check_generates_report` run against a minimal temp root fixture instead of `--root .`.
- Change `test_kernel_smoke.py::test_compileall_agency_ajax` to compile source dirs only (e.g. `agency ajax ui tests`) instead of `.`.
- Add explicit subprocess timeouts in long-running CLI smoke tests so failures are bounded (fail-fast instead of hang).
