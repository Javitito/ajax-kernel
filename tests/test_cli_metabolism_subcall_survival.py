from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

from agency.metabolism_doctor import run_doctor_metabolism


def _load_ajaxctl_module():
    loader = importlib.machinery.SourceFileLoader("ajaxctl_mod_survival", "bin/ajaxctl")
    spec = importlib.util.spec_from_loader("ajaxctl_mod_survival", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_cli_doctor_metabolism_command_registered() -> None:
    env = os.environ.copy()
    env["AJAX_DOCTOR_METABOLISM_CRITICAL_GAPS"] = "9999"
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "doctor", "metabolism", "--since-min", "180"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload.get("schema") == "ajax.doctor.metabolism.v0"


def test_metabolism_no_side_effects(tmp_path: Path) -> None:
    _write_json(tmp_path / "artifacts" / "capability_gaps" / "g.json", {"reason": "missing_efe_final"})
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    _ = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)

    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert before == after


def test_metabolism_reports_waiting_counts(tmp_path: Path) -> None:
    waiting = tmp_path / "artifacts" / "waiting_for_user" / "w1.json"
    _write_json(waiting, {"status": "WAITING_FOR_USER"})
    ts = time.time() - 3600.0
    os.utime(waiting, (ts, ts))

    payload = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)
    waiting_backlog = payload.get("waiting_backlog") or {}
    hints = payload.get("next_hint") or []

    assert waiting_backlog.get("count") == 1
    assert float(waiting_backlog.get("oldest_age_min") or 0.0) >= 59.0
    assert any("ops friction gc --dry-run" in str(h) for h in hints)


def test_subcall_local_does_not_require_role(monkeypatch) -> None:
    ajaxctl = _load_ajaxctl_module()
    captured = {}

    class _Outcome:
        ok = True
        terminal = "DONE"
        error = None
        output_text = ""

    def _fake_run_subcall(**kwargs):
        captured.update(kwargs)
        return _Outcome()

    monkeypatch.setitem(sys.modules, "agency.subcall", types.SimpleNamespace(run_subcall=_fake_run_subcall))

    rc = ajaxctl.main(["subcall", "--local", "hola"])
    assert rc == 0
    assert captured.get("role") == "survivor"
    assert captured.get("tier") == "T1"
    assert captured.get("prompt") == "hola"


def test_subcall_role_override_still_works(monkeypatch) -> None:
    ajaxctl = _load_ajaxctl_module()
    captured = {}

    class _Outcome:
        ok = True
        terminal = "DONE"
        error = None
        output_text = ""

    def _fake_run_subcall(**kwargs):
        captured.update(kwargs)
        return _Outcome()

    monkeypatch.setitem(sys.modules, "agency.subcall", types.SimpleNamespace(run_subcall=_fake_run_subcall))

    rc = ajaxctl.main(
        ["subcall", "--local", "hola", "--role", "reviewer", "--tier", "T2"]
    )
    assert rc == 0
    assert captured.get("role") == "reviewer"
    assert captured.get("tier") == "T2"


def test_help_lists_metabolism() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "doctor", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "metabolism" in (proc.stdout or "")


def test_cli_exit_codes(monkeypatch) -> None:
    ajaxctl = _load_ajaxctl_module()

    def _critical(*, root_dir, since_min):  # noqa: ARG001
        return {"schema": "ajax.doctor.metabolism.v0", "exit_code": 1}

    monkeypatch.setitem(
        sys.modules,
        "agency.metabolism_doctor",
        types.SimpleNamespace(run_doctor_metabolism=_critical),
    )
    rc = ajaxctl.main(["doctor", "metabolism", "--since-min", "180"])
    assert rc == 1


def test_receipt_written_if_metabolism_emits_receipt(tmp_path: Path, monkeypatch) -> None:
    ajaxctl = _load_ajaxctl_module()
    receipt = tmp_path / "artifacts" / "receipts" / "doctor_metabolism.json"
    _write_json(receipt, {"ok": True})

    def _with_receipt(*, root_dir, since_min):  # noqa: ARG001
        return {
            "schema": "ajax.doctor.metabolism.v0",
            "exit_code": 0,
            "receipt_path": str(receipt),
        }

    monkeypatch.setitem(
        sys.modules,
        "agency.metabolism_doctor",
        types.SimpleNamespace(run_doctor_metabolism=_with_receipt),
    )
    rc = ajaxctl.main(["doctor", "metabolism", "--since-min", "180"])
    assert rc == 0
    assert receipt.exists()
