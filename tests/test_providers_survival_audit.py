from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, List

import pytest

import agency.audits.providers_survival_audit as audit_mod


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_yaml_compatible_json(path: Path, payload: Dict[str, Any]) -> None:
    # YAML parser accepts JSON syntax; avoids depending on yaml emitter in tests.
    _write_json(path, payload)


def _base_policy(*, providers: List[str] | None = None) -> Dict[str, Any]:
    ladder = providers or ["alpha", "beta"]
    return {
        "schema": "ajax.provider_policy.v1",
        "rails": {
            "prod": {
                "roles": {
                    "brain": {"preference": list(ladder)},
                    "council": {"preference": list(ladder)},
                    "scout": {"preference": list(ladder)},
                }
            },
            "lab": {
                "roles": {
                    "brain": {"preference": list(ladder)},
                    "council": {"preference": list(ladder)},
                    "scout": {"preference": list(ladder)},
                }
            },
        },
        "providers": {
            "alpha": {"cost_class": "paid"},
            "beta": {"cost_class": "generous"},
        },
    }


def _base_status(*provider_ids: str) -> Dict[str, Any]:
    if not provider_ids:
        provider_ids = ("alpha", "beta")
    providers = {}
    for idx, pid in enumerate(provider_ids):
        providers[pid] = {
            "auth_state": "OK",
            "available_recent": True,
            "latency_p95_ms": 1200 + (idx * 100),
            "total_p95_ms": 2200 + (idx * 100),
            "ttft_p95_ms": 700 + (idx * 50),
            "breathing": {"status": "UP", "contract_status": "UP", "roles": {}},
            "transport": {"status": "UP"},
        }
    return {
        "schema": "ajax.providers_status.v1",
        "updated_ts": 1700000000.0,
        "updated_utc": "2023-11-14T22:13:20Z",
        "providers": providers,
    }


def _ledger_rows_ok(*provider_ids: str) -> List[Dict[str, Any]]:
    if not provider_ids:
        provider_ids = ("alpha", "beta")
    rows: List[Dict[str, Any]] = []
    for pid in provider_ids:
        for role in ("brain", "council", "scout"):
            rows.append(
                {
                    "provider": pid,
                    "model": "m1",
                    "role": role,
                    "status": "ok",
                    "reason": None,
                    "cooldown_until": None,
                    "cooldown_until_ts": None,
                }
            )
    return rows


def _base_ledger(*provider_ids: str) -> Dict[str, Any]:
    return {
        "schema": "ajax.provider_ledger.v1",
        "updated_ts": 1700000000.0,
        "updated_utc": "2023-11-14T22:13:20Z",
        "rows": _ledger_rows_ok(*provider_ids),
    }


def _base_timeouts() -> Dict[str, Any]:
    return {
        "schema": "ajax.provider_timeouts_policy.v1",
        "version": "test",
        "defaults": {
            "connect_timeout_ms": 1000,
            "first_output_timeout_ms": 15000,
            "stall_timeout_ms": 8000,
            "total_timeout_ms": 60000,
        },
    }


def _base_failure_policy() -> Dict[str, Any]:
    return {
        "schema": "ajax.provider_failure_policy.v1",
        "providers": {"cooldown_seconds_default": 120},
        "receipts": {"required_fields": ["schema", "ts_utc", "ok"]},
    }


def _write_minimal_repo(
    root: Path,
    *,
    policy_doc: Dict[str, Any] | None = None,
    status_doc: Dict[str, Any] | None = None,
    ledger_doc: Dict[str, Any] | None = None,
    timeouts_doc: Dict[str, Any] | None = None,
    failure_policy_doc: Dict[str, Any] | None = None,
) -> None:
    _write_json(root / "config" / "provider_policy.json", policy_doc or _base_policy())
    _write_json(root / "artifacts" / "health" / "providers_status.json", status_doc or _base_status())
    _write_json(root / "artifacts" / "provider_ledger" / "latest.json", ledger_doc or _base_ledger())
    _write_json(root / "config" / "provider_timeouts_policy.json", timeouts_doc or _base_timeouts())
    if failure_policy_doc is not None:
        _write_yaml_compatible_json(root / "config" / "provider_failure_policy.yaml", failure_policy_doc)


def _codes(out: Dict[str, Any]) -> List[str]:
    findings = out.get("findings") if isinstance(out.get("findings"), list) else []
    return [str(f.get("code")) for f in findings if isinstance(f, dict) and f.get("code")]


def test_providers_survival_audit_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_minimal_repo(tmp_path, failure_policy_doc=_base_failure_policy())
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000000.0)

    out = audit_mod.run_providers_survival_audit(tmp_path, mode="fast")

    assert out["schema"] == "ajax.audit.providers_survival.v0"
    assert out["ok"] is True
    assert out["read_only"] is True
    assert out["live_probes_invoked"] is False
    assert Path(str(out["artifact_path"])).exists()
    assert Path(str(out["artifact_md_path"])).exists()
    receipt_path = Path(str(out["receipt_path"]))
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key in ("ts_utc", "root", "ok", "summary_counts", "paths_written", "schema_version", "hashes"):
        assert key in receipt
    assert all(str(p).startswith(str(tmp_path / "artifacts")) for p in out["paths_written"])


def test_providers_survival_audit_fail_provider_in_ladder_without_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _base_policy(providers=["alpha", "ghost"])
    ledger = _base_ledger("alpha", "ghost")
    status = _base_status("alpha")
    _write_minimal_repo(
        tmp_path,
        policy_doc=policy,
        status_doc=status,
        ledger_doc=ledger,
        failure_policy_doc=_base_failure_policy(),
    )
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000001.0)

    out = audit_mod.run_providers_survival_audit(tmp_path)

    assert out["ok"] is False
    assert "policy_provider_missing_in_status" in _codes(out)


def test_providers_survival_audit_fail_ledger_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = {
        "schema": "ajax.provider_ledger.v1",
        "updated_ts": 1700000000.0,
        "updated_utc": "2023-11-14T22:13:20Z",
        "rows": [],
    }
    _write_minimal_repo(tmp_path, ledger_doc=ledger, failure_policy_doc=_base_failure_policy())
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000002.0)

    out = audit_mod.run_providers_survival_audit(tmp_path)

    assert out["ok"] is False
    assert "ledger_rows_empty" in _codes(out)


def test_providers_survival_audit_warn_hard_fail_with_expired_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _base_ledger()
    # Keep overall availability OK via beta; make alpha brain stale hard_fail with expired cooldown.
    for row in ledger["rows"]:
        if row["provider"] == "alpha" and row["role"] == "brain":
            row["status"] = "hard_fail"
            row["reason"] = "timeout"
            row["cooldown_until"] = "2023-11-14T22:12:00Z"
            row["cooldown_until_ts"] = 1699999920.0
            break
    _write_minimal_repo(tmp_path, ledger_doc=ledger, failure_policy_doc=_base_failure_policy())
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000000.0)

    out = audit_mod.run_providers_survival_audit(tmp_path)

    assert "ledger_hard_fail_cooldown_expired" in _codes(out)
    assert out["ok"] is True


def test_providers_survival_audit_warn_timeouts_policy_missing_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_minimal_repo(
        tmp_path,
        timeouts_doc={"schema": "ajax.provider_timeouts_policy.v1", "version": "test"},
        failure_policy_doc=_base_failure_policy(),
    )
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000003.0)

    out = audit_mod.run_providers_survival_audit(tmp_path)

    assert out["ok"] is True
    assert "timeouts_policy_defaults_missing" in _codes(out)


def test_providers_survival_audit_fast_mode_does_not_use_live_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_minimal_repo(tmp_path, failure_policy_doc=_base_failure_policy())
    monkeypatch.setattr(audit_mod, "_now_ts", lambda: 1700000004.0)
    calls = {"socket": 0}

    def _boom(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - would fail the test if called
        calls["socket"] += 1
        raise AssertionError("socket.create_connection should not be called by providers survival audit")

    monkeypatch.setattr(socket, "create_connection", _boom, raising=True)

    out = audit_mod.run_providers_survival_audit(tmp_path, mode="fast")

    assert calls["socket"] == 0
    assert out["live_probes_invoked"] is False
    assert out["read_only"] is True
