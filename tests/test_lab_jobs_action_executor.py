from __future__ import annotations

import json
from pathlib import Path

from agency.lab_worker import LabWorker


def _base_result(summary: str) -> dict:
    return {
        "outcome": "PASS",
        "efe_pass": True,
        "failure_codes": [],
        "evidence_refs": [],
        "next_action": "noop",
        "summary": summary,
    }


def test_lab_job_action_executor_default_dispatch(monkeypatch, tmp_path: Path) -> None:
    worker = LabWorker(tmp_path, worker_id="worker-test", idle_sleep_s=0)
    calls: list[str] = []

    def _fake_providers_probe(job: dict, path: Path) -> dict:  # noqa: ANN001
        calls.append("providers_probe")
        return _base_result("providers_probe default")

    monkeypatch.setattr(worker, "_execute_providers_probe", _fake_providers_probe, raising=True)
    result = worker._execute_job({"job_kind": "providers_probe", "job_id": "j1"}, tmp_path / "job.json")

    assert calls == ["providers_probe"]
    assert result["outcome"] == "PASS"
    assert result["action_executor"] == "builtin.providers_probe.v1"
    assert result["action_executor_source"] == "builtin_default"
    assert result["action_executor_handler"] == "_execute_providers_probe"


def test_lab_job_action_executor_override_via_registry_config(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "config" / "lab_action_executor_registry.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "schema": "ajax.lab_action_executor_registry.v1",
        "job_kinds": {
            "providers_probe": {
                "action_executor": "test.providers_probe.alt.v1",
                "handler": "_execute_capabilities_refresh",
            }
        },
    }
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worker = LabWorker(tmp_path, worker_id="worker-test", idle_sleep_s=0)
    calls: list[str] = []

    def _fake_providers_probe(job: dict, path: Path) -> dict:  # noqa: ANN001
        calls.append("providers_probe")
        return _base_result("providers_probe")

    def _fake_caps(job: dict, path: Path) -> dict:  # noqa: ANN001
        calls.append("capabilities_refresh")
        return _base_result("caps override")

    monkeypatch.setattr(worker, "_execute_providers_probe", _fake_providers_probe, raising=True)
    monkeypatch.setattr(worker, "_execute_capabilities_refresh", _fake_caps, raising=True)

    result = worker._execute_job({"job_kind": "providers_probe", "job_id": "j1"}, tmp_path / "job.json")

    assert calls == ["capabilities_refresh"]
    assert result["outcome"] == "PASS"
    assert result["action_executor"] == "test.providers_probe.alt.v1"
    assert result["action_executor_source"] == "config"
    assert result["action_executor_handler"] == "_execute_capabilities_refresh"


def test_lab_job_action_executor_invalid_registry_fails_closed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config" / "lab_action_executor_registry.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("{not-json", encoding="utf-8")

    worker = LabWorker(tmp_path, worker_id="worker-test", idle_sleep_s=0)
    result = worker._execute_job({"job_kind": "providers_probe", "job_id": "j1"}, tmp_path / "job.json")

    assert result["outcome"] == "FAIL"
    assert result["efe_pass"] is False
    assert "lab_action_executor_invalid" in (result.get("failure_codes") or [])
    assert str(cfg_path) in (result.get("evidence_refs") or [])
    assert result["next_action"] == "review_lab_action_executor_registry"

