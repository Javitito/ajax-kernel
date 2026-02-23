from __future__ import annotations

import json
from pathlib import Path

import agency.lab_worker as lab_worker_mod
import agency.provider_breathing as breathing_mod
from agency.lab_worker import LabWorker


def test_providers_probe_refreshes_providers_status_timestamp(
    monkeypatch, tmp_path: Path
) -> None:
    status_path = tmp_path / "artifacts" / "health" / "providers_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    old_doc = {
        "schema": "ajax.providers_status.v1",
        "updated_ts": 1000.0,
        "updated_utc": "1970-01-01T00:16:40Z",
        "providers": {"stub": {"auth_state": "UNKNOWN"}},
    }
    status_path.write_text(
        json.dumps(old_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    class _FakeBreathingLoop:
        def __init__(self, *, root_dir: Path, provider_configs: dict) -> None:
            self.root_dir = Path(root_dir)

        def run_once(self, *, roles=None):  # noqa: ANN001
            return {
                "schema": "ajax.providers_status.v1",
                "updated_at": 2000.0,
                "updated_ts": 1000.0,
                "updated_utc": "1970-01-01T00:33:20Z",
                "providers": {
                    "stub": {
                        "auth_state": "MISSING",
                        "breathing": {"status": "DOWN", "contract_status": "DOWN"},
                    }
                },
            }

    class _FakeLedger:
        def __init__(self, *, root_dir: Path) -> None:
            self.root_dir = Path(root_dir)

        def refresh(self, timeout_s: float = 2.0) -> dict:
            out = self.root_dir / "artifacts" / "provider_ledger" / "latest.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("{}", encoding="utf-8")
            return {"path": str(out), "updated_utc": "1970-01-01T00:33:20Z"}

    monkeypatch.setattr(
        breathing_mod,
        "_load_provider_configs",
        lambda root_dir: {"providers": {"stub": {"roles": ["brain"]}}},
        raising=True,
    )
    monkeypatch.setattr(
        breathing_mod,
        "ProviderBreathingLoop",
        _FakeBreathingLoop,
        raising=True,
    )
    monkeypatch.setattr(lab_worker_mod, "ProviderLedger", _FakeLedger, raising=True)

    worker = LabWorker(tmp_path, worker_id="worker-test", idle_sleep_s=0)
    result = worker._execute_providers_probe(
        {"job_kind": "providers_probe", "job_id": "j1", "mission_id": "lab_org"},
        tmp_path / "job.json",
    )

    assert result.get("outcome") == "PASS"
    assert str(status_path) in (result.get("evidence_refs") or [])
    refreshed = json.loads(status_path.read_text(encoding="utf-8"))
    assert refreshed.get("updated_utc") != old_doc["updated_utc"]
    assert float(refreshed.get("updated_ts") or 0.0) > float(old_doc["updated_ts"])
    assert float(refreshed.get("updated_ts") or 0.0) == 2000.0
