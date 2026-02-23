from __future__ import annotations

import json
import time
from pathlib import Path

import agency.lab_org as lab_org_mod
from agency.lab_control import LabStateStore
from agency.lab_org import lab_org_tick


def _setup_running_root(tmp_path: Path) -> Path:
    root = tmp_path / "ajax-kernel"
    root.mkdir(parents=True, exist_ok=True)
    store = LabStateStore(root)
    store.resume_lab_org("test_start", metadata={"source": "test"})
    return root


def _write_manifest(path: Path, challenges: list[dict]) -> None:
    payload = {"schema": "ajax.lab_org_manifest.v1", "micro_challenges": challenges}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _force_human_detected(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        lab_org_mod,
        "evaluate_explore_state",
        lambda root, policy, prev_state=None, now_ts=None: {
            "schema": "ajax.explore_state.v1",
            "state": "HUMAN_DETECTED",
            "trigger": None,
            "human_active": True,
            "human_active_reason": "test_human_active",
        },
        raising=True,
    )


def test_human_detected_blocks_non_maintenance_micro_challenges(monkeypatch, tmp_path: Path) -> None:
    root = _setup_running_root(tmp_path)
    _force_human_detected(monkeypatch)
    manifest = root / "config" / "lab_org_manifest_test.json"
    _write_manifest(
        manifest,
        [
            {
                "id": "ui_micro_challenge",
                "job_kind": "micro_challenge_ui",
                "enabled": True,
                "ui_intrusive": False,
                "cadence_s": 1,
                "budget_s": 20,
                "tags": ["explore"],
            }
        ],
    )

    receipt = lab_org_tick(root, manifest_path=manifest)

    assert receipt.get("human_active") is True
    assert receipt.get("mode") == "MAINTENANCE_ONLY"
    assert receipt.get("enqueued") is False
    assert receipt.get("job_kind") is None
    assert receipt.get("selected_job") is None
    assert receipt.get("skipped_reason") == "maintenance_only_no_due"
    assert "providers_probe" in (receipt.get("allowlist_used") or [])


def test_human_detected_allows_due_providers_probe(monkeypatch, tmp_path: Path) -> None:
    root = _setup_running_root(tmp_path)
    _force_human_detected(monkeypatch)
    manifest = root / "config" / "lab_org_manifest_test.json"
    _write_manifest(
        manifest,
        [
            {
                "id": "micro_challenge_ui",
                "job_kind": "micro_challenge_ui",
                "enabled": True,
                "ui_intrusive": False,
                "cadence_s": 0,
                "budget_s": 20,
                "tags": ["explore"],
            },
            {
                "id": "providers_probe",
                "job_kind": "providers_probe",
                "enabled": True,
                "ui_intrusive": False,
                "cadence_s": 0,
                "budget_s": 20,
                "tags": ["safe", "providers"],
            },
        ],
    )

    receipt = lab_org_tick(root, manifest_path=manifest)

    assert receipt.get("human_active") is True
    assert receipt.get("mode") == "MAINTENANCE_ONLY"
    assert receipt.get("enqueued") is True
    assert receipt.get("selected_job") == "providers_probe"
    assert receipt.get("job_kind") == "providers_probe"
    assert receipt.get("skipped_reason") is None


def test_human_detected_nothing_due_returns_maintenance_only_no_due(monkeypatch, tmp_path: Path) -> None:
    root = _setup_running_root(tmp_path)
    _force_human_detected(monkeypatch)
    manifest = root / "config" / "lab_org_manifest_test.json"
    _write_manifest(
        manifest,
        [
            {
                "id": "providers_probe",
                "job_kind": "providers_probe",
                "enabled": True,
                "ui_intrusive": False,
                "cadence_s": 3600,
                "budget_s": 20,
                "tags": ["safe", "providers"],
            }
        ],
    )
    state_path = root / "artifacts" / "lab" / "lab_org_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "updated_utc": "2026-02-21T00:00:00Z",
                "snapshot": {"last_job_ts": {"providers_probe": time.time()}},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = lab_org_tick(root, manifest_path=manifest)

    assert receipt.get("human_active") is True
    assert receipt.get("mode") == "MAINTENANCE_ONLY"
    assert receipt.get("enqueued") is False
    assert receipt.get("skipped_reason") == "maintenance_only_no_due"
    assert receipt.get("reason") == "maintenance_only_no_due"
