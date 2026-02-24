from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def _force_away(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        lab_org_mod,
        "evaluate_explore_state",
        lambda root, policy, prev_state=None, now_ts=None: {
            "schema": "ajax.explore_state.v1",
            "state": "AWAY",
            "trigger": None,
            "human_active": False,
            "human_active_reason": "test_away",
        },
        raising=True,
    )


def test_providers_audit_enqueues_only_when_away_and_hunger_budget_allow(monkeypatch, tmp_path: Path) -> None:
    root = _setup_running_root(tmp_path)
    _force_away(monkeypatch)
    monkeypatch.setattr(
        lab_org_mod,
        "get_current_hunger",
        lambda root_dir: SimpleNamespace(hunger=0.8, decision=SimpleNamespace(explore_budget=0.4)),
        raising=True,
    )
    manifest = root / "config" / "lab_org_manifest_test.json"
    _write_manifest(
        manifest,
        [
            {
                "id": "providers_audit",
                "job_kind": "providers_audit",
                "enabled": True,
                "ui_intrusive": False,
                "cadence_s": 30,
                "budget_s": 20,
                "min_hunger": 0.4,
                "min_explore_budget": 0.1,
                "tags": ["safe", "providers", "audit"],
            }
        ],
    )

    receipt = lab_org_tick(root, manifest_path=manifest)

    assert receipt.get("state") == "AWAY"
    assert receipt.get("enqueued") is True
    assert receipt.get("job_kind") == "providers_audit"
    hunger_gate = receipt.get("hunger_gate") or {}
    assert hunger_gate.get("applied") is True
    snap = hunger_gate.get("snapshot") or {}
    assert float(snap.get("hunger") or 0.0) == 0.8
    assert float(snap.get("explore_budget") or 0.0) == 0.4

    # 1 job/ventana: second tick inside cadence_s should not enqueue again.
    receipt2 = lab_org_tick(root, manifest_path=manifest)
    assert receipt2.get("enqueued") is False
    assert receipt2.get("job_kind") is None
    assert receipt2.get("skipped_reason") == "no_due_jobs"


def test_providers_audit_hunger_gate_blocks_when_budget_or_hunger_low(monkeypatch, tmp_path: Path) -> None:
    root = _setup_running_root(tmp_path)
    _force_away(monkeypatch)
    monkeypatch.setattr(
        lab_org_mod,
        "get_current_hunger",
        lambda root_dir: SimpleNamespace(hunger=0.2, decision=SimpleNamespace(explore_budget=0.05)),
        raising=True,
    )
    manifest = root / "config" / "lab_org_manifest_test.json"
    _write_manifest(
        manifest,
        [
            {
                "id": "providers_audit",
                "job_kind": "providers_audit",
                "enabled": True,
                "ui_intrusive": False,
                "cadence_s": 0,
                "budget_s": 20,
                "min_hunger": 0.4,
                "min_explore_budget": 0.1,
                "tags": ["safe", "providers", "audit"],
            }
        ],
    )

    receipt = lab_org_tick(root, manifest_path=manifest)

    assert receipt.get("enqueued") is False
    assert receipt.get("job_kind") is None
    assert receipt.get("skipped_reason") == "hunger_budget_gate"
    hunger_gate = receipt.get("hunger_gate") or {}
    skipped = hunger_gate.get("skipped") or []
    assert skipped
    assert skipped[0]["job_kind"] == "providers_audit"
    assert "hunger_below_threshold" in skipped[0]["reason"]
    assert "explore_budget_below_threshold" in skipped[0]["reason"]

