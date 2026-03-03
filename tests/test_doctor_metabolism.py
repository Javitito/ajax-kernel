from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agency.metabolism_doctor import run_doctor_metabolism


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_doctor_metabolism_reports_counts(tmp_path: Path) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(
        tmp_path / "artifacts" / "capability_gaps" / "g1.json",
        {
            "gap_id": "g1_missing_efe_final_m1",
            "created_at": now,
            "reason": "missing_efe_final",
            "efe_candidate_status": "generated",
        },
    )
    _write_json(
        tmp_path / "artifacts" / "capability_gaps" / "g2.json",
        {
            "gap_id": "g2_crystallize_failed_m2",
            "created_at": now,
            "reason": "crystallize_failed",
            "efe_candidate_status": "unsupported",
        },
    )
    _write_json(
        tmp_path / "artifacts" / "provider_ledger" / "latest.json",
        {
            "rows": [
                {"provider": "cloud", "reason": "quota_exhausted"},
                {"provider": "lmstudio", "reason": None},
            ]
        },
    )
    _write_json(
        tmp_path / "artifacts" / "receipts" / "router_ladder_decision_a.json",
        {"created_at": now, "ok": True, "local_fallback_used": True},
    )
    waiting = tmp_path / "artifacts" / "waiting_for_user" / "w1.json"
    _write_json(waiting, {"x": 1})

    payload = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)

    assert payload["gaps"]["missing_efe_final"] == 1
    assert payload["gaps"]["crystallize_failed"] == 1
    assert payload["efe_candidates"]["generated"] >= 1
    assert payload["efe_candidates"]["unsupported"] >= 1
    assert payload["provider"]["last_429_count"] == 1
    assert payload["provider"]["ladder_local_fallback"] == 1
    assert payload["waiting_backlog"]["count"] == 1


def test_doctor_metabolism_handles_missing_dirs(tmp_path: Path) -> None:
    payload = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)

    assert payload["gaps"]["missing_efe_final"] == 0
    assert payload["gaps"]["crystallize_failed"] == 0
    assert payload["waiting_backlog"]["count"] == 0
    assert isinstance(payload["next_hint"], list)


def test_doctor_metabolism_suggests_commands(tmp_path: Path) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(
        tmp_path / "artifacts" / "capability_gaps" / "g1.json",
        {"gap_id": "g1", "created_at": now, "reason": "missing_efe_final"},
    )
    _write_json(tmp_path / "artifacts" / "waiting_for_user" / "w.json", {"x": 1})
    _write_json(
        tmp_path / "artifacts" / "provider_ledger" / "latest.json",
        {"rows": [{"provider": "cloud", "reason": "429_tpm"}]},
    )

    payload = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)

    hints = payload["next_hint"]
    assert any("verify efe apply-candidate" in h for h in hints)
    assert any("ops friction gc --dry-run" in h for h in hints)
    assert any("doctor providers" in h for h in hints)


def test_doctor_metabolism_no_side_effects(tmp_path: Path) -> None:
    _write_json(tmp_path / "artifacts" / "capability_gaps" / "g.json", {"reason": "missing_efe_final"})
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    _ = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)

    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert before == after


def test_doctor_metabolism_exit_codes(tmp_path: Path, monkeypatch) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(
        tmp_path / "artifacts" / "capability_gaps" / "g1.json",
        {"gap_id": "g1", "created_at": now, "reason": "missing_efe_final"},
    )

    monkeypatch.setenv("AJAX_DOCTOR_METABOLISM_CRITICAL_GAPS", "1")
    critical = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)
    assert critical["critical"] is True
    assert critical["exit_code"] == 1

    monkeypatch.setenv("AJAX_DOCTOR_METABOLISM_CRITICAL_GAPS", "999")
    normal = run_doctor_metabolism(root_dir=tmp_path, since_min=180.0)
    assert normal["critical"] is False
    assert normal["exit_code"] == 0
