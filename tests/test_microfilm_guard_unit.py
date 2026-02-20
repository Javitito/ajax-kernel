from __future__ import annotations

from agency.microfilm_guard import (
    enforce_evidence_tiers,
    enforce_lab_prod_separation,
    enforce_ssc,
    enforce_undo_for_reversible,
    enforce_verify_before_done,
)


def test_enforce_ssc_blocks_when_actuation_and_snapshot_missing() -> None:
    out = enforce_ssc({"actuation": True, "snapshot0": None})
    assert out["ok"] is False
    assert out["code"] == "BLOCKED_SSC_MISSING"


def test_enforce_ssc_passes_when_no_actuation() -> None:
    out = enforce_ssc({"actuation": False, "snapshot0": None})
    assert out["ok"] is True


def test_enforce_ssc_passes_with_snapshot() -> None:
    out = enforce_ssc({"actuation": True, "snapshot0": {"ts": 123}})
    assert out["ok"] is True


def test_verify_before_done_blocks_without_verification() -> None:
    out = enforce_verify_before_done({"status": "DONE"}, {"ok": False})
    assert out["ok"] is False
    assert out["code"] == "BLOCKED_VERIFY_REQUIRED"


def test_verify_before_done_passes_with_verification() -> None:
    out = enforce_verify_before_done({"status": "DONE"}, {"ok": True})
    assert out["ok"] is True


def test_lab_prod_separation_blocks_lab_on_primary_display() -> None:
    out = enforce_lab_prod_separation({"rail": "lab", "display_target": "primary", "human_active": False})
    assert out["ok"] is False
    assert out["code"] == "BLOCKED_RAIL_MISMATCH"


def test_lab_prod_separation_blocks_lab_when_human_active() -> None:
    out = enforce_lab_prod_separation({"rail": "lab", "display_target": "dummy", "human_active": True})
    assert out["ok"] is False


def test_lab_prod_separation_blocks_prod_on_dummy() -> None:
    out = enforce_lab_prod_separation({"rail": "prod", "display_target": "dummy", "human_active": False})
    assert out["ok"] is False


def test_lab_prod_separation_passes_lab_dummy_human_inactive() -> None:
    out = enforce_lab_prod_separation({"rail": "lab", "display_target": "dummy", "human_active": False})
    assert out["ok"] is True


def test_evidence_tier_promotes_only_real_online() -> None:
    out = enforce_evidence_tiers({"driver_online": True}, {"ok": True, "verification_mode": "real"})
    assert out["promote_trust"] is True
    assert out["evidence_tier"] == "real_online"


def test_evidence_tier_rejects_synthetic_promotion() -> None:
    out = enforce_evidence_tiers({"driver_online": True}, {"ok": True, "verification_mode": "synthetic"})
    assert out["promote_trust"] is False


def test_evidence_tier_rejects_offline_promotion() -> None:
    out = enforce_evidence_tiers({"driver_online": False}, {"ok": True, "verification_mode": "real"})
    assert out["promote_trust"] is False


def test_undo_guard_blocks_reversible_without_undo() -> None:
    out = enforce_undo_for_reversible({"steps": [{"action": "app.launch"}], "metadata": {}})
    assert out["ok"] is False
    assert out["code"] == "BLOCKED_UNDO_REQUIRED"


def test_undo_guard_passes_reversible_with_undo() -> None:
    out = enforce_undo_for_reversible(
        {
            "steps": [{"action": "app.launch"}],
            "metadata": {"tx_paths": {"undo_plan_path": "artifacts/undo.json"}},
        }
    )
    assert out["ok"] is True


def test_undo_guard_passes_non_reversible() -> None:
    out = enforce_undo_for_reversible({"steps": [{"action": "query.status"}], "metadata": {}})
    assert out["ok"] is True
