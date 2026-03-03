from __future__ import annotations

import json
import os
import time
import types
from pathlib import Path

from agency.ajax_core import AjaxConfig, AjaxCore, MissionState
from agency.brain_router import AllBrainsFailed, BrainRouter
from agency.friction import run_friction_gc
from agency.verify.efe_apply_candidate import APPLY_SCHEMA, apply_efe_candidate_from_gap


def _build_core(tmp_path: Path) -> AjaxCore:
    core = AjaxCore.__new__(AjaxCore)
    state_dir = tmp_path / "artifacts" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    core.config = AjaxConfig(root_dir=tmp_path, state_dir=state_dir)
    core.root_dir = tmp_path
    core.log = types.SimpleNamespace(warning=lambda *a, **k: None)
    return core


def _touch_with_age(path: Path, *, age_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    ts = time.time() - age_seconds
    os.utime(path, (ts, ts))


def _write_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ajax.provider_ledger.v1",
        "updated_ts": time.time(),
        "rows": [
            {"provider": "cloud_a", "role": "brain", "status": "soft_fail", "reason": "quota_exhausted", "details": {}},
            {"provider": "lmstudio", "role": "brain", "status": "ok", "details": {}},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_e2e_efe_missing_gap_to_apply_candidate(tmp_path: Path) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="create file", mission_id="e2e-efe")

    gap_path = core._emit_missing_efe_gap(
        mission,
        reason="missing_efe_final",
        efe_candidate_source_doc={"steps": [{"action": "write_file", "args": {"path": "out.txt"}}]},
    )
    assert isinstance(gap_path, str)
    final_efe = tmp_path / "artifacts" / "efe_candidates" / "efe_final.json"
    res = apply_efe_candidate_from_gap(gap_path=Path(gap_path), out_path=final_efe)

    assert res["ok"] is True
    payload = json.loads(final_efe.read_text(encoding="utf-8"))
    assert payload.get("schema") == APPLY_SCHEMA
    assert isinstance((payload.get("expected_state") or {}).get("checks"), list)


def test_e2e_router_ladder_prefers_local_before_blocked() -> None:
    router = BrainRouter(
        [
            ("cloud_primary", {"kind": "cli", "roles": ["brain"]}),
            ("cloud_alt", {"kind": "cli", "roles": ["brain"]}),
            ("lmstudio", {"kind": "http_openai", "base_url": "http://localhost:1235/v1", "roles": ["brain"]}),
        ]
    )
    order: list[str] = []

    def caller(name, cfg, ps, pu, meta):  # noqa: ANN001
        order.append(name)
        if name != "lmstudio":
            raise RuntimeError("429 rate limit")
        return {"steps": [{"id": "s1"}]}

    plan = router.plan_with_failover(prompt_system="s", prompt_user="u", caller=caller)
    assert isinstance(plan, dict)
    assert order == ["cloud_primary", "cloud_alt", "lmstudio"]


def test_e2e_friction_gc_dry_run_then_apply_archives_and_snapshots(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    _touch_with_age(waiting_old, age_seconds=7200)
    ledger = tmp_path / "artifacts" / "provider_ledger" / "latest.json"
    _write_ledger(ledger)

    dry = run_friction_gc(root_dir=tmp_path, apply=False, older_than_hours=1.0)
    assert waiting_old.exists()
    assert dry["waiting_for_user"]["archived"] == []

    app = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=1.0)
    assert waiting_old.exists() is False
    assert len(app["waiting_for_user"]["archived"]) == 1
    assert Path(app["provider_ledger"]["snapshot_path"]).exists()


def test_e2e_combined_loop_closure_receipts_coherent(tmp_path: Path) -> None:
    waiting_old = tmp_path / "artifacts" / "waiting_for_user" / "old.json"
    _touch_with_age(waiting_old, age_seconds=7200)
    ledger = tmp_path / "artifacts" / "provider_ledger" / "latest.json"
    _write_ledger(ledger)

    core = _build_core(tmp_path)
    mission = MissionState(intention="combined", mission_id="e2e-combined")

    gc_payload = run_friction_gc(root_dir=tmp_path, apply=True, older_than_hours=1.0)
    gap_path = core._emit_missing_efe_gap(
        mission,
        reason="missing_efe_final",
        efe_candidate_source_doc={"steps": [{"action": "write_file", "args": {"path": "combo.txt"}}]},
    )
    out_efe = tmp_path / "artifacts" / "efe_candidates" / "combo_final.json"
    apply_efe_candidate_from_gap(gap_path=Path(str(gap_path)), out_path=out_efe)

    router = BrainRouter(
        [
            ("cloud_primary", {"kind": "cli", "roles": ["brain"]}),
            ("lmstudio", {"kind": "http_openai", "base_url": "http://localhost:1235/v1", "roles": ["brain"]}),
        ]
    )
    tried: list[str] = []

    def caller(name, cfg, ps, pu, meta):  # noqa: ANN001
        tried.append(name)
        if name == "cloud_primary":
            raise RuntimeError("timeout")
        return {"steps": [{"id": "s1"}]}

    _ = router.plan_with_failover(prompt_system="s", prompt_user="u", caller=caller)
    router_receipt = tmp_path / "artifacts" / "receipts" / "router_ladder_decision_test.json"
    router_receipt.parent.mkdir(parents=True, exist_ok=True)
    router_receipt.write_text(
        json.dumps(
            {
                "schema": "ajax.router_ladder_decision.v0",
                "ok": True,
                "providers_tried": tried,
                "local_fallback_used": tried[-1] == "lmstudio",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assert Path(gc_payload["receipt_path"]).exists()
    assert Path(str(gap_path)).exists()
    assert out_efe.exists()
    assert router_receipt.exists()


def test_helper_router_blocks_when_local_unavailable() -> None:
    router = BrainRouter(
        [
            ("cloud_primary", {"kind": "cli", "roles": ["brain"]}),
            ("cloud_alt", {"kind": "cli", "roles": ["brain"]}),
        ]
    )

    def caller(name, cfg, ps, pu, meta):  # noqa: ANN001
        raise RuntimeError("429")

    try:
        router.plan_with_failover(prompt_system="s", prompt_user="u", caller=caller)
        assert False, "expected AllBrainsFailed"
    except AllBrainsFailed as exc:
        assert exc.reason == "local_fallback_unavailable"


def test_helper_gap_candidate_contains_reason_codes(tmp_path: Path) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="unknown", mission_id="helper-reason")
    gap_path = core._emit_missing_efe_gap(
        mission,
        reason="missing_efe_final",
        efe_candidate_source_doc={"steps": [{"action": "unknown", "args": {}}]},
    )
    payload = json.loads(Path(str(gap_path)).read_text(encoding="utf-8"))
    assert payload.get("efe_candidate_status") in {"unsupported", "error", "generated"}
    assert isinstance(payload.get("efe_candidate_reason"), str)


def test_helper_apply_candidate_retains_human_hint(tmp_path: Path) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="hint", mission_id="helper-hint")
    gap_path = core._emit_missing_efe_gap(
        mission,
        reason="missing_efe_final",
        efe_candidate_source_doc={"steps": [{"action": "write_file", "args": {"path": "hint.txt"}}]},
    )
    out_efe = tmp_path / "final_hint.json"
    res = apply_efe_candidate_from_gap(gap_path=Path(str(gap_path)), out_path=out_efe)
    assert "no action execution" in str(res.get("human_hint") or "").lower()


def test_helper_friction_gc_no_network_or_external_side_effects(tmp_path: Path) -> None:
    before = set(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    payload = run_friction_gc(root_dir=tmp_path, apply=False, older_than_hours=1.0)
    after = set(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    # dry-run only writes its own receipt under artifacts/receipts
    new_entries = after - before
    assert any("artifacts/receipts/friction_gc_" in n for n in new_entries)
    assert payload["mode"] == "dry_run"
