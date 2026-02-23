from __future__ import annotations

import types
from pathlib import Path

import agency.ajax_core as ajax_core_mod
from agency.ajax_core import AjaxConfig, AjaxCore, MissionState


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_core(tmp_path: Path) -> AjaxCore:
    core = AjaxCore.__new__(AjaxCore)
    state_dir = tmp_path / "artifacts" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    core.config = AjaxConfig(root_dir=tmp_path, state_dir=state_dir)
    core.provider_configs = {"providers": {}}
    core.ledger = None
    core.log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    return core


def test_preflight_blocks_when_required_policy_missing(tmp_path: Path, monkeypatch) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="test", mode="auto", envelope=None, mission_id="m-policy-missing")

    monkeypatch.setattr(ajax_core_mod, "run_anchor_preflight", lambda **_k: {"ok": True}, raising=False)
    monkeypatch.setattr(
        ajax_core_mod,
        "build_starting_xi",
        lambda **_k: {"missing_players": [], "path": str(tmp_path / "artifacts" / "state" / "sx.json")},
        raising=False,
    )

    sx, res = core._preflight_starting_xi(mission)

    assert sx is None
    assert res is not None
    assert res.error == "BLOCKED_BY_POLICY_CONFIG_MISSING"
    assert isinstance(res.detail, dict)
    assert res.detail.get("terminal_status") == "BLOCKED"


def test_preflight_blocks_on_anchor_mismatch(tmp_path: Path, monkeypatch) -> None:
    core = _build_core(tmp_path)
    mission = MissionState(intention="test", mode="auto", envelope=None, mission_id="m-anchor")

    _write(tmp_path / "config" / "provider_policy.yaml", "schema: ajax.provider_policy.v1\nproviders: {}\n")
    _write(tmp_path / "config" / "provider_failure_policy.yaml", "planning:\n  max_attempts: 2\n")

    anchor_receipt = tmp_path / "artifacts" / "receipts" / "anchor_preflight_test.json"
    _write(anchor_receipt, "{}\n")
    monkeypatch.setattr(
        ajax_core_mod,
        "run_anchor_preflight",
        lambda **_k: {
            "ok": False,
            "status": "BLOCKED",
            "reason": "port_session_mismatch",
            "receipt_path": str(anchor_receipt),
        },
        raising=False,
    )
    monkeypatch.setattr(
        ajax_core_mod,
        "build_starting_xi",
        lambda **_k: {"missing_players": [], "path": str(tmp_path / "artifacts" / "state" / "sx.json")},
        raising=False,
    )

    sx, res = core._preflight_starting_xi(mission)

    assert sx is None
    assert res is not None
    assert res.error == "BLOCKED_BY_ANCHOR_MISMATCH"
    assert isinstance(res.detail, dict)
    assert res.detail.get("terminal_status") == "BLOCKED"
