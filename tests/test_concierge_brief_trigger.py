from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import agency.lab_org as lab_org_mod
from agency.concierge_brief import maybe_trigger_human_detected_brief
from agency.lab_control import LabStateStore
from agency.lab_org import lab_org_tick


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _setup_root(tmp_path: Path) -> Path:
    root = tmp_path / "ajax-kernel"
    root.mkdir(parents=True, exist_ok=True)
    store = LabStateStore(root)
    store.resume_lab_org("test_start", metadata={"source": "test"})
    manifest = root / "config" / "lab_org_manifest_test.json"
    _write_json(manifest, {"schema": "ajax.lab_org_manifest.v1", "micro_challenges": []})
    return root


def test_human_detected_event_generates_once_then_cooldown_blocks_second(monkeypatch, tmp_path: Path) -> None:
    root = _setup_root(tmp_path)
    generated_counter = {"n": 0}

    def _fake_generator(root_dir: Path, *, now_ts: float, **_kwargs: Any) -> Dict[str, Any]:
        generated_counter["n"] += 1
        ts = str(int(now_ts))
        brief_json = root_dir / "artifacts" / "concierge" / f"daily_brief_{ts}.json"
        brief_md = root_dir / "artifacts" / "concierge" / f"daily_brief_{ts}.md"
        receipt = root_dir / "artifacts" / "receipts" / f"daily_brief_{ts}.json"
        _write_json(brief_json, {"schema": "ajax.concierge.daily_brief.v1", "status": "ok"})
        brief_md.parent.mkdir(parents=True, exist_ok=True)
        brief_md.write_text("# brief\n", encoding="utf-8")
        _write_json(receipt, {"schema": "ajax.concierge.daily_brief.receipt.v1", "status": "ok"})
        return {
            "status": "ok",
            "generated": True,
            "brief_json_path": str(brief_json),
            "brief_md_path": str(brief_md),
            "receipt_path": str(receipt),
        }

    def _brief_hook(root_dir: Path, *, now_ts: float, **_kwargs: Any) -> Dict[str, Any]:
        return maybe_trigger_human_detected_brief(
            root_dir,
            now_ts=now_ts,
            cooldown_s=6 * 3600,
            generator=_fake_generator,
        )

    monkeypatch.setattr(
        lab_org_mod,
        "evaluate_explore_state",
        lambda root_dir, policy, prev_state=None, now_ts=None: {
            "schema": "ajax.explore_state.v1",
            "state": "HUMAN_DETECTED",
            "trigger": "AWAY->HUMAN_DETECTED",
            "human_active": True,
            "human_active_reason": "test",
        },
        raising=True,
    )
    monkeypatch.setattr(lab_org_mod, "maybe_trigger_human_detected_brief", _brief_hook, raising=True)

    now_values = iter([1_700_000_000.0, 1_700_000_100.0])
    monkeypatch.setattr(lab_org_mod.time, "time", lambda: next(now_values), raising=True)

    manifest = root / "config" / "lab_org_manifest_test.json"
    r1 = lab_org_tick(root, manifest_path=manifest)
    r2 = lab_org_tick(root, manifest_path=manifest)

    assert (r1.get("daily_brief") or {}).get("generated") is True
    assert (r2.get("daily_brief") or {}).get("generated") is False
    assert (r2.get("daily_brief") or {}).get("status") == "cooldown_skipped"
    assert generated_counter["n"] == 1

    briefs = sorted((root / "artifacts" / "concierge").glob("daily_brief_[0-9]*.json"))
    assert len(briefs) == 1
