from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agency.reporting.daily_brief_concierge import generate_daily_brief


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _assert_evidence_refs_shape(items: List[Dict[str, Any]]) -> None:
    for item in items:
        assert isinstance(item.get("kind"), str)
        assert item.get("kind")
        assert isinstance(item.get("path"), str)
        assert item.get("path")


def test_daily_brief_generates_artifacts_and_schema_with_leann(tmp_path: Path) -> None:
    # Arrange
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)

    gaps_dir = root / "artifacts" / "capability_gaps" / "open"
    _write_json(
        gaps_dir / "gap_a.json",
        {"gap_id": "gap_a", "summary": "provider fallback mismatch", "created_utc": "2026-02-27T10:00:00Z"},
    )
    _write_json(
        gaps_dir / "gap_b.json",
        {"gap_id": "gap_b", "summary": "bridge timeout unclear", "created_utc": "2026-02-27T09:00:00Z"},
    )
    _write_json(
        gaps_dir / "gap_c.json",
        {"gap_id": "gap_c", "summary": "doctor strict policy drift", "created_utc": "2026-02-27T08:00:00Z"},
    )

    index_base = root / ".leann" / "indexes" / "antigravity_skills_safe" / "documents.leann"
    meta_path = Path(f"{index_base}.meta.json")
    passages_path = Path(f"{index_base}.passages.jsonl")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"version": "1.0"}, ensure_ascii=False), encoding="utf-8")
    passages_path.write_text('{"id":"x1","text":"sample"}\n', encoding="utf-8")

    calls: Dict[str, int] = {"leann": 0}

    def _run_named_audit(name: str, root_dir: Path, run_id: Any = None, last: int = 1) -> Dict[str, Any]:
        _ = (root_dir, run_id, last)
        if name == "providers":
            return {
                "ok": True,
                "summary": {"critical": 0, "error": 0, "warn": 1},
                "findings": [{"id": "W1"}],
                "artifact_path": str(root / "artifacts" / "audit" / "providers" / "providers_survival_audit.json"),
                "receipt_path": str(root / "artifacts" / "receipts" / "providers_survival_audit_x.json"),
            }
        if name == "eki":
            return {
                "ok": True,
                "summary": {"critical": 0, "error": 0, "warn": 0},
                "findings": [],
                "artifact_path": str(root / "artifacts" / "audit" / "eki" / "eki_audit.json"),
                "receipt_path": str(root / "artifacts" / "receipts" / "eki_audit_x.json"),
            }
        raise AssertionError(f"unexpected audit: {name}")

    def _list_registered_audits() -> Dict[str, Any]:
        return {"providers": object(), "eki": object()}

    def _run_soak_check(root_dir: Path, rail: str, window_min: int, now_ts: float) -> Dict[str, Any]:
        _ = (root_dir, rail, window_min, now_ts)
        return {
            "ok": True,
            "outcome_code": "SOAK_PASS",
            "summary_paragraph": "all good",
            "receipt_path": str(root / "artifacts" / "receipts" / "soak_check_x.json"),
            "report_path": str(root / "artifacts" / "reports" / "soak_status.md"),
        }

    def _query_leann(collection: str, query: str, top_k: int = 5, fallback_grep: bool = True) -> List[Dict[str, Any]]:
        _ = (collection, top_k, fallback_grep)
        calls["leann"] += 1
        assert "gap_a" in query
        return [
            {
                "id": "hit-1",
                "text": "Use testing-qa and fix-review skills.",
                "score": 1.23,
                "source_mode": "vector",
                "metadata": {"path": "skills/testing-qa/SKILL.md"},
            }
        ]

    # Act
    result = generate_daily_brief(
        root=root,
        now_ts=1700000000.0,
        run_named_audit_fn=_run_named_audit,
        list_registered_audits_fn=_list_registered_audits,
        run_soak_check_fn=_run_soak_check,
        query_leann_fn=_query_leann,
    )

    # Assert
    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert calls["leann"] == 1

    json_path = Path(str(result["json_path"]))
    md_path = Path(str(result["md_path"]))
    receipt_path = Path(str(result["receipt_path"]))
    assert json_path.exists()
    assert md_path.exists()
    assert receipt_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "ajax.concierge.daily_brief.v1"
    assert payload["status"] == "ok"
    assert payload["read_only"] is True
    assert payload["mode"] == "proposal_only"
    assert payload["leann"]["status"] == "ok"
    assert len(payload["gaps"]["recent"]) == 3
    assert payload["audits"]["eki"]["status"] == "ok"
    assert payload["proposals"]
    assert all(p.get("mode") == "proposal_only" for p in payload["proposals"])
    _assert_evidence_refs_shape(payload["evidence_refs"])
    for claim in payload["claims"]:
        _assert_evidence_refs_shape(claim.get("evidence_refs") or [])


def test_daily_brief_fail_closed_when_leann_index_missing(tmp_path: Path) -> None:
    # Arrange
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "artifacts" / "capability_gaps" / "open" / "gap_only.json",
        {"gap_id": "gap_only", "summary": "missing index", "created_utc": "2026-02-27T12:00:00Z"},
    )

    def _run_named_audit(name: str, root_dir: Path, run_id: Any = None, last: int = 1) -> Dict[str, Any]:
        _ = (name, root_dir, run_id, last)
        return {"ok": True, "summary": {}, "findings": []}

    def _list_registered_audits() -> Dict[str, Any]:
        return {"providers": object()}

    def _run_soak_check(root_dir: Path, rail: str, window_min: int, now_ts: float) -> Dict[str, Any]:
        _ = (root_dir, rail, window_min, now_ts)
        return {"ok": True, "outcome_code": "SOAK_PASS", "summary_paragraph": "ok"}

    def _query_leann(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        raise AssertionError("query should not run when index is missing")

    # Act
    result = generate_daily_brief(
        root=root,
        now_ts=1700000000.0,
        run_named_audit_fn=_run_named_audit,
        list_registered_audits_fn=_list_registered_audits,
        run_soak_check_fn=_run_soak_check,
        query_leann_fn=_query_leann,
    )

    # Assert
    assert result["ok"] is False
    assert result["status"] == "capability_missing"
    assert result["exit_code"] == 2

    payload = json.loads(Path(str(result["json_path"])).read_text(encoding="utf-8"))
    assert payload["status"] == "capability_missing"
    assert payload["terminal"] == "GAP_LOGGED"
    assert payload["leann"]["status"] == "capability_missing"
    assert "verification_commands" in payload
    assert payload["read_only"] is True
