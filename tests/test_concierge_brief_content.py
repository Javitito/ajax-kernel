from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agency.concierge_brief import generate_daily_brief


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_daily_brief_content_schema_and_top3_with_mocks(tmp_path: Path) -> None:
    root = tmp_path
    now_ts = 1_700_000_000.0

    _write_json(
        root / "artifacts" / "receipts" / "r1.json",
        {
            "ts": now_ts - 10,
            "failure_codes": ["gap_auth", "gap_timeout"],
            "outcome_code": "gap_timeout",
        },
    )
    _write_json(root / "artifacts" / "receipts" / "r2.json", {"ts": now_ts - 20, "code": "gap_auth"})
    _write_json(
        root / "artifacts" / "receipts" / "r3.json",
        {
            "ts": now_ts - 30,
            "findings": [{"code": "gap_timeout"}, {"code": "gap_docs"}],
        },
    )

    index_base = root / ".leann" / "indexes" / "antigravity_skills_safe" / "documents.leann"
    _write_json(Path(f"{index_base}.meta.json"), {"version": "1.0"})
    Path(f"{index_base}.passages.jsonl").parent.mkdir(parents=True, exist_ok=True)
    Path(f"{index_base}.passages.jsonl").write_text('{"id":"1","text":"x"}\n', encoding="utf-8")

    def _run_named_audit(name: str, root_dir: Path, run_id: Any = None, last: int = 1) -> Dict[str, Any]:
        _ = (root_dir, run_id, last)
        if name == "providers":
            return {
                "ok": True,
                "summary": {"critical": 0, "error": 0, "warn": 1},
                "artifact_path": str(root / "artifacts" / "audit" / "providers" / "providers_survival_audit.json"),
                "receipt_path": str(root / "artifacts" / "receipts" / "providers_survival_audit_x.json"),
            }
        if name == "eki":
            return {
                "ok": True,
                "summary": {"critical": 0, "error": 0, "warn": 0},
                "artifact_path": str(root / "artifacts" / "audit" / "eki" / "eki_audit.json"),
                "receipt_path": str(root / "artifacts" / "receipts" / "eki_audit_x.json"),
            }
        raise AssertionError(f"unexpected audit name: {name}")

    def _list_registered_audits() -> Dict[str, Any]:
        return {"providers": object(), "eki": object()}

    def _run_soak_check(root_dir: Path, rail: str, window_min: int, now_ts: float) -> Dict[str, Any]:
        _ = (root_dir, rail, window_min, now_ts)
        return {
            "ok": True,
            "outcome_code": "SOAK_PASS",
            "signals": {"alive": {"ok": True}, "effective": {"ok": True}, "safe": {"ok": True}},
            "receipt_path": str(root / "artifacts" / "receipts" / "soak_check_x.json"),
            "report_path": str(root / "artifacts" / "reports" / "soak_status.md"),
        }

    def _query_leann(collection: str, query: str, top_k: int = 5, fallback_grep: bool = True) -> List[Dict[str, Any]]:
        _ = (collection, top_k, fallback_grep)
        return [
            {
                "id": "h1",
                "text": "---\nname: testing-qa\ndescription: \"Testing workflow\"\n---\n",
                "score": 1.0,
                "metadata": {},
            },
            {
                "id": "h2",
                "text": "---\nname: fix-review\ndescription: \"Verify fixes\"\n---\n",
                "score": 0.9,
                "metadata": {},
            },
            {
                "id": "h3",
                "text": "---\nname: api-documentation\ndescription: \"API docs\"\n---\n",
                "score": 0.8,
                "metadata": {},
            },
        ]

    result = generate_daily_brief(
        root,
        now_ts=now_ts,
        run_named_audit_fn=_run_named_audit,
        list_registered_audits_fn=_list_registered_audits,
        run_soak_check_fn=_run_soak_check,
        query_leann_fn=_query_leann,
    )

    assert result["generated"] is True
    assert result["status"] == "ok"

    payload = json.loads(Path(str(result["brief_json_path"])).read_text(encoding="utf-8"))
    assert payload["schema"] == "ajax.concierge.daily_brief.v1"
    assert payload["read_only"] is True
    assert payload["block_a"]["providers_counts"]["warn"] == 1
    assert payload["block_a"]["gates_counts"]["pass"] == 3
    assert payload["block_b"]["top_gap_codes"][0]["gap_code"] == "gap_timeout"
    assert payload["block_b"]["top_gap_codes"][1]["gap_code"] == "gap_auth"
    assert payload["block_b"]["top_gap_codes"][2]["gap_code"] == "gap_docs"
    assert len(payload["block_c"]["top_skills"]) == 3
    assert payload["block_c"]["top_skills"][0]["name"] == "testing-qa"
    assert payload["block_c"]["status"] == "ok"
