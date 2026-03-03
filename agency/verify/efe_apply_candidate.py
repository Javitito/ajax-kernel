from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

APPLY_SCHEMA = "ajax.verify.efe_apply_candidate.v0"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def apply_efe_candidate_from_gap(*, gap_path: Path, out_path: Path) -> Dict[str, Any]:
    gap_raw = json.loads(gap_path.read_text(encoding="utf-8"))
    if not isinstance(gap_raw, dict):
        raise ValueError("--gap must point to a JSON object")

    candidate_path_raw = gap_raw.get("efe_candidate_path")
    if not isinstance(candidate_path_raw, str) or not candidate_path_raw.strip():
        raise ValueError("gap_missing_efe_candidate_path")

    candidate_path = Path(candidate_path_raw).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = (gap_path.parent / candidate_path).resolve()
    if not candidate_path.exists():
        raise ValueError("efe_candidate_not_found")

    candidate_raw = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(candidate_raw, dict):
        raise ValueError("efe_candidate_invalid_json")

    expected_state = candidate_raw.get("expected_state")
    if not isinstance(expected_state, dict):
        raise ValueError("efe_candidate_missing_expected_state")

    final_payload = {
        "schema": APPLY_SCHEMA,
        "created_at": _utc_now(),
        "source_gap_path": str(gap_path),
        "source_candidate_path": str(candidate_path),
        "expected_state": expected_state,
        "human_hint": "Review/edit expected_state before execution; this helper performs no action execution.",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "gap_path": str(gap_path),
        "source_candidate_path": str(candidate_path),
        "out_path": str(out_path),
        "human_hint": final_payload["human_hint"],
    }
