from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agency.verify.efe_apply_candidate import apply_efe_candidate_from_gap


def _write_candidate(path: Path, expected_state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ajax.verify.efe_candidate.v0",
        "expected_state": expected_state,
        "ok": True,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_apply_candidate_outputs_efe_final_file(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    _write_candidate(candidate, {"checks": [{"kind": "fs", "path": "x.txt", "exists": True}]})
    gap = tmp_path / "gap.json"
    gap.write_text(json.dumps({"efe_candidate_path": str(candidate)}), encoding="utf-8")
    out = tmp_path / "efe_final.json"

    result = apply_efe_candidate_from_gap(gap_path=gap, out_path=out)

    assert result["ok"] is True
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload.get("expected_state"), dict)


def test_apply_candidate_rejects_missing_candidate_path(tmp_path: Path) -> None:
    gap = tmp_path / "gap.json"
    gap.write_text(json.dumps({"gap_id": "g1"}), encoding="utf-8")
    out = tmp_path / "efe_final.json"

    try:
        apply_efe_candidate_from_gap(gap_path=gap, out_path=out)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "gap_missing_efe_candidate_path" in str(exc)


def test_apply_candidate_preserves_deterministic_checks(tmp_path: Path) -> None:
    expected_state = {
        "checks": [
            {
                "kind": "fs",
                "path": "x.txt",
                "exists": True,
                "mtime": {"required": True},
                "size": {"required": True},
                "sha256": {"required": True},
            }
        ]
    }
    candidate = tmp_path / "candidate.json"
    _write_candidate(candidate, expected_state)
    gap = tmp_path / "gap.json"
    gap.write_text(json.dumps({"efe_candidate_path": str(candidate)}), encoding="utf-8")
    out = tmp_path / "efe_final.json"

    apply_efe_candidate_from_gap(gap_path=gap, out_path=out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("expected_state") == expected_state


def test_apply_candidate_outputs_human_editable_comment_or_hint(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    _write_candidate(candidate, {"checks": [{"kind": "process", "name": "python", "running": True}]})
    gap = tmp_path / "gap.json"
    gap.write_text(json.dumps({"efe_candidate_path": str(candidate)}), encoding="utf-8")
    out = tmp_path / "efe_final.json"

    proc = subprocess.run(
        [
            sys.executable,
            "bin/ajaxctl",
            "verify",
            "efe",
            "apply-candidate",
            "--gap",
            str(gap),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    hint = str(payload.get("human_hint") or "").lower()
    assert "review" in hint
    assert "no action execution" in hint
