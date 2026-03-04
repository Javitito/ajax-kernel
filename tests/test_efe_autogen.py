from __future__ import annotations

import json
from pathlib import Path

from agency.verify.efe_autogen import (
    CANDIDATE_SCHEMA,
    RECEIPT_SCHEMA,
    autogen_efe_candidate,
    autogen_efe_candidate_from_file,
    extract_action_descriptor,
    generate_expected_state,
)


def test_generate_expected_state_fs_includes_observables() -> None:
    expected, explain, unsupported, hint = generate_expected_state(
        {"kind": "fs", "params": {"paths": ["b.txt", "a.txt"], "exists": True}}
    )

    assert unsupported is None
    assert hint is None
    assert expected is not None
    checks = expected.get("checks")
    assert isinstance(checks, list) and len(checks) == 2
    assert checks[0]["path"] == "a.txt"
    assert checks[0]["mtime"]["required"] is True
    assert checks[0]["size"]["required"] is True
    assert checks[0]["sha256"]["required"] is True
    assert any("deterministic" in line.lower() for line in explain)


def test_generate_expected_state_fs_dedupes_paths() -> None:
    expected, *_ = generate_expected_state(
        {
            "kind": "fs",
            "params": {
                "paths": ["x.txt", "x.txt"],
                "file": "y.txt",
                "path": "x.txt",
            },
        }
    )

    assert expected is not None
    files = expected.get("files")
    assert files == [
        {"path": "x.txt", "must_exist": True},
        {"path": "y.txt", "must_exist": True},
    ]


def test_generate_expected_state_process_by_name() -> None:
    expected, _, unsupported, _ = generate_expected_state(
        {"kind": "process", "params": {"name": "lab_worker", "running": True}}
    )

    assert unsupported is None
    assert expected == {"checks": [{"kind": "process", "running": True, "name": "lab_worker"}]}


def test_generate_expected_state_process_by_pid_not_running() -> None:
    expected, _, unsupported, _ = generate_expected_state(
        {"kind": "process", "params": {"pid": 1234, "running": False}}
    )

    assert unsupported is None
    assert expected == {"checks": [{"kind": "process", "running": False, "pid": 1234}]}


def test_generate_expected_state_port_open_normalizes_localhost() -> None:
    expected, _, unsupported, _ = generate_expected_state(
        {"kind": "port", "params": {"host": "localhost", "port": 5012, "open": True}}
    )

    assert unsupported is None
    assert expected == {"checks": [{"kind": "port", "host": "127.0.0.1", "port": 5012, "open": True}]}


def test_generate_expected_state_port_closed() -> None:
    expected, _, unsupported, _ = generate_expected_state(
        {"kind": "port", "params": {"host": "127.0.0.1", "port": 5010, "open": False}}
    )

    assert unsupported is None
    assert expected == {"checks": [{"kind": "port", "host": "127.0.0.1", "port": 5010, "open": False}]}


def test_generate_expected_state_rejects_unknown_kind() -> None:
    expected, _, unsupported, hint = generate_expected_state({"kind": "network", "params": {}})

    assert expected is None
    assert unsupported == "unsupported_action_kind"
    assert "Unsupported descriptor kind" in str(hint)


def test_extract_action_descriptor_from_plan_step() -> None:
    descriptor = extract_action_descriptor(
        {
            "steps": [
                {"id": "s1", "action": "write_file", "args": {"path": "out.txt"}},
            ]
        }
    )

    assert descriptor is not None
    assert descriptor["kind"] == "fs"
    assert descriptor["params"]["paths"] == ["out.txt"]
    assert descriptor["step_id"] == "s1"


def test_extract_action_descriptor_from_root_kind_params() -> None:
    descriptor = extract_action_descriptor(
        {"kind": "process", "params": {"name": "python", "running": True}}
    )

    assert descriptor == {
        "kind": "process",
        "params": {"name": "python", "running": True},
        "source": "root",
    }


def test_efe_autogen_determinism_same_descriptor_same_expected_state() -> None:
    descriptor = {"kind": "fs", "params": {"paths": ["b", "a"], "exists": True}}
    first, *_ = generate_expected_state(descriptor)
    second, *_ = generate_expected_state(descriptor)

    assert first == second


def test_autogen_efe_candidate_writes_candidate_and_receipt(tmp_path: Path) -> None:
    out_path = tmp_path / "efe_candidate.json"
    receipts_dir = tmp_path / "receipts"

    result = autogen_efe_candidate(
        source_doc={"kind": "port", "params": {"port": 5012, "open": True}},
        out_path=out_path,
        source_path=None,
        receipts_dir=receipts_dir,
    )

    assert result["ok"] is True
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == CANDIDATE_SCHEMA
    assert payload["ok"] is True
    receipt = Path(result["receipt_path"])
    assert receipt.exists()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["schema"] == RECEIPT_SCHEMA


def test_autogen_efe_candidate_from_file_handles_unsupported(tmp_path: Path) -> None:
    source_path = tmp_path / "plan.json"
    source_path.write_text(json.dumps({"steps": [{"action": "do_unknown", "args": {}}]}), encoding="utf-8")
    out_path = tmp_path / "efe_candidate.json"

    result = autogen_efe_candidate_from_file(
        source_path=source_path,
        out_path=out_path,
        receipts_dir=tmp_path / "receipts",
    )

    assert result["ok"] is False
    assert result["unsupported_action_kind"] == "unsupported_action_kind"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["unsupported_action_kind"] == "unsupported_action_kind"
