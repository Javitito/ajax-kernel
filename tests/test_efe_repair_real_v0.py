from __future__ import annotations

import json
import socket
from pathlib import Path
import types

import agency.efe_repair as efe_repair_mod
from agency.ajax_core import AjaxConfig, AjaxCore
from agency.efe_repair import repair_plan_if_needed
from agency.expected_state import verify_efe
from agency.receipt_validator import validate_receipt


def _prepare_schema_tree(root: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "schemas" / "receipts"
    dst = root / "schemas" / "receipts"
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*.json"):
        dst.joinpath(path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _receipts_dir(root: Path) -> Path:
    path = root / "artifacts" / "receipts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_core(root: Path) -> AjaxCore:
    core = AjaxCore.__new__(AjaxCore)
    state_dir = root / "artifacts" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    core.config = AjaxConfig(root_dir=root, state_dir=state_dir)
    core.root_dir = root
    core.log = types.SimpleNamespace(warning=lambda *a, **k: None)
    core.actions_catalog = None
    return core


def _base_step(action: str, args: dict) -> dict:
    return {
        "id": "task-1",
        "intent": f"Execute {action}",
        "preconditions": {"expected_state": {}},
        "action": action,
        "args": args,
        "evidence_required": ["driver.active_window"],
        "success_spec": {"expected_state": {}},
        "on_fail": "abort",
    }


def test_repair_plan_template_first_fs_happy_path(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipts_dir = _receipts_dir(tmp_path)
    plan = {
        "goal": "Write artifact",
        "metadata": {"intention": "write artifact", "rail": "lab"},
        "steps": [_base_step("write_file", {"path": "artifacts/output/report.json"})],
    }

    result = repair_plan_if_needed(plan, receipts_dir=str(receipts_dir))

    assert result.success is True
    assert result.terminal == "DONE"
    assert result.repair_path == "template"
    assert result.template_id == "efe.fs_path_materialized.v0"
    step = result.plan["steps"][0]
    expected_state = step["success_spec"]["expected_state"]
    assert expected_state["files"][0]["path"] == "artifacts/output/report.json"
    report = validate_receipt(tmp_path, Path(str(result.receipt_path)))
    assert report.get("ok") is True
    assert report.get("schema_used") == "ajax.receipt.efe_repair.v1.schema.json"


def test_repair_plan_derivation_port_happy_path(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipts_dir = _receipts_dir(tmp_path)
    plan = {
        "goal": "Start localhost service",
        "metadata": {"intention": "start localhost service", "rail": "lab"},
        "steps": [_base_step("start_service", {"host": "127.0.0.1", "port": 5012})],
    }

    result = repair_plan_if_needed(plan, receipts_dir=str(receipts_dir))

    assert result.success is True
    assert result.repair_path == "derivation"
    expected_state = result.plan["steps"][0]["success_spec"]["expected_state"]
    assert expected_state["checks"][0]["kind"] == "port"
    assert expected_state["checks"][0]["port"] == 5012


def test_repair_plan_lab_only_gate_refuses_prod_even_for_template(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipts_dir = _receipts_dir(tmp_path)
    plan = {
        "goal": "Write artifact in prod",
        "metadata": {"intention": "write artifact", "rail": "prod"},
        "steps": [_base_step("write_file", {"path": "artifacts/output/prod.json"})],
    }

    result = repair_plan_if_needed(plan, receipts_dir=str(receipts_dir))

    assert result.success is False
    assert result.terminal == "WAITING_FOR_USER"
    assert result.candidate_path
    report = validate_receipt(tmp_path, Path(str(result.receipt_path)))
    assert report.get("ok") is True
    payload = json.loads(Path(str(result.receipt_path)).read_text(encoding="utf-8"))
    assert "rail_not_lab" in payload["auto_materialization_refused_reasons"]


def test_ajax_core_builds_waiting_plan_from_guarded_refusal(tmp_path: Path) -> None:
    core = _build_core(tmp_path)
    plan = {
        "goal": "Write artifact in prod",
        "metadata": {"intention": "write artifact in prod", "rail": "prod"},
        "steps": [_base_step("write_file", {"path": "artifacts/output/prod.json"})],
    }

    validated, repair_meta = core._validate_brain_plan_with_efe_repair(
        plan,
        intention="write artifact in prod",
        source="brain",
    )

    assert validated is None
    assert repair_meta["terminal"] == "WAITING_FOR_USER"
    wait_plan = core._build_missing_efe_plan(
        intention="write artifact in prod",
        source="brain",
        receipt_path=repair_meta.get("receipt"),
        terminal=repair_meta.get("terminal"),
        waiting_prompt=repair_meta.get("waiting_prompt"),
        boundary=repair_meta.get("boundary"),
        repair_path=repair_meta.get("repair_path"),
        template_id=repair_meta.get("template_id"),
        efe_candidate_path=repair_meta.get("candidate_path"),
        reason=repair_meta.get("reason"),
    )
    assert wait_plan.steps[0]["action"] == "await_user_input"
    assert wait_plan.metadata["efe_repair_terminal"] == "WAITING_FOR_USER"


def test_repair_plan_repo_patch_refuses_and_keeps_boundary(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipts_dir = _receipts_dir(tmp_path)
    plan = {
        "goal": "Patch repo file",
        "metadata": {"intention": "patch repo file", "rail": "lab"},
        "steps": [_base_step("write_file", {"path": "agency/ajax_core.py"})],
    }

    result = repair_plan_if_needed(plan, receipts_dir=str(receipts_dir))

    assert result.success is False
    assert result.terminal == "WAITING_FOR_USER"
    assert result.boundary["repair_path"] == "candidate"
    assert "mission_family_requires_boundary:repo_patch" in result.boundary["refusal_reasons"]


def test_repair_plan_waiting_path_when_candidate_generation_unavailable(tmp_path: Path, monkeypatch) -> None:
    _prepare_schema_tree(tmp_path)
    receipts_dir = _receipts_dir(tmp_path)
    plan = {
        "goal": "Unknown plan",
        "metadata": {"intention": "unknown", "rail": "lab"},
        "steps": [_base_step("mystery_action", {})],
    }

    def _boom(**_kwargs):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr(efe_repair_mod, "autogen_efe_candidate", _boom, raising=True)
    result = repair_plan_if_needed(plan, receipts_dir=str(receipts_dir))

    assert result.success is False
    assert result.terminal == "WAITING_FOR_USER"
    assert result.repair_path == "waiting"
    payload = json.loads(Path(str(result.receipt_path)).read_text(encoding="utf-8"))
    assert payload["repair_path"] == "waiting"


def test_verify_efe_receipt_schema_check_passes(tmp_path: Path) -> None:
    _prepare_schema_tree(tmp_path)
    receipt = tmp_path / "artifacts" / "receipts" / "efe_autogen_ok.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "schema": "ajax.receipt.efe_autogen.v0",
                "version": "v0",
                "created_at": "2026-03-11T20:00:00Z",
                "ok": True,
                "source_path": "plan.json",
                "efe_candidate_path": "efe_candidate.json",
                "descriptor_kind": "fs",
                "unsupported_action_kind": None,
                "hint": None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    ok, delta = verify_efe(
        {
            "files": [{"path": str(receipt), "must_exist": True}],
            "checks": [{"kind": "receipt_schema", "path": str(receipt), "schema": "ajax.receipt.efe_autogen.v0"}],
        },
        root_dir=tmp_path,
        timeout_s=0.1,
        poll_interval_s=0.01,
    )

    assert ok is True
    assert delta is None


def test_verify_efe_structured_output_and_port_checks_pass(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "subcalls" / "scout_demo.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "schema": "ajax.subcall.role_result.v1",
                "role": "scout",
                "provider_selected": "lmstudio",
                "result": "ok",
                "reason_code": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host, port = server.getsockname()
        ok, delta = verify_efe(
            {
                "files": [{"path": str(artifact), "must_exist": True}],
                "checks": [
                    {
                        "kind": "structured_output",
                        "path": str(artifact),
                        "format": "json",
                        "root_type": "object",
                        "required_keys": ["schema", "role", "provider_selected", "result", "reason_code"],
                    },
                    {"kind": "port", "host": host, "port": port, "open": True},
                ],
            },
            root_dir=tmp_path,
            timeout_s=0.1,
            poll_interval_s=0.01,
        )

    assert ok is True
    assert delta is None
