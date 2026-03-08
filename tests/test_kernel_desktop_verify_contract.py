from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import agency.desktop_operator_lab as operator_mod
import agency.desktop_roles as desktop_roles
import agency.desktop_verify_contract as contract


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(payload)


def _write_screenshot_fixture(root: Path) -> Path:
    screenshot = root / "fixtures" / "desktop_verify_fixture.png"
    _write_png(screenshot)
    _write_json(
        screenshot.with_suffix(".json"),
        {
            "active_window_title": "LAB Fixture Window",
            "focus_target": "search_box",
            "ui_affordances": ["search_box", "results_panel", "confirm_button"],
            "dialogs": [],
            "visual_markers": ["results_panel"],
            "visible_text": ["fixture ready"],
        },
    )
    return screenshot


def _fake_session_ok(root_dir: Path, *, required_rail: str = "lab", required_display: str = "dummy") -> dict:
    return {
        "schema": "ajax.lab.session_status.v0",
        "ok": True,
        "reason": "session_valid",
        "path": str(root_dir / "artifacts" / "lab" / "session" / "expected_session.json"),
        "rail": required_rail,
        "display_target": required_display,
        "invalid_reasons": [],
    }


def _fake_anchor_ok(root_dir: Path, rail: str, write_receipt: bool = True) -> dict:
    receipt = root_dir / "artifacts" / "receipts" / "anchor_preflight_fake.json"
    _write_json(receipt, {"schema": "ajax.anchor_preflight.v1", "ok": True, "rail": rail})
    return {
        "schema": "ajax.anchor_preflight.v1",
        "ok": True,
        "status": "READY",
        "rail": rail,
        "mismatches": [],
        "warnings": [],
        "receipt_path": str(receipt) if write_receipt else None,
    }


def test_desktop_verify_input_shape() -> None:
    verify_input = contract.build_desktop_verify_input(
        operation_class="focus_window",
        expected_efe_desktop={"active_window_title_contains": "fixture"},
        before_state={"screenshot_path": "before.png"},
        after_state={"screenshot_path": "after.png"},
        screenshot_before="before.png",
        screenshot_after="after.png",
        arbiter_context={"mode": "post"},
        runtime_metadata={"active_window_title": "LAB Fixture Window"},
        verify_spec={"name": "verify.window_open_and_focused.v0"},
    )
    assert verify_input["operation_class"] == "focus_window"
    assert verify_input["verification_contract_version"] == "desktop_v1"
    assert verify_input["verify_spec_ref"] == "verify.window_open_and_focused.v0"


def test_desktop_verification_result_shape() -> None:
    result = contract.build_desktop_verification_result(
        verdict="pass",
        mismatches=[],
        reason_code="ok",
        next_hint="",
        confidence="high",
    )
    assert result["verdict"] == "pass"
    assert result["mismatches"] == []
    assert result["confidence"] == "high"
    assert result["verification_contract_version"] == "desktop_v1"


def test_desktop_mismatch_shape() -> None:
    mismatch = contract.build_desktop_mismatch(
        "focus_target_contains",
        expected="search",
        observed="dialog",
        severity="high",
        note="Focus landed on the wrong control.",
    )
    assert mismatch["field"] == "focus_target_contains"
    assert mismatch["severity"] == "high"


def test_verify_contract_is_serializable() -> None:
    payload = {
        "verify_input": contract.build_desktop_verify_input(
            operation_class="open_app",
            expected_efe_desktop={"visual_markers_any": ["notepad.exe"]},
            before_state={"screenshot_path": "before.png"},
            after_state={"screenshot_path": "after.png"},
            verify_spec={"name": "verify.window_open_and_focused.v0"},
        ),
        "verification_result": contract.build_desktop_verification_result(
            verdict="uncertain",
            mismatches=[contract.build_desktop_mismatch("visual_markers_any", expected=["notepad.exe"], observed=[])],
            reason_code="verify_uncertain",
            next_hint="Collect stronger evidence.",
            confidence="low",
        ),
    }
    json.dumps(payload)


def test_arbiter_uses_shared_verify_contract(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)
    payload = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="post",
        screenshot_path=str(screenshot),
        objective="Validate shared desktop contract",
        strategy={"action_type": "focus_window", "target": {"title_contains": "fixture window", "focus_target": "search_box"}},
        expected_efe_desktop={
            "active_window_title_contains": "fixture",
            "focus_target_contains": "search",
            "affordances_any": ["results_panel"],
            "dialogs_absent": True,
        },
    )
    assert payload["ok"] is True
    assert payload["verify_input"]["verification_contract_version"] == "desktop_v1"
    assert payload["verification_result"]["verification_contract_version"] == "desktop_v1"


def test_compiler_uses_shared_verify_contract(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)
    scout = desktop_roles.run_desktop_scout(
        tmp_path,
        rail="lab",
        screenshot_path=str(screenshot),
        objective="Inspect shared contract fixture",
        strategy={"action_type": "focus_window"},
    )
    arbiter = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="post",
        screenshot_path=str(screenshot),
        objective="Validate shared contract fixture",
        strategy={"action_type": "focus_window", "target": {"title_contains": "fixture window", "focus_target": "search_box"}},
        expected_efe_desktop={
            "active_window_title_contains": "fixture",
            "focus_target_contains": "search",
            "affordances_any": ["results_panel"],
            "dialogs_absent": True,
        },
    )
    payload = desktop_roles.run_desktop_compiler(
        tmp_path,
        rail="lab",
        scout_artifact_path=str(scout["artifact_path"]),
        arbiter_artifact_paths=[str(arbiter["artifact_path"])],
        context={"recipe_name": "shared_contract_fixture"},
    )
    assert payload["ok"] is True
    assert payload["verification_contract_version"] == "desktop_v1"
    assert payload["verification_results"][0]["verification_contract_version"] == "desktop_v1"


def test_verification_contract_version_present() -> None:
    assert contract.DESKTOP_VERIFICATION_CONTRACT_VERSION == "desktop_v1"


def test_pass_result_shape() -> None:
    result = contract.build_desktop_verification_result(
        verdict="pass",
        mismatches=[],
        reason_code="ok",
        next_hint="",
        confidence="high",
    )
    assert result["verdict"] == "pass"
    assert result["confidence"] == "high"


def test_fail_result_shape() -> None:
    result = contract.build_desktop_verification_result(
        verdict="fail",
        mismatches=[contract.build_desktop_mismatch("dialogs_absent", expected=True, observed=["save changes"])],
        reason_code="verify_fail",
        next_hint="Handle the modal before retrying.",
        confidence="high",
    )
    assert result["verdict"] == "fail"
    assert result["mismatches"][0]["field"] == "dialogs_absent"


def test_uncertain_result_shape() -> None:
    result = contract.build_desktop_verification_result(
        verdict="uncertain",
        mismatches=[],
        reason_code="verify_evidence_incomplete",
        next_hint="Capture a stronger screenshot.",
        confidence="low",
    )
    assert result["verdict"] == "uncertain"
    assert result["reason_code"] == "verify_evidence_incomplete"


def test_receipts_include_shared_contract(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)
    payload = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="post",
        screenshot_path=str(screenshot),
        objective="Validate shared contract receipt",
        strategy={"action_type": "focus_window", "target": {"title_contains": "fixture window", "focus_target": "search_box"}},
        expected_efe_desktop={
            "active_window_title_contains": "fixture",
            "focus_target_contains": "search",
            "affordances_any": ["results_panel"],
            "dialogs_absent": True,
        },
    )
    receipt = json.loads(Path(str(payload["receipt_path"])).read_text(encoding="utf-8"))
    assert receipt["verification_contract_version"] == "desktop_v1"
    assert receipt["verification_result"]["verification_contract_version"] == "desktop_v1"


def test_no_allowlist_expansion_guard() -> None:
    assert operator_mod.ALLOWED_ACTIONS == {
        "click_coordinates",
        "click_named_target",
        "type_text",
        "safe_hotkey",
        "launch_test_app",
        "focus_window",
        "close_test_app",
    }


def test_verify_demo_registered_or_equivalent() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "desktop", "verify-demo", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--rail" in proc.stdout


def test_verify_demo_writes_shared_result(tmp_path: Path) -> None:
    payload = desktop_roles.run_desktop_verify_demo(tmp_path, rail="lab")
    artifact = Path(str(payload["artifact_path"]))
    assert artifact.exists()
    assert payload["verify_input"]["verification_contract_version"] == "desktop_v1"
    assert payload["verification_result"]["verification_contract_version"] == "desktop_v1"
