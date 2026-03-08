from __future__ import annotations

import agency.desktop_operation_specs as specs


def test_materialize_expected_efe_for_open_app() -> None:
    efe = specs.materialize_expected_efe_desktop(
        "open_app",
        {"process": "notepad.exe", "title_contains": "notepad"},
    )
    assert efe["visual_markers_any"] == ["notepad.exe"]
    assert efe["active_window_title_contains"] == "notepad"


def test_materialize_expected_efe_for_focus_window() -> None:
    efe = specs.materialize_expected_efe_desktop(
        "focus_window",
        {"process": "notepad.exe", "title_contains": "notes"},
    )
    assert efe["visual_markers_any"] == ["notepad.exe"]
    assert efe["active_window_title_contains"] == "notes"


def test_materialize_expected_efe_for_type_text() -> None:
    efe = specs.materialize_expected_efe_desktop(
        "type_text",
        {"process": "notepad.exe", "title_contains": "notepad", "text": "AJAX LAB DEMO", "focus_target": "editor"},
    )
    assert efe["visible_text_any"] == ["AJAX LAB DEMO"]
    assert efe["focus_target_contains"] == "editor"


def test_materialize_expected_efe_for_close_window_safe() -> None:
    efe = specs.materialize_expected_efe_desktop(
        "close_window_safe",
        {"process": "notepad.exe", "state": "not_foreground"},
    )
    assert efe["dialogs_absent"] is True
    assert efe["visual_markers_absent"] == ["notepad.exe"]


def test_materialize_expected_efe_for_dismiss_dialog() -> None:
    efe = specs.materialize_expected_efe_desktop(
        "dismiss_dialog",
        {"dialog_name": "save changes"},
    )
    assert efe["dialogs_absent"] is True


def test_build_verify_input_shape() -> None:
    verify_input = specs.build_verify_input(
        "open_app",
        {"action_type": "launch_test_app", "process": "notepad.exe", "title_contains": "notepad"},
        before_state={"screenshot_path": "before.png"},
        after_state={
            "screenshot_path": "after.png",
            "runtime_metadata": {"active_window_title": "Notepad", "visual_markers": ["notepad.exe"]},
            "post_arbiter": {"post_action_verdict": "pass", "mismatches": []},
        },
    )
    assert verify_input["operation_class"] == "open_app"
    assert verify_input["verify_spec_ref"].startswith("verify.")
    assert verify_input["screenshot_after"] == "after.png"
    assert verify_input["verification_contract_version"] == "desktop_v1"


def test_evaluate_desktop_efe_pass() -> None:
    result = specs.evaluate_desktop_efe(
        operation_class="open_app",
        context={"action_type": "launch_test_app", "process": "notepad.exe", "title_contains": "notepad"},
        before_state={"screenshot_path": "before.png"},
        after_state={
            "screenshot_path": "after.png",
            "runtime_metadata": {"active_window_title": "Notepad", "visual_markers": ["notepad.exe"]},
            "post_arbiter": {"post_action_verdict": "pass", "mismatches": []},
        },
    )
    verification_result = result["verification_result"]
    assert verification_result["verdict"] == "pass"
    assert verification_result["mismatches"] == []
    assert verification_result["verification_contract_version"] == "desktop_v1"


def test_evaluate_desktop_efe_fail() -> None:
    result = specs.evaluate_desktop_efe(
        operation_class="type_text",
        context={"action_type": "type_text", "process": "notepad.exe", "title_contains": "notepad", "text": "AJAX"},
        before_state={"screenshot_path": "before.png"},
        after_state={
            "screenshot_path": "after.png",
            "runtime_metadata": {"active_window_title": "Notepad", "visual_markers": ["notepad.exe"]},
            "post_arbiter": {"post_action_verdict": "pass", "mismatches": []},
        },
    )
    verification_result = result["verification_result"]
    assert verification_result["verdict"] == "fail"
    assert verification_result["mismatches"][0]["field"] == "visible_text_any"


def test_evaluate_desktop_efe_uncertain() -> None:
    result = specs.evaluate_desktop_efe(
        operation_class="open_app",
        context={"action_type": "launch_test_app", "process": "notepad.exe", "title_contains": "notepad"},
        before_state={"screenshot_path": "before.png"},
        after_state={
            "runtime_metadata": {"active_window_title": "Notepad", "visual_markers": ["notepad.exe"]},
            "post_arbiter": {"post_action_verdict": "uncertain", "mismatches": []},
        },
    )
    verification_result = result["verification_result"]
    assert verification_result["verdict"] == "uncertain"
    assert verification_result["reason_code"] in {"verify_evidence_incomplete", "verify_uncertain"}
