from __future__ import annotations

import json

import agency.desktop_operation_specs as specs


def test_operation_class_registry_exists() -> None:
    assert set(specs.OPERATION_CLASSES) >= {
        "open_app",
        "focus_window",
        "type_text",
        "close_window_safe",
        "dismiss_dialog",
    }


def test_expected_efe_templates_are_serializable() -> None:
    payloads = [
        specs.build_expected_efe_template("window_open_and_focused", process="notepad.exe", title_contains="notepad"),
        specs.build_expected_efe_template("text_present_in_window", process="notepad.exe", title_contains="notepad", text="AJAX"),
        specs.build_expected_efe_template("app_started", process="notepad.exe"),
        specs.build_expected_efe_template("dialog_visible", dialog_name="save changes"),
        specs.build_expected_efe_template("dialog_closed"),
        specs.build_expected_efe_template("expected_process_state", process="notepad.exe", state="not_foreground"),
    ]
    for payload in payloads:
        assert isinstance(payload, dict)
        json.dumps(payload)


def test_verify_specs_exist_for_core_classes() -> None:
    for name in ("open_app", "focus_window", "type_text", "close_window_safe", "dismiss_dialog"):
        assert name in specs.VERIFY_SPECS
        assert specs.VERIFY_SPECS[name]["name"].startswith("verify.")


def test_undo_specs_exist_for_core_classes() -> None:
    for name in ("open_app", "focus_window", "type_text", "close_window_safe", "dismiss_dialog"):
        assert name in specs.UNDO_SPECS
        assert specs.UNDO_SPECS[name]["name"].startswith("undo.")


def test_open_app_spec_shape() -> None:
    spec = specs.OPERATION_CLASSES["open_app"]
    assert spec["allowed_action_types"] == ["launch_test_app"]
    assert "window_open_and_focused" in spec["expected_templates"]
    assert spec["verify_spec"] == "verify.window_open_and_focused.v0"


def test_focus_window_spec_shape() -> None:
    spec = specs.OPERATION_CLASSES["focus_window"]
    assert spec["allowed_action_types"] == ["focus_window"]
    assert spec["expected_templates"] == ["window_open_and_focused"]


def test_type_text_spec_shape() -> None:
    spec = specs.OPERATION_CLASSES["type_text"]
    assert spec["allowed_action_types"] == ["type_text"]
    assert spec["expected_templates"] == ["text_present_in_window"]


def test_close_window_safe_spec_shape() -> None:
    spec = specs.OPERATION_CLASSES["close_window_safe"]
    assert spec["allowed_action_types"] == ["close_test_app"]
    assert "expected_process_state" in spec["expected_templates"]


def test_dismiss_dialog_spec_shape() -> None:
    spec = specs.OPERATION_CLASSES["dismiss_dialog"]
    assert spec["allowed_action_types"] == ["safe_hotkey"]
    assert spec["expected_templates"] == ["dialog_closed"]


def test_resolve_open_app_contract_merges_user_expected() -> None:
    contract = specs.resolve_operation_contract(
        action_type="launch_test_app",
        target={"process": "notepad.exe"},
        app="notepad.exe",
        expected_efe_desktop={"active_window_title_contains": "custom-notepad"},
    )
    assert contract["operation_class"] == "open_app"
    assert contract["expected_efe_desktop"]["active_window_title_contains"] == "custom-notepad"
    assert contract["expected_efe_desktop"]["visual_markers_any"] == ["notepad.exe"]


def test_resolve_type_text_contract_includes_text_probe() -> None:
    contract = specs.resolve_operation_contract(
        action_type="type_text",
        target={"process": "notepad.exe", "title_contains": "notepad"},
        text="AJAX LAB DEMO",
        expected_efe_desktop={},
    )
    assert contract["operation_class"] == "type_text"
    assert contract["expected_efe_desktop"]["visible_text_any"] == ["AJAX LAB DEMO"]


def test_resolve_dismiss_dialog_contract_from_safe_hotkey() -> None:
    contract = specs.resolve_operation_contract(
        action_type="safe_hotkey",
        hotkeys=["esc"],
        target={"dialog_name": "save changes"},
        expected_efe_desktop={},
    )
    assert contract["operation_class"] == "dismiss_dialog"
    assert contract["expected_efe_desktop"]["dialogs_absent"] is True
