from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import agency.desktop_operator_lab as operator_mod
from agency.receipt_validator import validate_receipt


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


def _gate_ok(tmp_path: Path) -> dict:
    receipt = tmp_path / "artifacts" / "receipts" / "anchor_preflight_fake.json"
    _write_json(receipt, {"schema": "ajax.anchor_preflight.v1", "ok": True})
    return {
        "ok": True,
        "reason_code": "ok",
        "next_hint": "",
        "session_status": {"ok": True, "reason": "session_valid"},
        "anchor": {"ok": True, "receipt_path": str(receipt)},
        "anchor_receipt_path": str(receipt),
    }


def _gate_session_missing(tmp_path: Path) -> dict:
    gate = _gate_ok(tmp_path)
    gate["ok"] = False
    gate["reason_code"] = "expected_session_missing"
    gate["next_hint"] = "Run lab session init"
    gate["session_status"] = {"ok": False, "reason": "expected_session_missing"}
    return gate


def _gate_anchor_missing(tmp_path: Path) -> dict:
    gate = _gate_ok(tmp_path)
    gate["ok"] = False
    gate["reason_code"] = "lab_anchor_missing"
    gate["next_hint"] = "Run doctor anchor"
    gate["anchor"] = {"ok": False, "mismatches": [{"code": "lab_anchor_missing"}]}
    return gate


def _capture_snapshot_factory(tmp_path: Path):
    calls = {"n": 0}

    def _capture(*, root_dir, job_id, mission_id, active_window, driver_url, context):  # noqa: ANN001
        calls["n"] += 1
        name = f"shot_{calls['n']}_{context}"
        png = Path(root_dir) / "artifacts" / "lab" / "observability" / "screenshots" / f"{name}.png"
        sidecar = png.with_suffix(".json")
        _write_png(png)
        _write_json(
            sidecar,
            {
                "active_window_title": "Notepad",
                "focus_target": "editor",
                "ui_affordances": ["editor_surface"],
                "visual_markers": ["notepad.exe"],
            },
        )
        return {"png_path": str(png), "json_path": str(sidecar)}

    return _capture


def _fake_scout(tmp_path: Path):
    def _run(root_dir, *, rail, screenshot_path, objective, strategy=None, metadata=None, context=None):  # noqa: ANN001
        path = Path(root_dir) / "artifacts" / "desktop" / "scout_fake.json"
        payload = {
            "schema": "ajax.desktop.scout.v1",
            "role": "desktop_scout",
            "rail": rail,
            "ok": True,
            "visual_risks": ["low_risk_fixture"],
            "candidate_paths": ["inspect:editor_surface"],
            "artifact_path": str(path),
        }
        _write_json(path, payload)
        return payload

    return _run


def _fake_arbiter_pass(tmp_path: Path):
    def _run(root_dir, *, rail, mode, screenshot_path, objective, strategy=None, expected_efe_desktop=None, metadata=None):  # noqa: ANN001
        suffix = "pre" if mode == "pre" else "post"
        path = Path(root_dir) / "artifacts" / "desktop" / f"arbiter_{suffix}_fake.json"
        payload = {
            "schema": "ajax.desktop.arbiter.v1",
            "role": "desktop_arbiter",
            "rail": rail,
            "mode": mode,
            "ok": True,
            "strategy_ok": True,
            "mismatches": [],
            "visual_risks": [],
            "post_action_verdict": "pass" if mode == "post" else "uncertain",
            "artifact_path": str(path),
        }
        _write_json(path, payload)
        return payload

    return _run


def _fake_arbiter_pre_fail(tmp_path: Path):
    def _run(root_dir, *, rail, mode, screenshot_path, objective, strategy=None, expected_efe_desktop=None, metadata=None):  # noqa: ANN001
        path = Path(root_dir) / "artifacts" / "desktop" / "arbiter_pre_fail.json"
        payload = {
            "schema": "ajax.desktop.arbiter.v1",
            "role": "desktop_arbiter",
            "rail": rail,
            "mode": mode,
            "ok": True,
            "strategy_ok": False,
            "mismatches": ["strategy_contains_destructive_or_secretive_terms"],
            "visual_risks": ["dialog_present:fixture"],
            "post_action_verdict": "uncertain",
            "artifact_path": str(path),
        }
        _write_json(path, payload)
        return payload

    return _run


class _FakeDriver:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.calls: list[tuple[str, dict]] = []

    def get_active_window(self) -> dict:
        return {"title": "Notepad", "process": "notepad.exe"}

    def find_text(self, text: str) -> dict:
        return {"ok": True, "matches": [{"text": text}]}

    def app_launch(self, process: str | None = None, path: str | None = None, args: list[str] | None = None) -> dict:
        self.calls.append(("app_launch", {"process": process, "path": path, "args": args}))
        return {"ok": True, "process": process or path}

    def window_focus(self, process: str | None = None, title_contains: str | None = None) -> dict:
        self.calls.append(("window_focus", {"process": process, "title_contains": title_contains}))
        return {"ok": True}

    def keyboard_type(self, text: str, submit: bool = False) -> dict:
        self.calls.append(("keyboard_type", {"text": text, "submit": submit}))
        return {"ok": True, "text": text}

    def keyboard_hotkey(self, *keys: str, ignore_safety: bool = False) -> dict:
        self.calls.append(("keyboard_hotkey", {"keys": list(keys), "ignore_safety": ignore_safety}))
        return {"ok": True, "keys": list(keys)}

    def mouse_click(self, x: int | None = None, y: int | None = None, button: str = "left") -> None:
        self.calls.append(("mouse_click", {"x": x, "y": y, "button": button}))

    def find_control(self, *, name: str | None = None, control_type: str | None = None) -> dict:
        return {"rect": {"x": 10, "y": 20, "width": 30, "height": 10}, "name": name, "control_type": control_type}


def _patch_operator_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(operator_mod, "build_desktop_lab_gate", lambda root_dir, rail: _gate_ok(Path(root_dir)), raising=True)
    monkeypatch.setattr(operator_mod, "capture_lab_snapshot", _capture_snapshot_factory(tmp_path), raising=True)
    monkeypatch.setattr(operator_mod, "run_desktop_scout", _fake_scout(tmp_path), raising=True)
    monkeypatch.setattr(operator_mod, "run_desktop_arbiter", _fake_arbiter_pass(tmp_path), raising=True)
    monkeypatch.setattr(operator_mod, "WindowsDriverClient", _FakeDriver, raising=True)
    monkeypatch.setattr(operator_mod, "close_editor_safe", lambda driver, action="dont_save", file_path=None, timeout_s=6.0, dialog_wait_s=1.5: {"ok": True, "method": "dont_save"}, raising=True)


def test_desktop_operator_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "desktop", "operate", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--action-type" in proc.stdout


def test_operator_rejects_non_lab_rail(tmp_path: Path) -> None:
    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="prod",
        action_type="launch_test_app",
        objective="Should fail outside LAB",
        expected_efe_desktop={"active_window_title_contains": "Notepad"},
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    assert payload["result"] == "fail_closed"
    assert payload["reason_code"] == "lab_only_command"


def test_operator_fails_closed_without_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operator_mod, "build_desktop_lab_gate", lambda root_dir, rail: _gate_session_missing(Path(root_dir)), raising=True)
    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="launch_test_app",
        objective="Need LAB session",
        expected_efe_desktop={"active_window_title_contains": "Notepad"},
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    assert payload["result"] == "fail_closed"
    assert payload["reason_code"] == "expected_session_missing"


def test_operator_fails_closed_without_anchor(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operator_mod, "build_desktop_lab_gate", lambda root_dir, rail: _gate_anchor_missing(Path(root_dir)), raising=True)
    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="launch_test_app",
        objective="Need LAB anchor",
        expected_efe_desktop={"active_window_title_contains": "Notepad"},
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    assert payload["result"] == "fail_closed"
    assert payload["reason_code"] == "lab_anchor_missing"


def test_operator_fails_closed_when_pre_verdict_not_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operator_mod, "build_desktop_lab_gate", lambda root_dir, rail: _gate_ok(Path(root_dir)), raising=True)
    monkeypatch.setattr(operator_mod, "capture_lab_snapshot", _capture_snapshot_factory(tmp_path), raising=True)
    monkeypatch.setattr(operator_mod, "run_desktop_scout", _fake_scout(tmp_path), raising=True)
    monkeypatch.setattr(operator_mod, "run_desktop_arbiter", _fake_arbiter_pre_fail(tmp_path), raising=True)
    monkeypatch.setattr(operator_mod, "WindowsDriverClient", _FakeDriver, raising=True)

    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="launch_test_app",
        objective="Pre-check must block",
        expected_efe_desktop={"active_window_title_contains": "Notepad"},
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    assert payload["result"] == "fail_closed"
    assert payload["reason_code"] == "pre_verdict_not_pass"


def test_operator_allows_only_allowlisted_actions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operator_mod, "build_desktop_lab_gate", lambda root_dir, rail: _gate_ok(Path(root_dir)), raising=True)
    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="delete_everything",
        objective="Should be blocked",
        expected_efe_desktop={"active_window_title_contains": "Notepad"},
    )
    assert payload["result"] == "fail_closed"
    assert payload["reason_code"] == "action_not_allowlisted"


def test_operator_writes_receipt(tmp_path: Path, monkeypatch) -> None:
    _patch_operator_success(monkeypatch, tmp_path)
    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="launch_test_app",
        objective="Launch test app",
        expected_efe_desktop={"active_window_title_contains": "Notepad", "visual_markers_any": ["notepad.exe"]},
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    assert Path(str(payload["receipt_path"])).exists()


def test_operator_receipt_schema_validates(tmp_path: Path, monkeypatch) -> None:
    _patch_operator_success(monkeypatch, tmp_path)
    receipt_schema = Path(__file__).resolve().parents[1] / "schemas" / "receipts" / "ajax.desktop.role_receipt.v1.schema.json"
    target_schema = tmp_path / "schemas" / "receipts" / receipt_schema.name
    target_schema.parent.mkdir(parents=True, exist_ok=True)
    target_schema.write_text(receipt_schema.read_text(encoding="utf-8"), encoding="utf-8")

    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="launch_test_app",
        objective="Launch test app",
        expected_efe_desktop={"active_window_title_contains": "Notepad", "visual_markers_any": ["notepad.exe"]},
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )

    report = validate_receipt(tmp_path, Path(str(payload["receipt_path"])))
    assert report["ok"] is True


def test_operator_writes_operation_artifact(tmp_path: Path, monkeypatch) -> None:
    _patch_operator_success(monkeypatch, tmp_path)
    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="launch_test_app",
        objective="Launch test app",
        expected_efe_desktop={"active_window_title_contains": "Notepad", "visual_markers_any": ["notepad.exe"]},
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    artifact = Path(str(payload["artifact_path"]))
    assert artifact.exists()
    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written["role"] == "desktop_operator_lab"


def test_operator_includes_expected_efe_desktop(tmp_path: Path, monkeypatch) -> None:
    _patch_operator_success(monkeypatch, tmp_path)
    expected = {"active_window_title_contains": "Notepad", "visual_markers_any": ["notepad.exe"]}
    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="launch_test_app",
        objective="Launch test app",
        expected_efe_desktop=expected,
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    assert payload["prepare"]["expected_efe_desktop"] == expected


def test_operator_includes_undo_info(tmp_path: Path, monkeypatch) -> None:
    _patch_operator_success(monkeypatch, tmp_path)
    payload = operator_mod.run_desktop_operate(
        tmp_path,
        rail="lab",
        action_type="launch_test_app",
        objective="Launch test app",
        expected_efe_desktop={"active_window_title_contains": "Notepad", "visual_markers_any": ["notepad.exe"]},
        app="notepad.exe",
        target={"process": "notepad.exe"},
    )
    assert "possible" in payload["undo_info"]
    assert isinstance(payload["undo_info"]["suggested_steps"], list)


def test_operate_demo_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "desktop", "operate-demo", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--rail" in proc.stdout


def test_operate_demo_writes_audit_artifact(tmp_path: Path, monkeypatch) -> None:
    calls = {"n": 0}

    def _fake_operate(root_dir, *, rail, action_type, objective, expected_efe_desktop, target=None, text=None, hotkeys=None, app=None, screenshot_path=None):  # noqa: ANN001
        calls["n"] += 1
        artifact = Path(root_dir) / "artifacts" / "desktop" / f"operation_demo_{calls['n']}.json"
        _write_json(artifact, {"role": "desktop_operator_lab", "result": "ok"})
        return {
            "role": "desktop_operator_lab",
            "result": "ok",
            "artifact_path": str(artifact),
            "reason_code": "ok",
        }

    monkeypatch.setattr(operator_mod, "run_desktop_operate", _fake_operate, raising=True)
    payload = operator_mod.run_desktop_operate_demo(tmp_path, rail="lab")
    assert Path(str(payload["artifact_path"])).exists()
    assert payload["schema"] == operator_mod.OPERATION_DEMO_SCHEMA
    assert len(payload["steps"]) == 2


def test_no_constitution_files_touched_guard() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        drive = repo_root.drive.rstrip(":").lower()
        wsl_root = f"/mnt/{drive}/{repo_root.as_posix().split(':', 1)[1].lstrip('/')}"
        proc = subprocess.run(
            ["wsl", "bash", "-lc", f"cd '{wsl_root}' && git diff --name-only"],
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode == 0
    touched = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    forbidden = [item for item in touched if item.startswith("PSEUDOCODE_MAP/")]
    assert forbidden == []


def test_no_microfilm_or_agents_touch_guard() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        drive = repo_root.drive.rstrip(":").lower()
        wsl_root = f"/mnt/{drive}/{repo_root.as_posix().split(':', 1)[1].lstrip('/')}"
        proc = subprocess.run(
            ["wsl", "bash", "-lc", f"cd '{wsl_root}' && git diff --name-only"],
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode == 0
    touched = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    forbidden = [item for item in touched if item in {"AGENTS.md", "MICROFILM.md"}]
    assert forbidden == []
