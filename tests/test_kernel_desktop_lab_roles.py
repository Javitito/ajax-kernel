from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import agency.desktop_roles as desktop_roles


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
    screenshot = root / "fixtures" / "desktop_fixture.png"
    _write_png(screenshot)
    _write_json(
        screenshot.with_suffix(".json"),
        {
            "active_window_title": "LAB Fixture Window",
            "focus_target": "search_box",
            "ui_affordances": ["search_box", "results_panel", "confirm_button"],
            "dialogs": [],
            "visual_markers": ["results_panel"],
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


def _fake_anchor_missing(root_dir: Path, rail: str, write_receipt: bool = True) -> dict:
    receipt = root_dir / "artifacts" / "receipts" / "anchor_preflight_fake.json"
    _write_json(receipt, {"schema": "ajax.anchor_preflight.v1", "ok": False, "rail": rail})
    return {
        "schema": "ajax.anchor_preflight.v1",
        "ok": False,
        "status": "BLOCKED",
        "rail": rail,
        "mismatches": [{"code": "lab_anchor_missing", "detail": "fixture blocked"}],
        "warnings": [],
        "receipt_path": str(receipt) if write_receipt else None,
    }


def test_desktop_scout_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "desktop", "scout", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--screenshot" in proc.stdout


def test_desktop_arbiter_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "desktop", "arbiter", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--mode" in proc.stdout


def test_desktop_compiler_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "desktop", "compile", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--scout-artifact" in proc.stdout


def test_scout_fails_closed_without_lab_anchor(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_missing, raising=True)

    payload = desktop_roles.run_desktop_scout(
        tmp_path,
        rail="lab",
        screenshot_path=str(screenshot),
        objective="Inspect fixture",
    )

    assert payload["ok"] is False
    assert payload["reason_code"] == "lab_anchor_missing"


def test_arbiter_fails_closed_without_lab_anchor(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_missing, raising=True)

    payload = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="pre",
        screenshot_path=str(screenshot),
        objective="Validate fixture strategy",
        strategy={"steps": ["review search_box"]},
    )

    assert payload["ok"] is False
    assert payload["reason_code"] == "lab_anchor_missing"


def test_compiler_fails_closed_on_missing_inputs(tmp_path: Path) -> None:
    payload = desktop_roles.run_desktop_compiler(
        tmp_path,
        rail="lab",
        scout_artifact_path=str(tmp_path / "missing_scout.json"),
        arbiter_artifact_paths=[],
    )

    assert payload["ok"] is False
    assert payload["reason_code"] == "missing_artifact_input"


def test_scout_writes_artifact(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)

    payload = desktop_roles.run_desktop_scout(
        tmp_path,
        rail="lab",
        screenshot_path=str(screenshot),
        objective="Inspect fixture desktop",
        strategy={"steps": ["review search_box"]},
    )

    artifact = Path(str(payload["artifact_path"]))
    assert payload["ok"] is True
    assert artifact.exists()
    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written["role"] == "desktop_scout"


def test_arbiter_writes_artifact(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)

    payload = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="pre",
        screenshot_path=str(screenshot),
        objective="Validate fixture strategy",
        strategy={"steps": ["review dialog state before continue"]},
    )

    artifact = Path(str(payload["artifact_path"]))
    assert payload["ok"] is True
    assert artifact.exists()
    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written["role"] == "desktop_arbiter"


def test_compiler_writes_artifact(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)

    scout = desktop_roles.run_desktop_scout(
        tmp_path,
        rail="lab",
        screenshot_path=str(screenshot),
        objective="Inspect fixture desktop",
        strategy={"steps": ["review search_box"]},
    )
    arbiter = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="post",
        screenshot_path=str(screenshot),
        objective="Validate fixture desktop",
        strategy={"steps": ["review search_box"]},
        expected_efe_desktop={"active_window_title_contains": "Fixture", "dialogs_absent": True},
    )

    payload = desktop_roles.run_desktop_compiler(
        tmp_path,
        rail="lab",
        scout_artifact_path=str(scout["artifact_path"]),
        arbiter_artifact_paths=[str(arbiter["artifact_path"])],
        context={"recipe_name": "fixture_recipe"},
    )

    artifact = Path(str(payload["artifact_path"]))
    assert payload["ok"] is True
    assert artifact.exists()
    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written["role"] == "desktop_compiler"
    assert written["verification_contract_version"] == "desktop_v1"
    assert isinstance(written["verification_results"], list)


def test_pre_verdict_shape(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)

    payload = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="pre",
        screenshot_path=str(screenshot),
        objective="Validate pre strategy",
        strategy={"steps": ["review dialog state before continue"]},
    )

    assert payload["mode"] == "pre"
    assert isinstance(payload["strategy_ok"], bool)
    assert payload["verification_result"]["verdict"] in {"pass", "fail", "uncertain"}
    assert isinstance(payload["verification_result"]["mismatches"], list)
    assert isinstance(payload["visual_risks"], list)
    assert isinstance(payload["next_hint"], str)
    assert payload["verify_input"]["verification_contract_version"] == "desktop_v1"


def test_post_verdict_shape(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)

    payload = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="post",
        screenshot_path=str(screenshot),
        objective="Validate post state",
        strategy={"steps": ["review search_box"]},
        expected_efe_desktop={
            "active_window_title_contains": "Fixture",
            "focus_target_contains": "search",
            "affordances_any": ["results_panel"],
            "dialogs_absent": True,
        },
    )

    assert payload["mode"] == "post"
    assert isinstance(payload["strategy_ok"], bool)
    assert payload["verification_result"]["verdict"] in {"pass", "fail", "uncertain"}
    assert isinstance(payload["verification_result"]["mismatches"], list)
    assert isinstance(payload["visual_risks"], list)
    assert isinstance(payload["next_hint"], str)
    assert payload["verify_input"]["verification_contract_version"] == "desktop_v1"


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
    forbidden = [item for item in touched if item.startswith("PSEUDOCODE_MAP/") or item in {"AGENTS.md", "MICROFILM.md"}]
    assert forbidden == []


def test_desktop_demo_writes_audit_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)

    payload = desktop_roles.run_desktop_demo(tmp_path, rail="lab")

    artifact = Path(str(payload["artifact_path"]))
    assert artifact.exists()
    assert payload["schema"] == desktop_roles.DEMO_SCHEMA
    assert isinstance(payload["steps"], list)


def test_compiler_uses_shared_verify_contract(tmp_path: Path, monkeypatch) -> None:
    screenshot = _write_screenshot_fixture(tmp_path)
    monkeypatch.setattr(desktop_roles, "validate_expected_session", _fake_session_ok, raising=True)
    monkeypatch.setattr(desktop_roles, "run_anchor_preflight", _fake_anchor_ok, raising=True)

    scout = desktop_roles.run_desktop_scout(
        tmp_path,
        rail="lab",
        screenshot_path=str(screenshot),
        objective="Inspect fixture desktop",
        strategy={"action_type": "focus_window"},
    )
    arbiter = desktop_roles.run_desktop_arbiter(
        tmp_path,
        rail="lab",
        mode="post",
        screenshot_path=str(screenshot),
        objective="Validate fixture desktop",
        strategy={"action_type": "focus_window", "target": {"title_contains": "fixture", "focus_target": "search_box"}},
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
    )
    assert payload["verification_results"][0]["verification_contract_version"] == "desktop_v1"


def test_verify_demo_registered_or_equivalent() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "desktop", "verify-demo", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--rail" in proc.stdout
