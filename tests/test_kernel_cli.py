import subprocess
import sys
import json
from pathlib import Path


def test_ajaxctl_help_exits_zero():
    proc = subprocess.run([sys.executable, "bin/ajaxctl", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0


def test_ajaxctl_whereami_root_override():
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "whereami", "--root", "."], capture_output=True, text=True
    )
    assert proc.returncode == 0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "repo_root:" in out


def test_ajaxctl_microfilm_check_generates_report():
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "microfilm", "check", "--root", "."],
        capture_output=True,
        text=True,
    )
    # FAIL (2) is valid in fail-closed environments; PASS (0) is also valid.
    assert proc.returncode in {0, 2}
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "Microfilm Compliance v1:" in out


def test_ajaxctl_soak_check_generates_report():
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "soak", "check", "--root", "."],
        capture_output=True,
        text=True,
    )
    assert proc.returncode in {0, 2}
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "Soak Gate v1:" in out


def test_ajaxctl_doctor_provider_alias_no_argparse_error():
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "doctor", "provider"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode in {0, 1, 2}
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "Alias detectado:" in out


def test_ajaxctl_lab_init_creates_minimum_files(tmp_path: Path):
    fake_root = tmp_path / "ajax-kernel"
    (fake_root / "agency").mkdir(parents=True, exist_ok=True)
    (fake_root / "bin").mkdir(parents=True, exist_ok=True)
    (fake_root / "config").mkdir(parents=True, exist_ok=True)
    (fake_root / "scripts" / "ops").mkdir(parents=True, exist_ok=True)
    (fake_root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (fake_root / "bin" / "ajaxctl").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "lab", "init", "--root", str(fake_root)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload.get("ok") is True
    assert (fake_root / "config" / "lab_org_manifest.yaml").exists()
    assert (fake_root / "config" / "explore_policy.yaml").exists()
    assert (fake_root / "scripts" / "ops" / "get_human_signal.ps1").exists()
    display_map = json.loads((fake_root / "config" / "display_map.json").read_text(encoding="utf-8"))
    assert isinstance(display_map.get("display_targets"), dict)
    assert display_map["display_targets"].get("lab") is not None


def test_ajaxctl_lab_init_accepts_parent_root_with_ajax_kernel_child(tmp_path: Path):
    legacy_root = tmp_path / "AJAX_HOME"
    fake_root = legacy_root / "ajax-kernel"
    (fake_root / "agency").mkdir(parents=True, exist_ok=True)
    (fake_root / "bin").mkdir(parents=True, exist_ok=True)
    (fake_root / "config").mkdir(parents=True, exist_ok=True)
    (fake_root / "scripts" / "ops").mkdir(parents=True, exist_ok=True)
    (fake_root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (fake_root / "bin" / "ajaxctl").write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "lab", "init", "--root", str(legacy_root)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload.get("ok") is True
    assert payload.get("root") == str(fake_root.resolve())
