from __future__ import annotations

from pathlib import Path

import agency.topology_doctor as topology_mod


def test_doctor_topology_detects_repo(monkeypatch, tmp_path: Path) -> None:
    ajax_home = tmp_path / "AJAX_HOME"
    kernel = ajax_home / "ajax-kernel"
    kernel.mkdir(parents=True, exist_ok=True)

    def _fake_git_toplevel(path: Path) -> str:
        if "ajax-kernel" in str(path):
            return str(kernel)
        return str(ajax_home)

    monkeypatch.setattr(topology_mod, "_git_toplevel", _fake_git_toplevel, raising=True)
    payload = topology_mod.inspect_topology(kernel_root=kernel, cwd=kernel)
    assert payload.get("location") == "ajax-kernel"
    assert payload.get("ok") is True
    assert payload.get("git_toplevel_kernel") == str(kernel)


def test_doctor_topology_outputs_recommendation(monkeypatch, tmp_path: Path) -> None:
    ajax_home = tmp_path / "AJAX_HOME"
    kernel = ajax_home / "ajax-kernel"
    kernel.mkdir(parents=True, exist_ok=True)

    def _fake_git_toplevel(path: Path) -> str:
        if "ajax-kernel" in str(path):
            return str(kernel)
        return str(ajax_home)

    monkeypatch.setattr(topology_mod, "_git_toplevel", _fake_git_toplevel, raising=True)
    payload = topology_mod.inspect_topology(kernel_root=kernel, cwd=ajax_home)
    assert payload.get("ok") is False
    assert payload.get("reason") == "running_from_root_not_kernel"
    recommended = payload.get("recommended") if isinstance(payload.get("recommended"), dict) else {}
    full = str(recommended.get("full") or "")
    assert str(kernel) in full
    assert "python bin/ajaxctl doctor topology" in full

