from __future__ import annotations

import subprocess

import agency.bridge_doctor as bridge_doctor


def test_bridge_doctor_prefers_probe_command_for_cli(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        bridge_doctor,
        "_load_provider_configs",
        lambda _root: {
            "providers": {
                "gemini_cli": {
                    "kind": "cli",
                    "probe_command": ["gemini", "--version"],
                    "infer_command": ["gemini", "--model", "auto", "-o", "json", "{prompt}"],
                },
                "codex_cli": {
                    "kind": "cli",
                    "probe_command": ["codex", "--version"],
                    "infer_command": ["cdx", "models", "--json"],
                },
                "qwen_cli": {
                    "kind": "cli",
                    "command": ["qwen", "models"],
                },
            }
        },
        raising=True,
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, capture_output, text, timeout, env, cwd, check):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(bridge_doctor.subprocess, "run", _fake_run, raising=True)

    payload = bridge_doctor.run_bridge_doctor(tmp_path, timeout_s=0.5)

    assert calls == [["gemini", "--version"], ["codex", "--version"]]
    assert payload["providers"]["gemini_cli"]["command_source"] == "probe"
    assert payload["providers"]["codex_cli"]["command_source"] == "probe"
    assert payload["providers"]["qwen_cli"]["status"] == "UNPROBED_COMMAND_TYPE"
    assert payload["providers"]["qwen_cli"]["command_source"] == "unprobeable"
    assert payload["summary"]["configured_cli_providers"] == 3
    assert payload["summary"]["probed_count"] == 2
    assert payload["summary"]["skipped_unprobed_count"] == 1
    assert payload["summary"]["coverage_ok"] is True
