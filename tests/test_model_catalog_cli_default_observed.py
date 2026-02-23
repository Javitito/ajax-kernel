from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import agency.models_registry as models_registry
import agency.provider_breathing as provider_breathing
import agency.provider_ledger as provider_ledger


def test_cli_model_effective_default_and_observed_helpers() -> None:
    status_entry = {
        "breathing": {
            "roles": {
                "brain": {
                    "command": "gemini --model auto -o json",
                }
            }
        }
    }
    assert provider_breathing._provider_model_effective(
        "gemini_cli", {"kind": "cli", "default_model": "gemini-2.5-pro"}
    ) == "DEFAULT"
    assert provider_breathing._provider_status_model_observed(status_entry) == "auto"

    assert provider_ledger._pick_model_id(
        "gemini_cli",
        {"kind": "cli", "default_model": "gemini-2.5-pro"},
        status_entry=status_entry,
    ) == "auto"
    assert provider_ledger._pick_model_id(
        "codex_brain",
        {"kind": "codex_cli_jsonl", "default_model": "gpt-5.1-codex-max"},
        status_entry={},
    ) == "DEFAULT"

    row = provider_ledger.LedgerRow(
        provider="codex_brain",
        model="DEFAULT",
        role="brain",
        status="ok",
    )
    assert row.to_dict()["model_effective"] == "DEFAULT"


def test_probe_cli_strips_model_flag_for_gemini_cli(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        seen["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(provider_breathing.subprocess, "run", _fake_run, raising=True)
    result = provider_breathing._probe_cli(
        "gemini_cli",
        {
            "kind": "cli",
            "command": ["gemini", "--model", "auto", "-o", "json", "{prompt}"],
            "default_model": "gemini-2.5-pro",
            "tier": "balanced",
        },
        role="brain",
        timeout_sec=2,
    )

    assert result.status == "UP"
    assert "--model" not in (result.command or "")
    assert seen["cmd"][:3] == ["gemini", "-o", "json"]


def test_probe_codex_jsonl_does_not_append_model_flag(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        seen["cmd"] = list(cmd)
        stdout = (
            '{"type":"item.completed","item":{"type":"assistant_message","text":"pong"}}\n'
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(provider_breathing.os, "name", "posix", raising=False)
    monkeypatch.setattr(provider_breathing.subprocess, "run", _fake_run, raising=True)

    result = provider_breathing._probe_codex_jsonl(
        "codex_brain",
        {"kind": "codex_cli_jsonl", "default_model": "gpt-5.1-codex-max", "tier": "premium"},
        role="brain",
        timeout_sec=2,
    )

    assert result.status == "UP"
    assert "--model" not in (result.command or "")
    assert "--model" not in " ".join(seen["cmd"])


def test_provider_breathing_records_default_and_observed_model(monkeypatch, tmp_path: Path) -> None:
    class _AuthState:
        def __init__(self) -> None:
            self.state = "OK"
            self.reason = ""

    class _FakeAuth:
        def __init__(self, *, root_dir: Path) -> None:
            self.root_dir = root_dir

        def auth_state(self, name: str, cfg: dict) -> _AuthState:  # noqa: ANN001
            return _AuthState()

        def ensure_auth_gap(self, name: str, auth_state: _AuthState) -> None:  # noqa: ANN001
            return None

    def _fake_transport(cfg: dict) -> provider_breathing.ProbeResult:  # noqa: ANN001
        return provider_breathing.ProbeResult(
            status="UP",
            checked_at=provider_breathing._now_ts(),
            command="gemini healthcheck",
        )

    def _fake_cli(provider: str, cfg: dict, role: str, timeout_sec: int) -> provider_breathing.ProbeResult:  # noqa: ANN001
        return provider_breathing.ProbeResult(
            status="UP",
            checked_at=provider_breathing._now_ts(),
            command="gemini --model auto -o json '{}'",
            signals={"valid_json": True},
        )

    monkeypatch.setattr(provider_breathing, "AuthManager", _FakeAuth, raising=True)
    monkeypatch.setattr(provider_breathing, "_probe_cli_transport", _fake_transport, raising=True)
    monkeypatch.setattr(provider_breathing, "_probe_cli", _fake_cli, raising=True)
    monkeypatch.setattr(provider_breathing, "_load_policy", lambda: {}, raising=True)

    loop = provider_breathing.ProviderBreathingLoop(
        root_dir=tmp_path,
        provider_configs={
            "providers": {
                "gemini_cli": {
                    "kind": "cli",
                    "command": ["gemini", "--model", "auto", "-o", "json", "{prompt}"],
                    "roles": ["brain"],
                    "tier": "balanced",
                }
            }
        },
    )
    status_doc = loop.run_once(roles=["brain"])
    entry = status_doc["providers"]["gemini_cli"]

    assert entry["model_effective"] == "DEFAULT"
    assert entry["model_observed"] == "auto"


def test_groq_dynamic_http_catalog_discovery_preserved(monkeypatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"id": "groq-live-model"}]}

    def _fake_get(url, headers=None, timeout=20):  # noqa: ANN001
        return _Resp()

    monkeypatch.setenv("GROQ_API_KEY", "x-test")
    monkeypatch.setattr(models_registry, "requests", type("R", (), {"get": staticmethod(_fake_get)})(), raising=True)

    models = models_registry._from_openai_http(
        "groq",
        {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key_env": "GROQ_API_KEY",
            "static_models": [{"id": "groq-static-model", "modalities": ["text"]}],
            "vision_hints": ["vision"],
        },
    )

    pairs = {(m.id, m.source) for m in models}
    assert ("groq-live-model", "http_openai") in pairs
    assert ("groq-static-model", "static") in pairs
