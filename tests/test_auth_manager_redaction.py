from __future__ import annotations

import json
import textwrap
from pathlib import Path

import agency.auth_manager as auth_manager
from agency.auth_manager import AuthManager
from agency.auth_provider_diagnostics import collect_provider_auth_diagnostics


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _prepare_root(root: Path) -> None:
    _write_yaml(
        root / "config" / "model_providers.yaml",
        """
        providers:
          groq:
            kind: http_openai
            base_url: "https://api.groq.com/openai/v1"
            api_key_env: "GROQ_API_KEY"
            roles: ["brain"]
            default_model: "llama-3.1-8b"
          qwen_cloud:
            kind: cli
            command: ["qwen", "--version"]
            roles: ["brain"]
            default_model: "qwen-7b"
          gemini_cli:
            kind: cli
            command: ["gemini", "-h"]
            roles: ["brain"]
            default_model: "gemini-flash"
        """,
    )


def test_auth_source_redacts_http_env(monkeypatch, tmp_path: Path) -> None:
    manager = AuthManager(root_dir=tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")
    source = manager.auth_source("groq", {"kind": "http_openai", "api_key_env": "GROQ_API_KEY"})
    assert source == "env:<present>"


def test_auth_source_redacts_wsl_file_paths(monkeypatch, tmp_path: Path) -> None:
    manager = AuthManager(root_dir=tmp_path)
    monkeypatch.delenv("QWEN_OAUTH", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        auth_manager,
        "_wsl_path_exists",
        lambda rel_path: rel_path == "~/.qwen/oauth_creds.json",
        raising=True,
    )
    monkeypatch.setattr(auth_manager, "_wsl_env_has", lambda _: False, raising=True)
    source = manager.auth_source("qwen_cloud", {"kind": "cli", "command": ["qwen", "--version"]})
    assert source == "file:<redacted>"


def test_auth_source_redacts_config_paths(monkeypatch, tmp_path: Path) -> None:
    manager = AuthManager(root_dir=tmp_path)
    monkeypatch.delenv("QWEN_OAUTH", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        auth_manager,
        "_wsl_path_exists",
        lambda rel_path: rel_path == "~/.qwen/settings.json",
        raising=True,
    )
    monkeypatch.setattr(auth_manager, "_wsl_env_has", lambda _: False, raising=True)
    source = manager.auth_source("qwen_cloud", {"kind": "cli", "command": ["qwen", "--version"]})
    assert source == "config:<redacted>"


def test_doctor_auth_payload_uses_redacted_auth_source(monkeypatch, tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")
    monkeypatch.setattr(
        auth_manager,
        "_wsl_path_exists",
        lambda rel_path: rel_path in {"~/.qwen/oauth_creds.json", "~/.gemini/oauth_creds.json"},
        raising=True,
    )
    monkeypatch.setattr(auth_manager, "_wsl_env_has", lambda _: False, raising=True)

    payload = collect_provider_auth_diagnostics(tmp_path, include_probes=False, write_artifact=False)
    rows = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    by_provider = {str(row.get("provider_name")): row for row in rows if isinstance(row, dict)}

    assert by_provider["groq"].get("auth_source") == "env:<present>"
    assert by_provider["qwen_cloud"].get("auth_source") == "file:<redacted>"
    assert by_provider["gemini_cli"].get("auth_source") == "file:<redacted>"

    dumped = json.dumps(payload, ensure_ascii=False)
    assert "oauth_creds.json" not in dumped
    assert "wsl_file:" not in dumped
    assert "env:GROQ_API_KEY" not in dumped
