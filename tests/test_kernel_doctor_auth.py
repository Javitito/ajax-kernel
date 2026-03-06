from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from agency.auth_provider_diagnostics import collect_provider_auth_diagnostics
from agency.council_subcall_layer import constitution_files_touched, resolve_role_strategy, run_doctor_council
from agency.subcall import ProviderCallResult, run_subcall


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_root(root: Path) -> None:
    _write_yaml(
        root / "config" / "provider_policy.yaml",
        """
        schema: ajax.provider_policy.v1
        defaults:
          save_codex_exclude_prefixes: [codex_]
        rails:
          lab:
            roles:
              scout:
                preference: [qwen_cloud, lmstudio]
              brain:
                preference: [groq, lmstudio]
              council:
                preference: [qwen_cloud, groq, lmstudio]
        """,
    )
    _write_yaml(
        root / "config" / "model_providers.yaml",
        """
        providers:
          groq:
            kind: http_openai
            base_url: "https://api.groq.com/openai/v1"
            api_key_env: "GROQ_API_KEY"
            roles: ["brain", "council", "scout"]
            tier: balanced
            default_model: "llama"
          lmstudio:
            kind: http_openai
            base_url: "http://127.0.0.1:1235/v1"
            roles: ["brain", "council", "scout"]
            tier: cheap
            default_model: "local-fast"
          qwen_cloud:
            kind: cli
            command: ["qwen", "-o", "json", "{prompt}"]
            probe_command: ["qwen", "--version"]
            infer_command: ["qwen", "-o", "json", "{prompt}"]
            roles: ["brain", "council", "scout"]
            tier: cheap
            default_model: "qwen-7b"
          gemini_cli:
            kind: cli
            command: ["gemini", "--model", "auto", "-o", "json", "{prompt}"]
            probe_command: ["gemini", "-h"]
            infer_command: ["gemini", "--model", "auto", "-o", "json", "{prompt}"]
            roles: ["brain", "scout"]
            tier: balanced
            default_model: "gemini-flash"
        """,
    )
    _write_yaml(
        root / "config" / "provider_failure_policy.yaml",
        """
        schema: ajax.provider_failure_policy.v1
        planning:
          max_attempts: 2
        """,
    )
    _write_yaml(
        root / "config" / "subcall_timeouts.yaml",
        """
        schema: ajax.subcall_timeouts.v1
        tiers:
          T0: 5
          T1: 8
          T2: 15
        """,
    )


def _write_providers_status(root: Path, providers: dict) -> None:
    _write_json(
        root / "artifacts" / "health" / "providers_status.json",
        {
            "providers": providers,
            "updated_utc": "2026-03-06T00:00:00Z",
        },
    )


def _provider_row(payload: dict, provider_name: str) -> dict:
    rows = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    for row in rows:
        if isinstance(row, dict) and row.get("provider_name") == provider_name:
            return row
    return {}


def test_doctor_auth_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "doctor", "auth", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "usage: ajaxctl doctor auth" in out


def test_doctor_auth_reports_auth_missing(tmp_path: Path, monkeypatch) -> None:
    _prepare_root(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    payload = collect_provider_auth_diagnostics(tmp_path, include_probes=False, write_artifact=False)
    groq = _provider_row(payload, "groq")
    assert groq.get("auth_state") == "missing"
    assert groq.get("reason_code") == "auth_missing"
    assert groq.get("effective_result") == "blocked"


def test_doctor_auth_reports_auth_present_without_secret_leak(tmp_path: Path, monkeypatch) -> None:
    _prepare_root(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value")
    payload = collect_provider_auth_diagnostics(tmp_path, include_probes=False, write_artifact=False)
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "super-secret-value" not in dumped
    groq = _provider_row(payload, "groq")
    assert groq.get("auth_state") == "present"


def test_doctor_auth_reports_timeout_vs_auth(tmp_path: Path, monkeypatch) -> None:
    _prepare_root(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "present-token")

    def _probe(provider: str, cfg: dict, timeout_s: float) -> dict:
        if provider == "groq":
            return {"reason_code": "provider_timeout", "reachability_state": "timeout", "quota_state": "ok", "probe": {}}
        if provider == "qwen_cloud":
            return {"reason_code": "auth_invalid", "reachability_state": "down", "quota_state": "unknown", "probe": {}}
        return {"reason_code": "ok", "reachability_state": "ok", "quota_state": "ok", "probe": {}}

    payload = collect_provider_auth_diagnostics(
        tmp_path,
        include_probes=True,
        write_artifact=False,
        probe_fn=_probe,
    )
    groq = _provider_row(payload, "groq")
    qwen = _provider_row(payload, "qwen_cloud")
    assert groq.get("reachability_state") == "timeout"
    assert groq.get("reason_code") == "provider_timeout"
    assert qwen.get("reason_code") == "auth_invalid"
    assert qwen.get("effective_result") == "blocked"


def test_doctor_auth_reports_cli_not_installed(tmp_path: Path) -> None:
    _prepare_root(tmp_path)

    def _probe(provider: str, cfg: dict, timeout_s: float) -> dict:
        if provider == "qwen_cloud":
            return {"reason_code": "cli_not_installed", "reachability_state": "down", "quota_state": "unknown", "probe": {}}
        return {"reason_code": "ok", "reachability_state": "ok", "quota_state": "ok", "probe": {}}

    payload = collect_provider_auth_diagnostics(
        tmp_path,
        include_probes=True,
        write_artifact=False,
        probe_fn=_probe,
    )
    qwen = _provider_row(payload, "qwen_cloud")
    assert qwen.get("reason_code") == "cli_not_installed"
    assert qwen.get("effective_result") == "blocked"


def test_strategy_avoids_blocked_provider_for_scout(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "config" / "provider_policy.yaml",
        """
        schema: ajax.provider_policy.v1
        rails:
          lab:
            roles:
              scout:
                preference: [blocked_http, lmstudio]
        """,
    )
    _write_yaml(
        tmp_path / "config" / "model_providers.yaml",
        """
        providers:
          blocked_http:
            kind: http_openai
            base_url: "https://example.invalid/v1"
            api_key_env: "BLOCKED_KEY"
            roles: ["scout"]
            tier: cheap
            default_model: "blocked"
          lmstudio:
            kind: http_openai
            base_url: "http://127.0.0.1:1235/v1"
            roles: ["scout"]
            tier: cheap
            default_model: "local-fast"
        """,
    )
    _write_providers_status(
        tmp_path,
        {
            "blocked_http": {"transport": {"status": "DOWN", "reason": "env_missing:BLOCKED_KEY"}},
            "lmstudio": {"transport": {"status": "UP", "reason": ""}},
        },
    )
    strategy = resolve_role_strategy(tmp_path, "scout", rail="lab")
    assert strategy.get("preferred_provider") == "lmstudio"
    assert "blocked_http" not in strategy.get("provider_ladder", [])


def test_strategy_avoids_timeout_provider_when_usable_fallback_exists(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "config" / "provider_policy.yaml",
        """
        schema: ajax.provider_policy.v1
        rails:
          lab:
            roles:
              scout:
                preference: [slow_local, fast_local]
        """,
    )
    _write_yaml(
        tmp_path / "config" / "model_providers.yaml",
        """
        providers:
          slow_local:
            kind: http_openai
            base_url: "http://127.0.0.1:19999/v1"
            roles: ["scout"]
            tier: cheap
            default_model: "slow"
          fast_local:
            kind: http_openai
            base_url: "http://127.0.0.1:18888/v1"
            roles: ["scout"]
            tier: cheap
            default_model: "fast"
        """,
    )
    _write_providers_status(
        tmp_path,
        {
            "slow_local": {"transport": {"status": "DOWN", "reason": "timeout"}},
            "fast_local": {"transport": {"status": "UP", "reason": ""}},
        },
    )
    strategy = resolve_role_strategy(tmp_path, "scout", rail="lab")
    assert strategy.get("preferred_provider") == "fast_local"
    assert strategy.get("provider_ladder", [None])[0] == "fast_local"


def test_subcall_receipt_includes_reason_code_and_fallbacks(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "config" / "provider_policy.yaml",
        """
        schema: ajax.provider_policy.v1
        rails:
          lab:
            roles:
              scout:
                preference: [local_a, local_b]
        """,
    )
    _write_yaml(
        tmp_path / "config" / "model_providers.yaml",
        """
        providers:
          local_a:
            kind: http_openai
            base_url: "http://127.0.0.1:19001/v1"
            roles: ["scout"]
            tier: cheap
            default_model: "a"
          local_b:
            kind: http_openai
            base_url: "http://127.0.0.1:19002/v1"
            roles: ["scout"]
            tier: cheap
            default_model: "b"
        """,
    )
    _write_yaml(
        tmp_path / "config" / "provider_failure_policy.yaml",
        """
        schema: ajax.provider_failure_policy.v1
        planning:
          max_attempts: 2
        """,
    )
    _write_yaml(
        tmp_path / "config" / "subcall_timeouts.yaml",
        """
        schema: ajax.subcall_timeouts.v1
        tiers:
          T0: 5
          T1: 8
          T2: 15
        """,
    )
    _write_providers_status(
        tmp_path,
        {
            "local_a": {"transport": {"status": "UP", "reason": ""}},
            "local_b": {"transport": {"status": "UP", "reason": ""}},
        },
    )
    seen: list[str] = []

    def _caller(provider, cfg, system_prompt, user_prompt, json_mode):  # noqa: ANN001
        seen.append(provider)
        if provider == "local_a":
            raise RuntimeError("timeout")
        return ProviderCallResult(text="ok", tokens=4)

    outcome = run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T1",
        prompt="ping",
        json_mode=False,
        read_ledger=False,
        max_attempts=2,
        human_present=False,
        caller=_caller,
    )
    receipt = json.loads((tmp_path / outcome.receipt_path).read_text(encoding="utf-8"))
    role_artifact = json.loads((tmp_path / str(outcome.role_artifact_path)).read_text(encoding="utf-8"))
    assert receipt.get("reason_code") == "ok"
    assert isinstance(receipt.get("ladder_tried"), list) and len(receipt.get("ladder_tried")) >= 2
    assert "local_a" in role_artifact.get("fallbacks_tried", [])
    assert seen[:2] == ["local_a", "local_b"]


def test_qwen_gemini_checks_are_optional_if_not_installed(tmp_path: Path) -> None:
    _prepare_root(tmp_path)

    def _probe(provider: str, cfg: dict, timeout_s: float) -> dict:
        if provider in {"qwen_cloud", "gemini_cli"}:
            return {"reason_code": "cli_not_installed", "reachability_state": "down", "quota_state": "unknown", "probe": {}}
        return {"reason_code": "ok", "reachability_state": "ok", "quota_state": "ok", "probe": {}}

    payload = collect_provider_auth_diagnostics(
        tmp_path,
        include_probes=True,
        write_artifact=False,
        probe_fn=_probe,
    )
    qwen = _provider_row(payload, "qwen_cloud")
    gemini = _provider_row(payload, "gemini_cli")
    assert qwen.get("reason_code") == "cli_not_installed"
    assert gemini.get("reason_code") == "cli_not_installed"


def test_doctor_auth_does_not_print_secrets(tmp_path: Path, monkeypatch) -> None:
    _prepare_root(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "TOP_SECRET_123")
    monkeypatch.setenv("GEMINI_API_KEY", "TOP_SECRET_456")
    payload = collect_provider_auth_diagnostics(tmp_path, include_probes=False, write_artifact=False)
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "TOP_SECRET_123" not in dumped
    assert "TOP_SECRET_456" not in dumped
    summary = str(payload.get("summary") or "")
    assert "TOP_SECRET_123" not in summary
    assert "TOP_SECRET_456" not in summary


def test_doctor_council_integration_with_auth_status(tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    _write_providers_status(
        tmp_path,
        {
            "qwen_cloud": {"transport": {"status": "DOWN", "reason": "env_missing:QWEN_OAUTH"}},
            "lmstudio": {"transport": {"status": "UP", "reason": ""}},
            "groq": {"transport": {"status": "DOWN", "reason": "env_missing:GROQ_API_KEY"}},
        },
    )
    payload = run_doctor_council(tmp_path)
    auth_doc = payload.get("auth_config") if isinstance(payload.get("auth_config"), dict) else {}
    blocked = auth_doc.get("blocked_providers") if isinstance(auth_doc.get("blocked_providers"), list) else []
    hints = payload.get("next_hint") if isinstance(payload.get("next_hint"), list) else []
    scout = payload.get("strategies", {}).get("scout", {})
    assert "qwen_cloud" in blocked
    assert any("doctor auth" in str(hint) for hint in hints)
    assert scout.get("preferred_provider") == "lmstudio"


def test_no_constitution_files_touched_guard() -> None:
    assert constitution_files_touched(["AGENTS.md"]) is True
    assert constitution_files_touched(["PSEUDOCODE_MAP/flow.md"]) is True
    assert constitution_files_touched(["agency/auth_provider_diagnostics.py"]) is False
