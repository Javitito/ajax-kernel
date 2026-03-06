from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from agency.council_subcall_layer import (
    constitution_files_touched,
    resolve_effective_role_strategy,
    resolve_role_strategy,
    run_council_demo,
    run_doctor_council,
)
from agency.subcall import ProviderCallResult, run_subcall


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _prepare_council_root(root: Path) -> None:
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
                preference: [cloud_balanced, local_fast]
              brain:
                preference: [cloud_balanced, cloud_strong]
              council:
                preference: [cloud_balanced, alt_council, cloud_strong]
        """,
    )
    _write_yaml(
        root / "config" / "model_providers.yaml",
        """
        providers:
          local_fast:
            kind: http_openai
            base_url: "http://127.0.0.1:1235/v1"
            roles: ["scout"]
            tier: cheap
            default_model: "local-fast"
          cloud_balanced:
            kind: http_openai
            base_url: "https://example.invalid/v1"
            api_key_env: "BALANCED_KEY"
            roles: ["scout", "brain", "council"]
            tier: balanced
            default_model: "balanced-model"
          alt_council:
            kind: http_openai
            base_url: "https://example.invalid/v1"
            api_key_env: "ALT_KEY"
            roles: ["council"]
            tier: balanced
            default_model: "alt-council-model"
          cloud_strong:
            kind: http_openai
            base_url: "https://example.invalid/v1"
            api_key_env: "STRONG_KEY"
            roles: ["brain", "council"]
            tier: premium
            default_model: "strong-model"
        """,
    )
    _write_yaml(
        root / "config" / "subcall_timeouts.yaml",
        """
        schema: ajax.subcall_timeouts.v1
        tiers:
          T0: 6
          T1: 8
          T2: 15
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_auth_runtime_snapshot(root: Path, providers: list[dict], *, stamp: str = "20260306T000000Z") -> Path:
    path = root / "artifacts" / "audits" / f"auth_provider_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ajax.doctor.auth.v1",
        "ts_utc": "2026-03-06T00:00:00Z",
        "providers": providers,
        "summary_counts": {"usable": 0, "degraded": 0, "blocked": 0},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _runtime_row(
    provider_name: str,
    *,
    effective_result: str,
    reason_code: str,
    auth_state: str = "present",
    reachability_state: str = "ok",
    quota_state: str = "ok",
) -> dict:
    return {
        "provider_name": provider_name,
        "configured": True,
        "kind": "http_openai",
        "auth_state": auth_state,
        "reachability_state": reachability_state,
        "quota_state": quota_state,
        "effective_result": effective_result,
        "reason_code": reason_code,
        "summary_status": reason_code,
    }


def test_role_strategy_resolution(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    strategy = resolve_role_strategy(tmp_path, "planner", rail="lab")
    assert strategy["role"] == "planner"
    assert strategy["provider_role"] == "brain"
    assert strategy["mode"] == "strong"
    assert strategy["default_tier"] == "T2"
    assert strategy["retries"] == 2


def test_scout_prefers_fast_or_local(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    strategy = resolve_role_strategy(tmp_path, "scout", rail="lab")
    assert strategy["preferred_provider"] == "local_fast"
    assert strategy["mode"] == "cheap"


def test_auditor_prefers_different_provider_from_coder_when_available(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="usable", reason_code="ok"),
            _runtime_row("cloud_strong", effective_result="usable", reason_code="ok"),
            _runtime_row("alt_council", effective_result="usable", reason_code="ok"),
            _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
        ],
    )
    coder = resolve_role_strategy(tmp_path, "coder", rail="lab")
    auditor = resolve_role_strategy(
        tmp_path,
        "auditor",
        rail="lab",
        coder_provider=str(coder.get("preferred_provider")),
    )
    assert coder["preferred_provider"] == "cloud_strong"
    assert auditor["preferred_provider"] != coder["preferred_provider"]


def test_role_strategy_filters_cli_not_installed_providers(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="blocked", reason_code="cli_not_installed"),
            _runtime_row("cloud_strong", effective_result="usable", reason_code="ok"),
            _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
        ],
    )
    strategy = resolve_role_strategy(tmp_path, "coder", rail="lab")
    discarded = strategy.get("discarded_providers") if isinstance(strategy.get("discarded_providers"), list) else []
    assert "cloud_balanced" not in strategy.get("provider_ladder", [])
    assert any(isinstance(item, dict) and item.get("provider") == "cloud_balanced" for item in discarded)


def test_resolve_effective_role_strategy_uses_auth_snapshot(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    effective = resolve_effective_role_strategy(
        role="scout",
        ladder=["cloud_balanced", "local_fast"],
        providers_cfg={
            "local_fast": {"kind": "http_openai", "base_url": "http://127.0.0.1:1235/v1", "tier": "cheap"},
            "cloud_balanced": {"kind": "http_openai", "base_url": "https://example.invalid/v1", "tier": "balanced"},
        },
        auth_snapshot={
            "local_fast": _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
            "cloud_balanced": _runtime_row(
                "cloud_balanced",
                effective_result="blocked",
                reason_code="cli_not_installed",
                reachability_state="down",
            ),
        },
        host_capabilities={"mode": "cheap", "rail": "lab"},
        coder_provider=None,
    )
    discarded = effective.get("discarded_providers") if isinstance(effective.get("discarded_providers"), list) else []
    assert effective.get("preferred_provider") == "local_fast"
    assert any(isinstance(item, dict) and item.get("provider") == "cloud_balanced" for item in discarded)


def test_scout_prefers_local_or_cheap_usable_provider(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="usable", reason_code="ok"),
            _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
        ],
    )
    strategy = resolve_role_strategy(tmp_path, "scout", rail="lab")
    assert strategy.get("preferred_provider") == "local_fast"


def test_coder_prefers_stronger_usable_provider(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="usable", reason_code="ok"),
            _runtime_row("cloud_strong", effective_result="usable", reason_code="ok"),
        ],
    )
    strategy = resolve_role_strategy(tmp_path, "coder", rail="lab")
    assert strategy.get("preferred_provider") == "cloud_strong"


def test_auditor_falls_back_to_same_provider_when_only_one_usable(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="usable", reason_code="ok"),
            _runtime_row("alt_council", effective_result="blocked", reason_code="cli_not_installed"),
            _runtime_row("cloud_strong", effective_result="blocked", reason_code="cli_not_installed"),
        ],
    )
    coder = resolve_role_strategy(tmp_path, "coder", rail="lab")
    auditor = resolve_role_strategy(tmp_path, "auditor", rail="lab", coder_provider=str(coder.get("preferred_provider")))
    assert auditor.get("preferred_provider") == coder.get("preferred_provider")
    hints = auditor.get("next_hint") if isinstance(auditor.get("next_hint"), list) else []
    assert any("solo hay un provider usable" in str(hint) for hint in hints)


def test_judge_uses_strongest_usable_provider(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="usable", reason_code="ok"),
            _runtime_row("cloud_strong", effective_result="usable", reason_code="ok"),
            _runtime_row("alt_council", effective_result="usable", reason_code="ok"),
        ],
    )
    strategy = resolve_role_strategy(tmp_path, "judge", rail="lab")
    assert strategy.get("preferred_provider") == "cloud_strong"


def test_role_timeout_differs_by_role(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="usable", reason_code="ok"),
            _runtime_row("cloud_strong", effective_result="usable", reason_code="ok"),
            _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
            _runtime_row("alt_council", effective_result="usable", reason_code="ok"),
        ],
    )
    scout = resolve_role_strategy(tmp_path, "scout", rail="lab")
    coder = resolve_role_strategy(tmp_path, "coder", rail="lab")
    auditor = resolve_role_strategy(tmp_path, "auditor", rail="lab")
    assert int(scout.get("timeout_seconds") or 0) < int(auditor.get("timeout_seconds") or 0)
    assert int(scout.get("timeout_seconds") or 0) < int(coder.get("timeout_seconds") or 0)


def test_subcall_writes_receipt_on_success(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)

    def _ok_caller(provider, cfg, system_prompt, user_prompt, json_mode):  # noqa: ANN001
        return ProviderCallResult(text='{"ok": true}', tokens=8)

    outcome = run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T1",
        prompt="ping",
        json_mode=False,
        read_ledger=False,
        max_attempts=1,
        human_present=False,
        caller=_ok_caller,
    )
    assert outcome.ok is True
    receipt = tmp_path / outcome.receipt_path
    assert receipt.exists()
    payload = _read_json(receipt)
    assert payload.get("ok") is True
    assert payload.get("reason_code") == "ok"


def test_subcall_writes_receipt_on_fail_closed(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)

    def _fail_caller(provider, cfg, system_prompt, user_prompt, json_mode):  # noqa: ANN001
        raise RuntimeError('http_401:{"error":{"code":"invalid_api_key"}}')

    outcome = run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T1",
        prompt="ping",
        json_mode=False,
        read_ledger=False,
        max_attempts=1,
        human_present=False,
        caller=_fail_caller,
    )
    assert outcome.ok is False
    receipt = tmp_path / outcome.receipt_path
    assert receipt.exists()
    payload = _read_json(receipt)
    assert payload.get("ok") is False
    assert payload.get("reason_code") == "auth_invalid"


def test_subcall_fail_closed_on_missing_provider_with_hint(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "config" / "provider_policy.yaml",
        """
        schema: ajax.provider_policy.v1
        rails:
          lab:
            roles:
              scout:
                preference: [ghost_provider]
        """,
    )
    _write_yaml(tmp_path / "config" / "model_providers.yaml", "providers: {}\n")

    outcome = run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T1",
        prompt="ping",
        json_mode=False,
        read_ledger=False,
        max_attempts=1,
        human_present=False,
    )
    assert outcome.ok is False
    assert outcome.reason_code == "no_provider_available"
    assert any("doctor council" in hint for hint in outcome.next_hint)


def test_subcall_fallback_sequence(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
            _runtime_row("cloud_balanced", effective_result="usable", reason_code="ok"),
        ],
    )
    calls: list[str] = []

    def _caller(provider, cfg, system_prompt, user_prompt, json_mode):  # noqa: ANN001
        calls.append(provider)
        if provider == "local_fast":
            raise RuntimeError("provider_down")
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
    assert outcome.ok is True
    assert outcome.provider_chosen == "cloud_balanced"
    assert calls == ["local_fast", "cloud_balanced"]


def test_doctor_council_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "doctor", "council", "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode in {0, 1}
    payload = json.loads(proc.stdout)
    assert payload.get("schema") == "ajax.doctor.council.v1"


def test_doctor_council_reports_missing_auth_without_secret_leak(tmp_path: Path, monkeypatch) -> None:
    _prepare_council_root(tmp_path)
    monkeypatch.setenv("BALANCED_KEY", "very-secret-value")
    payload = run_doctor_council(tmp_path)
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "very-secret-value" not in dumped
    auth_doc = payload.get("auth_config") if isinstance(payload.get("auth_config"), dict) else {}
    missing = auth_doc.get("missing_auth_providers") if isinstance(auth_doc.get("missing_auth_providers"), list) else []
    assert "cloud_strong" in missing


def test_doctor_council_shows_discarded_providers(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="blocked", reason_code="cli_not_installed"),
            _runtime_row("cloud_strong", effective_result="usable", reason_code="ok"),
            _runtime_row("alt_council", effective_result="usable", reason_code="ok"),
            _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
        ],
    )
    payload = run_doctor_council(tmp_path)
    effective = payload.get("effective_roles") if isinstance(payload.get("effective_roles"), dict) else {}
    coder = effective.get("coder") if isinstance(effective.get("coder"), dict) else {}
    discarded = coder.get("discarded_providers") if isinstance(coder.get("discarded_providers"), list) else []
    assert any(isinstance(item, dict) and item.get("provider") == "cloud_balanced" for item in discarded)
    summary = str(payload.get("summary") or "")
    assert "discarded=[" in summary


def test_council_demo_registered() -> None:
    proc = subprocess.run(
        [sys.executable, "bin/ajaxctl", "council", "demo", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "usage: ajaxctl council demo" in out


def test_council_demo_writes_audit_artifact(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)

    def _fake_subcall_runner(**kwargs):  # noqa: ANN003
        role = kwargs.get("role") or "unknown"
        return SimpleNamespace(
            ok=True,
            terminal="DONE",
            provider_chosen=f"{role}_provider",
            reason_code="ok",
            next_hint=[],
            receipt_path=f"artifacts/receipts/{role}.json",
            role_artifact_path=f"artifacts/subcalls/{role}_x.json",
            ladder_tried=[{"provider": f"{role}_provider", "ok": True}],
        )

    payload = run_council_demo(tmp_path, subcall_runner=_fake_subcall_runner)
    assert payload.get("ok") is True
    artifact_path = Path(str(payload.get("artifact_path")))
    assert artifact_path.exists()
    written = _read_json(artifact_path)
    steps = written.get("steps") if isinstance(written.get("steps"), list) else []
    assert len(steps) == 4


def test_council_demo_does_not_attempt_discarded_providers(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="blocked", reason_code="cli_not_installed"),
            _runtime_row("cloud_strong", effective_result="usable", reason_code="ok"),
            _runtime_row("alt_council", effective_result="usable", reason_code="ok"),
            _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
        ],
    )

    def _fake_subcall_runner(**kwargs):  # noqa: ANN003
        role = kwargs.get("role") or "unknown"
        strategy = resolve_role_strategy(tmp_path, str(role), rail="lab")
        selected = strategy.get("preferred_provider")
        return SimpleNamespace(
            ok=True,
            terminal="DONE",
            provider_chosen=selected,
            reason_code="ok",
            next_hint=[],
            receipt_path=f"artifacts/receipts/{role}.json",
            role_artifact_path=f"artifacts/subcalls/{role}_x.json",
            ladder_tried=[{"provider": selected, "ok": True}],
        )

    payload = run_council_demo(tmp_path, subcall_runner=_fake_subcall_runner)
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    assert steps
    for step in steps:
        discarded = step.get("discarded_providers") if isinstance(step.get("discarded_providers"), list) else []
        ladder = step.get("provider_ladder") if isinstance(step.get("provider_ladder"), list) else []
        discarded_names = {str(item.get("provider")) for item in discarded if isinstance(item, dict)}
        assert not (set(str(item) for item in ladder) & discarded_names)


def test_subcall_receipt_includes_discarded_providers(tmp_path: Path) -> None:
    _prepare_council_root(tmp_path)
    _write_auth_runtime_snapshot(
        tmp_path,
        [
            _runtime_row("cloud_balanced", effective_result="blocked", reason_code="cli_not_installed"),
            _runtime_row("local_fast", effective_result="usable", reason_code="ok"),
        ],
    )

    def _ok_caller(provider, cfg, system_prompt, user_prompt, json_mode):  # noqa: ANN001
        return ProviderCallResult(text="ok", tokens=4)

    outcome = run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T1",
        prompt="ping",
        json_mode=False,
        read_ledger=False,
        max_attempts=1,
        human_present=False,
        caller=_ok_caller,
    )
    receipt = _read_json(tmp_path / outcome.receipt_path)
    discarded = receipt.get("discarded_providers") if isinstance(receipt.get("discarded_providers"), list) else []
    assert any(isinstance(item, dict) and item.get("provider") == "cloud_balanced" for item in discarded)
    role_artifact_rel = str(receipt.get("role_artifact_path") or "")
    role_artifact_path = tmp_path / role_artifact_rel if role_artifact_rel else Path("")
    assert role_artifact_path.exists()
    role_artifact = _read_json(role_artifact_path)
    role_discarded = role_artifact.get("discarded_providers") if isinstance(role_artifact.get("discarded_providers"), list) else []
    assert any(isinstance(item, dict) and item.get("provider") == "cloud_balanced" for item in role_discarded)


def test_no_constitution_files_touched_guard() -> None:
    assert constitution_files_touched(["AGENTS.md"]) is True
    assert constitution_files_touched(["PSEUDOCODE_MAP/flow.md"]) is True
    assert constitution_files_touched(["agency/subcall.py", "tests/test_kernel_cli.py"]) is False


def test_no_secret_leak_in_council_outputs(tmp_path: Path, monkeypatch) -> None:
    _prepare_council_root(tmp_path)
    monkeypatch.setenv("BALANCED_KEY", "ULTRA_SECRET_BALANCED")
    payload = run_doctor_council(tmp_path)
    dumped = json.dumps(payload, ensure_ascii=False)
    assert "ULTRA_SECRET_BALANCED" not in dumped
