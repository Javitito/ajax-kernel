from __future__ import annotations

import json
import textwrap
from pathlib import Path

import agency.provider_ledger as provider_ledger_mod
from agency.provider_ledger import ProviderLedger
from agency.subcall import ProviderCallResult, run_subcall


def _write_configs(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "provider_ledger").mkdir(parents=True, exist_ok=True)
    (root / "config" / "provider_policy.yaml").write_text(
        textwrap.dedent(
            """
            schema: ajax.provider_policy.v1
            defaults:
              cooldowns:
                timeout: 600
                bridge_error: 20
              probe_failures_threshold: 2
              probe_failures_cooldown_seconds: 900
              save_codex_exclude_prefixes: [codex_]
            rails:
              lab:
                roles:
                  scout:
                    preference: [probe_cli, fallback_cli]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "config" / "model_providers.yaml").write_text(
        textwrap.dedent(
            """
            providers:
              probe_cli:
                kind: cli
                command: ["probe-cli", "{prompt}"]
                roles: ["scout"]
                tier: cheap
              fallback_cli:
                kind: cli
                command: ["fallback-cli", "{prompt}"]
                roles: ["scout"]
                tier: cheap
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _row(doc: dict, provider: str = "probe_cli", role: str = "scout") -> dict:
    for row in doc.get("rows", []):
        if row.get("provider") == provider and row.get("role") == role:
            return row
    raise AssertionError(f"row not found for {provider}:{role}")


def test_probe_failures_reach_threshold_and_set_cooldown(tmp_path, monkeypatch) -> None:
    _write_configs(tmp_path)
    ledger = ProviderLedger(root_dir=tmp_path)

    monkeypatch.setattr(ProviderLedger, "_auth_status", lambda self, provider, cfg: None, raising=True)
    monkeypatch.setattr(ProviderLedger, "_status_doc", lambda self: {}, raising=True)

    responses = [(False, "bridge_error", "probe_err_1"), (False, "bridge_error", "probe_err_2")]

    def _fake_probe(self, provider, cfg):  # noqa: ANN001
        if provider == "probe_cli":
            return responses.pop(0)
        return True, None, "probe_ok"

    monkeypatch.setattr(ProviderLedger, "_probe_cli_presence", _fake_probe, raising=True)

    first = ledger.refresh()
    first_row = _row(first)
    assert first_row["consecutive_probe_failures"] == 1
    assert first_row["reason"] == "bridge_error"
    assert first_row["cooldown_until_ts"] is None

    second = ledger.refresh()
    second_row = _row(second)
    assert second_row["consecutive_probe_failures"] == 2
    assert second_row["reason"] == "probe_failed"
    assert second_row["cooldown_until_ts"] is not None


def test_subcall_skips_provider_in_cooldown(tmp_path, monkeypatch) -> None:
    _write_configs(tmp_path)
    ledger_doc = {
        "schema": "ajax.provider_ledger.v1",
        "rows": [
            {
                "provider": "probe_cli",
                "role": "scout",
                "status": "soft_fail",
                "reason": "probe_failed",
                "cooldown_until": "2099-01-01T00:00:00Z",
                "cooldown_until_ts": 4070908800.0,
                "consecutive_probe_failures": 2,
            },
            {
                "provider": "fallback_cli",
                "role": "scout",
                "status": "ok",
                "reason": None,
                "cooldown_until": None,
                "cooldown_until_ts": None,
                "consecutive_probe_failures": 0,
            },
        ],
    }
    (tmp_path / "artifacts" / "provider_ledger" / "latest.json").write_text(
        json.dumps(ledger_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(ProviderLedger, "refresh", lambda self: ledger_doc, raising=True)

    called: list[str] = []

    def _fake_caller(provider, cfg, system_prompt, user_prompt, json_mode):  # noqa: ANN001
        called.append(provider)
        return ProviderCallResult(text='{"ok": true}', tokens=10)

    outcome = run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T1",
        prompt="ping",
        json_mode=False,
        read_ledger=True,
        max_attempts=2,
        human_present=False,
        caller=_fake_caller,
    )

    assert outcome.ok is True
    assert outcome.provider_chosen == "fallback_cli"
    assert called == ["fallback_cli"]
    assert outcome.ladder_tried[0]["provider"] == "probe_cli"
    assert outcome.ladder_tried[0]["reason"] == "cooldown_active"
    assert outcome.ladder_tried[0]["skipped"] is True


def test_probe_success_clears_probe_cooldown(tmp_path, monkeypatch) -> None:
    _write_configs(tmp_path)
    ledger = ProviderLedger(root_dir=tmp_path)

    monkeypatch.setattr(ProviderLedger, "_auth_status", lambda self, provider, cfg: None, raising=True)
    monkeypatch.setattr(ProviderLedger, "_status_doc", lambda self: {}, raising=True)

    responses = [
        (False, "bridge_error", "probe_err_1"),
        (False, "bridge_error", "probe_err_2"),
        (True, None, "probe_ok"),
    ]

    def _fake_probe(self, provider, cfg):  # noqa: ANN001
        if provider == "probe_cli":
            return responses.pop(0)
        return True, None, "probe_ok"

    monkeypatch.setattr(ProviderLedger, "_probe_cli_presence", _fake_probe, raising=True)
    ts_values = iter([1000.0, 1001.0, 2000.0])
    monkeypatch.setattr(provider_ledger_mod, "_now_ts", lambda: next(ts_values), raising=True)

    ledger.refresh()
    ledger.refresh()
    third = ledger.refresh()
    third_row = _row(third)
    assert third_row["status"] == "ok"
    assert third_row["reason"] is None
    assert third_row["cooldown_until_ts"] is None
    assert third_row["consecutive_probe_failures"] == 0
