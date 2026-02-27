from __future__ import annotations

import textwrap

from agency.subcall import ProviderCallResult, run_subcall


def test_subcall_skips_cli_without_prompt_placeholder(tmp_path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    (tmp_path / "config" / "provider_policy.yaml").write_text(
        textwrap.dedent(
            """
            schema: ajax.provider_policy.v1
            defaults:
              save_codex_exclude_prefixes: [codex_]
            rails:
              lab:
                roles:
                  scout:
                    preference: [bad_cli, good_cli]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (tmp_path / "config" / "model_providers.yaml").write_text(
        textwrap.dedent(
            """
            providers:
              bad_cli:
                kind: cli
                command: ["qwen", "models"]
                roles: ["scout"]
                tier: "cheap"
              good_cli:
                kind: cli
                command: ["python", "-c", "print('ok')", "{prompt}"]
                roles: ["scout"]
                tier: "cheap"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    called: list[str] = []

    def _fake_caller(provider, cfg, system_prompt, user_prompt, json_mode):  # noqa: ANN001
        called.append(provider)
        return ProviderCallResult(text='{"status":"ok"}', tokens=12)

    outcome = run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T1",
        prompt="ping",
        json_mode=False,
        read_ledger=False,
        max_attempts=2,
        human_present=False,
        caller=_fake_caller,
    )

    assert outcome.ok is True
    assert outcome.provider_chosen == "good_cli"
    assert called == ["good_cli"]
    assert outcome.ladder_tried[0]["provider"] == "bad_cli"
    assert outcome.ladder_tried[0]["reason"] == "incompatible_for_subcall"
    assert outcome.ladder_tried[0]["detail"] == "command_has_no_prompt_placeholder"
    assert outcome.ladder_tried[0]["skipped"] is True
