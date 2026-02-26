from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import agency.subcall as subcall


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


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
                preference: [test_cli]
        """,
    )
    _write_yaml(
        root / "config" / "model_providers.yaml",
        """
        providers:
          test_cli:
            kind: cli
            command: ["python", "-c", "print('ok')", "{prompt}"]
            roles: ["scout"]
            tier: cheap
            timeout_seconds: 40
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


def test_subcall_uses_timeout_map_by_tier(tmp_path, monkeypatch) -> None:
    _prepare_root(tmp_path)

    captured_timeouts: list[float] = []

    def _fake_run(cmd, input, capture_output, text, timeout, check):  # noqa: ANN001
        captured_timeouts.append(float(timeout))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subcall.subprocess, "run", _fake_run, raising=True)

    out_t0 = subcall.run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T0",
        prompt="ping",
        json_mode=False,
        read_ledger=False,
        max_attempts=1,
        human_present=False,
    )
    out_t2 = subcall.run_subcall(
        root_dir=tmp_path,
        role="scout",
        tier="T2",
        prompt="ping",
        json_mode=False,
        read_ledger=False,
        max_attempts=1,
        human_present=False,
    )

    assert out_t0.ok is True
    assert out_t2.ok is True
    assert captured_timeouts == [6.0, 15.0]
