from __future__ import annotations

import json
from pathlib import Path

import pytest

from agency.policy_contract import validate_policy_contract


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("missing_file", ["provider_policy.yaml", "provider_failure_policy.yaml"])
def test_policy_contract_blocks_when_required_yaml_missing(tmp_path: Path, missing_file: str) -> None:
    cfg = tmp_path / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    if missing_file != "provider_policy.yaml":
        _write(cfg / "provider_policy.yaml", "schema: ajax.provider_policy.v1\nproviders: {}\n")
    if missing_file != "provider_failure_policy.yaml":
        _write(cfg / "provider_failure_policy.yaml", "planning:\n  max_attempts: 2\n")

    result = validate_policy_contract(tmp_path, sync_json=True, write_receipt=True)

    assert result.ok is False
    assert result.status == "BLOCKED"
    assert result.reason == "missing_policy_files"
    assert missing_file in "\n".join(result.missing_files)
    assert result.receipt_path is not None
    assert Path(result.receipt_path).exists()


def test_policy_contract_syncs_provider_policy_json_from_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    _write(
        cfg / "provider_policy.yaml",
        "schema: ajax.provider_policy.v1\nproviders:\n  groq:\n    cost_class: generous\n",
    )
    _write(
        cfg / "provider_failure_policy.yaml",
        "planning:\n  max_attempts: 3\nproviders:\n  cooldown_seconds_default: 90\n",
    )

    result = validate_policy_contract(tmp_path, sync_json=True, write_receipt=True)

    assert result.ok is True
    assert result.status == "READY"
    target = tmp_path / "config" / "provider_policy.json"
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload.get("schema") == "ajax.provider_policy.v1"
    assert payload.get("providers", {}).get("groq", {}).get("cost_class") == "generous"
