from __future__ import annotations

from agency.bridge_doctor import evaluate_bridge_health


def _payload(*, coverage_ok: bool, provider_statuses: dict[str, str]) -> dict:
    return {
        "summary": {"coverage_ok": coverage_ok},
        "providers": {name: {"status": status} for name, status in provider_statuses.items()},
    }


def test_bridge_exit_non_strict_allows_warnings_when_coverage_ok() -> None:
    payload = _payload(coverage_ok=True, provider_statuses={"qwen_cloud": "TIMEOUT"})

    verdict = evaluate_bridge_health(payload, strict=False)

    assert verdict["coverage_ok"] is True
    assert verdict["ok"] is True
    assert verdict["exit_code"] == 0
    assert verdict["warning_providers"] == ["qwen_cloud"]


def test_bridge_exit_fails_when_coverage_not_ok() -> None:
    payload = _payload(coverage_ok=False, provider_statuses={})

    verdict = evaluate_bridge_health(payload, strict=False)

    assert verdict["coverage_ok"] is False
    assert verdict["ok"] is False
    assert verdict["exit_code"] == 2
    assert verdict["warning_providers"] == []


def test_bridge_exit_strict_fails_on_any_warning_gap() -> None:
    payload = _payload(coverage_ok=True, provider_statuses={"gemini_cli": "AUTH"})

    verdict = evaluate_bridge_health(payload, strict=True)

    assert verdict["coverage_ok"] is True
    assert verdict["ok"] is False
    assert verdict["exit_code"] == 2
    assert verdict["warning_providers"] == ["gemini_cli"]
