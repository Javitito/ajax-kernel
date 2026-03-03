from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from agency.brain_router import AllBrainsFailed, BrainProviderError, BrainRouter


CLOUD_CFG = {"kind": "cli", "tier": "balanced", "roles": ["brain"]}
LOCAL_CFG = {
    "kind": "http_openai",
    "base_url": "http://localhost:1235/v1",
    "tier": "cheap",
    "roles": ["brain"],
}


def _plan() -> Dict[str, Any]:
    return {"steps": [{"id": "s1", "action": "noop", "success_spec": {"expected_state": {"checks": ["ok"]}}}]}


def _caller_factory(order: List[str], outcomes: Dict[str, Any]):
    def _caller(name, cfg, system, user, meta):  # noqa: ANN001
        order.append(name)
        out = outcomes[name]
        if isinstance(out, Exception):
            raise out
        return out

    return _caller


def test_cloud_success_skips_local_even_if_local_declared_first() -> None:
    providers: List[Tuple[str, Dict[str, Any]]] = [
        ("lmstudio", LOCAL_CFG),
        ("groq", CLOUD_CFG),
        ("qwen_cloud", CLOUD_CFG),
    ]
    router = BrainRouter(providers)
    order: List[str] = []

    plan = router.plan_with_failover(
        prompt_system="s",
        prompt_user="u",
        caller=_caller_factory(order, {"groq": _plan(), "qwen_cloud": _plan(), "lmstudio": _plan()}),
    )

    assert isinstance(plan, dict)
    assert order == ["groq"]


def test_local_only_configuration_executes_local_directly() -> None:
    router = BrainRouter([("lmstudio", LOCAL_CFG)])
    order: List[str] = []

    plan = router.plan_with_failover(
        prompt_system="s",
        prompt_user="u",
        caller=_caller_factory(order, {"lmstudio": _plan()}),
    )

    assert isinstance(plan, dict)
    assert order == ["lmstudio"]


def test_transient_429_tries_alt_cloud_then_local() -> None:
    providers = [
        ("cloud_a", CLOUD_CFG),
        ("cloud_b", CLOUD_CFG),
        ("lmstudio", LOCAL_CFG),
    ]
    router = BrainRouter(providers)
    order: List[str] = []
    outcomes = {
        "cloud_a": BrainProviderError("429 too many requests", error_code="quota_exhausted"),
        "cloud_b": BrainProviderError("timeout", error_code="timeout"),
        "lmstudio": _plan(),
    }

    plan = router.plan_with_failover(
        prompt_system="s",
        prompt_user="u",
        caller=_caller_factory(order, outcomes),
    )

    assert isinstance(plan, dict)
    assert order == ["cloud_a", "cloud_b", "lmstudio"]


def test_timeout_uses_alt_cloud_before_local() -> None:
    providers = [
        ("cloud_a", CLOUD_CFG),
        ("cloud_b", CLOUD_CFG),
        ("lmstudio", LOCAL_CFG),
    ]
    router = BrainRouter(providers)
    order: List[str] = []
    outcomes = {
        "cloud_a": RuntimeError("timeout waiting for output"),
        "cloud_b": _plan(),
        "lmstudio": _plan(),
    }

    plan = router.plan_with_failover(
        prompt_system="s",
        prompt_user="u",
        caller=_caller_factory(order, outcomes),
    )

    assert isinstance(plan, dict)
    assert order == ["cloud_a", "cloud_b"]


def test_provider_lock_violation_triggers_local_fallback() -> None:
    providers = [
        ("cloud_a", CLOUD_CFG),
        ("lmstudio", LOCAL_CFG),
    ]
    router = BrainRouter(providers)
    order: List[str] = []
    outcomes = {
        "cloud_a": RuntimeError("provider_lock_violation"),
        "lmstudio": _plan(),
    }

    plan = router.plan_with_failover(
        prompt_system="s",
        prompt_user="u",
        caller=_caller_factory(order, outcomes),
    )

    assert isinstance(plan, dict)
    assert order == ["cloud_a", "lmstudio"]


def test_local_unavailable_blocks_with_typed_reason() -> None:
    router = BrainRouter(
        [
            ("cloud_a", CLOUD_CFG),
            ("cloud_b", CLOUD_CFG),
        ]
    )
    order: List[str] = []
    outcomes = {
        "cloud_a": RuntimeError("429 quota exhausted"),
        "cloud_b": RuntimeError("timeout"),
    }

    with pytest.raises(AllBrainsFailed) as exc:
        router.plan_with_failover(
            prompt_system="s",
            prompt_user="u",
            caller=_caller_factory(order, outcomes),
        )

    assert order == ["cloud_a", "cloud_b"]
    assert exc.value.reason == "local_fallback_unavailable"


def test_local_fallback_failed_returns_typed_reason() -> None:
    router = BrainRouter(
        [
            ("cloud_a", CLOUD_CFG),
            ("lmstudio", LOCAL_CFG),
        ]
    )
    order: List[str] = []
    outcomes = {
        "cloud_a": RuntimeError("timeout"),
        "lmstudio": RuntimeError("local bridge down"),
    }

    with pytest.raises(AllBrainsFailed) as exc:
        router.plan_with_failover(
            prompt_system="s",
            prompt_user="u",
            caller=_caller_factory(order, outcomes),
        )

    assert order == ["cloud_a", "lmstudio"]
    assert exc.value.reason == "local_fallback_failed"


def test_hard_failure_does_not_trigger_local_fallback() -> None:
    router = BrainRouter(
        [
            ("cloud_a", CLOUD_CFG),
            ("lmstudio", LOCAL_CFG),
        ]
    )
    order: List[str] = []
    outcomes = {
        "cloud_a": RuntimeError("auth_missing"),
        "lmstudio": _plan(),
    }

    with pytest.raises(AllBrainsFailed) as exc:
        router.plan_with_failover(
            prompt_system="s",
            prompt_user="u",
            caller=_caller_factory(order, outcomes),
        )

    assert order == ["cloud_a"]
    assert exc.value.reason == "all_brains_failed"
