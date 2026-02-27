from __future__ import annotations

import agency.model_router as model_router


def _patch_io_loaders(monkeypatch) -> None:
    monkeypatch.setattr(model_router, "_load_slots", lambda _path: {}, raising=True)
    monkeypatch.setattr(model_router, "_load_inventory", lambda _path: {}, raising=True)
    monkeypatch.setattr(model_router, "_load_ledger", lambda _path: {}, raising=True)
    monkeypatch.setattr(model_router, "_load_status", lambda _path: {}, raising=True)


def _vision_only_providers() -> dict:
    return {
        "lmstudio_vision": {
            "kind": "http_openai",
            "base_url": "http://localhost:1235/v1",
            "roles": ["vision"],
            "modes": ["planner", "vision", "verifier"],
            "tier": "cheap",
            "default_model": "qwen/qwen3-vl-4b",
        }
    }


def test_pick_model_vision_cheap_alias_routes_without_traceback(monkeypatch) -> None:
    _patch_io_loaders(monkeypatch)

    provider, cfg = model_router.pick_model(
        "vision_plan",
        providers_cfg=_vision_only_providers(),
        cost_mode="cheap",
    )

    assert provider == "lmstudio_vision"
    policy = cfg["_decision_trace"]["policy"]
    assert policy["requested_budget_mode"] == "cheap"
    assert policy["budget_mode"] == "balanced"


def test_pick_model_vision_premium_falls_back_to_balanced(monkeypatch) -> None:
    _patch_io_loaders(monkeypatch)
    monkeypatch.delenv("AJAX_PREMIUM_ONLY", raising=False)

    provider, cfg = model_router.pick_model(
        "vision_plan",
        providers_cfg=_vision_only_providers(),
        cost_mode="premium",
    )

    assert provider == "lmstudio_vision"
    policy = cfg["_decision_trace"]["policy"]
    assert policy["mode"] == "vision_budget_fallback"
    assert policy["fallback_reason"] == "vision_no_premium_provider"
    assert policy["budget_mode"] == "balanced"


def test_pick_model_vision_uses_eco_default_budget(monkeypatch) -> None:
    _patch_io_loaders(monkeypatch)
    monkeypatch.delenv("AJAX_COST_MODE", raising=False)
    monkeypatch.delenv("AJAX_VISION_DEFAULT_COST_MODE", raising=False)

    provider, cfg = model_router.pick_model(
        "vision_plan",
        providers_cfg=_vision_only_providers(),
        cost_mode=None,
    )

    assert provider == "lmstudio_vision"
    policy = cfg["_decision_trace"]["policy"]
    assert policy["requested_budget_mode"] == "eco"
    assert policy["budget_mode"] == "balanced"
