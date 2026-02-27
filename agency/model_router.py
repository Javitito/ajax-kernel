from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import json

import yaml  # type: ignore


_TIER_ORDER = ["cheap", "balanced", "premium"]
_TIER_CANONICAL = {"cheap": "free", "balanced": "standard", "premium": "premium"}
_COST_MODE_ALIASES = {
    "cheap": "balanced",
    "eco": "balanced",
    "local": "emergency",
    "standard": "balanced",
    "premium_fast": "premium",
}
_TASK_ROLE_MAP = {
    "brain_plan": ("brain", "planner"),
    "brain_chat": ("brain", "chatter"),
    "council_review": ("council", "verifier"),
    "scout": ("scout", "chatter"),
    "vision_plan": ("vision", "planner"),
    "verifier": ("verifier", "verifier"),
}
_TASK_SCORE_DIM = {
    "brain_plan": "planning",
    "brain_chat": "analysis",
    "council_review": "verifier",
    "scout": "analysis",
    "vision_plan": "vision",
    "verifier": "verifier",
}

def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _premium_only_enabled() -> bool:
    raw = os.getenv("AJAX_PREMIUM_ONLY")
    if raw is None:
        return True
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _allow_tier_downgrade_on_emergency() -> bool:
    raw = os.getenv("AJAX_ALLOW_TIER_DOWNGRADE_ON_EMERGENCY")
    if raw is None:
        return False
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_local_provider(cfg: Any) -> bool:
    if not isinstance(cfg, dict):
        return False
    kind = str(cfg.get("kind") or "").strip().lower()
    if kind == "static":
        return True
    base_url = cfg.get("base_url")
    if isinstance(base_url, str):
        url = base_url.strip().lower()
        if "localhost" in url or "127.0.0.1" in url:
            return True
    return False


def _load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("providers", {}) if isinstance(data, dict) else {}


def _load_slots(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(raw) or {}
        return json.loads(raw)
    except Exception:
        return {}


def _load_inventory(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("providers") or []
    except Exception:
        return {}
    inv: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        prov = str(item.get("provider") or "")
        mid = str(item.get("id") or "")
        if not prov or not mid:
            continue
        tier_raw = str(item.get("model_tier") or item.get("tier") or "").strip().lower()
        inv.setdefault(prov, {})
        inv[prov][mid] = {"tier": tier_raw or "balanced"}
    return inv


def _load_ledger(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tier_allows(tier: str, cost_mode: str) -> bool:
    tier = (tier or "balanced").lower()
    cost_mode = _normalize_cost_mode((cost_mode or "premium").lower())
    if cost_mode == "emergency":
        if not _allow_tier_downgrade_on_emergency():
            return tier == "premium"
        return tier in {"cheap", "balanced", "premium"}
    if cost_mode in {"balanced", "save_codex"}:
        return tier in {"cheap", "balanced"}
    return tier in {"cheap", "balanced", "premium"}  # premium mode


def _normalize_cost_mode(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return "premium"
    return _COST_MODE_ALIASES.get(value, value)


def _read_human_active_flag(base_path: Path) -> Optional[bool]:
    paths = [
        base_path / "state" / "human_active.flag",
        base_path / "artifacts" / "state" / "human_active.flag",
        base_path / "artifacts" / "policy" / "human_active.flag",
    ]
    for path in paths:
        try:
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                continue
            if raw.startswith("{"):
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict) and "human_active" in data:
                        return bool(data.get("human_active"))
                except Exception:
                    pass
            lowered = raw.lower()
            if "true" in lowered:
                return True
            if "false" in lowered:
                return False
        except Exception:
            continue
    return None


def _default_cost_mode(base_path: Path) -> str:
    human_active = _read_human_active_flag(base_path)
    if human_active is True:
        return "save_codex"
    return "premium"


def pick_model(
    task_kind: str,
    intent: str | None = None,
    providers_cfg: Dict[str, Any] | None = None,
    cost_mode: str | None = None,
    slot: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Selecciona un provider según el tipo de tarea, roles y tier.
    Respeta `disabled:true` en YAML y `AJAX_COST_MODE` (premium|balanced|emergency).
    Si se proporciona `slot`, usa config/model_slots.* para forzar modelo preferente.
    """
    if providers_cfg is None:
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "model_providers.yaml"
        if not cfg_path.exists():
            raise RuntimeError("model_providers.yaml no encontrado")
        providers_cfg = _load_config(cfg_path)

    base_path = Path(__file__).resolve().parents[1]
    slots_cfg = _load_slots(base_path / "config" / "model_slots.json")
    inventory = _load_inventory(base_path / "config" / "model_inventory_cloud.json")
    ledger_path = base_path / "artifacts" / "provider_ledger" / "latest.json"
    ledger_doc = _load_ledger(ledger_path)
    status_doc = _load_status(base_path / "artifacts" / "health" / "providers_status.json")
    if os.getenv("AJAX_ALLOW_DEGRADED_PLANNING") is not None:
        allow_degraded_planning = _env_truthy("AJAX_ALLOW_DEGRADED_PLANNING")
    else:
        allow_degraded_planning = False
        try:
            from agency.provider_failure_policy import load_provider_failure_policy, planning_allow_degraded_planning

            failure_policy = load_provider_failure_policy(base_path)
            allow_degraded_planning = planning_allow_degraded_planning(failure_policy, default=False)
        except Exception:
            allow_degraded_planning = False
    slot_missing_flag: Dict[str, Any] = {}
    mapping = _TASK_ROLE_MAP.get(task_kind.lower())
    if not mapping:
        raise RuntimeError(f"pick_model: tarea desconocida {task_kind}")
    desired_role, desired_mode = mapping

    cloud_first_tasks = {"brain_plan", "brain_chat", "council_review", "vision_plan"}
    cloud_first = task_kind.lower() in cloud_first_tasks
    raw_cost = (cost_mode or os.getenv("AJAX_COST_MODE") or "").strip()
    if not raw_cost:
        if task_kind.lower() == "vision_plan":
            raw_cost = os.getenv("AJAX_VISION_DEFAULT_COST_MODE") or "eco"
        else:
            raw_cost = _default_cost_mode(base_path)
    requested_cost_mode = raw_cost.strip().lower()
    cost_mode = _normalize_cost_mode(requested_cost_mode)
    allow_local_text = _env_truthy("AJAX_ALLOW_LOCAL_TEXT") or cost_mode == "emergency"
    allow_local_vision = desired_role == "vision"
    allow_local_any = allow_local_text or allow_local_vision
    if cloud_first and cost_mode not in {"premium", "balanced", "emergency", "save_codex"}:
        # Cloud-first estricto: deliberación opera en modo premium (modelos "premium" preferidos).
        cost_mode = "premium"
    # Cargar scores si existen (ledger opcional)
    scores_path = Path(__file__).resolve().parents[1] / "config" / "model_scores_cloud.json"
    scores_data: Dict[str, Any] = {}
    if scores_path.exists():
        try:
            scores_data = yaml.safe_load(scores_path.read_text(encoding="utf-8")) or {}
        except Exception:
            scores_data = {}

    score_dim = _TASK_SCORE_DIM.get(task_kind.lower())
    policy_mode = "premium_first" if task_kind.lower() == "brain_chat" else "success_first"
    decision: Dict[str, Any] = {
        "policy": {
            "mode": policy_mode,
            "budget_mode": cost_mode,
            "requested_budget_mode": requested_cost_mode,
            "premium_rule": None,
            "fallback_reason": None,
            "allow_local_text": bool(allow_local_text),
            "allow_local_vision": bool(allow_local_vision),
            "ledger_path": str(ledger_path),
        },
        "candidates": [],
    }
    candidates: list[tuple[float, int, int, str, Dict[str, Any]]] = []
    pref_order: Dict[str, int] = {}
    policy_doc: Dict[str, Any] = {}
    scoreboard: Dict[str, Any] = {}
    try:
        from agency import provider_policy  # type: ignore

        rail = os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"
        policy_doc = provider_policy.load_provider_policy(base_path)
        if cost_mode == "premium":
            pref_list = provider_policy.preferred_providers(policy_doc, rail=rail, role=desired_role)
            pref_order = {name: idx for idx, name in enumerate(pref_list or []) if isinstance(name, str)}
    except Exception:
        policy_doc = {}
        pref_order = {}
    try:
        from agency import provider_scoreboard  # type: ignore

        scoreboard_path = base_path / "artifacts" / "state" / "provider_scoreboard.json"
        scoreboard = provider_scoreboard.load_scoreboard(scoreboard_path)
    except Exception:
        scoreboard = {}
    if task_kind.lower() == "brain_chat" and cost_mode == "premium" and not pref_order:
        pref_order = {name: idx for idx, name in enumerate(["codex_brain", "gemini_cli", "qwen_cloud", "groq"])}

    policy_providers = policy_doc.get("providers") if isinstance(policy_doc, dict) else None
    policy_providers = policy_providers if isinstance(policy_providers, dict) else {}
    defaults = policy_doc.get("defaults") if isinstance(policy_doc, dict) else {}
    exclude_prefixes: list[str] = []
    if cost_mode == "save_codex":
        if isinstance(defaults, dict):
            exclude_prefixes = defaults.get("save_codex_exclude_prefixes") or []
        if not isinstance(exclude_prefixes, list) or not exclude_prefixes:
            exclude_prefixes = ["codex_"]
        exclude_prefixes = [str(p) for p in exclude_prefixes if str(p)]

    def _policy_state_for_role(provider: str, role: str) -> Optional[str]:
        ent = policy_providers.get(provider)
        if not isinstance(ent, dict):
            return None
        cap = "vision" if role == "vision" else "text"
        state = ent.get(f"policy_state_{cap}") or ent.get("policy_state")
        if isinstance(state, str) and state.strip():
            return state.strip().lower()
        return None

    ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else None
    ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
    ledger_by_provider: Dict[str, Dict[str, Any]] = {}
    for row in ledger_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("role") or "").strip().lower() != desired_role:
            continue
        prov = str(row.get("provider") or "").strip()
        if prov and prov not in ledger_by_provider:
            ledger_by_provider[prov] = row

    def _canonical_tier_label(raw: Optional[str]) -> str:
        label = str(raw or "").strip().lower()
        if label in {"free", "standard", "premium"}:
            return label
        return _TIER_CANONICAL.get(label, _TIER_CANONICAL["balanced"])

    # Intentar slot explícito
    if slot and isinstance(slots_cfg, dict):
        slot_entry = slots_cfg.get(slot) or {}
        preferred = slot_entry.get(cost_mode) or slot_entry.get(cost_mode.lower())
        if cost_mode == "emergency" and not preferred:
            preferred = slot_entry.get("cheap") or slot_entry.get("balanced")
        if preferred and isinstance(preferred, str) and ":" in preferred:
            prov_name, model_id = preferred.split(":", 1)
            if exclude_prefixes and any(prov_name.startswith(prefix) for prefix in exclude_prefixes):
                slot_missing_flag = {
                    "_slot_missing": True,
                    "_slot_requested": preferred,
                    "_slot_blocked": True,
                    "_slot_block_reason": "cost_mode_save_codex",
                }
                cfg = None
            else:
                cfg = providers_cfg.get(prov_name)
            if not isinstance(cfg, dict) or cfg.get("disabled"):
                cfg = None
            if cfg is not None and cloud_first and _is_local_provider(cfg):
                if desired_role != "vision" and not allow_local_text:
                    cfg = None
            if cfg is not None:
                tier = str(cfg.get("tier", "balanced")).lower()
                if _tier_allows(tier, cost_mode):
                    cfg = {**cfg, "_selected_model": model_id}
                    # validar contra inventario; si falta, se buscará fallback
                    if prov_name in inventory and model_id not in inventory.get(prov_name, {}):
                        slot_missing_flag = {"_slot_missing": True, "_slot_requested": preferred}
                    else:
                        ledger_row = ledger_by_provider.get(prov_name)
                        if isinstance(ledger_row, dict) and str(ledger_row.get("status") or "") != "ok":
                            slot_missing_flag = {
                                "_slot_missing": True,
                                "_slot_requested": preferred,
                                "_slot_blocked": True,
                                "_slot_block_reason": ledger_row.get("reason"),
                            }
                        else:
                            policy_state = _policy_state_for_role(prov_name, desired_role)
                            if policy_state in {"disallowed", "blocked", "deny"} and not allow_local_text:
                                slot_missing_flag = {
                                    "_slot_missing": True,
                                    "_slot_requested": preferred,
                                    "_slot_blocked": True,
                                    "_slot_block_reason": "policy_disallowed",
                                }
                            else:
                                cfg["_decision_trace"] = {
                                    "policy": {
                                        "mode": "slot_override",
                                        "budget_mode": cost_mode,
                                        "premium_rule": None,
                                        "allow_local_text": bool(allow_local_text),
                                        "allow_local_vision": bool(allow_local_vision),
                                        "ledger_path": str(ledger_path),
                                    },
                                    "candidates": [],
                                    "selected": {
                                        "provider": prov_name,
                                        "model": model_id,
                                        "tier": _canonical_tier_label(cfg.get("tier")),
                                    },
                                }
                                return prov_name, cfg

    def _evaluate_candidates(
        *,
        include_local: bool,
        premium_only: bool = False,
    ) -> tuple[list[tuple[float, int, int, str, Dict[str, Any]]], list[Dict[str, Any]]]:
        out: list[tuple[float, int, int, str, Dict[str, Any]]] = []
        entries: list[Dict[str, Any]] = []
        providers_status = (status_doc or {}).get("providers") if isinstance(status_doc, dict) else {}
        if not isinstance(providers_status, dict):
            providers_status = {}

        def _reject(entry: Dict[str, Any], code: str, evidence: Optional[str] = None) -> None:
            entry["eligible"] = False
            entry.setdefault("reject_codes", []).append(code)
            if evidence:
                entry["evidence_ref"] = evidence

        for name, cfg_any in providers_cfg.items():
            entry = {
                "provider": name,
                "model": None,
                "tier": _canonical_tier_label((cfg_any or {}).get("tier")),
                "eligible": True,
                "reject_codes": [],
                "supports_premium": bool((cfg_any or {}).get("supports_premium", False)),
                "requires_auth": bool((cfg_any or {}).get("requires_auth", False)),
                "evidence_ref": None,
                "ledger_status": None,
                "ledger_reason": None,
                "cooldown_until": None,
                "cooldown_until_ts": None,
                "policy_state": {"text": _policy_state_for_role(name, "brain"), "vision": _policy_state_for_role(name, "vision")},
            }
            entries.append(entry)
            if exclude_prefixes and any(name.startswith(prefix) for prefix in exclude_prefixes):
                _reject(entry, "COST_MODE_SAVE_CODEX")
                continue
            if not isinstance(cfg_any, dict):
                _reject(entry, "CONFIG_INVALID")
                continue
            st_entry = providers_status.get(name) if isinstance(providers_status, dict) else None
            availability = str((st_entry or {}).get("availability") or "").strip().lower() if isinstance(st_entry, dict) else ""
            if task_kind.lower() == "brain_plan" and availability == "degraded" and not allow_degraded_planning:
                _reject(entry, "AVAILABILITY_DEGRADED", evidence=str(base_path / "artifacts" / "health" / "providers_status.json"))
                continue
            policy_state = _policy_state_for_role(name, desired_role)
            if policy_state in {"disallowed", "blocked", "deny"}:
                if desired_role == "vision":
                    _reject(entry, "POLICY_DISALLOWED")
                    continue
                if not allow_local_text:
                    _reject(entry, "POLICY_DISALLOWED")
                    continue
            if cfg_any.get("disabled"):
                _reject(entry, "DISABLED")
                continue
            if cloud_first and not include_local and _is_local_provider(cfg_any):
                if desired_role == "vision":
                    if not allow_local_vision:
                        _reject(entry, "LOCAL_BLOCKED")
                        continue
                else:
                    _reject(entry, "LOCAL_TEXT_BLOCKED")
                    continue
            tier = str(cfg_any.get("tier", "balanced")).lower()
            if premium_only and tier != "premium":
                _reject(entry, "PREMIUM_ONLY")
                continue
            if not _tier_allows(tier, cost_mode):
                _reject(entry, "TIER_BLOCKED")
                continue
            roles = cfg_any.get("roles") or []
            if not isinstance(roles, list) or desired_role not in [r.lower() for r in roles]:
                _reject(entry, "ROLE_MISMATCH")
                continue
            modes = cfg_any.get("modes")
            if modes:
                modes_l = [m.lower() for m in modes] if isinstance(modes, list) else []
                if desired_mode not in modes_l:
                    _reject(entry, "MODE_MISMATCH")
                    continue
            if not include_local and not cloud_first and _is_local_provider(cfg_any):
                if desired_role == "vision":
                    if not allow_local_vision:
                        _reject(entry, "LOCAL_BLOCKED")
                        continue
                else:
                    _reject(entry, "LOCAL_TEXT_BLOCKED")
                continue
            model_id = cfg_any.get("default_model")
            if not model_id:
                models_map = cfg_any.get("models")
                if isinstance(models_map, dict):
                    model_id = models_map.get("balanced") or models_map.get("fast") or next(iter(models_map.values()), None)
            ledger_key = f"{name}:{model_id}" if model_id else None
            raw_score = 0.0
            if ledger_key and ledger_key in scores_data and isinstance(scores_data[ledger_key], dict):
                s = scores_data[ledger_key].get("scores", {})
                raw_score = float(s.get(score_dim, 0.0)) if isinstance(s, dict) and score_dim else 0.0
            scoreboard_score = None
            if scoreboard and not name.startswith("codex_") and tier in {"cheap", "balanced"}:
                try:
                    from agency import provider_scoreboard  # type: ignore

                    min_samples = int(os.getenv("AJAX_SCOREBOARD_MIN_SAMPLES", "3") or 3)
                    cooldown_minutes = int(os.getenv("AJAX_SCOREBOARD_COOLDOWN_MIN", "15") or 15)
                    scoreboard_state = provider_scoreboard.promotion_state(
                        scoreboard,
                        provider=name,
                        model=model_id,
                        min_samples=min_samples,
                        cooldown_minutes=cooldown_minutes,
                    )
                    entry["scoreboard_state"] = scoreboard_state
                    if scoreboard_state.get("eligible") is False:
                        entry["reorder_decision"] = f"ineligible:{scoreboard_state.get('reason')}"
                    elif scoreboard_state.get("reorder_allowed"):
                        scoreboard_score = provider_scoreboard.score_for(scoreboard, provider=name, model=model_id)
                        entry["reorder_decision"] = "ok"
                    else:
                        entry["reorder_decision"] = scoreboard_state.get("reason")
                except Exception:
                    scoreboard_score = None
            if isinstance(scoreboard_score, (int, float)):
                raw_score += float(scoreboard_score)
                entry["scoreboard_score"] = float(scoreboard_score)
            if inventory and name in inventory and model_id and model_id not in inventory.get(name, {}):
                _reject(entry, "MODEL_NOT_AVAILABLE")
                continue
            model_meta = (inventory.get(name, {}) if inventory else {}).get(model_id or "", {})
            entry["model"] = model_id
            entry["tier"] = _canonical_tier_label(model_meta.get("tier") or tier)
            ledger_row = ledger_by_provider.get(name)
            if isinstance(ledger_row, dict):
                entry["ledger_status"] = ledger_row.get("status")
                entry["ledger_reason"] = ledger_row.get("reason")
                entry["cooldown_until"] = ledger_row.get("cooldown_until")
                entry["cooldown_until_ts"] = ledger_row.get("cooldown_until_ts")
                status = str(ledger_row.get("status") or "")
                reason = str(ledger_row.get("reason") or "")
                cooldown_ts = ledger_row.get("cooldown_until_ts")
                if (
                    status == "soft_fail"
                    and isinstance(cooldown_ts, (int, float))
                    and cooldown_ts > time.time()
                ):
                    _reject(entry, "COOLDOWN_ACTIVE", evidence=str(ledger_path))
                    continue
                if status and status != "ok":
                    if reason in {"quota_exhausted", "429_tpm"}:
                        _reject(entry, "QUOTA_EXHAUSTED", evidence=str(ledger_path))
                    elif reason == "auth":
                        _reject(entry, "AUTH_FAIL", evidence=str(ledger_path))
                    elif reason in {"timeout", "bridge_error"}:
                        _reject(entry, "HEALTH_FAIL", evidence=str(ledger_path))
                    else:
                        _reject(entry, "LEDGER_UNAVAILABLE", evidence=str(ledger_path))
                    continue
            api_key_env = cfg_any.get("api_key_env")
            if entry["requires_auth"] and api_key_env and not os.getenv(api_key_env):
                _reject(entry, "AUTH_REQUIRED", evidence=f"env:{api_key_env}")
                continue
            if cloud_first:
                tier_rank = {"premium": 0, "balanced": 1, "cheap": 2}.get(tier, 1)
            else:
                tier_rank = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else 1
            local_penalty = 1000 if _is_local_provider(cfg_any) and desired_role != "vision" else 0
            pref_rank = (pref_order.get(name, len(pref_order) + 10) if pref_order else 0) + local_penalty
            out.append((raw_score, pref_rank, tier_rank, name, {**cfg_any, "_selected_model": model_id}))
        return out, entries

    include_local_initial = allow_local_any
    if cost_mode == "premium":
        premium_only = _premium_only_enabled()
        candidates, entries = _evaluate_candidates(
            include_local=include_local_initial, premium_only=premium_only
        )
        decision["policy"]["premium_only"] = premium_only
        if not candidates and not premium_only:
            decision["policy"]["fallback_reason"] = "premium_unavailable"
            candidates, entries = _evaluate_candidates(
                include_local=include_local_initial, premium_only=False
            )
        elif not candidates and premium_only:
            decision["policy"]["fallback_reason"] = "premium_only_blocked"
        if not candidates and desired_role == "vision":
            # Vision pulse must remain budget-aware: fallback to eco/local instead of crashing.
            cost_mode = "balanced"
            decision["policy"]["mode"] = "vision_budget_fallback"
            decision["policy"]["budget_mode"] = cost_mode
            decision["policy"]["fallback_reason"] = "vision_no_premium_provider"
            candidates, entries = _evaluate_candidates(
                include_local=include_local_initial, premium_only=False
            )
    elif cost_mode == "emergency":
        premium_only = not _allow_tier_downgrade_on_emergency()
        candidates, entries = _evaluate_candidates(
            include_local=include_local_initial, premium_only=premium_only
        )
        decision["policy"]["premium_only"] = premium_only
        if not candidates and not premium_only:
            decision["policy"]["fallback_reason"] = "emergency_unavailable"
    else:
        candidates, entries = _evaluate_candidates(
            include_local=include_local_initial, premium_only=False
        )
    decision["candidates"] = entries
    if cloud_first and not candidates and allow_local_text:
        # cloud_quorum_failed => permitir local solo con override explícito.
        candidates, entries = _evaluate_candidates(include_local=True, premium_only=False)
        decision["candidates"] = entries
        decision["policy"]["mode"] = "success_first_local_fallback"

    if not candidates:
        raise RuntimeError(f"No hay provider disponible para rol '{desired_role}' con cost_mode={cost_mode}")

    # Orden: preferencia explícita -> score -> tier (para desempatar)
    if pref_order:
        candidates.sort(key=lambda t: (t[1], -t[0], t[2]))
    else:
        candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    decision["ranked_candidates"] = [name for _, __, ___, name, ____ in candidates]
    decision["fallback_chain"] = list(decision["ranked_candidates"])
    _, _, _, name, cfg = candidates[0]
    if slot_missing_flag:
        cfg.update(slot_missing_flag)
        cfg["_slot_selected"] = f"{name}:{cfg.get('_selected_model')}" if cfg.get("_selected_model") else name
    selected_entry = next((entry for entry in decision["candidates"] if entry.get("provider") == name), None)
    decision["selected"] = {
        "provider": name,
        "model": cfg.get("_selected_model"),
        "tier": selected_entry.get("tier") if isinstance(selected_entry, dict) else _canonical_tier_label(cfg.get("tier")),
    }
    cfg["_decision_trace"] = decision
    return name, cfg
