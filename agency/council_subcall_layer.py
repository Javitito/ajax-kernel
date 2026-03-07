from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agency.provider_failure_policy import load_provider_failure_policy, planning_max_attempts
from agency.provider_policy import env_rail, load_provider_policy, preferred_providers

try:
    from agency.auth_provider_diagnostics import collect_provider_auth_diagnostics
except Exception:  # pragma: no cover
    collect_provider_auth_diagnostics = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


COUNCIL_EXEC_ROLES = ("planner", "scout", "coder", "auditor", "judge")
ROLE_ALIASES = {
    "reviewer": "auditor",
    "summarizer": "coder",
    "validator": "judge",
    "survivor": "coder",
}
CLI_SUBCALL_ROLES = (
    "planner",
    "scout",
    "coder",
    "auditor",
    "judge",
    "reviewer",
    "summarizer",
    "validator",
    "survivor",
)
CONSTITUTION_GUARDS = (
    "AGENTS.md",
    "docs/AJAX_SCI_KERNEL.md",
    "docs/AJAX_POLICY_CHALLENGE_LOOP.md",
    "PSEUDOCODE_MAP/",
)

_ROLE_PROFILES: Dict[str, Dict[str, str]] = {
    "planner": {"provider_role": "brain", "mode": "strong", "default_tier": "T2"},
    "scout": {"provider_role": "scout", "mode": "cheap", "default_tier": "T0"},
    "coder": {"provider_role": "brain", "mode": "balanced", "default_tier": "T1"},
    "auditor": {"provider_role": "council", "mode": "balanced", "default_tier": "T1"},
    "judge": {"provider_role": "council", "mode": "strong", "default_tier": "T2"},
}

_MODE_TO_COST_MODE = {
    "cheap": "save_codex",
    "balanced": "balanced",
    "strong": "premium",
}
_ROLE_TIMEOUT_SECONDS = {
    "scout": 6,
    "auditor": 10,
    "coder": 12,
    "planner": 14,
    "judge": 14,
}
_BLOCKED_REASON_CODES = {"auth_missing", "auth_invalid", "auth_expired", "cli_not_installed", "config_missing"}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ts_label() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _read_doc(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    try:
        if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
            data = yaml.safe_load(raw) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_model_providers(root_dir: Path) -> Dict[str, Dict[str, Any]]:
    root = Path(root_dir)
    data = _read_doc(root / "config" / "model_providers.yaml")
    if not data:
        data = _read_doc(root / "config" / "model_providers.json")
    providers = data.get("providers") if isinstance(data, dict) else {}
    if not isinstance(providers, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for provider, cfg in providers.items():
        if isinstance(provider, str) and isinstance(cfg, dict):
            out[provider] = cfg
    return out


def _load_timeout_map(root_dir: Path) -> Dict[str, int]:
    root = Path(root_dir)
    data = _read_doc(root / "config" / "subcall_timeouts.yaml")
    if not data:
        data = _read_doc(root / "config" / "subcall_timeouts.json")
    tiers = data.get("tiers") if isinstance(data.get("tiers"), dict) else data
    if not isinstance(tiers, dict):
        return {}
    out: Dict[str, int] = {}
    for tier in ("T0", "T1", "T2"):
        raw = tiers.get(tier)
        try:
            if raw is None:
                continue
            out[tier] = max(1, int(raw))
        except Exception:
            continue
    return out


def _git_head(root_dir: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            head = str(proc.stdout or "").strip()
            if head:
                return head
    except Exception:
        pass
    return "unknown"


def runtime_identity(root_dir: Path) -> Dict[str, str]:
    root = Path(root_dir).resolve()
    return {
        "ajaxctl_path": str((root / "bin" / "ajaxctl").resolve()),
        "platform": platform.platform(),
        "git_head": _git_head(root),
    }


def canonical_role(role: str) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in _ROLE_PROFILES:
        return normalized
    return ROLE_ALIASES.get(normalized, normalized)


def role_to_provider_role(role: str) -> str:
    role_c = canonical_role(role)
    profile = _ROLE_PROFILES.get(role_c) or {}
    return str(profile.get("provider_role") or "brain")


def role_to_default_tier(role: str) -> str:
    role_c = canonical_role(role)
    profile = _ROLE_PROFILES.get(role_c) or {}
    return str(profile.get("default_tier") or "T1")


def role_to_mode(role: str) -> str:
    role_c = canonical_role(role)
    profile = _ROLE_PROFILES.get(role_c) or {}
    return str(profile.get("mode") or "balanced")


def mode_to_cost_mode(mode: str) -> str:
    return _MODE_TO_COST_MODE.get(str(mode or "").strip().lower(), "balanced")


def _provider_supports_role(cfg: Dict[str, Any], provider_role: str) -> bool:
    roles = cfg.get("roles")
    if not isinstance(roles, list):
        return False
    roles_l = {str(item).strip().lower() for item in roles if isinstance(item, str)}
    return provider_role in roles_l


def _provider_tier(cfg: Dict[str, Any]) -> str:
    tier = str(cfg.get("tier") or "balanced").strip().lower()
    if tier not in {"cheap", "balanced", "premium"}:
        return "balanced"
    return tier


def _is_local_provider(provider: str, cfg: Dict[str, Any]) -> bool:
    name = str(provider or "").strip().lower()
    if any(token in name for token in ("lmstudio", "ollama", "local")):
        return True
    base_url = str(cfg.get("base_url") or "").strip().lower()
    if base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost"):
        return True
    return False


def _tier_priority(mode: str, tier: str) -> int:
    mode_n = str(mode or "balanced").strip().lower()
    tier_n = str(tier or "balanced").strip().lower()
    if mode_n == "cheap":
        return {"cheap": 0, "balanced": 1, "premium": 2}.get(tier_n, 3)
    if mode_n == "strong":
        return {"premium": 0, "balanced": 1, "cheap": 2}.get(tier_n, 3)
    return {"balanced": 0, "cheap": 1, "premium": 2}.get(tier_n, 3)


def _pick_model_for_provider(cfg: Dict[str, Any], mode: str) -> Optional[str]:
    models = cfg.get("models")
    mode_n = str(mode or "balanced").strip().lower()
    key_map = {"cheap": ("fast", "cheap"), "balanced": ("balanced", "default"), "strong": ("smart", "strong")}
    if isinstance(models, dict):
        for key in key_map.get(mode_n, ("balanced",)):
            val = models.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    for key in ("default_model", "model"):
        val = cfg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _fallback_ladder_from_inventory(
    providers_cfg: Dict[str, Dict[str, Any]], *, provider_role: str
) -> List[str]:
    out: List[str] = []
    for provider, cfg in providers_cfg.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("disabled"):
            continue
        if _provider_supports_role(cfg, provider_role):
            out.append(provider)
    return out


def _provider_runtime_map_from_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        provider = str(row.get("provider_name") or "").strip()
        if not provider:
            continue
        out[provider] = dict(row)
    return out


def _latest_auth_runtime_payload(root_dir: Path) -> Dict[str, Any]:
    audits_dir = Path(root_dir).resolve() / "artifacts" / "audits"
    if not audits_dir.exists():
        return {}
    candidates = sorted(
        audits_dir.glob("auth_provider_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}
    payload = _read_doc(candidates[0])
    if not isinstance(payload.get("providers"), list):
        return {}
    return payload


def _live_auth_runtime_payload(root_dir: Path) -> Dict[str, Any]:
    if collect_provider_auth_diagnostics is None:
        return {}
    try:
        return collect_provider_auth_diagnostics(
            root_dir=Path(root_dir).resolve(),
            include_probes=False,
            write_artifact=False,
        )
    except Exception:
        return {}


def _provider_runtime_map(root_dir: Path) -> Dict[str, Dict[str, Any]]:
    cached = _latest_auth_runtime_payload(root_dir)
    if cached:
        return _provider_runtime_map_from_payload(cached)
    live = _live_auth_runtime_payload(root_dir)
    if live:
        return _provider_runtime_map_from_payload(live)
    return {}


def _runtime_quality_rank(row: Dict[str, Any]) -> int:
    effective = str(row.get("effective_result") or "").strip().lower()
    if effective == "usable":
        return 0
    if effective == "degraded":
        return 1
    if effective == "blocked":
        return 3
    return 2


def _role_tier_rank(role: str, tier: str) -> int:
    tier_n = str(tier or "balanced").strip().lower()
    role_c = canonical_role(role)
    if role_c in {"planner", "judge", "coder"}:
        order = {"premium": 0, "balanced": 1, "cheap": 2}
        return order.get(tier_n, 3)
    if role_c == "auditor":
        order = {"balanced": 0, "cheap": 1, "premium": 2}
        return order.get(tier_n, 3)
    return {"cheap": 0, "balanced": 1, "premium": 2}.get(tier_n, 3)


def resolve_effective_role_strategy(
    *,
    role: str,
    ladder: List[str],
    providers_cfg: Dict[str, Dict[str, Any]],
    auth_snapshot: Dict[str, Dict[str, Any]],
    host_capabilities: Optional[Dict[str, Any]] = None,
    coder_provider: Optional[str] = None,
) -> Dict[str, Any]:
    role_c = canonical_role(role)
    host_caps = host_capabilities or {}
    discarded: List[Dict[str, str]] = []
    candidates: List[str] = []
    for provider in ladder:
        row = auth_snapshot.get(provider) or {}
        reason_code = str(row.get("reason_code") or "").strip()
        effective = str(row.get("effective_result") or "").strip().lower()
        if reason_code in _BLOCKED_REASON_CODES or effective == "blocked":
            discarded.append({"provider": provider, "reason_code": reason_code or "blocked"})
            continue
        candidates.append(provider)

    pos = {provider: idx for idx, provider in enumerate(candidates)}

    def _sort_key(provider: str) -> Any:
        cfg = providers_cfg.get(provider, {})
        row = auth_snapshot.get(provider, {})
        runtime_rank = _runtime_quality_rank(row if isinstance(row, dict) else {})
        tier_rank = _role_tier_rank(role_c, _provider_tier(cfg))
        local_rank = 0 if _is_local_provider(provider, cfg) else 1
        base = pos.get(provider, 999)
        if role_c == "scout":
            return (runtime_rank, local_rank, _tier_priority("cheap", _provider_tier(cfg)), base)
        if role_c in {"planner", "judge"}:
            return (runtime_rank, tier_rank, local_rank, base)
        if role_c == "coder":
            return (runtime_rank, tier_rank, local_rank, base)
        return (runtime_rank, tier_rank, local_rank, base)

    ordered = sorted(candidates, key=_sort_key)

    if role_c == "auditor" and coder_provider and ordered:
        coder_n = str(coder_provider).strip()
        if ordered[0] == coder_n:
            for idx, provider in enumerate(ordered):
                if provider != coder_n:
                    ordered.insert(0, ordered.pop(idx))
                    break
        if len(ordered) == 1 and ordered[0] == coder_n:
            discarded.append({"provider": coder_n, "reason_code": "only_provider_usable"})

    availability_by_provider: Dict[str, Dict[str, Any]] = {}
    for provider in ordered:
        row = auth_snapshot.get(provider)
        if not isinstance(row, dict):
            continue
        availability_by_provider[provider] = {
            "effective_result": row.get("effective_result"),
            "reason_code": row.get("reason_code"),
            "reachability_state": row.get("reachability_state"),
            "auth_state": row.get("auth_state"),
            "quota_state": row.get("quota_state"),
        }

    model_by_provider: Dict[str, Optional[str]] = {}
    mode = str(host_caps.get("mode") or role_to_mode(role_c))
    for provider in ordered:
        model_by_provider[provider] = _pick_model_for_provider(providers_cfg.get(provider, {}), mode)

    preferred_provider = ordered[0] if ordered else None
    preferred_model = model_by_provider.get(preferred_provider) if preferred_provider else None
    return {
        "provider_ladder": ordered,
        "preferred_provider": preferred_provider,
        "preferred_model": preferred_model,
        "fallback_providers": ordered[1:] if len(ordered) > 1 else [],
        "model_by_provider": model_by_provider,
        "availability_by_provider": availability_by_provider,
        "discarded_providers": discarded,
    }


def resolve_role_strategy(
    root_dir: Path,
    role: str,
    *,
    rail: Optional[str] = None,
    coder_provider: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(root_dir).resolve()
    role_c = canonical_role(role)
    if role_c not in _ROLE_PROFILES:
        raise ValueError(f"invalid_role:{role}")

    provider_role = role_to_provider_role(role_c)
    mode = role_to_mode(role_c)
    default_tier = role_to_default_tier(role_c)
    rail_n = str(rail or env_rail() or "lab").strip().lower()
    if rail_n not in {"lab", "prod"}:
        rail_n = "lab"

    policy_doc = load_provider_policy(root)
    providers_cfg = _load_model_providers(root)
    timeout_map = _load_timeout_map(root)
    try:
        retries = planning_max_attempts(load_provider_failure_policy(root), default=2)
    except Exception:
        retries = 2
    retries = max(1, int(retries))

    ladder = preferred_providers(policy_doc, rail=rail_n, role=provider_role)
    if not ladder:
        ladder = _fallback_ladder_from_inventory(providers_cfg, provider_role=provider_role)
    else:
        inventory = _fallback_ladder_from_inventory(providers_cfg, provider_role=provider_role)
        for provider in inventory:
            if provider not in ladder:
                ladder.append(provider)

    role_ladder: List[str] = []
    for provider in ladder:
        cfg = providers_cfg.get(provider)
        if not isinstance(cfg, dict):
            continue
        if cfg.get("disabled"):
            continue
        if not _provider_supports_role(cfg, provider_role):
            continue
        role_ladder.append(provider)

    runtime_map = _provider_runtime_map(root)
    effective = resolve_effective_role_strategy(
        role=role_c,
        ladder=role_ladder,
        providers_cfg=providers_cfg,
        auth_snapshot=runtime_map,
        host_capabilities={"mode": mode, "rail": rail_n},
        coder_provider=coder_provider,
    )
    filtered = effective.get("provider_ladder") if isinstance(effective.get("provider_ladder"), list) else []
    filtered = [str(provider) for provider in filtered if str(provider).strip()]
    discarded_providers = (
        effective.get("discarded_providers") if isinstance(effective.get("discarded_providers"), list) else []
    )

    suppressed_providers: List[Dict[str, str]] = [
        {"provider": str(item.get("provider") or ""), "reason_code": str(item.get("reason_code") or "")}
        for item in discarded_providers
        if isinstance(item, dict) and str(item.get("provider") or "").strip()
    ]
    deprioritized_providers: List[Dict[str, str]] = []
    for provider in filtered:
        row = runtime_map.get(provider) or {}
        reason_code = str(row.get("reason_code") or "")
        if reason_code in {"provider_timeout", "provider_down", "quota_exhausted"}:
            deprioritized_providers.append({"provider": provider, "reason_code": reason_code})

    model_by_provider = effective.get("model_by_provider") if isinstance(effective.get("model_by_provider"), dict) else {}
    availability_by_provider = (
        effective.get("availability_by_provider") if isinstance(effective.get("availability_by_provider"), dict) else {}
    )
    preferred_provider = effective.get("preferred_provider")
    preferred_provider = str(preferred_provider).strip() if isinstance(preferred_provider, str) else None
    preferred_model = effective.get("preferred_model")
    preferred_model = str(preferred_model).strip() if isinstance(preferred_model, str) else None

    tier_timeout = int(timeout_map.get(default_tier) or 20)
    role_timeout = int(_ROLE_TIMEOUT_SECONDS.get(role_c) or tier_timeout)
    timeout_seconds = max(1, max(tier_timeout, role_timeout))

    next_hint: List[str] = []
    if not filtered:
        next_hint = [
            f"python bin/ajaxctl doctor providers --roles {provider_role}",
            "python bin/ajaxctl doctor council",
            "python bin/ajaxctl doctor auth",
            "revisa config/provider_policy.yaml y config/model_providers.yaml",
        ]
    if role_c == "auditor" and coder_provider and filtered:
        coder_n = str(coder_provider).strip()
        if len(filtered) == 1 and filtered[0] == coder_n:
            next_hint.append("solo hay un provider usable para auditor; comparte provider con coder")

    return {
        "schema": "ajax.council.role_strategy.v1",
        "role": role_c,
        "provider_role": provider_role,
        "mode": mode,
        "cost_mode": mode_to_cost_mode(mode),
        "default_tier": default_tier,
        "preferred_provider": preferred_provider,
        "fallback_providers": effective.get("fallback_providers") if isinstance(effective.get("fallback_providers"), list) else (filtered[1:] if len(filtered) > 1 else []),
        "provider_ladder": filtered,
        "preferred_model": preferred_model,
        "model_by_provider": model_by_provider,
        "availability_by_provider": availability_by_provider,
        "suppressed_providers": suppressed_providers,
        "discarded_providers": discarded_providers,
        "deprioritized_providers": deprioritized_providers,
        "timeout_seconds": max(1, int(timeout_seconds)),
        "retries": retries,
        "strategy_ok": bool(filtered),
        "next_hint": next_hint,
        "rail": rail_n,
    }


def resolve_strategy_map(root_dir: Path, *, rail: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    root = Path(root_dir).resolve()
    strategies: Dict[str, Dict[str, Any]] = {}
    coder_provider: Optional[str] = None
    for role in COUNCIL_EXEC_ROLES:
        strategy = resolve_role_strategy(root, role, rail=rail, coder_provider=coder_provider)
        if role == "coder":
            coder_provider = str(strategy.get("preferred_provider") or "") or None
        if role == "auditor":
            strategy = resolve_role_strategy(root, role, rail=rail, coder_provider=coder_provider)
        strategies[role] = strategy
    return strategies


def _latest_role_subcall(root_dir: Path, role: str) -> Optional[Dict[str, Any]]:
    target_role = canonical_role(role)
    out_dir = Path(root_dir) / "artifacts" / "subcalls"
    if not out_dir.exists():
        return None
    candidates = sorted(
        out_dir.glob(f"{target_role}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    latest = candidates[0]
    payload = _read_doc(latest)
    if not payload:
        return {"path": str(latest), "error": "invalid_json"}
    payload = dict(payload)
    payload["path"] = str(latest)
    return payload


def _provider_auth_snapshot(provider: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(cfg.get("kind") or "unknown")
    api_key_env = cfg.get("api_key_env")
    api_key_env_name = str(api_key_env).strip() if isinstance(api_key_env, str) and api_key_env.strip() else ""
    auth_required = bool(api_key_env_name)
    auth_present = True
    auth_state = "present"
    if auth_required:
        auth_present = bool(os.getenv(api_key_env_name))
        auth_state = "present" if auth_present else "missing"
    elif kind == "cli":
        infer_cmd = cfg.get("infer_command") or cfg.get("command")
        if not infer_cmd:
            auth_present = False
            auth_state = "missing_cli_command"
    return {
        "provider": provider,
        "kind": kind,
        "auth_required": bool(auth_required),
        "auth_env": api_key_env_name if auth_required else None,
        "auth_state": auth_state,
        "config_present": True,
    }


def run_doctor_council(root_dir: Path) -> Dict[str, Any]:
    root = Path(root_dir).resolve()
    strategies = resolve_strategy_map(root)
    providers_cfg = _load_model_providers(root)
    auth_runtime = _latest_auth_runtime_payload(root)
    if not auth_runtime:
        auth_runtime = _live_auth_runtime_payload(root)
    runtime_map = _provider_runtime_map_from_payload(auth_runtime) if auth_runtime else {}

    providers_in_use: List[str] = []
    for role in COUNCIL_EXEC_ROLES:
        strategy = strategies.get(role) or {}
        ladder = strategy.get("provider_ladder") if isinstance(strategy.get("provider_ladder"), list) else []
        for provider in ladder:
            provider_s = str(provider)
            if provider_s and provider_s not in providers_in_use:
                providers_in_use.append(provider_s)
        suppressed = strategy.get("suppressed_providers") if isinstance(strategy.get("suppressed_providers"), list) else []
        for item in suppressed:
            if not isinstance(item, dict):
                continue
            provider_s = str(item.get("provider") or "").strip()
            if provider_s and provider_s not in providers_in_use:
                providers_in_use.append(provider_s)

    auth_entries: List[Dict[str, Any]] = []
    for provider in providers_in_use:
        cfg = providers_cfg.get(provider)
        if not isinstance(cfg, dict):
            auth_entries.append(
                {
                    "provider": provider,
                    "kind": "unknown",
                    "auth_required": False,
                    "auth_env": None,
                    "auth_state": "provider_missing_in_config",
                    "config_present": False,
                    "reachability_state": "unknown",
                    "quota_state": "unknown",
                    "effective_result": "blocked",
                    "reason_code": "config_missing",
                    "next_hint": "revisa config/model_providers.yaml",
                }
            )
            continue
        base_entry = _provider_auth_snapshot(provider, cfg)
        runtime_row = runtime_map.get(provider) or {}
        if runtime_row:
            base_entry.update(
                {
                    "auth_state": runtime_row.get("auth_state") or base_entry.get("auth_state"),
                    "reachability_state": runtime_row.get("reachability_state") or "unknown",
                    "quota_state": runtime_row.get("quota_state") or "unknown",
                    "effective_result": runtime_row.get("effective_result") or "degraded",
                    "reason_code": runtime_row.get("reason_code") or "unknown_provider_failure",
                    "next_hint": runtime_row.get("next_hint"),
                }
            )
        else:
            base_entry.update(
                {
                    "reachability_state": "unknown",
                    "quota_state": "unknown",
                    "effective_result": "degraded",
                    "reason_code": "unknown_provider_failure",
                }
            )
        auth_entries.append(base_entry)

    missing_auth = [
        item["provider"]
        for item in auth_entries
        if bool(item.get("config_present"))
        and (
            str(item.get("auth_state") or "").strip().lower() in {"missing", "expired", "auth_missing"}
            or str(item.get("reason_code") or "").strip() in {"auth_missing", "auth_invalid", "auth_expired"}
        )
    ]
    missing_strategy = [role for role, strategy in strategies.items() if not bool(strategy.get("strategy_ok"))]
    blocked_providers = [
        item.get("provider")
        for item in auth_entries
        if str(item.get("effective_result") or "").strip().lower() == "blocked"
    ]
    timeout_providers = [
        item.get("provider")
        for item in auth_entries
        if str(item.get("reason_code") or "").strip() == "provider_timeout"
    ]
    blocked_providers = [str(item) for item in blocked_providers if str(item)]
    timeout_providers = [str(item) for item in timeout_providers if str(item)]

    effective_roles: Dict[str, Dict[str, Any]] = {}
    for role in COUNCIL_EXEC_ROLES:
        strategy = strategies.get(role) if isinstance(strategies.get(role), dict) else {}
        fallbacks = strategy.get("fallback_providers") if isinstance(strategy.get("fallback_providers"), list) else []
        discarded = strategy.get("discarded_providers") if isinstance(strategy.get("discarded_providers"), list) else []
        effective_roles[role] = {
            "preferred_provider": strategy.get("preferred_provider"),
            "fallback_providers": [str(item) for item in fallbacks if str(item).strip()],
            "discarded_providers": [
                {
                    "provider": str(item.get("provider") or ""),
                    "reason_code": str(item.get("reason_code") or ""),
                }
                for item in discarded
                if isinstance(item, dict) and str(item.get("provider") or "").strip()
            ],
            "timeout_seconds": int(strategy.get("timeout_seconds") or 0),
            "mode": strategy.get("mode"),
        }

    last_by_role: Dict[str, Optional[Dict[str, Any]]] = {}
    for role in COUNCIL_EXEC_ROLES:
        last_by_role[role] = _latest_role_subcall(root, role)

    next_hint: List[str] = []
    if missing_auth:
        for item in auth_entries:
            if item.get("provider") in missing_auth and item.get("auth_env"):
                next_hint.append(f"set {item['auth_env']}=<token>")
            elif item.get("provider") in missing_auth and item.get("next_hint"):
                next_hint.append(str(item.get("next_hint")))
    if missing_strategy:
        next_hint.append("revisa config/provider_policy.yaml (rails.<rail>.roles.<role>.preference)")
        next_hint.append("revisa config/model_providers.yaml (providers.<id>.roles)")
    if blocked_providers or timeout_providers:
        next_hint.append("python bin/ajaxctl doctor auth")
    for role in COUNCIL_EXEC_ROLES:
        if not last_by_role.get(role):
            next_hint.append(f'python bin/ajaxctl subcall --role {role} "health check {role}"')
    next_hint.append("python bin/ajaxctl council demo")
    # keep hints stable and compact
    dedup_hints: List[str] = []
    for hint in next_hint:
        if hint not in dedup_hints:
            dedup_hints.append(hint)

    executable = not missing_strategy
    payload: Dict[str, Any] = {
        "schema": "ajax.doctor.council.v1",
        "ts_utc": _utc_now(),
        "roles_available": list(COUNCIL_EXEC_ROLES),
        "strategies": strategies,
        "auth_config": {
            "providers": auth_entries,
            "missing_auth_providers": missing_auth,
            "blocked_providers": blocked_providers,
            "timeout_providers": timeout_providers,
        },
        "auth_runtime": {
            "schema": auth_runtime.get("schema"),
            "summary_counts": auth_runtime.get("summary_counts"),
        },
        "effective_roles": effective_roles,
        "last_subcall_by_role": last_by_role,
        "executable": bool(executable),
        "next_hint": dedup_hints,
        "runtime_identity": runtime_identity(root),
    }
    payload["summary"] = format_doctor_council_summary(payload)
    return payload


def format_doctor_council_summary(payload: Dict[str, Any]) -> str:
    lines: List[str] = ["AJAX Doctor council"]
    lines.append(f"executable: {bool(payload.get('executable'))}")
    roles = payload.get("roles_available") if isinstance(payload.get("roles_available"), list) else []
    if roles:
        lines.append("roles: " + ", ".join(str(role) for role in roles))
    strategies = payload.get("strategies") if isinstance(payload.get("strategies"), dict) else {}
    effective_roles = payload.get("effective_roles") if isinstance(payload.get("effective_roles"), dict) else {}
    for role in COUNCIL_EXEC_ROLES:
        strategy = strategies.get(role) if isinstance(strategies.get(role), dict) else {}
        compact = effective_roles.get(role) if isinstance(effective_roles.get(role), dict) else {}
        fallbacks = compact.get("fallback_providers") if isinstance(compact.get("fallback_providers"), list) else []
        discarded = compact.get("discarded_providers") if isinstance(compact.get("discarded_providers"), list) else []
        discarded_tokens: List[str] = []
        for item in discarded:
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider") or "").strip()
            reason = str(item.get("reason_code") or "").strip()
            if provider:
                discarded_tokens.append(f"{provider}:{reason or 'discarded'}")
        timeout_s = compact.get("timeout_seconds")
        lines.append(
            f"{role}: mode={strategy.get('mode')} provider={strategy.get('preferred_provider') or 'none'} model={strategy.get('preferred_model') or 'none'} "
            f"fallbacks=[{','.join(str(item) for item in fallbacks)}] discarded=[{','.join(discarded_tokens)}] timeout_s={timeout_s}"
        )
    auth_doc = payload.get("auth_config") if isinstance(payload.get("auth_config"), dict) else {}
    missing_auth = auth_doc.get("missing_auth_providers") if isinstance(auth_doc.get("missing_auth_providers"), list) else []
    lines.append(f"missing_auth_providers: {len(missing_auth)}")
    blocked_providers = auth_doc.get("blocked_providers") if isinstance(auth_doc.get("blocked_providers"), list) else []
    timeout_providers = auth_doc.get("timeout_providers") if isinstance(auth_doc.get("timeout_providers"), list) else []
    lines.append(f"blocked_providers: {len(blocked_providers)}")
    lines.append(f"timeout_providers: {len(timeout_providers)}")
    last_by_role = payload.get("last_subcall_by_role") if isinstance(payload.get("last_subcall_by_role"), dict) else {}
    for role in COUNCIL_EXEC_ROLES:
        latest = last_by_role.get(role) if isinstance(last_by_role.get(role), dict) else {}
        if latest:
            lines.append(
                f"last_{role}: result={latest.get('result') or latest.get('status') or 'unknown'} provider={latest.get('provider_selected') or 'none'} reason_code={latest.get('reason_code') or 'none'}"
            )
        else:
            lines.append(f"last_{role}: none")
    hints = payload.get("next_hint") if isinstance(payload.get("next_hint"), list) else []
    if hints:
        lines.append("next_hint:")
        for hint in hints:
            lines.append(f"- {hint}")
    return "\n".join(lines)


def run_council_demo(
    root_dir: Path,
    *,
    prompt: Optional[str] = None,
    subcall_runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    root = Path(root_dir).resolve()
    if subcall_runner is None:
        from agency.subcall import run_subcall as _run_subcall  # local import to avoid cycle

        subcall_runner = _run_subcall

    base_prompt = str(prompt or "").strip()
    if not base_prompt:
        base_prompt = (
            "Mini demo textual del concilio: analiza un cambio documental sin tocar codigo y resume riesgos."
        )

    role_prompts = {
        "scout": f"[SCOUT] {base_prompt} Devuelve 3 hechos y 1 riesgo.",
        "coder": f"[CODER] {base_prompt} Propone un plan corto de 3 pasos.",
        "auditor": f"[AUDITOR] {base_prompt} Critica el plan en 2 riesgos verificables.",
        "judge": f"[JUDGE] {base_prompt} Sintetiza decision final en 3 lineas.",
    }

    steps: List[Dict[str, Any]] = []
    for role in ("scout", "coder", "auditor", "judge"):
        strategy = resolve_role_strategy(root, role)
        tier = str(strategy.get("default_tier") or "T1")
        retries = int(strategy.get("retries") or 2)
        provider_ladder = strategy.get("provider_ladder") if isinstance(strategy.get("provider_ladder"), list) else []
        discarded = strategy.get("discarded_providers") if isinstance(strategy.get("discarded_providers"), list) else []
        try:
            outcome = subcall_runner(
                root_dir=root,
                role=role,
                tier=tier,
                prompt=role_prompts[role],
                json_mode=False,
                read_ledger=True,
                max_attempts=retries,
                human_present=False,
            )
            role_ok = bool(getattr(outcome, "ok", False))
            steps.append(
                {
                    "role": role,
                    "ok": role_ok,
                    "terminal": getattr(outcome, "terminal", None),
                    "provider_selected": getattr(outcome, "provider_chosen", None),
                    "reason_code": getattr(outcome, "reason_code", "ok" if role_ok else "subcall_failed"),
                    "next_hint": getattr(outcome, "next_hint", []),
                    "receipt_path": getattr(outcome, "receipt_path", None),
                    "artifact_path": getattr(outcome, "role_artifact_path", None),
                    "ladder_tried": getattr(outcome, "ladder_tried", []),
                    "provider_ladder": [str(item) for item in provider_ladder if str(item).strip()],
                    "discarded_providers": [
                        {
                            "provider": str(item.get("provider") or ""),
                            "reason_code": str(item.get("reason_code") or ""),
                        }
                        for item in discarded
                        if isinstance(item, dict) and str(item.get("provider") or "").strip()
                    ],
                }
            )
        except Exception as exc:
            steps.append(
                {
                    "role": role,
                    "ok": False,
                    "terminal": "GAP_LOGGED",
                    "provider_selected": None,
                    "reason_code": "demo_subcall_exception",
                    "next_hint": [f'python bin/ajaxctl subcall --role {role} "{base_prompt}"'],
                    "receipt_path": None,
                    "artifact_path": None,
                    "error": str(exc)[:240],
                    "ladder_tried": [],
                    "provider_ladder": [str(item) for item in provider_ladder if str(item).strip()],
                    "discarded_providers": [
                        {
                            "provider": str(item.get("provider") or ""),
                            "reason_code": str(item.get("reason_code") or ""),
                        }
                        for item in discarded
                        if isinstance(item, dict) and str(item.get("provider") or "").strip()
                    ],
                }
            )

    ok = all(bool(step.get("ok")) for step in steps)
    payload: Dict[str, Any] = {
        "schema": "ajax.council.demo.v1",
        "ts_utc": _utc_now(),
        "ok": bool(ok),
        "status": "DONE" if ok else "FAIL_CLOSED",
        "reason_code": "ok" if ok else "council_demo_role_failed",
        "steps": steps,
        "runtime_identity": runtime_identity(root),
        "next_hint": [
            "python bin/ajaxctl doctor council",
            'python bin/ajaxctl subcall --role scout "sanity check"',
        ],
    }

    out_dir = root / "artifacts" / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"council_demo_{_ts_label()}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload["artifact_path"] = str(out_path)
    return payload


def constitution_files_touched(paths: List[str]) -> bool:
    for path in paths:
        raw = str(path or "").replace("\\", "/").lstrip("./")
        if raw == "AGENTS.md":
            return True
        if raw in {"docs/AJAX_SCI_KERNEL.md", "docs/AJAX_POLICY_CHALLENGE_LOOP.md"}:
            return True
        if raw.startswith("PSEUDOCODE_MAP/"):
            return True
    return False
