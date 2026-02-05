from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def load_provider_failure_policy(root_dir: Path, *, policy_path: Optional[Path] = None) -> Dict[str, Any]:
    path = policy_path or (Path(root_dir) / "config" / "provider_failure_policy.yaml")
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def planning_max_attempts(policy: Dict[str, Any], *, default: int = 2) -> int:
    raw = None
    if isinstance(policy, dict):
        planning = policy.get("planning")
        if isinstance(planning, dict):
            raw = planning.get("max_attempts")
    try:
        val = int(raw)
        if val > 0:
            return min(10, val)
    except Exception:
        pass
    return max(1, int(default or 1))


def planning_allow_degraded_planning(policy: Dict[str, Any], *, default: bool = False) -> bool:
    raw = None
    if isinstance(policy, dict):
        planning = policy.get("planning")
        if isinstance(planning, dict):
            raw = planning.get("allow_degraded_planning")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def on_no_plan_terminal(policy: Dict[str, Any], *, default: str = "WAITING_FOR_USER") -> str:
    raw = None
    if isinstance(policy, dict):
        planning = policy.get("planning")
        if isinstance(planning, dict):
            on_no_plan = planning.get("on_no_plan")
            if isinstance(on_no_plan, dict):
                raw = on_no_plan.get("terminal")
    val = str(raw or "").strip().upper()
    if val in {"WAITING_FOR_USER", "ASK_USER", "GAP_LOGGED"}:
        return val
    return str(default or "WAITING_FOR_USER").strip().upper() or "WAITING_FOR_USER"


def force_ask_user_on_severe(policy: Dict[str, Any], *, default: bool = True) -> bool:
    raw = None
    if isinstance(policy, dict):
        planning = policy.get("planning")
        if isinstance(planning, dict):
            on_no_plan = planning.get("on_no_plan")
            if isinstance(on_no_plan, dict):
                raw = on_no_plan.get("force_ask_user_on_severe")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def quota_exhausted_tokens(policy: Dict[str, Any]) -> List[str]:
    default_tokens = [
        "429",
        "rate limit",
        "rate_limit",
        "quota",
        "insufficient_quota",
        "insufficient quota",
        "no capacity",
        "capacity",
        "capacity available",
        "too many requests",
        "quota exhausted",
    ]
    tokens: List[str] = []
    raw_list = None
    if isinstance(policy, dict):
        providers = policy.get("providers")
        if isinstance(providers, dict):
            raw_list = providers.get("classify_as_quota_exhausted")
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, str) and item.strip():
                tokens.append(item.strip().lower())
    if not tokens:
        tokens = [t.strip().lower() for t in default_tokens if isinstance(t, str) and t.strip()]
    out: List[str] = []
    for tok in tokens:
        tok_n = str(tok or "").strip().lower()
        if tok_n and tok_n not in out:
            out.append(tok_n)
    return out


def cooldown_seconds_default(policy: Dict[str, Any], *, default: int = 90) -> int:
    raw = None
    if isinstance(policy, dict):
        providers = policy.get("providers")
        if isinstance(providers, dict):
            raw = providers.get("cooldown_seconds_default")
    try:
        val = int(raw)
        if val > 0:
            return max(30, min(val, 600))
    except Exception:
        pass
    return max(30, min(int(default or 90), 600))


def cooldown_seconds_for_reason(
    policy: Dict[str, Any],
    *,
    reason: Optional[str],
    default: Optional[int] = None,
) -> Optional[int]:
    key = str(reason or "").strip().lower()
    if not key:
        return default
    raw = None
    if isinstance(policy, dict):
        cooldowns = policy.get("cooldowns")
        if isinstance(cooldowns, dict):
            raw = cooldowns.get(key)
    try:
        val = int(raw)
        if val > 0:
            return max(5, min(val, 3600))
    except Exception:
        return default
    return default


def receipt_required_fields(policy: Dict[str, Any]) -> List[str]:
    default_fields = ["schema", "ts_utc", "ok", "reason", "provider", "error_code"]
    raw_list = None
    if isinstance(policy, dict):
        receipts = policy.get("receipts")
        if isinstance(receipts, dict):
            raw_list = receipts.get("required_fields")
    fields: List[str] = []
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, str) and item.strip():
                fields.append(item.strip())
    if not fields:
        fields = list(default_fields)
    out: List[str] = []
    for key in fields:
        key_n = str(key or "").strip()
        if key_n and key_n not in out:
            out.append(key_n)
    return out
