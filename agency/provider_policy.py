from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _normalize_rail(raw: Optional[str]) -> str:
    val = (raw or "").strip().lower()
    if val in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def load_provider_policy(root_dir: Path, *, policy_path: Optional[Path] = None) -> Dict[str, Any]:
    path = policy_path or (Path(root_dir) / "config" / "provider_policy.yaml")
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def preferred_providers(policy: Dict[str, Any], *, rail: str, role: str) -> List[str]:
    rail_n = _normalize_rail(rail)
    role_n = (role or "").strip().lower()
    rails = policy.get("rails") if isinstance(policy, dict) else None
    if not isinstance(rails, dict):
        return []
    rail_entry = rails.get(rail_n)
    if not isinstance(rail_entry, dict):
        return []
    roles = rail_entry.get("roles")
    if not isinstance(roles, dict):
        return []
    role_entry = roles.get(role_n)
    if not isinstance(role_entry, dict):
        return []
    pref = role_entry.get("preference") or role_entry.get("preferred") or []
    if isinstance(pref, str):
        pref = [pref]
    if not isinstance(pref, list):
        return []
    out: List[str] = []
    for item in pref:
        if isinstance(item, str) and item.strip():
            pid = item.strip()
            if pid not in out:
                out.append(pid)
    return out


def cost_class(policy: Dict[str, Any], provider: str) -> Optional[str]:
    providers = policy.get("providers") if isinstance(policy, dict) else None
    if not isinstance(providers, dict):
        return None
    ent = providers.get(provider)
    if not isinstance(ent, dict):
        return None
    cc = ent.get("cost_class")
    if isinstance(cc, str) and cc.strip() in {"free", "generous", "paid"}:
        return cc.strip()
    return None


def policy_cooldown_seconds(policy: Dict[str, Any], reason: str) -> Optional[int]:
    defaults = policy.get("defaults") if isinstance(policy, dict) else None
    if not isinstance(defaults, dict):
        return None
    cds = defaults.get("cooldowns")
    if not isinstance(cds, dict):
        return None
    raw = cds.get(reason) or cds.get(str(reason).lower())
    try:
        if raw is None:
            return None
        return max(0, int(raw))
    except Exception:
        return None


def env_rail() -> str:
    return _normalize_rail(os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE"))

