from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

DEFAULT_POLICY = {
    "alert_level": "normal",
    "ask_user_timeout_seconds": 60,
    "ask_user_on_timeout": "abort",
    "allow_app_launch": True,
    "max_app_launches_per_mission": 3,
    "per_app_limits": {},
    "driver_request_timeout_seconds": 5,
    "driver_failure_window_seconds": 60,
    "driver_failure_threshold": 3,
    "driver_recovery_cooldown_seconds": 20,
}

VALID_ALERT_LEVELS = {"cautious", "normal", "bold"}
VALID_ON_TIMEOUT = {"abort", "log_only", "future_fallback", "safe_autonomous"}


def _policy_path(root_dir: Path) -> Path:
    return root_dir / "config" / "security_policy.yaml"


def load_security_policy(root_dir: Path) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    path = _policy_path(root_dir)
    if not path.exists() or yaml is None:
        return policy
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            policy.update(data)
    except Exception:
        return policy
    # Normalizar valores
    lvl = str(policy.get("alert_level") or "").lower()
    policy["alert_level"] = lvl if lvl in VALID_ALERT_LEVELS else DEFAULT_POLICY["alert_level"]
    on_timeout = str(policy.get("ask_user_on_timeout") or "").lower()
    policy["ask_user_on_timeout"] = on_timeout if on_timeout in VALID_ON_TIMEOUT else DEFAULT_POLICY["ask_user_on_timeout"]
    try:
        policy["ask_user_timeout_seconds"] = int(policy.get("ask_user_timeout_seconds", DEFAULT_POLICY["ask_user_timeout_seconds"]))
    except Exception:
        policy["ask_user_timeout_seconds"] = DEFAULT_POLICY["ask_user_timeout_seconds"]
    policy["allow_app_launch"] = bool(policy.get("allow_app_launch", DEFAULT_POLICY["allow_app_launch"]))
    try:
        policy["max_app_launches_per_mission"] = int(policy.get("max_app_launches_per_mission", DEFAULT_POLICY["max_app_launches_per_mission"]))
    except Exception:
        policy["max_app_launches_per_mission"] = DEFAULT_POLICY["max_app_launches_per_mission"]
    per_limits = policy.get("per_app_limits") or {}
    policy["per_app_limits"] = per_limits if isinstance(per_limits, dict) else {}
    try:
        policy["driver_request_timeout_seconds"] = float(policy.get("driver_request_timeout_seconds", DEFAULT_POLICY["driver_request_timeout_seconds"]))
    except Exception:
        policy["driver_request_timeout_seconds"] = DEFAULT_POLICY["driver_request_timeout_seconds"]
    try:
        policy["driver_failure_window_seconds"] = float(policy.get("driver_failure_window_seconds", DEFAULT_POLICY["driver_failure_window_seconds"]))
    except Exception:
        policy["driver_failure_window_seconds"] = DEFAULT_POLICY["driver_failure_window_seconds"]
    try:
        policy["driver_failure_threshold"] = int(policy.get("driver_failure_threshold", DEFAULT_POLICY["driver_failure_threshold"]))
    except Exception:
        policy["driver_failure_threshold"] = DEFAULT_POLICY["driver_failure_threshold"]
    try:
        policy["driver_recovery_cooldown_seconds"] = float(policy.get("driver_recovery_cooldown_seconds", DEFAULT_POLICY["driver_recovery_cooldown_seconds"]))
    except Exception:
        policy["driver_recovery_cooldown_seconds"] = DEFAULT_POLICY["driver_recovery_cooldown_seconds"]
    return policy


def get_alert_level(root_dir: Path) -> str:
    return load_security_policy(root_dir).get("alert_level", DEFAULT_POLICY["alert_level"])


def policy_snapshot(root_dir: Path) -> str:
    """
    Devuelve un JSON legible con la política vigente (útil para logs).
    """
    try:
        return json.dumps(load_security_policy(root_dir), ensure_ascii=False)
    except Exception:
        return json.dumps(DEFAULT_POLICY, ensure_ascii=False)
