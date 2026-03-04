from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_ENV_KEY = "VISION_LOCAL_ALLOWED"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_DENY = "vision_local_disabled"
_CLI_HINT = "python bin/ajaxctl vision tag-screen --allow-local"
_PROCESS_ALLOW_LOCAL = False


def set_local_vision_allowed_for_process(enabled: bool = True) -> None:
    global _PROCESS_ALLOW_LOCAL
    _PROCESS_ALLOW_LOCAL = bool(enabled)


def is_local_vision_allowed(*, allow_local: bool = False) -> bool:
    """
    Returns True only when VISION_LOCAL_ALLOWED is explicitly set to a truthy value.
    """
    if bool(allow_local) or bool(_PROCESS_ALLOW_LOCAL):
        return True
    val = os.getenv(_ENV_KEY)
    if val is None:
        return False
    return val.strip().lower() in _TRUE_VALUES


def ensure_local_vision_allowed(context: str = "", *, allow_local: bool = False) -> None:
    """
    Raises PermissionError if local vision routines are not allowed.
    """
    if not is_local_vision_allowed(allow_local=allow_local):
        suffix = f":{context}" if context else ""
        raise PermissionError(
            f"{_DENY}{suffix} (next_hint: set {_ENV_KEY}=true or run `{_CLI_HINT}`)"
        )


def _providers_status_path(root_dir: Optional[Path] = None) -> Path:
    root = Path(root_dir) if root_dir is not None else Path.cwd()
    return root / "artifacts" / "health" / "providers_status.json"


def _load_providers_status(root_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = _providers_status_path(root_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _provider_vision_up(provider_row: Dict[str, Any]) -> tuple[bool, str]:
    transport = provider_row.get("transport") if isinstance(provider_row.get("transport"), dict) else {}
    breathing = provider_row.get("breathing") if isinstance(provider_row.get("breathing"), dict) else {}
    roles = breathing.get("roles") if isinstance(breathing.get("roles"), dict) else {}
    role_vision = roles.get("vision") if isinstance(roles.get("vision"), dict) else {}

    transport_status = str(transport.get("status") or "").upper()
    role_status = str(role_vision.get("status") or breathing.get("status") or "").upper()
    if transport_status in {"UP", "DEGRADED"} and role_status in {"UP", "DEGRADED"}:
        return True, ""
    reason = (
        str(provider_row.get("unavailable_reason") or "")
        or str(role_vision.get("reason") or "")
        or str(transport.get("reason") or "")
        or "provider_not_up"
    )
    return False, reason


def select_local_vision_provider(
    *,
    root_dir: Optional[Path] = None,
    providers_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status_doc = providers_status if isinstance(providers_status, dict) else _load_providers_status(root_dir)
    providers = status_doc.get("providers") if isinstance(status_doc.get("providers"), dict) else {}
    preferred_order = ["lmstudio_vision"]
    for name, row in providers.items():
        if name in preferred_order:
            continue
        caps = row.get("capabilities") if isinstance(row, dict) else {}
        if isinstance(caps, dict) and bool(caps.get("vision_local")):
            preferred_order.append(str(name))

    for name in preferred_order:
        row = providers.get(name) if isinstance(providers.get(name), dict) else None
        if not isinstance(row, dict):
            continue
        up, reason = _provider_vision_up(row)
        if up:
            return {
                "provider": name,
                "up": True,
                "status": "UP",
                "reason": "",
            }
        if name == "lmstudio_vision":
            return {
                "provider": name,
                "up": False,
                "status": "DOWN",
                "reason": reason,
            }
    return {
        "provider": None,
        "up": False,
        "status": "DOWN",
        "reason": "no_local_vision_provider",
    }
