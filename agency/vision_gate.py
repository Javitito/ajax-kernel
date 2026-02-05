from __future__ import annotations

import os

_ENV_KEY = "VISION_LOCAL_ALLOWED"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_DENY = "vision_local_disabled"


def is_local_vision_allowed() -> bool:
    """
    Returns True only when VISION_LOCAL_ALLOWED is explicitly set to a truthy value.
    """
    val = os.getenv(_ENV_KEY)
    if val is None:
        return False
    return val.strip().lower() in _TRUE_VALUES


def ensure_local_vision_allowed(context: str = "") -> None:
    """
    Raises PermissionError if local vision routines are not allowed.
    """
    if not is_local_vision_allowed():
        suffix = f":{context}" if context else ""
        raise PermissionError(f"{_DENY}{suffix} (set {_ENV_KEY}=1 to enable)")
