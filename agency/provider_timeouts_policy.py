from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_provider_timeouts_policy(
    root_dir: Path, *, policy_path: Optional[Path] = None
) -> Tuple[Dict[str, Any], Optional[str]]:
    path = policy_path or (Path(root_dir) / "config" / "provider_timeouts_policy.json")
    if not path.exists():
        return {}, None
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return {}, None
    try:
        doc = json.loads(raw)
    except Exception:
        return {}, None
    policy = doc if isinstance(doc, dict) else {}
    try:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except Exception:
        digest = None
    return policy, digest


def policy_version(policy: Dict[str, Any]) -> Optional[str]:
    if not isinstance(policy, dict):
        return None
    val = policy.get("version")
    return str(val) if isinstance(val, str) and val.strip() else None
