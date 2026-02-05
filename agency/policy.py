"""Policy loader for agency routing decisions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:  # Optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

DEFAULT_POLICY_TEXT = """# Default agency routing policy
planner:
  primary: qwen
  fallback:
    - gemini
    - codex
  risk_threshold:
    high: 0.7
    medium: 0.4
council:
  enabled: false
  budget:
    tokens: 1500
    turns: 2
executor:
  primary: qwen
verifier:
  primary: qwen
  critic: gemini
allowlist:
  urls:
    - "https://www.youtube.com/*"
    - "https://music.youtube.com/*"
    - "https://open.spotify.com/*"
sensitive_actions:
  actions:
    - "file.delete"
    - "proc.kill"
  require_confirm: true
supervision:
  timeouts_s:
    planner: 60
    executor: 120
    verifier: 45
    tool: 30
  heartbeat_s: 6
  retries:
    planner: 1
    executor: 2
    verifier: 1
  backoff_s:
    - 2
    - 5
    - 10
  resume_on_restart: true
  circuit_breaker:
    window_s: 60
    failure_threshold: 5
    cool_down_s: 120
routing:
  planner:
    high_risk: "bin/qwen_task.py --role planner --json"
    normal: "bin/qwen_task.py --role planner --json"
  executor: "bin/qwen_task.py --role executor --json"
  verifier:
    primary: "bin/qwen_task.py --role verifier --json"
    fallback: "bin/gemini_task.py --role verifier --json"

council:
  enabled: false

budgets: { steps: 4, seconds: 90, tokens: 6000 }

tool_enforcement: auto
require_verified_for_final: true
min_json_purity: 0.95
min_tool_use_rate: 0.80
"""

DEFAULT_POLICY: Dict[str, Any] = {
    "planner": {
        "primary": "qwen",
        "fallback": ["gemini", "codex"],
        "risk_threshold": {"high": 0.7, "medium": 0.4},
    },
    "council": {"enabled": False, "budget": {"tokens": 1500, "turns": 2}},
    "executor": {"primary": "qwen"},
    "verifier": {"primary": "qwen", "critic": "gemini"},
    "allowlist": {
        "urls": [
            "https://www.youtube.com/*",
            "https://music.youtube.com/*",
            "https://open.spotify.com/*",
        ]
    },
    "sensitive_actions": {
        "actions": ["file.delete", "proc.kill"],
        "require_confirm": True,
    },
    "supervision": {
        "timeouts_s": {"planner": 60, "executor": 120, "verifier": 45, "tool": 30},
        "heartbeat_s": 6,
        "retries": {"planner": 1, "executor": 2, "verifier": 1},
        "backoff_s": [2, 5, 10],
        "resume_on_restart": True,
        "circuit_breaker": {
            "window_s": 60,
            "failure_threshold": 5,
            "cool_down_s": 120,
        },
    },
    "routing": {
        "planner": {
            "high_risk": "bin/qwen_task.py --role planner --json",
            "normal": "bin/qwen_task.py --role planner --json"
        },
        "executor": "bin/qwen_task.py --role executor --json",
        "verifier": {
            "primary": "bin/qwen_task.py --role verifier --json",
            "fallback": "bin/gemini_task.py --role verifier --json"
        }
    },
    "council": {"enabled": False},
    "budgets": {"steps": 4, "seconds": 90, "tokens": 6000},
    "tool_enforcement": "auto",
    "require_verified_for_final": True,
    "min_json_purity": 0.95,
    "min_tool_use_rate": 0.80
}


@dataclass
class PolicySnapshot:
    config: Dict[str, Any]
    text: str
    source: Path


def _parse_policy(text: str) -> Dict[str, Any]:
    if yaml is not None:
        try:
            parsed = yaml.safe_load(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, dict):
            return parsed_json
    except json.JSONDecodeError:
        pass
    return DEFAULT_POLICY


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_policy(path: Path | None = None) -> PolicySnapshot:
    env_path = os.getenv("AGENCY_POLICY_FILE")
    candidate = Path(env_path) if env_path else None
    if path is not None:
        candidate = path
    if candidate and candidate.exists():
        text = candidate.read_text(encoding="utf-8")
        config = _parse_policy(text)
    else:
        text = DEFAULT_POLICY_TEXT
        config = DEFAULT_POLICY.copy()

    # Load policy.overrides.yml if it exists and merge with base config
    overrides_path = Path("policy.overrides.yml")
    if overrides_path.exists():
        overrides_text = overrides_path.read_text(encoding="utf-8")
        overrides_config = _parse_policy(overrides_text)
        config = deep_merge(config, overrides_config)
        # Update text to include overrides (or create new text from merged config)
        text = DEFAULT_POLICY_TEXT  # We'll keep the original text and apply overrides during runtime

    return PolicySnapshot(config=config, text=text, source=candidate or Path("<default>"))


def write_policy(snapshot: PolicySnapshot, path: Path) -> None:
    path.write_text(snapshot.text, encoding="utf-8")


__all__ = ["PolicySnapshot", "load_policy", "write_policy", "DEFAULT_POLICY"]
