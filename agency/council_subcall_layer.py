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

    filtered: List[str] = []
    for provider in ladder:
        cfg = providers_cfg.get(provider)
        if not isinstance(cfg, dict):
            continue
        if cfg.get("disabled"):
            continue
        if not _provider_supports_role(cfg, provider_role):
            continue
        filtered.append(provider)

    if filtered:
        original_pos = {provider: idx for idx, provider in enumerate(filtered)}
        if mode == "cheap":
            filtered = sorted(
                filtered,
                key=lambda provider: (
                    0 if _is_local_provider(provider, providers_cfg.get(provider, {})) else 1,
                    _tier_priority(mode, _provider_tier(providers_cfg.get(provider, {}))),
                    original_pos.get(provider, 999),
                ),
            )
        else:
            filtered = sorted(
                filtered,
                key=lambda provider: (
                    _tier_priority(mode, _provider_tier(providers_cfg.get(provider, {}))),
                    original_pos.get(provider, 999),
                ),
            )

    if role_c == "auditor" and coder_provider and filtered:
        coder_n = str(coder_provider).strip()
        if filtered[0] == coder_n:
            for idx, provider in enumerate(filtered):
                if provider != coder_n:
                    filtered.insert(0, filtered.pop(idx))
                    break

    model_by_provider: Dict[str, Optional[str]] = {}
    for provider in filtered:
        cfg = providers_cfg.get(provider, {})
        model_by_provider[provider] = _pick_model_for_provider(cfg, mode)

    preferred_provider = filtered[0] if filtered else None
    preferred_model = model_by_provider.get(preferred_provider) if preferred_provider else None
    timeout_seconds = int(timeout_map.get(default_tier) or 20)
    if preferred_provider:
        cfg = providers_cfg.get(preferred_provider, {})
        try:
            timeout_seconds = int(cfg.get("timeout_seconds") or timeout_seconds)
        except Exception:
            pass

    next_hint: List[str] = []
    if not filtered:
        next_hint = [
            f"python bin/ajaxctl doctor providers --roles {provider_role}",
            "python bin/ajaxctl doctor council",
            "revisa config/provider_policy.yaml y config/model_providers.yaml",
        ]

    return {
        "schema": "ajax.council.role_strategy.v1",
        "role": role_c,
        "provider_role": provider_role,
        "mode": mode,
        "cost_mode": mode_to_cost_mode(mode),
        "default_tier": default_tier,
        "preferred_provider": preferred_provider,
        "fallback_providers": filtered[1:] if len(filtered) > 1 else [],
        "provider_ladder": filtered,
        "preferred_model": preferred_model,
        "model_by_provider": model_by_provider,
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

    providers_in_use: List[str] = []
    for role in COUNCIL_EXEC_ROLES:
        strategy = strategies.get(role) or {}
        ladder = strategy.get("provider_ladder") if isinstance(strategy.get("provider_ladder"), list) else []
        for provider in ladder:
            provider_s = str(provider)
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
                }
            )
            continue
        auth_entries.append(_provider_auth_snapshot(provider, cfg))

    missing_auth = [
        item["provider"]
        for item in auth_entries
        if bool(item.get("config_present")) and bool(item.get("auth_required")) and item.get("auth_state") != "present"
    ]
    missing_strategy = [role for role, strategy in strategies.items() if not bool(strategy.get("strategy_ok"))]

    last_by_role: Dict[str, Optional[Dict[str, Any]]] = {}
    for role in COUNCIL_EXEC_ROLES:
        last_by_role[role] = _latest_role_subcall(root, role)

    next_hint: List[str] = []
    if missing_auth:
        for item in auth_entries:
            if item.get("provider") in missing_auth and item.get("auth_env"):
                next_hint.append(f"set {item['auth_env']}=<token>")
    if missing_strategy:
        next_hint.append("revisa config/provider_policy.yaml (rails.<rail>.roles.<role>.preference)")
        next_hint.append("revisa config/model_providers.yaml (providers.<id>.roles)")
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
        },
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
    for role in COUNCIL_EXEC_ROLES:
        strategy = strategies.get(role) if isinstance(strategies.get(role), dict) else {}
        lines.append(
            f"{role}: mode={strategy.get('mode')} provider={strategy.get('preferred_provider') or 'none'} model={strategy.get('preferred_model') or 'none'}"
        )
    auth_doc = payload.get("auth_config") if isinstance(payload.get("auth_config"), dict) else {}
    missing_auth = auth_doc.get("missing_auth_providers") if isinstance(auth_doc.get("missing_auth_providers"), list) else []
    lines.append(f"missing_auth_providers: {len(missing_auth)}")
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

