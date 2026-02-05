from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from agency.provider_ledger import ProviderLedger
from agency.provider_failure_policy import load_provider_failure_policy, planning_max_attempts
from agency.provider_policy import env_rail, load_provider_policy, preferred_providers


STRICT_JSON_ONLY = (
    "Output strictly one JSON object and nothing else. "
    "No markdown, no code fences, no extra text. "
    "The output must start with '{' and end with '}'."
)


_SUBCALL_ROLES = {"scout", "reviewer", "summarizer", "validator"}
_TASK_TIERS = {"T0", "T1", "T2"}


def _now_ts() -> float:
    return time.time()


def _ts_slug(ts: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts or _now_ts()))


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _confirmo_premium(prompt: str) -> bool:
    raw = (prompt or "").strip()
    if not raw:
        return False
    return bool(re.search(r"\bconfirmo\s+premium\b", raw, flags=re.IGNORECASE))


def _read_human_active_flag(root_dir: Path) -> Optional[bool]:
    paths = [
        root_dir / "state" / "human_active.flag",
        root_dir / "artifacts" / "state" / "human_active.flag",
        root_dir / "artifacts" / "policy" / "human_active.flag",
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


def _default_cost_mode(root_dir: Path) -> str:
    human_active = _read_human_active_flag(root_dir)
    if human_active is True:
        return "save_codex"
    return "premium"


def _safe_relpath(path: Path, root_dir: Path) -> str:
    try:
        return os.path.relpath(path, root_dir)
    except Exception:
        return str(path)


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    s = s.strip("`").strip()
    if s.lower().startswith("json"):
        s = s[len("json") :].strip()
    return s


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = _strip_code_fences(text).strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


def _extract_codex_jsonl(raw: str) -> Tuple[Optional[str], Optional[str]]:
    last_text: Optional[str] = None
    last_error: Optional[str] = None
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error":
            msg = event.get("message")
            if isinstance(msg, str) and msg.strip():
                last_error = msg.strip()
        if event.get("type") == "item.completed":
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") not in {"agent_message", "assistant_message", "message"}:
                continue
            text = item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                last_text = text.strip()
    return last_text, last_error


def _estimate_tokens(text: str) -> int:
    # Estimador conservador (no depende de tokenizers externos).
    raw = (text or "").strip()
    if not raw:
        return 0
    return max(1, int(len(raw) / 4))


def _load_model_providers(root_dir: Path) -> Dict[str, Any]:
    cfg_yaml_path = root_dir / "config" / "model_providers.yaml"
    cfg_json_path = root_dir / "config" / "model_providers.json"
    data: Any = None
    if yaml is not None and cfg_yaml_path.exists():
        try:
            data = yaml.safe_load(cfg_yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = None
    if data is None and cfg_json_path.exists():
        try:
            data = json.loads(cfg_json_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        return {}
    providers = data.get("providers")
    return providers if isinstance(providers, dict) else {}


def _subcall_role_to_provider_role(role: str) -> str:
    role_n = (role or "").strip().lower()
    if role_n == "scout":
        return "scout"
    if role_n == "reviewer":
        return "council"
    if role_n == "validator":
        return "council"
    return "brain"  # summarizer default


def _compile_subcall_prompts(*, role: str, tier: str, prompt: str, json_mode: bool) -> Tuple[str, str]:
    role_n = (role or "").strip().lower()
    tier_u = (tier or "").strip().upper()
    prompt = str(prompt or "").strip()

    if role_n == "validator":
        system = (
            "You are AJAX Validator.\n"
            "Validate the given input against the requested constraints.\n"
            "Output MUST follow this JSON schema:\n"
            '{"ok": <bool>, "errors": <list[str]>}\n'
            "Set ok=false if any error.\n"
        )
    elif role_n == "reviewer":
        system = (
            "You are AJAX Reviewer.\n"
            "Review the given content for correctness and policy compliance.\n"
            "If output is JSON, use a single object and include a clear decision.\n"
        )
    elif role_n == "summarizer":
        system = (
            "You are AJAX Summarizer.\n"
            "Summarize the given content concisely and accurately.\n"
        )
    else:
        system = (
            "You are AJAX Scout.\n"
            "Explore options and propose candidates; avoid unsafe instructions.\n"
        )

    system = system.rstrip() + f"\nTask tier: {tier_u}\n"
    if json_mode:
        system = system.rstrip() + "\n\n" + STRICT_JSON_ONLY

    user = prompt
    if json_mode and tier_u == "T0" and role_n == "validator" and not user:
        user = "Return a JSON object following the schema."
    return system.strip(), user.strip()


def _validate_validator_schema(obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(obj.get("ok"), bool):
        return "schema_error:missing_or_invalid_ok"
    errors = obj.get("errors")
    if not isinstance(errors, list) or any(not isinstance(e, str) for e in errors):
        return "schema_error:missing_or_invalid_errors"
    return None


@dataclass
class ProviderCallResult:
    text: str
    tokens: int


ProviderCaller = Callable[[str, Dict[str, Any], str, str, bool], ProviderCallResult]


def _default_call_provider(provider: str, cfg: Dict[str, Any], system_prompt: str, user_prompt: str, json_mode: bool) -> ProviderCallResult:
    kind = (cfg.get("kind") or "").strip().lower()
    model_id = cfg.get("_selected_model") or cfg.get("default_model") or cfg.get("model")
    timeout_s = int(cfg.get("timeout_seconds") or 60)

    if kind == "codex_cli_jsonl":
        cmd_template = cfg.get("command") or ["codex", "exec", "--model", "{model}", "--json"]
        cmd: List[str] = []
        for token in cmd_template:
            if token == "{model}":
                cmd.append(str(model_id or ""))
            else:
                cmd.append(str(token))
        if "--skip-git-repo-check" not in cmd:
            cmd.append("--skip-git-repo-check")
        full_prompt = system_prompt.rstrip() + "\n\n" + user_prompt
        proc = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=os.environ.copy(),
        )
        raw = (proc.stdout or "").strip()
        last_text, last_error = _extract_codex_jsonl(raw)
        if proc.returncode != 0:
            err = (last_error or "").strip() or (proc.stderr or "").strip() or raw[:200] or "codex_failed"
            raise RuntimeError(f"codex_rc_{proc.returncode}:{err[:200]}")
        text = last_text or raw
        return ProviderCallResult(text=text.strip(), tokens=_estimate_tokens(full_prompt) + _estimate_tokens(text))

    if kind == "cli":
        cmd_template = cfg.get("command") or []
        if isinstance(cmd_template, str):
            cmd_template = [cmd_template]
        full_prompt = system_prompt.rstrip() + "\n\n" + user_prompt
        cmd: List[str] = []
        use_stdin = False
        i = 0
        while i < len(cmd_template):
            token = cmd_template[i]
            next_token = cmd_template[i + 1] if i + 1 < len(cmd_template) else None
            if token in {"--prompt", "-p", "--prompt-file"} and next_token == "{prompt}":
                use_stdin = True
                i += 2
                continue
            if token == "{prompt}":
                use_stdin = True
                i += 1
                continue
            if token == "{model}":
                cmd.append(str(model_id or ""))
                i += 1
                continue
            cmd.append(str(token))
            i += 1
        if not cmd:
            raise RuntimeError("cli_command_missing")
        proc = subprocess.run(
            cmd,
            input=full_prompt if use_stdin else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or (proc.stdout or "").strip() or f"cli_rc_{proc.returncode}"
            raise RuntimeError(f"cli_rc_{proc.returncode}:{err[:200]}")
        text = (proc.stdout or "").strip()
        return ProviderCallResult(text=text, tokens=_estimate_tokens(full_prompt) + _estimate_tokens(text))

    if kind == "http_openai":
        if requests is None:
            raise RuntimeError("requests_unavailable")
        base_url = (cfg.get("base_url") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("base_url_missing")
        api_key_env = cfg.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0 if json_mode else 0.1,
            "max_tokens": int(cfg.get("max_tokens") or (300 if json_mode else 800)),
        }
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"http_{resp.status_code}:{resp.text[:200]}")
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        content = str(content).strip()
        usage = data.get("usage") if isinstance(data, dict) else None
        total_tokens = None
        if isinstance(usage, dict):
            try:
                total_tokens = int(usage.get("total_tokens"))
            except Exception:
                total_tokens = None
        return ProviderCallResult(
            text=content,
            tokens=int(total_tokens) if isinstance(total_tokens, int) else (_estimate_tokens(system_prompt + user_prompt) + _estimate_tokens(content)),
        )

    raise RuntimeError(f"unsupported_provider_kind:{kind or 'unknown'}")


@dataclass
class SubcallOutcome:
    ok: bool
    terminal: str  # DONE | ASK_USER | GAP_LOGGED
    tier: str
    role: str
    provider_chosen: Optional[str]
    ladder_tried: List[Dict[str, Any]]
    tokens: int
    latency_ms: int
    error: Optional[str]
    output_text: Optional[str]
    output_json: Optional[Dict[str, Any]]
    output_path: Optional[str]
    receipt_path: str


def run_subcall(
    *,
    root_dir: Path,
    role: str,
    tier: str,
    prompt: str,
    json_mode: bool = False,
    read_ledger: bool = True,
    max_attempts: Optional[int] = None,
    human_present: Optional[bool] = None,
    caller: Optional[ProviderCaller] = None,
    allow_premium_subcall: bool = False,
    caller_provider: Optional[str] = None,
    force_provider: Optional[str] = None,
) -> SubcallOutcome:
    """
    DEV_DELEGATION subcall:
    - Route: config/provider_policy.yaml + ProviderLedger (no silent fallback).
    - Attempts: max_attempts providers (default via config/provider_failure_policy.yaml).
    - JSON mode: one repair pass per provider attempt.
    - Always emits receipt to artifacts/receipts/subcall_<ts>.json.
    """
    root_dir = Path(root_dir)
    role_n = (role or "").strip().lower()
    tier_u = (tier or "").strip().upper()
    if role_n not in _SUBCALL_ROLES:
        raise ValueError(f"invalid_role:{role}")
    if tier_u not in _TASK_TIERS:
        raise ValueError(f"invalid_tier:{tier}")

    started = _now_ts()
    ts = _ts_slug(started)
    receipt_dir = root_dir / "artifacts" / "receipts"
    receipt_path = receipt_dir / f"subcall_{ts}.json"
    out_dir = root_dir / "artifacts" / "subcalls"
    out_path: Optional[Path] = None

    if human_present is None:
        try:
            human_present = bool(sys.stdin.isatty() and sys.stdout.isatty())  # type: ignore[name-defined]
        except Exception:
            human_present = False

    caller = caller or _default_call_provider
    provider_role = _subcall_role_to_provider_role(role_n)
    rail = env_rail()
    cost_mode = (os.getenv("AJAX_COST_MODE") or "").strip().lower()
    if not cost_mode:
        cost_mode = _default_cost_mode(root_dir)
    allow_premium_subcall = bool(allow_premium_subcall)
    if _confirmo_premium(prompt):
        allow_premium_subcall = True
    caller_provider = (caller_provider or os.getenv("AJAX_CALLER_PROVIDER") or "").strip()
    if max_attempts is None:
        try:
            failure_policy = load_provider_failure_policy(root_dir)
        except Exception:
            failure_policy = {}
        max_attempts = planning_max_attempts(failure_policy, default=2)
    else:
        max_attempts = max(1, int(max_attempts or 1))

    policy_doc = load_provider_policy(root_dir)
    ladder = preferred_providers(policy_doc, rail=rail, role=provider_role)
    if isinstance(force_provider, str) and force_provider.strip():
        forced = force_provider.strip()
        if forced in ladder:
            ladder = [forced] + [p for p in ladder if p != forced]
        else:
            ladder = [forced] + list(ladder or [])

    providers_cfg = _load_model_providers(root_dir)
    defaults = policy_doc.get("defaults") if isinstance(policy_doc, dict) else {}
    prefixes = []
    if isinstance(defaults, dict):
        prefixes = defaults.get("save_codex_exclude_prefixes") or []
    if not isinstance(prefixes, list) or not prefixes:
        prefixes = ["codex_"]
    prefixes = [str(p) for p in prefixes if str(p)]

    ledger_doc: Dict[str, Any] = {}
    ok_set: set[str] = set()
    availability: Dict[str, Any] = {}
    if read_ledger:
        try:
            ledger = ProviderLedger(root_dir=root_dir)
            ledger_doc = ledger.refresh()
            rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else None
            rows = rows if isinstance(rows, list) else []
            ok_set = set(ProviderLedger.ok_providers(rows, role=provider_role))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("role") or "").strip().lower() != provider_role:
                    continue
                prov = str(row.get("provider") or "").strip()
                if not prov:
                    continue
                availability[prov] = {
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                    "cooldown_until": row.get("cooldown_until"),
                    "cost_class": row.get("cost_class"),
                }
        except Exception:
            ledger_doc = {}
            ok_set = set()

    ladder_tried: List[Dict[str, Any]] = []
    provider_chosen: Optional[str] = None
    total_tokens = 0
    output_text: Optional[str] = None
    output_json: Optional[Dict[str, Any]] = None
    terminal = "GAP_LOGGED"
    error: Optional[str] = None
    blocked = False
    blocked_reason: Optional[str] = None

    def _is_codex(provider: str) -> bool:
        return provider.startswith("codex_")

    def _exclude_by_prefix(provider: str) -> bool:
        for prefix in prefixes:
            if provider.startswith(prefix):
                return True
        return False

    def _eligible(provider: str) -> bool:
        cfg_any = providers_cfg.get(provider)
        if not isinstance(cfg_any, dict):
            return False
        if cfg_any.get("disabled"):
            return False
        if caller_provider and _is_codex(caller_provider) and _is_codex(provider):
            nonlocal blocked, blocked_reason
            blocked = True
            blocked_reason = blocked_reason or "self_call_guard"
            return False
        if cost_mode == "save_codex" and _exclude_by_prefix(provider) and not allow_premium_subcall:
            return False
        if role_n in {"scout", "reviewer", "summarizer"} and not allow_premium_subcall and _is_codex(provider):
            return False
        roles = cfg_any.get("roles") or []
        roles_l = [str(r).strip().lower() for r in roles] if isinstance(roles, list) else []
        if provider_role not in roles_l:
            return False
        tier_cfg = str(cfg_any.get("tier") or "balanced").strip().lower()
        if cost_mode in {"balanced", "save_codex"} and tier_cfg == "premium":
            if cost_mode == "save_codex" and allow_premium_subcall:
                pass
            else:
                return False
        if read_ledger and provider not in ok_set:
            return False
        return True

    eligible_ladder = [p for p in ladder if _eligible(p)]

    receipt_payload: Dict[str, Any] = {
        "schema": "ajax.subcall_receipt.v1",
        "ts": started,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "tier": tier_u,
        "role": role_n,
        "provider_role": provider_role,
        "rail": rail,
        "cost_mode": cost_mode,
        "provider_chosen": None,
        "ladder_tried": [],
        "tokens": 0,
        "latency_ms": 0,
        "ok": False,
        "error": None,
        "output_path": None,
        "blocked": blocked,
        "blocked_reason": blocked_reason,
        "caller_provider": caller_provider or None,
        "premium_allowed": bool(allow_premium_subcall),
        # Multi-model governance (optional fields; never silent):
        "primary_model": None,
        "used_model": None,
        "fallback_reason": None,
        "availability": availability or None,
        "confidence": None,
    }
    receipt_payload["blocked"] = blocked
    receipt_payload["blocked_reason"] = blocked_reason

    try:
        if not ladder:
            error = f"no_policy_ladder:role={provider_role} rail={rail}"
            terminal = "ASK_USER" if human_present else "GAP_LOGGED"
            return _finalize_subcall(
                root_dir=root_dir,
                receipt_path=receipt_path,
                receipt_payload=receipt_payload,
                started=started,
                tier=tier_u,
                role=role_n,
                provider_chosen=None,
                ladder_tried=[],
                tokens=0,
                output_text=None,
                output_json=None,
                output_path=None,
                ok=False,
                terminal=terminal,
                error=error,
            )

        if not eligible_ladder:
            error = f"no_provider_available:role={provider_role} rail={rail}"
            terminal = "ASK_USER" if human_present else "GAP_LOGGED"
            return _finalize_subcall(
                root_dir=root_dir,
                receipt_path=receipt_path,
                receipt_payload=receipt_payload,
                started=started,
                tier=tier_u,
                role=role_n,
                provider_chosen=None,
                ladder_tried=[],
                tokens=0,
                output_text=None,
                output_json=None,
                output_path=None,
                ok=False,
                terminal=terminal,
                error=error,
            )

        primary_provider = eligible_ladder[0]
        receipt_payload["primary_model"] = {"provider_id": primary_provider, "model_id": (providers_cfg.get(primary_provider) or {}).get("default_model")}

        attempts = 0
        for provider in eligible_ladder:
            if attempts >= max(1, int(max_attempts or 1)):
                break
            attempts += 1
            cfg_any = providers_cfg.get(provider)
            cfg = dict(cfg_any) if isinstance(cfg_any, dict) else {}
            if cfg.get("default_model") and not cfg.get("_selected_model"):
                cfg["_selected_model"] = cfg.get("default_model")

            system_prompt, user_prompt = _compile_subcall_prompts(role=role_n, tier=tier_u, prompt=prompt, json_mode=json_mode)
            attempt_started = _now_ts()
            attempt_tokens = 0
            attempt_text: Optional[str] = None
            attempt_error: Optional[str] = None
            repaired = False

            try:
                res = caller(provider, cfg, system_prompt, user_prompt, json_mode)
                attempt_text = res.text
                attempt_tokens = int(res.tokens or 0)
                if json_mode:
                    parsed = _parse_json_object(attempt_text)
                    schema_err = None
                    if parsed is not None and role_n == "validator":
                        schema_err = _validate_validator_schema(parsed)
                    if parsed is None or schema_err:
                        repair_user = (
                            "Return ONLY valid JSON matching the required schema. No prose.\n"
                            "Repair the INVALID_OUTPUT into valid JSON without changing intent.\n\n"
                            f"INVALID_OUTPUT:\n{(attempt_text or '')[:8000]}\n"
                        )
                        repaired = True
                        res2 = caller(provider, cfg, system_prompt, repair_user, True)
                        attempt_text = res2.text
                        attempt_tokens += int(res2.tokens or 0)
                        parsed2 = _parse_json_object(attempt_text)
                        schema_err2 = None
                        if parsed2 is not None and role_n == "validator":
                            schema_err2 = _validate_validator_schema(parsed2)
                        if parsed2 is None or schema_err2:
                            raise RuntimeError(f"json_parse_error:{schema_err2 or 'invalid_json'}:{(attempt_text or '')[:120]}")
                        output_json = parsed2
                        output_text = json.dumps(parsed2, ensure_ascii=False, indent=2) + "\n"
                    else:
                        output_json = parsed
                        output_text = json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"
                else:
                    output_text = str(attempt_text or "").strip() + "\n"
                provider_chosen = provider
                total_tokens += attempt_tokens
                ladder_tried.append(
                    {
                        "provider": provider,
                        "ok": True,
                        "error": None,
                        "latency_ms": int((_now_ts() - attempt_started) * 1000),
                        "tokens": attempt_tokens,
                        "repaired": bool(repaired),
                    }
                )
                break
            except Exception as exc:
                attempt_error = str(exc)
                ladder_tried.append(
                    {
                        "provider": provider,
                        "ok": False,
                        "error": attempt_error[:200],
                        "latency_ms": int((_now_ts() - attempt_started) * 1000),
                        "tokens": attempt_tokens,
                        "repaired": bool(repaired),
                    }
                )
                continue

        ok = bool(provider_chosen and output_text is not None)
        latency_ms = int((_now_ts() - started) * 1000)
        if ok:
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = "json" if json_mode else "txt"
            out_path = out_dir / f"subcall_{ts}.{suffix}"
            out_path.write_text(output_text or "", encoding="utf-8")
            terminal = "DONE"
            error = None
        else:
            terminal = "ASK_USER" if human_present else "GAP_LOGGED"
            error = ladder_tried[-1]["error"] if ladder_tried else "all_attempts_failed"

        fallback_reason = None
        if provider_chosen and primary_provider and provider_chosen != primary_provider:
            fallback_reason = "provider_failover"

        receipt_payload["provider_chosen"] = provider_chosen
        receipt_payload["used_model"] = (
            {"provider_id": provider_chosen, "model_id": (providers_cfg.get(provider_chosen) or {}).get("default_model")}
            if provider_chosen
            else None
        )
        receipt_payload["fallback_reason"] = fallback_reason

        return _finalize_subcall(
            root_dir=root_dir,
            receipt_path=receipt_path,
            receipt_payload=receipt_payload,
            started=started,
            tier=tier_u,
            role=role_n,
            provider_chosen=provider_chosen,
            ladder_tried=ladder_tried,
            tokens=int(total_tokens),
            output_text=output_text,
            output_json=output_json,
            output_path=out_path,
            ok=ok,
            terminal=terminal,
            error=error,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = int((_now_ts() - started) * 1000)
        error = str(exc)
        terminal = "ASK_USER" if human_present else "GAP_LOGGED"
        return _finalize_subcall(
            root_dir=root_dir,
            receipt_path=receipt_path,
            receipt_payload=receipt_payload,
            started=started,
            tier=tier_u,
            role=role_n,
            provider_chosen=provider_chosen,
            ladder_tried=ladder_tried,
            tokens=int(total_tokens),
            output_text=None,
            output_json=None,
            output_path=None,
            ok=False,
            terminal=terminal,
            error=error,
            latency_ms=latency_ms,
        )


def _finalize_subcall(
    *,
    root_dir: Path,
    receipt_path: Path,
    receipt_payload: Dict[str, Any],
    started: float,
    tier: str,
    role: str,
    provider_chosen: Optional[str],
    ladder_tried: List[Dict[str, Any]],
    tokens: int,
    output_text: Optional[str],
    output_json: Optional[Dict[str, Any]],
    output_path: Optional[Path],
    ok: bool,
    terminal: str,
    error: Optional[str],
    latency_ms: Optional[int] = None,
) -> SubcallOutcome:
    if latency_ms is None:
        latency_ms = int((_now_ts() - started) * 1000)
    receipt_payload["provider_chosen"] = provider_chosen
    receipt_payload["ladder_tried"] = list(ladder_tried)
    receipt_payload["tokens"] = int(tokens or 0)
    receipt_payload["latency_ms"] = int(latency_ms)
    receipt_payload["ok"] = bool(ok)
    receipt_payload["terminal"] = str(terminal)
    receipt_payload["error"] = str(error)[:400] if error else None
    receipt_payload["output_path"] = _safe_relpath(output_path, root_dir) if output_path else None

    gap_path: Optional[Path] = None
    if terminal == "GAP_LOGGED" and not ok:
        try:
            ts_slug = receipt_path.stem
            if ts_slug.startswith("subcall_"):
                ts_slug = ts_slug[len("subcall_") :]
            gap_dir = root_dir / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            gap_path = gap_dir / f"subcall_{role}_{tier}_{ts_slug}.json"
            gap_payload: Dict[str, Any] = {
                "capability_family": f"dev_delegation.subcall.{role}",
                "symptom": "subcall_failed",
                "symptoms": ["subcall_failed"],
                "created_at": receipt_payload.get("ts_utc"),
                "evidence_paths": [receipt_payload.get("output_path"), _safe_relpath(receipt_path, root_dir)],
                "detail": {
                    "terminal": terminal,
                    "tier": tier,
                    "role": role,
                    "provider_chosen": provider_chosen,
                    "provider_role": receipt_payload.get("provider_role"),
                    "rail": receipt_payload.get("rail"),
                    "error": receipt_payload.get("error"),
                    "ladder_tried": list(ladder_tried),
                },
            }
            gap_path.write_text(json.dumps(gap_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            receipt_payload["gap_path"] = _safe_relpath(gap_path, root_dir)
        except Exception:
            gap_path = None

    try:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    return SubcallOutcome(
        ok=bool(ok),
        terminal=str(terminal),
        tier=tier,
        role=role,
        provider_chosen=provider_chosen,
        ladder_tried=list(ladder_tried),
        tokens=int(tokens or 0),
        latency_ms=int(latency_ms),
        error=str(error) if error else None,
        output_text=output_text,
        output_json=output_json,
        output_path=_safe_relpath(output_path, root_dir) if output_path else None,
        receipt_path=_safe_relpath(receipt_path, root_dir),
    )
