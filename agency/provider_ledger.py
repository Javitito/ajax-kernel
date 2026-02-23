from __future__ import annotations

import json
import os
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from agency.auth_manager import AuthManager
except Exception:  # pragma: no cover
    AuthManager = None  # type: ignore

from agency.provider_failure_policy import (
    cooldown_seconds_default as failure_cooldown_seconds_default,
    load_provider_failure_policy,
    quota_exhausted_tokens,
)
from agency.provider_policy import load_provider_policy


_CLI_DEFAULT_MODEL_PROVIDERS = {"codex_brain", "gemini_cli", "qwen_cli"}


def _now_ts() -> float:
    return time.time()


def _iso_utc(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or _now_ts()))


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        return float(raw)
    except Exception:
        return None


def _normalize_reason_from_text(text: str, *, quota_tokens: Optional[Iterable[str]] = None) -> Optional[str]:
    msg = (text or "").strip().lower()
    if not msg:
        return None
    tokens = list(quota_tokens) if quota_tokens else []
    if not tokens:
        tokens = [
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
        ]
    if any(
        tok in msg
        for tok in (str(t).strip().lower() for t in tokens if str(t).strip())
    ):
        return "quota_exhausted"
    if any(tok in msg for tok in ("timeout", "timed out", "read timed out", "etimedout")):
        return "timeout"
    if any(tok in msg for tok in ("401", "403", "auth", "api key", "unauthorized", "forbidden", "expired", "missing")):
        return "auth"
    if any(tok in msg for tok in ("connection", "refused", "dns", "name or service", "not found", "exit 127")):
        return "bridge_error"
    return "bridge_error"


def _default_cooldown_seconds(reason: str) -> int:
    reason = (reason or "").strip().lower()
    if reason in {"quota_exhausted", "429_tpm"}:
        return 90
    if reason == "timeout":
        return 30
    if reason == "bridge_error":
        return 20
    return 60


def _extract_model_flag_from_command(command: Any) -> Optional[str]:
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        tokens = shlex.split(command)
    except Exception:
        tokens = command.strip().split()
    for idx, tok in enumerate(tokens):
        if tok == "--model" and idx + 1 < len(tokens):
            val = str(tokens[idx + 1]).strip()
            return val or None
        if tok.startswith("--model="):
            val = tok.split("=", 1)[1].strip()
            return val or None
    return None


def _observed_model_from_status_entry(entry: Dict[str, Any]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    for candidate in (
        entry.get("transport"),
        ((entry.get("transport") or {}).get("evidence") if isinstance(entry.get("transport"), dict) else None),
    ):
        if isinstance(candidate, dict):
            observed = _extract_model_flag_from_command(candidate.get("command"))
            if observed:
                return observed
    breathing = entry.get("breathing") if isinstance(entry.get("breathing"), dict) else {}
    roles = breathing.get("roles") if isinstance(breathing.get("roles"), dict) else {}
    for role_data in roles.values():
        if not isinstance(role_data, dict):
            continue
        observed = _extract_model_flag_from_command(role_data.get("command"))
        if observed:
            return observed
        evidence = role_data.get("evidence")
        if isinstance(evidence, dict):
            observed = _extract_model_flag_from_command(evidence.get("command"))
            if observed:
                return observed
    return None


def _pick_model_id(provider: str, cfg: Dict[str, Any], *, status_entry: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if not isinstance(cfg, dict):
        return None
    kind = str(cfg.get("kind") or "").strip().lower()
    provider_n = str(provider or "").strip()
    if provider_n in _CLI_DEFAULT_MODEL_PROVIDERS and kind in {"cli", "codex_cli_jsonl"}:
        observed = _observed_model_from_status_entry(status_entry or {})
        return observed or "DEFAULT"
    model_id = cfg.get("_selected_model") or cfg.get("default_model") or cfg.get("model")
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    models = cfg.get("models")
    if isinstance(models, dict):
        for key in ("balanced", "fast", "smart", "premium", "cheap"):
            val = models.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for val in models.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _is_local_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return "localhost" in u or "127.0.0.1" in u


def _default_cost_class(provider: str, cfg: Dict[str, Any]) -> str:
    kind = str(cfg.get("kind") or "").strip().lower()
    if kind == "static":
        return "free"
    base_url = cfg.get("base_url")
    if isinstance(base_url, str) and _is_local_url(base_url):
        return "free"
    # Heurística conservadora: cloud => paid, salvo override de policy.
    return "paid"


def _capabilities_for_cfg(cfg: Dict[str, Any]) -> Dict[str, bool]:
    roles = cfg.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    roles_l = {str(r).strip().lower() for r in roles if r}
    kind = str(cfg.get("kind") or "").strip().lower()
    base_url = cfg.get("base_url")
    is_local = bool(kind == "static")
    if isinstance(base_url, str) and _is_local_url(base_url):
        is_local = True
    return {
        "text_local": bool(is_local and roles_l.intersection({"brain", "council", "scout"})),
        "vision_local": bool(is_local and "vision" in roles_l),
    }


@dataclass
class LedgerRow:
    provider: str
    model: Optional[str]
    role: str
    status: str  # ok | soft_fail | hard_fail
    reason: Optional[str] = None  # quota_exhausted | bridge_error | auth | timeout | ...
    cooldown_until_ts: Optional[float] = None
    last_ok_ts: Optional[float] = None
    last_fail_ts: Optional[float] = None
    cost_class: str = "paid"  # free | generous | paid
    details: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    policy_state_text: Optional[str] = None
    policy_state_vision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_effective": self.model,
            "role": self.role,
            "status": self.status,
            "reason": self.reason,
            "cooldown_until": _iso_utc(self.cooldown_until_ts) if self.cooldown_until_ts else None,
            "cooldown_until_ts": self.cooldown_until_ts,
            "last_ok_ts": self.last_ok_ts,
            "last_fail_ts": self.last_fail_ts,
            "cost_class": self.cost_class,
            "details": dict(self.details),
            "capabilities": dict(self.capabilities),
            "policy_state": {"text": self.policy_state_text, "vision": self.policy_state_vision},
        }


class ProviderLedger:
    """
    Ledger/Budgeter de disponibilidad por provider/model/role.

    Objetivo:
    - No usa LLMs (solo señales locales + probes ligeros).
    - Persiste en artifacts/provider_ledger/latest.json.
    - Regla: status!=ok => no cuenta para quorum.
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        provider_configs: Optional[Dict[str, Any]] = None,
        policy_path: Optional[Path] = None,
        ledger_path: Optional[Path] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.ledger_path = ledger_path or (self.root_dir / "artifacts" / "provider_ledger" / "latest.json")
        self.policy_path = policy_path or (self.root_dir / "config" / "provider_policy.yaml")
        self._provider_configs = provider_configs
        self._failure_policy_doc: Optional[Dict[str, Any]] = None

    def _failure_policy(self) -> Dict[str, Any]:
        if isinstance(self._failure_policy_doc, dict):
            return self._failure_policy_doc
        try:
            self._failure_policy_doc = load_provider_failure_policy(self.root_dir)
        except Exception:
            self._failure_policy_doc = {}
        return self._failure_policy_doc

    def _quota_tokens(self) -> List[str]:
        try:
            return quota_exhausted_tokens(self._failure_policy())
        except Exception:
            return []

    def _load_provider_configs(self) -> Dict[str, Any]:
        if isinstance(self._provider_configs, dict):
            return self._provider_configs
        cfg_yaml_path = self.root_dir / "config" / "model_providers.yaml"
        cfg_json_path = self.root_dir / "config" / "model_providers.json"
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

        return data if isinstance(data, dict) else {"providers": {}}

    def _load_policy(self) -> Dict[str, Any]:
        try:
            doc = load_provider_policy(self.root_dir, policy_path=self.policy_path)
        except Exception:
            doc = {}
        return doc if isinstance(doc, dict) else {}

    def load_latest(self) -> Dict[str, Any]:
        doc = _safe_read_json(self.ledger_path)
        return doc if isinstance(doc, dict) else {}

    def _policy_cost_class(self, policy: Dict[str, Any], provider: str, cfg: Dict[str, Any]) -> str:
        providers = policy.get("providers") if isinstance(policy, dict) else None
        if isinstance(providers, dict):
            ent = providers.get(provider)
            if isinstance(ent, dict):
                cc = ent.get("cost_class")
                if isinstance(cc, str) and cc.strip() in {"free", "generous", "paid"}:
                    return cc.strip()
        return _default_cost_class(provider, cfg)

    def _policy_cooldown_seconds(self, policy: Dict[str, Any], reason: str) -> int:
        defaults = policy.get("defaults") if isinstance(policy, dict) else None
        cooldowns = (defaults or {}).get("cooldowns") if isinstance(defaults, dict) else None
        if isinstance(cooldowns, dict):
            raw = cooldowns.get(reason) or cooldowns.get(str(reason).lower())
            try:
                if raw is not None:
                    return max(0, int(raw))
            except Exception:
                pass
        base = _default_cooldown_seconds(reason)
        reason_n = (reason or "").strip().lower()
        if reason_n in {"quota_exhausted", "429_tpm"}:
            try:
                base = int(failure_cooldown_seconds_default(self._failure_policy(), default=base))
            except Exception:
                base = base
        return base

    def _policy_state(self, policy: Dict[str, Any], provider: str, cap: str) -> Optional[str]:
        providers = policy.get("providers") if isinstance(policy, dict) else None
        if not isinstance(providers, dict):
            return None
        ent = providers.get(provider)
        if not isinstance(ent, dict):
            return None
        key = f"policy_state_{cap}"
        state = ent.get(key)
        if not state:
            state = ent.get("policy_state")
        if isinstance(state, str) and state.strip():
            return state.strip().lower()
        return None

    def _docs_gap_path(self, provider: str) -> Path:
        return self.root_dir / "artifacts" / "capability_gaps" / f"provider_docs_unclear_{provider}.json"

    def _docs_gap_recent(self, path: Path, now: float, ttl_s: int = 3600) -> bool:
        data = _safe_read_json(path)
        last_ts = _parse_float(data.get("last_emitted_ts"))
        if last_ts and (now - last_ts) < ttl_s:
            return True
        return False

    def _build_docs_queries(self, provider: str, reason: str) -> List[str]:
        base = provider.replace("_", " ").strip()
        queries = []
        if reason == "quota_exhausted":
            queries.extend(
                [
                    f"{base} api rate limits",
                    f"{base} api quota limits",
                    f"{base} tokens per minute limit",
                ]
            )
        if reason == "auth":
            queries.extend(
                [
                    f"{base} api key setup",
                    f"{base} authentication error",
                    f"{base} api key missing",
                ]
            )
        queries.append(f"{base} api models list endpoint")
        return queries

    def _maybe_emit_docs_gap(
        self,
        *,
        provider: str,
        reason: str,
        role: str,
        model_id: Optional[str],
        detail: Optional[str],
        prev_row: Dict[str, Any],
        now: float,
    ) -> None:
        if reason not in {"quota_exhausted", "auth"}:
            return
        prev_reason = prev_row.get("reason") if isinstance(prev_row.get("reason"), str) else None
        prev_status = prev_row.get("status") if isinstance(prev_row.get("status"), str) else None
        if not prev_reason or prev_reason != reason:
            return
        if prev_status == "ok":
            return
        gap_path = self._docs_gap_path(provider)
        if self._docs_gap_recent(gap_path, now, ttl_s=3600):
            return
        prev_gap = _safe_read_json(gap_path)
        count = int(prev_gap.get("count") or 0) + 1
        first_seen = prev_gap.get("first_seen_utc") or _iso_utc(now)
        queries = self._build_docs_queries(provider, reason)
        suggested_cmds = [f'python bin/ajaxctl lab web --topic "{q}" --strict' for q in queries]
        payload = {
            "capability_gap_id": f"PROVIDER_DOCS_UNCLEAR_{provider}".upper(),
            "capability_family": "provider_docs_unclear",
            "created_at": _iso_utc(now),
            "provider": provider,
            "role": role,
            "model": model_id,
            "reason": reason,
            "detail": detail,
            "evidence_paths": [str(self.ledger_path)],
            "suggested_queries": queries,
            "suggested_commands": suggested_cmds,
            "notes": "LAB-only web discovery recommended; do not run in PROD.",
            "first_seen_utc": first_seen,
            "last_emitted_utc": _iso_utc(now),
            "last_emitted_ts": now,
            "count": count,
        }
        gap_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_write_json(gap_path, payload)

    def _simulation_override(self) -> Dict[str, Any]:
        raw = os.getenv("AJAX_LEDGER_SIMULATE") or ""
        raw = raw.strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _provider_roles(self, cfg: Dict[str, Any]) -> List[str]:
        roles = cfg.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        out = [str(r).strip().lower() for r in roles if isinstance(r, str) and r.strip()]
        # Si no hay roles declarados, asumir roles básicos para evitar fallos "silenciosos"
        # en configuraciones minimalistas/tests. Vision NO se asume por defecto.
        if not out:
            out = ["brain", "council", "scout"]
        # ledger normaliza "planner/chatter" => roles funcionales, no modos
        return sorted(set(out))

    def _status_doc(self) -> Dict[str, Any]:
        return _safe_read_json(self.root_dir / "artifacts" / "health" / "providers_status.json")

    def _status_entry(self, status_doc: Dict[str, Any], provider: str) -> Dict[str, Any]:
        try:
            entry = ((status_doc or {}).get("providers") or {}).get(provider)
            return entry if isinstance(entry, dict) else {}
        except Exception:
            return {}

    def _reason_from_status_doc(self, status_entry: Dict[str, Any], role: str) -> Optional[str]:
        # 1) Breathing por rol (si existe): señal más fresca que un historial de fallos.
        breathing = status_entry.get("breathing")
        if isinstance(breathing, dict):
            roles = breathing.get("roles")
            if isinstance(roles, dict):
                probe = roles.get(role)
                if isinstance(probe, dict):
                    st = str(probe.get("status") or "").strip().upper()
                    if st == "UP":
                        return None
                    msg = str(probe.get("reason") or "") + " " + str(probe.get("error") or "")
                    reason = _normalize_reason_from_text(msg, quota_tokens=self._quota_tokens())
                    if reason:
                        return reason
        # 2) Eventos recientes (ranker): útil para cooldown (429/etc) cuando no hay breathing.
        events = status_entry.get("events")
        if isinstance(events, list) and events:
            for ev in reversed(events[-15:]):
                if not isinstance(ev, dict):
                    continue
                ok = ev.get("ok")
                if ok is True:
                    return None
                msg = str(ev.get("error") or ev.get("outcome") or "")
                reason = _normalize_reason_from_text(msg, quota_tokens=self._quota_tokens())
                if reason:
                    return reason
        return None

    def _auth_status(self, provider: str, cfg: Dict[str, Any]) -> Optional[str]:
        if AuthManager is None:
            return None
        try:
            auth = AuthManager(root_dir=self.root_dir)
        except Exception:
            return None
        # Mejorar detección para bridges (provider_cli_bridge.py)
        cfg_eff = cfg
        try:
            kind = str(cfg.get("kind") or "").lower()
            cmd = cfg.get("command") or []
            cmd_list = cmd if isinstance(cmd, list) else []
            cmd_str = " ".join(str(x) for x in cmd_list)
            if kind == "cli" and "provider_cli_bridge.py" in cmd_str:
                if provider in {"qwen_cloud", "qwen_cli"}:
                    cfg_eff = {"kind": "cli", "command": ["qwen"]}
                if provider in {"gemini_cli"}:
                    cfg_eff = {"kind": "cli", "command": ["gemini"]}
        except Exception:
            cfg_eff = cfg

        try:
            st = auth.auth_state(provider, cfg_eff)
            if st.state in {"MISSING", "EXPIRED"} and AuthManager.is_web_auth_required(cfg_eff):  # type: ignore[attr-defined]
                # No degradar en silencio: persistir evidencia local y gap accionable.
                try:
                    auth.persist_auth_state(provider, st)
                except Exception:
                    pass
                try:
                    auth.ensure_auth_gap(provider, st)
                except Exception:
                    pass
                return "auth"
        except Exception:
            return None
        return None

    def _probe_http_models(self, provider: str, cfg: Dict[str, Any], timeout_s: float) -> Tuple[bool, Optional[str], str]:
        if requests is None:
            return False, "bridge_error", "requests_missing"
        base_url = str(cfg.get("base_url") or "").rstrip("/")
        if not base_url:
            return False, "bridge_error", "base_url_missing"
        url = f"{base_url}/models"
        api_key_env = cfg.get("api_key_env")
        headers: Dict[str, str] = {}
        if isinstance(api_key_env, str) and api_key_env.strip():
            api_key = os.getenv(api_key_env.strip())
            if not api_key:
                return False, "auth", f"env_missing:{api_key_env.strip()}"
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s)
        except Exception as exc:
            reason = _normalize_reason_from_text(str(exc), quota_tokens=self._quota_tokens()) or "bridge_error"
            if "timeout" in str(exc).lower():
                reason = "timeout"
            return False, reason, str(exc)[:200]
        if resp.status_code == 429:
            return False, "quota_exhausted", (resp.text or "")[:200]
        if resp.status_code in {401, 403}:
            return False, "auth", f"http_{resp.status_code}"
        if resp.status_code >= 400:
            return False, "bridge_error", f"http_{resp.status_code}"
        return True, None, "ok"

    def _probe_cli_presence(self, provider: str, cfg: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
        kind = str(cfg.get("kind") or "").strip().lower()
        cmd = cfg.get("command") or []
        cmd_list = cmd if isinstance(cmd, list) else []

        # Deducción best-effort del binario real (evita invocar LLM via provider_cli_bridge).
        tool = str(cmd_list[0]) if cmd_list else ""
        try:
            if kind == "cli" and cmd_list and "provider_cli_bridge.py" in " ".join(str(x) for x in cmd_list):
                if provider in {"qwen_cloud", "qwen_cli"}:
                    tool = "qwen"
                elif provider == "gemini_cli":
                    tool = "gemini"
        except Exception:
            pass

        if not tool:
            return False, "bridge_error", "cli_command_missing"
        if tool in {"python", "python3", "py"}:
            # Validar que el script exista si aplica
            if any(isinstance(x, str) and x.endswith(".py") for x in cmd_list[1:]):
                for part in cmd_list[1:]:
                    if isinstance(part, str) and part.endswith(".py"):
                        path = (self.root_dir / part).resolve() if not os.path.isabs(part) else Path(part)
                        if not path.exists():
                            return False, "bridge_error", f"script_missing:{part}"
            return True, None, "ok"
        if shutil.which(tool) is None:
            # Por defecto, el ledger no es estricto con la presencia del binario CLI, porque:
            # - algunos entornos usan wrappers/aliases que `which` no ve en tests herméticos,
            # - y el fallo real se detecta por `providers_status.json` cuando se intente invocar.
            #
            # Modo estricto: exporta `AJAX_LEDGER_STRICT_CLI=1`.
            if _env_truthy("AJAX_LEDGER_STRICT_CLI"):
                return False, "bridge_error", f"cli_missing:{tool}"
            return True, None, f"cli_not_verified:{tool}"
        return True, None, "ok"

    def refresh(self, *, timeout_s: float = 2.0) -> Dict[str, Any]:
        policy = self._load_policy()
        providers_cfg = (self._load_provider_configs().get("providers") or {}) if isinstance(self._load_provider_configs(), dict) else {}
        providers_cfg = providers_cfg if isinstance(providers_cfg, dict) else {}

        prev = self.load_latest()
        prev_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in (prev.get("rows") or []):
            if not isinstance(row, dict):
                continue
            prov = str(row.get("provider") or "").strip()
            role = str(row.get("role") or "").strip().lower()
            if prov and role:
                prev_rows[(prov, role)] = row

        status_doc = self._status_doc()
        simulate = self._simulation_override()

        rows: List[LedgerRow] = []
        now = _now_ts()

        for provider, cfg_any in providers_cfg.items():
            if not isinstance(cfg_any, dict):
                continue
            if cfg_any.get("disabled"):
                continue
            kind = str(cfg_any.get("kind") or "").strip().lower()
            if not kind or kind == "static":
                continue

            roles = self._provider_roles(cfg_any)
            if not roles:
                continue
            st_entry = self._status_entry(status_doc, str(provider))
            model_id = _pick_model_id(str(provider), cfg_any, status_entry=st_entry)
            cost_class = self._policy_cost_class(policy, str(provider), cfg_any)
            capabilities = _capabilities_for_cfg(cfg_any)
            policy_state_text = self._policy_state(policy, str(provider), "text")
            policy_state_vision = self._policy_state(policy, str(provider), "vision")

            for role in roles:
                prev_row = prev_rows.get((str(provider), role), {})
                prev_cd = _parse_float(prev_row.get("cooldown_until_ts"))
                prev_reason = prev_row.get("reason") if isinstance(prev_row.get("reason"), str) else None
                prev_status = prev_row.get("status") if isinstance(prev_row.get("status"), str) else None
                last_ok_ts = _parse_float(prev_row.get("last_ok_ts"))
                last_fail_ts = _parse_float(prev_row.get("last_fail_ts"))

                if prev_cd and now < prev_cd and prev_status and prev_status != "ok":
                    rows.append(
                        LedgerRow(
                            provider=str(provider),
                            model=model_id,
                            role=role,
                            status=str(prev_status),
                            reason=str(prev_reason) if prev_reason else None,
                            cooldown_until_ts=prev_cd,
                            last_ok_ts=last_ok_ts,
                            last_fail_ts=last_fail_ts,
                            cost_class=cost_class,
                            details={"source": "cooldown"},
                            capabilities=capabilities,
                            policy_state_text=policy_state_text,
                            policy_state_vision=policy_state_vision,
                        )
                    )
                    continue

                # --- Simulation override (tests / acceptance)
                sim_ent = simulate.get(str(provider))
                if isinstance(sim_ent, dict):
                    pick = sim_ent.get(role) or sim_ent.get("*")
                    if isinstance(pick, dict):
                        sim_status = str(pick.get("status") or "").strip()
                        sim_reason = pick.get("reason")
                        reason = str(sim_reason).strip() if isinstance(sim_reason, str) and sim_reason.strip() else None
                        status = sim_status if sim_status in {"ok", "soft_fail", "hard_fail"} else "hard_fail"
                        cd_until = None
                        if status != "ok" and reason in {"quota_exhausted", "429_tpm", "timeout", "bridge_error"}:
                            cd_until = now + float(self._policy_cooldown_seconds(policy, reason))
                        rows.append(
                            LedgerRow(
                                provider=str(provider),
                                model=model_id,
                                role=role,
                                status=status,
                                reason=reason,
                                cooldown_until_ts=cd_until,
                                last_ok_ts=last_ok_ts,
                                last_fail_ts=now if status != "ok" else last_fail_ts,
                                cost_class=cost_class,
                                details={"source": "simulate"},
                                capabilities=capabilities,
                                policy_state_text=policy_state_text,
                                policy_state_vision=policy_state_vision,
                            )
                        )
                        continue

                # --- Auth gate (no network)
                auth_reason = self._auth_status(str(provider), cfg_any)
                if auth_reason == "auth":
                    rows.append(
                        LedgerRow(
                            provider=str(provider),
                            model=model_id,
                            role=role,
                            status="hard_fail",
                            reason="auth",
                            cooldown_until_ts=None,
                            last_ok_ts=last_ok_ts,
                            last_fail_ts=now,
                            cost_class=cost_class,
                            details={"source": "auth"},
                            capabilities=capabilities,
                            policy_state_text=policy_state_text,
                            policy_state_vision=policy_state_vision,
                        )
                    )
                    self._maybe_emit_docs_gap(
                        provider=str(provider),
                        reason="auth",
                        role=role,
                        model_id=model_id,
                        detail="auth_gate",
                        prev_row=prev_row,
                        now=now,
                    )
                    continue

                # --- Local status signals (no network)
                inferred = self._reason_from_status_doc(st_entry, role)
                if inferred in {"quota_exhausted", "timeout", "bridge_error"}:
                    cd_until = now + float(self._policy_cooldown_seconds(policy, inferred))
                    rows.append(
                        LedgerRow(
                            provider=str(provider),
                            model=model_id,
                            role=role,
                            status="soft_fail",
                            reason=inferred,
                            cooldown_until_ts=cd_until,
                            last_ok_ts=last_ok_ts,
                            last_fail_ts=now,
                            cost_class=cost_class,
                            details={"source": "providers_status"},
                            capabilities=capabilities,
                            policy_state_text=policy_state_text,
                            policy_state_vision=policy_state_vision,
                        )
                    )
                    self._maybe_emit_docs_gap(
                        provider=str(provider),
                        reason=inferred,
                        role=role,
                        model_id=model_id,
                        detail="providers_status",
                        prev_row=prev_row,
                        now=now,
                    )
                    continue
                if inferred == "auth":
                    rows.append(
                        LedgerRow(
                            provider=str(provider),
                            model=model_id,
                            role=role,
                            status="hard_fail",
                            reason="auth",
                            cooldown_until_ts=None,
                            last_ok_ts=last_ok_ts,
                            last_fail_ts=now,
                            cost_class=cost_class,
                            details={"source": "providers_status"},
                            capabilities=capabilities,
                            policy_state_text=policy_state_text,
                            policy_state_vision=policy_state_vision,
                        )
                    )
                    self._maybe_emit_docs_gap(
                        provider=str(provider),
                        reason="auth",
                        role=role,
                        model_id=model_id,
                        detail="providers_status",
                        prev_row=prev_row,
                        now=now,
                    )
                    continue

                # --- Lightweight probe
                ok = False
                reason = None
                detail = ""
                if kind == "http_openai":
                    base_url = cfg_any.get("base_url")
                    base_url_s = str(base_url).strip() if isinstance(base_url, str) else ""
                    # Por defecto evitamos probes remotos (red) para no introducir dependencia azarosa del proveedor.
                    # Se permite probe HTTP solo si es local (localhost/127.0.0.1) o si se fuerza explícitamente.
                    probe_remote = _env_truthy("AJAX_LEDGER_PROBE_REMOTE_HTTP")
                    if not base_url_s:
                        ok, reason, detail = True, None, "probe_skipped:base_url_missing"
                    elif _is_local_url(base_url_s) or probe_remote:
                        ok, reason, detail = self._probe_http_models(str(provider), cfg_any, timeout_s=timeout_s)
                    else:
                        ok, reason, detail = True, None, "probe_skipped:remote_http_disabled"
                else:
                    ok, reason, detail = self._probe_cli_presence(str(provider), cfg_any)
                if ok:
                    rows.append(
                        LedgerRow(
                            provider=str(provider),
                            model=model_id,
                            role=role,
                            status="ok",
                            reason=None,
                            cooldown_until_ts=None,
                            last_ok_ts=now,
                            last_fail_ts=last_fail_ts,
                            cost_class=cost_class,
                            details={"source": "probe", "detail": detail},
                            capabilities=capabilities,
                            policy_state_text=policy_state_text,
                            policy_state_vision=policy_state_vision,
                        )
                    )
                    continue

                status = "soft_fail"
                cooldown_until_ts = None
                if reason == "auth":
                    status = "hard_fail"
                elif reason == "bridge_error":
                    # cli_missing/script_missing => hard_fail; remote connection => soft_fail
                    if "missing" in detail.lower() or "cli_missing" in detail.lower() or "script_missing" in detail.lower():
                        status = "hard_fail"
                    else:
                        status = "soft_fail"
                else:
                    status = "soft_fail"

                if status == "soft_fail" and reason in {"quota_exhausted", "429_tpm", "timeout", "bridge_error"}:
                    cooldown_until_ts = now + float(self._policy_cooldown_seconds(policy, reason))
                rows.append(
                    LedgerRow(
                        provider=str(provider),
                        model=model_id,
                        role=role,
                        status=status,
                        reason=reason,
                        cooldown_until_ts=cooldown_until_ts,
                        last_ok_ts=last_ok_ts,
                        last_fail_ts=now,
                        cost_class=cost_class,
                        details={"source": "probe", "detail": detail},
                        capabilities=capabilities,
                        policy_state_text=policy_state_text,
                        policy_state_vision=policy_state_vision,
                    )
                )
                if reason:
                    self._maybe_emit_docs_gap(
                        provider=str(provider),
                        reason=str(reason),
                        role=role,
                        model_id=model_id,
                        detail=detail,
                        prev_row=prev_row,
                        now=now,
                    )

        doc = {
            "schema": "ajax.provider_ledger.v1",
            "updated_ts": now,
            "updated_utc": _iso_utc(now),
            "path": str(self.ledger_path),
            "rows": [r.to_dict() for r in rows],
        }
        _safe_write_json(self.ledger_path, doc)
        return doc

    @staticmethod
    def ok_providers(rows: Iterable[Dict[str, Any]], *, role: str) -> List[str]:
        role_n = (role or "").strip().lower()
        out: List[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("role") or "").strip().lower() != role_n:
                continue
            if str(row.get("status") or "") != "ok":
                continue
            prov = str(row.get("provider") or "").strip()
            if prov and prov not in out:
                out.append(prov)
        return out

    def record_failure(self, provider: str, role: str, reason: str, detail: str = "") -> None:
        """Registrar un fallo en tiempo real y aplicar cooldown."""
        now = _now_ts()
        policy = self._load_policy()
        cooldown_s = float(self._policy_cooldown_seconds(policy, reason))
        
        doc = self.load_latest()
        rows = doc.get("rows", [])
        updated = False
        
        for row in rows:
            if row.get("provider") == provider and row.get("role") == role:
                row["status"] = "soft_fail" if reason != "auth" else "hard_fail"
                row["reason"] = reason
                row["last_fail_ts"] = now
                row["cooldown_until_ts"] = now + cooldown_s
                row["details"]["source"] = "realtime_failure"
                row["details"]["detail"] = detail[:200]
                updated = True
                break
        
        if updated:
            doc["updated_ts"] = now
            doc["updated_utc"] = _iso_utc(now)
            _safe_write_json(self.ledger_path, doc)

    def record_success(self, provider: str, role: str) -> None:
        """Registrar un éxito en tiempo real y limpiar cooldown."""
        now = _now_ts()
        doc = self.load_latest()
        rows = doc.get("rows", [])
        updated = False
        
        for row in rows:
            if row.get("provider") == provider and row.get("role") == role:
                row["status"] = "ok"
                row["reason"] = None
                row["last_ok_ts"] = now
                row["cooldown_until_ts"] = None
                row["details"]["source"] = "realtime_success"
                updated = True
                break
        
        if updated:
            doc["updated_ts"] = now
            doc["updated_utc"] = _iso_utc(now)
            _safe_write_json(self.ledger_path, doc)
