"""
BrainRouter v1.1 — failover exhaustivo entre proveedores Brain.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple


class AllBrainsFailed(RuntimeError):
    def __init__(self, errors: List[str]) -> None:
        super().__init__("all_brains_failed")
        self.errors = errors


class TemporaryBrainFailure(RuntimeError):
    pass


class BrainProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        error_detail: Optional[str] = None,
        stderr_excerpt: Optional[str] = None,
        timeout_s: Optional[int] = None,
        parse_error: bool = False,
        raw_plan_excerpt: Optional[str] = None,
        stage: Optional[str] = None,
        timings: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.error_detail = error_detail or message
        self.stderr_excerpt = stderr_excerpt
        self.timeout_s = timeout_s
        self.parse_error = bool(parse_error)
        self.raw_plan_excerpt = raw_plan_excerpt
        self.stage = stage
        self.timings = timings

class BrainRouter:
    def __init__(self, providers: List[Tuple[str, Dict[str, Any]]], logger: Optional[logging.Logger] = None, ledger: Any = None) -> None:
        self.providers = providers
        self.log = logger or logging.getLogger("brain_router")
        self.ledger = ledger

    @classmethod
    def from_config(cls, config: Dict[str, Any], logger: Optional[logging.Logger] = None, ledger: Any = None) -> "BrainRouter":
        providers_cfg = config.get("providers", {}) if isinstance(config, dict) else {}
        ordered: List[Tuple[str, Dict[str, Any]]] = []
        for name, cfg in providers_cfg.items():
            if cfg.get("disabled"):
                continue
            roles = cfg.get("roles") or []
            if isinstance(roles, list) and any(r.lower() == "brain" for r in roles):
                ordered.append((name, cfg))
        # mantener el orden declarado en YAML
        return cls(ordered, logger=logger, ledger=ledger)

    def _is_transient(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        transient_tokens = [
            "429",
            "rate limit",
            "rate_limit",
            "quota",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "too many requests",
        ]
        return any(tok in msg for tok in transient_tokens)

    def _is_contract_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(tok in msg for tok in ["brain_output no es un dict", "invalid json", "json inválido", "empty_plan"])

    def plan_with_failover(
        self,
        prompt_system: str,
        prompt_user: str,
        meta: Optional[Dict[str, Any]] = None,
        exclude: Optional[set[str]] = None,
        pool: Optional[List[str]] = None,
        max_attempts: Optional[int] = None,
        caller=None,
        attempt_collector: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Itera proveedores brain en orden. caller(name,cfg,system,user,meta)->plan(dict)
        Si todos fallan, lanza AllBrainsFailed con el listado de errores.
        """
        if caller is None:
            raise RuntimeError("BrainRouter requiere un caller para invocar proveedores")
        errors: List[str] = []
        exclude = exclude or set()
        providers = list(self.providers)
        if pool:
            wanted = [str(p) for p in pool if isinstance(p, str) and p.strip()]
            by_name = {n: c for n, c in providers}
            ordered: List[Tuple[str, Dict[str, Any]]] = []
            seen: set[str] = set()
            for n in wanted:
                cfg = by_name.get(n)
                if cfg is None:
                    continue
                ordered.append((n, cfg))
                seen.add(n)
            # Persistencia: si se agota la alineación, aún podemos probar el resto.
            for n, cfg in providers:
                if n in seen:
                    continue
                ordered.append((n, cfg))
            providers = ordered

        attempts_used = 0
        for name, cfg in providers:
            if name in exclude:
                continue
            if isinstance(max_attempts, int) and max_attempts > 0 and attempts_used >= max_attempts:
                break
            try:
                self.log.info("BrainRouter: trying provider=%s", name)
            except Exception:
                pass
            started = time.time()
            try:
                attempts_used += 1
                plan = caller(name, cfg, prompt_system, prompt_user, meta)
                if not isinstance(plan, dict):
                    raise ValueError("brain_output no es un dict")
                
                # Éxito real-time
                if self.ledger:
                    try:
                        self.ledger.record_success(name, "brain")
                    except Exception:
                        pass

                if not plan.get("steps"):
                    # Deterministic contract repair pass (same provider, once):
                    # ask for strict schema and non-empty steps with per-step success_spec.
                    repair_meta = dict(meta or {})
                    if not repair_meta.get("contract_repair_attempted"):
                        repair_meta["contract_repair_attempted"] = True
                        repair_user = (
                            prompt_user.rstrip()
                            + "\n\nReturn plan strictly following the schema; steps must be NON-EMPTY; "
                            + "every step MUST include success_spec (EFE); every step MUST set on_fail='abort'. JSON only. "
                            + "If you cannot produce a plan, return JSON with steps empty and include no_plan_reason."
                        )
                        plan = caller(name, cfg, prompt_system, repair_user, repair_meta)
                        if not isinstance(plan, dict):
                            raise ValueError("brain_output no es un dict")
                    if not plan.get("steps"):
                        raise BrainProviderError(
                            "empty_plan",
                            error_code="empty_plan",
                            error_detail="empty_plan",
                            raw_plan_excerpt=str((plan or {}).get("_raw_plan_excerpt") or "")[:500],
                            stage="normalize",
                        )
                timings = None
                if isinstance(plan, dict):
                    timings = plan.get("_timings") if isinstance(plan.get("_timings"), dict) else None
                    if timings is not None:
                        try:
                            plan.pop("_timings", None)
                        except Exception:
                            pass
                if attempt_collector is not None:
                    attempt_collector.append(
                        {
                            "role": "brain",
                            "id": name,
                            "provider": name,
                            "model": cfg.get("_selected_model") or cfg.get("default_model") or cfg.get("model"),
                            "tier": cfg.get("tier"),
                            "ok": True,
                            "result": "ok",
                            "stage": "parse",
                            "raw_plan_excerpt": (plan.get("_raw_plan_excerpt") if isinstance(plan, dict) else None),
                            "latency_ms": int((time.time() - started) * 1000),
                            "ts": started,
                            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                            "t_start": timings.get("t_start") if isinstance(timings, dict) else None,
                            "t_connect_ok": timings.get("t_connect_ok") if isinstance(timings, dict) else None,
                            "t_first_output": timings.get("t_first_output") if isinstance(timings, dict) else None,
                            "t_last_output": timings.get("t_last_output") if isinstance(timings, dict) else None,
                            "t_end": timings.get("t_end") if isinstance(timings, dict) else None,
                            "timeout_kind": timings.get("timeout_kind") if isinstance(timings, dict) else "NONE",
                            "client_abort": timings.get("client_abort") if isinstance(timings, dict) else None,
                            "exit_code": timings.get("exit_code") if isinstance(timings, dict) else None,
                            "stderr_tail": timings.get("stderr_tail") if isinstance(timings, dict) else None,
                            "bytes_rx": timings.get("bytes_rx") if isinstance(timings, dict) else None,
                            "tokens_rx": timings.get("tokens_rx") if isinstance(timings, dict) else None,
                        }
                    )
                return plan
            except Exception as exc:
                attempts_used += 0
                msg = str(exc)
                error_code = getattr(exc, "error_code", None)
                error_detail = getattr(exc, "error_detail", None)
                stderr_excerpt = getattr(exc, "stderr_excerpt", None)
                timeout_s = getattr(exc, "timeout_s", None)
                parse_error = bool(getattr(exc, "parse_error", False))
                raw_plan_excerpt = getattr(exc, "raw_plan_excerpt", None)
                stage = getattr(exc, "stage", None)
                if not error_code:
                    msg_lower = msg.lower()
                    if msg_lower.startswith("brain_http_"):
                        error_code = msg_lower.split(":", 1)[0].replace("brain_", "")
                    elif msg_lower.startswith("cli_rc_"):
                        error_code = "cli_rc"
                    elif msg_lower.startswith("cli_failed"):
                        error_code = "cli_failed"
                    elif "auth_missing" in msg_lower:
                        error_code = "auth_missing"
                    elif "auth_expired" in msg_lower:
                        error_code = "auth_expired"
                    elif "timeout" in msg_lower:
                        error_code = "timeout"
                    elif "json" in msg_lower or "parse" in msg_lower:
                        error_code = "parse_error"
                    elif "empty_plan" in msg_lower:
                        error_code = "empty_plan"
                    else:
                        error_code = "unknown"
                if not error_detail:
                    error_detail = msg
                timings = getattr(exc, "timings", None)
                if attempt_collector is not None:
                    attempt_collector.append(
                        {
                            "role": "brain",
                            "id": name,
                            "provider": name,
                            "model": cfg.get("_selected_model") or cfg.get("default_model") or cfg.get("model"),
                            "tier": cfg.get("tier"),
                            "ok": False,
                            "result": msg,
                            "error_code": error_code,
                            "error_detail": error_detail,
                            "stderr_excerpt": stderr_excerpt,
                            "timeout_s": timeout_s,
                            "parse_error": parse_error,
                            "raw_plan_excerpt": raw_plan_excerpt,
                            "stage": stage or ("parse" if parse_error else ("normalize" if error_code == "empty_plan" else None)),
                            "latency_ms": int((time.time() - started) * 1000),
                            "transient": self._is_transient(exc) or self._is_contract_error(exc),
                            "ts": started,
                            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                            "t_start": timings.get("t_start") if isinstance(timings, dict) else None,
                            "t_connect_ok": timings.get("t_connect_ok") if isinstance(timings, dict) else None,
                            "t_first_output": timings.get("t_first_output") if isinstance(timings, dict) else None,
                            "t_last_output": timings.get("t_last_output") if isinstance(timings, dict) else None,
                            "t_end": timings.get("t_end") if isinstance(timings, dict) else None,
                            "timeout_kind": timings.get("timeout_kind") if isinstance(timings, dict) else None,
                            "client_abort": timings.get("client_abort") if isinstance(timings, dict) else None,
                            "exit_code": timings.get("exit_code") if isinstance(timings, dict) else None,
                            "stderr_tail": timings.get("stderr_tail") if isinstance(timings, dict) else None,
                            "bytes_rx": timings.get("bytes_rx") if isinstance(timings, dict) else None,
                            "tokens_rx": timings.get("tokens_rx") if isinstance(timings, dict) else None,
                        }
                    )
                if self._is_transient(exc) or self._is_contract_error(exc):
                    # Fallo transitorio real-time
                    if self.ledger:
                        try:
                            # Inferir razón (429, timeout, etc.)
                            reason = "bridge_error"
                            msg_l = str(exc).lower()
                            if "429" in msg_l or "quota" in msg_l or "rate limit" in msg_l:
                                reason = "quota_exhausted"
                            elif "timeout" in msg_l:
                                reason = "timeout"
                            elif "auth" in msg_l:
                                reason = "auth"
                            self.ledger.record_failure(name, "brain", reason, detail=str(exc))
                        except Exception:
                            pass
                    
                    try:
                        self.log.warning("BrainRouter: provider %s failed (%s); switching...", name, msg)
                    except Exception:
                        pass
                    errors.append(f"{name}:{msg}")
                    continue
                # fallo duro: anotar y seguir con el siguiente (no reintentar el mismo)
                errors.append(f"{name}:hard:{msg}")
                continue
        raise AllBrainsFailed(errors or ["no_brain_providers"])
