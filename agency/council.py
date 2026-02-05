from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import requests

try:
    from agency.model_router import pick_model
except ImportError:  # pragma: no cover
    pick_model = None  # type: ignore

from agency.method_pack import AJAX_METHOD_PACK


@dataclass
class CouncilVerdict:
    """
    Representa la decisión del Council sobre un plan.

    El JSON de salida del modelo DEBE tener:
      - verdict: "approve" | "needs_revision" | "reject"
      - reason: explicación breve (una línea)
      - suggested_fix: sugerencia breve (opcional)
      - escalation_hint: "await_user_input" | "try_stronger_model" | "unsafe"
    """
    verdict: str
    reason: str
    suggested_fix: Optional[str] = None
    escalation_hint: Optional[str] = None
    debug_notes: Optional[str] = None
    council_degraded: Optional[bool] = None
    council_degraded_reason: Optional[str] = None
    providers_tried: Optional[str] = None

    @property
    def approved(self) -> bool:
        return self.verdict == "approve"

    @property
    def feedback(self) -> str:
        # alias para el código legado que espera .feedback
        return self.reason


@dataclass
class CouncilProviderDecision:
    provider: str
    decision: str  # "pass" | "fail" | "unavailable"
    reasons: str = ""
    verdict: Optional[CouncilVerdict] = None
    latency_ms: Optional[int] = None


@dataclass
class CouncilConsensusResult:
    provider_results: List[CouncilProviderDecision] = field(default_factory=list)
    final_decision: str = "veto"  # "approved" | "veto" | "blocked"
    final_verdict: str = "reject"  # "approve" | "needs_revision" | "reject"
    reason: str = ""
    degraded: bool = False
    degraded_reason: Optional[str] = None
    escalation_hint: Optional[str] = None
    suggested_fix: Optional[str] = None
    debug_notes: Optional[str] = None

    @property
    def approved(self) -> bool:
        return self.final_decision == "approved"

    def to_verdict(self) -> CouncilVerdict:
        verdict = self.final_verdict if self.final_verdict else ("approve" if self.approved else "reject")
        hint = self.escalation_hint if self.escalation_hint else (None if self.approved else "blocked")
        debug = self.debug_notes or "; ".join(f"{d.provider}={d.decision}" for d in self.provider_results)
        reason = self.reason or ("consensus_pass" if self.approved else "consensus_blocked")
        return CouncilVerdict(
            verdict=verdict,
            reason=reason,
            suggested_fix=self.suggested_fix,
            escalation_hint=hint,
            debug_notes=debug or None,
            council_degraded=self.degraded,
            council_degraded_reason=self.degraded_reason,
            providers_tried=debug or None,
        )


COUNCIL_STATE_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "governance" / "council_state.json"
)
DEGRADED_ACTION_WHITELIST = {"keyboard.type", "keyboard.hotkey", "mouse.click", "window.focus"}
SAFE_HOTKEYS = {
    ("alt", "tab"),
    ("shift", "alt", "tab"),
    ("ctrl", "tab"),
    ("ctrl", "shift", "tab"),
    ("win", "d"),
    ("esc",),
}

STRICT_JSON_ONLY_SUFFIX = """
────────────────────────────────────
SALIDA ESTRICTA (JSON ONLY)
────────────────────────────────────
Devuelve ESTRICTAMENTE un (1) objeto JSON y NADA más.
- Prohibido Markdown / ``` / fences / texto extra.
- La salida debe empezar con "{" y terminar con "}".
- Usa comillas dobles para claves y strings.
- No incluyas saltos de línea dentro de strings (si necesitas separar ideas, usa '; ').
- En este modo, pon "debug_notes": "" (vacío) para minimizar riesgo de JSON inválido.
""".strip()


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    # Quitar fences de forma tolerante.
    s = s.strip()
    s = s.strip("`").strip()
    if s.lower().startswith("json"):
        s = s[len("json") :].strip()
    return s


def _parse_json_object(text: str) -> Dict[str, Any]:
    """
    Extractor robusto (TICKET COUNCIL-JSON-EXTRACTOR-004):
    - intenta json.loads directo
    - si falla, intenta substring entre el primer '{' y el último '}'
    - si falla, intenta la última línea que parezca un JSON completo
    """
    raw = _strip_code_fences(text)
    raw = raw.strip()
    if not raw:
        raise ValueError("empty")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                data = json.loads(ln)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    raise ValueError("invalid_json")


def _extract_codex_jsonl(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Codex `--json` imprime eventos JSONL. Extrae:
    - último mensaje del agente (texto)
    - último error (si existe)
    """
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
        ev_type = event.get("type")
        if ev_type == "error":
            msg = event.get("message")
            if isinstance(msg, str) and msg.strip():
                last_error = msg.strip()
        if ev_type == "item.completed":
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type not in {"agent_message", "assistant_message", "message"}:
                continue
            text = item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                last_text = text.strip()
    return last_text, last_error


def _now_ts() -> float:
    return time.time()


def load_council_state(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or COUNCIL_STATE_PATH
    default = {"mode": "normal", "reason": None, "timestamp": _now_ts()}
    if not target.exists():
        return default
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        mode = str(data.get("mode") or "normal")
        reason = data.get("reason")
        ts = float(data.get("timestamp") or _now_ts())
        return {"mode": mode, "reason": reason, "timestamp": ts}
    except Exception:
        return default


def persist_council_state(mode: str, reason: Optional[str] = None, path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or COUNCIL_STATE_PATH
    payload = {"mode": mode, "reason": reason, "timestamp": _now_ts()}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return payload


def set_council_mode(mode: str, reason: Optional[str] = None, path: Optional[Path] = None) -> Dict[str, Any]:
    prev = load_council_state(path)
    if prev.get("mode") != mode or (reason and prev.get("reason") != reason):
        try:
            print(f"🏛️ COUNCIL mode change: {prev.get('mode')} -> {mode} ({reason or 'sin motivo'})")
        except Exception:
            pass
    return persist_council_state(mode, reason, path)


def _normalize_hotkeys(keys: List[str]) -> Tuple[str, ...]:
    normalized = tuple(k.strip().lower() for k in keys if isinstance(k, str) and k.strip())
    return normalized


def is_action_allowed_in_degraded(action: Optional[str], args: Optional[Dict[str, Any]] = None) -> bool:
    if not action:
        return False
    action = action.strip()
    if action not in DEGRADED_ACTION_WHITELIST:
        return False
    if action != "keyboard.hotkey":
        return True
    keys_raw = []
    if args and isinstance(args, dict):
        keys_raw = args.get("keys") or []
    if isinstance(keys_raw, str):
        keys_raw = [keys_raw]
    if not isinstance(keys_raw, list):
        return False
    normalized = _normalize_hotkeys(keys_raw)
    return normalized in SAFE_HOTKEYS


class Council:
    """
    Adaptador liviano para una segunda opinión (Consejo).
    Usa un provider con rol 'council' o un fallback (groq / gemini / codex_brain)
    para validar planes antes de ejecutarlos en el PC del usuario.
    """

    def __init__(self, provider_configs: Dict[str, Any]):
        self.provider_configs = provider_configs
        self._root_dir = Path(__file__).resolve().parents[1]
        self.timeout_per_provider_sec = int(os.getenv("AJAX_COUNCIL_TIMEOUT_PER_PROVIDER_SEC", "20") or 20)
        self.max_attempts = int(os.getenv("AJAX_COUNCIL_MAX_ATTEMPTS", "3") or 3)

    def _providers_cfg(self) -> Dict[str, Any]:
        if isinstance(self.provider_configs, dict):
            return self.provider_configs.get("providers", {}) or {}
        return {}

    def _is_disabled(self, name: str, cfg: Dict[str, Any]) -> bool:
        if cfg.get("disabled"):
            return True
        if name == "codex_brain" and os.environ.get("DISABLE_CODEX_COUNCIL"):
            return True
        return False

    def _status_path(self) -> Path:
        return self._root_dir / "artifacts" / "health" / "providers_status.json"

    def _record_provider_attempt(self, provider: str, decision: CouncilProviderDecision) -> None:
        try:
            from agency import provider_ranker  # type: ignore
        except Exception:
            provider_ranker = None  # type: ignore
        if provider_ranker is None:
            return
        try:
            ok = decision.decision in {"pass", "fail"}
            provider_ranker.record_attempt(
                self._status_path(),
                provider=provider,
                ok=ok,
                latency_ms=decision.latency_ms,
                outcome=decision.decision,
                error=decision.reasons or None,
            )
        except Exception:
            return

    @staticmethod
    def _normalize_rail(raw: Optional[str]) -> str:
        val = (raw or "").strip().lower()
        if val in {"prod", "production", "live"}:
            return "prod"
        return "lab"

    @staticmethod
    def _extract_risk_level(context: Optional[Dict[str, Any]]) -> str:
        # Preferir misión si existe (AjaxCore pasa mission_envelope en brain_input)
        try:
            env = (context or {}).get("mission_envelope") or {}
            gov = env.get("governance") or {}
            rl = str(gov.get("risk_level") or "").strip().lower()
            if rl in {"low", "medium", "high"}:
                return rl
        except Exception:
            pass
        return "medium"

    @classmethod
    def _required_quorum(cls, *, rail: str, risk_level: str) -> int:
        raw_override = (os.getenv("AJAX_COUNCIL_QUORUM") or os.getenv("AJAX_COUNCIL_QUORUM_REQUIRED") or "").strip()
        if raw_override:
            try:
                val = int(raw_override)
                if val in {1, 2}:
                    return val
            except Exception:
                pass
        rail_n = cls._normalize_rail(rail)
        rl = (risk_level or "medium").strip().lower()
        if rail_n == "prod":
            return 2
        if rl == "low":
            return 1
        return 2

    @staticmethod
    def _capability_human_permission_switch(context: Optional[Dict[str, Any]]) -> bool:
        try:
            caps = (context or {}).get("capabilities") or {}
            return bool(caps.get("human_permission_switch"))
        except Exception:
            return False

    def _default_pool(self) -> List[str]:
        # Canon: pool determinista via policy, fallback a orden histórico.
        try:
            from agency import provider_policy  # type: ignore
        except Exception:
            provider_policy = None  # type: ignore
        cost_mode = (os.getenv("AJAX_COST_MODE") or "").strip().lower()
        if not cost_mode:
            cost_mode = self._default_cost_mode()
        if provider_policy is not None:
            try:
                rail = self._normalize_rail(os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE"))
                pol = provider_policy.load_provider_policy(self._root_dir)
                pref = provider_policy.preferred_providers(pol, rail=rail, role="council")
                if pref:
                    if cost_mode == "save_codex":
                        pref = [p for p in pref if not str(p).startswith("codex_")]
                    return pref
            except Exception:
                pass
        pool = ["lmstudio", "groq", "qwen_cloud", "gemini_cli", "codex_brain"]
        if cost_mode == "save_codex":
            pool = [p for p in pool if not str(p).startswith("codex_")]
        return pool

    def _default_cost_mode(self) -> str:
        try:
            paths = [
                self._root_dir / "state" / "human_active.flag",
                self._root_dir / "artifacts" / "state" / "human_active.flag",
                self._root_dir / "artifacts" / "policy" / "human_active.flag",
            ]
            for path in paths:
                if not path.exists():
                    continue
                raw = path.read_text(encoding="utf-8").strip()
                if not raw:
                    continue
                if raw.startswith("{"):
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict) and "human_active" in data:
                            if data.get("human_active"):
                                return "save_codex"
                            return "premium"
                    except Exception:
                        pass
                lowered = raw.lower()
                if "true" in lowered:
                    return "save_codex"
                if "false" in lowered:
                    return "premium"
        except Exception:
            pass
        return "premium"

    def _rank_pool(self, pool: List[str], *, role: str, rail: str = "lab", risk_level: str = "medium") -> List[str]:
        providers_cfg = self._providers_cfg()
        try:
            from agency import provider_ranker  # type: ignore
        except Exception:
            provider_ranker = None  # type: ignore
        policy_doc: Dict[str, Any] = {}
        try:
            from agency import provider_policy  # type: ignore

            policy_doc = provider_policy.load_provider_policy(self._root_dir)
        except Exception:
            policy_doc = {}
        policy_providers = policy_doc.get("providers") if isinstance(policy_doc, dict) else None
        policy_providers = policy_providers if isinstance(policy_providers, dict) else {}
        allow_local_override = (os.getenv("AJAX_ALLOW_LOCAL_TEXT") or "").strip().lower() in {"1", "true", "yes", "on"}
        filtered_pool = []
        for p in pool:
            ent = policy_providers.get(p)
            if isinstance(ent, dict):
                state = str(ent.get("policy_state") or "").strip().lower()
                if state in {"disallowed", "blocked", "deny"} and not allow_local_override:
                    continue
            filtered_pool.append(p)
        if provider_ranker is None:
            return [p for p in filtered_pool if p in providers_cfg]
        try:
            status = provider_ranker.load_status(self._status_path())
            scoreboard_doc: Dict[str, Any] = {}
            try:
                from agency import provider_scoreboard  # type: ignore

                scoreboard_doc = provider_scoreboard.load_scoreboard(
                    self._root_dir / "artifacts" / "state" / "provider_scoreboard.json"
                )
            except Exception:
                scoreboard_doc = {}
            prefer_tier = (os.getenv("AJAX_COST_MODE") or "premium").strip().lower()
            if prefer_tier not in {"premium", "balanced", "cheap", "emergency"}:
                prefer_tier = "premium"
            # v0: tier bias + order de pool como tie-breaker
            return provider_ranker.rank_providers(
                filtered_pool,
                providers_cfg=providers_cfg,
                status=status,
                scoreboard=scoreboard_doc,
                prefer_tier=prefer_tier,
                role=role,
                rail=rail,
                risk_level=risk_level,
            )
        except Exception:
            return [p for p in filtered_pool if p in providers_cfg]

    def _try_review_with_pool(
        self,
        *,
        pool: List[str],
        exclude: Optional[set[str]],
        rail: str,
        risk_level: str,
        intention: str,
        plan_json: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        timeout_per_provider_sec: int,
        max_attempts: int,
        pre_ranked: bool = False,
        model_overrides: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[CouncilProviderDecision], List[CouncilProviderDecision]]:
        providers_cfg = self._providers_cfg()
        attempts: List[CouncilProviderDecision] = []
        exclude = exclude or set()

        ranked = list(pool) if pre_ranked else self._rank_pool(pool, role="council")
        try:
            from agency import provider_ranker  # type: ignore
        except Exception:
            provider_ranker = None  # type: ignore
        try:
            status = provider_ranker.load_status(self._status_path()) if provider_ranker else {}
        except Exception:
            status = {}
        try:
            from agency.auth_manager import AuthManager  # type: ignore
        except Exception:
            AuthManager = None  # type: ignore
        auth = AuthManager(root_dir=self._root_dir) if AuthManager else None
        for provider_name in ranked:
            if provider_name in exclude:
                continue
            cfg = providers_cfg.get(provider_name)
            if not isinstance(cfg, dict) or self._is_disabled(provider_name, cfg):
                continue

            # AUTH_REQUIRED => soft-skip, gap, y fallback inmediato (no invocar provider)
            if auth is not None:
                try:
                    astate = auth.auth_state(provider_name, cfg)
                    auth.persist_auth_state(provider_name, astate)
                    auth.ensure_auth_gap(provider_name, astate)
                    if astate.state in {"MISSING", "EXPIRED"} and AuthManager.is_web_auth_required(cfg):  # type: ignore[attr-defined]
                        decision = CouncilProviderDecision(
                            provider=provider_name,
                            decision="unavailable",
                            reasons=f"auth_{astate.state.lower()}:{astate.reason}",
                            latency_ms=0,
                        )
                        attempts.append(decision)
                        self._record_provider_attempt(provider_name, decision)
                        if len(attempts) >= max_attempts:
                            break
                        continue
                except Exception:
                    pass

            provider_timeout = timeout_per_provider_sec
            try:
                if provider_ranker:
                    provider_timeout = max(
                        provider_timeout,
                        int(
                            provider_ranker.recommended_timeout_seconds(
                                provider_name=provider_name,
                                provider_cfg=cfg,
                                status=status,
                                role="council",
                                rail=rail,
                                risk_level=risk_level,
                            )
                        ),
                    )
                provider_timeout = max(provider_timeout, int(cfg.get("timeout_seconds") or 0))
            except Exception:
                provider_timeout = timeout_per_provider_sec

            cfg_use = cfg
            if model_overrides and provider_name in model_overrides:
                try:
                    cfg_use = dict(cfg)
                    cfg_use["_selected_model"] = model_overrides[provider_name]
                except Exception:
                    cfg_use = cfg
            decision = self._invoke_provider(
                provider_name,
                cfg_use,
                intention,
                plan_json,
                system_prompt,
                user_prompt,
                timeout_override_sec=provider_timeout,
            )
            attempts.append(decision)
            self._record_provider_attempt(provider_name, decision)
            # Fallback SOLO en errores técnicos (unavailable)
            if decision.decision != "unavailable":
                return decision, attempts
            if len(attempts) >= max_attempts:
                break

        return None, attempts

    @staticmethod
    def _normalize_escalation_hint(raw: Any) -> Optional[str]:
        hint = str(raw or "").strip()
        if not hint:
            return None
        hint = hint.lower()
        allowed = {"await_user_input", "try_stronger_model", "unsafe", "blocked"}
        return hint if hint in allowed else None

    @classmethod
    def _build_verdict(cls, payload: Dict[str, Any]) -> CouncilVerdict:
        verdict = payload.get("verdict") or "reject"
        reason = payload.get("reason") or payload.get("feedback") or ""
        hint = cls._normalize_escalation_hint(payload.get("escalation_hint"))
        # Normalización defensiva: si falta el hint, inferir uno conservador
        if hint is None:
            v = str(verdict or "").strip().lower()
            if v == "needs_revision":
                hint = "try_stronger_model"
            elif v == "reject":
                hint = "await_user_input"
        return CouncilVerdict(
            verdict=verdict,
            reason=reason,
            suggested_fix=payload.get("suggested_fix") or payload.get("suggestion"),
            escalation_hint=hint,
            debug_notes=payload.get("debug_notes"),
        )

    def _select_council_provider(self) -> Tuple[str, Dict[str, Any]]:
        providers_cfg = self._providers_cfg()
        cost_mode = os.getenv("AJAX_COST_MODE", "premium")

        # Preferimos usar el router si existe
        if pick_model:
            prov, cfg = pick_model(
                "council_review",
                providers_cfg=providers_cfg,
                cost_mode=cost_mode,
                intent=None,
                slot="council.main",
            )
            if cfg.get("_slot_missing"):
                try:
                    print(
                        f"🏛️ COUNCIL slot council.main faltante "
                        f"({cfg.get('_slot_requested')}), usando fallback "
                        f"{cfg.get('_slot_selected') or prov}"
                    )
                except Exception:
                    pass
            return prov, cfg

        # 1) Provider con rol "council"
        for name, cfg in providers_cfg.items():
            roles = cfg.get("roles") or []
            if self._is_disabled(name, cfg):
                continue
            if isinstance(roles, list) and "council" in [r.lower() for r in roles]:
                return name, cfg

        # 2) Fallbacks razonables
        if "groq" in providers_cfg and not self._is_disabled("groq", providers_cfg["groq"]):
            return "groq", providers_cfg["groq"]
        if "gemini_cli" in providers_cfg and not self._is_disabled(
            "gemini_cli", providers_cfg["gemini_cli"]
        ):
            return "gemini_cli", providers_cfg["gemini_cli"]
        if "codex_brain" in providers_cfg and not self._is_disabled(
            "codex_brain", providers_cfg["codex_brain"]
        ):
            return "codex_brain", providers_cfg["codex_brain"]

        raise RuntimeError("No hay provider disponible para council")

    def _select_council_providers(self) -> List[Tuple[str, Dict[str, Any]]]:
        providers_cfg = self._providers_cfg()
        selected: List[Tuple[str, Dict[str, Any]]] = []
        try:
            primary = self._select_council_provider()
            selected.append(primary)
        except Exception:
            pass

        # Segundo proveedor distinto si es posible
        if providers_cfg:
            for name, cfg in providers_cfg.items():
                if selected and name == selected[0][0]:
                    continue
                if self._is_disabled(name, cfg):
                    continue
                roles = cfg.get("roles") or []
                if isinstance(roles, list) and "council" in [r.lower() for r in roles]:
                    selected.append((name, cfg))
                    break

        fallback_order = ("gemini_cli", "codex_brain", "qwen_cloud", "qwen_cli", "groq")
        if selected and len(selected) < 2:
            for cand in fallback_order:
                cfg = providers_cfg.get(cand)
                if not cfg or self._is_disabled(cand, cfg):
                    continue
                if any(cand == prov for prov, _ in selected):
                    continue
                selected.append((cand, cfg))
                break

        if not selected:
            raise RuntimeError("No hay provider disponible para council")
        return selected[:2]

    @staticmethod
    def _guard_tool_plan(tool_plan: Optional[Dict[str, Any]], intention: str) -> Optional[CouncilVerdict]:
        if not tool_plan or not isinstance(tool_plan, dict):
            return None
        required = tool_plan.get("required") or []
        satisfied = tool_plan.get("satisfied") or {}
        reasons = tool_plan.get("reasons") or []
        risk_flags = tool_plan.get("risk_flags") or {}
        missing = [req for req in required if not satisfied.get(req)]
        if not missing:
            return None
        high_risk = bool(risk_flags.get("high_risk_intent") or any("high_risk" in str(r) for r in reasons))
        historical = "memory.leann_history" in required or any("historic" in str(r) for r in reasons)
        if high_risk or historical:
            return CouncilVerdict(
                verdict="reject",
                reason=f"tool_plan_required_missing:{','.join(missing)}",
                suggested_fix="adjunta memoria LEANN/snippets o ejecuta heartbeat antes de continuar",
                escalation_hint="await_user_input",
                debug_notes=f"tool_plan_guard:{intention[:80]}",
            )
        return CouncilVerdict(
            verdict="needs_revision",
            reason=f"tool_plan_incomplete:{','.join(missing)}",
            suggested_fix="completa los requisitos del ToolPlan",
            escalation_hint="await_user_input",
            debug_notes="tool_plan_guard_soft",
        )

    def _build_prompts(
        self,
        intention: str,
        plan_json: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        actions_catalog: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        system_prompt = (
            AJAX_METHOD_PACK
            + "\n\n"
            + """Eres el Council de AJAX, el último filtro de seguridad antes de ejecutar un plan sobre el ordenador del usuario.

Tu misión NO es quedar bien, ni ser simpático.
Tu misión es SIEMPRE elegir el SIGUIENTE PASO MÁS SEGURO hacia la intención del usuario.

No existen subtareas.
Solo existen tareas completas encadenadas.

Una tarea solo puede delegar en otra tarea completa, nunca en un atajo.

Regla estructural (no negociable):
- Cada Step es una TAREA atomizada (no un “paso ligero”).
- Cada step DEBE incluir: id, intent, preconditions.expected_state, success_spec.expected_state, evidence_required, on_fail="abort", action, args.
- Si falta alguno: tu verdict NO puede ser approve. Debe ser needs_revision con suggested_fix indicando “regenera el plan con el schema completo”.

Robustez (cuando hay selección ambigua):
- Si el plan debe seleccionar un ítem entre resultados (búsqueda/lista), debe:
  (a) tener método de selección determinista + verificación con >=2 señales independientes, o
  (b) incluir un step explícito action="await_user_input" para pedir confirmación/selección humana.
- Para intenciones tipo "reproducir media", exige verificación con >=2 señales independientes (p.ej. título + reproduciendo/progreso).

Piensa así:
- Si hay al menos UNA forma segura o reversible de avanzar (preguntar, clarificar, escalar modelo, esperar, reparar algo), la misión SIGUE VIVA.
- Solo aceptas rendirte si:
  • el usuario la cancela explícitamente, o
  • seguir sería inherentemente inseguro (privacidad, dinero, trabajo no guardado, mundo físico), o
  • es técnicamente imposible (sin red prolongada y la tarea la requiere, etc.).

Recibirás como contexto:
- La intención del usuario (texto libre).
- El plan propuesto (JSON).
- knowledge_context.signals → estado del sistema:
  • safety_profile: "tonterias" | "normal" | "delicado" | "paranoico"
  • driver_health: "healthy" | "degraded" | "down"
  • known_risks: lista de riesgos conocidos (p.ej. "multiple_ubuntu_windows")
  • otros metadatos del entorno.

DEBES usar esas señales. No estás en el vacío.
- driver_health:
  • "down": acciones de escritorio son muy inciertas. Si la acción depende del driver, prefiere "await_user_input" o "unsafe"; si no depende, puedes seguir.
  • "degraded": NO bloquees acciones reversibles de escritorio (minimizar, mover, foco) en perfiles "tonterias"/"normal"; sí pide aclaración/precaución en acciones sensibles (guardar/cerrar apps/archivos, sesiones críticas).
  • "healthy": procede según el resto de reglas.
- safety_profile ∈ {"delicado","paranoico"} y la acción toca escritorio: prefiere "await_user_input" o mejoras de plan antes que "approve".

────────────────────────────────────
MODOS DE SEGURIDAD DEL USUARIO
────────────────────────────────────

El perfil de seguridad actual del usuario es: {{current_safety_profile}}

Aplica estas políticas estrictas:

"tonterias":
  - Acciones reversibles de escritorio (minimizar ventanas, cambiar foco, mover ventanas) se pueden aprobar con ambigüedad leve.
  - Pide aclaración solo si ves riesgo medio/alto de dejar al usuario peor que antes.
  - No uses "unsafe" para acciones claramente reversibles.

"normal":
  - Pregunta cuando haya ambigüedad razonable sobre QUÉ objeto se toca:
    "esta ventana", "este archivo", "lo que suena", etc.
  - Acepta pequeños riesgos en acciones reversibles, pero nunca en:
    datos sensibles, sesiones críticas, dinero, mundo físico o trabajo no guardado.

"delicado":
  - Ante cualquier referencia deíctica ("esta ventana", "este archivo", "la canción que suena")
    tiende a pedir aclaración salvo que el plan tenga verificación robusta.
  - Si dudas, pide aclaración. No des nada por hecho.

"paranoico":
  - Solo apruebes planes con alta claridad y, si tocan datos/sesiones/mundo físico,
    con confirmación explícita del usuario o doble verificación independiente.
  - El resto es aclaración ("await_user_input") o "unsafe".

────────────────────────────────────
PRIMEROS PRINCIPIOS
────────────────────────────────────

1) El mundo es caótico y asíncrono:
   títulos cambian, ventanas se duplican, el usuario puede tocar cosas mientras tanto.
2) Un falso positivo (ejecutar algo malo) es muchísimo más caro que un falso negativo
   (pedir aclaración, negarse o escalar modelo).
3) Toda referencia deíctica ("esta ventana", "este archivo", "lo que suena") es ambigua
   hasta demostrar lo contrario con señales sólidas (foco + título + proceso, etc.).
4) El éxito no es "ejecutar el plan", es "no dejar al usuario peor que antes".
5) Operaciones sobre:
   - datos sensibles
   - sesiones críticas (correo, banco, administración)
   - archivos de trabajo no guardado
   - dispositivos físicos (puerta, alarma, domótica)
   requieren máxima prudencia.

────────────────────────────────────
ROL DEL COUNCIL
────────────────────────────────────

Eres a la vez:

1) Council EXIGENTE:
   - Rechaza planes mediocres.
   - No aceptas criterios de éxito débiles (p.ej. solo title_match cuando hay riesgos).
   - No apruebas por educación, ni por prisa.

2) Council POSIBILISTA:
   - Si el plan es inseguro pero arreglable con:
     • una pregunta al usuario,
     • un mejor criterio de éxito,
     • escalar modelo,
     • arreglar un servicio (driver, red),
     entonces tu trabajo NO es bloquear, es indicar el siguiente paso seguro.
   - Solo devuelves "unsafe" cuando incluso con aclaraciones o más modelo
     seguiría siendo demasiado peligroso.

────────────────────────────────────
FORMATO DE SALIDA (OBLIGATORIO)
────────────────────────────────────

Tu salida DEBE ser SIEMPRE un único objeto JSON válido, sin texto antes ni después.

Formato exacto:

{
  "verdict": "approve" | "needs_revision" | "reject",
  "reason": "explicación breve en UNA sola línea, sin saltos de línea ni comillas dobles internas",
  "suggested_fix": "sugerencia breve en una sola línea o \"\" si no aplica",
  "escalation_hint": "await_user_input" | "try_stronger_model" | "unsafe",
  "debug_notes": "deliberación interna resumida; puede ser varias frases, pero sin saltos de línea"
}

Significado:

- "verdict":
  - "approve": el plan es suficientemente seguro para el modo de seguridad actual.
  - "needs_revision": el plan tiene problemas que podrían arreglarse (más información,
    mejor criterio de éxito, más razonamiento).
  - "reject": el plan es inaceptable en su forma actual.

- "escalation_hint":
  - "await_user_input":
      El siguiente paso SEGURO es pedir aclaración o confirmación al usuario y,
      después de su respuesta, continuar la MISMA misión.
      Ejemplo típico: órdenes ambiguas tipo "minimiza todas las ventanas menos ésta"
      sin saber con certeza cuál es "esta".

  - "try_stronger_model":
      El plan es débil, incompleto o ingenuo, pero un modelo más fuerte probablemente
      sí podría encontrar un plan seguro sin molestar aún al usuario.
      Ejemplo típico: múltiples ventanas, overlays, casos visuales complejos.

  - "unsafe":
      La naturaleza de la orden es inherentemente arriesgada (privacidad, dinero,
      trabajo no guardado, mundo físico) y ningún modelo debería ejecutarla sin
      confirmación fuerte y medidas extra.
      Aquí el orquestador debe tratar la misión como peligrosamente delicada.

- "debug_notes":
  - Resumen de tu deliberación interna (mini-brainstorm): qué viste en el plan,
    qué señales del sistema pesaron (driver_health, safety_profile, known_risks),
    qué alternativas consideraste.

────────────────────────────────────
GUÍA PRÁCTICA PARA CASOS TÍPICOS
────────────────────────────────────

1) Orden reversible ambigua:
   Intento: "minimiza todas las ventanas menos ésta"
   - Si no puedes identificar con claridad cuál es "esta" ventana:
     • verdict = "needs_revision"
     • escalation_hint = "await_user_input"
     • reason = "falta confirmación explícita de qué ventana mantener visible"
     • suggested_fix = "pregunta al usuario qué ventana debe quedar visible antes de minimizar las demás"
   - Si el modo de seguridad es "tonterias" y solo hay una ventana visible razonable:
     puedes aprobar, pero solo si no hay riesgos conocidos en signals (p.ej. "nunca minimizar vscode").

2) Driver degradado o inestable:
   - Si signals.driver_health es "degraded" o "down":
     • Sé más conservador al aprobar acciones que dependen del driver.
     • Si el plan asume un driver sano, considera "needs_revision" + "try_stronger_model"
       o "await_user_input" para que el orquestador pueda arreglar infra o esperar.

3) Operaciones sobre datos sensibles o sesiones:
   - Cerrar sesión bancaria, borrar archivos, cerrar aplicaciones con trabajo no guardado:
     • Tienden a "needs_revision" + "await_user_input" o directamente "unsafe" según contexto.

────────────────────────────────────
REGLA FINAL
────────────────────────────────────

Nunca devuelvas texto fuera del objeto JSON.
Nunca omitas "escalation_hint".
Es mejor pedir aclaración o escalar modelo tres veces
que ejecutar una sola vez algo que deje al usuario peor que antes.

        Tu respuesta: SOLO el objeto JSON con esos cinco campos. Nada antes, nada después, nada fuera de las llaves."""
        )
        # Endurecer "JSON-only" (TICKET COUNCIL-ROBUST-PARSE-003)
        system_prompt += (
            "\n\nOUTPUT STRICT:\n"
            "- Output strictly one JSON object.\n"
            "- No markdown, no code fences, no extra text.\n"
            "- Output must start with '{' and end with '}'.\n"
            "- Keep all values on a single line (no raw newlines inside strings).\n"
        )
        evidence = {
            "installed_apps": (context or {}).get("installed_apps"),
            "knowledge_context": (context or {}).get("knowledge_context"),
            "observation": (context or {}).get("observation"),
            "tool_plan": (context or {}).get("tool_plan"),
        }
        user_prompt = (
            f"Intención: {intention}\n"
            f"Plan:\n{json.dumps(plan_json, ensure_ascii=False, indent=2)}\n"
            f"CONTEXTUAL EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
        )
        if actions_catalog:
            user_prompt += (
                f"\nAcciones permitidas: "
                f"{json.dumps(actions_catalog, ensure_ascii=False)}"
            )
        return system_prompt, user_prompt

    def _call_cli_json(self, cmd: List[str], prompt: str, timeout: int) -> str:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or (proc.stdout or "").strip() or f"cli_rc_{proc.returncode}"
            raise RuntimeError(err[:200])
        return proc.stdout.strip()

    def _invoke_provider(
        self,
        provider_name: str,
        cfg: Dict[str, Any],
        intention: str,
        plan_json: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        *,
        timeout_override_sec: Optional[int] = None,
        strict_json_only: bool = False,
    ) -> CouncilProviderDecision:
        kind = (cfg.get("kind") or "").lower()
        model = (
            cfg.get("_selected_model")
            or cfg.get("default_model")
            or cfg.get("model")
            or "llama-3.1-8b-instant"
        )
        timeout = int(timeout_override_sec or cfg.get("timeout_seconds") or 20)
        if strict_json_only:
            system_prompt = system_prompt.rstrip() + "\n\n" + STRICT_JSON_ONLY_SUFFIX
        started = time.time()
        try:
            print(
                f"🏛️ COUNCIL provider={provider_name} model={model} "
                f"tier={cfg.get('tier','unknown')}"
            )
        except Exception:
            pass
        print(f"🏛️ COUNCIL: Revisando plan para '{intention}' con {provider_name}...")

        def _build_decision(ver: Optional[CouncilVerdict]) -> CouncilProviderDecision:
            if ver is None:
                return CouncilProviderDecision(
                    provider=provider_name,
                    decision="unavailable",
                    reasons="empty_verdict",
                    latency_ms=int((time.time() - started) * 1000),
                )
            decision = "pass" if ver.approved else "fail"
            reasons = ver.reason or ver.feedback or ""
            if not ver.approved and not reasons:
                return CouncilProviderDecision(
                    provider=provider_name,
                    decision="unavailable",
                    reasons="invalid_reviewer_output:empty_reason",
                    latency_ms=int((time.time() - started) * 1000),
                )
            if ver.debug_notes:
                try:
                    print(f"🧠 COUNCIL debug ({provider_name}): {ver.debug_notes}")
                except Exception:
                    pass
            summary = "aprobado" if ver.approved else f"veto:{reasons}"
            try:
                print(f"🏛️ COUNCIL {provider_name}: {summary}")
            except Exception:
                pass
            return CouncilProviderDecision(
                provider=provider_name,
                decision=decision,
                reasons=reasons,
                verdict=ver,
                latency_ms=int((time.time() - started) * 1000),
            )

        try:
            if kind == "codex_cli_jsonl":
                cmd_template = cfg.get("command") or [
                    "codex",
                    "exec",
                    "--model",
                    "{model}",
                    "--json",
                ]
                cmd = [model if t == "{model}" else t for t in cmd_template]
                prompt = system_prompt.rstrip() + "\n\n" + user_prompt
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
                raw = (proc.stdout or "").strip()
                last_text, last_error = _extract_codex_jsonl(raw)
                if proc.returncode != 0:
                    err = (last_error or "").strip() or (proc.stderr or "").strip() or raw[:200] or "codex_failed"
                    return CouncilProviderDecision(
                        provider=provider_name,
                        decision="unavailable",
                        reasons=f"codex_rc_{proc.returncode}:{err[:200]}",
                        latency_ms=int((time.time() - started) * 1000),
                    )
                candidate = last_text or raw
                try:
                    data = _parse_json_object(candidate)
                    ver = self._build_verdict(data)
                    return _build_decision(ver)
                except Exception:
                    if not strict_json_only:
                        return self._invoke_provider(
                            provider_name,
                            cfg,
                            intention,
                            plan_json,
                            system_prompt,
                            user_prompt,
                            timeout_override_sec=timeout_override_sec,
                            strict_json_only=True,
                        )
                    return CouncilProviderDecision(
                        provider=provider_name,
                        decision="unavailable",
                        reasons=f"invalid_reviewer_output:parse_error:{candidate[:200]}",
                        latency_ms=int((time.time() - started) * 1000),
                    )

            if kind == "cli":
                prompt = (system_prompt.rstrip() + "\n\n" + user_prompt).encode("utf-8", errors="ignore").decode(
                    "utf-8", errors="ignore"
                )
                cmd_template = cfg.get("command") or []
                cmd: List[str] = []
                used_prompt_placeholder = False
                for token in cmd_template:
                    if token == "{model}":
                        cmd.append(str(model or ""))
                        continue
                    if token == "{prompt}":
                        cmd.append(prompt)
                        used_prompt_placeholder = True
                        continue
                    cmd.append(str(token))
                if not cmd:
                    raise RuntimeError("cli_command_missing")
                if used_prompt_placeholder:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                else:
                    proc = subprocess.run(
                        cmd,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                if proc.returncode != 0:
                    err = (proc.stderr or "").strip() or (proc.stdout or "").strip() or f"cli_rc_{proc.returncode}"
                    raise RuntimeError(err[:200])
                raw = (proc.stdout or "").strip()
                try:
                    data = _parse_json_object(raw)
                    ver = self._build_verdict(data)
                    return _build_decision(ver)
                except Exception:
                    if not strict_json_only:
                        return self._invoke_provider(
                            provider_name,
                            cfg,
                            intention,
                            plan_json,
                            system_prompt,
                            user_prompt,
                            timeout_override_sec=timeout_override_sec,
                            strict_json_only=True,
                        )
                    return CouncilProviderDecision(
                        provider=provider_name,
                        decision="unavailable",
                        reasons=f"invalid_reviewer_output:parse_error:{raw[:200]}",
                        latency_ms=int((time.time() - started) * 1000),
                    )

            base_url = (cfg.get("base_url") or "").rstrip("/")
            if not base_url:
                raise RuntimeError("base_url_missing")
            api_key_env = cfg.get("api_key_env")
            api_key = os.getenv(api_key_env) if api_key_env else None
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0 if strict_json_only else 0.1,
                "max_tokens": 200,
            }
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if resp.status_code >= 400:
                return CouncilProviderDecision(
                    provider=provider_name,
                    decision="unavailable",
                    reasons=f"http_{resp.status_code}:{resp.text[:200]}",
                    latency_ms=int((time.time() - started) * 1000),
                )

            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
                or ""
            )
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[len("json") :].strip()

            try:
                payload = _parse_json_object(content)
                ver = self._build_verdict(payload)
                return _build_decision(ver)
            except Exception:
                if not strict_json_only:
                    return self._invoke_provider(
                        provider_name,
                        cfg,
                        intention,
                        plan_json,
                        system_prompt,
                        user_prompt,
                        timeout_override_sec=timeout_override_sec,
                        strict_json_only=True,
                    )
                return CouncilProviderDecision(
                    provider=provider_name,
                    decision="unavailable",
                    reasons=f"invalid_reviewer_output:parse_error:{content[:200]}",
                    latency_ms=int((time.time() - started) * 1000),
                )
        except Exception as exc:
            return CouncilProviderDecision(
                provider=provider_name,
                decision="unavailable",
                reasons=str(exc),
                latency_ms=int((time.time() - started) * 1000),
            )

    @staticmethod
    def _providers_tried_debug(decisions: List[CouncilProviderDecision]) -> str:
        parts: List[str] = []
        for d in decisions:
            part = f"{d.provider}:{d.decision}"
            if d.reasons:
                part += f"({d.reasons})"
            parts.append(part)
        return ", ".join(parts)

    @classmethod
    def _pick_fail_guidance(cls, failures: List[CouncilProviderDecision]) -> Tuple[str, str, Optional[str]]:
        """
        Devuelve (final_verdict, escalation_hint, suggested_fix) para un veto real.
        Priorización: unsafe > await_user_input > try_stronger_model > blocked.
        """
        # Extraer CouncilVerdict si existe
        verdicts: List[CouncilVerdict] = [f.verdict for f in failures if f.verdict is not None]  # type: ignore[list-item]
        # Fallback defensivo
        if not verdicts:
            return "reject", "blocked", None

        def _prio(v: CouncilVerdict) -> int:
            hint = cls._normalize_escalation_hint(v.escalation_hint) or "blocked"
            order = {"unsafe": 0, "await_user_input": 1, "try_stronger_model": 2, "blocked": 3}
            return order.get(hint, 3)

        chosen = sorted(verdicts, key=_prio)[0]
        hint = cls._normalize_escalation_hint(chosen.escalation_hint) or "blocked"
        # Addendum A: try_stronger_model => pursuit (needs_revision)
        if hint == "try_stronger_model":
            return "needs_revision", hint, chosen.suggested_fix
        if hint == "await_user_input":
            return "needs_revision", hint, chosen.suggested_fix
        if hint == "unsafe":
            return "reject", hint, chosen.suggested_fix
        return "reject", "blocked", chosen.suggested_fix

    def review_plan_with_council(
        self,
        intention: str,
        plan_json: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        actions_catalog: Optional[Dict[str, Any]] = None,
    ) -> CouncilConsensusResult:
        try:
            system_prompt, user_prompt = self._build_prompts(intention, plan_json, context, actions_catalog)
        except Exception:
            system_prompt, user_prompt = "", ""

        rail = self._normalize_rail(os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE"))
        risk_level = self._extract_risk_level(context)
        quorum = self._required_quorum(rail=rail, risk_level=risk_level)
        timeout_pp = int(self.timeout_per_provider_sec or 20)
        max_attempts = int(self.max_attempts or 3)
        pool1 = self._default_pool()
        pool2 = self._default_pool()
        pre_ranked = False
        model_overrides: Optional[Dict[str, str]] = None
        # Preflight: si llega Starting XI en contexto, usarlo como roster determinista por rol.
        sx = (context or {}).get("starting_xi") if context else None
        if isinstance(sx, dict):
            try:
                council = sx.get("council") or {}
                r1 = council.get("role1") or {}
                r2 = council.get("role2") or {}
                sx_quorum = None
                q = sx.get("quorum") if isinstance(sx.get("quorum"), dict) else {}
                try:
                    if isinstance(q, dict):
                        eff = q.get("council_effective")
                        req = q.get("council_required")
                        if isinstance(eff, (int, float)) and int(eff) > 0:
                            sx_quorum = int(eff)
                        elif isinstance(req, (int, float)) and int(req) > 0:
                            sx_quorum = int(req)
                except Exception:
                    sx_quorum = None

                def _pool_from(role_entry: Any) -> List[str]:
                    out: List[str] = []
                    if not isinstance(role_entry, dict):
                        return out
                    prim = role_entry.get("primary")
                    bench = role_entry.get("bench") or []
                    for ent in [prim, *bench]:
                        if not isinstance(ent, dict):
                            continue
                        prov = ent.get("provider")
                        if isinstance(prov, str) and prov.strip() and prov not in out:
                            out.append(prov)
                    return out

                def _models_from(role_entry: Any, acc: Dict[str, str]) -> None:
                    if not isinstance(role_entry, dict):
                        return
                    prim = role_entry.get("primary")
                    bench = role_entry.get("bench") or []
                    for ent in [prim, *bench]:
                        if not isinstance(ent, dict):
                            continue
                        prov = ent.get("provider")
                        mid = ent.get("model")
                        if isinstance(prov, str) and prov.strip() and isinstance(mid, str) and mid.strip():
                            acc[prov] = mid

                p1 = _pool_from(r1)
                p2 = _pool_from(r2)
                if p1:
                    pool1 = p1
                    pre_ranked = True
                if p2:
                    pool2 = p2
                    pre_ranked = True
                elif p1:
                    # Para quorum=1, rol2 puede reusar pool1 (excluirá el usado) si se activa quorum>=2.
                    pool2 = p1
                if sx_quorum is not None:
                    quorum = sx_quorum
                overrides: Dict[str, str] = {}
                _models_from(r1, overrides)
                _models_from(r2, overrides)
                model_overrides = overrides or None
            except Exception:
                pre_ranked = False
                model_overrides = None

        # Ledger refresh (no LLM): filtrar providers no-ok antes de quorum/intent.
        ledger_degraded = False
        ledger_degraded_reason: Optional[str] = None
        ledger_ok: set[str] = set()
        ledger_path: Optional[str] = None
        try:
            from agency.provider_ledger import ProviderLedger  # type: ignore
        except Exception:
            ProviderLedger = None  # type: ignore
        if ProviderLedger is not None:
            try:
                pool1_before = list(pool1)
                pool2_before = list(pool2)
                doc = ProviderLedger(root_dir=self._root_dir, provider_configs=self.provider_configs).refresh(timeout_s=1.5) or {}
                ledger_path = doc.get("path") if isinstance(doc, dict) else None
                ledger_rows = doc.get("rows") if isinstance(doc, dict) else None
                ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
                ledger_ok = set(ProviderLedger.ok_providers(ledger_rows, role="council"))
                pool1 = [p for p in pool1 if p in ledger_ok]
                pool2 = [p for p in pool2 if p in ledger_ok]
                if model_overrides:
                    model_overrides = {k: v for k, v in model_overrides.items() if k in ledger_ok}
                filtered = sorted((set(pool1_before) | set(pool2_before)) - (set(pool1) | set(pool2)))
                if filtered:
                    ledger_degraded = True
                    ledger_degraded_reason = f"ledger_filtered:{','.join(filtered[:3])}"
            except Exception:
                ledger_ok = set()

        # Degraded quorum constitucional: si el ledger no permite quorum 2, bajar a 1 en LAB o intents low-risk.
        if quorum >= 2 and ProviderLedger is not None:
            ok_count = len(ledger_ok)
            if ok_count == 0:
                ledger_degraded = True
                ledger_degraded_reason = "ledger_no_ok_providers_for_council"
            elif ok_count < quorum:
                allow_degrade = (
                    rail != "prod"
                    or risk_level == "low"
                    or (os.getenv("AJAX_ALLOW_COUNCIL_QUORUM_DEGRADE") or "").strip().lower() in {"1", "true", "yes", "on"}
                )
                if allow_degrade:
                    ledger_degraded = True
                    ledger_degraded_reason = f"ledger_quorum_degraded:{quorum}->1"
                    quorum = 1
                else:
                    ledger_degraded = True
                    ledger_degraded_reason = f"ledger_quorum_unmet:{ok_count}<{quorum}"

        all_attempts: List[CouncilProviderDecision] = []
        final_decisions: List[CouncilProviderDecision] = []

        used: set[str] = set()
        d1, attempts1 = self._try_review_with_pool(
            pool=pool1,
            exclude=None,
            rail=rail,
            risk_level=risk_level,
            intention=intention,
            plan_json=plan_json,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_per_provider_sec=timeout_pp,
            max_attempts=max_attempts,
            pre_ranked=pre_ranked,
            model_overrides=model_overrides,
        )
        all_attempts.extend(attempts1)
        if d1:
            final_decisions.append(d1)
            used.add(d1.provider)

        if quorum >= 2:
            d2, attempts2 = self._try_review_with_pool(
                pool=pool2,
                exclude=used,
                rail=rail,
                risk_level=risk_level,
                intention=intention,
                plan_json=plan_json,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_per_provider_sec=timeout_pp,
                max_attempts=max_attempts,
                pre_ranked=pre_ranked,
                model_overrides=model_overrides,
            )
            all_attempts.extend(attempts2)
            if d2:
                final_decisions.append(d2)
                used.add(d2.provider)

        passes = [d for d in final_decisions if d.decision == "pass"]
        fails = [d for d in final_decisions if d.decision == "fail"]
        degraded = ledger_degraded or any(d.decision == "unavailable" for d in all_attempts)
        degraded_reason = None
        if ledger_degraded_reason:
            degraded_reason = ledger_degraded_reason
        elif degraded:
            unavailable = [d.provider for d in all_attempts if d.decision == "unavailable"]
            if unavailable:
                degraded_reason = f"provider_unavailable:{','.join(unavailable[:3])}"

        tried_debug = self._providers_tried_debug(all_attempts)
        ledger_note = f" ledger_ok={len(ledger_ok)}" if ProviderLedger is not None else ""
        ledger_note += f" ledger_path={ledger_path}" if ledger_path else ""
        debug_notes = f"rail={rail} risk={risk_level} quorum={quorum};{ledger_note}; providers_tried: {tried_debug}"

        # 1) Veto real (reject/needs_revision del Council)
        if fails:
            final_verdict, hint, suggested_fix = self._pick_fail_guidance(fails)
            # Addendum A: reject/needs_revision con try_stronger_model -> pursuit
            reason = fails[0].reasons or (fails[0].verdict.reason if fails[0].verdict else "council_veto")
            return CouncilConsensusResult(
                provider_results=all_attempts,
                final_decision="veto",
                final_verdict=final_verdict,
                escalation_hint=hint,
                suggested_fix=suggested_fix,
                reason=reason,
                degraded=degraded,
                degraded_reason=degraded_reason,
                debug_notes=debug_notes,
            )

        # 2) Aprobado con quorum suficiente
        if len(passes) >= quorum and quorum > 0:
            if degraded:
                used_fallback = ",".join(sorted({d.provider for d in passes})) or "unknown"
                reason = f"Council degraded ({degraded_reason or 'provider_slow'}), used fallback={used_fallback}"
            else:
                reason = "council_approved"
            return CouncilConsensusResult(
                provider_results=all_attempts,
                final_decision="approved",
                final_verdict="approve",
                escalation_hint=None,
                suggested_fix=None,
                reason=reason,
                degraded=degraded,
                degraded_reason=degraded_reason,
                debug_notes=debug_notes,
            )

        # 3) Bloqueado por falta de quorum (solo errores técnicos / unavailable)
        blocked_code = "BLOCKED_BY_COUNCIL_QUORUM"
        if any(
            (d.decision == "unavailable" and d.reasons and "invalid_reviewer_output" in d.reasons)
            for d in all_attempts
        ):
            blocked_code = "BLOCKED_BY_COUNCIL_INVALID_REVIEW"
        blocked_reason = f"{blocked_code} (providers tried: {tried_debug})"
        human_switch = self._capability_human_permission_switch(context)
        if rail == "prod" and human_switch and passes:
            used_pass = ",".join(sorted({d.provider for d in passes}))
            question = f"No pude alcanzar quorum del Council. ¿Autorizas continuar con revisión parcial ({used_pass})?"
            return CouncilConsensusResult(
                provider_results=all_attempts,
                final_decision="blocked",
                final_verdict="needs_revision",
                escalation_hint="await_user_input",
                suggested_fix=question,
                reason=blocked_reason,
                degraded=degraded,
                degraded_reason=degraded_reason,
                debug_notes=debug_notes,
            )
        return CouncilConsensusResult(
            provider_results=all_attempts,
            final_decision="blocked",
            final_verdict="reject",
            escalation_hint="blocked",
            suggested_fix=None,
            reason=blocked_reason,
            degraded=degraded,
            degraded_reason=degraded_reason,
            debug_notes=debug_notes,
        )

    def review_plan(
        self,
        intention: str,
        plan_json: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        actions_catalog: Optional[Dict[str, Any]] = None,
    ) -> CouncilVerdict:
        tp_verdict = self._guard_tool_plan((context or {}).get("tool_plan") if context else None, intention)
        if tp_verdict:
            return tp_verdict
        # Deterministic robustness gate (non-ad-hoc):
        # If the plan explicitly pauses for human selection/confirmation, it is safe to approve.
        try:
            steps = plan_json.get("steps") if isinstance(plan_json, dict) else None
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    if str(step.get("action") or "").strip() == "await_user_input":
                        return CouncilVerdict(
                            verdict="approve",
                            reason="await_user_input_present",
                            suggested_fix="",
                            escalation_hint=None,
                            debug_notes="robustness_gate:explicit_human_confirmation_step",
                        )
        except Exception:
            pass
        consensus = self.review_plan_with_council(intention, plan_json, context=context, actions_catalog=actions_catalog)
        return consensus.to_verdict()
