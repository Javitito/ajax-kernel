from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _iso_utc(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _file_age_seconds(path: Path) -> Optional[int]:
    try:
        return int(max(0.0, time.time() - float(path.stat().st_mtime)))
    except Exception:
        return None


def _normalize_rail(raw: Optional[str]) -> str:
    val = (raw or "").strip().lower()
    return "prod" if val in {"prod", "production", "live"} else "lab"


def _normalize_risk_level(raw: Optional[str]) -> str:
    val = (raw or "").strip().lower()
    return val if val in {"low", "medium", "high"} else "medium"

def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_local_provider(cfg: Any) -> bool:
    if not isinstance(cfg, dict):
        return False
    kind = str(cfg.get("kind") or "").strip().lower()
    if kind == "static":
        return True
    base_url = cfg.get("base_url")
    if isinstance(base_url, str):
        url = base_url.strip().lower()
        if "localhost" in url or "127.0.0.1" in url:
            return True
    return False


def required_council_quorum(*, rail: str, risk_level: str) -> int:
    raw_override = (os.getenv("AJAX_COUNCIL_QUORUM") or os.getenv("AJAX_COUNCIL_QUORUM_REQUIRED") or "").strip()
    if raw_override:
        try:
            val = int(raw_override)
            if val in {1, 2}:
                return val
        except Exception:
            pass
    rail_n = _normalize_rail(rail)
    rl = _normalize_risk_level(risk_level)
    if rail_n == "prod":
        return 2
    if rl == "low":
        return 1
    return 2


def _providers_with_role(providers_cfg: Dict[str, Any], role: str) -> List[str]:
    role_l = (role or "").strip().lower()
    out: List[str] = []
    for name, cfg in (providers_cfg or {}).items():
        if not isinstance(cfg, dict) or cfg.get("disabled"):
            continue
        roles = cfg.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        roles_l = {str(r).strip().lower() for r in roles if r}
        if role_l in roles_l:
            out.append(str(name))
    return out


def _providers_with_vision_models(inventory: Dict[str, Any]) -> Set[str]:
    providers: Set[str] = set()
    for item in (inventory or {}).get("providers") or []:
        if not isinstance(item, dict):
            continue
        prov = str(item.get("provider") or "").strip()
        mods = item.get("modalities") or []
        if not prov or not isinstance(mods, list):
            continue
        if any(str(m).lower() == "vision" for m in mods):
            providers.add(prov)
    return providers


def _providers_with_static_vision_hint(providers_cfg: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for name, cfg in (providers_cfg or {}).items():
        if not isinstance(cfg, dict) or cfg.get("disabled"):
            continue
        kind = str(cfg.get("kind") or "").strip().lower()
        if kind == "static":
            continue
        static_models = cfg.get("static_models") or cfg.get("models") or []
        if not isinstance(static_models, list):
            continue
        for item in static_models:
            if isinstance(item, dict):
                mods = item.get("modalities") or []
                if isinstance(mods, list) and any(str(m).lower() == "vision" for m in mods):
                    out.add(str(name))
                    break
    return out


def _pick_model_for_provider(
    *,
    provider: str,
    cfg: Dict[str, Any],
    cost_mode: str,
    inventory_models: Optional[Set[str]],
) -> Tuple[Optional[str], List[str]]:
    """
    Elige modelo para un provider:
    - preferir cfg.models[cost_mode] si existe, luego default_model/model
    - si hay inventario y el modelo no existe, fallback al primero presente (determinista)
    Devuelve (model_id, notes[]).
    """
    notes: List[str] = []
    model_id: Optional[str] = None
    models_map = cfg.get("models")
    if isinstance(models_map, dict):
        model_id = models_map.get(cost_mode) or models_map.get(cost_mode.lower()) or None
    if not model_id:
        model_id = cfg.get("_selected_model") or cfg.get("default_model") or cfg.get("model")
    if not model_id:
        notes.append("model_missing")
        return None, notes
    if inventory_models is not None and model_id not in inventory_models:
        notes.append("model_not_in_inventory")
        if inventory_models:
            model_id = sorted(inventory_models)[0]
            notes.append("model_fallback_to_inventory_first")
    return str(model_id), notes


@dataclass(frozen=True)
class Player:
    provider: str
    model: Optional[str]
    tier: Optional[str]
    auth_state: Optional[str]
    availability: Optional[str]
    latency_ms: Optional[int]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "auth_state": self.auth_state,
            "availability": self.availability,
            "latency_ms": self.latency_ms,
            "notes": list(self.notes),
        }


def _player_from_status(
    *,
    provider: str,
    cfg: Dict[str, Any],
    status_doc: Dict[str, Any],
    role: str,
    cost_mode: str,
    inventory_by_provider: Dict[str, Set[str]],
) -> Player:
    st = ((status_doc or {}).get("providers") or {}).get(provider) if isinstance(status_doc, dict) else None
    st = st if isinstance(st, dict) else {}
    auth_state = st.get("auth_state")
    tier = cfg.get("tier")
    breathing = st.get("breathing") if isinstance(st, dict) else None
    probe = None
    if isinstance(breathing, dict):
        roles = breathing.get("roles")
        if isinstance(roles, dict):
            probe = roles.get(role)
    probe = probe if isinstance(probe, dict) else {}
    availability = probe.get("status") if isinstance(probe, dict) else None
    latency_ms = probe.get("latency_ms") if isinstance(probe, dict) else None
    try:
        latency_ms = int(latency_ms) if latency_ms is not None else None
    except Exception:
        latency_ms = None

    model_id, model_notes = _pick_model_for_provider(
        provider=provider,
        cfg=cfg,
        cost_mode=cost_mode,
        inventory_models=inventory_by_provider.get(provider),
    )
    notes: List[str] = []
    notes.extend(model_notes)
    return Player(
        provider=provider,
        model=model_id,
        tier=str(tier) if tier is not None else None,
        auth_state=str(auth_state) if auth_state is not None else None,
        availability=str(availability) if availability is not None else None,
        latency_ms=latency_ms,
        notes=notes,
    )


def _role_reason(primary: Optional[Dict[str, Any]], status_doc: Dict[str, Any]) -> str:
    """
    Motivo ultra-corto por rol.
    Formato estable: "auth_ok + p95=123ms + tier=cheap" (fallback a lat=... si no hay p95).
    """
    if not isinstance(primary, dict):
        return "no_primary"
    prov = primary.get("provider")
    if not isinstance(prov, str) or not prov.strip():
        return "no_primary"
    st = ((status_doc or {}).get("providers") or {}).get(prov) if isinstance(status_doc, dict) else None
    st = st if isinstance(st, dict) else {}
    auth_state = str(primary.get("auth_state") or st.get("auth_state") or "").strip().upper()
    auth_tok = "auth_ok" if auth_state == "OK" else f"auth_{auth_state.lower()}" if auth_state else "auth_unknown"

    p95 = None
    try:
        p95_raw = st.get("latency_p95_ms")
        if isinstance(p95_raw, (int, float)) and p95_raw > 0:
            p95 = int(p95_raw)
    except Exception:
        p95 = None
    lat = None
    try:
        lat_raw = primary.get("latency_ms")
        if isinstance(lat_raw, (int, float)) and lat_raw > 0:
            lat = int(lat_raw)
    except Exception:
        lat = None
    lat_tok = f"p95={p95}ms" if p95 is not None else f"lat={lat}ms" if lat is not None else "lat=unknown"

    tier = str(primary.get("tier") or "").strip().lower()
    tier_tok = f"tier={tier}" if tier else "tier=unknown"

    return f"{auth_tok} + {lat_tok} + {tier_tok}"


def _fallback_ladder(players: List[Any]) -> List[str]:
    """
    Orden determinista de sustitución (sin sorpresas): lista de "provider:model" (o "provider" si no hay modelo).
    """
    ladder: List[str] = []
    for p in players:
        if not isinstance(p, dict):
            continue
        prov = p.get("provider")
        if not isinstance(prov, str) or not prov.strip():
            continue
        model = p.get("model")
        rung = f"{prov}:{model}" if isinstance(model, str) and model.strip() else prov
        ladder.append(rung)
    return ladder


def _extract_inventory_maps(root_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Set[str]]]:
    inv_path = root_dir / "config" / "model_inventory_cloud.json"
    inv_doc = _safe_read_json(inv_path)
    inv_by_provider: Dict[str, Set[str]] = {}
    for item in inv_doc.get("providers") or []:
        if not isinstance(item, dict):
            continue
        prov = str(item.get("provider") or "").strip()
        mid = str(item.get("id") or "").strip()
        if not prov or not mid:
            continue
        inv_by_provider.setdefault(prov, set()).add(mid)
    return inv_doc, inv_by_provider


def _rank(
    pool: List[str],
    *,
    root_dir: Path,
    providers_cfg: Dict[str, Any],
    status_doc: Dict[str, Any],
    role: str,
    rail: str,
    risk_level: str,
    prefer_tier: str = "premium",
) -> List[str]:
    try:
        from agency import provider_ranker  # type: ignore
    except Exception:
        provider_ranker = None  # type: ignore
    if provider_ranker is None:
        return [p for p in pool if p in providers_cfg]
    try:
        scoreboard_doc: Dict[str, Any] = {}
        try:
            from agency import provider_scoreboard  # type: ignore

            scoreboard_doc = provider_scoreboard.load_scoreboard(
                Path(root_dir) / "artifacts" / "state" / "provider_scoreboard.json"
            )
        except Exception:
            scoreboard_doc = {}
        return provider_ranker.rank_providers(
            pool,
            providers_cfg=providers_cfg,
            status=status_doc,
            scoreboard=scoreboard_doc,
            prefer_tier=prefer_tier,
            role=role,
            rail=rail,
            risk_level=risk_level,
        )
    except Exception:
        return [p for p in pool if p in providers_cfg]


def _sync_inventory_best_effort(root_dir: Path) -> Dict[str, Any]:
    out = root_dir / "config" / "model_inventory_cloud.json"
    try:
        max_age = int(os.getenv("AJAX_STARTING_XI_INVENTORY_MAX_AGE_SEC", "3600") or 3600)
    except Exception:
        max_age = 3600
    if max_age > 0 and out.exists():
        age = _file_age_seconds(out)
        if age is not None and age <= max_age:
            return {"ok": True, "skipped": True, "reason": "fresh_inventory", "age_sec": age, "path": str(out)}
        # En preflight, por defecto NO refrescamos inventario si está stale (para no bloquear misiones).
        force = (os.getenv("AJAX_STARTING_XI_FORCE_INVENTORY_REFRESH") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not force:
            return {"ok": False, "reason": "stale_inventory", "age_sec": age, "path": str(out)}
    try:
        from agency import models_registry  # type: ignore
    except Exception:
        return {"ok": False, "reason": "models_registry_unavailable"}
    try:
        models = models_registry.discover_models()
    except Exception as exc:
        return {"ok": False, "reason": "discover_models_failed", "error": str(exc)[:200]}
    items: List[Dict[str, Any]] = []
    for m in models:
        items.append(
            {
                "provider": getattr(m, "provider", None),
                "id": getattr(m, "id", None),
                "source": getattr(m, "source", None),
                "modalities": getattr(m, "modalities", None),
                "raw": getattr(m, "raw", None),
            }
        )
    payload = {"providers": items, "fetched_at": _iso_utc()}
    try:
        _safe_write_json(out, payload)
        return {"ok": True, "count": len(items), "path": str(out)}
    except Exception as exc:
        return {"ok": False, "reason": "write_inventory_failed", "error": str(exc)[:200]}


def _run_breathing_best_effort(root_dir: Path, provider_configs: Dict[str, Any]) -> Dict[str, Any]:
    status_path = root_dir / "artifacts" / "health" / "providers_status.json"
    try:
        max_age = int(os.getenv("AJAX_STARTING_XI_BREATHING_MAX_AGE_SEC", "180") or 180)
    except Exception:
        max_age = 180
    if max_age > 0 and status_path.exists():
        doc = _safe_read_json(status_path)
        updated_at = doc.get("updated_at")
        age: Optional[int] = None
        try:
            if isinstance(updated_at, (int, float)) and float(updated_at) > 0:
                age = int(max(0.0, time.time() - float(updated_at)))
        except Exception:
            age = None
        if age is None:
            age = _file_age_seconds(status_path)
        if age is not None and age <= max_age:
            return {"ok": True, "skipped": True, "reason": "fresh_status", "age_sec": age, "path": str(status_path)}
        # En preflight, por defecto NO refrescamos breathing si está stale (para no bloquear misiones).
        force = (os.getenv("AJAX_STARTING_XI_FORCE_BREATHING_REFRESH") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not force:
            return {"ok": False, "reason": "stale_status", "age_sec": age, "path": str(status_path)}
    try:
        from agency.provider_breathing import ProviderBreathingLoop  # type: ignore
    except Exception:
        return {"ok": False, "reason": "provider_breathing_unavailable"}
    try:
        # Preflight debe ser rápido: cap de timeout por probe (evita bloquear la misión por providers lentos).
        try:
            timeout_cap = int(os.getenv("AJAX_STARTING_XI_BREATHING_TIMEOUT_CAP_SEC", "12") or 12)
        except Exception:
            timeout_cap = 12
        capped_cfg: Dict[str, Any] = {"providers": {}}
        providers = (provider_configs or {}).get("providers") if isinstance(provider_configs, dict) else None
        if isinstance(providers, dict):
            for name, cfg in providers.items():
                if not isinstance(cfg, dict):
                    continue
                cfg2 = dict(cfg)
                try:
                    raw_to = int(cfg2.get("timeout_seconds") or 0)
                except Exception:
                    raw_to = 0
                if raw_to > 0:
                    cfg2["timeout_seconds"] = min(raw_to, timeout_cap)
                else:
                    cfg2["timeout_seconds"] = timeout_cap
                capped_cfg["providers"][name] = cfg2
        loop = ProviderBreathingLoop(root_dir=root_dir, provider_configs=capped_cfg)
        loop.run_once(roles=["brain", "council", "scout", "vision"])
        return {"ok": True, "timeout_cap_sec": timeout_cap, "path": str(status_path)}
    except Exception as exc:
        return {"ok": False, "reason": "provider_breathing_failed", "error": str(exc)[:200]}


def _summarize_fix_hints(hints: Iterable[str]) -> str:
    ordered = []
    for k in ("auth", "quota", "timeout"):
        if k in hints:
            ordered.append(k)
    return "/".join(ordered) if ordered else "none"


def build_starting_xi(
    *,
    root_dir: Path,
    provider_configs: Dict[str, Any],
    rail: str,
    risk_level: str,
    cost_mode: str,
    run_breathing: bool = True,
    run_inventory_sync: bool = True,
    vision_required: bool = True,
) -> Dict[str, Any]:
    """
    Construye Starting XI por rol y lo persiste en artifacts/health/starting_xi.json.
    Devuelve el payload completo (incluye missing_players si aplica).
    """
    rail_n = _normalize_rail(rail)
    risk_n = _normalize_risk_level(risk_level)
    cost_n = (cost_mode or "premium").strip().lower()

    providers_cfg = (provider_configs or {}).get("providers") or {}
    if not isinstance(providers_cfg, dict):
        providers_cfg = {}

    # Policy (budgeter) para orden de preferencia por rail/rol.
    policy_doc: Dict[str, Any] = {}
    pref_brain: List[str] = []
    pref_council: List[str] = []
    pref_vision: List[str] = []
    exclude_prefixes: List[str] = []
    allow_premium_codex = _env_truthy("AJAX_ALLOW_PREMIUM_CODEX")
    try:
        from agency import provider_policy as provider_policy_mod  # type: ignore
    except Exception:
        provider_policy_mod = None  # type: ignore
    if provider_policy_mod is not None:
        try:
            policy_doc = provider_policy_mod.load_provider_policy(root_dir) or {}
            pref_brain = provider_policy_mod.preferred_providers(policy_doc, rail=rail_n, role="brain")
            pref_council = provider_policy_mod.preferred_providers(policy_doc, rail=rail_n, role="council")
            pref_vision = provider_policy_mod.preferred_providers(policy_doc, rail=rail_n, role="vision")
            defaults = policy_doc.get("defaults") if isinstance(policy_doc, dict) else {}
            if cost_n == "save_codex" and not allow_premium_codex:
                if isinstance(defaults, dict):
                    exclude_prefixes = defaults.get("save_codex_exclude_prefixes") or []
                if not isinstance(exclude_prefixes, list) or not exclude_prefixes:
                    exclude_prefixes = ["codex_"]
                exclude_prefixes = [str(p) for p in exclude_prefixes if str(p)]
        except Exception:
            policy_doc = {}
            pref_brain = []
            pref_council = []
            pref_vision = []
    strict_order = {
        "brain": bool(pref_brain),
        "council": bool(pref_council),
        "vision": bool(pref_vision),
    }

    # Ledger (no LLM): disponibilidad + cooldown por provider/model/role.
    ledger_doc: Dict[str, Any] = {}
    ledger_result: Dict[str, Any] = {"ok": False, "reason": "not_run"}
    ok_by_role: Dict[str, set[str]] = {}
    try:
        from agency.provider_ledger import ProviderLedger  # type: ignore
    except Exception:
        ProviderLedger = None  # type: ignore
    if ProviderLedger is not None:
        try:
            ledger_doc = ProviderLedger(root_dir=root_dir, provider_configs=provider_configs).refresh(timeout_s=1.5) or {}
            ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else None
            ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
            ok_by_role = {
                "brain": set(ProviderLedger.ok_providers(ledger_rows, role="brain")),
                "council": set(ProviderLedger.ok_providers(ledger_rows, role="council")),
                "vision": set(ProviderLedger.ok_providers(ledger_rows, role="vision")),
                "scout": set(ProviderLedger.ok_providers(ledger_rows, role="scout")),
            }
            ledger_result = {
                "ok": True,
                "path": ledger_doc.get("path"),
                "updated_ts": ledger_doc.get("updated_ts"),
                "updated_utc": ledger_doc.get("updated_utc"),
            }
        except Exception as exc:
            ledger_doc = {}
            ledger_result = {"ok": False, "reason": "ledger_failed", "error": str(exc)[:200]}

    breathing_result = _run_breathing_best_effort(root_dir, provider_configs) if run_breathing else {"ok": False, "reason": "disabled"}
    inventory_result = _sync_inventory_best_effort(root_dir) if run_inventory_sync else {"ok": False, "reason": "disabled"}

    status_doc = _safe_read_json(root_dir / "artifacts" / "health" / "providers_status.json")
    inv_doc, inv_by_provider = _extract_inventory_maps(root_dir)

    allow_local_text = _env_truthy("AJAX_ALLOW_LOCAL_TEXT") or cost_n == "emergency"
    allow_local_vision = True
    allow_local_override_env = None
    if _env_truthy("AJAX_ALLOW_LOCAL_TEXT"):
        allow_local_override_env = "AJAX_ALLOW_LOCAL_TEXT"
    elif cost_n == "emergency":
        allow_local_override_env = "cost_mode_emergency"
    policy_providers = policy_doc.get("providers") if isinstance(policy_doc, dict) else None
    policy_providers = policy_providers if isinstance(policy_providers, dict) else {}

    vision_role_pool = _providers_with_role(providers_cfg, "vision")
    vision_fallback = _providers_with_vision_models(inv_doc) | _providers_with_static_vision_hint(providers_cfg)
    vision_pool = list(vision_role_pool)
    for p in sorted(vision_fallback):
        if p in vision_pool:
            continue
        vision_pool.append(p)

    def _apply_preference(candidates: List[str], preference: List[str]) -> List[str]:
        ordered: List[str] = []
        for p in preference:
            if p in candidates and p not in ordered:
                ordered.append(p)
        for p in candidates:
            if p not in ordered:
                ordered.append(p)
        return ordered

    def _invokable_provider(name: str) -> bool:
        cfg = providers_cfg.get(name) or {}
        kind = str(cfg.get("kind") or "").strip().lower()
        return bool(kind and kind != "static")

    vision_pool = [p for p in vision_pool if p in providers_cfg and not providers_cfg.get(p, {}).get("disabled") and _invokable_provider(p)]

    brain_pool = _apply_preference(_providers_with_role(providers_cfg, "brain"), pref_brain)
    council_pool = _apply_preference(_providers_with_role(providers_cfg, "council"), pref_council)
    vision_pool = _apply_preference(vision_pool, pref_vision)
    if exclude_prefixes:
        brain_pool = [p for p in brain_pool if not any(p.startswith(prefix) for prefix in exclude_prefixes)]
        council_pool = [p for p in council_pool if not any(p.startswith(prefix) for prefix in exclude_prefixes)]
        vision_pool = [p for p in vision_pool if not any(p.startswith(prefix) for prefix in exclude_prefixes)]
    if bool(ledger_result.get("ok")):
        # Regla: status!=ok => no cuenta para quorum; se filtra del roster.
        brain_ok = ok_by_role.get("brain") or set()
        council_ok = ok_by_role.get("council") or set()
        vision_ok = ok_by_role.get("vision") or set()
        brain_pool = [p for p in brain_pool if p in brain_ok]
        council_pool = [p for p in council_pool if p in council_ok]
        vision_pool = [p for p in vision_pool if p in vision_ok]
    quorum = required_council_quorum(rail=rail_n, risk_level=risk_n)

    def _filter_pool(
        pool: List[str],
        *,
        role: str,
        include_local_text: bool,
        include_local_vision: bool,
    ) -> List[str]:
        out: List[str] = []
        for p in pool:
            if p not in providers_cfg:
                continue
            cfg = providers_cfg.get(p) or {}
            if not isinstance(cfg, dict) or cfg.get("disabled"):
                continue
            ent = policy_providers.get(p)
            if isinstance(ent, dict):
                state = str(ent.get("policy_state") or "").strip().lower()
                cap_key = "policy_state_vision" if role == "vision" else "policy_state_text"
                cap_state = str(ent.get(cap_key) or state).strip().lower()
                if cap_state in {"disallowed", "blocked", "deny"}:
                    if role == "vision":
                        continue
                    if not include_local_text:
                        continue
            if not _invokable_provider(p):
                continue
            if _is_local_provider(cfg):
                if role == "vision":
                    if not include_local_vision:
                        continue
                else:
                    if not include_local_text:
                        continue
            out.append(p)
        return out

    def _mk_player(provider: Optional[str], role: str) -> Optional[Dict[str, Any]]:
        if not provider:
            return None
        cfg = providers_cfg.get(provider)
        if not isinstance(cfg, dict):
            return None
        return _player_from_status(
            provider=provider,
            cfg=cfg,
            status_doc=status_doc,
            role=role,
            cost_mode=cost_n,
            inventory_by_provider=inv_by_provider,
        ).to_dict()

    def _build_payload(
        *,
        include_local_text: bool,
        include_local_vision: bool,
        cloud_quorum_failed: bool,
        cloud_only_missing: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        brain_pool_eff = _filter_pool(brain_pool, role="brain", include_local_text=include_local_text, include_local_vision=False)
        council_pool_eff = _filter_pool(
            council_pool, role="council", include_local_text=include_local_text, include_local_vision=False
        )
        vision_pool_eff = _filter_pool(
            vision_pool, role="vision", include_local_text=False, include_local_vision=include_local_vision
        )

        brain_ranked = (
            list(brain_pool_eff)
            if strict_order.get("brain")
            else _rank(
                brain_pool_eff,
                root_dir=root_dir,
                providers_cfg=providers_cfg,
                status_doc=status_doc,
                role="brain",
                rail=rail_n,
                risk_level=risk_n,
                prefer_tier=cost_n,
            )
        )
        council_ranked = (
            list(council_pool_eff)
            if strict_order.get("council")
            else _rank(
                council_pool_eff,
                root_dir=root_dir,
                providers_cfg=providers_cfg,
                status_doc=status_doc,
                role="council",
                rail=rail_n,
                risk_level=risk_n,
                prefer_tier=cost_n,
            )
        )
        vision_ranked = (
            list(vision_pool_eff)
            if strict_order.get("vision")
            else _rank(
                vision_pool_eff,
                root_dir=root_dir,
                providers_cfg=providers_cfg,
                status_doc=status_doc,
                role="vision",
                rail=rail_n,
                risk_level=risk_n,
                prefer_tier=cost_n,
            )
        )

        brain_primary = brain_ranked[0] if brain_ranked else None
        brain_bench = [p for p in brain_ranked if p and p != brain_primary] if brain_primary else []

        council1_primary = council_ranked[0] if council_ranked else None
        council2_primary = None
        for cand in council_ranked[1:]:
            if cand != council1_primary:
                council2_primary = cand
                break
        council1_bench = (
            [p for p in council_ranked if p and p not in {council1_primary, council2_primary}]
            if council1_primary
            else []
        )
        council2_bench = (
            [p for p in council_ranked if p and p not in {council1_primary, council2_primary}]
            if council2_primary
            else []
        )

        vision_primary = vision_ranked[0] if vision_ranked else None
        vision_bench = [p for p in vision_ranked if p and p != vision_primary] if vision_primary else []

        payload: Dict[str, Any] = {
            "schema": "ajax.starting_xi.v1",
            "created_utc": _iso_utc(),
            "rail": rail_n,
            "risk_level": risk_n,
            "cost_mode": cost_n,
            "policy": {
                "online_first": True,
                "cloud_first": True,
                "include_local_text": bool(include_local_text),
                "include_local_vision": bool(include_local_vision),
                "allow_local_text": bool(allow_local_text),
                "allow_local_override_env": allow_local_override_env,
                "cloud_quorum_failed": bool(cloud_quorum_failed),
                "vision_required": bool(vision_required),
            },
            "inputs": {
                "providers_status_path": str(root_dir / "artifacts" / "health" / "providers_status.json"),
                "provider_ledger_path": str(root_dir / "artifacts" / "provider_ledger" / "latest.json"),
                "model_inventory_path": str(root_dir / "config" / "model_inventory_cloud.json"),
            },
            "preflight": {
                "ledger": ledger_result,
                "breathing": breathing_result,
                "inventory": inventory_result,
            },
            "quorum": {"council_required": quorum},
            "brain": {},
            "council": {"role1": {}, "role2": {}},
            "vision": {},
            "missing_players": [],
            "optional_missing_players": [],
            "fix_hints": "none",
        }
        if cloud_only_missing is not None:
            payload["policy"]["cloud_only_missing_players"] = cloud_only_missing

        brain_primary_p = _mk_player(brain_primary, "brain")
        brain_bench_p: List[Dict[str, Any]] = []
        for p in brain_bench:
            if not p:
                continue
            player = _mk_player(p, "brain")
            if isinstance(player, dict):
                brain_bench_p.append(player)
        payload["brain"] = {
            "primary": brain_primary_p,
            "bench": brain_bench_p,
            "reason": _role_reason(brain_primary_p, status_doc),
            "fallback_ladder": _fallback_ladder(brain_bench_p),
        }

        c1_primary_p = _mk_player(council1_primary, "council")
        c1_bench_p: List[Dict[str, Any]] = []
        for p in council1_bench:
            if not p:
                continue
            player = _mk_player(p, "council")
            if isinstance(player, dict):
                c1_bench_p.append(player)
        payload["council"]["role1"] = {
            "primary": c1_primary_p,
            "bench": c1_bench_p,
            "reason": _role_reason(c1_primary_p, status_doc),
            "fallback_ladder": _fallback_ladder(c1_bench_p),
        }

        c2_primary_p = _mk_player(council2_primary, "council")
        c2_bench_p: List[Dict[str, Any]] = []
        for p in council2_bench:
            if not p:
                continue
            player = _mk_player(p, "council")
            if isinstance(player, dict):
                c2_bench_p.append(player)
        payload["council"]["role2"] = {
            "primary": c2_primary_p,
            "bench": c2_bench_p,
            "reason": _role_reason(c2_primary_p, status_doc),
            "fallback_ladder": _fallback_ladder(c2_bench_p),
        }

        vision_primary_p = _mk_player(vision_primary, "vision")
        vision_bench_p: List[Dict[str, Any]] = []
        for p in vision_bench:
            if not p:
                continue
            player = _mk_player(p, "vision")
            if isinstance(player, dict):
                vision_bench_p.append(player)
        payload["vision"] = {
            "primary": vision_primary_p,
            "bench": vision_bench_p,
            "reason": _role_reason(vision_primary_p, status_doc),
            "fallback_ladder": _fallback_ladder(vision_bench_p),
        }

        missing: List[Dict[str, Any]] = []
        optional_missing: List[Dict[str, Any]] = []
        fix_hints: Set[str] = set()
        if brain_primary_p is None:
            missing.append({"role": "brain", "reason": "no_viable_provider"})
        if c1_primary_p is None:
            missing.append({"role": "council.role1", "reason": "no_viable_provider"})
        if quorum >= 2 and c2_primary_p is None:
            missing.append({"role": "council.role2", "reason": "quorum_2_requires_second_provider"})
        if vision_primary_p is None and vision_required:
            missing.append({"role": "vision", "reason": "no_viable_provider"})
        elif vision_primary_p is None:
            optional_missing.append({"role": "vision", "reason": "no_viable_provider"})

        # Hints desde status (auth/quota/timeout) para los roles que fallaron
        providers_status = (status_doc.get("providers") if isinstance(status_doc, dict) else {}) or {}
        for m in [*missing, *optional_missing]:
            role = str(m.get("role") or "")
            pool = brain_pool_eff if role == "brain" else council_pool_eff if role.startswith("council") else vision_pool_eff
            for prov in pool:
                st = providers_status.get(prov) if isinstance(providers_status, dict) else None
                if not isinstance(st, dict):
                    continue
                astate = str(st.get("auth_state") or "").upper()
                if astate in {"MISSING", "EXPIRED"}:
                    fix_hints.add("auth")
                breathing = st.get("breathing")
                roles = breathing.get("roles") if isinstance(breathing, dict) else None
                if isinstance(roles, dict):
                    probe = roles.get("brain" if role == "brain" else "council" if role.startswith("council") else "vision")
                    if isinstance(probe, dict):
                        reason = str(probe.get("reason") or "").lower()
                        err = str(probe.get("error") or "").lower()
                        if any(tok in reason or tok in err for tok in ("429", "quota", "rate limit", "rate_limit")):
                            fix_hints.add("quota")
                        if any(tok in reason or tok in err for tok in ("timeout", "timed out", "latency")):
                            fix_hints.add("timeout")

        payload["missing_players"] = missing
        payload["optional_missing_players"] = optional_missing
        payload["fix_hints"] = _summarize_fix_hints(fix_hints)
        return payload

    payload = _build_payload(
        include_local_text=allow_local_text,
        include_local_vision=allow_local_vision,
        cloud_quorum_failed=False,
    )
    if allow_local_text and payload.get("missing_players"):
        cloud_only_missing = payload.get("missing_players") if isinstance(payload.get("missing_players"), list) else None
        payload = _build_payload(
            include_local_text=True,
            include_local_vision=allow_local_vision,
            cloud_quorum_failed=True,
            cloud_only_missing=cloud_only_missing,
        )

    out_path = root_dir / "artifacts" / "health" / "starting_xi.json"
    _safe_write_json(out_path, payload)
    payload["path"] = str(out_path)
    return payload


def format_console_lines(starting_xi: Dict[str, Any]) -> List[str]:
    """
    Devuelve exactamente 4 líneas para consola (Starting XI / Bench / Risk / Fix).
    """
    def _fmt_player(p: Any) -> str:
        if not isinstance(p, dict):
            return "-"
        prov = str(p.get("provider") or "-")
        mid = p.get("model")
        return f"{prov}:{mid}" if mid else prov

    brain = _fmt_player(((starting_xi.get("brain") or {}).get("primary")))
    c1 = _fmt_player((((starting_xi.get("council") or {}).get("role1") or {}).get("primary")))
    c2 = _fmt_player((((starting_xi.get("council") or {}).get("role2") or {}).get("primary")))
    vis = _fmt_player(((starting_xi.get("vision") or {}).get("primary")))

    def _fmt_bench(xs: Any) -> str:
        if not isinstance(xs, list) or not xs:
            return "-"
        try:
            max_items = int(os.getenv("AJAX_STARTING_XI_BENCH_MAX", "3") or 3)
        except Exception:
            max_items = 3
        items = [_fmt_player(x) for x in xs]
        if len(items) > max_items:
            return ",".join(items[:max_items]) + ",…"
        return ",".join(items)

    b_bench = _fmt_bench(((starting_xi.get("brain") or {}).get("bench")))
    c1_bench = _fmt_bench((((starting_xi.get("council") or {}).get("role1") or {}).get("bench")))
    c2_bench = _fmt_bench((((starting_xi.get("council") or {}).get("role2") or {}).get("bench")))
    v_bench = _fmt_bench(((starting_xi.get("vision") or {}).get("bench")))

    rail = str(starting_xi.get("rail") or "lab")
    risk = str(starting_xi.get("risk_level") or "medium")
    quorum = ((starting_xi.get("quorum") or {}).get("council_required")) or 0
    fix = str(starting_xi.get("fix_hints") or "none")

    return [
        f"Starting XI: Brain={brain} Council1={c1} Council2={c2} Vision={vis}",
        f"Bench: Brain={b_bench} Council1={c1_bench} Council2={c2_bench} Vision={v_bench}",
        f"Risk/rail: {rail} {risk} -> council_quorum={quorum}",
        f"If blocked: fix={fix} (auth/quota/timeout)",
    ]
