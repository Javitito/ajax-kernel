from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_WINDOW_EVENTS = 50
_RATE_WINDOW_EVENTS = 20
_TIMEOUT_TOKENS = ("timeout", "timed out", "read timed out", "etimedout")


def _pick_model_id(cfg: Any) -> Optional[str]:
    if not isinstance(cfg, dict):
        return None
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


def load_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"providers": {}, "updated_at": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("providers", {})
            return data
    except Exception:
        pass
    return {"providers": {}, "updated_at": None}


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    # p in [0,100]
    p = max(0.0, min(100.0, p))
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def _is_timeout_event(ev: Any) -> bool:
    if not isinstance(ev, dict):
        return False
    msg = str(ev.get("error") or ev.get("outcome") or "").lower()
    return any(tok in msg for tok in _TIMEOUT_TOKENS)


def _compute_rates(events: List[Any], window: int = _RATE_WINDOW_EVENTS) -> tuple[Optional[float], Optional[float]]:
    if not events:
        return None, None
    recent = events[-window:] if len(events) >= window else list(events)
    total = len(recent)
    if total <= 0:
        return None, None
    failures = 0
    timeouts = 0
    for ev in recent:
        if isinstance(ev, dict) and ev.get("ok") is False:
            failures += 1
        if _is_timeout_event(ev):
            timeouts += 1
    return timeouts / total, failures / total


def record_attempt(
    path: Path,
    *,
    provider: str,
    ok: bool,
    latency_ms: Optional[int],
    ttft_ms: Optional[int] = None,
    total_ms: Optional[int] = None,
    outcome: str,
    error: Optional[str] = None,
) -> None:
    data = load_status(path)
    providers = data.setdefault("providers", {})
    entry = providers.setdefault(provider, {})
    events = entry.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        entry["events"] = events

    total_val = None
    if isinstance(total_ms, (int, float)) and total_ms > 0:
        total_val = int(total_ms)
    elif isinstance(latency_ms, (int, float)) and latency_ms > 0:
        total_val = int(latency_ms)
    ttft_val = int(ttft_ms) if isinstance(ttft_ms, (int, float)) and ttft_ms > 0 else None
    events.append(
        {
            "ts": time.time(),
            "ok": bool(ok),
            "latency_ms": int(latency_ms) if isinstance(latency_ms, int) else None,
            "total_ms": total_val,
            "ttft_ms": ttft_val,
            "outcome": str(outcome or ""),
            "error": str(error)[:200] if error else None,
        }
    )
    if len(events) > _WINDOW_EVENTS:
        entry["events"] = events[-_WINDOW_EVENTS:]
        events = entry["events"]

    latencies: List[float] = []
    totals: List[float] = []
    ttfts: List[float] = []
    ok_events = 0
    recent = events[-10:] if len(events) >= 10 else list(events)
    for ev in recent:
        if isinstance(ev, dict) and ev.get("ok") is True:
            ok_events += 1
        lm = ev.get("latency_ms") if isinstance(ev, dict) else None
        if isinstance(lm, (int, float)) and lm > 0:
            latencies.append(float(lm))
        tm = ev.get("total_ms") if isinstance(ev, dict) else None
        if isinstance(tm, (int, float)) and tm > 0:
            totals.append(float(tm))
        tt = ev.get("ttft_ms") if isinstance(ev, dict) else None
        if isinstance(tt, (int, float)) and tt > 0:
            ttfts.append(float(tt))
    p95 = _percentile(latencies, 95.0)
    p95_total = _percentile(totals or latencies, 95.0)
    p95_ttft = _percentile(ttfts, 95.0)
    entry["available_recent"] = ok_events > 0
    entry["latency_p95_ms"] = int(p95) if p95 is not None else None
    entry["total_p95_ms"] = int(p95_total) if p95_total is not None else None
    entry["ttft_p95_ms"] = int(p95_ttft) if p95_ttft is not None else None
    timeout_rate, failure_rate = _compute_rates(events, window=_RATE_WINDOW_EVENTS)
    entry["timeout_rate_recent"] = round(timeout_rate, 4) if isinstance(timeout_rate, float) else None
    entry["failure_rate_recent"] = round(failure_rate, 4) if isinstance(failure_rate, float) else None

    data["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _get_auth_state(st: Any) -> Optional[str]:
    if not isinstance(st, dict):
        return None
    raw = st.get("auth_state") or (st.get("auth") or {}).get("state")
    if not raw:
        return None
    val = str(raw).strip().upper()
    return val if val in {"OK", "MISSING", "EXPIRED"} else None


def _role_probe(st: Dict[str, Any], role: str) -> Optional[Dict[str, Any]]:
    breathing = st.get("breathing")
    if not isinstance(breathing, dict):
        return None
    roles = breathing.get("roles")
    if not isinstance(roles, dict):
        return None
    probe = roles.get(role)
    return probe if isinstance(probe, dict) else None


def rank_providers(
    pool: List[str],
    *,
    providers_cfg: Dict[str, Any],
    status: Dict[str, Any],
    scoreboard: Optional[Dict[str, Any]] = None,
    prefer_tier: str = "premium",
    role: str = "council",
    rail: str = "lab",
    risk_level: str = "medium",
) -> List[str]:
    """
    Ranker v0.1 (auth-aware):
    - auth_state MISSING/EXPIRED => excluye provider
    - breathing probe status por rol (UP/DEGRADED/DOWN) ajusta score
    - penaliza timeout_rate_recent y latency_p95_ms
    - tier preference (+1/0/-1)
    Tie-breaker: original pool order.
    """
    prefer_tier = (prefer_tier or "premium").lower()
    tier_bias = {"cheap": 0, "balanced": 0, "premium": 0}
    if prefer_tier == "premium":
        tier_bias = {"premium": 2, "balanced": 1, "cheap": 0}
    elif prefer_tier == "balanced":
        tier_bias = {"balanced": 2, "premium": 1, "cheap": 0}
    elif prefer_tier == "emergency":
        tier_bias = {"cheap": 2, "balanced": 1, "premium": 0}
    else:
        tier_bias = {"balanced": 2, "premium": 1, "cheap": 0}

    providers_status = (status or {}).get("providers") if isinstance(status, dict) else {}
    if not isinstance(providers_status, dict):
        providers_status = {}

    scored: List[tuple[int, int, str]] = []
    role_n = (role or "council").strip().lower()
    allow_local_text = _env_truthy("AJAX_ALLOW_LOCAL_TEXT") or prefer_tier == "emergency"
    allow_local_vision = role_n == "vision"
    rail_n = (rail or "lab").strip().lower()
    rl = (risk_level or "medium").strip().lower()
    for idx, name in enumerate(pool):
        cfg = providers_cfg.get(name)
        if not isinstance(cfg, dict):
            continue
        if cfg.get("disabled"):
            continue
        tier = str(cfg.get("tier") or "balanced").lower()
        score = 0
        st = providers_status.get(name) if isinstance(providers_status, dict) else None

        auth_state = _get_auth_state(st)
        if auth_state in {"MISSING", "EXPIRED"}:
            # Excluir solo si el provider realmente requiere auth web.
            # Si el status rolling está desalineado con la cfg actual, no debe vetar providers "local/cli".
            try:
                from agency.auth_manager import AuthManager  # type: ignore

                if AuthManager.is_web_auth_required(cfg):
                    continue
            except Exception:
                continue

        probe = _role_probe(st, role_n) if isinstance(st, dict) else None
        probe_status = str(probe.get("status") or "").upper() if isinstance(probe, dict) else ""
        if probe_status == "UP":
            score += 3
        elif probe_status == "DEGRADED":
            score += 1
        elif probe_status == "DOWN":
            score -= 3

        if isinstance(st, dict) and st.get("available_recent") is True:
            score += 2
        p95 = st.get("latency_p95_ms") if isinstance(st, dict) else None
        if isinstance(p95, (int, float)):
            if p95 < 1500:
                score += 2
            elif p95 < 3000:
                score += 1
            elif p95 > 12000:
                score -= 2
            elif p95 > 6000:
                score -= 1
        tr = st.get("timeout_rate_recent") if isinstance(st, dict) else None
        if isinstance(tr, (int, float)):
            if tr >= 0.5:
                score -= 6
            elif tr >= 0.2:
                score -= 3
            elif tr >= 0.1:
                score -= 1

        if isinstance(scoreboard, dict) and scoreboard:
            try:
                from agency import provider_scoreboard  # type: ignore

                model_id = _pick_model_id(cfg)
                min_samples = int(os.getenv("AJAX_SCOREBOARD_MIN_SAMPLES", "3") or 3)
                cooldown_minutes = int(os.getenv("AJAX_SCOREBOARD_COOLDOWN_MIN", "15") or 15)
                state = provider_scoreboard.promotion_state(
                    scoreboard,
                    provider=name,
                    model=model_id,
                    min_samples=min_samples,
                    cooldown_minutes=cooldown_minutes,
                )
                if state.get("eligible") is False:
                    score -= 6
                elif state.get("reorder_allowed"):
                    sb_score = provider_scoreboard.score_for(scoreboard, provider=name, model=model_id)
                    if isinstance(sb_score, (int, float)):
                        score += float(sb_score) * 8.0
            except Exception:
                pass

        score += tier_bias.get(tier, 0)
        # Online-first: providers locales solo entran con override explícito/emergency.
        if _is_local_provider(cfg):
            if role_n == "vision":
                if not allow_local_vision:
                    continue
            else:
                if not allow_local_text:
                    continue
        # PROD/high risk: weak bias away from cheap
        if rail_n == "prod" or rl in {"high"}:
            if tier == "cheap":
                score -= 1
            if tier == "premium":
                score += 1
        scored.append((score, -idx, name))

    scored.sort(reverse=True)
    return [name for _, __, name in scored]


def recommended_timeout_seconds(
    *,
    provider_name: str,
    provider_cfg: Dict[str, Any],
    status: Optional[Dict[str, Any]],
    role: str,
    rail: str,
    risk_level: str,
) -> int:
    """
    Timeout adaptativo (v0):
    - base: cfg.timeout_seconds o tier default
    - si breathing probe DEGRADED con latency_ms => subir
    - si PROD/high => subir mínimo
    """
    tier = str(provider_cfg.get("tier") or "balanced").lower()
    base = provider_cfg.get("timeout_seconds")
    if isinstance(base, (int, float)):
        timeout = int(base)
    else:
        timeout = 60 if tier == "premium" else 30 if tier == "balanced" else 20

    role_n = (role or "").strip().lower()
    if role_n == "brain":
        timeout = max(timeout, 20)
    if role_n == "council":
        timeout = max(timeout, 20)

    rail_n = (rail or "lab").strip().lower()
    rl = (risk_level or "medium").strip().lower()
    if rail_n == "prod" or rl in {"high"}:
        timeout = max(timeout, 30)

    st = None
    try:
        st = ((status or {}).get("providers") or {}).get(provider_name)
    except Exception:
        st = None
    probe = _role_probe(st, role_n) if isinstance(st, dict) else None
    if isinstance(probe, dict) and str(probe.get("status") or "").upper() == "DEGRADED":
        lm = probe.get("latency_ms")
        if isinstance(lm, (int, float)) and lm > 0:
            timeout = max(timeout, int(lm / 1000) + 5)
        else:
            timeout = int(timeout * 1.5)

    # Bench p95 (total) from providers_status -> use as floor.
    if isinstance(st, dict):
        p95_total = st.get("total_p95_ms")
        if not isinstance(p95_total, (int, float)) or p95_total <= 0:
            p95_total = st.get("latency_p95_ms")
        if isinstance(p95_total, (int, float)) and p95_total > 0:
            timeout = max(timeout, int(p95_total / 1000) + 5)

    return max(5, min(int(timeout), 120))
