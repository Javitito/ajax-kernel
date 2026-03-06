from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agency.auth_manager import AuthManager

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


_BLOCKED_REASON_CODES = {
    "auth_missing",
    "auth_invalid",
    "auth_expired",
    "cli_not_installed",
    "config_missing",
    "provider_down",
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
    root = Path(root_dir).resolve()
    data = _read_doc(root / "config" / "model_providers.yaml")
    if not data:
        data = _read_doc(root / "config" / "model_providers.json")
    providers = data.get("providers") if isinstance(data, dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(providers, dict):
        return out
    for provider, cfg in providers.items():
        if not isinstance(provider, str) or not isinstance(cfg, dict):
            continue
        if str(cfg.get("kind") or "").strip().lower() == "static":
            continue
        out[provider] = dict(cfg)
    return out


def _load_providers_status(root_dir: Path) -> Dict[str, Any]:
    doc = _read_doc(Path(root_dir).resolve() / "artifacts" / "health" / "providers_status.json")
    providers = doc.get("providers")
    if isinstance(providers, dict):
        return providers
    return {}


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


def _runtime_identity(root_dir: Path) -> Dict[str, str]:
    root = Path(root_dir).resolve()
    return {
        "ajaxctl_path": str((root / "bin" / "ajaxctl").resolve()),
        "platform": platform.platform(),
        "git_head": _git_head(root),
    }


def _auth_state_public(raw: str) -> str:
    state = str(raw or "").strip().upper()
    if state == "OK":
        return "present"
    if state == "MISSING":
        return "missing"
    if state == "EXPIRED":
        return "expired"
    return "unknown"


def _sanitize_text(text: str) -> str:
    raw = str(text or "")
    redacted = re.sub(
        r"(?i)\b(api[_-]?key|token|authorization)\b\s*[:=]\s*([^\s,;]+)",
        r"\1=<redacted>",
        raw,
    )
    return redacted[:240]


def _classify_reason_code(raw_reason: str) -> str:
    reason = str(raw_reason or "").strip().lower()
    if not reason:
        return "ok"
    if "auth_expired" in reason or ("expired" in reason and "auth" in reason):
        return "auth_expired"
    if "env_missing" in reason or "auth_missing" in reason:
        return "auth_missing"
    if "invalid_api_key" in reason or "http_401" in reason or "unauthorized" in reason or "http_403" in reason:
        return "auth_invalid"
    if "cli_missing" in reason or "not recognized as an internal or external command" in reason:
        return "cli_not_installed"
    if "command not found" in reason or "no such file or directory" in reason:
        return "cli_not_installed"
    if "http_429" in reason or "quota" in reason or "rate_limit" in reason or "rate limit" in reason:
        return "quota_exhausted"
    if "timeout" in reason or "timed out" in reason:
        return "provider_timeout"
    if "config_missing" in reason or "provider_missing_in_config" in reason:
        return "config_missing"
    if "connection refused" in reason or "transport_down" in reason or "provider_down" in reason:
        return "provider_down"
    if "down" in reason:
        return "provider_down"
    return "unknown_provider_failure"


def _reason_to_reachability(reason_code: str, status_hint: Optional[str] = None) -> str:
    status = str(status_hint or "").strip().upper()
    if status == "UP" and reason_code == "ok":
        return "ok"
    if reason_code == "provider_timeout":
        return "timeout"
    if reason_code in {"ok"}:
        return "ok"
    if reason_code in {
        "auth_missing",
        "auth_invalid",
        "auth_expired",
        "cli_not_installed",
        "provider_down",
        "quota_exhausted",
    }:
        return "down"
    return "unknown"


def _reason_to_quota(reason_code: str) -> str:
    if reason_code == "quota_exhausted":
        return "limited"
    if reason_code in {"ok", "provider_timeout", "provider_down", "cli_not_installed"}:
        return "ok"
    return "unknown"


def _effective_result(*, auth_state: str, reachability_state: str, reason_code: str) -> str:
    if reason_code in _BLOCKED_REASON_CODES:
        return "blocked"
    if auth_state in {"missing", "expired"}:
        return "blocked"
    if reachability_state == "ok" and reason_code == "ok":
        return "usable"
    if reason_code in {"provider_timeout", "quota_exhausted"}:
        return "degraded"
    if reachability_state in {"timeout"}:
        return "degraded"
    if reachability_state in {"down"}:
        return "blocked"
    return "degraded"


def _next_hint(provider: str, cfg: Dict[str, Any], reason_code: str) -> str:
    api_key_env = str(cfg.get("api_key_env") or "").strip()
    kind = str(cfg.get("kind") or "").strip().lower()
    if reason_code == "auth_missing":
        if api_key_env:
            return f"set {api_key_env}=<token> && python bin/ajaxctl doctor auth"
        if kind == "cli":
            return f"reautentica CLI de {provider} y reintenta: python bin/ajaxctl doctor auth"
    if reason_code == "auth_invalid":
        return f"renueva credenciales de {provider} y reintenta: python bin/ajaxctl doctor auth"
    if reason_code == "auth_expired":
        return f"renueva sesión/token de {provider} y reintenta: python bin/ajaxctl doctor auth"
    if reason_code == "cli_not_installed":
        return f"instala CLI de {provider} o ajusta PATH; luego python bin/ajaxctl doctor auth"
    if reason_code == "provider_timeout":
        return f"verifica latencia/bridge de {provider} y timeout policy; luego python bin/ajaxctl doctor auth"
    if reason_code == "quota_exhausted":
        return f"espera cooldown o usa fallback; verifica con python bin/ajaxctl doctor auth"
    if reason_code == "provider_down":
        return f"levanta/repara {provider} y verifica con python bin/ajaxctl doctor auth"
    if reason_code == "config_missing":
        return "revisa config/model_providers.yaml y config/provider_policy.yaml"
    return "python bin/ajaxctl doctor council"


def _wsl_binary_available(binary: str) -> bool:
    if os.name != "nt" or shutil.which("wsl.exe") is None:
        return False
    try:
        proc = subprocess.run(
            ["wsl.exe", "--", "bash", "-lc", f"command -v {binary} >/dev/null 2>&1"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _cli_binary_available(binary: str) -> bool:
    if not binary:
        return False
    if shutil.which(binary):
        return True
    return _wsl_binary_available(binary)


def _resolve_probe_command(cfg: Dict[str, Any]) -> List[str]:
    cmd = cfg.get("probe_command")
    if not isinstance(cmd, list) or not cmd:
        cmd = cfg.get("infer_command") or cfg.get("command")
    if not isinstance(cmd, list) or not cmd:
        return []
    model = str(cfg.get("default_model") or cfg.get("model") or "auto")
    out: List[str] = []
    for token in cmd:
        token_s = str(token)
        if token_s == "{prompt}":
            out.append("healthcheck")
            continue
        if token_s == "{model}":
            out.append(model)
            continue
        out.append(token_s)
    return out


def _probe_cli_provider(provider: str, cfg: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    cmd = _resolve_probe_command(cfg)
    if not cmd:
        return {
            "reason_code": "config_missing",
            "reachability_state": "unknown",
            "quota_state": "unknown",
            "probe": {"provider": provider, "error": "missing_command"},
        }
    binary = str(cmd[0]).strip()
    if not _cli_binary_available(binary):
        return {
            "reason_code": "cli_not_installed",
            "reachability_state": "down",
            "quota_state": "unknown",
            "probe": {"provider": provider, "command": " ".join(cmd), "error": "cli_missing"},
        }

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "reason_code": "provider_timeout",
            "reachability_state": "timeout",
            "quota_state": "ok",
            "probe": {"provider": provider, "command": " ".join(cmd), "error": "timeout"},
        }
    except Exception as exc:
        return {
            "reason_code": "provider_down",
            "reachability_state": "down",
            "quota_state": "unknown",
            "probe": {"provider": provider, "command": " ".join(cmd), "error": _sanitize_text(str(exc))},
        }

    elapsed_ms = int((time.time() - start) * 1000)
    stdout_tail = _sanitize_text(str(proc.stdout or "")[-200:])
    stderr_tail = _sanitize_text(str(proc.stderr or "")[-200:])
    merged = f"{stderr_tail} {stdout_tail}".strip().lower()
    if proc.returncode == 0:
        return {
            "reason_code": "ok",
            "reachability_state": "ok",
            "quota_state": "ok",
            "probe": {
                "provider": provider,
                "command": " ".join(cmd),
                "latency_ms": elapsed_ms,
                "exit_code": proc.returncode,
            },
        }
    reason_code = _classify_reason_code(merged)
    if reason_code == "ok":
        reason_code = "provider_down"
    return {
        "reason_code": reason_code,
        "reachability_state": _reason_to_reachability(reason_code),
        "quota_state": _reason_to_quota(reason_code),
        "probe": {
            "provider": provider,
            "command": " ".join(cmd),
            "latency_ms": elapsed_ms,
            "exit_code": proc.returncode,
            "stderr_tail": stderr_tail,
            "stdout_tail": stdout_tail,
        },
    }


def _probe_http_provider(provider: str, cfg: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    if requests is None:
        return {
            "reason_code": "unknown_provider_failure",
            "reachability_state": "unknown",
            "quota_state": "unknown",
            "probe": {"provider": provider, "error": "requests_unavailable"},
        }
    base_url = str(cfg.get("base_url") or "").strip()
    if not base_url:
        return {
            "reason_code": "config_missing",
            "reachability_state": "unknown",
            "quota_state": "unknown",
            "probe": {"provider": provider, "error": "base_url_missing"},
        }
    endpoint = base_url.rstrip("/") + "/models"
    headers: Dict[str, str] = {}
    api_key_env = str(cfg.get("api_key_env") or "").strip()
    if api_key_env and os.getenv(api_key_env):
        headers["Authorization"] = "Bearer <redacted>"
    start = time.time()
    try:
        real_headers = {"Content-Type": "application/json"}
        if api_key_env and os.getenv(api_key_env):
            real_headers["Authorization"] = f"Bearer {os.getenv(api_key_env)}"
        resp = requests.get(endpoint, headers=real_headers, timeout=max(0.5, float(timeout_s)))
    except Exception as exc:
        err = _sanitize_text(str(exc))
        reason_code = _classify_reason_code(err)
        if reason_code == "ok":
            reason_code = "provider_down"
        return {
            "reason_code": reason_code,
            "reachability_state": _reason_to_reachability(reason_code),
            "quota_state": _reason_to_quota(reason_code),
            "probe": {"provider": provider, "endpoint": endpoint, "error": err, "headers": headers},
        }
    latency_ms = int((time.time() - start) * 1000)
    status = int(getattr(resp, "status_code", 0) or 0)
    if status == 200:
        return {
            "reason_code": "ok",
            "reachability_state": "ok",
            "quota_state": "ok",
            "probe": {"provider": provider, "endpoint": endpoint, "http_status": status, "latency_ms": latency_ms, "headers": headers},
        }
    if status == 401:
        reason_code = "auth_missing" if api_key_env and not os.getenv(api_key_env) else "auth_invalid"
    elif status == 403:
        reason_code = "auth_invalid"
    elif status == 404:
        reason_code = "provider_down"
    elif status == 429:
        reason_code = "quota_exhausted"
    elif status >= 500:
        reason_code = "provider_down"
    else:
        reason_code = "unknown_provider_failure"
    return {
        "reason_code": reason_code,
        "reachability_state": _reason_to_reachability(reason_code),
        "quota_state": _reason_to_quota(reason_code),
        "probe": {"provider": provider, "endpoint": endpoint, "http_status": status, "latency_ms": latency_ms, "headers": headers},
    }


def _probe_provider(provider: str, cfg: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    kind = str(cfg.get("kind") or "").strip().lower()
    if kind in {"cli", "codex_cli_jsonl"}:
        return _probe_cli_provider(provider, cfg, timeout_s)
    if kind == "http_openai":
        return _probe_http_provider(provider, cfg, timeout_s)
    return {
        "reason_code": "config_missing",
        "reachability_state": "unknown",
        "quota_state": "unknown",
        "probe": {"provider": provider, "error": f"unsupported_kind:{kind or 'unknown'}"},
    }


def _snapshot_for_provider(provider: str, providers_status: Dict[str, Any]) -> Dict[str, Any]:
    entry = providers_status.get(provider) if isinstance(providers_status, dict) else None
    if not isinstance(entry, dict):
        return {
            "reason_code": "unknown_provider_failure",
            "reachability_state": "unknown",
            "quota_state": "unknown",
            "snapshot_reason": "provider_not_observed",
        }
    transport = entry.get("transport") if isinstance(entry.get("transport"), dict) else {}
    breathing = entry.get("breathing") if isinstance(entry.get("breathing"), dict) else {}
    raw_status = str(transport.get("status") or breathing.get("status") or "").strip().upper()
    raw_reason = str(transport.get("reason") or breathing.get("reason") or transport.get("error") or "").strip()
    reason_code = _classify_reason_code(raw_reason)
    if raw_status == "UP" and reason_code in {"ok", "unknown_provider_failure"}:
        reason_code = "ok"
    reachability_state = "ok" if raw_status == "UP" and reason_code == "ok" else _reason_to_reachability(reason_code, raw_status)
    return {
        "reason_code": reason_code,
        "reachability_state": reachability_state,
        "quota_state": _reason_to_quota(reason_code),
        "snapshot_reason": _sanitize_text(raw_reason),
    }


def collect_provider_auth_diagnostics(
    root_dir: Path,
    *,
    include_probes: bool = True,
    write_artifact: bool = False,
    probe_timeout_s: float = 2.0,
    probe_fn: Optional[Callable[[str, Dict[str, Any], float], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    root = Path(root_dir).resolve()
    providers_cfg = _load_model_providers(root)
    providers_status = _load_providers_status(root)
    auth_mgr = AuthManager(root_dir=root)
    probe_runner = probe_fn or _probe_provider

    providers_rows: List[Dict[str, Any]] = []
    for provider in sorted(providers_cfg.keys()):
        cfg = providers_cfg.get(provider) or {}
        if bool(cfg.get("disabled")):
            continue
        configured = bool(cfg)
        auth = auth_mgr.auth_state(provider, cfg)
        auth_state = _auth_state_public(auth.state)

        snapshot = _snapshot_for_provider(provider, providers_status)
        probe_payload: Dict[str, Any] = {}
        state = dict(snapshot)
        if include_probes:
            try:
                probe_payload = probe_runner(provider, cfg, probe_timeout_s) or {}
            except Exception as exc:
                probe_payload = {
                    "reason_code": "unknown_provider_failure",
                    "reachability_state": "unknown",
                    "quota_state": "unknown",
                    "probe": {"provider": provider, "error": _sanitize_text(str(exc))},
                }
            probe_reason = str(probe_payload.get("reason_code") or "ok")
            if probe_reason == "ok":
                state.update(
                    {
                        "reason_code": "ok",
                        "reachability_state": probe_payload.get("reachability_state") or "ok",
                        "quota_state": probe_payload.get("quota_state") or "ok",
                    }
                )
            elif probe_reason != "ok":
                state.update(
                    {
                        "reason_code": probe_reason,
                        "reachability_state": probe_payload.get("reachability_state") or state.get("reachability_state"),
                        "quota_state": probe_payload.get("quota_state") or state.get("quota_state"),
                    }
                )

        reason_code = str(state.get("reason_code") or "unknown_provider_failure")
        if not configured:
            reason_code = "config_missing"
        if auth_state == "missing":
            reason_code = "auth_missing"
        elif auth_state == "expired":
            reason_code = "auth_expired"
        elif reason_code == "auth_missing" and auth_state == "present":
            reason_code = "auth_invalid" if include_probes else "unknown_provider_failure"

        reachability_state = str(state.get("reachability_state") or _reason_to_reachability(reason_code))
        quota_state = str(state.get("quota_state") or _reason_to_quota(reason_code))
        result = _effective_result(
            auth_state=auth_state,
            reachability_state=reachability_state,
            reason_code=reason_code,
        )
        next_hint = _next_hint(provider, cfg, reason_code)

        row: Dict[str, Any] = {
            "provider_name": provider,
            "configured": configured,
            "kind": str(cfg.get("kind") or "unknown"),
            "auth_state": auth_state,
            "reachability_state": reachability_state,
            "quota_state": quota_state,
            "effective_result": result,
            "reason_code": reason_code,
            "next_hint": next_hint,
            "auth_source": auth_mgr.auth_source(provider, cfg),
            "snapshot_reason": state.get("snapshot_reason"),
        }
        if include_probes and isinstance(probe_payload.get("probe"), dict):
            row["probe"] = probe_payload.get("probe")
        providers_rows.append(row)

    counts = {"usable": 0, "degraded": 0, "blocked": 0}
    for row in providers_rows:
        state = str(row.get("effective_result") or "degraded")
        if state not in counts:
            state = "degraded"
        counts[state] += 1

    blocked = [row for row in providers_rows if row.get("effective_result") == "blocked"]
    degraded = [row for row in providers_rows if row.get("effective_result") == "degraded"]
    next_hint: List[str] = []
    if blocked:
        next_hint.append("python bin/ajaxctl doctor auth")
        next_hint.append("python bin/ajaxctl doctor council")
    if degraded:
        next_hint.append("revisa timeout/quota de providers degradados y reintenta subcall")
    if not providers_rows:
        next_hint.append("revisa config/model_providers.yaml")

    dedup_hints: List[str] = []
    for hint in next_hint:
        if hint not in dedup_hints:
            dedup_hints.append(hint)

    payload: Dict[str, Any] = {
        "schema": "ajax.doctor.auth.v1",
        "ts_utc": _utc_now(),
        "providers": providers_rows,
        "summary_counts": counts,
        "next_hint": dedup_hints,
        "runtime_identity": _runtime_identity(root),
    }
    payload["summary"] = format_doctor_auth_summary(payload)

    if write_artifact:
        out_dir = root / "artifacts" / "audits"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"auth_provider_{_ts_label()}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload["artifact_path"] = str(out_path)
    return payload


def format_doctor_auth_summary(payload: Dict[str, Any]) -> str:
    rows = payload.get("providers") if isinstance(payload.get("providers"), list) else []
    counts = payload.get("summary_counts") if isinstance(payload.get("summary_counts"), dict) else {}
    lines = ["AJAX Doctor auth"]
    lines.append(f"providers: {len(rows)}")
    lines.append(f"usable: {int(counts.get('usable') or 0)}")
    lines.append(f"degraded: {int(counts.get('degraded') or 0)}")
    lines.append(f"blocked: {int(counts.get('blocked') or 0)}")
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"{row.get('provider_name')}: auth={row.get('auth_state')} reachability={row.get('reachability_state')} quota={row.get('quota_state')} result={row.get('effective_result')} reason={row.get('reason_code')}"
        )
    hints = payload.get("next_hint") if isinstance(payload.get("next_hint"), list) else []
    if hints:
        lines.append("next_hint:")
        for hint in hints:
            lines.append(f"- {hint}")
    return "\n".join(lines)


def run_doctor_auth(root_dir: Path, *, probe_timeout_s: float = 2.0) -> Dict[str, Any]:
    return collect_provider_auth_diagnostics(
        root_dir=Path(root_dir).resolve(),
        include_probes=True,
        write_artifact=True,
        probe_timeout_s=probe_timeout_s,
    )
