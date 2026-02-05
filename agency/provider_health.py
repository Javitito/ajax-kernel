from __future__ import annotations

import json
import time
import os
import platform
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agency.provider_breathing import ProviderBreathingLoop, _load_provider_configs  # type: ignore


def _iso_utc(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _detect_public_ip(timeout: float = 3.0) -> Optional[str]:
    try:
        with urllib.request.urlopen("https://ifconfig.me/ip", timeout=timeout) as resp:  # type: ignore[arg-type]
            return resp.read().decode("utf-8").strip()
    except Exception:
        return None


def _collect_route_hint() -> Optional[str]:
    cmds = [
        ["ip", "route"],
        ["netstat", "-nr"],
    ]
    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4, check=False)
        except Exception:
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            text = proc.stdout.strip()
            return text[:800]
    return None


def _gather_doctor_env(root_dir: Path, ts: float) -> Tuple[Dict[str, Any], Optional[str]]:
    proxies = {key.lower(): os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")}
    proxy_env = {k: v for k, v in proxies.items() if v}
    sandbox = bool(
        os.environ.get("WSL_DISTRO_NAME")
        or "microsoft" in platform.release().lower()
        or os.environ.get("CONTAINER") == "1"
    )
    route_hint = _collect_route_hint()
    public_ip = _detect_public_ip()
    env_dir = root_dir / "artifacts" / "health" / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = env_dir / f"doctor_env_{int(ts)}.txt"
    lines = [
        f"ts={_iso_utc(ts)}",
        f"sandbox={sandbox}",
        f"proxy_env={proxy_env}",
        f"public_ip={public_ip}",
        "route_hint:",
        route_hint or "<none>",
    ]
    try:
        snapshot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        snapshot_path = None
    doctor_env = {
        "vpn_detected": None,
        "proxy_env": proxy_env,
        "sandbox": sandbox,
        "public_ip": public_ip,
        "route_hint": (route_hint[:200] if route_hint else None),
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
    }
    return doctor_env, str(snapshot_path) if snapshot_path else None


class ProviderFailureCode:
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    CLIENT_TIMEOUT = "CLIENT_TIMEOUT"
    INFRA_BRIDGE = "INFRA_BRIDGE"
    SANDBOX_LISTEN_EPERM = "SANDBOX_LISTEN_EPERM"
    BINARY_MISSING = "BINARY_MISSING"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    HTTP_403 = "HTTP_403"
    ENV_BLOCKED = "ENV_BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class ProviderFailure:
    provider: str
    code: str
    message: str
    stage: str
    context: Dict[str, Any] = field(default_factory=dict)
    remediation: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "context": self.context,
            "remediation": self.remediation,
            "ts": self.ts,
            "ts_utc": _iso_utc(self.ts),
        }


def _remediation_for_code(code: str, provider: str) -> List[str]:
    if code == ProviderFailureCode.QUOTA_EXHAUSTED:
        return [
            f"Se detectó cuota/capacidad agotada para {provider}. Espera al cooldown o aumenta presupuesto/cuota.",
            "Escala a un provider alternativo por ladder/policy para mantener continuidad.",
            "Opcional: ejecuta `bin/ajaxctl doctor providers` para verificar si el estado cambió.",
        ]
    if code == ProviderFailureCode.TIMEOUT:
        return [
            f"Ejecuta `bin/ajaxctl doctor providers` para validar la salud de {provider}.",
            "Escala a un provider alternativo o aumenta el presupuesto/tier temporalmente.",
        ]
    if code == ProviderFailureCode.CLIENT_TIMEOUT:
        return [
            f"Se agotó el timeout del cliente al invocar {provider}.",
            "Reintenta con un timeout mayor o revisa la latencia local (WSL/bridge).",
            "Ejecuta `bin/ajaxctl doctor providers` si sospechas degradación real.",
        ]
    if code == ProviderFailureCode.INFRA_BRIDGE:
        return [
            "El bridge/CLI falló antes de contactar el modelo.",
            "Ejecuta `bin/ajaxctl doctor bridge` para diagnosticar binarios, env vars y ping.",
            "Repara el bridge o reautoriza credenciales antes de penalizar al provider.",
        ]
    if code == ProviderFailureCode.SANDBOX_LISTEN_EPERM:
        return [
            f"Ejecuta el CLI de {provider} fuera del sandbox o habilita el permiso para abrir sockets en el host.",
            "Configura el CLI para que no abra listeners en 0.0.0.0 durante el arranque (modo headless).",
            "Sugerido: abrir INCIDENT de salud de providers para derivar la remediación.",
        ]
    if code == ProviderFailureCode.BINARY_MISSING:
        return [
            f"Instala el binario de {provider} y agrégalo al PATH.",
            "Verifica variables tipo QWEN_RUN/GEMINI_MODEL para apuntar a rutas válidas.",
        ]
    if code == ProviderFailureCode.AUTH_REQUIRED:
        return [
            f"Renueva las credenciales de {provider} (bin/ajaxctl permit ... o flujo web).",
            "Vuelve a ejecutar `doctor providers` para confirmar el estado.",
        ]
    if code == ProviderFailureCode.HTTP_403:
        return [
            f"El endpoint de {provider} devolvió HTTP 403. Revisa permisos de API key y whitelists de IP.",
            "Si el acceso depende de VPN o allowlists, confirma que la IP actual esté autorizada.",
        ]
    if code == ProviderFailureCode.ENV_BLOCKED:
        return [
            "Se detectó un posible bloqueo ambiental (VPN/proxy). Deshabilita proxies/VPN o usa una red permitida.",
            "Valida la conectividad directa (`curl <endpoint>`) fuera del sandbox.",
            "Sugerido: abrir INCIDENT de salud de providers para seguimiento automático.",
        ]
    return [
        f"Revisa artifacts/health/providers_status.json para más contexto de {provider}.",
        "Consulta los logs del provider y considera abrir un incidente LAB.",
    ]


def classify_provider_failure(
    provider: str,
    message: str,
    *,
    stage: str,
    context: Optional[Dict[str, Any]] = None,
    root_dir: Optional[Path] = None,
) -> ProviderFailure:
    msg = (message or "").strip()
    msg_lower = msg.lower()
    code = ProviderFailureCode.UNKNOWN
    ctx = dict(context or {})
    http_status = ctx.get("http_status")
    stderr = str(ctx.get("stderr") or "").lower()
    proxy_env = ctx.get("proxy_env") or {}
    explicit_code = str(ctx.get("error_code") or "").strip().lower()
    quota_tokens: List[str] = []
    if root_dir is not None:
        try:
            from agency.provider_failure_policy import load_provider_failure_policy, quota_exhausted_tokens

            policy = load_provider_failure_policy(root_dir)
            quota_tokens = quota_exhausted_tokens(policy)
        except Exception:
            quota_tokens = []
    if not quota_tokens:
        quota_tokens = ["429", "rate limit", "rate_limit", "quota", "no capacity", "capacity", "quota exhausted"]

    if explicit_code == "client_timeout":
        code = ProviderFailureCode.CLIENT_TIMEOUT
    elif explicit_code == "infra_bridge_error":
        code = ProviderFailureCode.INFRA_BRIDGE
    elif "listen" in msg_lower and "eperm" in msg_lower:
        code = ProviderFailureCode.SANDBOX_LISTEN_EPERM
    elif "listen" in stderr and "eperm" in stderr:
        code = ProviderFailureCode.SANDBOX_LISTEN_EPERM
    elif "sandbox_listen_eperm" in msg_lower:
        code = ProviderFailureCode.SANDBOX_LISTEN_EPERM
    elif "command not found" in msg_lower or "no such file" in msg_lower or "not recognized" in msg_lower:
        code = ProviderFailureCode.BINARY_MISSING
    elif http_status == 429:
        code = ProviderFailureCode.QUOTA_EXHAUSTED
    elif quota_tokens and any(tok in msg_lower for tok in quota_tokens):
        code = ProviderFailureCode.QUOTA_EXHAUSTED
    elif "timeout" in msg_lower or "timed out" in msg_lower or "deadline" in msg_lower:
        code = ProviderFailureCode.TIMEOUT
    elif "auth_missing" in msg_lower or "auth_expired" in msg_lower or "api key" in msg_lower or "unauthorized" in msg_lower:
        code = ProviderFailureCode.AUTH_REQUIRED
    elif http_status == 403 or "http_403" in msg_lower:
        code = ProviderFailureCode.HTTP_403
    elif http_status == 401:
        code = ProviderFailureCode.AUTH_REQUIRED
    elif http_status == 407 or (proxy_env and any(proxy_env.values())):
        code = ProviderFailureCode.ENV_BLOCKED
    if code == ProviderFailureCode.UNKNOWN and msg:
        code = ProviderFailureCode.UNAVAILABLE
    remediation = _remediation_for_code(code, provider)
    ctx = dict(context or {})
    if msg:
        ctx.setdefault("detail", msg)
    evidence = ctx.get("evidence")
    if evidence:
        ctx.setdefault("evidence", evidence)
    return ProviderFailure(provider=provider, code=code, message=msg, stage=stage, context=ctx, remediation=remediation)


def run_provider_doctor(root_dir: Path, *, roles: Optional[List[str]] = None) -> Tuple[Path, Dict[str, Any]]:
    """
    Ejecuta probes ligeros (ProviderBreathingLoop) y persiste un snapshot con clasificación estructurada.
    """
    provider_configs = _load_provider_configs(root_dir)
    loop = ProviderBreathingLoop(root_dir=root_dir, provider_configs=provider_configs)
    status_doc = loop.run_once(roles=roles or ["brain"])
    ts = time.time()
    doctor_env, env_path = _gather_doctor_env(root_dir, ts)
    payload: Dict[str, Any] = {
        "schema": "ajax.provider_doctor.v2",
        "ran_ts": ts,
        "ran_utc": _iso_utc(ts),
        "providers": {},
        "doctor_env": doctor_env,
    }
    providers = status_doc.get("providers") if isinstance(status_doc, dict) else {}
    if not isinstance(providers, dict):
        providers = {}
    preferred_roles = ("brain", "council", "scout", "vision")
    for name, entry in providers.items():
        if not isinstance(entry, dict):
            continue
        breathing = entry.get("breathing") if isinstance(entry.get("breathing"), dict) else {}
        status = breathing.get("status") or "UNKNOWN"
        reason = breathing.get("reason") or ""
        roles_map = breathing.get("roles") if isinstance(breathing.get("roles"), dict) else {}
        role_entry = None
        if isinstance(roles_map, dict):
            for role in preferred_roles:
                maybe = roles_map.get(role)
                if isinstance(maybe, dict):
                    role_entry = maybe
                    break
            if role_entry is None and roles_map:
                role_entry = next(iter(roles_map.values()))
        evidence = {}
        signals = {}
        probe_kind = "SNAPSHOT"
        if isinstance(role_entry, dict):
            evidence = role_entry.get("evidence") or {}
            if not evidence:
                if role_entry.get("latency_ms") is not None:
                    evidence["latency_ms"] = role_entry.get("latency_ms")
            for key in ("http_status", "endpoint", "command", "stderr_excerpt"):
                if role_entry.get(key) and key not in evidence:
                    evidence[key] = role_entry.get(key)
            probe_kind = role_entry.get("probe_kind") or "ACTIVE"
            if role_entry.get("error") and not reason:
                reason = role_entry.get("reason") or ""
            signals = role_entry.get("signals") if isinstance(role_entry.get("signals"), dict) else {}
            if signals and "latency_ms" in signals and "latency_ms" not in evidence:
                evidence["latency_ms"] = signals.get("latency_ms")
        failure_list: List[Dict[str, Any]] = []
        incident_suggested = False
        if status != "UP":
            error_text = ""
            if role_entry and role_entry.get("error"):
                error_text = str(role_entry.get("error"))
            elif reason:
                error_text = str(reason)
            failure = classify_provider_failure(
                name,
                error_text,
                stage="doctor",
                context={
                    "reason": reason,
                    "roles": sorted(list(roles_map.keys())) if isinstance(roles_map, dict) else [],
                    "http_status": role_entry.get("http_status") if role_entry else None,
                    "stderr": role_entry.get("stderr_excerpt") if role_entry else None,
                    "proxy_env": doctor_env.get("proxy_env"),
                    "evidence": evidence,
                },
                root_dir=root_dir,
            )
            failure_list.append(failure.to_dict())
            if failure.code in {ProviderFailureCode.SANDBOX_LISTEN_EPERM, ProviderFailureCode.HTTP_403, ProviderFailureCode.ENV_BLOCKED}:
                incident_suggested = True
        payload["providers"][name] = {
            "status": status,
            "reason": reason,
            "last_probe_utc": breathing.get("last_probe_utc"),
            "probe_kind": probe_kind,
            "evidence": evidence,
            "failures": failure_list,
            "incident_suggestion": incident_suggested,
            "signals": signals or {},
        }
    payload["incident_suggestion"] = any(
        data.get("incident_suggestion")
        for data in payload["providers"].values()
        if isinstance(data, dict)
    )
    artifacts_dir = root_dir / "artifacts" / "health" / "providers"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / f"doctor_{int(ts)}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path, payload


def _resolve_ping_timeouts(
    root_dir: Path,
    *,
    provider_name: str,
    provider_cfg: Dict[str, Any],
    planning: bool,
    timeout_ttft_ms: Optional[int],
    timeout_total_ms: Optional[int],
    timeout_stall_ms: Optional[int],
) -> Dict[str, Optional[int]]:
    policy, _ = ({}, None)
    try:
        from agency.provider_timeouts_policy import load_provider_timeouts_policy  # type: ignore

        policy, _ = load_provider_timeouts_policy(root_dir)
    except Exception:
        policy = {}
    defaults = (policy.get("defaults") if isinstance(policy, dict) else {}) or {}
    rail_cfg = {}
    rails = policy.get("rails") if isinstance(policy, dict) else None
    if isinstance(rails, dict):
        rail_cfg = rails.get("lab", {}) if isinstance(rails.get("lab"), dict) else {}
    rail_defaults = rail_cfg.get("defaults") if isinstance(rail_cfg, dict) else {}
    providers = policy.get("providers") if isinstance(policy, dict) else None
    provider_overrides = (
        providers.get(provider_name, {}) if isinstance(providers, dict) else {}
    )
    stage_key = "planning" if planning else "default"
    stage_overrides = (
        provider_overrides.get(stage_key) if isinstance(provider_overrides, dict) else None
    )

    def _pick(key: str) -> Optional[int]:
        for src in (stage_overrides, rail_defaults, defaults):
            if isinstance(src, dict) and key in src:
                raw = src.get(key)
                if raw is None:
                    return None
                try:
                    val = int(raw)
                    if val > 0:
                        return val
                except Exception:
                    return None
        return None

    total_ms = timeout_total_ms or _pick("total_timeout_ms")
    ttft_ms = timeout_ttft_ms or _pick("first_output_timeout_ms")
    stall_ms = timeout_stall_ms or _pick("stall_timeout_ms")
    connect_ms = _pick("connect_timeout_ms")
    if total_ms is None:
        base = (
            provider_cfg.get("planning_timeout_seconds")
            if planning
            else provider_cfg.get("timeout_seconds")
        )
        try:
            total_ms = int(float(base) * 1000) if base is not None else None
        except Exception:
            total_ms = None
    return {
        "connect_timeout_ms": connect_ms,
        "first_output_timeout_ms": ttft_ms,
        "stall_timeout_ms": stall_ms,
        "total_timeout_ms": total_ms,
    }


def _run_cli_with_timeouts(
    cmd: List[str],
    *,
    input_text: Optional[str],
    timeout_cfg: Dict[str, Any],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    t_start = time.monotonic()
    meta = {
        "t_start": t_start,
        "t_connect_ok": t_start,
        "t_first_output": None,
        "t_last_output": None,
        "t_end": None,
    }
    bytes_stdout = 0
    bytes_stderr = 0
    stdout_chunks: List[bytes] = []
    stderr_chunks: List[bytes] = []

    timeout_kind = "NONE"
    client_abort = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=env,
            cwd=cwd,
        )
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timings": {**meta, "t_end": time.monotonic()},
            "timeout_kind": "NONE",
            "client_abort": False,
            "bytes_stdout": 0,
            "bytes_stderr": 0,
        }

    if proc.stdin and input_text is not None:
        try:
            proc.stdin.write(input_text.encode("utf-8", errors="ignore"))
        except Exception:
            pass
        try:
            proc.stdin.close()
        except Exception:
            pass

    import threading
    import queue

    def reader(stream, q, name):
        try:
            fd = stream.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                q.put((name, chunk))
        except Exception:
            pass
        q.put((name, b""))

    out_q = queue.Queue()
    if proc.stdout:
        threading.Thread(
            target=reader, args=(proc.stdout, out_q, "stdout"), daemon=True
        ).start()
    if proc.stderr:
        threading.Thread(
            target=reader, args=(proc.stderr, out_q, "stderr"), daemon=True
        ).start()

    total_ms = timeout_cfg.get("total_timeout_ms")
    first_output_ms = timeout_cfg.get("first_output_timeout_ms")
    stall_ms = timeout_cfg.get("stall_timeout_ms")
    total_deadline = (
        t_start + (float(total_ms) / 1000.0) if isinstance(total_ms, (int, float)) else None
    )
    first_output_deadline = (
        t_start + (float(first_output_ms) / 1000.0)
        if isinstance(first_output_ms, (int, float))
        else None
    )
    stall_timeout_s = (float(stall_ms) / 1000.0) if isinstance(stall_ms, (int, float)) else None

    while True:
        rc = proc.poll()
        if rc is not None and out_q.empty():
            break

        try:
            name, chunk = out_q.get(timeout=0.05)
            if not chunk:
                continue
            now = time.monotonic()
            if name == "stdout":
                if meta["t_first_output"] is None:
                    meta["t_first_output"] = now
                bytes_stdout += len(chunk)
                stdout_chunks.append(chunk)
            else:
                bytes_stderr += len(chunk)
                stderr_chunks.append(chunk)
            meta["t_last_output"] = now
        except queue.Empty:
            pass

        now = time.monotonic()
        if (
            first_output_deadline
            and meta["t_first_output"] is None
            and now > first_output_deadline
        ):
            timeout_kind = "TTFT"
            client_abort = True
            break
        if (
            stall_timeout_s
            and meta["t_last_output"] is not None
            and (now - float(meta["t_last_output"])) > stall_timeout_s
        ):
            timeout_kind = "STALL"
            client_abort = True
            break
        if total_deadline and now > total_deadline:
            timeout_kind = "TOTAL"
            client_abort = True
            break

    if client_abort:
        try:
            proc.terminate()
        except Exception:
            pass
        if timeout_kind in {"TTFT", "STALL"}:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    try:
        proc.wait(timeout=2)
    except Exception:
        pass

    meta["t_end"] = time.monotonic()
    stdout_bytes = b"".join(stdout_chunks)
    stderr_bytes = b"".join(stderr_chunks)
    out_text = stdout_bytes.decode("utf-8", errors="ignore")
    err_text = stderr_bytes.decode("utf-8", errors="ignore")

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": out_text,
        "stderr": err_text,
        "timings": meta,
        "timeout_kind": timeout_kind,
        "client_abort": client_abort,
        "bytes_stdout": bytes_stdout,
        "bytes_stderr": bytes_stderr,
    }


def run_provider_ping(
    root_dir: Path,
    *,
    provider: str,
    model: Optional[str] = None,
    timeout_ttft_ms: Optional[int] = None,
    timeout_total_ms: Optional[int] = None,
    timeout_stall_ms: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    cfg_doc = _load_provider_configs(root_dir)
    providers_cfg = cfg_doc.get("providers") if isinstance(cfg_doc, dict) else None
    provider_cfg = providers_cfg.get(provider) if isinstance(providers_cfg, dict) else None
    if not isinstance(provider_cfg, dict):
        raise RuntimeError(f"provider_not_found:{provider}")
    kind = str(provider_cfg.get("kind") or "").strip().lower()
    selected_model = model or provider_cfg.get("default_model") or provider_cfg.get("model")

    auth_source = None
    try:
        from agency.auth_manager import AuthManager  # type: ignore

        auth = AuthManager(root_dir=root_dir)
        auth_source = auth.auth_source(provider, provider_cfg)
    except Exception:
        auth_source = None

    timeout_cfg = _resolve_ping_timeouts(
        root_dir,
        provider_name=provider,
        provider_cfg=provider_cfg,
        planning=True,
        timeout_ttft_ms=timeout_ttft_ms,
        timeout_total_ms=timeout_total_ms,
        timeout_stall_ms=timeout_stall_ms,
    )

    prompt = "ping"
    cmd = None
    use_stdin = False
    endpoint = None
    env_override: Dict[str, str] = {}

    if kind == "cli":
        cmd_template = provider_cfg.get("command") or []
        cmd = []
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
                cmd.append(selected_model or "")
                i += 1
                continue
            cmd.append(token)
            i += 1
        endpoint = f"cli:{cmd[0]}" if cmd else "cli"
    elif kind == "codex_cli_jsonl":
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check"]
        if selected_model:
            cmd.extend(["--model", str(selected_model).strip()])
        use_stdin = True
        endpoint = "codex_cli_jsonl"
        env_override["CODEX_REASONING_EFFORT"] = os.getenv("AJAX_CODEX_PLANNING_EFFORT", "medium")
    else:
        raise RuntimeError(f"provider_ping_unsupported_kind:{kind or 'unknown'}")

    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    result = _run_cli_with_timeouts(
        cmd,
        input_text=(prompt if use_stdin else None),
        timeout_cfg=timeout_cfg,
        env=env,
    )
    timings = result.get("timings") or {}
    t_start = timings.get("t_start")
    t_first = timings.get("t_first_output")
    t_end = timings.get("t_end")
    ttft_ms = None
    total_ms = None
    if isinstance(t_start, (int, float)) and isinstance(t_first, (int, float)) and t_first >= t_start:
        ttft_ms = int((float(t_first) - float(t_start)) * 1000)
    if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)) and t_end >= t_start:
        total_ms = int((float(t_end) - float(t_start)) * 1000)
    timeout_kind = str(result.get("timeout_kind") or "NONE")
    outcome = "ok"
    if result.get("client_abort") and timeout_kind in {"TTFT", "STALL", "TOTAL"}:
        outcome = timeout_kind.lower()
    elif not result.get("ok"):
        outcome = f"rc_{result.get('returncode')}"

    payload: Dict[str, Any] = {
        "schema": "ajax.provider_ping.v1",
        "provider": provider,
        "model": selected_model,
        "kind": kind,
        "endpoint": endpoint,
        "auth_source": auth_source,
        "timeout_cfg": timeout_cfg,
        "outcome": outcome,
        "ok": outcome == "ok",
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "timeout_kind": timeout_kind,
        "stdout_tail": (result.get("stdout") or "")[-200:],
        "stderr_tail": (result.get("stderr") or "")[-200:],
        "ts": time.time(),
        "ts_utc": _iso_utc(),
    }

    artifacts_dir = root_dir / "artifacts" / "health" / "providers"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / f"ping_{provider}_{int(payload['ts'])}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path, payload
