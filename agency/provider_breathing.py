from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from agency.auth_manager import AuthManager

PIDFILE = ROOT / "artifacts" / "health" / "provider_breathing.pid"

STRICT_JSON_ONLY = (
    "Output strictly one JSON object and nothing else. "
    "No markdown, no code fences, no extra text. "
    "The output must start with '{' and end with '}'."
)

_QUOTA_TOKENS: Optional[List[str]] = None


def _quota_exhausted_tokens() -> List[str]:
    global _QUOTA_TOKENS
    if isinstance(_QUOTA_TOKENS, list) and _QUOTA_TOKENS:
        return _QUOTA_TOKENS
    try:
        from agency.provider_failure_policy import load_provider_failure_policy, quota_exhausted_tokens

        policy = load_provider_failure_policy(ROOT)
        tokens = quota_exhausted_tokens(policy)
        _QUOTA_TOKENS = tokens if tokens else None
    except Exception:
        _QUOTA_TOKENS = None
    return _QUOTA_TOKENS or ["429", "rate limit", "quota", "no capacity", "capacity", "quota exhausted"]


def _load_env_file(path: Path) -> None:
    """Carga variables de un .env sencillo (key=value) si no están ya en el entorno."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _now_ts() -> float:
    return time.time()


def _iso_now(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or _now_ts()))


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"providers": {}, "updated_at": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"providers": {}, "updated_at": None}
    except Exception:
        return {"providers": {}, "updated_at": None}


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


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


def _load_provider_configs(root_dir: Path) -> Dict[str, Any]:
    cfg_yaml_path = root_dir / "config" / "model_providers.yaml"
    cfg_json_path = root_dir / "config" / "model_providers.json"
    data: Any = None

    if cfg_yaml_path.exists() and yaml is not None:
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


@dataclass
class ProbeResult:
    status: str  # UP | DEGRADED | DOWN
    latency_ms: Optional[int] = None
    reason: str = ""
    error: str = ""
    checked_at: float = 0.0
    probe_kind: str = "ACTIVE"
    http_status: Optional[int] = None
    endpoint: Optional[str] = None
    command: Optional[str] = None
    stderr_excerpt: Optional[str] = None
    signals: Optional[Dict[str, Any]] = None

    def evidence(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.http_status is not None:
            payload["http_status"] = self.http_status
        if self.endpoint:
            payload["endpoint"] = self.endpoint
        if self.command:
            payload["command"] = self.command
        if self.stderr_excerpt:
            payload["stderr_excerpt"] = self.stderr_excerpt
        return payload

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "status": self.status,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
            "error": self.error,
            "checked_at": self.checked_at,
            "checked_utc": _iso_now(self.checked_at),
            "probe_kind": self.probe_kind,
            "evidence": self.evidence(),
        }
        if self.http_status is not None:
            out["http_status"] = self.http_status
        if self.endpoint:
            out["endpoint"] = self.endpoint
        if self.command:
            out["command"] = self.command
        if self.stderr_excerpt:
            out["stderr_excerpt"] = self.stderr_excerpt
        if self.signals:
            out["signals"] = dict(self.signals)
        return out


def _latency_threshold_ms(tier: str) -> int:
    tier_n = (tier or "balanced").lower()
    if tier_n == "cheap":
        return 6000
    if tier_n == "premium":
        return 12000
    return 8000


def _is_local_provider(cfg: Dict[str, Any]) -> bool:
    kind = str(cfg.get("kind") or "").strip().lower()
    if kind == "static":
        return True
    base_url = cfg.get("base_url")
    if isinstance(base_url, str):
        url = base_url.strip().lower()
        if "localhost" in url or "127.0.0.1" in url:
            return True
    return False


def _capabilities_for_cfg(cfg: Dict[str, Any]) -> Dict[str, bool]:
    roles = cfg.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    roles_l = {str(r).strip().lower() for r in roles if r}
    is_local = _is_local_provider(cfg)
    return {
        "text_local": bool(is_local and roles_l.intersection({"brain", "council", "scout"})),
        "vision_local": bool(is_local and "vision" in roles_l),
    }


def _load_policy() -> Dict[str, Any]:
    path_yaml = ROOT / "config" / "provider_policy.yaml"
    path_json = ROOT / "config" / "provider_policy.json"
    data: Any = None

    if yaml is not None and path_yaml.exists():
        try:
            data = yaml.safe_load(path_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            data = None

    if data is None and path_json.exists():
        try:
            data = json.loads(path_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    return data if isinstance(data, dict) else {}


def _policy_state(policy_doc: Dict[str, Any], provider: str, cap: str) -> Optional[str]:
    providers = policy_doc.get("providers") if isinstance(policy_doc, dict) else None
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


def _probe_http_transport(cfg: Dict[str, Any], timeout_sec: int) -> ProbeResult:
    if requests is None:
        return ProbeResult(status="DOWN", reason="requests_missing", checked_at=_now_ts(), probe_kind="TRANSPORT")
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        return ProbeResult(status="DOWN", reason="base_url_missing", checked_at=_now_ts(), probe_kind="TRANSPORT")
    url = f"{base_url}/models"
    api_key_env = cfg.get("api_key_env")
    headers = {}
    if api_key_env:
        api_key = os.getenv(str(api_key_env))
        if not api_key:
            return ProbeResult(status="DOWN", reason="auth_missing", checked_at=_now_ts(), probe_kind="TRANSPORT", endpoint=url)
        headers["Authorization"] = f"Bearer {api_key}"
    started = _now_ts()
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_sec)
    except Exception as exc:
        msg = str(exc)
        lowered = msg.lower()
        reason = "timeout" if "timeout" in lowered else "unavailable"
        return ProbeResult(
            status="DOWN",
            reason=reason,
            error=msg[:200],
            checked_at=_now_ts(),
            probe_kind="TRANSPORT",
            endpoint=url,
        )
    latency_ms = int((_now_ts() - started) * 1000)
    signals = {"latency_ms": latency_ms}
    if resp.status_code in {401, 403}:
        return ProbeResult(
            status="DOWN",
            reason="auth_missing",
            error=(resp.text or "")[:200],
            checked_at=_now_ts(),
            probe_kind="TRANSPORT",
            http_status=resp.status_code,
            endpoint=url,
            signals=signals,
        )
    if resp.status_code == 429:
        return ProbeResult(
            status="UP",
            reason="quota_exhausted",
            checked_at=_now_ts(),
            probe_kind="TRANSPORT",
            http_status=resp.status_code,
            endpoint=url,
            signals=signals,
        )
    if resp.status_code >= 400:
        return ProbeResult(
            status="DOWN",
            reason=f"http_{resp.status_code}",
            error=(resp.text or "")[:200],
            checked_at=_now_ts(),
            probe_kind="TRANSPORT",
            http_status=resp.status_code,
            endpoint=url,
            signals=signals,
        )
    return ProbeResult(
        status="UP",
        checked_at=_now_ts(),
        probe_kind="TRANSPORT",
        http_status=resp.status_code,
        endpoint=url,
        signals=signals,
    )


def _probe_cli_transport(cfg: Dict[str, Any]) -> ProbeResult:
    cmd_template = cfg.get("command") or []
    if not isinstance(cmd_template, list) or not cmd_template:
        return ProbeResult(status="DOWN", reason="cli_command_missing", checked_at=_now_ts(), probe_kind="TRANSPORT")
    tool = str(cmd_template[0])
    cmd_str = " ".join(str(x) for x in cmd_template)
    if tool in {"python", "python3", "py"}:
        script_path = None
        for part in cmd_template[1:]:
            if isinstance(part, str) and part.endswith(".py"):
                script_path = part
                break
        if script_path:
            path = Path(script_path)
            if not path.is_absolute():
                path = ROOT / script_path
            if not path.exists():
                return ProbeResult(
                    status="DOWN",
                    reason=f"script_missing:{script_path}",
                    checked_at=_now_ts(),
                    probe_kind="TRANSPORT",
                    command=cmd_str,
                )
        return ProbeResult(status="UP", checked_at=_now_ts(), probe_kind="TRANSPORT", command=cmd_str)
    if shutil.which(tool) is None:
        return ProbeResult(
            status="DOWN",
            reason=f"cli_missing:{tool}",
            checked_at=_now_ts(),
            probe_kind="TRANSPORT",
            command=cmd_str,
        )
    return ProbeResult(status="UP", checked_at=_now_ts(), probe_kind="TRANSPORT", command=cmd_str)


def _probe_codex_transport(cfg: Dict[str, Any], timeout_sec: int = 3) -> ProbeResult:
    now = _now_ts()
    if os.name == "nt":
        wsl = shutil.which("wsl.exe")
        if not wsl:
            return ProbeResult(status="DOWN", reason="wsl_missing", checked_at=now, probe_kind="TRANSPORT", command="wsl.exe (missing)")
        distro = (os.getenv("AJAX_WSL_DISTRO") or os.getenv("WSL_DISTRO_NAME") or "Ubuntu").strip()
        cmd = [wsl, "-d", distro, "--", "bash", "-lc", "command -v codex >/dev/null 2>&1"]
        cmd_str = " ".join(cmd)
        started = _now_ts()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
        except subprocess.TimeoutExpired:
            return ProbeResult(status="DOWN", reason="timeout", checked_at=_now_ts(), probe_kind="TRANSPORT", command=cmd_str)
        except Exception as exc:
            return ProbeResult(status="DOWN", reason="unavailable", error=str(exc)[:200], checked_at=_now_ts(), probe_kind="TRANSPORT", command=cmd_str)
        latency_ms = int((_now_ts() - started) * 1000)
        err = (proc.stderr or "").strip()
        out = (proc.stdout or "").strip()
        signals = {"latency_ms": latency_ms}
        if proc.returncode != 0:
            msg = err or out or f"rc_{proc.returncode}"
            return ProbeResult(
                status="DOWN",
                latency_ms=latency_ms,
                reason="cli_missing:codex",
                error=msg[:200],
                checked_at=_now_ts(),
                probe_kind="TRANSPORT",
                command=cmd_str,
                stderr_excerpt=msg[:200] or None,
                signals=signals,
            )
        return ProbeResult(status="UP", latency_ms=latency_ms, checked_at=_now_ts(), probe_kind="TRANSPORT", command=cmd_str, signals=signals)

    if shutil.which("codex") is None:
        return ProbeResult(status="DOWN", reason="cli_missing:codex", checked_at=now, probe_kind="TRANSPORT", command="codex")
    return ProbeResult(status="UP", checked_at=now, probe_kind="TRANSPORT", command="codex")


def _probe_http_openai(provider: str, cfg: Dict[str, Any], role: str, timeout_sec: int) -> ProbeResult:
    if requests is None:
        return ProbeResult(status="DOWN", reason="requests_missing", checked_at=_now_ts(), probe_kind="ACTIVE")
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        return ProbeResult(status="DOWN", reason="base_url_missing", checked_at=_now_ts(), probe_kind="ACTIVE")
    model = cfg.get("default_model") or cfg.get("model") or "llama-3.1-8b-instant"
    api_key_env = cfg.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    system = STRICT_JSON_ONLY
    if role == "brain":
        want = {
            "plan_id": "probe",
            "steps": [{"id": "step-1", "action": "window.focus", "args": {"title_contains": "Brave"}}],
            "success_spec": {"type": "window_title_contains", "text": "Brave"},
        }
        user = f"Return exactly this JSON: {json.dumps(want, ensure_ascii=False)}"
    elif role == "council":
        want = {
            "verdict": "approve",
            "reason": "probe_ok",
            "suggested_fix": "",
            "escalation_hint": "",
            "debug_notes": "",
        }
        user = f"Return exactly this JSON: {json.dumps(want, ensure_ascii=False)}"
    else:
        user = "Return exactly: OK"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    started = _now_ts()
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout_sec)
    except Exception as exc:
        msg = str(exc)
        lowered = msg.lower()
        reason = "timeout" if "timeout" in lowered else "unavailable"
        return ProbeResult(
            status="DEGRADED",
            reason=reason,
            error=msg[:200],
            checked_at=_now_ts(),
            endpoint=base_url,
        )
    latency_ms = int((_now_ts() - started) * 1000)
    signals = {"latency_ms": latency_ms}
    if resp.status_code >= 400:
        return ProbeResult(
            status="DEGRADED",
            reason=f"http_{resp.status_code}",
            error=(resp.text or "")[:200],
            checked_at=_now_ts(),
            latency_ms=latency_ms,
            http_status=resp.status_code,
            endpoint=base_url,
            signals=signals,
        )
    try:
        data = resp.json()
        content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    except Exception as exc:
        return ProbeResult(
            status="DEGRADED",
            reason="bad_json",
            error=str(exc)[:200],
            checked_at=_now_ts(),
            latency_ms=latency_ms,
            endpoint=base_url,
            signals=signals,
        )
    if role in {"brain", "council"}:
        parsed = _parse_json_object(content)
        if not isinstance(parsed, dict):
            return ProbeResult(
                status="DEGRADED",
                reason="invalid_json",
                error=content[:200],
                checked_at=_now_ts(),
                latency_ms=latency_ms,
                endpoint=base_url,
                signals=signals,
            )
        signals["valid_json"] = True
    tier = str(cfg.get("tier") or "balanced")
    if latency_ms > _latency_threshold_ms(tier):
        signals["latency_note"] = "slow"
    return ProbeResult(status="UP", latency_ms=latency_ms, checked_at=_now_ts(), endpoint=base_url, signals=signals)


def _probe_cli(provider: str, cfg: Dict[str, Any], role: str, timeout_sec: int) -> ProbeResult:
    cmd_template = cfg.get("command") or []
    if not isinstance(cmd_template, list) or not cmd_template:
        return ProbeResult(status="DEGRADED", reason="cli_command_missing", checked_at=_now_ts())
    model = cfg.get("default_model") or cfg.get("model") or ""
    system = STRICT_JSON_ONLY
    if role == "brain":
        want = {
            "plan_id": "probe",
            "steps": [{"id": "step-1", "action": "window.focus", "args": {"title_contains": "Brave"}}],
            "success_spec": {"type": "window_title_contains", "text": "Brave"},
        }
        prompt = system + "\n" + f"Return exactly this JSON: {json.dumps(want, ensure_ascii=False)}"
    elif role == "council":
        want = {
            "verdict": "approve",
            "reason": "probe_ok",
            "suggested_fix": "",
            "escalation_hint": "",
            "debug_notes": "",
        }
        prompt = system + "\n" + f"Return exactly this JSON: {json.dumps(want, ensure_ascii=False)}"
    else:
        prompt = "Return exactly: OK"
    cmd: List[str] = []
    used_prompt = False
    for tok in cmd_template:
        if tok == "{model}":
            cmd.append(str(model))
        elif tok == "{prompt}":
            cmd.append(prompt)
            used_prompt = True
        else:
            cmd.append(str(tok))
    cmd_str = " ".join(cmd)
    started = _now_ts()
    try:
        proc = subprocess.run(
            cmd,
            input=None if used_prompt else prompt,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        latency_ms = int((_now_ts() - started) * 1000)
        stderr = exc.stderr if isinstance(getattr(exc, "stderr", None), str) else ""
        stdout = exc.stdout if isinstance(getattr(exc, "stdout", None), str) else ""
        combined = "\n".join([s for s in (stderr.strip(), stdout.strip()) if s])
        msg = (combined or str(exc)).strip()
        lowered = msg.lower()
        tokens = _quota_exhausted_tokens()
        reason = "quota_exhausted" if any(tok in lowered for tok in tokens) else "timeout"
        return ProbeResult(
            status="DEGRADED",
            latency_ms=latency_ms,
            reason=reason,
            error=msg[:200],
            checked_at=_now_ts(),
            command=cmd_str,
        )
    except Exception as exc:
        msg = str(exc)
        lowered = msg.lower()
        reason = "sandbox_listen_eperm" if ("eperm" in lowered and ("listen" in lowered or "bind" in lowered)) else "unavailable"
        return ProbeResult(status="DEGRADED", reason=reason, error=msg[:200], checked_at=_now_ts(), command=cmd_str)
    latency_ms = int((_now_ts() - started) * 1000)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    signals = {"latency_ms": latency_ms}
    if proc.returncode != 0:
        msg = err or out or f"cli_rc_{proc.returncode}"
        lowered = msg.lower()
        reason = "sandbox_listen_eperm" if ("eperm" in lowered and ("listen" in lowered or "bind" in lowered)) else "unavailable"
        return ProbeResult(
            status="DEGRADED",
            latency_ms=latency_ms,
            reason=reason,
            error=msg[:200],
            checked_at=_now_ts(),
            command=cmd_str,
            stderr_excerpt=err[:200] or None,
            signals=signals,
        )
    if role in {"brain", "council"}:
        parsed = _parse_json_object(out)
        if not isinstance(parsed, dict):
            return ProbeResult(
                status="DEGRADED",
                latency_ms=latency_ms,
                reason="invalid_json",
                error=out[:200],
                checked_at=_now_ts(),
                signals=signals,
            )
        signals["valid_json"] = True
    tier = str(cfg.get("tier") or "balanced")
    if latency_ms > _latency_threshold_ms(tier):
        signals["latency_note"] = "slow"
    return ProbeResult(status="UP", latency_ms=latency_ms, checked_at=_now_ts(), command=cmd_str, signals=signals)


def _probe_codex_jsonl(provider: str, cfg: Dict[str, Any], role: str, timeout_sec: int) -> ProbeResult:
    model = cfg.get("default_model") or cfg.get("model") or ""
    if os.name == "nt":
        wsl = shutil.which("wsl.exe")
        if not wsl:
            return ProbeResult(
                status="DEGRADED",
                reason="wsl_missing",
                error="wsl.exe not found",
                checked_at=_now_ts(),
                command="wsl.exe (missing)",
            )
        distro = (os.getenv("AJAX_WSL_DISTRO") or os.getenv("WSL_DISTRO_NAME") or "Ubuntu").strip()
        model_arg = f" --model {shlex.quote(str(model))}" if isinstance(model, str) and model.strip() else ""
        cmd = [
            wsl,
            "-d",
            distro,
            "--",
            "bash",
            "-lc",
            f"echo ping | codex exec --json --skip-git-repo-check{model_arg}",
        ]
        cmd_str = "wsl: " + " ".join(cmd)
        stdin = None
    else:
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check"]
        if isinstance(model, str) and model.strip():
            cmd.extend(["--model", str(model).strip()])
        cmd_str = "native: " + " ".join(cmd)
        stdin = "ping\n"
    started = _now_ts()
    try:
        proc = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        msg = str(exc)
        return ProbeResult(
            status="DEGRADED",
            reason="timeout",
            error=msg[:200],
            checked_at=_now_ts(),
            command=cmd_str,
        )
    except Exception as exc:
        msg = str(exc)
        lowered = msg.lower()
        reason = "sandbox_listen_eperm" if ("eperm" in lowered and ("listen" in lowered or "bind" in lowered)) else "unavailable"
        return ProbeResult(status="DEGRADED", reason=reason, error=msg[:200], checked_at=_now_ts(), command=cmd_str)
    latency_ms = int((_now_ts() - started) * 1000)
    raw = (proc.stdout or "").strip()
    signals = {"latency_ms": latency_ms}
    last_text, last_error = _extract_codex_jsonl(raw)
    jsonl_events = 0
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            json.loads(line)
        except Exception:
            continue
        jsonl_events += 1
    signals["jsonl_events"] = jsonl_events
    if proc.returncode != 0:
        msg = (last_error or "").strip() or (proc.stderr or "").strip() or raw[:200] or f"codex_rc_{proc.returncode}"
        lowered = msg.lower()
        reason = "sandbox_listen_eperm" if ("eperm" in lowered and ("listen" in lowered or "bind" in lowered)) else "unavailable"
        return ProbeResult(
            status="DEGRADED",
            latency_ms=latency_ms,
            reason=reason,
            error=msg[:200],
            checked_at=_now_ts(),
            command=cmd_str,
            stderr_excerpt=(proc.stderr or "")[:200] or None,
            signals=signals,
        )
    if jsonl_events <= 0:
        return ProbeResult(
            status="DEGRADED",
            latency_ms=latency_ms,
            reason="invalid_jsonl",
            error=(raw or "")[:200],
            checked_at=_now_ts(),
            command=cmd_str,
            signals=signals,
        )
    signals["valid_json"] = True
    tier = str(cfg.get("tier") or "premium")
    if latency_ms > _latency_threshold_ms(tier):
        signals["latency_note"] = "slow"
    return ProbeResult(status="UP", latency_ms=latency_ms, checked_at=_now_ts(), command=cmd_str, signals=signals)


class ProviderBreathingLoop:
    def __init__(self, *, root_dir: Path, provider_configs: Dict[str, Any]) -> None:
        self.root_dir = Path(root_dir)
        self.provider_configs = provider_configs
        self.status_path = self.root_dir / "artifacts" / "health" / "providers_status.json"
        self.auth = AuthManager(root_dir=self.root_dir)

    def run_once(self, *, roles: Optional[List[str]] = None) -> Dict[str, Any]:
        now = _now_ts()
        roles_filter = {r.lower() for r in (roles or ["brain", "council", "scout", "vision"])}
        providers_cfg = (self.provider_configs or {}).get("providers") or {}
        policy_doc = _load_policy()

        status_doc = _safe_read_json(self.status_path)
        providers_status = status_doc.get("providers")
        if not isinstance(providers_status, dict):
            providers_status = {}
            status_doc["providers"] = providers_status

        for name, cfg in providers_cfg.items():
            if not isinstance(cfg, dict) or cfg.get("disabled"):
                continue
            declared_roles = cfg.get("roles") or []
            if isinstance(declared_roles, str):
                declared_roles = [declared_roles]
            declared_roles_l = {str(r).lower() for r in declared_roles if r}
            probe_roles = sorted(declared_roles_l & roles_filter)
            if not probe_roles:
                continue

            entry = providers_status.get(name)
            if not isinstance(entry, dict):
                entry = {}
                providers_status[name] = entry
            entry["capabilities"] = _capabilities_for_cfg(cfg)
            entry["policy_state"] = {
                "text": _policy_state(policy_doc, name, "text"),
                "vision": _policy_state(policy_doc, name, "vision"),
            }
            auth_state = self.auth.auth_state(name, cfg)
            entry["auth_state"] = auth_state.state
            entry["auth_reason"] = auth_state.reason
            entry["auth_checked_at"] = now
            self.auth.ensure_auth_gap(name, auth_state)

            kind = str(cfg.get("kind") or "").lower()
            if auth_state.state in {"MISSING", "EXPIRED"} and AuthManager.is_web_auth_required(cfg):
                transport_res = ProbeResult(
                    status="DOWN",
                    reason="auth_missing",
                    error=auth_state.reason,
                    checked_at=now,
                    probe_kind="TRANSPORT",
                )
            elif kind == "http_openai":
                transport_res = _probe_http_transport(cfg, int(cfg.get("timeout_seconds") or 20))
            elif kind == "codex_cli_jsonl":
                transport_res = _probe_codex_transport(cfg, timeout_sec=3)
            else:
                transport_res = _probe_cli_transport(cfg)
            entry["transport"] = transport_res.to_dict()
            provider_up = transport_res.status == "UP"

            breathing = entry.get("breathing")
            if not isinstance(breathing, dict):
                breathing = {}
                entry["breathing"] = breathing
            roles_map = breathing.get("roles")
            if not isinstance(roles_map, dict):
                roles_map = {}
                breathing["roles"] = roles_map

            contract_worst = "UP"
            contract_reason = ""
            available_recent = False
            for role in probe_roles:
                if not provider_up:
                    res = ProbeResult(
                        status="DOWN",
                        reason="transport_down",
                        error=transport_res.reason,
                        checked_at=now,
                        probe_kind="CONTRACT",
                    )
                else:
                    timeout_sec = int(cfg.get("timeout_seconds") or 20)
                    if kind == "http_openai":
                        res = _probe_http_openai(name, cfg, role, timeout_sec)
                    elif kind == "codex_cli_jsonl":
                        res = _probe_codex_jsonl(name, cfg, role, timeout_sec)
                    else:
                        res = _probe_cli(name, cfg, role, timeout_sec)
                    res.probe_kind = "CONTRACT"
                roles_map[role] = res.to_dict()
                if res.status == "DOWN":
                    contract_worst = "DOWN"
                    contract_reason = res.reason or contract_reason
                elif res.status == "DEGRADED" and contract_worst != "DOWN":
                    contract_worst = "DEGRADED"
                    contract_reason = res.reason or contract_reason
                if isinstance(res.signals, dict) and res.signals.get("valid_json"):
                    available_recent = True

            breathing["status"] = "UP" if provider_up else "DOWN"
            breathing["reason"] = transport_res.reason or ""
            breathing["contract_status"] = contract_worst
            breathing["contract_reason"] = contract_reason
            breathing["last_probe_ts"] = now
            breathing["last_probe_utc"] = _iso_now(now)
            entry["available_recent"] = bool(available_recent)
            entry["available_recent_checked_at"] = now
            reason = ""
            policy_state = entry.get("policy_state") if isinstance(entry.get("policy_state"), dict) else {}
            policy_text = str((policy_state or {}).get("text") or "").strip().lower()
            if policy_text in {"disallowed", "blocked", "deny"}:
                reason = "policy_disallowed"
            auth_state = str(entry.get("auth_state") or "").upper()
            if not reason and auth_state == "MISSING":
                reason = "auth_missing"
            elif not reason and auth_state == "EXPIRED":
                reason = "auth_expired"
            elif not reason and breathing.get("status") == "DOWN":
                reason = "transport_down"
            elif not reason and breathing.get("contract_status") == "DOWN":
                reason = "contract_down"
            entry["unavailable_reason"] = reason or None

        status_doc["updated_at"] = now
        status_doc["updated_utc"] = _iso_now(now)
        _safe_write_json(self.status_path, status_doc)
        return status_doc


def _already_running(pidfile: Path) -> bool:
    if not pidfile.exists():
        return False
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except Exception:
        try:
            pidfile.unlink()
        except Exception:
            pass
        return False


def run_loop(interval: float, max_cycles: Optional[int]) -> int:
    provider_configs = _load_provider_configs(ROOT)
    loop = ProviderBreathingLoop(root_dir=ROOT, provider_configs=provider_configs)
    cycles = 0
    while True:
        cycles += 1
        loop.run_once()
        if max_cycles and cycles >= max_cycles:
            break
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Provider breathing loop: probes ligeros y estado rolling.")
    parser.add_argument("--interval", type=float, default=float(os.getenv("AJAX_PROVIDER_BREATHING_INTERVAL_SEC", "180")), help="Intervalo entre ciclos (s)")
    parser.add_argument("--max-cycles", type=int, default=None, help="Máximo de ciclos (para test).")
    parser.add_argument("--daemon", action="store_true", help="Ejecutar en background (pidfile).")
    parser.add_argument("--pidfile", type=Path, default=PIDFILE, help="Ruta pidfile.")
    args = parser.parse_args(argv)
    _load_env_file(ROOT / ".env")

    if args.daemon:
        if _already_running(args.pidfile):
            print("Provider breathing ya está en ejecución.")
            return 0
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--interval",
                str(args.interval),
                "--max-cycles",
                str(args.max_cycles) if args.max_cycles else "0",
                "--pidfile",
                str(args.pidfile),
            ],
            cwd=ROOT,
            start_new_session=True,
        )
        args.pidfile.parent.mkdir(parents=True, exist_ok=True)
        args.pidfile.write_text(str(proc.pid), encoding="utf-8")
        print(f"Provider breathing iniciado (pid={proc.pid}).")
        return 0

    if args.pidfile:
        try:
            args.pidfile.parent.mkdir(parents=True, exist_ok=True)
            args.pidfile.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass
    try:
        return run_loop(args.interval, args.max_cycles if args.max_cycles and args.max_cycles > 0 else None)
    finally:
        try:
            args.pidfile.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
