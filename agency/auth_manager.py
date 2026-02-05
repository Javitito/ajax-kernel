from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import shutil


@dataclass(frozen=True)
class AuthState:
    state: str  # "OK" | "MISSING" | "EXPIRED"
    reason: str = ""
    instructions: str = ""
    checked_at: float = 0.0


def _now_ts() -> float:
    return time.time()


def _iso_now(ts: Optional[float] = None) -> str:
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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        return


_WSL_HOME: Optional[str] = None
_WSL_HOME_CHECKED = False


def _wsl_available() -> bool:
    return os.name == "nt" and shutil.which("wsl.exe") is not None


def _get_wsl_home() -> Optional[str]:
    global _WSL_HOME, _WSL_HOME_CHECKED
    if _WSL_HOME_CHECKED:
        return _WSL_HOME
    _WSL_HOME_CHECKED = True
    if not _wsl_available():
        _WSL_HOME = None
        return None
    try:
        proc = subprocess.run(
            ["wsl.exe", "--", "bash", "-lc", "echo -n $HOME"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            home = proc.stdout.strip()
            _WSL_HOME = home or None
        else:
            _WSL_HOME = None
    except Exception:
        _WSL_HOME = None
    return _WSL_HOME


def _wsl_path_exists(rel_path: str) -> bool:
    home = _get_wsl_home()
    if not home:
        return False
    target = rel_path.replace("~", home, 1)
    try:
        proc = subprocess.run(
            ["wsl.exe", "--", "bash", "-lc", f"if [ -s '{target}' ]; then exit 0; else exit 1; fi"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _wsl_env_has(key: str) -> bool:
    if not _wsl_available():
        return False
    k = str(key).strip()
    if not k:
        return False
    try:
        proc = subprocess.run(
            ["wsl.exe", "--", "bash", "-lc", f"if [ -n \"${k}\" ]; then exit 0; else exit 1; fi"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


class AuthManager:
    """
    Detecta auth requerida por provider y emite gaps accionables.

    Heurística v0 (best-effort):
    - http_openai con api_key_env => requiere env var.
    - CLI qwen => requiere QWEN_OAUTH u OPENAI_API_KEY o settings.json con auth.
    - CLI gemini => requiere credenciales locales o env vars (GEMINI_API_KEY/GOOGLE_API_KEY).
    - codex_cli_jsonl => requiere auth local (~/.codex/auth.json) o OPENAI_API_KEY.
    """

    def __init__(self, *, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.gaps_dir = self.root_dir / "artifacts" / "capability_gaps"
        self.health_dir = self.root_dir / "artifacts" / "health"
        self.status_path = self.health_dir / "providers_status.json"

    @staticmethod
    def is_web_auth_required(provider_cfg: Dict[str, Any]) -> bool:
        kind = str(provider_cfg.get("kind") or "").lower()
        if kind == "http_openai":
            return bool(provider_cfg.get("api_key_env"))
        if kind == "codex_cli_jsonl":
            return True
        if kind == "cli":
            cmd = provider_cfg.get("command") or []
            tool = str(cmd[0]) if isinstance(cmd, list) and cmd else ""
            return tool in {"qwen", "gemini"}
        return False

    def auth_state(self, provider_name: str, provider_cfg: Dict[str, Any]) -> AuthState:
        now = _now_ts()
        kind = str(provider_cfg.get("kind") or "").lower()
        if kind == "http_openai":
            api_key_env = provider_cfg.get("api_key_env")
            if not api_key_env:
                return AuthState(state="OK", checked_at=now)
            env_key = str(api_key_env)
            if os.getenv(env_key):
                return AuthState(state="OK", checked_at=now)
            return AuthState(
                state="MISSING",
                reason=f"env_missing:{env_key}",
                instructions=f"Configura `{env_key}` en tu entorno o en `.env`.",
                checked_at=now,
            )

        if kind == "codex_cli_jsonl":
            if os.name == "nt":
                if _wsl_path_exists("~/.codex/auth.json") or _wsl_env_has("OPENAI_API_KEY") or _wsl_env_has("CODEX_API_KEY"):
                    return AuthState(state="OK", reason="wsl_credentials", checked_at=now)
            else:
                if os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY"):
                    return AuthState(state="OK", checked_at=now)
                auth_path = Path("~/.codex/auth.json").expanduser()
                if auth_path.exists() and auth_path.stat().st_size > 0:
                    return AuthState(state="OK", checked_at=now)
            return AuthState(
                state="MISSING",
                reason="codex_auth_missing",
                instructions="Inicia sesión/configura Codex CLI (p.ej. `codex login`) o exporta `OPENAI_API_KEY`.",
                checked_at=now,
            )

        if kind == "cli":
            cmd = provider_cfg.get("command") or []
            tool = str(cmd[0]) if isinstance(cmd, list) and cmd else ""
            if tool == "qwen":
                oauth_path = Path("~/.qwen/oauth_creds.json").expanduser()
                if oauth_path.exists() and oauth_path.stat().st_size > 0:
                    return AuthState(state="OK", checked_at=now)
                if os.getenv("QWEN_OAUTH") or os.getenv("OPENAI_API_KEY"):
                    return AuthState(state="OK", checked_at=now)
                settings_path = Path("~/.qwen/settings.json").expanduser()
                data = _safe_read_json(settings_path)
                auth = (((data.get("security") or {}) if isinstance(data, dict) else {}).get("auth") or {})
                if isinstance(auth, dict) and any(auth.values()):
                    return AuthState(state="OK", checked_at=now)
                if _wsl_path_exists("~/.qwen/oauth_creds.json") or _wsl_path_exists("~/.qwen/settings.json"):
                    return AuthState(state="OK", reason="wsl_credentials", checked_at=now)
                return AuthState(
                    state="MISSING",
                    reason="qwen_auth_missing",
                    instructions=(
                        "Configura Qwen Code auth en `~/.qwen/settings.json` o exporta `QWEN_OAUTH` "
                        "(alternativa: `OPENAI_API_KEY` si tu CLI usa auth tipo OpenAI)."
                    ),
                    checked_at=now,
                )
            if tool == "gemini":
                oauth_path = Path("~/.gemini/oauth_creds.json").expanduser()
                if oauth_path.exists() and oauth_path.stat().st_size > 0:
                    return AuthState(state="OK", checked_at=now)
                if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                    return AuthState(state="OK", checked_at=now)
                adc = Path("~/.config/gcloud/application_default_credentials.json").expanduser()
                if adc.exists() and adc.stat().st_size > 0:
                    return AuthState(state="OK", checked_at=now)
                if _wsl_path_exists("~/.gemini/oauth_creds.json") or _wsl_path_exists("~/.config/gcloud/application_default_credentials.json"):
                    return AuthState(state="OK", reason="wsl_credentials", checked_at=now)
                return AuthState(
                    state="MISSING",
                    reason="gemini_auth_missing",
                    instructions="Configura credenciales para Gemini CLI (ADC de gcloud o `GEMINI_API_KEY`).",
                    checked_at=now,
                )

        return AuthState(state="OK", checked_at=now)

    def auth_source(self, provider_name: str, provider_cfg: Dict[str, Any]) -> Optional[str]:
        """
        Best-effort: devuelve la fuente de auth detectada (env|file|wsl_file).
        """
        kind = str(provider_cfg.get("kind") or "").lower()
        if kind == "http_openai":
            api_key_env = provider_cfg.get("api_key_env")
            if api_key_env and os.getenv(str(api_key_env)):
                return f"env:{api_key_env}"
            return None
        if kind == "codex_cli_jsonl":
            if os.getenv("OPENAI_API_KEY"):
                return "env:OPENAI_API_KEY"
            if os.getenv("CODEX_API_KEY"):
                return "env:CODEX_API_KEY"
            auth_path = Path("~/.codex/auth.json").expanduser()
            if auth_path.exists() and auth_path.stat().st_size > 0:
                return "file:~/.codex/auth.json"
            if _wsl_path_exists("~/.codex/auth.json"):
                return "wsl_file:~/.codex/auth.json"
            if _wsl_env_has("OPENAI_API_KEY"):
                return "wsl_env:OPENAI_API_KEY"
            if _wsl_env_has("CODEX_API_KEY"):
                return "wsl_env:CODEX_API_KEY"
            return None
        if kind == "cli":
            cmd = provider_cfg.get("command") or []
            tool = str(cmd[0]) if isinstance(cmd, list) and cmd else ""
            if tool == "qwen":
                if os.getenv("QWEN_OAUTH"):
                    return "env:QWEN_OAUTH"
                if os.getenv("OPENAI_API_KEY"):
                    return "env:OPENAI_API_KEY"
                oauth_path = Path("~/.qwen/oauth_creds.json").expanduser()
                if oauth_path.exists() and oauth_path.stat().st_size > 0:
                    return "file:~/.qwen/oauth_creds.json"
                settings_path = Path("~/.qwen/settings.json").expanduser()
                if settings_path.exists() and settings_path.stat().st_size > 0:
                    return "file:~/.qwen/settings.json"
                if _wsl_path_exists("~/.qwen/oauth_creds.json"):
                    return "wsl_file:~/.qwen/oauth_creds.json"
                if _wsl_path_exists("~/.qwen/settings.json"):
                    return "wsl_file:~/.qwen/settings.json"
                if _wsl_env_has("QWEN_OAUTH"):
                    return "wsl_env:QWEN_OAUTH"
                if _wsl_env_has("OPENAI_API_KEY"):
                    return "wsl_env:OPENAI_API_KEY"
                return None
            if tool == "gemini":
                if os.getenv("GEMINI_API_KEY"):
                    return "env:GEMINI_API_KEY"
                if os.getenv("GOOGLE_API_KEY"):
                    return "env:GOOGLE_API_KEY"
                oauth_path = Path("~/.gemini/oauth_creds.json").expanduser()
                if oauth_path.exists() and oauth_path.stat().st_size > 0:
                    return "file:~/.gemini/oauth_creds.json"
                adc = Path("~/.config/gcloud/application_default_credentials.json").expanduser()
                if adc.exists() and adc.stat().st_size > 0:
                    return "file:~/.config/gcloud/application_default_credentials.json"
                if _wsl_path_exists("~/.gemini/oauth_creds.json"):
                    return "wsl_file:~/.gemini/oauth_creds.json"
                if _wsl_path_exists("~/.config/gcloud/application_default_credentials.json"):
                    return "wsl_file:~/.config/gcloud/application_default_credentials.json"
                if _wsl_env_has("GEMINI_API_KEY"):
                    return "wsl_env:GEMINI_API_KEY"
                if _wsl_env_has("GOOGLE_API_KEY"):
                    return "wsl_env:GOOGLE_API_KEY"
                return None
        return None

    def persist_auth_state(self, provider_name: str, auth_state: AuthState) -> None:
        data = _safe_read_json(self.status_path)
        providers = data.get("providers")
        if not isinstance(providers, dict):
            providers = {}
            data["providers"] = providers
        entry = providers.get(provider_name)
        if not isinstance(entry, dict):
            entry = {}
            providers[provider_name] = entry
        entry["auth_state"] = auth_state.state
        entry["auth_reason"] = auth_state.reason
        entry["auth_checked_at"] = auth_state.checked_at or _now_ts()
        data["updated_at"] = _now_ts()
        _safe_write_json(self.status_path, data)

    def ensure_auth_gap(self, provider_name: str, auth_state: AuthState) -> Optional[Path]:
        if auth_state.state not in {"MISSING", "EXPIRED"}:
            return None
        self.gaps_dir.mkdir(parents=True, exist_ok=True)
        gap_path = self.gaps_dir / f"auth_required_{provider_name}.json"
        previous = _safe_read_json(gap_path)
        occurrences = 0
        try:
            occurrences = int(previous.get("occurrences") or 0)
        except Exception:
            occurrences = 0
        payload = {
            "gap_id": f"AUTH_REQUIRED_{provider_name}",
            "capability_family": "auth.required",
            "provider": provider_name,
            "auth_state": auth_state.state,
            "reason": auth_state.reason,
            "instructions": auth_state.instructions,
            "occurrences": occurrences + 1,
            "created_at": _iso_now(),
            "evidence_paths": [str(self.status_path)],
        }
        _safe_write_json(gap_path, payload)
        return gap_path
