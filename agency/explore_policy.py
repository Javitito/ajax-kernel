from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
    return data if isinstance(data, dict) else None


def load_explore_policy(root_dir: Path) -> Dict[str, Any]:
    cfg_path = Path(root_dir) / "config" / "explore_policy.yaml"
    if cfg_path.exists() and yaml is not None:
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def state_rules(cfg: Dict[str, Any], state: str) -> Dict[str, Any]:
    states = cfg.get("states") if isinstance(cfg.get("states"), dict) else {}
    if isinstance(states, dict):
        entry = states.get(state)
        if isinstance(entry, dict):
            return entry
    # Back-compat: allow older shape policy.away / policy.human_detected
    pol = cfg.get("policy") if isinstance(cfg.get("policy"), dict) else {}
    key = "away" if state == "AWAY" else "human_detected"
    legacy = pol.get(key)
    return legacy if isinstance(legacy, dict) else {}


def _latest_display_probe_receipt(root_dir: Path) -> Optional[Dict[str, Any]]:
    base = Path(root_dir) / "artifacts" / "ops" / "display_probe"
    if not base.exists():
        return None
    try:
        candidates = [p for p in base.iterdir() if p.is_dir()]
    except Exception:
        return None
    for folder in sorted(candidates, key=lambda p: p.name, reverse=True):
        receipt = folder / "receipt.json"
        payload = _read_json(receipt)
        if isinstance(payload, dict):
            return payload
    return None


def dummy_display_ok(root_dir: Path) -> bool:
    receipt = _latest_display_probe_receipt(root_dir)
    if not isinstance(receipt, dict):
        return False
    return bool(receipt.get("lab_zone_ok"))


def read_human_signal(root_dir: Path, *, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Query Windows host for human signals.
    Returns schema ajax.human_signal.v1-ish:
      {ok,last_input_age_sec,session_unlocked,error,ts_utc}
    """
    cfg = policy or load_explore_policy(root_dir)
    signal_cfg = cfg.get("human_signal") if isinstance(cfg.get("human_signal"), dict) else {}
    script_rel = signal_cfg.get("ps_script") or "scripts/ops/get_human_signal.ps1"
    script_path = (Path(root_dir) / str(script_rel)).resolve()
    try:
        timeout_s = float(signal_cfg.get("timeout_s") or 2.5)
    except Exception:
        timeout_s = 2.5

    payload: Dict[str, Any] = {
        "schema": "ajax.human_signal.v1",
        "ok": False,
        "last_input_age_sec": None,
        "session_unlocked": None,
        "error": None,
        "ts_utc": _utc_now(),
        "probe": {
            "script": str(script_path),
            "script_windows": None,
            "timeout_s": timeout_s,
            "command": "powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File <script>",
        },
    }

    if not script_path.exists():
        payload["error"] = "human_signal_script_missing"
        return payload

    def _to_windows_path(posix_path: str) -> Optional[str]:
        raw = (posix_path or "").strip()
        if not raw:
            return None
        # Heuristic: if already looks like Windows drive path, keep as-is.
        if len(raw) >= 3 and raw[1:3] == ":\\":
            return raw
        try:
            proc = subprocess.run(
                ["wslpath", "-w", raw],
                capture_output=True,
                text=False,
                timeout=1.0,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        win = _decode(proc.stdout or b"", "utf-8").strip()
        return win or None

    def _decode(b: bytes, encoding: str) -> str:
        try:
            return b.decode(encoding, errors="replace")
        except Exception:
            return ""

    def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    script_arg = str(script_path)
    script_win = _to_windows_path(script_arg)
    if script_win:
        payload["probe"]["script_windows"] = script_win
        script_arg = script_win

    try:
        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script_arg,
            ],
            capture_output=True,
            text=False,
            timeout=max(0.5, timeout_s),
        )
    except Exception as exc:
        payload["error"] = f"human_signal_probe_failed:{str(exc)[:160]}"
        return payload

    stdout_b = proc.stdout or b""
    stderr_b = proc.stderr or b""
    stdout_u8 = _decode(stdout_b, "utf-8").strip()
    stderr_u8 = _decode(stderr_b, "utf-8").strip()
    if proc.returncode != 0:
        payload["error"] = f"human_signal_rc_{proc.returncode}:{(stderr_u8 or stdout_u8)[:160]}"
        payload["probe"]["stderr_excerpt"] = stderr_u8[:200]
        payload["probe"]["stdout_excerpt"] = stdout_u8[:200]
        return payload

    data = _parse_json_from_text(stdout_u8)
    if data is None:
        stdout_1252 = _decode(stdout_b, "cp1252").strip()
        data = _parse_json_from_text(stdout_1252)
        if data is None:
            payload["error"] = f"human_signal_json_parse_failed:{stdout_u8[:160] or '<empty>'}"
            payload["probe"]["stdout_excerpt"] = stdout_u8[:200]
            payload["probe"]["stderr_excerpt"] = stderr_u8[:200]
            return payload

    payload["ok"] = bool(data.get("ok", True))
    payload["last_input_age_sec"] = data.get("last_input_age_sec")
    payload["session_unlocked"] = data.get("session_unlocked")
    if data.get("error"):
        payload["error"] = str(data.get("error"))[:200]
    return payload


def compute_human_active(
    signal: Dict[str, Any],
    *,
    threshold_s: float,
    unknown_as_human: bool,
) -> Tuple[bool, str]:
    # If the probe explicitly reports failure, do not trust any payload fields
    # (some bootstrap stubs emit placeholder values that would look "active").
    if signal.get("ok") is False:
        return (bool(unknown_as_human), "signal_not_ok")
    age = signal.get("last_input_age_sec")
    unlocked = signal.get("session_unlocked")
    try:
        age_v = float(age) if age is not None else None
    except Exception:
        age_v = None
    unlocked_v = None
    if isinstance(unlocked, bool):
        unlocked_v = unlocked
    if age_v is None or unlocked_v is None:
        return (bool(unknown_as_human), "unknown_signal")
    active = (age_v < float(threshold_s)) and bool(unlocked_v)
    return (bool(active), "ok")


def evaluate_explore_state(
    root_dir: Path,
    *,
    policy: Optional[Dict[str, Any]] = None,
    prev_state: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = policy or load_explore_policy(root_dir)
    pol = cfg.get("policy") if isinstance(cfg.get("policy"), dict) else {}
    try:
        threshold_s = float(pol.get("human_active_threshold_s") or 90)
    except Exception:
        threshold_s = 90.0
    unknown_as_human = bool(pol.get("unknown_signal_as_human", True))
    signal = read_human_signal(root_dir, policy=cfg)
    active, reason = compute_human_active(signal, threshold_s=threshold_s, unknown_as_human=unknown_as_human)
    state = "HUMAN_DETECTED" if active else "AWAY"
    trigger = None
    if prev_state and prev_state != state:
        trigger = f"{prev_state}->{state}"
    return {
        "schema": "ajax.explore_state.v1",
        "ts_utc": _utc_now(),
        "now_ts": float(now_ts or time.time()),
        "prev_state": prev_state,
        "state": state,
        "trigger": trigger,
        "human_active": bool(active),
        "human_active_reason": reason,
        "human_signal": signal,
    }
