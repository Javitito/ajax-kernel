#!/usr/bin/env python3
"""
Heartbeat unificado para AJAX.
Realiza checks ligeros sobre web (5002), RAG (8000) y driver (5010) y genera
``artifacts/health/ajax_heartbeat.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib import error, request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agency.driver_keys import load_ajax_driver_api_key
from agency.exercise_schema import normalize_verification, make_gap, now_iso

DEFAULT_OUT = ROOT / "artifacts" / "health" / "ajax_heartbeat.json"
VISION_CANARY = ROOT / "artifacts" / "vision" / "vision_canary_send_button.json"
VOICE_CANARY = ROOT / "artifacts" / "voice" / "voice_canary.json"
LEANN_CANARY = ROOT / "artifacts" / "health" / "leann_canary.json"
HISTORY_OUT = ROOT / "artifacts" / "health" / "ajax_heartbeat_history.jsonl"


def _read_env_or_file(env_keys: Tuple[str, ...], path_keys: Tuple[str, ...]) -> str:
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return val.strip()
    for path in path_keys:
        p = Path(os.path.expanduser(path))
        if p.exists():
            try:
                return p.read_text(encoding="utf-8").strip()
            except Exception:
                continue
    return ""


def _fetch(url: str, *, headers: Optional[Dict[str, str]] = None, method: str = "GET",
           data: Optional[bytes] = None, timeout: float = 6.0) -> Dict[str, Any]:
    req = request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            text = body.decode("utf-8", errors="replace") if body else ""
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "body": text}
    except error.HTTPError as exc:
        body = exc.read()
        text = body.decode("utf-8", errors="replace") if body else ""
        return {"ok": False, "status": exc.code, "error": str(exc), "body": text}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc), "body": ""}


def _status_label(ok: bool, *, degraded: bool = False) -> str:
    if ok:
        return "green"
    return "yellow" if degraded else "red"


def check_web_health() -> Dict[str, Any]:
    endpoints = [
        ("health", "http://127.0.0.1:5002/health"),
        ("adam_health", "http://127.0.0.1:5002/api/adam/health"),
        ("root", "http://127.0.0.1:5002/"),
    ]
    tried = []
    for name, url in endpoints:
        res = _fetch(url)
        ok = res.get("ok", False)
        status_txt = f"{name}:{res.get('status') or res.get('error', 'no response')}"
        tried.append(status_txt)
        if ok:
            return {"status": _status_label(True), "detail": status_txt}
    return {"status": _status_label(False), "detail": "; ".join(tried)}


def check_rag_health() -> Dict[str, Any]:
    token = _read_env_or_file(
        ("LEANN_RAG_AUTH_TOKEN", "AUTH_TOKEN"),
        ("~/.leann/rag_auth_token",),
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    res = _fetch("http://127.0.0.1:8000/health", headers=headers)
    ok = res.get("ok", False)
    degraded = not token and not ok  # sin token puede devolver 401; no es fallo duro del stack
    detail = f"http {res.get('status')}" if res.get("status") else res.get("error", "no response")
    if not token:
        detail = f"{detail} (sin token)"
    return {"status": _status_label(ok, degraded=degraded), "detail": detail}


def check_rag_query() -> Dict[str, Any]:
    api_key = _read_env_or_file(
        ("LEANN_WEB_API_KEY", "WEB_API_KEY", "API_KEY"),
        ("~/.leann/web_api_key",),
    )
    if not api_key:
        return {"status": "yellow", "detail": "sin web_api_key; query no ejecutada"}
    payload = json.dumps({"q": "canary", "k": 1}).encode("utf-8")
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    res = _fetch("http://127.0.0.1:5002/api/rag/query", headers=headers, method="POST", data=payload)
    ok = res.get("ok", False)
    detail = f"http {res.get('status')}" if res.get("status") else res.get("error", "no response")
    # considera éxito si hay 200 y cuerpo no vacío
    if ok and not res.get("body"):
        ok = False
        detail = f"{detail}; cuerpo vacío"
    return {"status": _status_label(ok), "detail": detail}


def check_driver() -> Dict[str, Any]:
    api_key = load_ajax_driver_api_key() or ""
    headers = {"X-AJAX-KEY": api_key} if api_key else {}
    res = _fetch("http://127.0.0.1:5010/health", headers=headers)
    ok = res.get("ok", False)
    detail = f"http {res.get('status')}" if res.get("status") else res.get("error", "no response")
    return {"status": _status_label(ok), "detail": detail}


def check_vision() -> Dict[str, Any]:
    if not VISION_CANARY.exists():
        return {"status": "green", "detail": "vision_canary_not_run"}
    try:
        data = json.loads(VISION_CANARY.read_text(encoding="utf-8"))
        status = str(data.get("status", "unknown")).lower()
        detail = data.get("detail") or "vision_canary"
        return {"status": status, "detail": detail}
    except Exception as exc:
        return {"status": "red", "detail": f"vision_canary_error:{exc}"}


def check_voice() -> Dict[str, Any]:
    if not VOICE_CANARY.exists():
        return {"status": "green", "detail": "voice_canary_not_run"}
    try:
        data = json.loads(VOICE_CANARY.read_text(encoding="utf-8"))
        status = str(data.get("status", "unknown")).lower()
        detail = data.get("detail") or "voice_canary"
        return {"status": status, "detail": detail}
    except Exception as exc:
        return {"status": "red", "detail": f"voice_canary_error:{exc}"}


def check_leann_canary() -> Dict[str, Any]:
    if not LEANN_CANARY.exists():
        return {"status": "green", "detail": "leann_canary_not_run"}
    try:
        data = json.loads(LEANN_CANARY.read_text(encoding="utf-8"))
        status = _status_label(data.get("status") == "green", degraded=data.get("status") == "yellow")
        detail = data.get("detail") or "leann_canary"
        return {"status": status, "detail": detail}
    except Exception as exc:
        return {"status": "red", "detail": f"leann_canary_error:{exc}"}


def aggregate_status(subsystems: Dict[str, Dict[str, Any]]) -> Tuple[str, list]:
    order = {"green": 0, "yellow": 1, "red": 2}
    worst = "green"
    problems = []
    for name, report in subsystems.items():
        status = report.get("status", "red")
        if order.get(status, 2) > order.get(worst, 0):
            worst = status
        if status != "green":
            problems.append(name)
    return worst, problems


def main() -> int:
    out_path = Path(sys.argv[sys.argv.index("--out") + 1]).expanduser() if "--out" in sys.argv else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subsystems = {
        "web_ui": check_web_health(),
        "rag_api": check_rag_health(),
        "rag_query": check_rag_query(),
        "driver": check_driver(),
        "vision": check_vision(),
        "voice": check_voice(),
        "leann": check_leann_canary(),
    }
    overall, problems = aggregate_status(subsystems)
    payload = {
        "timestamp": now_iso(),
        "status": overall,
        "subsystems": subsystems,
        "problems": problems,
    }
    if subsystems.get("driver", {}).get("status") != "green":
        payload["ok"] = False
        payload["reason"] = "DRIVER_DEGRADED"
        payload["recommended_action"] = "wait_or_restart"
    outcome = "success" if overall == "green" else ("neutral" if overall == "yellow" else "fail")
    gaps = []
    for name, report in subsystems.items():
        st = str(report.get("status", "unknown"))
        if st != "green":
            gaps.append(make_gap("subsystem", f"{name}:{st}", severity="medium", evidence=[report.get("detail", "")]))
    # Gap explícito para el caso crítico del driver (fail-fast)
    if subsystems.get("driver", {}).get("status") == "red":
        gaps.append(make_gap("infra", "DRIVER_UNAVAILABLE", severity="critical", evidence=[subsystems["driver"].get("detail", "")]))
    if not subsystems.get("vision") or subsystems["vision"].get("status") == "green":
        pass
    payload["verification"] = normalize_verification(
        {
            "outcome": outcome,
            "ok": overall == "green",
            "notes": problems,
            "metrics": {"subsystems_checked": len(subsystems)},
        },
        default_outcome=outcome,
    )
    payload["gaps"] = gaps
    payload["observability_gaps"] = [g["code"] for g in gaps]
    payload["tool_gaps"] = []
    payload["tools_used"] = ["infra.heartbeat"]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        HISTORY_OUT.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    # No forzamos código de error; el consumidor decide qué hacer con status=red
    return 0


if __name__ == "__main__":
    sys.exit(main())
