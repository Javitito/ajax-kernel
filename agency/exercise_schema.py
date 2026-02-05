from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def normalize_verification(data: Optional[Dict[str, Any]], *, default_outcome: str = "neutral") -> Dict[str, Any]:
    """
    Normaliza estructura de verificación para ejercicios.
    - outcome: success|fail|neutral|unknown (lowercase)
    - ok: False si outcome neutral/unknown
    - is_terminal: True salvo neutral/unknown
    """
    base = dict(data or {})
    report = base.get("report") if isinstance(base.get("report"), dict) else {}
    outcome = (base.get("outcome") or report.get("outcome") or default_outcome or "").lower() or "unknown"
    ok_flag = bool(base.get("ok") or report.get("ok"))
    if outcome in {"neutral", "unknown"}:
        ok_flag = False
    if outcome == "fail":
        ok_flag = False
    base["outcome"] = outcome
    base["ok"] = ok_flag
    base["is_terminal"] = outcome not in {"neutral", "unknown"}
    base.setdefault("report", report)
    base.setdefault("notes", base.get("notes") or report.get("advice") or [])
    return base


def make_gap(kind: str, code: str, severity: str = "low", evidence: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    ev_list = [str(e) for e in (evidence or []) if str(e)]
    return {
        "kind": str(kind),
        "code": str(code),
        "severity": str(severity),
        "evidence": ev_list,
    }


def extract_tools_used_from_steps(steps: Optional[Iterable[Any]]) -> List[str]:
    tools: List[str] = []
    if not steps:
        return tools
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or step.get("tool") or "").strip()
        if action:
            tools.append(action)
        # algunos ejercicios guardan nested audit con name
        audit = step.get("audit")
        if isinstance(audit, dict):
            name = str(audit.get("name") or "").strip()
            if name:
                tools.append(name)
    return sorted(set(t for t in tools if t))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
