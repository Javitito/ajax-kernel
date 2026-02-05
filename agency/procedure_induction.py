from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "artifacts" / "exercises"


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _collect_visual_evidence(run_dir: Path) -> List[str]:
    visuals: List[str] = []
    candidates = [
        run_dir / "screenshots",
        run_dir / "vision_tiles",
        run_dir / "artifacts" / "vision_tiles",
    ]
    for folder in candidates:
        if folder.exists() and folder.is_dir():
            for item in folder.iterdir():
                if not item.is_file():
                    continue
                try:
                    visuals.append(str(item.relative_to(run_dir)))
                except Exception:
                    visuals.append(str(item))
    return visuals


def _confidence(plan_steps: List[Any], verification: Optional[Dict[str, Any]], audit: Optional[Dict[str, Any]]) -> float:
    score = 0.45
    if plan_steps:
        score += 0.1
    if audit:
        score += 0.05
        if audit.get("errors"):
            score -= 0.05
        if audit.get("result_ok"):
            score += 0.05
    ver_payload = verification or (audit.get("verification") if audit else None) or {}
    if isinstance(ver_payload, dict):
        outcome = (ver_payload.get("outcome") or (ver_payload.get("report") or {}).get("outcome") or "").lower()
        ok_flag = bool(ver_payload.get("ok") or (ver_payload.get("report") or {}).get("ok"))
        if ok_flag:
            score += 0.15
        if outcome in {"neutral", "unknown"}:
            score -= 0.05
        if outcome == "fail":
            score -= 0.1
    return max(0.0, min(1.0, round(score, 2)))


def _normalize_verification(verification: Optional[Dict[str, Any]], audit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = verification or (audit.get("verification") if audit else {}) or {}
    payload = dict(base) if isinstance(base, dict) else {}
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}

    outcome = (payload.get("outcome") or report.get("outcome") or "").lower() or "unknown"
    ok_flag = bool(payload.get("ok") or report.get("ok"))

    if outcome in {"neutral", "unknown"}:
        ok_flag = False

    payload["outcome"] = outcome
    payload["ok"] = ok_flag
    payload["is_terminal"] = outcome not in {"neutral", "unknown"}
    payload.setdefault("report", report)
    payload.setdefault("notes", payload.get("notes") or report.get("advice") or [])
    return payload


def induce_procedures(run_dir: Path, *, run_id: Optional[str] = None, out: Optional[Path] = None) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    run_id = run_id or run_dir.name

    plan_path = run_dir / "plan.json"
    audit_path = run_dir / "audit_log.json"
    verification_path = run_dir / "verification.json"

    plan = _safe_load_json(plan_path) or {}
    audit = _safe_load_json(audit_path)
    verification = _safe_load_json(verification_path)
    plan_steps: List[Any] = (plan.get("plan") if isinstance(plan, dict) else None) or []
    audit_actions: List[Dict[str, Any]] = (audit.get("actions") if isinstance(audit, dict) else None) or []
    visuals = _collect_visual_evidence(run_dir)

    obs_gaps: List[str] = []
    if not plan_steps:
        obs_gaps.append("missing_plan_steps")
    if audit is None:
        obs_gaps.append("missing_audit_log")
    if verification is None and not (audit and audit.get("verification")):
        obs_gaps.append("missing_verification")
    if not visuals:
        obs_gaps.append("no_visual_evidence")

    tool_gaps: List[str] = []
    for step in plan_steps:
        action = ""
        if isinstance(step, dict):
            action = str(step.get("tool") or step.get("action") or "").strip()
        if action and not any(a.get("name") == action for a in audit_actions):
            tool_gaps.append(f"no_audit_for:{action}")
    gaps: List[Dict[str, str]] = [{"kind": "observability", "code": code} for code in sorted(set(obs_gaps))]
    gaps.extend({"kind": "tool", "code": code} for code in sorted(set(tool_gaps)))

    steps_summary: List[Dict[str, Any]] = []
    for idx, step in enumerate(plan_steps, start=1):
        if isinstance(step, dict):
            action = str(step.get("tool") or step.get("action") or "unknown")
            args = step.get("args") if isinstance(step.get("args"), dict) else {
                k: v for k, v in step.items() if k not in {"tool", "action"}
            }
        else:
            action = "unknown"
            args = {}
        audit_match = next((a for a in audit_actions if a.get("name") == action), None)
        steps_summary.append(
            {
                "step_id": idx,
                "action": action,
                "args": args if isinstance(args, dict) else {},
                "audit": audit_match,
            }
        )

    verification_payload = _normalize_verification(verification, audit)
    confidence = _confidence(plan_steps, verification_payload, audit)
    audit_errors = (audit or {}).get("errors") or []
    audit_errors = [e for e in audit_errors if str(e) != "verification_outcome:neutral"]

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at_ts": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "plan_path": str(plan_path),
            "audit_log_path": str(audit_path),
            "verification_path": str(verification_path),
            "visual_assets": visuals,
        },
        "procedures": [
            {
                "name": f"{run_id}_procedure",
                "source": "plan",
                "confidence": confidence,
                "steps": steps_summary,
                "verification": verification_payload,
                "observations": {
                    "audit_errors": audit_errors,
                    "visual_evidence": visuals,
                },
            }
        ],
        "gaps": gaps,
        "observability_gaps": sorted(set(obs_gaps)),
        "tool_gaps": sorted(set(tool_gaps)),
    }

    target = out
    if target is None:
        target = DEFAULT_OUT_DIR / f"procedure_induction_{int(payload['created_at_ts'])}.json"
    if not target.is_absolute():
        target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Procedure induction (analitico, sin ejecucion).")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directorio de run con plan/audit/verification.")
    parser.add_argument("--run-id", type=str, help="Identificador de run (por defecto nombre del directorio).")
    parser.add_argument(
        "--out",
        type=Path,
        help="Ruta de salida JSON (por defecto artifacts/exercises/procedure_induction_<ts>.json).",
    )
    args = parser.parse_args()

    payload = induce_procedures(args.run_dir, run_id=args.run_id, out=args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
