from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _fmt_verification(ver: Optional[Dict[str, Any]]) -> str:
    if not ver:
        return "verification: missing"
    outcome = ver.get("outcome", "unknown")
    ok = ver.get("ok", False)
    notes = ver.get("notes") or []
    return f"verification: {outcome} (ok={ok})" + (f" notes: {notes}" if notes else "")


def _gaps_section(gaps: List[Dict[str, Any]]) -> str:
    if not gaps:
        return "- gaps: none"
    lines = ["- gaps:"]
    for g in gaps:
        lines.append(f"  - {g.get('kind','?')}::{g.get('code','?')} severity={g.get('severity','?')}")
    return "\n".join(lines)


def _tool_use_notes(notes_path: Path) -> str:
    data = _load_json(notes_path) or {}
    lines = ["- tool use notes:"]
    if not data:
        lines.append("  - none")
        return "\n".join(lines)
    for tool, entries in data.items():
        lines.append(f"  - {tool}: {len(entries)} uses")
    return "\n".join(lines)


def _breaker_snapshot() -> Dict[str, Any]:
    gov_dir = ROOT / "artifacts" / "governance"
    mission = _load_json(gov_dir / "mission_breaker_state.json") or {}
    infra = _load_json(gov_dir / "infra_breaker_state.json") or {}
    return {"mission": mission, "infra": infra}


def _gaps_snapshot() -> List[Dict[str, Any]]:
    gaps_dir = ROOT / "artifacts" / "capability_gaps"
    gaps: List[Dict[str, Any]] = []
    if gaps_dir.exists():
        for p in gaps_dir.glob("*.json"):
            g = _load_json(p)
            if isinstance(g, dict):
                g["path"] = str(p)
                gaps.append(g)
    return gaps


def compute_intervention(now_ts: float, heartbeat: Dict[str, Any], gym: Dict[str, Any]) -> Dict[str, Any]:
    breakers = _breaker_snapshot()
    gaps = _gaps_snapshot()
    triggered: List[str] = []
    actions: List[str] = []

    hb_ver = heartbeat.get("verification") if isinstance(heartbeat.get("verification"), dict) else {}
    hb_status = heartbeat.get("status")
    hb_ok = bool(hb_ver.get("ok"))
    hb_outcome = str(hb_ver.get("outcome") or "").lower()
    if hb_status == "red" or hb_outcome == "fail" or not hb_ok:
        triggered.append("heartbeat_red")
        actions.append("jobs/fix_infra_basic.json")
        actions.append("Start-AjaxDriver.ps1")

    gym_ver = gym.get("verification") if isinstance(gym.get("verification"), dict) else {}
    if not gym_ver.get("ok", False):
        # exercise_required gaps
        req_fail = [g for g in gym.get("gaps") or [] if g.get("kind") == "exercise_required"]
        if req_fail:
            triggered.append("exercise_required_failed")
            actions.append("resolver ejercicios required (ver gym_daily)")

    mission_block = bool((breakers.get("mission") or {}).get("mission_should_block") or (breakers.get("mission") or {}).get("last_blocked"))
    infra_block = bool((breakers.get("infra") or {}).get("infra_should_block") or (breakers.get("infra") or {}).get("last_blocked"))
    if mission_block or infra_block:
        triggered.append("breaker_tripped")
        actions.append("inspeccionar artifacts/governance/*breaker_state.json")

    mv = _load_json(ROOT / "artifacts" / "motivo_vital" / "latest.json") or {}
    autonomy = (mv.get("dimensions") or {}).get("autonomy")
    if isinstance(autonomy, (int, float)) and autonomy < 0.5:
        triggered.append("autonomy_low")
        actions.append("planear ejercicio nuevo + mejora observabilidad/tool")

    level = "green"
    reason = "Salud estable; no intervenir."
    if triggered:
        level = "yellow"
        reason = "Hay señales a vigilar; dejar marcapasos un ciclo."
    if "heartbeat_red" in triggered or "exercise_required_failed" in triggered or "breaker_tripped" in triggered:
        level = "red"
        reason = "Señales críticas detectadas (heartbeat/breaker/required)."

    return {
        "timestamp": now_ts,
        "level": level,
        "reason": reason,
        "triggered_rules": triggered,
        "recommended_actions": actions,
        "sources": {
            "heartbeat": "artifacts/health/ajax_heartbeat.json",
            "gym": "artifacts/exercises/gym_daily.json",
            "breakers": "artifacts/governance/",
            "mv": "artifacts/motivo_vital/latest.json",
        },
    }


def build_daily_report(root: Path = ROOT) -> str:
    gym = _load_json(root / "artifacts" / "exercises" / "gym_daily.json") or {}
    heartbeat = _load_json(root / "artifacts" / "health" / "ajax_heartbeat.json") or {}
    mv = _load_json(root / "artifacts" / "motivo_vital" / "latest.json") or {}
    tool_notes_path = root / "artifacts" / "tools" / "tool_use_notes.json"
    now_ts = time.time()
    intervention = compute_intervention(now_ts, heartbeat, gym)

    lines: List[str] = []
    lines.append("# Daily Report")
    lines.append("")
    lines.append("## Cómo leer este reporte")
    lines.append("- Revisa Gym para ver si los ejercicios requeridos pasaron (ok=true).")
    lines.append("- Heartbeat indica salud de subsistemas; rojo implica revisar red/driver/web.")
    lines.append("- Motivo Vital muestra tendencia y dimensiones bajas.")
    lines.append("- Tools muestra uso acumulado; mira gaps y notas para priorizar.")
    lines.append("")
    # Resumen breve
    lines.append("## Resumen human-friendly")
    gym_ok = gym.get("verification", {}).get("ok")
    hb_status = heartbeat.get("status", "unknown")
    mv_score = mv.get("score", "?")
    tools_used = ", ".join(gym.get("tools_used") or [])
    lines.append(f"- Gym: outcome={gym.get('verification', {}).get('outcome','?')} ok={gym_ok}")
    lines.append(f"- Heartbeat: status={hb_status}")
    lines.append(f"- MV score: {mv_score}")
    lines.append(f"- Tools used: {tools_used or 'n/a'}")
    lines.append("")
    lines.append("## Gym")
    lines.append("_Ejercicios diarios y verificación; required debe ser success para ok=true._")
    lines.append(f"- status: {gym.get('verification', {}).get('outcome', 'unknown')} (ok={gym.get('verification', {}).get('ok')})")
    lines.append(_gaps_section(gym.get("gaps") or []))
    lines.append("- tools_used: " + ", ".join(gym.get("tools_used") or []))
    lines.append("")
    lines.append("## Heartbeat")
    lines.append("_Salud web/rag/driver/visión/voz/LEANN; rojo implica bloqueo de ejecuciones normales._")
    lines.append(f"- status: {heartbeat.get('status','unknown')}")
    lines.append(f"- {_fmt_verification(heartbeat.get('verification') if isinstance(heartbeat.get('verification'), dict) else None)}")
    lines.append(_gaps_section(heartbeat.get("gaps") or []))
    lines.append("")
    lines.append("## Motivo Vital")
    lines.append("_Tendencia evolutiva; mira dimensiones bajas para priorizar mejoras._")
    lines.append(f"- score: {mv.get('score','?')}")
    lines.append("- dimensions:")
    for name, val in (mv.get("dimensions") or {}).items():
        lines.append(f"  - {name}: {val}")
    if mv.get("deltas"):
        lines.append("- deltas:")
        for name, val in mv["deltas"].items():
            lines.append(f"  - {name}: {val:+.4f}")
    lines.append("")
    lines.append("## Tools")
    lines.append("_Uso acumulado de herramientas (Tool Use Notes)._")
    lines.append(_tool_use_notes(tool_notes_path))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- gym: artifacts/exercises/gym_daily.json")
    lines.append("- heartbeat: artifacts/health/ajax_heartbeat.json")
    lines.append("- motivo_vital: artifacts/motivo_vital/latest.json")
    lines.append(f"- tool_use_notes: {tool_notes_path}")
    lines.append("")
    # Próximos pasos sugeridos (determinista)
    lines.append("## Próximos pasos sugeridos")
    suggestions: List[str] = []
    # Heartbeat rojo/fail
    hb_outcome = (heartbeat.get("verification") or {}).get("outcome", "")
    hb_ok = (heartbeat.get("verification") or {}).get("ok", False)
    if heartbeat.get("status") == "red" or hb_outcome == "fail" or not hb_ok:
        suggestions.append("- Revisar heartbeat: web_ui/rag_api/driver; lanzar fix_infra_basic y comprobar driver 5010/web_ui.")
    # Required fails en Gym
    req_gaps = [g for g in gym.get("gaps") or [] if g.get("kind") == "exercise_required"]
    if req_gaps:
        suggestions.append("- Resolver ejercicios required fallidos antes de continuar: " + ", ".join(set(g.get("code","?") for g in req_gaps)))
    # Autonomy baja
    autonomy = (mv.get("dimensions") or {}).get("autonomy")
    if isinstance(autonomy, (int, float)) and autonomy < 0.5:
        suggestions.append("- Planear 1 ejercicio nuevo y 1 mejora de tool/observability para subir autonomía.")
    if not suggestions:
        suggestions.append("- Mantener rutina: Gym diario + report; sin bloqueos críticos detectados.")
    lines.extend(suggestions)
    lines.append("")
    lines.append("## ¿Debes intervenir?")
    icon = "✅" if intervention["level"] == "green" else ("⚠️" if intervention["level"] == "yellow" else "🛑")
    lines.append(f"{icon} Nivel: {intervention['level']} — {intervention['reason']}")
    if intervention["triggered_rules"]:
        lines.append(f"- Reglas activadas: {', '.join(intervention['triggered_rules'])}")
    if intervention["recommended_actions"]:
        lines.append("- Acciones recomendadas:")
        for act in intervention["recommended_actions"]:
            lines.append(f"  - {act}")
    else:
        lines.append("- Acciones recomendadas: ninguna")
    return "\n".join(lines)


def persist_report(report: str, heartbeat: Dict[str, Any], gym: Dict[str, Any]) -> None:
    reports_dir = ROOT / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "daily_report.md").write_text(report, encoding="utf-8")
    try:
        intervention = compute_intervention(time.time(), heartbeat, gym)
        (reports_dir / "intervention_status.json").write_text(
            json.dumps(intervention, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def main() -> int:
    heartbeat = _load_json(ROOT / "artifacts" / "health" / "ajax_heartbeat.json") or {}
    gym = _load_json(ROOT / "artifacts" / "exercises" / "gym_daily.json") or {}
    report = build_daily_report()
    print(report)
    persist_report(report, heartbeat, gym)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
