"""Protocolo de Autonomía Supervisada (PAS) para AJAX.

Este módulo materializa las Leyes y SOP de AGENTS.md sin inventar roles nuevos:
- CEO orquesta vía broker (Jobs JSON)
- Council sigue filtrando seguridad
- AJAX ejecuta misiones y produce evidencia

Entrada principal: run_agentic_session(agent_goal, constraints)
Devuelve un resumen estructurado (AgentResult) y persiste evidencia JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request

from agency.ajax_heartbeat import DEFAULT_OUT as HEARTBEAT_PATH
from agency.broker import AgencyBroker
from agency.contract import AgencyBudget, AgencyContext, AgencyJob, AgencyResult
from agency.driver_keys import load_ajax_driver_api_key
from agency.leann_context import get_leann_context
try:
    from agency.anti_optimism_guard import validate_output as _validate_output_bundle, get_guard as _get_guard
    from agency.types import OutputBundle
except Exception:  # pragma: no cover - opcional
    _validate_output_bundle = None  # type: ignore
    _get_guard = None  # type: ignore
    OutputBundle = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
PAS_DIR = ROOT / "artifacts" / "pas_sessions"
HEARTBEAT_PATH = Path(HEARTBEAT_PATH)
HEARTBEAT_SCRIPT = ROOT / "agency" / "ajax_heartbeat.py"
try:
    from agency.circuit_breaker import InfraBreaker, speak_instability_alert
except Exception:  # pragma: no cover - opcional
    InfraBreaker = None  # type: ignore

    def speak_instability_alert(kind: str) -> None:  # type: ignore
        return

INFRA_BREAKER = InfraBreaker() if InfraBreaker else None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clean_text(val: Any, limit: int = 320) -> str:
    if val is None:
        return ""
    text = str(val)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _maybe_parse_output_bundle(answer: str) -> Optional["OutputBundle"]:
    if OutputBundle is None:
        return None
    raw = (answer or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "claims" not in data and "hypothesis" not in data:
        return None
    try:
        return OutputBundle.from_dict(data)
    except Exception:
        return None


def _apply_anti_optimism_guard(answer: str, rail: str) -> Tuple[str, Optional[str], Optional[str]]:
    if _validate_output_bundle is None:
        return answer, None, None
    bundle = _maybe_parse_output_bundle(answer)
    if not bundle:
        return answer, None, None
    result = _validate_output_bundle(bundle, original_text=answer, rail=rail)
    if result.action == "DEGRADED_TO_HYPOTHESIS" and result.bundle is not None:
        return (
            json.dumps(result.bundle.to_dict(), ensure_ascii=False, indent=2),
            result.action,
            result.receipt_path,
        )
    if result.action == "SOFT_BLOCK":
        guard = _get_guard(rail) if _get_guard else None
        message = guard.format_soft_block(result) if guard else "SOFT_BLOCK"
        return message, result.action, result.receipt_path
    return answer, result.action, result.receipt_path


def _to_windows_path(path: Path) -> str:
    raw = str(path)
    if raw.startswith("/mnt/") and len(raw.split("/")) >= 4:
        parts = raw.split("/")
        drive = parts[2]
        rest = "\\".join(parts[3:])
        return f"{drive.upper()}:\\{rest}"
    return raw


def _to_windows_path(path: Path) -> str:
    raw = str(path)
    if raw.startswith("/mnt/") and len(raw.split("/")) >= 4:
        parts = raw.split("/")
        drive = parts[2]
        rest = "\\".join(parts[3:])
        return f"{drive.upper()}:\\{rest}"
    return raw


@contextmanager
def _env_override(**kwargs: Optional[str]) -> Any:
    old: Dict[str, Optional[str]] = {}
    for key, value in kwargs.items():
        old[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass
class AgentConstraints:
    max_missions: int = 3
    max_runtime_sec: int = 420
    max_consecutive_failures: int = 2

    @classmethod
    def from_any(cls, raw: Any) -> "AgentConstraints":
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            return cls(
                max_missions=int(raw.get("max_missions", 3) or 3),
                max_runtime_sec=int(raw.get("max_runtime_sec", 420) or 420),
                max_consecutive_failures=int(raw.get("max_consecutive_failures", 2) or 2),
            )
        return cls()

    def to_dict(self) -> Dict[str, int]:
        return {
            "max_missions": self.max_missions,
            "max_runtime_sec": self.max_runtime_sec,
            "max_consecutive_failures": self.max_consecutive_failures,
        }


@dataclass
class MissionReport:
    intent: str
    job_id: str
    ok: bool
    answer: str
    evidence: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class AgentResult:
    status: str
    final_state_summary: str
    missions_run: List[MissionReport]
    gaps_detected: List[str] = field(default_factory=list)
    next_suggestions: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    session_id: str = ""
    summary_path: Optional[str] = None
    heartbeat_gate: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "final_state_summary": self.final_state_summary,
            "missions_run": [m.to_dict() for m in self.missions_run],
            "gaps_detected": self.gaps_detected,
            "next_suggestions": self.next_suggestions,
            "elapsed_sec": self.elapsed_sec,
            "session_id": self.session_id,
            "summary_path": self.summary_path,
            "heartbeat_gate": self.heartbeat_gate,
        }


def _is_self_heal_goal(text: str) -> bool:
    lowered = text.lower()
    tokens = ("self heal", "self-heal", "latido", "heartbeat", "self_recover", "recupera", "verde")
    return any(tok in lowered for tok in tokens)


def _heartbeat_status() -> Dict[str, Any]:
    if HEARTBEAT_PATH.exists():
        data = _read_json(HEARTBEAT_PATH)
        if isinstance(data, dict):
            return data
    return {"status": "unknown", "problems": ["missing_heartbeat"]}


def _run_cmd(cmd: List[str], *, timeout: float = 120.0, text: bool = True) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=text, timeout=timeout)
        ok = proc.returncode == 0
        if text:
            out = (proc.stdout or "") + (proc.stderr or "")
        else:
            out_bytes = (proc.stdout or b"") + (proc.stderr or b"")
            try:
                out = out_bytes.decode("utf-8", errors="ignore")
            except Exception:
                out = ""
        return ok, out.strip()
    except Exception as exc:
        return False, f"exception:{exc}"


def _run_heartbeat_probe(session_id: str) -> MissionReport:
    ok, output = _run_cmd(["python", str(HEARTBEAT_SCRIPT)])
    hb = _heartbeat_status()
    status = str(hb.get("status", "unknown")).lower()
    artifacts = [str(HEARTBEAT_PATH)]
    meta = {"probe": True, "hb_status_after": status, "detail": _clean_text(output)}
    return MissionReport(
        intent="Generar latido base (agency/ajax_heartbeat.py)",
        job_id=f"{session_id}_bootstrap",
        ok=ok,
        answer=f"Latido tras probe: {status}",
        evidence=artifacts,
        artifacts=artifacts,
        errors=[] if ok else [output or "heartbeat_probe_failed"],
        meta=meta,
    )


def _detect_planner_failure(res: AgencyResult) -> Tuple[bool, str]:
    if res.ok:
        return False, ""
    lower_answer = res.answer.lower()
    no_artifacts = not res.artifacts and not res.actions
    errors = " | ".join(res.errors) if res.errors else ""
    if ("aborted at phase" in lower_answer or "planner" in lower_answer or not res.confidence) and no_artifacts:
        reason = errors or res.answer or "planner_no_plan"
        return True, _clean_text(reason, 200)
    if not res.metrics.tokens_in and not res.metrics.tokens_out and no_artifacts:
        return True, _clean_text(errors or "planner_no_tokens", 200)
    return False, ""


def _run_skeleton_self_heal(session_id: str, mission_index: int) -> MissionReport:
    # Skeleton: leer latido, intentar regenerarlo, volver a leer
    ok_probe, output = _run_cmd(["python", str(HEARTBEAT_SCRIPT)])
    hb_after = _heartbeat_status()
    status = str(hb_after.get("status", "unknown")).lower()
    artifacts = [str(HEARTBEAT_PATH)]
    answer = f"Skeleton PAS: latido tras intento: {status}"
    ok = ok_probe and status == "green"
    meta = {
        "fallback": "skeleton",
        "hb_status_after": status,
        "detail": _clean_text(output),
    }
    errors = [] if ok else [f"skeleton_result:{status}"]
    return MissionReport(
        intent="PAS skeleton: regenerar latido y verificar",
        job_id=f"{session_id}_m{mission_index}_skeleton",
        ok=ok,
        answer=answer,
        evidence=artifacts,
        artifacts=artifacts,
        errors=errors,
        meta=meta,
    )


def _probe_driver_health(timeout: float = 3.0) -> Tuple[bool, str]:
    try:
        headers = {}
        api_key = load_ajax_driver_api_key()
        if api_key:
            headers["X-AJAX-KEY"] = api_key
        req = request.Request("http://127.0.0.1:5010/health", method="GET", headers=headers)
        with request.urlopen(req, timeout=timeout) as resp:
            ok = 200 <= resp.status < 300
            return ok, f"http {resp.status}"
    except Exception as exc:
        return False, str(exc)


def _start_driver(actions: List[Dict[str, Any]], attempt: int) -> bool:
    ps1 = ROOT / "Start-AjaxDriver.ps1"
    started = False
    # Prefer PowerShell script if available
    if ps1.exists():
        ps1_win = _to_windows_path(ps1)
        ok, out = _run_cmd(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1_win],
            text=False,
            timeout=40,
        )
        actions.append({"step": f"driver_ps1_attempt_{attempt}", "cmd": ps1_win, "ok": ok, "detail": _clean_text(out)})
        started = started or ok
    # Fallback: python os_driver.py in background
    driver_cmd = ["python", str(ROOT / "drivers" / "os_driver.py"), "--host", "127.0.0.1", "--port", "5010"]
    try:
        proc = subprocess.Popen(driver_cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        actions.append({"step": f"driver_py_spawn_{attempt}", "cmd": " ".join(driver_cmd), "ok": True, "detail": f"pid={proc.pid}"})
        started = True
    except Exception as exc:
        actions.append({"step": f"driver_py_spawn_{attempt}", "cmd": " ".join(driver_cmd), "ok": False, "detail": str(exc)})
    # Probe health up to 15s
    for i in range(5):
        time.sleep(3)
        ok, detail = _probe_driver_health()
        actions.append({"step": f"driver_health_probe_{attempt}_{i+1}", "cmd": "GET 5010/health", "ok": ok, "detail": detail})
        if ok:
            return True
    return False


def _run_fix_infra(session_id: str, attempt: int) -> MissionReport:
    actions: List[Dict[str, Any]] = []

    def record(label: str, cmd: List[str], *, text: bool = True, timeout: float = 120.0) -> None:
        ok, out = _run_cmd(cmd, text=text, timeout=timeout)
        actions.append({"step": label, "cmd": " ".join(cmd), "ok": ok, "detail": _clean_text(out)})

    # Liberar puerto 5002
    record("kill_5002", ["bash", "-lc", "pids=$(lsof -i :5002 -t || true); [ -n \"$pids\" ] && kill -9 $pids || true"])

    # Web: restart y fallback a script
    record("web_restart", ["systemctl", "--user", "restart", "leann-web.service"])
    web_script = ROOT / "start_web_interface.sh"
    if web_script.exists():
        record("web_script", ["bash", str(web_script)])

    # RAG + QueryDrop
    record("rag_restart", ["systemctl", "--user", "restart", "leann-rag.service"])
    record("querydrop_restart", ["systemctl", "--user", "restart", "leann-querydrop.service"])

    # Driver
    driver_ok = _start_driver(actions, attempt)

    hb_before = _heartbeat_status()
    ok_probe, output = _run_cmd(["python", str(HEARTBEAT_SCRIPT)])
    hb_after = _heartbeat_status()
    status = str(hb_after.get("status", "unknown")).lower()

    artifacts = [str(HEARTBEAT_PATH)]
    meta = {
        "fix_infra_attempt": attempt,
        "actions": actions,
        "driver_ok": driver_ok,
        "hb_before": hb_before,
        "hb_after_status": status,
        "probe_output": _clean_text(output),
    }
    ok_total = driver_ok and ok_probe and status in {"green", "yellow"}
    errors = [] if ok_total else [f"fix_infra_status:{status}"]
    return MissionReport(
        intent=f"FIX_INFRA attempt {attempt} (libera 5002, web, rag, querydrop, driver, heartbeat)",
        job_id=f"{session_id}_fixinfra_{attempt}",
        ok=ok_total,
        answer=f"FIX_INFRA latido tras intento {attempt}: {status}",
        evidence=artifacts,
        artifacts=artifacts,
        errors=errors,
        meta=meta,
    )


def _rag_to_context(agent_goal: str) -> AgencyContext:
    rag = get_leann_context(agent_goal, mode="persona+system", timeout_seconds=1.5)
    rag_snippets = rag.get("rag_snippets") or []
    notes = rag.get("notes") or []
    rag_items: List[Dict[str, Any]] = []
    for item in rag_snippets:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content")
            source = item.get("source") or item.get("doc_id") or item.get("path")
            if text:
                rag_items.append({"text": str(text), "source": source})
        else:
            rag_items.append({"text": str(item), "source": None})
    note_items = [str(n) for n in notes if n]
    return AgencyContext(rag=rag_items, notes=note_items)


def _collect_gap_ids(result: AgencyResult) -> List[str]:
    gaps: List[str] = []
    for artifact in result.artifacts:
        if "capability_gap" in artifact.path or artifact.type == "capability_gap":
            gaps.append(Path(artifact.path).stem)
    for err in result.errors:
        if "capability_gap" in err:
            gaps.append(err)
    return gaps


def _plan_is_noop(job_id: str) -> bool:
    plan_path = ROOT / "runs" / job_id / "plan.json"
    if not plan_path.exists():
        return False
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        steps = data.get("plan") or []
        if not steps:
            return True
        return all((step.get("tool") or "noop").lower() == "noop" for step in steps if isinstance(step, dict))
    except Exception:
        return False


def _build_job(
    session_id: str,
    mission_index: int,
    mission_intent: str,
    agent_goal: str,
    constraints: AgentConstraints,
    bucket: str,
) -> AgencyJob:
    context = _rag_to_context(agent_goal)
    remaining_seconds = max(constraints.max_runtime_sec, 60)
    budget = AgencyBudget(steps=4, seconds=remaining_seconds, tokens=6000)
    metadata = {
        "bucket": bucket,
        "task_type": bucket,
        "risk": "maintenance" if bucket == "maintenance" else "normal",
        "pas_session_id": session_id,
        "mission_index": mission_index,
        "pas_goal": agent_goal,
        "planner_policy": "auto",
        "job_kind": "mission",
    }
    job_id = f"{session_id}_m{mission_index}"
    return AgencyJob(
        job_id=job_id,
        goal=mission_intent,
        context=context,
        budget=budget,
        metadata=metadata,
    )


def _persist_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit_capability_gap(session_id: str, hb: Dict[str, Any], reason: str, attempts: Optional[List[Dict[str, Any]]] = None) -> Path:
    gap_dir = ROOT / "artifacts" / "capability_gaps"
    gap_dir.mkdir(parents=True, exist_ok=True)
    gap_id = f"infra_self_heal_failed_{session_id}"
    payload = {
        "gap_id": gap_id,
        "capability_family": "infra.self_heal",
        "symptoms": hb.get("problems") or [],
        "heartbeat_status": hb,
        "reason": reason,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_paths": [
            str(HEARTBEAT_PATH),
            str(ROOT / "artifacts" / "health" / "ajax_heartbeat_history.jsonl"),
        ],
        "next_actions": [
            "Director: registrar en research_backlog.yaml",
            "Scout: buscar patrones de supervisión/auto-restart para web/driver",
        ],
    }
    if attempts:
        payload["fix_infra_attempts"] = attempts
    path = gap_dir / f"{gap_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_agentic_session(agent_goal: str, constraints: Optional[AgentConstraints] = None, *, session_id: Optional[str] = None,
                        persist_path: Optional[Path] = None) -> AgentResult:
    """Ejecuta el PAS para un goal de alto nivel."""
    if not agent_goal or not str(agent_goal).strip():
        raise ValueError("agent_goal vacío")

    constraints = constraints or AgentConstraints()
    session = session_id or f"pas_{uuid.uuid4().hex[:8]}"
    start = time.time()
    hb = _heartbeat_status()
    hb_status = str(hb.get("status", "unknown")).lower()
    is_self_heal = _is_self_heal_goal(agent_goal)
    infra_breaker = INFRA_BREAKER
    infra_blocked = False
    if infra_breaker:
        try:
            infra_blocked = infra_breaker.infra_should_block(time.time())
        except Exception:
            infra_blocked = False

    missions: List[MissionReport] = []
    gaps_detected: List[str] = []
    next_suggestions: List[str] = []
    consecutive_failures = 0
    status = "partial"
    final_summary = ""
    fix_infra_attempts = 0
    summary_path = persist_path or (PAS_DIR / f"{session}.json")

    if infra_blocked:
        final_summary = "Infra breaker activo; autonomía pausada."
        status = "aborted_by_gate"
        speak_instability_alert("infra")
        result = AgentResult(
            status=status,
            final_state_summary=final_summary,
            missions_run=[],
            gaps_detected=[],
            next_suggestions=["Revisar infraestructura manualmente antes de relanzar PAS."],
            elapsed_sec=time.time() - start,
            session_id=session,
            summary_path=str(summary_path),
            heartbeat_gate=hb,
        )
        _persist_summary(summary_path, result.to_dict())
        return result

    gate_block = hb_status != "green" and not is_self_heal

    if gate_block:
        status = "aborted_by_gate"
        final_summary = f"Heartbeat en estado {hb_status}; PAS bloqueado salvo self-heal."
        result = AgentResult(
            status=status,
            final_state_summary=final_summary,
            missions_run=[],
            gaps_detected=[],
            next_suggestions=["Lanza el goal self_heal_heartbeat para desbloquear."] if hb_status != "green" else [],
            elapsed_sec=time.time() - start,
            session_id=session,
            summary_path=str(summary_path),
            heartbeat_gate=hb,
        )
        if infra_breaker:
            try:
                infra_breaker.infra_register_failure(kind=hb_status or "heartbeat", now=time.time(), meta={"source": "pas_gate"})
            except Exception:
                pass
        _persist_summary(summary_path, result.to_dict())
        return result

    bucket = "maintenance" if is_self_heal else "general"

    def _run_job_with_hint(job_obj: AgencyJob, planner_hint: Optional[str] = None) -> AgencyResult:
        if not planner_hint:
            broker_local = AgencyBroker()
            return broker_local.run_job(job_obj)
        hint_env = {}
        if planner_hint == "groq":
            hint_env = {
                "AGENCY_PLANNER_CMD": "bin/groq_task.py --role planner --json",
                "PLANNER_MODE": "cloud-first",
            }
        with _env_override(**hint_env):
            broker_local = AgencyBroker()
            return broker_local.run_job(job_obj)

    # Si no hay latido, crea misión de probe para no fallar en PLAN
    if is_self_heal and ("missing_heartbeat" in (hb.get("problems") or []) or hb_status == "unknown"):
        probe_mission = _run_heartbeat_probe(session)
        missions.append(probe_mission)
        hb = _heartbeat_status()
        hb_status = str(hb.get("status", "unknown")).lower()
        if hb_status == "green":
            status = "success"
            final_summary = "Heartbeat generado y en verde tras probe inicial."
            elapsed_total = time.time() - start
            result = AgentResult(
                status=status,
                final_state_summary=final_summary,
                missions_run=missions,
                gaps_detected=[],
                next_suggestions=[],
                elapsed_sec=elapsed_total,
                session_id=session,
                summary_path=str(summary_path),
                heartbeat_gate=hb,
        )
        _persist_summary(summary_path, result.to_dict())
        return result

    # Si heartbeat está en rojo por infra, intenta FIX_INFRA antes de misiones planificadas
    if is_self_heal and hb_status == "red":
        fix_infra_attempts += 1
        fix_report = _run_fix_infra(session, fix_infra_attempts)
        missions.append(fix_report)
        hb = _heartbeat_status()
        hb_status = str(hb.get("status", "unknown")).lower()
        if infra_breaker and hb_status != "green":
            try:
                infra_breaker.infra_register_failure(kind="self_heal_bootstrap", now=time.time(), meta={"hb_status": hb_status})
            except Exception:
                pass
        if hb_status == "green":
            status = "success"
            final_summary = "Heartbeat recuperado a verde tras FIX_INFRA."
            elapsed_total = time.time() - start
            result = AgentResult(
                status=status,
                final_state_summary=final_summary,
                missions_run=missions,
                gaps_detected=[],
                next_suggestions=[],
                elapsed_sec=elapsed_total,
                session_id=session,
                summary_path=str(summary_path),
                heartbeat_gate=hb,
            )
            _persist_summary(summary_path, result.to_dict())
            return result

    for mission_index in range(1, constraints.max_missions + 1):
        elapsed = time.time() - start
        if elapsed >= constraints.max_runtime_sec:
            final_summary = f"Límite de tiempo alcanzado ({constraints.max_runtime_sec}s)."
            status = "partial" if missions else "failed"
            break

        # Evaluar éxito antes de lanzar siguiente misión (Ley de la Intención)
        if is_self_heal:
            hb = _heartbeat_status()
            if str(hb.get("status", "")).lower() == "green":
                status = "success"
                final_summary = "Heartbeat ya en verde; no se requieren más misiones."
                break
            if hb_status == "red" and fix_infra_attempts < 3:
                fix_infra_attempts += 1
                fix_report = _run_fix_infra(session, fix_infra_attempts)
                missions.append(fix_report)
                hb = _heartbeat_status()
                hb_status = str(hb.get("status", "unknown")).lower()
                if infra_breaker and hb_status != "green":
                    try:
                        infra_breaker.infra_register_failure(kind="self_heal_attempt", now=time.time(), meta={"hb_status": hb_status, "attempt": fix_infra_attempts})
                    except Exception:
                        pass
                if hb_status == "green":
                    status = "success"
                    final_summary = "Heartbeat recuperado a verde tras FIX_INFRA."
                    break

        problems = hb.get("problems") or []
        mission_intent = agent_goal
        if is_self_heal:
            mission_intent = (
                f"[PAS self-heal] Recupera el heartbeat a verde. Estado actual: {hb_status}. "
                f"Problemas: {', '.join(problems) if problems else 'sin detalle'}."
            )

        job = _build_job(session, mission_index, mission_intent, agent_goal, constraints, bucket)
        agency_res: AgencyResult = _run_job_with_hint(job)
        mission_artifacts = [artifact.path for artifact in agency_res.artifacts]
        mission_meta: Dict[str, Any] = {"confidence": agency_res.confidence}
        planner_failed, planner_reason = _detect_planner_failure(agency_res)
        if not planner_failed and _plan_is_noop(job.job_id):
            planner_failed = True
            planner_reason = "noop_plan"
        if planner_failed:
            mission_meta["planner_failure"] = planner_reason
            # Reintento con planner alternativo (groq)
            agency_res = _run_job_with_hint(job, planner_hint="groq")
            mission_artifacts = [artifact.path for artifact in agency_res.artifacts]
            mission_meta["fallback_planner"] = "groq"
            if not agency_res.ok:
                mission_meta["fallback_planner_result"] = _clean_text(agency_res.answer)
        if planner_failed and not agency_res.ok:
            # Skeleton defensivo si sigue sin plan útil
            skeleton = _run_skeleton_self_heal(session, mission_index)
            missions.append(skeleton)
            agency_res = AgencyResult(
                ok=skeleton.ok,
                answer=skeleton.answer,
                actions=[],
                artifacts=[],
                metrics=agency_res.metrics,
                confidence=agency_res.confidence,
                errors=skeleton.errors,
            )
            mission_artifacts = skeleton.artifacts
            mission_meta["skeleton_used"] = True

        rail = str(os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or "prod").strip().lower()
        guarded_answer, guard_action, guard_receipt = _apply_anti_optimism_guard(
            str(agency_res.answer or ""), rail=rail
        )
        if guard_action:
            mission_meta["anti_optimism_action"] = guard_action
            if guard_receipt:
                mission_meta["anti_optimism_receipt"] = guard_receipt
                mission_artifacts.append(guard_receipt)
        agency_ok = bool(agency_res.ok)
        errors_list = list(agency_res.errors or [])
        if guard_action == "SOFT_BLOCK":
            agency_ok = False
            if "anti_optimism_soft_block" not in errors_list:
                errors_list.append("anti_optimism_soft_block")

        mission = MissionReport(
            intent=mission_intent,
            job_id=job.job_id,
            ok=agency_ok,
            answer=_clean_text(guarded_answer),
            evidence=mission_artifacts,
            artifacts=mission_artifacts,
            errors=errors_list,
            meta=mission_meta,
        )
        missions.append(mission)
        gaps_detected.extend(_collect_gap_ids(agency_res))

        if agency_res.ok:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if infra_breaker and is_self_heal:
                try:
                    infra_breaker.infra_register_failure(kind="self_heal_mission", now=time.time(), meta={"errors": agency_res.errors})
                except Exception:
                    pass
            if planner_failed:
                next_suggestions.append(f"planner_failure:{planner_reason}")
            if consecutive_failures >= constraints.max_consecutive_failures:
                status = "failed"
                final_summary = (
                    f"{consecutive_failures} fallos consecutivos. Último error: "
                    f"{_clean_text(agency_res.errors[0]) if agency_res.errors else 'sin detalle'}"
                )
                break

        # Verificación de éxito tras la misión
        if is_self_heal:
            hb_post = _heartbeat_status()
            hb_status = str(hb_post.get("status", "unknown")).lower()
            if hb_status == "green":
                status = "success"
                final_summary = "Heartbeat recuperado (verde) tras misión PAS."
                hb = hb_post
                break
        else:
            if agency_res.ok:
                status = "success"
                final_summary = "Goal cumplido según misión más reciente."
                break

    else:
        status = status or "partial"
        final_summary = final_summary or "Límite de misiones alcanzado."

    elapsed_total = time.time() - start
    if status != "success" and not final_summary:
        planner_flag = any(m.meta.get("planner_failure") for m in missions)
        if planner_flag:
            final_summary = "No se alcanzó éxito: planner sin plan usable (fallback aplicado)."
        else:
            final_summary = "No se alcanzó éxito antes de agotar límites."
    if status != "success":
        next_suggestions.append("Revisar artifacts de la última misión en runs/ y ajustar constraints/goal.")
    gap_path: Optional[Path] = None
    if status != "success" and is_self_heal:
        attempts_summary = [m.meta for m in missions if m.meta.get("fix_infra_attempt")]
        gap_path = _emit_capability_gap(session, hb, reason=final_summary, attempts=attempts_summary)
        gaps_detected.append(gap_path.stem)
        if infra_breaker:
            try:
                infra_breaker.infra_register_failure(kind="self_heal_failed", now=time.time(), meta={"reason": final_summary})
            except Exception:
                pass

    result = AgentResult(
        status=status,
        final_state_summary=final_summary,
        missions_run=missions,
        gaps_detected=sorted(set(gaps_detected)),
        next_suggestions=next_suggestions,
        elapsed_sec=elapsed_total,
        session_id=session,
        summary_path=str(summary_path),
        heartbeat_gate=hb,
    )
    _persist_summary(summary_path, result.to_dict())
    return result


__all__ = [
    "AgentConstraints",
    "AgentResult",
    "MissionReport",
    "run_agentic_session",
]
