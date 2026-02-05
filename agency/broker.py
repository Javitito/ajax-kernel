"""Minimal broker orchestrating the LEANN agency workflow.

This is an MVP implementation: it wires the contract, router, council, planner,
executors, and verifier while keeping the logic pluggable. The broker prefers
real CLIs (Gemini/Qwen/Codex) but gracefully falls back to stub behaviours so
it remains useful during local development.
"""

from __future__ import annotations

import copy
import json
import os
import random
import select
import shlex
import signal
import subprocess
import sys
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import urllib.request
import importlib.util
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Ensure script execution finds agency/*
    sys.path.insert(0, str(ROOT))

from agency.plan_runner import run_job_plan


if __package__ is None:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from agency.council import is_action_allowed_in_degraded, load_council_state
except Exception:  # pragma: no cover - fallback

    def load_council_state(path=None):  # type: ignore
        return {"mode": "normal"}

    def is_action_allowed_in_degraded(action=None, args=None):  # type: ignore
        return False


try:
    from agency.circuit_breaker import InfraBreaker, speak_instability_alert
except Exception:  # pragma: no cover - fallback
    InfraBreaker = None  # type: ignore

    def speak_instability_alert(kind: str) -> None:  # type: ignore
        return


try:
    from agency.infraguard import (
        evaluate_plan_actions,
        write_audit_log,
        classify_action,
        ActionAudit,
    )
except Exception:  # pragma: no cover

    def evaluate_plan_actions(plan_actions, council_state, infra_blocked):  # type: ignore
        return True, "", []

    def write_audit_log(path, **kwargs):  # type: ignore
        return

    def classify_action(action):  # type: ignore
        return "moderate", "default"

    class ActionAudit:  # type: ignore
        def __init__(self, name, args, classification, allowed, reason=None):
            self.name = name
            self.args = args
            self.classification = classification
            self.allowed = allowed
            self.reason = reason

        def to_dict(self):
            return {
                "name": self.name,
                "args": self.args,
                "classification": self.classification,
                "allowed": self.allowed,
                "reason": self.reason,
            }


try:  # Allow running as module or script
    from .contract import (
        AgencyJob,
        AgencyResult,
        AgentArtifact,
        AgentMetrics,
        ToolCall,
    )
    from .policy import load_policy, write_policy
    from .registry import RegistrySnapshot, RegistrySpec, collect_registry, write_registry
    from .runs import RunPaths
except ImportError:  # pragma: no cover - fallback for "python agency/broker.py"
    from agency.contract import (  # type: ignore
        AgencyJob,
        AgencyResult,
        AgentArtifact,
        AgentMetrics,
        ToolCall,
    )
    from agency.policy import load_policy, write_policy  # type: ignore
    from agency.registry import RegistrySnapshot, RegistrySpec, collect_registry, write_registry  # type: ignore
    from agency.runs import RunPaths  # type: ignore


JsonDict = Dict[str, Any]
BACKGROUND_MODE_ACTIVE = False

try:
    from agency.anti_optimism_guard import validate_output as _validate_output_bundle, get_guard
    from agency.types import OutputBundle
except Exception:  # pragma: no cover - optional dependency
    _validate_output_bundle = None  # type: ignore
    get_guard = None  # type: ignore
    OutputBundle = None  # type: ignore


def _set_background_mode(active: bool) -> None:
    global BACKGROUND_MODE_ACTIVE
    BACKGROUND_MODE_ACTIVE = active


try:
    from tools.verifier import verify_run, verify_player_state
except Exception:  # pragma: no cover - optional dependency
    verify_run = None  # type: ignore
    verify_player_state = None  # type: ignore

try:
    from tools.browser import BrowserController
except Exception:  # pragma: no cover - optional dependency
    BrowserController = None  # type: ignore

try:
    from agency.executor import execute as local_executor_execute
except Exception:  # pragma: no cover - fallback
    local_executor_execute = None  # type: ignore


@dataclass
class ExecutionState:
    phase: str = "PLAN"
    step_index: int = 0
    active_role: Optional[str] = None
    retries: Dict[str, int] = field(
        default_factory=lambda: {"planner": 0, "executor": 0, "verifier": 0}
    )
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "step_index": self.step_index,
            "active_role": self.active_role,
            "retries": self.retries,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionState":
        return cls(
            phase=str(data.get("phase", "PLAN")),
            step_index=int(data.get("step_index", 0)),
            active_role=data.get("active_role"),
            retries={k: int(v) for k, v in (data.get("retries") or {}).items()},
            last_error=data.get("last_error"),
        )


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Terminate process and its children."""

    if proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover - windows path
            proc.send_signal(signal.SIGBREAK)
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                text=True,
            )
    except Exception:
        pass
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def _ensure_sequence_backoff(values: Iterable[Any]) -> List[float]:
    seq: List[float] = []
    for value in values:
        try:
            seq.append(float(value))
        except (TypeError, ValueError):
            continue
    return seq or [2.0, 5.0, 10.0]


@dataclass
class LMStudioHealthResult:
    """Resultado detallado del health check de LM Studio."""

    server_up: bool = False
    model_ready: bool = False
    status_code: Optional[int] = None
    error: Optional[str] = None
    response_body: Optional[str] = None


def _lmstudio_health_detailed(
    url: Optional[str] = None, timeout: float = 1.5
) -> LMStudioHealthResult:
    """
    Verifica el estado de LM Studio distinguiendo entre:
    - server_up: El servidor responde HTTP (cualquier código)
    - model_ready: El servidor responde 200 y tiene modelos disponibles

    Returns:
        LMStudioHealthResult con el estado detallado
    """
    result = LMStudioHealthResult()
    target = url or os.getenv("LMSTUDIO_HEALTH_URL") or "http://127.0.0.1:1235/models"

    try:
        req = urllib.request.Request(target, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result.status_code = getattr(resp, "status", 200)
            result.server_up = True

            # Leer respuesta para verificar modelos disponibles
            try:
                body = resp.read().decode("utf-8", errors="ignore")
                result.response_body = body[:1000]  # Limitar tamaño

                # Verificar si hay modelos cargados
                if result.status_code == 200:
                    try:
                        data = json.loads(body)
                        if isinstance(data, dict) and "data" in data:
                            models = data.get("data", [])
                            result.model_ready = len(models) > 0
                        elif isinstance(data, list):
                            result.model_ready = len(data) > 0
                        else:
                            result.model_ready = True  # Asumir OK si responde 200
                    except json.JSONDecodeError:
                        # Si no es JSON válido pero responde 200, asumir OK
                        result.model_ready = True
                else:
                    # Servidor up pero modelo no listo (404, 503, etc.)
                    result.model_ready = False

            except Exception as e:
                result.error = f"Error leyendo respuesta: {e}"
                result.model_ready = False

    except urllib.error.HTTPError as e:
        # Servidor responde pero con error HTTP
        result.server_up = True
        result.status_code = e.code
        result.model_ready = False
        result.error = f"HTTP {e.code}: {e.reason}"

        # 404 específicamente indica que el endpoint no existe o modelo no cargado
        if e.code == 404:
            result.error = "Modelo no cargado o endpoint /models no disponible (404)"

    except Exception as e:
        # Servidor no responde en absoluto
        result.server_up = False
        result.model_ready = False
        result.error = str(e)

    return result


def _lmstudio_healthy(url: Optional[str] = None, timeout: float = 1.5) -> bool:
    """
    Verificación simple de salud (backwards compatible).
    Considera "healthy" si el servidor responde 200-299.
    """
    result = _lmstudio_health_detailed(url, timeout)
    return result.server_up and result.status_code is not None and 200 <= result.status_code < 300


def _wrap_background_command(cmd: List[str]) -> List[str]:
    if not BACKGROUND_MODE_ACTIVE or not cmd:
        return cmd
    first = cmd[0]
    if first in {"ionice", "nice"}:
        return cmd
    return ["ionice", "-c2", "-n7", "nice", "-n", "10", *cmd]


def _guard_degraded_plan(plan_data: JsonDict) -> Optional[Dict[str, Any]]:
    try:
        state = load_council_state()
    except Exception:
        return None
    mode = str(state.get("mode") or "normal").lower()
    if mode != "degraded":
        return None
    raw_actions = []
    if isinstance(plan_data, dict):
        raw_actions = plan_data.get("plan") or plan_data.get("actions") or []
    blocked = []
    actions: List[ToolCall] = []
    for item in raw_actions:
        try:
            actions.append(ToolCall.from_any(item))
        except Exception:
            blocked.append({"index": len(actions), "action": None, "reason": "invalid_action"})
    for idx, action in enumerate(actions):
        if not is_action_allowed_in_degraded(action.tool, action.args):
            blocked.append({"index": idx, "action": action.tool})
    if blocked:
        return {"state": state, "blocked": blocked}
    return None


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


def _apply_anti_optimism_guard(
    answer: str, *, rail: str
) -> Tuple[str, Optional[str], Optional[str]]:
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
        guard = get_guard(rail) if get_guard else None
        message = guard.format_soft_block(result) if guard else "SOFT_BLOCK"
        return message, result.action, result.receipt_path
    return answer, result.action, result.receipt_path


def supervise_call(
    command: Optional[List[str] | str],
    role: str,
    payload: JsonDict,
    supervision: Dict[str, Any],
    fallback_command: Optional[str] = None,
) -> Tuple[bool, JsonDict, str, str, int, float, bool, Optional[str]]:
    """Run CLI with watchdog behaviour returning parsed JSON result."""

    if not command:
        return (
            False,
            {"ok": False, "error": f"No command configured for {role}"},
            "",
            "",
            0,
            0.0,
            False,
            None,
        )

    timeouts = supervision.get("timeouts_s", {})
    timeout_s = float(timeouts.get(role, timeouts.get("tool", 60)))
    heartbeat_s = float(supervision.get("heartbeat_s", 6))
    retries_cfg = supervision.get("retries", {})
    retries = int(retries_cfg.get(role, 0))
    backoff_seq = _ensure_sequence_backoff(supervision.get("backoff_s", [2, 5, 10]))

    commands_to_try: List[Tuple[List[str], bool]] = []
    base_primary = command if isinstance(command, list) else shlex.split(str(command))
    primary_cmd = _wrap_background_command(list(base_primary))
    commands_to_try.append((primary_cmd, False))
    if fallback_command:
        fallback_list = shlex.split(fallback_command)
        commands_to_try.append((_wrap_background_command(fallback_list), True))

    last_result: JsonDict = {"ok": False, "error": "unknown"}
    last_stdout = ""
    last_stderr = ""
    attempts_used = 0
    last_duration_ms = 0.0
    last_timed_out = False
    last_command_repr: Optional[str] = None

    for cmd_args, is_fallback in commands_to_try:
        attempts = 1 if is_fallback else retries + 1
        for attempt in range(attempts):
            attempts_used += 1
            stdout_buffer: List[str] = []
            stderr_buffer: List[str] = []
            command_repr = " ".join(cmd_args)
            last_command_repr = command_repr

            popen_kwargs: Dict[str, Any] = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "bufsize": 1,
            }
            if sys.platform != "win32":
                popen_kwargs["preexec_fn"] = os.setsid
            else:  # pragma: no cover
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            try:
                proc = subprocess.Popen(cmd_args, **popen_kwargs)
            except FileNotFoundError as exc:
                last_result = {"ok": False, "error": f"command_not_found: {exc}"}
                last_timed_out = False
                break

            attempt_start = time.time()
            timed_out = False
            try:
                proc.stdin.write(json.dumps(payload))  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
            except Exception:
                pass
            finally:
                if proc.stdin:
                    proc.stdin.close()

            start = time.time()
            hb_deadline = start + heartbeat_s
            overall_deadline = start + timeout_s

            while True:
                if proc.poll() is not None:
                    break
                try:
                    ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.5)  # type: ignore[arg-type]
                except Exception:
                    ready = []
                for stream in ready:
                    if stream is None:
                        continue
                    try:
                        chunk = stream.readline()
                    except Exception:
                        chunk = ""
                    if not chunk:
                        continue
                    if stream is proc.stdout:
                        stdout_buffer.append(chunk)
                    else:
                        stderr_buffer.append(chunk)
                        if '"heartbeat"' in chunk:
                            hb_deadline = time.time() + heartbeat_s

                now = time.time()
                if now > overall_deadline or now > hb_deadline:
                    kill_process_tree(proc)
                    stderr_buffer.append("[broker] timeout or heartbeat missed\n")
                    timed_out = True
                    break

            try:
                proc.wait(timeout=1)
            except Exception:
                kill_process_tree(proc)

            duration_ms = (time.time() - attempt_start) * 1000
            last_duration_ms = duration_ms
            last_timed_out = timed_out

            stdout_text = "".join(stdout_buffer)
            stderr_text = "".join(stderr_buffer)
            last_stdout = stdout_text
            last_stderr = stderr_text

            if proc.returncode == 0 and stdout_text.strip():
                try:
                    parsed = json.loads(stdout_text)
                    return (
                        True,
                        parsed,
                        stdout_text,
                        stderr_text,
                        attempts_used,
                        duration_ms,
                        timed_out,
                        command_repr,
                    )
                except json.JSONDecodeError as exc:
                    last_result = {"ok": False, "error": f"invalid_json: {exc}"}
            else:
                exit_reason = proc.returncode
                last_result = {
                    "ok": False,
                    "error": f"exit_code_{exit_reason}",
                    "stderr": stderr_text[-400:],
                }

            if attempt < attempts - 1 and not is_fallback:
                backoff = backoff_seq[min(attempt, len(backoff_seq) - 1)] + random.uniform(0, 1)
                time.sleep(backoff)

        if last_result.get("ok"):
            break

    return (
        False,
        last_result,
        last_stdout,
        last_stderr,
        attempts_used,
        last_duration_ms,
        last_timed_out,
        last_command_repr,
    )


def _write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_line(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _compact_file(path: Path, max_lines: int = 200) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_lines:
        return
    keep = lines[: max_lines // 2] + ["… (compacted) …"] + lines[-max_lines // 2 :]
    path.write_text("\n".join(keep) + "\n", encoding="utf-8")


@dataclass
class RouterDecision:
    use_council: bool
    reason: str

    def to_dict(self) -> JsonDict:
        return {"use_council": self.use_council, "reason": self.reason}


class Router:
    """Very small heuristic router deciding whether to call the council."""

    def decide(self, job: AgencyJob) -> RouterDecision:
        goal = job.goal.lower()
        requires = job.metadata.get("requires_council")
        if isinstance(requires, bool):
            return RouterDecision(bool(requires), "flag-from-metadata")

        long_goal = len(goal) > 200
        has_questions = goal.count("?") >= 2
        high_risk = "riesgo" in goal or "seguro" in goal
        ambiguous = any(token in goal for token in ("tal vez", "quizá", "ambigu"))
        use_council = long_goal or has_questions or high_risk or ambiguous
        reason = (
            "long-goal"
            if long_goal
            else "multi-question"
            if has_questions
            else "risk"
            if high_risk
            else "heuristic"
        )
        return RouterDecision(use_council=use_council, reason=reason)


class AgentInvoker:
    """Wraps a CLI agent or a Python callable."""

    def __init__(
        self,
        name: str,
        command: Optional[List[str]] = None,
        fallback: Optional[Callable[[JsonDict], JsonDict]] = None,
        timeout: int = 60,
    ) -> None:
        self.name = name
        self.command = command
        self.fallback = fallback or self._default_fallback
        self.timeout = timeout

    def invoke(self, payload: JsonDict, log_path: Path) -> JsonDict:
        if self.command:
            try:
                completed = subprocess.run(
                    self.command,
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    timeout=self.timeout,
                    check=False,
                )
                log_path.write_text(
                    f"COMMAND: {' '.join(self.command)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n",
                    encoding="utf-8",
                )
                if completed.stdout:
                    return json.loads(completed.stdout)
            except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
                pass
        response = self.fallback(payload)
        log_path.write_text(
            "FALLBACK RESPONSE:\n" + json.dumps(response, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return response

    def invoke_supervised(
        self,
        payload: JsonDict,
        log_path: Path,
        supervision: Dict[str, Any],
        role: str,
        override_command: Optional[str] = None,
        fallback_command: Optional[str] = None,
    ) -> Tuple[bool, JsonDict, Dict[str, Any]]:
        command = None
        if override_command:
            if isinstance(override_command, dict):
                override_command = override_command.get("cmd")

            if override_command and isinstance(override_command, str):
                command = shlex.split(override_command)
        elif self.command:
            command = self.command

        if command:
            (
                success,
                result,
                stdout_text,
                stderr_text,
                attempts,
                duration_ms,
                timed_out,
                command_used,
            ) = supervise_call(
                command,
                role,
                payload,
                supervision,
                fallback_command=fallback_command,
            )
            log_payload = {
                "command": command_used or " ".join(command),
                "stdout": stdout_text,
                "stderr": stderr_text,
                "attempts": attempts,
                "duration_ms": duration_ms,
                "timed_out": timed_out,
            }
            log_path.write_text(
                json.dumps(log_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            if success:
                meta = {
                    "attempts": attempts,
                    "duration_ms": duration_ms,
                    "timed_out": timed_out,
                    "command": command_used or " ".join(command),
                }
                return True, result, meta
        response = self.fallback(payload)
        log_path.write_text(
            "FALLBACK RESPONSE:\n" + json.dumps(response, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        fallback_command_repr = override_command or (
            " ".join(self.command) if self.command else fallback_command
        )
        meta = {
            "attempts": 0,
            "duration_ms": 0.0,
            "timed_out": False,
            "command": fallback_command_repr or "internal-fallback",
        }
        return True, response, meta

    def _default_fallback(self, payload: JsonDict) -> JsonDict:
        return {
            "summary": f"Stubbed response from {self.name}",
            "suggestions": ["Recopila estado", "Ejecuta acción principal", "Valida resultado"],
        }


def _load_cli_command(env_var: str, default: Optional[str]) -> Optional[List[str]]:
    value = os.getenv(env_var, default)
    if not value:
        return None
    return value.split()


def _default_brief_fallback(payload: JsonDict) -> JsonDict:
    goal = payload.get("goal", "")
    return {
        "questions": ["¿Hay dependencias externas?", "¿Necesitamos confirmación humana?"],
        "risks": ["Datos insuficientes"] if len(goal) > 120 else [],
        "steps": [
            "Revisar contexto RAG",
            "Clarificar puntos críticos",
            "Definir plan de 3-4 pasos",
        ],
        "brief": f"Resumen generado automáticamente para: {goal[:80]}",
    }


def _default_plan_fallback(payload: JsonDict) -> JsonDict:
    goal = payload.get("goal", "")
    capabilities = payload.get("capabilities", [])
    metadata = payload.get("metadata") or {}
    notes: List[str] = []
    steps: List[Dict[str, Any]] = []

    target_url = metadata.get("target_url") or metadata.get("url")
    volume = metadata.get("volume", 0.4)
    if target_url:
        steps.append(
            {
                "id": "step-1",
                "tool": "browser.play",
                "args": {
                    "url": target_url,
                    "volume": volume,
                    "title": metadata.get("title") or goal[:60],
                },
            }
        )
        notes.append("Plan heurístico: ejecutar browser.play con URL proporcionada")

    handled = {step.get("tool") for step in steps}
    for capability in capabilities[:5]:
        if capability in handled:
            continue
        steps.append(
            {
                "id": f"step-{len(steps) + 1}",
                "tool": capability,
                "args": {"note": f"Usar {capability} orientado al objetivo"},
            }
        )

    if not steps:
        steps.append({"id": "step-1", "tool": "noop", "args": {"note": "Analizar manualmente"}})

    notes.append(f"Plan generado automáticamente para: {goal[:100]}")
    return {
        "plan": steps,
        "notes": notes,
        "metadata": metadata,
    }


# -----------------------------
# Single-agent skills runner
# -----------------------------
def run_skill(skill: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load skills/<skill>.py and invoke main(request) returning a normalized payload.

    Also records artifacts/skills_last_run.json and updates artifacts/skills_metrics.json
    with minimal telemetry summary.
    """
    args = args or {}
    # Guardrails: nombre de skill seguro (sin rutas)
    if not isinstance(skill, str) or not re.match(r"^[a-zA-Z0-9_]{1,64}$", skill):
        return {"ok": False, "error": "invalid_skill_name"}

    base_dir = Path.cwd()
    skills_dir = base_dir / "skills"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    path = (skills_dir / f"{skill}.py").resolve()
    try:
        skills_root = skills_dir.resolve(strict=False)
    except Exception:
        skills_root = skills_dir
    # El path debe quedar dentro de skills/
    if not str(path).startswith(str(skills_root)) or not path.exists():
        return {"ok": False, "error": f"skill_not_found: {skill}"}

    spec = importlib.util.spec_from_file_location(f"skills.{skill}", str(path))
    if spec is None or spec.loader is None:
        return {"ok": False, "error": "skill_import_error"}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": f"skill_exec_error: {exc}"}

    if not hasattr(module, "main"):
        return {"ok": False, "error": "skill_missing_main"}

    request_payload = {"args": args}
    try:
        result = module.main(request_payload)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        result = {"ok": False, "error": f"skill_runtime_error: {exc}"}

    # Normalize and persist artifacts
    ok = bool(result.get("ok"))
    telemetry = result.get("telemetry") or {}
    data = result.get("data") or {}
    last_run = {"skill": skill, "ok": ok, "telemetry": telemetry, "data": data}
    _write_json(artifacts_dir / "skills_last_run.json", last_run)

    # Update metrics
    metrics_path = artifacts_dir / "skills_metrics.json"
    try:
        current = (
            json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        )
    except Exception:
        current = {}
    by_skill = current.get("by_skill") or {}
    sk = by_skill.get(skill) or {}
    # Minimal rollup (overwrite with last; real system could aggregate windows)
    sk.update(
        {
            "p95_ms": telemetry.get("p95_ms", sk.get("p95_ms", 0)),
            "ok_ratio": 1.0 if ok else 0.0,
            "cost_eur": telemetry.get("cost_eur", sk.get("cost_eur", 0.0)),
            "energy_wh": telemetry.get("energy_wh", sk.get("energy_wh", 0.0)),
        }
    )
    by_skill[skill] = sk
    # Recompute summary (simple):
    p95_list = [float(v.get("p95_ms", 0)) for v in by_skill.values()] or [0.0]
    cost_sum = sum(float(v.get("cost_eur", 0.0)) for v in by_skill.values())
    energy_sum = sum(float(v.get("energy_wh", 0.0)) for v in by_skill.values())
    current["by_skill"] = by_skill
    current["summary"] = {
        "p95_ms": max(p95_list),
        "cost_eur": round(cost_sum, 6),
        "energy_wh": round(energy_sum, 6),
    }
    _write_json(metrics_path, current)

    return {"ok": ok, "skill": skill, "telemetry": telemetry, "data": data}


def _default_executor_fallback(payload: JsonDict) -> JsonDict:
    if local_executor_execute is not None:
        try:
            job_payload = {
                "id": payload.get("job_id", "run"),
                "plan": payload.get("plan", []),
            }
            return local_executor_execute(job_payload)
        except Exception:
            pass

    actions = []
    for step in payload.get("plan", []):
        actions.append(
            {
                "tool": step.get("tool", "noop"),
                "args": step.get("args", {"note": step.get("action", "")}),
            }
        )
    return {"actions": actions, "artifacts": []}


def _default_verifier_fallback(payload: JsonDict) -> JsonDict:
    return {
        "ok": True,
        "confidence": 0.6,
        "notes": ["Verificación simulada"],
    }


class Council:
    def __init__(self, advisors: Iterable[AgentInvoker]) -> None:
        self.advisors = list(advisors)

    def gather_brief(self, job: AgencyJob, run_dir: RunPaths) -> JsonDict:
        payload = {
            "goal": job.goal,
            "context": job.context.to_dict(),
            "budget": job.budget.to_dict(),
        }
        aggregated: Dict[str, Any] = {"brief": "", "questions": [], "steps": [], "risks": []}

        for idx, advisor in enumerate(self.advisors, start=1):
            log_path = run_dir.logs_dir / f"council_{advisor.name}_{idx}.log"
            response = advisor.invoke({"mode": "briefing", **payload}, log_path)
            aggregated["questions"].extend(response.get("questions", []))
            aggregated["steps"].extend(response.get("steps", []))
            aggregated["risks"].extend(response.get("risks", []))
            brief = response.get("brief") or response.get("summary")
            if brief:
                aggregated["brief"] += f"[{advisor.name}] {brief}\n"

        aggregated["brief"] = aggregated["brief"].strip() or "Brief mínimo generado por broker"
        _write_json(run_dir.root / "council_brief.json", aggregated)
        return aggregated


class Planner:
    def __init__(self, backend: AgentInvoker) -> None:
        self.backend = backend

    def make_plan(
        self,
        job: AgencyJob,
        brief: Optional[JsonDict],
        run_dir: RunPaths,
        supervision: Dict[str, Any],
        override_command: Optional[str] = None,
        fallback_command: Optional[str] = None,
    ) -> Tuple[JsonDict, bool, Dict[str, Any]]:
        payload = {
            "mode": "plan",
            "goal": job.goal,
            "capabilities": job.capabilities,
            "brief": brief,
            "context": job.context.to_dict(),
            "metadata": job.metadata,
        }
        ok, response, meta = self.backend.invoke_supervised(
            payload,
            run_dir.logs_dir / "planner.log",
            supervision,
            role="planner",
            override_command=override_command,
            fallback_command=fallback_command,
        )
        rail = (
            str(
                (payload.get("metadata") or {}).get("rail")
                or os.getenv("AJAX_RAIL")
                or os.getenv("AJAX_ENV")
                or "prod"
            )
            .strip()
            .lower()
        )
        if isinstance(response, dict) and ("claims" in response or "hypothesis" in response):
            bundle = OutputBundle.from_dict(response) if OutputBundle else None
            if bundle and _validate_output_bundle:
                result = _validate_output_bundle(
                    bundle, original_text=json.dumps(response, ensure_ascii=False), rail=rail
                )
                guard = get_guard(rail) if get_guard else None
                note = None
                if result.action == "SOFT_BLOCK" and guard:
                    note = guard.format_soft_block(result)
                elif result.bundle is not None:
                    note = json.dumps(result.bundle.to_dict(), ensure_ascii=False, indent=2)
                response = {"plan": [], "notes": [note] if note else []}
                ok = False
                meta = dict(meta or {})
                meta["anti_optimism_action"] = result.action
                meta["anti_optimism_receipt"] = result.receipt_path
        _write_json(run_dir.root / "plan.json", response)
        response.setdefault("metadata", payload.get("metadata"))
        return response, ok, meta


class Executors:
    def __init__(self, backend: AgentInvoker) -> None:
        self.backend = backend

    def run(
        self,
        job: AgencyJob,
        plan: JsonDict,
        run_dir: RunPaths,
        supervision: Dict[str, Any],
        override_command: Optional[str] = None,
        fallback_command: Optional[str] = None,
    ) -> Tuple[List[ToolCall], List[AgentArtifact], bool, Dict[str, Any]]:
        payload = {
            "mode": "execute",
            "goal": job.goal,
            "capabilities": job.capabilities,
            "plan": plan.get("plan") or [],
            "job_id": job.job_id,
        }
        ok, response, meta = self.backend.invoke_supervised(
            payload,
            run_dir.logs_dir / "executor.log",
            supervision,
            role="executor",
            override_command=override_command,
            fallback_command=fallback_command,
        )
        _write_json(run_dir.root / "execution.json", response)
        actions = [ToolCall.from_any(item) for item in response.get("actions", [])]
        artifacts = [AgentArtifact.from_any(item) for item in response.get("artifacts", [])]
        existing_paths = {artifact.path for artifact in artifacts}
        for artifact in self._execute_local_actions(actions, run_dir):
            if artifact.path in existing_paths:
                continue
            artifacts.append(artifact)
            existing_paths.add(artifact.path)
        return actions, artifacts, ok, meta

    @staticmethod
    def _write_tool_error(artifacts_dir: Path, url: str, error: str) -> None:
        payload = {"url": url, "error": error, "ts": time.time()}
        error_path = artifacts_dir / "tool_error.json"
        error_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _execute_local_actions(
        self, actions: List[ToolCall], run_dir: RunPaths
    ) -> List[AgentArtifact]:
        produced: List[AgentArtifact] = []
        for action in actions:
            if action.tool != "browser.play":
                continue
            url = action.args.get("url") if isinstance(action.args, dict) else None
            if not url:
                continue
            raw_volume = action.args.get("volume") if isinstance(action.args, dict) else None
            if isinstance(raw_volume, (int, float)):
                volume = (
                    int(raw_volume * 100)
                    if isinstance(raw_volume, float) and raw_volume <= 1
                    else int(raw_volume)
                )
            else:
                volume = 40
            title = action.args.get("title") if isinstance(action.args, dict) else None
            try:
                if BrowserController is None:
                    raise RuntimeError("BrowserController unavailable")
                controller = BrowserController(run_dir.artifacts_dir)
                controller.play(url, volume=max(0, min(volume, 100)), title=title)
                produced.append(
                    AgentArtifact(type="player_state", path="artifacts/player_state.json")
                )
                screenshot_path = run_dir.artifacts_dir / "screenshot.png"
                if screenshot_path.exists():
                    produced.append(
                        AgentArtifact(type="screenshot", path="artifacts/screenshot.png")
                    )
            except Exception as exc:  # noqa: BLE001
                self._write_tool_error(run_dir.artifacts_dir, url, str(exc))
                produced.append(
                    AgentArtifact(
                        type="tool_error",
                        path="artifacts/tool_error.json",
                        description=str(exc),
                    )
                )
        return produced


class Verifier:
    def __init__(self, backend: AgentInvoker) -> None:
        self.backend = backend

    def check(
        self,
        job: AgencyJob,
        actions: List[ToolCall],
        artifacts: List[AgentArtifact],
        run_dir: RunPaths,
        supervision: Dict[str, Any],
        override_command: Optional[str] = None,
        fallback_command: Optional[str] = None,
    ) -> Tuple[JsonDict, bool, Dict[str, Any]]:
        payload = {
            "mode": "verify",
            "goal": job.goal,
            "actions": [action.to_dict() for action in actions],
            "artifacts": [artifact.to_dict() for artifact in artifacts],
        }
        # Pasar criterios de aceptación si el job los define en metadata
        try:
            if isinstance(job.metadata, dict) and job.metadata.get("acceptance_criteria"):
                payload["acceptance_criteria"] = job.metadata.get("acceptance_criteria")
        except Exception:
            pass
        ok, response, meta = self.backend.invoke_supervised(
            payload,
            run_dir.logs_dir / "verifier.log",
            supervision,
            role="verifier",
            override_command=override_command,
            fallback_command=fallback_command,
        )

        if verify_run is not None:
            try:
                report = verify_run(run_dir.root)
                response.setdefault("report", report)
                response.setdefault("metrics", {})
                advice = report.get("advice", [])
                existing_notes = response.get("notes")
                if isinstance(existing_notes, list):
                    notes_bucket = existing_notes
                elif existing_notes:
                    notes_bucket = [existing_notes]
                else:
                    notes_bucket = []
                response["notes"] = notes_bucket
                if isinstance(advice, list):
                    notes_bucket.extend(advice)
                elif advice:
                    notes_bucket.append(advice)
                response.setdefault("artifacts", []).append(
                    {
                        "type": "verifier_report",
                        "path": "artifacts/verifier_report.json",
                    }
                )
                # Handle the new outcome field: "success", "failure", or "neutral"
                outcome = report.get("outcome")
                if outcome == "failure":
                    response["ok"] = False
                else:  # "success" or "neutral" are both considered OK
                    response["ok"] = True
            except Exception as exc:  # pragma: no cover - best effort
                response.setdefault("errors", []).append(f"local_verifier_failed: {exc}")

        _write_json(run_dir.root / "verification.json", response)
        return response, ok, meta


class AgencyBroker:
    """Main entry point that executes the broker loop."""

    def __init__(self) -> None:
        self.router = Router()
        self.logger = logging.getLogger(__name__)
        self.policy_snapshot = load_policy()
        self.supervision_cfg_base = copy.deepcopy(
            self.policy_snapshot.config.get("supervision", {})
        )
        self.supervision_cfg = copy.deepcopy(self.supervision_cfg_base)
        self.routing_cfg = self.policy_snapshot.config.get("routing", {})
        # Mindset (meta-capa de propósito)
        self.mindset = self.policy_snapshot.config.get(
            "mindset",
            {
                "goal": "Mejorar continuamente eficiencia, seguridad y claridad del ecosistema",
                "heuristics": [
                    "buscar inconsistencias",
                    "sugerir optimizaciones",
                    "auto-refinar rutas",
                ],
                "tone": "constructivo, curioso, autocrítico",
            },
        )
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._codex_quota_cache: Optional[Dict[str, Any]] = None
        self._codex_quota_mtime: Optional[float] = None
        self.background_mode = False
        self.infra_breaker = InfraBreaker() if InfraBreaker else None

        council_invokers = [
            AgentInvoker(
                name="qwen",
                command=_load_cli_command("AGENCY_QWEN_CMD", "bin/qwen_task.py"),
                fallback=_default_brief_fallback,
            ),
            AgentInvoker(
                name="gemini",
                command=_load_cli_command("AGENCY_GEMINI_CMD", "bin/gemini_task.py"),
                fallback=_default_brief_fallback,
            ),
            AgentInvoker(
                name="codex",
                command=_load_cli_command("AGENCY_CODEX_CMD", "bin/codex_task.py"),
                fallback=_default_brief_fallback,
            ),
            AgentInvoker(
                name="groq",
                command=_load_cli_command("AGENCY_GROQ_CMD", "bin/groq_task.py"),
                fallback=_default_brief_fallback,
            ),
        ]
        # Optional built-in advisors for skepticism and red-teaming if available
        try:
            skeptic_cmd = _load_cli_command("AGENCY_COUNCIL_SKEPTIC", "bin/council_skeptic.py")
            if skeptic_cmd:
                council_invokers.append(
                    AgentInvoker(
                        name="skeptic", command=skeptic_cmd, fallback=_default_brief_fallback
                    )
                )
        except Exception:
            pass
        try:
            redteam_cmd = _load_cli_command("AGENCY_COUNCIL_REDTEAM", "bin/council_redteam.py")
            if redteam_cmd:
                council_invokers.append(
                    AgentInvoker(
                        name="redteam", command=redteam_cmd, fallback=_default_brief_fallback
                    )
                )
        except Exception:
            pass
        self.council = Council(advisors=council_invokers)

        planner_invoker = AgentInvoker(
            name="planner",
            command=_load_cli_command("AGENCY_PLANNER_CMD", None),
            fallback=_default_plan_fallback,
        )
        executor_invoker = AgentInvoker(
            name="executor",
            command=_load_cli_command("AGENCY_EXECUTOR_CMD", None),
            fallback=_default_executor_fallback,
        )
        verifier_invoker = AgentInvoker(
            name="verifier",
            command=_load_cli_command("AGENCY_VERIFIER_CMD", None),
            fallback=_default_verifier_fallback,
        )
        groq_invoker = AgentInvoker(
            name="groq",
            command=_load_cli_command("AGENCY_GROQ_CMD", "bin/groq_task.py"),
            fallback=_default_brief_fallback,
        )

        self.planner = Planner(planner_invoker)
        self.executors = Executors(executor_invoker)
        self.verifier = Verifier(verifier_invoker)
        self.groq_agent = groq_invoker  # Agente especial para tareas rápidas de Groq
        self.registry_snapshot = self._collect_registry_snapshot()

    def _collect_registry_snapshot(self) -> Optional[RegistrySnapshot]:
        specs: List[RegistrySpec] = []

        def add_spec(name: str, role: str, command: Optional[List[str]]) -> None:
            if command:
                specs.append(RegistrySpec(name=name, role=role, command=list(command)))

        add_spec("planner", "planner", self.planner.backend.command)
        add_spec("executor", "executor", self.executors.backend.command)
        add_spec("verifier", "verifier", self.verifier.backend.command)

        for idx, advisor in enumerate(self.council.advisors, start=1):
            add_spec(f"council_{idx}", "council", advisor.command)

        if not specs:
            return None

        try:
            return collect_registry(specs)
        except Exception:
            return None

    def _is_background_mode(self) -> bool:
        return (Path("config") / "mode.background").exists()

    def _apply_background_policy(self, job: AgencyJob) -> None:
        job.budget.steps = max(1, int(job.budget.steps * 0.6))
        job.budget.seconds = max(1, int(job.budget.seconds * 0.6))
        try:
            job.budget.tokens = max(1, int(job.budget.tokens))
        except Exception:
            pass

        timeouts = self.supervision_cfg.get("timeouts_s")
        if isinstance(timeouts, dict):
            for key, value in list(timeouts.items()):
                try:
                    timeouts[key] = float(value) * 1.5
                except (TypeError, ValueError):
                    continue

        backoff = self.supervision_cfg.get("backoff_s")
        if isinstance(backoff, list):
            scaled: List[float] = []
            for value in backoff:
                try:
                    scaled.append(float(value) * 1.5)
                except (TypeError, ValueError):
                    continue
            if scaled:
                self.supervision_cfg["backoff_s"] = scaled

    def _load_or_initialize_state(self, run_dir: RunPaths) -> ExecutionState:
        if self.supervision_cfg.get("resume_on_restart") and run_dir.state_file.exists():
            try:
                data = json.loads(run_dir.state_file.read_text(encoding="utf-8"))
                state = ExecutionState.from_dict(data)
                return state
            except Exception:
                pass
        state = ExecutionState()
        self._write_state(run_dir, state)
        return state

    def _write_state(self, run_dir: RunPaths, state: ExecutionState) -> None:
        run_dir.state_file.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _append_journal(self, run_dir: RunPaths, event: Dict[str, Any]) -> None:
        event_payload = {
            "ts": time.time(),
            **event,
        }
        with run_dir.journal_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")

    def _update_state(
        self, run_dir: RunPaths, state: ExecutionState, phase: str, role: Optional[str] = None
    ) -> None:
        state.phase = phase
        state.active_role = role
        self._write_state(run_dir, state)

    def _record_attempt(
        self,
        run_dir: RunPaths,
        state: ExecutionState,
        role: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        command = meta.get("command") if meta else None
        breaker_state = self._breaker_state_label(role, command) if command else "n/a"
        state.retries[role] = state.retries.get(role, 0) + (0 if success else 1)
        if not success:
            state.last_error = (details or {}).get("error") if details else None
        else:
            state.last_error = None
        self._write_state(run_dir, state)
        payload: Dict[str, Any] = {"event": "role_result", "role": role, "success": success}
        if details:
            payload.update(details)
        self._append_journal(run_dir, payload)
        metrics_event = {
            "run_id": run_dir.root.name,
            "role": role,
            "ok": success,
            "lat_ms": float(meta.get("duration_ms", 0.0)) if meta else 0.0,
            "timeout": bool(meta.get("timed_out")) if meta else False,
            "attempts": int(meta.get("attempts", 0)) if meta else 0,
            "breaker_state": breaker_state,
            "command": command or "internal",
            "ts": time.time(),
        }
        self._append_metrics(run_dir, metrics_event)
        if command:
            self._update_circuit(role, command, success)

    def _finalize_result(self, run_dir: RunPaths, payload: Dict[str, Any], bucket: str) -> None:
        enriched: Dict[str, Any] = dict(payload)
        enriched.setdefault("bucket", bucket)
        enriched.setdefault("job_id", run_dir.root.name)
        if "started_at" not in enriched:
            try:
                enriched["started_at"] = run_dir.root.stat().st_mtime
            except OSError:
                pass
        finished_at = enriched.get("finished_at")
        if not isinstance(finished_at, (int, float)):
            finished_at = time.time()
            enriched["finished_at"] = finished_at
        if "finished_at_iso" not in enriched:
            enriched["finished_at_iso"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_at)
            )

        try:
            run_dir.write_result(enriched)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[broker-warn] no se pudo escribir result.json: {exc}")
            return

        self._invoke_latest_update(run_dir)

    def _invoke_latest_update(self, run_dir: RunPaths) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "update_latest.py"
        if not script.exists() or not run_dir.result_file.exists():
            return

        python = sys.executable or "python3"
        cmd = [python, str(script), str(run_dir.result_file), str(run_dir.root)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[broker-warn] update_latest.py lanzó excepción: {exc}")
            return

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            detail = f": {stderr}" if stderr else ""
            print(f"[broker-warn] update_latest.py fallo{detail}")

    @staticmethod
    def _dedupe(commands: Iterable[str]) -> List[str]:
        seen: List[str] = []
        for cmd in commands:
            if cmd and cmd not in seen:
                seen.append(cmd)
        return seen

    @staticmethod
    def _load_heartbeat_status() -> Dict[str, Any]:
        path = Path("artifacts/health/ajax_heartbeat.json")
        if not path.exists():
            return {"status": "unknown", "problems": ["missing_heartbeat"], "subsystems": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("heartbeat is not a JSON object")
            payload.setdefault("status", "unknown")
            if "problems" not in payload:
                payload["problems"] = []
            return payload
        except Exception as exc:  # pragma: no cover - best effort
            return {"status": "unknown", "problems": [f"parse_error:{exc}"], "subsystems": {}}

    @staticmethod
    def _job_bypasses_heartbeat(job: AgencyJob) -> bool:
        meta = job.metadata or {}
        risk = str(meta.get("risk", "")).lower()
        bucket = str(
            meta.get("bucket") or meta.get("task_type") or meta.get("category", "")
        ).lower()
        goal = job.goal.lower()
        if risk in {"maintenance", "canary", "health", "diagnostic", "low"}:
            return True
        if bucket in {"maintenance", "canary", "health", "diagnostic", "diag"}:
            return True
        tokens = (
            "canary",
            "health",
            "heartbeat",
            "diagnost",
            "mantenimiento",
            "maintenance",
            "latido",
            "self-heal",
        )
        return any(tok in goal for tok in tokens)

    def _infer_job_bucket(self, job: AgencyJob) -> str:
        metadata = job.metadata or {}
        for key in ("bucket", "intent", "category", "domain", "task_type"):
            value = metadata.get(key)
            if value:
                return str(value).lower()
        goal = job.goal.lower()
        if any(keyword in goal for keyword in ("código", "codigo", "code", "script", "programa")):
            return "code"
        if any(keyword in goal for keyword in ("rag", "documento", "resumen", "consulta")):
            return "rag"
        if any(keyword in goal for keyword in ("browser", "navegador", "web", "url")):
            return "browser"
        if any(keyword in goal for keyword in ("excel", "word", "notepad", "office")):
            return "office"
        return "general"

    def _planner_commands_from_config(
        self, planner_cfg: Dict[str, Any], job: AgencyJob
    ) -> List[str]:
        commands: List[str] = []
        if not planner_cfg:
            return commands
        bucket = self._infer_job_bucket(job)
        by_bucket = planner_cfg.get("by_bucket") or {}
        if by_bucket:
            bucket_lower = bucket.lower()
            for key, values in by_bucket.items():
                if not isinstance(values, list):
                    continue
                if key.lower() == bucket_lower:
                    commands.extend(values)
                    break
            if not commands:
                for fallback_key in ("general", "default"):
                    values = by_bucket.get(fallback_key)
                    if isinstance(values, list):
                        commands.extend(values)
                        break
            if not commands:
                first_bucket = next(iter(by_bucket.values()), [])
                if isinstance(first_bucket, list):
                    commands.extend(first_bucket)

        if not commands and isinstance(planner_cfg.get("candidates"), list):
            commands.extend(planner_cfg.get("candidates", []))
        # Optional runtime preference: PLANNER_MODE=local-first|cloud-first
        try:
            mode = os.getenv("PLANNER_MODE", "").lower()
            if mode in {"local-first", "cloud-first"} and commands:

                def is_local(cmd: str) -> bool:
                    return "lmstudio_task.py" in cmd or "--model lfm2moe" in cmd

                locals_first = [c for c in commands if is_local(c)]
                clouds = [c for c in commands if not is_local(c)]
                commands = (
                    (locals_first + clouds) if mode == "local-first" else (clouds + locals_first)
                )
        except Exception:
            pass
        # Healthcheck LM Studio to protect SLO
        try:
            if commands and ("lmstudio_task.py" in commands[0] or "--model lfm2moe" in commands[0]):
                required = os.getenv("LMSTUDIO_HEALTH_REQUIRED", "1") != "0"
                if not _lmstudio_healthy():
                    if required:
                        self.logger.warning(
                            "[routing] LM Studio is DOWN. Skipping local-first option to protect SLO."
                        )
                        first = commands.pop(0)
                        commands.append(first)
                    else:
                        self.logger.info("[routing] LM Studio is DOWN (not required).")
        except Exception as exc:
            self.logger.error(f"[routing] Error during LM Studio healthcheck: {exc}")
        return self._dedupe(commands)

    def _load_codex_quota(self) -> Optional[Dict[str, Any]]:
        path = Path("artifacts/codex_quota.json")
        if not path.exists():
            self._codex_quota_cache = None
            self._codex_quota_mtime = None
            return None
        mtime = path.stat().st_mtime
        if self._codex_quota_mtime != mtime:
            try:
                self._codex_quota_cache = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._codex_quota_cache = None
            self._codex_quota_mtime = mtime
        return self._codex_quota_cache

    def _ensure_codex_quota(self) -> Optional[Dict[str, Any]]:
        cached = self._load_codex_quota()
        if cached is not None:
            return cached
        script_path = Path("scripts/quota.py")
        out_path = Path("artifacts/codex_quota.json")
        if not script_path.exists():
            return None
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                ["python", str(script_path), "--out", str(out_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                try:
                    self._codex_quota_cache = json.loads(completed.stdout.strip().splitlines()[-1])
                except json.JSONDecodeError:
                    self._codex_quota_cache = None
            else:
                self._codex_quota_cache = None
        except Exception:
            self._codex_quota_cache = None
        self._codex_quota_mtime = out_path.stat().st_mtime if out_path.exists() else None
        return self._codex_quota_cache

    def _job_requests_codex(self, job: AgencyJob) -> bool:
        policy_extras = getattr(job.policy, "extras", {}) or {}
        if policy_extras.get("emergency_codex"):
            return True
        return bool(job.metadata.get("emergency_codex"))

    def _codex_allowed_for_job(self, job: AgencyJob) -> bool:
        if self._job_requests_codex(job):
            return True
        quota = self._ensure_codex_quota()
        if not quota:
            return False
        if not bool(quota.get("ok_to_use", False)) or quota.get("override_required"):
            return False
        routing_allow = bool(
            (self.routing_cfg or {}).get("metadata", {}).get("codex_allowed", True)
        )
        return routing_allow

    def _filter_codex_commands(
        self, commands: List[str], job: AgencyJob, allow_codex: bool
    ) -> List[str]:
        if allow_codex:
            return commands
        return [cmd for cmd in commands if "bin/codex_task.py" not in cmd]

    def _filter_commands_by_circuit(self, role: str, commands: List[str]) -> List[str]:
        filtered: List[str] = []
        for cmd in commands:
            if not cmd:
                continue
            if self._is_circuit_open(role, cmd):
                continue
            filtered.append(cmd)
        return filtered

    def _is_task_suitable_for_groq(self, job: AgencyJob) -> bool:
        """Determina si una tarea es apropiada para ser ejecutada por Groq basándose en su complejidad y tipo."""
        goal = job.goal.lower()
        metadata = job.metadata or {}

        # Verificar metadatos explícitos que indiquen preferencia por Groq
        if metadata.get("model_preference") == "groq":
            return True

        # Tareas que típicamente se benefician de la velocidad de Groq
        fast_tasks_keywords = [
            "resumen",
            "resume",
            "summarize",
            "clasifica",
            "classify",
            "formatea",
            "format",
            "extrae",
            "extract",
            "convierte",
            "convert",
            "simple",
            "básico",
            "basic",
            "breve",
            "short",
            "rápido",
            "quick",
            "etiqueta",
            "tag",
            "cuenta",
            "count",
        ]

        # Tareas complejas que NO son apropiadas para Groq
        complex_tasks_keywords = [
            "análisis profundo",
            "deep analysis",
            "estratégico",
            "strategic",
            "planificación",
            "planning",
            "investigación",
            "research",
            "desarrolla",
            "develop",
            "crea",
            "create",
            "implementa",
            "implement",
            "diseña",
            "design",
            "optimiza",
            "optimize",
            "resuelve",
            "solve",
            "complejo",
            "complex",
            "detallado",
            "detailed",
        ]

        # Verificar si es una tarea rápida
        is_fast_task = any(keyword in goal for keyword in fast_tasks_keywords)

        # Verificar si es una tarea compleja
        is_complex_task = any(keyword in goal for keyword in complex_tasks_keywords)

        # Tareas cortas tienden a ser más adecuadas para Groq
        is_short_task = len(goal) < 100

        # Si es explícitamente compleja, no es para Groq
        if is_complex_task:
            return False

        # Si es corta y rápida, probablemente sí
        if is_short_task and is_fast_task:
            return True

        # Por defecto, tareas simples
        return not is_complex_task

    def _default_command_for_role(self, role: str) -> str:
        mapping = {
            "planner": "bin/gemini_task.py --role planner --json",
            "executor": "bin/qwen_task.py --role executor --json",
            "verifier": "bin/qwen_task.py --role verifier --json",
        }
        return mapping.get(role, "")

    def _legacy_routing(
        self, role: str, job: AgencyJob, routing: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        if role == "planner":
            risk = str(job.metadata.get("risk_level", "normal")).lower()
            primary = (
                routing.get("planner.high_risk")
                if risk in {"high", "critical"}
                else routing.get("planner.normal")
            )
            fallback = (
                routing.get("planner.normal")
                if primary != routing.get("planner.normal")
                else routing.get("planner.high_risk")
            )
        elif role == "executor":
            primary = routing.get("executor")
            fallback = routing.get("executor.fallback")
        elif role == "verifier":
            primary = routing.get("verifier.primary")
            fallback = routing.get("verifier.fallback")
        else:
            primary = None
            fallback = None

        default_cmd = self._default_command_for_role(role)
        candidates = [cmd for cmd in (primary, fallback, default_cmd) if cmd]
        candidates = self._filter_commands_by_circuit(role, self._dedupe(candidates))
        if not candidates:
            return None, None
        primary_cmd = candidates[0]
        fallback_cmd = candidates[1] if len(candidates) > 1 else None
        return primary_cmd, fallback_cmd

    def _routing_for_role(
        self, role: str, job: AgencyJob
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        env = os.environ.get(f"AGENCY_{role.upper()}_CMD")
        snapshot: Dict[str, Any] = {
            "role": role,
            "source": "env" if env else "routing_cfg",
            "env_override": env,
            "raw_candidates": [],
            "pre_circuit_candidates": [],
            "filtered_candidates": [],
            "allow_codex": False,
        }

        if env:
            snapshot.update({"selected": env, "fallback": None})
            return env, None, snapshot

        routing = self.routing_cfg or {}
        commands: List[str] = []
        raw_candidates: List[str] = []
        allow_codex = False

        # Lógica especial para usar Groq en tareas apropiadas
        use_groq_for_this_task = self._is_task_suitable_for_groq(job)

        if role == "planner":
            planner_cfg = (
                routing.get("planner", {}) if isinstance(routing.get("planner", {}), dict) else {}
            )
            raw_candidates = self._planner_commands_from_config(planner_cfg, job)
            allow_codex = self._codex_allowed_for_job(job)
            commands = self._filter_codex_commands(list(raw_candidates), job, allow_codex)

            # Si la tarea es apropiada para Groq, ponerlo como primera opción
            if use_groq_for_this_task:
                commands.insert(0, "bin/groq_task.py --role planner --json")
        else:
            role_cfg = routing.get(role, {}) if isinstance(routing.get(role, {}), dict) else {}
            raw_candidates = (
                role_cfg.get("candidates") if isinstance(role_cfg.get("candidates"), list) else []
            )
            commands = self._filter_codex_commands(list(raw_candidates), job, allow_codex=False)

            # Si la tarea es apropiada para Groq, considerarlo como opción
            if use_groq_for_this_task:
                if role == "executor":
                    commands.insert(0, "bin/groq_task.py --role executor --json")
                elif role == "verifier":
                    commands.insert(0, "bin/groq_task.py --role verifier --json")

        deduped = self._dedupe(commands)
        pre_circuit = list(deduped)
        filtered = self._filter_commands_by_circuit(role, deduped)

        snapshot.update(
            {
                "raw_candidates": raw_candidates,
                "pre_circuit_candidates": pre_circuit,
                "filtered_candidates": filtered,
                "allow_codex": allow_codex,
                "use_groq_for_this_task": use_groq_for_this_task,  # Añadir información de decisión
            }
        )

        if not filtered:
            primary, fallback = self._legacy_routing(role, job, routing)
            snapshot.update(
                {
                    "source": "legacy",
                    "selected": primary,
                    "fallback": fallback,
                }
            )
            return primary, fallback, snapshot

        primary = filtered[0]
        fallback = filtered[1] if len(filtered) > 1 else None
        snapshot.update({"selected": primary, "fallback": fallback})
        # Record planner selection usage (best-effort)
        if role == "planner":
            try:
                day = time.strftime("%Y-%m-%d", time.localtime())
                usage_path = Path("data/planner_usage.json")
                usage: Dict[str, Dict[str, int]] = {}
                if usage_path.exists():
                    try:
                        usage = json.loads(usage_path.read_text(encoding="utf-8"))
                    except Exception:
                        usage = {}
                bucket = usage.setdefault(day, {})
                key = (
                    "lmstudio"
                    if (primary and ("lmstudio_task.py" in primary or "lfm2moe" in primary))
                    else (
                        "qwen"
                        if (primary and "qwen_task.py" in primary)
                        else (
                            "gemini"
                            if (primary and "gemini_task.py" in primary)
                            else (
                                "codex"
                                if (primary and "codex_task.py" in primary)
                                else (
                                    "groq"
                                    if (
                                        primary
                                        and (
                                            "groq_task.py" in primary or " groq" in primary.lower()
                                        )
                                    )
                                    else "other"
                                )
                            )
                        )
                    )
                )
                bucket[key] = int(bucket.get(key, 0)) + 1
                usage_path.parent.mkdir(parents=True, exist_ok=True)
                usage_path.write_text(
                    json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        return primary, fallback, snapshot

    def _breaker_key(self, role: str, command: str) -> str:
        return f"{role}:{command}"

    def _is_circuit_open(self, role: str, command: str) -> bool:
        key = self._breaker_key(role, command)
        info = self.circuit_breakers.get(key)
        if not info:
            return False
        now = time.time()
        window = float(self.supervision_cfg.get("circuit_breaker", {}).get("window_s", 60))
        cool_down = float(self.supervision_cfg.get("circuit_breaker", {}).get("cool_down_s", 120))

        failures: List[float] = [ts for ts in info.get("failures", []) if now - ts <= window]
        info["failures"] = failures
        state = info.get("state", "closed")
        opened_at = info.get("opened_at")

        if state == "open":
            if opened_at and now - opened_at >= cool_down:
                info["state"] = "half_open"
                info["failures"] = []
                info["opened_at"] = None
                return False
            return True
        return False

    def _breaker_state_label(self, role: str, command: Optional[str]) -> str:
        if not command:
            return "n/a"
        key = self._breaker_key(role, command)
        info = self.circuit_breakers.get(key)
        if not info:
            return "closed"
        return str(info.get("state", "closed"))

    def _update_circuit(self, role: str, command: str, success: bool) -> None:
        key = self._breaker_key(role, command)
        info = self.circuit_breakers.setdefault(
            key, {"state": "closed", "failures": [], "opened_at": None}
        )
        cfg = self.supervision_cfg.get("circuit_breaker", {})
        threshold = int(cfg.get("failure_threshold", 5))
        window = float(cfg.get("window_s", 60))
        now = time.time()

        if success:
            info["state"] = "closed"
            info["failures"] = []
            info["opened_at"] = None
            return

        failures: List[float] = info.get("failures", [])
        failures = [ts for ts in failures if now - ts <= window]
        failures.append(now)
        info["failures"] = failures

        if len(failures) >= threshold:
            info["state"] = "open"
            info["opened_at"] = now

    def _append_metrics(self, run_dir: RunPaths, payload: Dict[str, Any]) -> None:
        payload.setdefault("ts", time.time())
        with run_dir.metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _increment_token_totals(self, tokens_in: int, tokens_out: int) -> None:
        if not tokens_in and not tokens_out:
            return
        usage_path = Path("data/token_usage.json")
        day = time.strftime("%Y-%m-%d", time.localtime())
        totals: Dict[str, Dict[str, int]] = {}
        if usage_path.exists():
            try:
                totals = json.loads(usage_path.read_text(encoding="utf-8"))
            except Exception:
                totals = {}
        day_totals = totals.setdefault(day, {"tokens_in": 0, "tokens_out": 0})
        day_totals["tokens_in"] = int(day_totals.get("tokens_in", 0)) + int(tokens_in or 0)
        day_totals["tokens_out"] = int(day_totals.get("tokens_out", 0)) + int(tokens_out or 0)
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        usage_path.write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_job(self, job: AgencyJob) -> AgencyResult:
        self.background_mode = self._is_background_mode()
        self.supervision_cfg = copy.deepcopy(self.supervision_cfg_base)
        if self.background_mode:
            self._apply_background_policy(job)
        previous_mode = BACKGROUND_MODE_ACTIVE
        _set_background_mode(self.background_mode)

        run_dir = RunPaths.create(job.job_id)
        run_dir.write_job(job.to_dict())
        write_policy(self.policy_snapshot, run_dir.root / "policy.yml")
        if self.registry_snapshot:
            write_registry(self.registry_snapshot, run_dir.root / "registry.json")
        council_state = load_council_state()
        infra_state: Dict[str, Any] = {}
        infra_block = False
        if self.infra_breaker:
            try:
                infra_state = dict(self.infra_breaker.state)
            except Exception:
                infra_state = {}
            try:
                infra_block = self.infra_breaker.infra_should_block(time.time())
                infra_state["blocked"] = infra_block
            except Exception:
                infra_block = False
        audit_actions: List[ActionAudit] = []
        audit_verification: Optional[Dict[str, Any]] = None

        notes_path = run_dir.root / "NOTES.md"
        _append_line(notes_path, f"# Run {job.job_id}")
        _append_line(notes_path, f"Goal: {job.goal}")
        _compact_file(notes_path)
        bucket = self._infer_job_bucket(job)

        heartbeat = self._load_heartbeat_status()
        hb_status = str(heartbeat.get("status", "unknown")).lower()
        hb_problems = heartbeat.get("problems", [])
        hb_bypass = self._job_bypasses_heartbeat(job)
        _append_line(
            notes_path, f"Heartbeat status={hb_status} bypass={hb_bypass} problems={hb_problems}"
        )
        _compact_file(notes_path)
        if hb_status != "green" and not hb_bypass:
            if self.infra_breaker:
                try:
                    self.infra_breaker.infra_register_failure(
                        kind=hb_status or "heartbeat",
                        now=time.time(),
                        meta={"problems": hb_problems},
                    )
                except Exception:
                    pass
            errors = [f"heartbeat_status:{hb_status}"]
            if hb_problems:
                errors.append("heartbeat_problems:" + ",".join(str(p) for p in hb_problems))
            result = AgencyResult(
                ok=False,
                answer=f"Latido AJAX en estado {hb_status}; job bloqueado salvo mantenimiento/canary.",
                actions=[],
                artifacts=[],
                metrics=AgentMetrics(),
                confidence=0.0,
                errors=errors,
            )
            self._finalize_result(run_dir, result.to_dict(), bucket)
            _set_background_mode(previous_mode)
            return result

        infra_block = False
        if self.infra_breaker:
            try:
                infra_block = self.infra_breaker.infra_should_block(time.time())
            except Exception:
                infra_block = False
        is_maintenance = hb_bypass or bucket in {
            "maintenance",
            "canary",
            "health",
            "diagnostic",
            "diag",
        }
        if infra_block and not is_maintenance:
            speak_instability_alert("infra")
            result = AgencyResult(
                ok=False,
                answer="Infra breaker activado: misiones normales bloqueadas.",
                actions=[],
                artifacts=[],
                metrics=AgentMetrics(),
                confidence=0.0,
                errors=["infra_breaker_tripped"],
            )
            self._finalize_result(run_dir, result.to_dict(), bucket)
            _set_background_mode(previous_mode)
            return result

        job_kind = str((job.metadata or {}).get("job_kind", "")).lower()
        if job_kind == "pas":
            from agency.agentic_loop import AgentConstraints, run_agentic_session

            pas_goal = str((job.metadata or {}).get("pas_goal") or job.goal)
            pas_constraints = AgentConstraints.from_any((job.metadata or {}).get("pas_constraints"))
            pas_result = run_agentic_session(
                pas_goal,
                constraints=pas_constraints,
                session_id=job.job_id,
                persist_path=run_dir.root / "pas_result.json",
            )
            artifacts: List[AgentArtifact] = []
            if pas_result.summary_path:
                artifacts.append(
                    AgentArtifact(
                        type="pas_summary",
                        path=str(pas_result.summary_path),
                        description="Resumen PAS (autonomía supervisada)",
                        meta={"status": pas_result.status},
                    )
                )
            for mission in pas_result.missions_run:
                for art in mission.artifacts:
                    artifacts.append(
                        AgentArtifact(
                            type="mission_artifact",
                            path=str(art),
                            description=f"PAS misión {mission.job_id}",
                        )
                    )
            errors = []
            if pas_result.status != "success":
                errors.append(f"pas_status:{pas_result.status}")
                errors.extend(pas_result.next_suggestions or [])
            result = AgencyResult(
                ok=pas_result.status == "success",
                answer=pas_result.final_state_summary,
                actions=[],
                artifacts=artifacts,
                metrics=AgentMetrics(),
                confidence=0.0,
                errors=errors,
            )
            self._finalize_result(run_dir, result.to_dict(), bucket)
            _set_background_mode(previous_mode)
            return result

        def _job_requires_planner(job_obj: AgencyJob) -> bool:
            meta = getattr(job_obj, "metadata", {}) or {}
            planner_policy = meta.get("planner_policy", "auto")
            job_kind = meta.get("job_kind")
            if planner_policy == "never":
                return False
            if job_kind == "maintenance":
                return False
            return True

        if not _job_requires_planner(job):
            self.logger.info(
                "Broker: executing job %s via plan_runner (planner disabled by policy)", job.job_id
            )
            guard = _guard_degraded_plan({"plan": job.metadata.get("actions", [])})
            if guard:
                detail_path = run_dir.root / "council_guard.json"
                _write_json(detail_path, guard)
                reason = "Council en modo degradado; acciones fuera de white-list."
                result = AgencyResult(
                    ok=False,
                    answer=reason,
                    actions=[],
                    artifacts=[],
                    metrics=AgentMetrics(),
                    confidence=0.0,
                    errors=["council_degraded_mode"],
                )
                self._finalize_result(run_dir, result.to_dict(), bucket)
                _set_background_mode(previous_mode)
                return result
            plan_actions_raw = (
                job.metadata.get("actions", []) if isinstance(job.metadata, dict) else []
            )
            allowed_critical, blocked_reason, audited = evaluate_plan_actions(
                plan_actions_raw, council_state, infra_block
            )
            if audited:
                audit_actions.extend(audited)
            if not allowed_critical and blocked_reason:
                speak_instability_alert(
                    "infra" if blocked_reason == "infra_breaker_tripped" else "mission"
                )
                result = AgencyResult(
                    ok=False,
                    answer=f"Acción crítica bloqueada ({blocked_reason}).",
                    actions=[],
                    artifacts=[],
                    metrics=AgentMetrics(),
                    confidence=0.0,
                    errors=["critical_action_blocked"],
                )
                audit_path = run_dir.root / "audit_log.json"
                write_audit_log(
                    audit_path,
                    mission_id=job.job_id,
                    intent=job.goal,
                    council_state=council_state,
                    infra_state=infra_state,
                    actions=audit_actions,
                    verification=None,
                    final_status="blocked_by_guard",
                    result_ok=False,
                    errors=result.errors,
                )
                self._finalize_result(run_dir, result.to_dict(), bucket)
                _set_background_mode(previous_mode)
                return result
            try:
                res = run_job_plan(job)
                return AgencyResult(
                    ok=bool(res.get("ok")),
                    answer="Plan runner executed (planner disabled)",
                    actions=[],
                    artifacts=res.get("artifacts", []),
                    errors=[res.get("error")] if res.get("error") else [],
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Broker: plan_runner failed for job %s", job.job_id)
                return AgencyResult(ok=False, answer=str(exc), errors=[str(exc)])

        decision = self.router.decide(job)
        _write_json(run_dir.root / "router.json", decision.to_dict())

        quota_snapshot = self._ensure_codex_quota()
        routing_trace: Dict[str, Any] = {
            "bucket": bucket,
            "candidates_by_role": {},
            "selected": {},
            "signals": {
                "quota": quota_snapshot,
                "bench_source": "models_bench_report.json",
                "policy": "policy.overrides.yml",
            },
        }
        routing_trace_path = run_dir.root / "routing_decision.json"

        def _persist_routing(role: str, snapshot: Dict[str, Any]) -> None:
            routing_trace["candidates_by_role"][role] = {
                "env_override": snapshot.get("env_override"),
                "raw": snapshot.get("raw_candidates"),
                "pre_circuit": snapshot.get("pre_circuit_candidates"),
                "filtered": snapshot.get("filtered_candidates"),
                "allow_codex": snapshot.get("allow_codex"),
                "source": snapshot.get("source"),
            }
            selected = snapshot.get("selected")
            if selected is not None:
                routing_trace["selected"][role] = selected
            fallback_cmd = snapshot.get("fallback")
            if fallback_cmd:
                routing_trace.setdefault("fallbacks", {})[role] = fallback_cmd
            elif "fallbacks" in routing_trace and role in routing_trace["fallbacks"]:
                # Limpia cuando ya no hay fallback
                routing_trace["fallbacks"].pop(role)
                if not routing_trace["fallbacks"]:
                    routing_trace.pop("fallbacks")
            _write_json(routing_trace_path, routing_trace)
            if role == "planner":
                chosen = routing_trace.get("selected", {}).get("planner")
                if chosen:
                    parts = chosen.split()
                    planner_label = parts[0] if parts else chosen
                    for part in parts:
                        if part.startswith("bin/"):
                            planner_label = part
                            break
                    quota_ok = bool(quota_snapshot and quota_snapshot.get("ok_to_use"))
                    print(f"[routing] bucket={bucket} planner={planner_label} quota_ok={quota_ok}")

        brief_path = run_dir.root / "council_brief.json"
        brief: Optional[JsonDict] = None
        if brief_path.exists():
            try:
                brief = json.loads(brief_path.read_text(encoding="utf-8"))
            except Exception:
                brief = None

        if brief is None and decision.use_council:
            brief = self.council.gather_brief(job, run_dir)
            _append_line(notes_path, "Council consulted: yes")
        elif not decision.use_council:
            _append_line(notes_path, "Council consulted: no")

        def load_json(path: Path) -> Optional[JsonDict]:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    return None
            return None

        state = self._load_or_initialize_state(run_dir)
        if state.phase == "DONE":
            # Run ya completado previamente; devolver resultado persistido sin marcar fallo spurious
            persisted = load_json(run_dir.root / "result.json") or {}
            ok_prev = bool(persisted.get("ok"))
            if ok_prev:
                try:
                    errors = persisted.get("errors") or []
                    answer = persisted.get("answer") or f"Run {job.job_id} already completed"
                    result = AgencyResult(
                        ok=ok_prev,
                        answer=answer,
                        actions=[ToolCall(**a) for a in persisted.get("actions", [])]
                        if persisted.get("actions")
                        else [],
                        artifacts=[AgentArtifact(**a) for a in persisted.get("artifacts", [])]
                        if persisted.get("artifacts")
                        else [],
                        metrics=AgentMetrics(**persisted.get("metrics", {}))
                        if persisted.get("metrics")
                        else AgentMetrics(),
                        confidence=float(persisted.get("confidence", 0.0)),
                        errors=errors,
                    )
                    _set_background_mode(previous_mode)
                    return result
                except Exception:
                    pass
            # Si estaba en DONE pero sin éxito, limpiamos estado para relanzar fresco
            for fname in [
                "plan.json",
                "execution.json",
                "verification.json",
                "result.json",
                "audit_log.json",
                "switchboard.json",
                "routing_decision.json",
            ]:
                try:
                    (run_dir.root / fname).unlink()
                except Exception:
                    pass
            for fname in ["verifier_report.json", "tool_error.json"]:
                try:
                    (run_dir.artifacts_dir / fname).unlink()
                except Exception:
                    pass
            state = ExecutionState()
            self._write_state(run_dir, state)

        plan_data = load_json(run_dir.root / "plan.json")
        execution_data = load_json(run_dir.root / "execution.json")
        verification_data = load_json(run_dir.root / "verification.json")

        # Escribir archivo de switchboard si existe información
        switchboard_decision = {
            "route": getattr(self, "current_route", "unknown"),
            "rule": getattr(self, "applied_rule", "default"),
        }
        (run_dir.root / "switchboard.json").write_text(json.dumps(switchboard_decision, indent=2))

        actions: List[ToolCall] = []
        artifacts: List[AgentArtifact] = []

        if execution_data:
            actions = [ToolCall.from_any(item) for item in execution_data.get("actions", [])]
            artifacts = [
                AgentArtifact.from_any(item) for item in execution_data.get("artifacts", [])
            ]

        if verification_data:
            extra_artifacts = verification_data.get("artifacts") or []
            for item in extra_artifacts:
                try:
                    artifact = AgentArtifact.from_any(item)
                except ValueError:
                    continue
                artifacts.append(artifact)

        while state.phase not in {"DONE", "ABORT"}:
            if state.phase == "PLAN":
                self._append_journal(run_dir, {"event": "phase_start", "phase": "PLAN"})
                # Si el job ya trae acciones y (opcional) pide saltar planificación, crea plan sintético
                if job.metadata.get("actions") and job.policy.to_dict().get("skip_planner", True):
                    primary_cmd, fallback_cmd, routing_snapshot = self._routing_for_role(
                        "planner", job
                    )
                    routing_snapshot["source"] = "skip_planner"
                    _persist_routing("planner", routing_snapshot)
                    plan = {"goal": job.goal, "plan": job.metadata["actions"]}
                    _write_json(run_dir.root / "plan.json", plan)  # usa helper local
                    state.phase = "EXEC"  # avanza a ejecución
                else:
                    # ... flujo normal de planificación ...
                    primary_cmd, fallback_cmd, routing_snapshot = self._routing_for_role(
                        "planner", job
                    )
                    _persist_routing("planner", routing_snapshot)
                    state.active_role = "planner"
                    plan_data, cli_ok, meta = self.planner.make_plan(
                        job,
                        brief,
                        run_dir,
                        self.supervision_cfg,
                        override_command=primary_cmd,
                        fallback_command=fallback_cmd,
                    )
                    self._record_attempt(
                        run_dir,
                        state,
                        "planner",
                        success=cli_ok and bool(plan_data.get("plan")),
                        details=None if cli_ok else {"error": "planner_fallback"},
                        meta=meta,
                    )
                    if not plan_data.get("plan"):
                        state.phase = "ABORT"
                        break
                    for note in plan_data.get("notes", []):
                        _append_line(notes_path, f"PLAN: {note}")
                state.step_index = 1
                self._update_state(run_dir, state, "EXEC")
                continue

            if state.phase == "EXEC":
                self._append_journal(run_dir, {"event": "phase_start", "phase": "EXEC"})
                if not plan_data:
                    plan_data = load_json(run_dir.root / "plan.json") or {}
                guard = _guard_degraded_plan(plan_data or {})
                if guard:
                    _write_json(run_dir.root / "council_guard.json", guard)
                    state.phase = "ABORT"
                    state.last_error = "council_degraded_mode"
                    result = AgencyResult(
                        ok=False,
                        answer="Council en modo degradado; plan bloqueado fuera de white-list.",
                        actions=[],
                        artifacts=[],
                        metrics=AgentMetrics(),
                        confidence=0.0,
                        errors=["council_degraded_mode"],
                    )
                    self._finalize_result(run_dir, result.to_dict(), bucket)
                    _set_background_mode(previous_mode)
                    return result
                plan_actions_raw = (plan_data or {}).get("plan") or []
                allowed_critical, blocked_reason, audited = evaluate_plan_actions(
                    plan_actions_raw, council_state, infra_block
                )
                if audited:
                    audit_actions.extend(audited)
                if not allowed_critical and blocked_reason:
                    speak_instability_alert(
                        "infra" if blocked_reason == "infra_breaker_tripped" else "mission"
                    )
                    result = AgencyResult(
                        ok=False,
                        answer=f"Acción crítica bloqueada ({blocked_reason}).",
                        actions=[],
                        artifacts=[],
                        metrics=AgentMetrics(),
                        confidence=0.0,
                        errors=["critical_action_blocked"],
                    )
                    audit_path = run_dir.root / "audit_log.json"
                    write_audit_log(
                        audit_path,
                        mission_id=job.job_id,
                        intent=job.goal,
                        council_state=council_state,
                        infra_state=infra_state,
                        actions=audit_actions,
                        verification=None,
                        final_status="blocked_by_guard",
                        result_ok=False,
                        errors=result.errors,
                    )
                    self._finalize_result(run_dir, result.to_dict(), bucket)
                    _set_background_mode(previous_mode)
                    return result
                primary_cmd, fallback_cmd, routing_snapshot = self._routing_for_role(
                    "executor", job
                )
                _persist_routing("executor", routing_snapshot)
                state.active_role = "executor"
                actions, artifacts, cli_ok, meta = self.executors.run(
                    job,
                    plan_data or {},
                    run_dir,
                    self.supervision_cfg,
                    override_command=primary_cmd,
                    fallback_command=fallback_cmd,
                )
                if actions:
                    for act in actions:
                        classification, cls_reason = classify_action(
                            {"tool": act.tool, "args": act.args}
                        )
                        audit_actions.append(
                            ActionAudit(
                                name=act.tool,
                                args=act.args,
                                classification=classification,
                                allowed=True,
                                reason=cls_reason,
                            )
                        )
                self._record_attempt(
                    run_dir,
                    state,
                    "executor",
                    success=cli_ok,
                    details=None if cli_ok else {"error": "executor_fallback"},
                    meta=meta,
                )
                _append_line(notes_path, f"Executed {len(actions)} actions")
                # Guardar execution.json para que verify_run pueda inferir checks
                # Convertir objetos ToolCall y AgentArtifact a diccionarios
                execution_dict = {
                    "actions": [action.to_dict() for action in actions],
                    "artifacts": [artifact.to_dict() for artifact in artifacts],
                }
                _write_json(run_dir.root / "execution.json", execution_dict)
                state.step_index = 2
                self._update_state(run_dir, state, "VERIFY")
                continue

            if state.phase == "VERIFY":
                self._append_journal(run_dir, {"event": "phase_start", "phase": "VERIFY"})
                if not execution_data:
                    execution_data = load_json(run_dir.root / "execution.json") or {}
                if not actions:
                    actions = [
                        ToolCall.from_any(item) for item in execution_data.get("actions", [])
                    ]
                if not artifacts:
                    artifacts = [
                        AgentArtifact.from_any(item) for item in execution_data.get("artifacts", [])
                    ]

                primary_cmd, fallback_cmd, routing_snapshot = self._routing_for_role(
                    "verifier", job
                )
                _persist_routing("verifier", routing_snapshot)
                state.active_role = "verifier"
                verification_data, cli_ok, meta = self.verifier.check(
                    job,
                    actions,
                    artifacts,
                    run_dir,
                    self.supervision_cfg,
                    override_command=primary_cmd,
                    fallback_command=fallback_cmd,
                )
                success_flag = bool(verification_data.get("ok"))
                self._record_attempt(
                    run_dir,
                    state,
                    "verifier",
                    success=cli_ok and success_flag,
                    details=None if cli_ok else {"error": "verifier_fallback"},
                    meta=meta,
                )
                extra_artifacts = verification_data.get("artifacts") or []
                for item in extra_artifacts:
                    try:
                        artifact = AgentArtifact.from_any(item)
                    except ValueError:
                        continue
                    artifacts.append(artifact)
                state.step_index = 3
                self._update_state(run_dir, state, "SYNTH")
                continue

            if state.phase == "SYNTH":
                self._append_journal(run_dir, {"event": "phase_start", "phase": "SYNTH"})
                verification_data = (
                    verification_data or load_json(run_dir.root / "verification.json") or {}
                )
                report_path = run_dir.root / "artifacts" / "verifier_report.json"
                verifier_report = load_json(report_path)

                require_verified_for_final = self.policy_snapshot.config.get(
                    "require_verified_for_final", True
                )

                if verifier_report:
                    outcome = str(verifier_report.get("outcome", "failure")).lower()
                    report_errors = [str(err) for err in verifier_report.get("errors", []) if err]
                else:
                    outcome = str(verification_data.get("report", {}).get("outcome", "")) or (
                        "success" if verification_data.get("ok") else "failure"
                    )
                    outcome = outcome.lower()
                    report_errors = [str(err) for err in verification_data.get("errors", []) if err]

                allowed_success_states = (
                    {"success"} if require_verified_for_final else {"success", "neutral"}
                )
                verdict = outcome in allowed_success_states

                if not verifier_report and "success" not in outcome and not verification_data:
                    verdict = False if require_verified_for_final else True

                if outcome == "failure":
                    verdict = False

                answer_lines = [f"Objetivo: {job.goal}"]
                if brief:
                    answer_lines.append(f"Brief del consejo: {brief.get('brief', '').strip()}")
                answer_lines.append("Ejecución:")
                for action in actions:
                    answer_lines.append(f"- {action.tool}: {action.args}")
                answer_lines.append(f"Verificación: {outcome.upper()}")

                answer = "\n".join(answer_lines)
                rail = (
                    str(
                        self.policy_snapshot.config.get("rail")
                        or os.getenv("AJAX_RAIL")
                        or os.getenv("AJAX_ENV")
                        or "prod"
                    )
                    .strip()
                    .lower()
                )
                guarded_answer, guard_action, guard_receipt = _apply_anti_optimism_guard(
                    answer, rail=rail
                )
                if guard_action:
                    answer = guarded_answer
                    if guard_receipt:
                        artifacts.append(
                            AgentArtifact(
                                type="receipt",
                                path=guard_receipt,
                                description="anti_optimism_guard",
                            )
                        )
                    if guard_action == "SOFT_BLOCK":
                        verdict = False
                        errors.append("anti_optimism_soft_block")
                        errors.append("ask_user_required")

                metrics = AgentMetrics(
                    tokens_in=verification_data.get("metrics", {}).get("tokens_in", 0),
                    tokens_out=verification_data.get("metrics", {}).get("tokens_out", 0),
                    latency_ms=verification_data.get("metrics", {}).get("latency_ms", 0),
                )

                errors = report_errors
                if not verdict:
                    fallback_error = f"verification_outcome:{outcome or 'unknown'}"
                    if fallback_error not in errors:
                        errors.append(fallback_error)

                result = AgencyResult(
                    ok=verdict,
                    answer=answer,
                    actions=actions,
                    artifacts=artifacts,
                    metrics=metrics,
                    confidence=float(verification_data.get("confidence", 0.5)),
                    errors=errors,
                )
                result_payload = result.to_dict()
                self._increment_token_totals(metrics.tokens_in, metrics.tokens_out)
                audit_verification = {
                    "outcome": outcome,
                    "ok": verdict,
                    "errors": errors,
                    "report": verification_data,
                }
                audit_path = run_dir.root / "audit_log.json"
                write_audit_log(
                    audit_path,
                    mission_id=job.job_id,
                    intent=job.goal,
                    council_state=council_state,
                    infra_state=infra_state,
                    actions=audit_actions,
                    verification=audit_verification,
                    final_status="success" if verdict else "failure",
                    result_ok=verdict,
                    errors=errors,
                )
                state.step_index = 4
                self._update_state(run_dir, state, "DONE")
                self._append_journal(run_dir, {"event": "phase_end", "phase": "SYNTH"})
                _compact_file(notes_path)
                self._finalize_result(run_dir, result_payload, bucket)
                # Mindset: reflexión operativa mínima
                try:
                    impro_lines = [
                        "# Mindset Reflection",
                        f"goal: {self.mindset.get('goal', '')}",
                        f"heuristics: {', '.join(self.mindset.get('heuristics', []))}",
                        f"tone: {self.mindset.get('tone', '')}",
                        "",
                        "## Señales del run",
                        f"verdict: {'OK' if verdict else 'FAIL'}",
                        f"errors: {', '.join(errors) if errors else 'none'}",
                        f"latency_ms: {metrics.latency_ms}",
                        "",
                        "## Sugerencias (heurísticas)",
                        "- buscar inconsistencias en ruteo/verificador si FAIL",
                        "- sugerir optimizaciones (reordenar candidatos, reducir timeouts si alta latencia)",
                        "- auto-refinar rutas (quota_routing_guard) si proveedor no disponible",
                    ]
                    (run_dir.artifacts_dir / "mindset_reflection.md").write_text(
                        "\n".join(impro_lines), encoding="utf-8"
                    )
                except Exception:
                    pass
                _set_background_mode(previous_mode)
                return result

            break

        # Aborted run fallback result
        verification_data = verification_data or {}
        result = AgencyResult(
            ok=False,
            answer=f"Run {job.job_id} aborted at phase {state.phase}",
            actions=actions,
            artifacts=artifacts,
            metrics=AgentMetrics(),
            confidence=0.0,
            errors=[state.last_error or "unknown"],
        )
        result_payload = result.to_dict()
        self._append_journal(
            run_dir, {"event": "aborted", "phase": state.phase, "error": state.last_error}
        )
        audit_path = run_dir.root / "audit_log.json"
        write_audit_log(
            audit_path,
            mission_id=job.job_id,
            intent=job.goal,
            council_state=council_state,
            infra_state=infra_state,
            actions=audit_actions,
            verification=audit_verification,
            final_status="aborted",
            result_ok=False,
            errors=[state.last_error or "unknown"],
        )
        self._finalize_result(run_dir, result_payload, bucket)
        _set_background_mode(previous_mode)
        return result


def load_job(path: Path) -> AgencyJob:
    return AgencyJob.load(path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a LEANN agency job through the broker")
    parser.add_argument("job", type=Path, help="Path to job.json")
    parser.add_argument("--print-result", action="store_true", help="Print final result to stdout")

    args = parser.parse_args()

    job = load_job(args.job)
    broker = AgencyBroker()
    result = broker.run_job(job)

    if args.print_result:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
