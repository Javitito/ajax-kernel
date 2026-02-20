from __future__ import annotations

import json
import hashlib
import logging
import os
import time
import subprocess
import shutil
import selectors
import urllib.parse
import csv
import difflib
import contextlib
import random
from io import StringIO
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple, Set
import uuid
import re
import unicodedata

import requests  # type: ignore
from agency.leann_query_client import query_leann  # type: ignore
from agency.lab_control import LabStateStore, DEFAULT_PROBE_TTL_SECONDS

# Órganos existentes (manejo suave para deps opcionales)
try:
    from agency.windows_driver_client import WindowsDriverClient
except ImportError:  # pragma: no cover
    WindowsDriverClient = None  # type: ignore

try:
    from agency.actuator import Actuator, create_default_actuator
except ImportError:  # pragma: no cover
    Actuator = None  # type: ignore
    create_default_actuator = None  # type: ignore

try:
    from agency.models_registry import discover_models, list_vision_models
except ImportError:  # pragma: no cover
    discover_models = None  # type: ignore
    list_vision_models = None  # type: ignore

try:
    from agency.starting_xi import build_starting_xi, format_console_lines
except ImportError:  # pragma: no cover
    build_starting_xi = None  # type: ignore
    format_console_lines = None  # type: ignore

try:
    from agency.quota_gate import QuotaGate
except ImportError:  # pragma: no cover
    QuotaGate = None  # type: ignore
    format_console_lines = None  # type: ignore

try:
    from agency.contract import AgencyJob
except ImportError:  # pragma: no cover
    AgencyJob = None  # type: ignore

try:
    from agency.plan_runner import run_job_plan
except ImportError:  # pragma: no cover
    run_job_plan = None  # type: ignore

try:
    from agency.history import MissionHistoryRecorder
except ImportError:  # pragma: no cover
    MissionHistoryRecorder = None  # type: ignore

try:
    from agency.crystallization import CrystallizationEngine
except ImportError:  # pragma: no cover
    CrystallizationEngine = None  # type: ignore

try:
    from agency.actions_catalog import ActionCatalog
except ImportError:  # pragma: no cover
    ActionCatalog = None  # type: ignore

try:
    from agency.council import (
        Council,
        CouncilVerdict,
        is_action_allowed_in_degraded,
        load_council_state,
    )
except ImportError:  # pragma: no cover
    Council = None  # type: ignore
    CouncilVerdict = None  # type: ignore

    def load_council_state(path=None):  # type: ignore
        return {"mode": "normal", "reason": None, "timestamp": time.time()}  # type: ignore

    def is_action_allowed_in_degraded(action=None, args=None):  # type: ignore
        return False


try:
    from agency.circuit_breaker import MissionBreaker, speak_instability_alert
except ImportError:  # pragma: no cover
    MissionBreaker = None  # type: ignore

    def speak_instability_alert(kind: str) -> None:  # type: ignore
        return


try:
    from agency.success_evaluator import evaluate_success
except ImportError:  # pragma: no cover
    evaluate_success = None  # type: ignore
try:
    from agency.efe_repair import repair_plan_if_needed
except ImportError:  # pragma: no cover
    repair_plan_if_needed = None  # type: ignore
try:
    from agency.rigor_selector import decide_rigor, RigorStrategy
except ImportError:  # pragma: no cover
    decide_rigor = None  # type: ignore
    RigorStrategy = None  # type: ignore
try:
    from agency.model_router import pick_model
except ImportError:  # pragma: no cover
    pick_model = None  # type: ignore

try:
    from agency.auth_manager import AuthManager
except ImportError:  # pragma: no cover
    AuthManager = None  # type: ignore

try:
    from agency import executor
except ImportError:  # pragma: no cover
    executor = None  # type: ignore

try:
    from agency import habits as habits_mod
except ImportError:  # pragma: no cover
    habits_mod = None  # type: ignore

try:
    from agency import explorer
except ImportError:  # pragma: no cover
    explorer = None  # type: ignore

try:
    from agency.arbiter import PlanCandidate, choose_best
except ImportError:  # pragma: no cover
    PlanCandidate = None  # type: ignore
    choose_best = None  # type: ignore

try:
    from agency.brain_router import BrainRouter, TemporaryBrainFailure, BrainProviderError
except ImportError:  # pragma: no cover
    BrainRouter = None  # type: ignore
    TemporaryBrainFailure = None  # type: ignore
    BrainProviderError = None  # type: ignore
try:
    from agency.brain_router import AllBrainsFailed
except ImportError:  # pragma: no cover
    AllBrainsFailed = None  # type: ignore

try:
    from agency import leann_context
except ImportError:  # pragma: no cover
    leann_context = None  # type: ignore

try:
    from agency import system_signals
except ImportError:  # pragma: no cover
    system_signals = None  # type: ignore

try:
    from agency.mission_envelope import (
        MissionEnvelope,
        SuccessContract,
        MissionError,
        ExecutionEvent,
        Hypothesis,
        GovernanceSpec,
    )
except ImportError:  # pragma: no cover
    MissionEnvelope = None  # type: ignore
    SuccessContract = None  # type: ignore
    MissionError = None  # type: ignore
    ExecutionEvent = None  # type: ignore
    Hypothesis = None  # type: ignore
    GovernanceSpec = None  # type: ignore

try:
    from agency.brain_prompt import build_brain_prompts, mapper_system_prompt
except ImportError:  # pragma: no cover
    build_brain_prompts = None  # type: ignore
    mapper_system_prompt = None  # type: ignore
try:
    from agency.plan_normalizer import normalize_plan  # type: ignore
except ImportError:  # pragma: no cover
    normalize_plan = None  # type: ignore
try:
    from agency.legacy_plan_adapter import adapt_legacy_plan  # type: ignore
except ImportError:  # pragma: no cover
    adapt_legacy_plan = None  # type: ignore

try:
    from agency.leann_client import LeannClient
except ImportError:  # pragma: no cover
    LeannClient = None  # type: ignore
try:
    from agency.method_pack import AJAX_METHOD_PACK
except ImportError:  # pragma: no cover
    AJAX_METHOD_PACK = ""  # type: ignore
try:
    from agency.contract_enforcer import ContractEnforcer
except ImportError:  # pragma: no cover
    ContractEnforcer = None  # type: ignore

try:
    from agency.tool_inventory import load_inventory, load_heartbeat_snapshot
except ImportError:  # pragma: no cover
    load_inventory = None  # type: ignore
    load_heartbeat_snapshot = None  # type: ignore
try:
    from agency.tool_inventory import load_tool_use_notes
except ImportError:  # pragma: no cover
    load_tool_use_notes = None  # type: ignore

try:
    from agency.tool_policy import select_tool_plan
except ImportError:  # pragma: no cover
    select_tool_plan = None  # type: ignore

try:
    from agency.tool_schema import ToolPlan as PolicyToolPlan
except ImportError:  # pragma: no cover
    PolicyToolPlan = None  # type: ignore

try:
    from agency import os_inventory
except ImportError:
    os_inventory = None

try:
    from agency.security_policy import load_security_policy
except ImportError:  # pragma: no cover
    load_security_policy = None  # type: ignore

try:
    from agency.incidents import IncidentReporter  # type: ignore
except ImportError:  # pragma: no cover
    IncidentReporter = None  # type: ignore

try:
    from agency.provider_health import (
        ProviderFailure,
        ProviderFailureCode,
        classify_provider_failure,
    )
except ImportError:  # pragma: no cover
    ProviderFailure = None  # type: ignore
    ProviderFailureCode = None  # type: ignore
    classify_provider_failure = None  # type: ignore

try:
    from agency.provider_failure_policy import (
        cooldown_seconds_default as failure_cooldown_seconds_default,
        cooldown_seconds_for_reason as failure_cooldown_seconds_for_reason,
        load_provider_failure_policy,
        planning_allow_degraded_planning as failure_allow_degraded_planning,
        planning_max_attempts as failure_planning_max_attempts,
        force_ask_user_on_severe as failure_force_ask_user_on_severe,
        on_no_plan_terminal as failure_on_no_plan_terminal,
        receipt_required_fields as failure_receipt_required_fields,
    )
except ImportError:  # pragma: no cover
    failure_cooldown_seconds_default = None  # type: ignore
    failure_cooldown_seconds_for_reason = None  # type: ignore
    load_provider_failure_policy = None  # type: ignore
    failure_allow_degraded_planning = None  # type: ignore
    failure_planning_max_attempts = None  # type: ignore
    failure_force_ask_user_on_severe = None  # type: ignore
    failure_on_no_plan_terminal = None  # type: ignore
    failure_receipt_required_fields = None  # type: ignore

try:
    from agency.provider_timeouts_policy import (
        load_provider_timeouts_policy,
        policy_version as timeouts_policy_version,
    )
except ImportError:  # pragma: no cover
    load_provider_timeouts_policy = None  # type: ignore
    timeouts_policy_version = None  # type: ignore

try:
    from agency.provider_breathing import ProviderBreathingLoop
except ImportError:  # pragma: no cover
    ProviderBreathingLoop = None  # type: ignore
try:
    from agency.provider_ledger import ProviderLedger
except ImportError:  # pragma: no cover
    ProviderLedger = None  # type: ignore
try:
    from agency import provider_ranker
except ImportError:  # pragma: no cover
    provider_ranker = None  # type: ignore
try:
    from agency.policy_contract import validate_policy_contract
except ImportError:  # pragma: no cover
    validate_policy_contract = None  # type: ignore
try:
    from agency.anchor_preflight import run_anchor_preflight
except ImportError:  # pragma: no cover
    run_anchor_preflight = None  # type: ignore

try:
    from agency.display_targets import load_display_map
except ImportError:  # pragma: no cover
    load_display_map = None  # type: ignore

try:
    from agency.microfilm_guard import (
        enforce_ssc as microfilm_enforce_ssc,
        enforce_verify_before_done as microfilm_enforce_verify_before_done,
        enforce_lab_prod_separation as microfilm_enforce_lab_prod_separation,
        enforce_evidence_tiers as microfilm_enforce_evidence_tiers,
        enforce_undo_for_reversible as microfilm_enforce_undo_for_reversible,
    )
except ImportError:  # pragma: no cover
    microfilm_enforce_ssc = None  # type: ignore
    microfilm_enforce_verify_before_done = None  # type: ignore
    microfilm_enforce_lab_prod_separation = None  # type: ignore
    microfilm_enforce_evidence_tiers = None  # type: ignore
    microfilm_enforce_undo_for_reversible = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

_CORE_VERSION = "2.0"
_WAITING_MISSION_SCHEMA = "ajax.waiting_mission.v1"
_AWAIT_USER_SENTINEL = "__await_user__"
TERMINAL_STATES = {
    "COMPLETED",
    "ABORTED_BY_USER",
    "CANCELLED",
    "IRRECOVERABLY_UNSAFE",
    "TECHNICALLY_IMPOSSIBLE",
    "GAP_LOGGED",
    "DERIVED_TO_LAB_EXPERIMENT",
    "DERIVED_TO_LAB_PROBE",
    "PAUSED_FOR_LAB",
    "BLOCKED",
    "BLOCKED_BY_COUNCIL_VETO",
}

MISSION_ORCHESTRATOR_PROMPT = (
    "Eres el núcleo de control de AJAX (Mission Orchestrator).\n\n"
    "No existen subtareas.\n"
    "Solo existen tareas completas encadenadas.\n\n"
    "Una tarea solo puede delegar en otra tarea completa, nunca en un atajo.\n\n"
    "ASK_USER no finaliza la misión: transiciona a WAITING_FOR_USER y reanuda con la respuesta del usuario (misma mission_id).\n\n"
    "Ley fundamental: mientras exista UNA sola opción segura de avanzar hacia la intención del usuario "
    "— preguntar, aclarar, escalar modelo, esperar, reparar infraestructura, pedir permiso o delegar — "
    "la misión NO puede declararse fallida.\n\n"
    "Finales legítimos: COMPLETED, ABORTED_BY_USER, CANCELLED, IRRECOVERABLY_UNSAFE, TECHNICALLY_IMPOSSIBLE.\n"
    "Solo declaras un final cuando el usuario aborta, el Council marca inseguro irrecuperable, "
    "o tras intentos razonables el avance es técnicamente imposible. Tu trabajo es elegir siempre el "
    "siguiente paso seguro."
)

LAB_PROBE_TRIGGER_KEYS = {
    "selector_no_encontrado",
    "selector_no_encontrada",
    "selector_missing",
    "senales_insuficientes_efe",
    "signals_insuficientes_efe",
    "signals_insufficient_efe",
    "council_veto_condicionado",
    "council_blocked",
    "ambiguity_block",
}


@dataclass
class AjaxConfig:
    root_dir: Path
    state_dir: Path
    rag_project: str = "AJAX"
    user_id: str = "primary"

    @classmethod
    def from_env(cls) -> "AjaxConfig":
        root = Path(__file__).resolve().parents[1]
        user_id = os.getenv("AJAX_USER_ID") or os.getenv("AJAX_DEFAULT_USER") or "primary"
        return cls(root_dir=root, state_dir=root / "artifacts" / "state", user_id=user_id)


@dataclass
class AjaxStateSnapshot:
    version: str
    timestamp: float
    last_intention: Optional[str]
    last_plan_id: Optional[str] = None
    last_result_summary: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "AjaxStateSnapshot":
        return cls(_CORE_VERSION, time.time(), None, None, None, {})


@dataclass
class AjaxObservation:
    timestamp: float
    foreground: Optional[Dict[str, Any]] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AjaxPlan:
    id: Optional[str] = None
    summary: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    plan_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_spec: Optional[Dict[str, Any]] = None


@dataclass
class AjaxExecutionResult:
    success: bool
    detail: Any = None
    error: Optional[str] = None
    path: Optional[str] = None
    plan_id: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    mission_id: Optional[str] = None


class LibrarySelectionError(RuntimeError):
    def __init__(self, reason: str, *, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass
class LibrarySkillCandidate:
    skill_id: str
    kind: Literal["habit", "action"]
    label: str
    confidence: float
    plan_json: Optional[Dict[str, Any]] = None
    action_name: Optional[str] = None
    args_schema: Optional[Dict[str, Any]] = None
    habit_id: Optional[str] = None


@dataclass
class MissionInfraIssue:
    component: str
    recoverable: bool = True
    retry_in: int = 8
    detail: Optional[str] = None


@dataclass
class MissionCouncilSignal:
    verdict: Optional[str] = None
    reason: Optional[str] = None
    suggested_fix: Optional[str] = None
    escalation_hint: Optional[str] = None


@dataclass
class AskUserRequest:
    question: str
    reason: Optional[str] = None
    timeout_seconds: int = 60
    on_timeout: str = "abort"
    alert_level: str = "normal"
    escalation_trace: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "reason": self.reason,
            "timeout_seconds": self.timeout_seconds,
            "on_timeout": self.on_timeout,
            "alert_level": self.alert_level,
            "escalation_trace": self.escalation_trace or [],
        }


@dataclass
class AskUserPayload:
    question: str
    context: Dict[str, Any]
    options: List[Dict[str, str]]
    default: str = "retry_escalate_brain"
    expects: str = "menu_choice"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "context": self.context,
            "options": self.options,
            "default": self.default,
            "expects": self.expects,
        }


@dataclass
class MissionStep:
    kind: Literal[
        "EXECUTE_ACTION",
        "ASK_USER",
        "UPGRADE_MODEL",
        "FIX_INFRA",
        "WAIT_AND_RETRY",
        "REQUEST_PERMISSION",
        "LEARN_FROM_USER",
        "GIVE_UP",
    ]
    question: Optional[str] = None
    action: Optional[Any] = None
    next_model: Optional[str] = None
    final_status: Optional[str] = None
    retry_in: Optional[int] = None
    component: Optional[str] = None
    seconds: Optional[int] = None


@dataclass
class MissionState:
    intention: str
    mode: Literal["auto", "dry"] = "auto"
    status: str = "IN_PROGRESS"
    mission_id: str = field(default_factory=lambda: f"mission-{uuid.uuid4().hex[:8]}")
    started_at: float = field(default_factory=time.time)
    envelope: Optional["MissionEnvelope"] = None
    pending_plan: Optional[AjaxPlan] = None
    last_plan: Optional[AjaxPlan] = None
    last_result: Optional[AjaxExecutionResult] = None
    attempts: int = 0
    plan_attempts: int = 0
    cost_mode: Optional[str] = None
    max_attempts: int = 3
    feedback: Optional[str] = None
    infra_issue: Optional[MissionInfraIssue] = None
    council_signal: Optional[MissionCouncilSignal] = None
    needs_explicit_permission: bool = False
    permission_granted: bool = False
    permission_question: Optional[str] = None
    lab_job_id: Optional[str] = None
    waiting_cycles: int = 0
    last_user_reply: Optional[str] = None
    brain_exclude: Set[str] = field(default_factory=set)
    last_mission_error: Optional["MissionError"] = None
    retry_after: float = 0.0
    await_user_input: bool = False
    user_cancelled: bool = False
    brain_attempts: List[Dict[str, Any]] = field(default_factory=list)
    ask_user_request: Optional[AskUserRequest] = None
    app_launch_counts: Dict[str, int] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    provider_cooldowns: Dict[str, float] = field(default_factory=dict)
    provider_failures: List[Dict[str, Any]] = field(default_factory=list)
    pending_user_options: List[Dict[str, str]] = field(default_factory=list)
    brain_cost_mode_override: Optional[str] = None
    brain_retry_level: int = 0
    premium_rule: Literal["never", "if_needed", "now"] = "if_needed"
    progress_token: Optional[str] = None
    progress_token_prev: Optional[str] = None
    progress_no_change_count: int = 0
    loop_guard: bool = False

    def next_pending_action(self) -> Optional[AjaxPlan]:
        return self.pending_plan


@dataclass
class AjaxHealthReport:
    ok: bool
    driver_ok: bool
    rag_ok: bool
    models_count: int
    vision_models_count: int
    notes: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AjaxCore:
    """
    Núcleo de AJAX 2.0: inicializa órganos, carga estado y ejecuta el ciclo OODA.
    """

    _CHAT_ACTION_WORDS = (
        "abre",
        "abrir",
        "open",
        "play",
        "pon",
        "reproduce",
        "lanza",
        "inicia",
        "ejecuta",
        "busca",
        "verifica",
        "escucha",
        "escuchar",
        "cancion",
        "canción",
        "track",
        "song",
        "video",
        "música",
        "musica",
        "spotify",
        "youtube",
        "navega",
        "abre",
        "entra",
    )
    _CHAT_DESIRE_HINTS = (
        "me gustaria",
        "me gustaría",
        "me encantaria",
        "me encantaría",
        "quiero",
        "quisiera",
        "necesito",
        "me ayudas",
        "puedo pedirte",
        "tengo ganas",
        "me gustaría hacer",
        "me apetece",
    )
    _CHAT_CONFIRMATION_TOKENS = {
        "si",
        "sí",
        "vale",
        "ok",
        "okay",
        "okey",
        "dale",
        "claro",
        "hazlo",
        "adelante",
        "listo",
        "procedamos",
        "cuando quieras",
        "como siempre",
        "lo de siempre",
        "igual que siempre",
    }

    def __init__(self, config: Optional[AjaxConfig] = None):
        self.log = logging.getLogger("ajax.core")
        self.config = config or AjaxConfig.from_env()
        self.root_dir = self.config.root_dir
        self.state_dir = self.config.state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.auto_crystallize_enabled = self._load_auto_crystallize_flag()
        self.lab_control = LabStateStore(self.config.root_dir)
        self.health: Optional[AjaxHealthReport] = None
        self.provider_configs: Dict[str, Any] = self._load_provider_configs()
        self._last_brain_selection_trace: Optional[Dict[str, Any]] = None
        self.chat_history: List[Tuple[str, str]] = []  # (role, content)
        self.last_action_intent: Optional[str] = None
        self._chat_leann_cache: Dict[str, Any] = {
            "ts": 0.0,
            "key": None,
            "snippets": [],
            "had_snippets": False,
        }
        self._chat_leann_profile_cache: Dict[str, Any] = {"ts": 0.0, "key": None, "card": None}
        self._chat_lite = False
        self.default_user_id: str = (
            getattr(self.config, "user_id", None)
            or os.getenv("AJAX_DEFAULT_USER")
            or os.getenv("AJAX_USER_ID")
            or "primary"
        )
        self._chat_intent_profiles: Dict[str, Dict[str, Any]] = {}
        self._chat_user_labels: Dict[str, str] = {}
        self.council = Council(self.provider_configs) if Council else None
        self._tool_inventory: Optional[List[Any]] = None
        # Política de seguridad y parámetros del driver (antes de inicializar driver)
        default_policy = {
            "alert_level": "normal",
            "ask_user_timeout_seconds": 60,
            "ask_user_on_timeout": "abort",
            "driver_request_timeout_seconds": 5,
            "driver_failure_window_seconds": 60,
            "driver_failure_threshold": 3,
            "driver_recovery_cooldown_seconds": 20,
            "allow_app_launch": True,
            "max_app_launches_per_mission": 3,
            "per_app_limits": {},
            # Budgets de misión (pursuit)
            "mission_attempt_budget": 5,
            "mission_retry_budget_seconds": 90,
        }
        self.security_policy: Dict[str, Any] = (
            load_security_policy(self.config.root_dir) if load_security_policy else default_policy
        )
        self.alert_level: str = str(self.security_policy.get("alert_level") or "normal")
        try:
            self.mission_attempt_budget = int(
                os.getenv("AJAX_MISSION_ATTEMPT_BUDGET")
                or self.security_policy.get("mission_attempt_budget", 5)
                or 5
            )
        except Exception:
            self.mission_attempt_budget = 5
        try:
            self.mission_retry_budget_seconds = int(
                os.getenv("AJAX_MISSION_RETRY_BUDGET_SEC")
                or self.security_policy.get("mission_retry_budget_seconds", 90)
                or 90
            )
        except Exception:
            self.mission_retry_budget_seconds = 90
        try:
            self.driver_timeout = float(
                self.security_policy.get("driver_request_timeout_seconds", 5)
            )
        except Exception:
            self.driver_timeout = 5.0
        try:
            self.driver_failure_window = float(
                self.security_policy.get("driver_failure_window_seconds", 60)
            )
        except Exception:
            self.driver_failure_window = 60.0
        try:
            self.driver_failure_threshold = int(
                self.security_policy.get("driver_failure_threshold", 3)
            )
        except Exception:
            self.driver_failure_threshold = 3
        try:
            self.driver_recovery_cooldown = float(
                self.security_policy.get("driver_recovery_cooldown_seconds", 20)
            )
        except Exception:
            self.driver_recovery_cooldown = 20.0
        self._driver_cb = {
            "status": "up",
            "failures": [],
            "down_since": None,
        }  # failures = list[{"ts": float, "reason": str}]
        self.mission_breaker = MissionBreaker(self.root_dir) if MissionBreaker else None

        self.log.info("🔥 AjaxCore: Inicializando órganos...")
        self.constitution = self._load_constitution()
        self.orchestrator_prompt = MISSION_ORCHESTRATOR_PROMPT
        self.actions_catalog = self._load_actions_catalog()
        self.driver = self._init_driver()
        if (os.getenv("AJAX_SKIP_DRIVER_AUTOSTART") or "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self._ensure_driver_running()
        self.actuator = self._init_actuator()
        self.contract_enforcer = (
            ContractEnforcer(root_dir=self.config.root_dir)
            if ContractEnforcer is not None
            else None
        )
        self.rag_client = self._init_rag_client()
        if (os.getenv("AJAX_SKIP_CAPABILITIES_DISCOVERY") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            self.capabilities = {}
        else:
            self.capabilities = self._discover_capabilities()
        self.state = self._load_last_snapshot()
        self.ledger = (
            ProviderLedger(root_dir=self.config.root_dir, provider_configs=self.provider_configs)
            if ProviderLedger is not None
            else None
        )
        router_cls = self._get_brain_router_cls()
        self.brain_router = (
            router_cls.from_config(self.provider_configs, logger=self.log, ledger=self.ledger)
            if router_cls
            else None
        )
        self.mission_history = (
            MissionHistoryRecorder(self.config.root_dir, logger=self.log)
            if MissionHistoryRecorder
            else None
        )
        self._last_brain_attempts: List[Dict[str, Any]] = []
        if (os.getenv("AJAX_SKIP_HEALTH_ON_BOOT") or "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                self.health = self.health_check(strict=False)
            except Exception as exc:  # pragma: no cover
                self.log.warning("Health check inicial falló: %s", exc)
        try:
            self._check_slots_health()
        except Exception:
            pass
        self.log.info("✅ AjaxCore: Vivo y listo.")

    # --- Inicialización de órganos ---
    def _init_driver(self):
        if WindowsDriverClient is None:
            self.log.warning("Driver no disponible (WindowsDriverClient no importado)")
            return None
        try:
            return WindowsDriverClient(timeout_s=getattr(self, "driver_timeout", 5.0))
        except Exception as exc:  # pragma: no cover - dependencia externa
            self.log.error("No se pudo inicializar el driver: %s", exc)
            return None

    def _wsl_to_windows_path(self, path: str) -> str:
        if path.startswith("/mnt/"):
            parts = path.split("/")
            if len(parts) >= 4:
                drive = parts[2]
                rest = "\\".join(parts[3:])
                return f"{drive.upper()}:\\{rest}"
        return path

    def _resolve_driver_host_port(self) -> Tuple[str, int]:
        env_url = os.getenv("OS_DRIVER_URL") or ""
        if env_url:
            raw = env_url if "://" in env_url else f"http://{env_url}"
            parsed = urllib.parse.urlparse(raw)
            host = parsed.hostname or "127.0.0.1"
            try:
                port = int(parsed.port or 5010)
            except Exception:
                port = 5010
            return host, port
        env_host = os.getenv("OS_DRIVER_HOST")
        if env_host:
            try:
                port = int(os.getenv("OS_DRIVER_PORT") or 5010)
            except Exception:
                port = 5010
            return env_host, port
        return "127.0.0.1", 5010

    def _driver_health(self) -> bool:
        if not self.driver:
            return False
        if self._driver_status() == "down":
            return False
        try:
            res = self.driver.health()
            ok = bool(res.get("ok", False)) if isinstance(res, dict) else True
            if ok:
                self._driver_cb = {"status": "up", "failures": [], "down_since": None}
            return ok
        except Exception:
            self._register_driver_failure("health_failed")
            return False

    def _driver_online(self) -> bool:
        if not self.driver:
            return False
        try:
            res = self.driver.health()
            if isinstance(res, dict):
                return bool(res.get("ok", False))
            return True
        except Exception:
            return False

    def _build_plan_from_habit(self, habit: "habits.Habit") -> Dict[str, Any]:
        return {
            "plan_id": f"habit:{habit.id}",
            "steps": habit.steps,
            "success_contract": {"type": "check_last_step_status"},
            "metadata": {
                "intention": habit.intent_pattern,
                "source": "habit",
                "habit_id": habit.id,
            },
        }

    def _start_driver(self) -> bool:
        ps = shutil.which("powershell.exe") or shutil.which("powershell")
        if not ps:
            # fallback: ruta típica en Windows desde WSL
            cand = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
            if os.path.exists(cand):
                ps = cand
        if not ps:
            return False
        host, port = self._resolve_driver_host_port()
        repo_posix = str(self.config.root_dir)
        repo_win = self._wsl_to_windows_path(repo_posix).replace("/", "\\")
        vpy = os.path.join(repo_win, ".venv_os_driver", "Scripts", "python.exe")
        py_cmd = vpy if os.path.exists(vpy) else "python"
        cmd = [
            ps,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"Start-Process -WindowStyle Hidden '{py_cmd}' -ArgumentList 'drivers\\os_driver.py --host {host} --port {port}' -WorkingDirectory '{repo_win}'",
        ]
        try:
            subprocess.run(cmd, check=False, timeout=10)
            for _ in range(5):
                time.sleep(1.5)
                if self._driver_health():
                    return True
            return False
        except Exception as exc:
            self.log.warning("No se pudo arrancar el driver: %s", exc)
            return False

    def _ensure_driver_running(self) -> None:
        if self._driver_status() == "down":
            self.log.warning("Circuit breaker: driver en estado DOWN, no se intenta autostart.")
            return
        if self._driver_health():
            return
        self.log.info("Driver no responde; intentando arrancarlo...")
        ok = self._start_driver()
        if not ok:
            self.log.warning("Driver sigue no disponible tras intento de arranque.")
            # reintentar inicializar cliente por si el arranque creó el servicio
            try:
                self.driver = self._init_driver()
            except Exception:
                pass
            self._register_driver_failure("driver_unavailable")

    def _load_actions_catalog(self):
        if ActionCatalog is None:
            self.log.warning("ActionCatalog no disponible")
            return None
        try:
            catalog = ActionCatalog(self.root_dir)
            self.log.info("ActionCatalog cargado (%d acciones)", len(catalog.list_actions()))
            return catalog
        except Exception as exc:  # pragma: no cover
            self.log.warning("No se pudo cargar ActionCatalog: %s", exc)
            return None

    def _load_provider_configs(self) -> Dict[str, Any]:
        if yaml is None:
            return {}
        path = self.root_dir / "config" / "model_providers.yaml"
        if not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data
        except Exception as exc:  # pragma: no cover
            self.log.warning("No se pudo cargar model_providers.yaml: %s", exc)
            return {}

    def _get_brain_router_cls(self):
        if BrainRouter is not None:
            return BrainRouter
        try:
            from agency import brain_router as br  # type: ignore

            return getattr(br, "BrainRouter", None)
        except Exception:
            return None

    def _check_slots_health(self) -> Dict[str, Any]:
        """
        Verifica si los modelos declarados en config/model_slots.json existen en el inventario.
        Guarda el resumen en state.notes["model_slots_health"].
        """
        slots_path = self.root_dir / "config" / "model_slots.json"
        inv_path = self.root_dir / "config" / "model_inventory_cloud.json"
        if not slots_path.exists() or not inv_path.exists():
            return {"ok": False, "reason": "slots_or_inventory_missing"}
        try:
            slots = json.loads(slots_path.read_text(encoding="utf-8"))
        except Exception:
            return {"ok": False, "reason": "slots_parse_error"}
        try:
            inventory = json.loads(inv_path.read_text(encoding="utf-8"))
            inv_list = inventory.get("providers") or []
        except Exception:
            return {"ok": False, "reason": "inventory_parse_error"}
        present = {(str(item.get("provider")), str(item.get("id"))) for item in inv_list if item}
        health_entries = []
        missing = 0
        for slot, costs in slots.items():
            if not isinstance(costs, dict):
                continue
            for cost_mode, val in costs.items():
                if not isinstance(val, str) or ":" not in val:
                    continue
                provider, model_id = val.split(":", 1)
                ok = (provider, model_id) in present
                if not ok:
                    missing += 1
                health_entries.append(
                    {
                        "slot": slot,
                        "cost": cost_mode,
                        "provider": provider,
                        "model": model_id,
                        "ok": ok,
                        "reason": None if ok else "not_in_inventory",
                    }
                )
        summary = {"ok": missing == 0, "missing": missing, "entries": health_entries}
        try:
            self.state.notes["model_slots_health"] = summary
        except Exception:
            pass
        if missing > 0:
            try:
                self.log.warning("Slots sin inventario: %d", missing)
            except Exception:
                pass
        return summary

    def _slots_missing_entries(self) -> List[Dict[str, Any]]:
        """
        Devuelve lista estable de slots faltantes según state.notes["model_slots_health"].
        Cada entry: {slot,cost,provider,model,reason}.
        """
        try:
            health = (
                (self.state.notes or {}).get("model_slots_health")
                if hasattr(self, "state")
                else None
            )
        except Exception:
            health = None
        if not isinstance(health, dict):
            try:
                health = self._check_slots_health()
            except Exception:
                health = None
        if not isinstance(health, dict):
            return []
        entries = health.get("entries")
        if not isinstance(entries, list):
            return []
        out: List[Dict[str, Any]] = []
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            if ent.get("ok") is True:
                continue
            out.append(
                {
                    "slot": ent.get("slot"),
                    "cost": ent.get("cost"),
                    "provider": ent.get("provider"),
                    "model": ent.get("model"),
                    "reason": ent.get("reason") or "not_in_inventory",
                }
            )
        return out

    def _init_actuator(self):
        if create_default_actuator is None:
            self.log.warning("Actuator factory no disponible")
            return None
        try:
            return create_default_actuator()
        except Exception as exc:  # pragma: no cover
            self.log.error("No se pudo inicializar el Actuator: %s", exc)
            return None

    def _init_rag_client(self):
        if LeannClient is None:
            return None
        try:
            return LeannClient()
        except Exception:
            return None

    def _register_driver_failure(self, reason: str) -> None:
        try:
            import time

            now = time.time()
            window = getattr(self, "driver_failure_window", 60.0)
            threshold = getattr(self, "driver_failure_threshold", 3)
            failures_raw = self._driver_cb.get("failures", []) or []
            failures = []
            for item in failures_raw:
                ts = None
                if isinstance(item, dict):
                    ts = item.get("ts")
                    rs = str(item.get("reason") or "")
                else:
                    ts = item
                    rs = ""
                try:
                    ts_f = float(ts)  # type: ignore[arg-type]
                except Exception:
                    continue
                if now - ts_f <= window:
                    failures.append({"ts": ts_f, "reason": rs})
            failures.append({"ts": now, "reason": reason})
            self._driver_cb["failures"] = failures
            if len(failures) >= threshold:
                if self._driver_cb.get("status") != "down":
                    self._driver_cb["down_since"] = now
                    self._driver_cb["status"] = "down"
                    self.log.warning(
                        "Circuit breaker: driver marcado como down tras %d fallos recientes (%s)",
                        len(failures),
                        reason,
                    )
        except Exception:
            pass

    def _format_plan_timeout_message(self, detail: Dict[str, Any]) -> str:
        timeout = detail.get("timeout_seconds")
        last = detail.get("last_logged_step") or {}
        step_id = last.get("step_id") or last.get("action") or "desconocido"
        step_status = last.get("status") or "sin_estado"
        try:
            timeout_txt = f"{float(timeout):.1f}s"
        except Exception:
            timeout_txt = "tiempo límite"
        return f"El plan_runner no respondió tras {timeout_txt} (último paso {step_id} · estado={step_status})."

    def _annotate_plan_runner_result(
        self, mission: MissionState, result: AjaxExecutionResult
    ) -> None:
        if result.path != "plan_runner":
            return
        detail = result.detail if isinstance(result.detail, dict) else None
        if not detail:
            return
        status = detail.get("status")
        if status == "plan_future_timeout":
            mission.notes["planning_incomplete"] = True
            last_step = detail.get("last_logged_step") or {}
            mission.notes["planning_last_step"] = last_step
            friendly = self._format_plan_timeout_message(detail)
            result.error = friendly
            if MissionError:
                mission.last_mission_error = MissionError(
                    kind="plan_error", step_id=None, reason="plan_future_timeout"
                )
        elif status == "driver_unreachable":
            mission.notes["driver_issue"] = "driver_unreachable"
            if not mission.infra_issue or mission.infra_issue.component != "driver":
                mission.infra_issue = MissionInfraIssue(
                    component="driver", recoverable=True, retry_in=8, detail="driver_unreachable"
                )
            if MissionError:
                mission.last_mission_error = MissionError(
                    kind="infra_error", step_id=None, reason="driver_unreachable"
                )
            result.error = result.error or "driver_unreachable"
            self._register_driver_failure("driver_unreachable")

    def _driver_status(self) -> str:
        status = self._driver_cb.get("status", "unknown")
        if status == "down":
            try:
                cooldown = getattr(self, "driver_recovery_cooldown", 20.0)
                down_since = self._driver_cb.get("down_since")
                if down_since and (time.time() - float(down_since)) >= cooldown:
                    self._driver_cb = {"status": "up", "failures": [], "down_since": None}
                    status = "up"
            except Exception:
                pass
        return status

    def driver_breaker_state(self) -> Dict[str, Any]:
        try:
            return {
                "status": self._driver_status(),
                "failures": list(self._driver_cb.get("failures", [])),
                "down_since": self._driver_cb.get("down_since"),
                "window_seconds": getattr(self, "driver_failure_window", None),
                "threshold": getattr(self, "driver_failure_threshold", None),
                "cooldown_seconds": getattr(self, "driver_recovery_cooldown", None),
            }
        except Exception:
            return {"status": "unknown"}

    def _load_constitution(self):
        # Placeholder: cargar constitución/políticas si existe un loader
        try:
            const_path = self.config.root_dir / "AGENTS.md"
            if const_path.exists():
                return const_path.read_text(encoding="utf-8")
        except Exception:
            pass
        return None

    def _mission_log_dir(self) -> Path:
        path = self.config.root_dir / "artifacts" / "missions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _sanitize_filename(self, text: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
        return safe.strip("_") or "mission"

    def _safe_serialize(self, obj: Any) -> Any:
        if obj is None:
            return None
        try:
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
        except Exception:
            pass
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return {k: self._safe_serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._safe_serialize(v) for v in obj]
        if hasattr(obj, "__dict__"):
            try:
                return dict(obj.__dict__)
            except Exception:
                return str(obj)
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    def _save_mission_attempt(
        self,
        envelope: Optional["MissionEnvelope"],
        attempt: int,
        plan: Optional["AjaxPlan"],
        contract: Optional["SuccessContract"],
        result: "AjaxExecutionResult",
        evaluation: Optional[Dict[str, Any]],
        mission_error: Optional["MissionError"],
    ) -> Optional[str]:
        if envelope is None:
            return None
        try:
            snapshot0 = None
            try:
                snapshot0 = (
                    envelope.metadata.get("snapshot0")
                    if isinstance(envelope.metadata, dict)
                    else None
                )
            except Exception:
                snapshot0 = None
            payload = {
                "mission_id": envelope.mission_id,
                "intent": envelope.original_intent,
                "attempt": attempt,
                "governance": self._safe_serialize(envelope.governance),
                "ajax_layers": {
                    "comprension": "intención + knowledge_context (LEANN + signals)",
                    "inventario": "catálogo de acciones + modelos detectados",
                    "exploracion": "planificación Brain + Council con signals",
                    "ejecucion": "actuador/driver + plan_runner",
                    "aprendizaje": "logs y feedback (mission_log, success_eval)",
                },
                "evidence": {
                    "snapshot0": self._safe_serialize(snapshot0),
                },
                "plan": self._safe_serialize(plan),
                "success_contract": self._safe_serialize(contract),
                "execution_log": self._safe_serialize(envelope.execution_log),
                "execution_result": self._safe_serialize(result),
                "success_evaluation": self._safe_serialize(evaluation),
                "mission_error": self._safe_serialize(mission_error),
                "timestamp": time.time(),
            }
            log_dir = self._mission_log_dir()
            fname = f"{self._sanitize_filename(envelope.mission_id)}_attempt{attempt}.json"
            fpath = log_dir / fname
            fpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            envelope.metadata["last_mission_log"] = str(fpath)
            self.state.notes["last_mission_log"] = str(fpath)
            return str(fpath)
        except Exception as exc:
            try:
                self.log.warning("No se pudo escribir mission log: %s", exc)
            except Exception:
                pass
        return None

    @staticmethod
    def _iso_utc(ts: Optional[float] = None) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))

    def _snapshot_dir(self) -> Path:
        out = self.root_dir / "artifacts" / "missions" / "snapshots"
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return out

    def _tx_dir(self, mission_id: str) -> Path:
        out = (
            self.root_dir
            / "artifacts"
            / "missions"
            / "transactions"
            / self._sanitize_filename(str(mission_id or "mission"))
        )
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return out

    @staticmethod
    def _write_json_best_effort(path: Path, payload: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    def _tx_prepare(self, *, mission: "MissionState", plan: "AjaxPlan", attempt: int) -> Dict[str, str]:
        """
        PREPARE (transaccional): persistir un "undo plan" y un registro de transacción.
        Best-effort: nunca debe bloquear ejecución.
        """
        mission_id = str(mission.mission_id or "")
        tx_dir = self._tx_dir(mission_id)
        undo_path = tx_dir / f"undo_attempt{int(attempt)}.json"
        tx_path = tx_dir / f"tx_attempt{int(attempt)}.json"

        snap0 = None
        try:
            if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
                snap0 = mission.envelope.metadata.get("snapshot0")
        except Exception:
            snap0 = None
        snap0_path = None
        try:
            if isinstance(snap0, dict):
                sp = snap0.get("snapshot_path")
                if isinstance(sp, str) and sp.strip():
                    snap0_path = sp.strip()
        except Exception:
            snap0_path = None

        preexisting_pids = None
        try:
            if isinstance(getattr(plan, "metadata", None), dict):
                preexisting_pids = plan.metadata.get("preexisting_pids")
        except Exception:
            preexisting_pids = None

        prepare = {
            "ts": time.time(),
            "ts_utc": self._iso_utc(),
            "snapshot0_path": snap0_path,
            "preexisting_pids": preexisting_pids,
            "fg_window": (snap0 or {}).get("fg_window") if isinstance(snap0, dict) else None,
        }

        undo_plan = {
            "schema": "ajax.undo_plan.v1",
            "mission_id": mission_id,
            "attempt": int(attempt),
            "prepare": prepare,
            "actions": [
                {
                    "type": "restore_focus_snapshot0",
                    "note": "Best-effort: restore foco/ventana al foreground de snapshot0.",
                },
                {
                    "type": "taskkill_newly_launched_processes",
                    "note": "Best-effort: matar PIDs de app.launch (excluyendo preexisting_pids) si la verificación falla.",
                },
            ],
        }
        self._write_json_best_effort(undo_path, undo_plan)

        tx_doc = {
            "schema": "ajax.transaction.v1",
            "mission_id": mission_id,
            "attempt": int(attempt),
            "plan_id": getattr(plan, "plan_id", None) or getattr(plan, "id", None),
            "prepare": prepare,
            "apply": None,
            "verify": None,
            "undo": None,
            "paths": {"undo_plan": str(undo_path)},
        }
        self._write_json_best_effort(tx_path, tx_doc)

        # Attach paths for later finalize + evidence.
        try:
            plan.metadata = plan.metadata or {}
            if isinstance(plan.metadata, dict):
                plan.metadata["tx_paths"] = {"tx_path": str(tx_path), "undo_plan_path": str(undo_path)}
        except Exception:
            pass
        try:
            if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
                mission.envelope.metadata["tx_paths"] = {"tx_path": str(tx_path), "undo_plan_path": str(undo_path)}
        except Exception:
            pass
        return {"tx_path": str(tx_path), "undo_plan_path": str(undo_path)}

    def _tx_finalize(
        self,
        *,
        mission: "MissionState",
        plan: "AjaxPlan",
        attempt: int,
        result: "AjaxExecutionResult",
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        VERIFY/UNDO (transaccional): actualizar el registro con apply/verify y undo (si aplica).
        Best-effort: nunca debe bloquear.
        """
        tx_path = None
        try:
            if isinstance(getattr(plan, "metadata", None), dict):
                tx_path = (plan.metadata.get("tx_paths") or {}).get("tx_path")
        except Exception:
            tx_path = None
        if not isinstance(tx_path, str) or not tx_path.strip():
            return None
        path = Path(tx_path)
        doc: Dict[str, Any] = {}
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            doc = {
                "schema": "ajax.transaction.v1",
                "mission_id": mission.mission_id,
                "attempt": int(attempt),
            }

        apply_part = None
        try:
            apply_part = {
                "ts": time.time(),
                "ts_utc": self._iso_utc(),
                "path": result.path,
                "error": result.error,
                "detail": result.detail if isinstance(result.detail, dict) else None,
            }
        except Exception:
            apply_part = None

        verify_part = None
        try:
            verify_part = {
                "ts": time.time(),
                "ts_utc": self._iso_utc(),
                "ok": bool(result.success),
                "evaluation": evaluation if isinstance(evaluation, dict) else None,
            }
        except Exception:
            verify_part = None

        undo_part = None
        if not bool(result.success):
            cleanup = None
            try:
                cleanup = (mission.notes or {}).get("cleanup") if isinstance(mission.notes, dict) else None
            except Exception:
                cleanup = None
            if cleanup is None:
                try:
                    if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
                        cleanup = mission.envelope.metadata.get("cleanup")
                except Exception:
                    cleanup = None
            undo_part = {
                "ts": time.time(),
                "ts_utc": self._iso_utc(),
                "performed": bool(cleanup is not None),
                "cleanup": cleanup,
            }

        doc["apply"] = apply_part
        doc["verify"] = verify_part
        doc["undo"] = undo_part
        self._write_json_best_effort(path, doc)
        return str(path)

    def _tasklist_snapshot(self, processes: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        uniq = []
        seen = set()
        for p in processes:
            p = str(p or "").strip()
            if not p or p.lower() in seen:
                continue
            seen.add(p.lower())
            uniq.append(p)
        for proc_name in uniq:
            try:
                proc = subprocess.run(
                    ["tasklist.exe", "/FO", "CSV", "/NH", "/FI", f"IMAGENAME eq {proc_name}"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                out[proc_name] = {
                    "rc": proc.returncode,
                    "stdout": (proc.stdout or "").strip()[:2000],
                    "stderr": (proc.stderr or "").strip()[:2000],
                }
            except Exception as exc:
                out[proc_name] = {"error": str(exc)[:200]}
        return out

    @staticmethod
    def _normalize_image_name(raw: Optional[str]) -> Optional[str]:
        """
        Normaliza un process/path a "foo.exe" (lower) para tasklist/taskkill.
        """
        s = str(raw or "").strip()
        if not s:
            return None
        s = s.replace("\\", "/")
        base = os.path.basename(s).strip()
        if not base:
            return None
        if "." not in base:
            base = f"{base}.exe"
        return base.lower()

    def _tasklist_pids(self, image_name: str) -> Set[int]:
        """
        Devuelve PIDs actuales para un IMAGENAME (best-effort).
        """
        pids: Set[int] = set()
        name = (image_name or "").strip()
        if not name:
            return pids
        try:
            proc = subprocess.run(
                ["tasklist.exe", "/FO", "CSV", "/NH", "/FI", f"IMAGENAME eq {name}"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if proc.returncode != 0:
                return pids
            txt = (proc.stdout or "").strip()
            if not txt or "INFO:" in txt.upper():
                return pids
            for row in csv.reader(StringIO(txt)):
                if not row or len(row) < 2:
                    continue
                try:
                    pid = int(str(row[1]).strip())
                except Exception:
                    continue
                if pid > 0:
                    pids.add(pid)
        except Exception:
            return pids
        return pids

    def _collect_preexisting_pids(self, plan: AjaxPlan) -> Dict[str, List[int]]:
        """
        Snapshot minimal por proceso objetivo (para cleanup): {image_name: [pid,...]}.
        """
        targets: Set[str] = set()
        for st in plan.steps or []:
            if not isinstance(st, dict):
                continue
            if str(st.get("action") or "").strip() != "app.launch":
                continue
            args = st.get("args") if isinstance(st.get("args"), dict) else {}
            proc = args.get("process")
            path = args.get("path")
            name = self._normalize_image_name(path or proc)
            if name:
                targets.add(name)
        out: Dict[str, List[int]] = {}
        for name in sorted(targets):
            out[name] = sorted(self._tasklist_pids(name))
        return out

    def _cleanup_after_failure(
        self, mission: MissionState, plan: AjaxPlan, result: AjaxExecutionResult
    ) -> None:
        """
        Cleanup/orden mínimo:
        - Cerrar (taskkill) procesos lanzados por la misión cuando sea posible (PID tracking).
        - Restaurar foco/entorno hacia Snapshot-0 (best-effort).
        """
        preexisting = {}
        try:
            preexisting = (plan.metadata or {}).get("preexisting_pids") if plan.metadata else {}
        except Exception:
            preexisting = {}
        preexisting_map: Dict[str, Set[int]] = {}
        if isinstance(preexisting, dict):
            for k, v in preexisting.items():
                if not isinstance(k, str) or not isinstance(v, list):
                    continue
                pids = set()
                for item in v:
                    try:
                        pid = int(item)
                    except Exception:
                        continue
                    if pid > 0:
                        pids.add(pid)
                preexisting_map[k.lower()] = pids

        steps_detail = []
        try:
            if result.path == "plan_runner" and isinstance(result.detail, dict):
                steps_detail = (
                    result.detail.get("steps")
                    if isinstance(result.detail.get("steps"), list)
                    else []
                )
        except Exception:
            steps_detail = []

        launched: List[Dict[str, Any]] = []
        killed: List[int] = []
        for st in steps_detail:
            if not isinstance(st, dict):
                continue
            det = st.get("detail")
            if not isinstance(det, dict):
                continue
            if str(det.get("action") or "").strip() != "app.launch":
                continue
            if bool(det.get("skipped_launch")):
                continue
            resp = det.get("result") if isinstance(det.get("result"), dict) else {}
            pid_raw = resp.get("pid") if isinstance(resp, dict) else None
            try:
                pid = int(pid_raw) if pid_raw is not None else None
            except Exception:
                pid = None
            image = self._normalize_image_name(
                det.get("path")
                or det.get("process")
                or (resp.get("process") if isinstance(resp, dict) else None)
            )
            launched.append({"image": image, "pid": pid})
            if not pid or pid <= 0 or not image:
                continue
            pre = preexisting_map.get(image.lower(), set())
            if pid in pre:
                continue
            try:
                tk = subprocess.run(
                    ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if tk.returncode == 0:
                    killed.append(pid)
            except Exception:
                pass

        # Restaurar foco hacia Snapshot-0 (best-effort).
        restored_focus = False
        try:
            snap0 = (
                mission.envelope.metadata.get("snapshot0")
                if mission.envelope
                and isinstance(getattr(mission.envelope, "metadata", None), dict)
                else None
            )
        except Exception:
            snap0 = None
        try:
            fg = (snap0 or {}).get("fg_window") if isinstance(snap0, dict) else None
            proc0 = (fg or {}).get("process") if isinstance(fg, dict) else None
            title0 = (fg or {}).get("title") if isinstance(fg, dict) else None
            if self.driver and proc0:
                title_contains = str(title0 or "").strip()
                if title_contains:
                    title_contains = title_contains[:80]
                self.driver.window_focus(process=str(proc0), title_contains=title_contains or None)
                restored_focus = True
        except Exception:
            restored_focus = False

        cleanup_summary = {
            "launched": launched,
            "killed_pids": killed,
            "restored_focus": restored_focus,
        }
        try:
            mission.notes["cleanup"] = cleanup_summary
        except Exception:
            pass
        try:
            if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
                mission.envelope.metadata["cleanup"] = cleanup_summary
        except Exception:
            pass

    def _capture_snapshot0(self, envelope: "MissionEnvelope") -> Dict[str, Any]:
        """
        Snapshot-0 obligatorio: inventario inicial antes de cualquier acción.
        Mejor-esfuerzo (no debe bloquear la misión si falla).
        """
        ts = time.time()
        snap: Dict[str, Any] = {
            "ts": ts,
            "utc": self._iso_utc(ts),
            "rail": (
                os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"
            ),
            "cost_mode": os.getenv("AJAX_COST_MODE"),
            "driver": {"available": bool(self.driver)},
            "driver_health": None,
            "fg_window": None,
            "screenshot_path": None,
            "ui_inspect": None,
            "processes": None,
        }
        if not self.driver:
            snap["driver"]["available"] = False
        else:
            try:
                snap["driver_health"] = self.driver.health()
            except Exception as exc:
                snap["driver_health"] = {"ok": False, "error": str(exc)}
            try:
                snap["fg_window"] = self.driver.get_active_window()
            except Exception as exc:
                snap["fg_window"] = {"error": str(exc)}
            try:
                shot = self.driver.screenshot()
                snap["screenshot_path"] = str(shot)
            except Exception as exc:
                snap["screenshot_path"] = None
                snap["driver"].setdefault("errors", []).append(f"screenshot:{exc}")
            try:
                snap["ui_inspect"] = self.driver.inspect_window()
            except Exception as exc:
                snap["ui_inspect"] = {"ok": False, "error": str(exc)}
            try:
                fg_proc = str(
                    (
                        (snap.get("fg_window") or {}).get("process")
                        if isinstance(snap.get("fg_window"), dict)
                        else ""
                    )
                    or ""
                )
                proc_candidates = [
                    p
                    for p in [fg_proc, "brave.exe", "chrome.exe", "msedge.exe", "firefox.exe"]
                    if p
                ]
                snap["processes"] = self._tasklist_snapshot(proc_candidates)
            except Exception:
                snap["processes"] = None

        # Persistir snapshot a disco para evidencia durable
        try:
            path = self._snapshot_dir() / f"{envelope.mission_id}_snapshot0.json"
            path.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            snap["snapshot_path"] = str(path)
        except Exception:
            pass
        return snap

    def _discover_capabilities(self) -> Dict[str, Any]:
        caps: Dict[str, Any] = {}
        if discover_models:
            try:
                caps["models_all"] = [m.__dict__ for m in discover_models()]
            except Exception as exc:
                self.log.warning("No se pudieron descubrir modelos (all): %s", exc)
        if list_vision_models:
            try:
                caps["models_vision"] = [m.__dict__ for m in list_vision_models()]
            except Exception as exc:
                self.log.warning("No se pudieron descubrir modelos de visión: %s", exc)
        # Capability flag (micro-note): permiso humano opcional para gates (no asume que esté cableado)
        try:
            caps["human_permission_switch"] = (
                os.getenv("AJAX_HUMAN_PERMISSION_SWITCH") or ""
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        except Exception:
            caps["human_permission_switch"] = False
        return caps

    # --- Estado ---
    def _load_last_snapshot(self) -> AjaxStateSnapshot:
        path = self.config.state_dir / "ajax_state_latest.json"
        if path.exists():
            try:
                return AjaxStateSnapshot(**json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                self.log.warning("Snapshot corrupto, usando vacío: %s", exc)
        return AjaxStateSnapshot.empty()

    def snapshot(self, note: Optional[str] = None) -> None:
        if note:
            self.state.notes["note"] = note
        self.state.timestamp = time.time()
        path = self.config.state_dir / "ajax_state_latest.json"
        path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")

    def _waiting_mission_path(self) -> Path:
        return self.config.state_dir / "waiting_mission.json"

    def _load_waiting_mission_payload(self) -> Optional[Dict[str, Any]]:
        path = self._waiting_mission_path()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if str(payload.get("schema") or "").strip() != _WAITING_MISSION_SCHEMA:
            return None
        if "pending" in payload and payload.get("pending") is None:
            return None
        status = str(payload.get("status") or "").strip().upper()
        if (
            status == "PAUSED_BY_USER"
            or bool(payload.get("paused_by_user"))
            or bool(payload.get("parked_by_user"))
        ):
            return None
        return payload

    def _clear_waiting_mission(
        self,
        mission_id: Optional[str] = None,
        *,
        cancelled: bool = False,
        cancel_reason: Optional[str] = None,
    ) -> None:
        path = self._waiting_mission_path()
        try:
            if path.exists():
                path.unlink()
        except Exception:
            return
        if mission_id:
            try:
                payload_dir = self.config.root_dir / "artifacts" / "waiting_for_user"
                payload_path = payload_dir / f"{mission_id}.json"
                if payload_path.exists():
                    try:
                        doc = json.loads(payload_path.read_text(encoding="utf-8"))
                    except Exception:
                        doc = None
                    if isinstance(doc, dict):
                        doc["consumed_utc"] = self._iso_utc()
                        if cancelled:
                            doc["cancelled"] = True
                            doc["cancelled_utc"] = self._iso_utc()
                            if cancel_reason:
                                doc["cancel_reason"] = str(cancel_reason)
                        payload_path.write_text(
                            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                        )
            except Exception:
                pass

    def _persist_waiting_mission(
        self,
        mission: MissionState,
        *,
        question: str,
        user_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Persist a non-terminal mission that is waiting on user input so it can be resumed later.
        """
        expects = None
        try:
            if isinstance(user_payload, dict):
                expects = user_payload.get("expects")
        except Exception:
            expects = None
        payload_doc: Dict[str, Any] = {
            "schema": _WAITING_MISSION_SCHEMA,
            "saved_utc": self._iso_utc(),
            "last_prompted_at": self._iso_utc(),
            "mission_id": mission.mission_id,
            "intention": mission.intention,
            "mode": mission.mode,
            "started_at": float(mission.started_at or time.time()),
            "status": str(mission.status or "WAITING_FOR_USER"),
            "parked_by_user": bool((mission.notes or {}).get("parked_by_user")),
            "paused_by_user": bool(
                (mission.notes or {}).get("paused_by_user")
                or (mission.notes or {}).get("parked_by_user")
            ),
            "question": question,
            "pending_question": question,
            "loop_guard": bool(getattr(mission, "loop_guard", False)),
            "progress_token": getattr(mission, "progress_token", None),
            "progress_token_prev": getattr(mission, "progress_token_prev", None),
            "progress_no_change_count": int(getattr(mission, "progress_no_change_count", 0) or 0),
            "ask_user_request": mission.ask_user_request.to_dict()
            if mission.ask_user_request
            else None,
            "mission": {
                "attempts": int(mission.attempts or 0),
                "plan_attempts": int(mission.plan_attempts or 0),
                "max_attempts": int(mission.max_attempts or 0),
                "cost_mode": mission.cost_mode,
                "waiting_cycles": int(getattr(mission, "waiting_cycles", 0) or 0),
                "brain_exclude": sorted([str(x) for x in (mission.brain_exclude or set())]),
                "brain_attempts": self._safe_serialize(
                    getattr(mission, "brain_attempts", None) or []
                ),
                "brain_cost_mode_override": mission.brain_cost_mode_override,
                "brain_retry_level": int(getattr(mission, "brain_retry_level", 0) or 0),
                "premium_rule": mission.premium_rule,
                "feedback": mission.feedback,
                "infra_issue": self._safe_serialize(mission.infra_issue)
                if mission.infra_issue
                else None,
                "council_signal": self._safe_serialize(mission.council_signal)
                if mission.council_signal
                else None,
                "needs_explicit_permission": bool(mission.needs_explicit_permission),
                "permission_granted": bool(mission.permission_granted),
                "permission_question": mission.permission_question,
                "lab_job_id": mission.lab_job_id,
                "app_launch_counts": self._safe_serialize(mission.app_launch_counts or {}),
                "notes": self._safe_serialize(mission.notes or {}),
                "loop_guard": bool(getattr(mission, "loop_guard", False)),
                "progress_token": getattr(mission, "progress_token", None),
                "progress_token_prev": getattr(mission, "progress_token_prev", None),
                "progress_no_change_count": int(
                    getattr(mission, "progress_no_change_count", 0) or 0
                ),
                "pending_plan": self._safe_serialize(mission.pending_plan)
                if mission.pending_plan
                else None,
                "last_plan": self._safe_serialize(mission.last_plan) if mission.last_plan else None,
                "pending_user_options": mission.pending_user_options or [],
            },
            "envelope": self._safe_serialize(mission.envelope.to_dict())
            if mission.envelope and hasattr(mission.envelope, "to_dict")
            else None,
            "default_option": (user_payload or {}).get("default")
            if isinstance(user_payload, dict)
            else None,
            "default_option_id": (user_payload or {}).get("default")
            if isinstance(user_payload, dict)
            else None,
            "expects": expects
            or ("menu_choice" if mission.pending_user_options else "user_answer"),
            "pending_default_confirm": False,
            "ask_user_payload": user_payload if isinstance(user_payload, dict) else None,
        }
        try:
            user_payload_path: Optional[Path] = None
            payload_clone = user_payload.copy() if isinstance(user_payload, dict) else None
            if payload_clone:
                payload_dir = self.config.root_dir / "artifacts" / "waiting_for_user"
                payload_dir.mkdir(parents=True, exist_ok=True)
                user_payload_path = payload_dir / f"{mission.mission_id}.json"
                payload_record = {
                    "mission_id": mission.mission_id,
                    "intention": mission.intention,
                    "pending_question": question,
                    "expects": expects
                    or ("menu_choice" if mission.pending_user_options else "user_answer"),
                    "default_option_id": (payload_clone or {}).get("default"),
                    "last_prompted_at": self._iso_utc(),
                    "pending_default_confirm": False,
                    "payload": payload_clone,
                }
                user_payload_path.write_text(
                    json.dumps(payload_record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                mission.notes["_waiting_payload_path"] = str(user_payload_path)
            path = self._waiting_mission_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                if getattr(self, "lab_control", None):
                    self.lab_control.sync_foreground_from_waiting()
            except Exception:
                pass
            return str(path)
        except Exception:
            return None

    def _record_waiting_payload_escalation(self, mission: MissionState, *, option_id: str) -> None:
        payload_path = None
        try:
            if isinstance(mission.notes, dict):
                payload_path = mission.notes.get("_waiting_payload_path")
        except Exception:
            payload_path = None
        if not payload_path:
            try:
                payload_path = str(
                    self.config.root_dir
                    / "artifacts"
                    / "waiting_for_user"
                    / f"{mission.mission_id}.json"
                )
            except Exception:
                payload_path = None
        if not payload_path:
            return
        path = Path(str(payload_path))
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
        sx_path = None
        sx_inputs = None
        try:
            if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
                sx_path = mission.envelope.metadata.get("starting_xi_path")
        except Exception:
            sx_path = None
        try:
            if not sx_path and isinstance(mission.notes, dict):
                sx = mission.notes.get("starting_xi")
                if isinstance(sx, dict):
                    sx_path = sx.get("path")
                    sx_inputs = sx.get("inputs")
        except Exception:
            sx_inputs = None
        roster_refs: List[str] = []
        if isinstance(sx_path, str) and sx_path:
            roster_refs.append(sx_path)
        if isinstance(sx_inputs, dict):
            lp = sx_inputs.get("provider_ledger_path")
            if isinstance(lp, str) and lp:
                roster_refs.append(lp)
        payload["escalation"] = {
            "selected_option": option_id,
            "cost_mode": mission.cost_mode,
            "brain_cost_mode_override": mission.brain_cost_mode_override,
            "starting_xi_path": sx_path,
            "roster_refs": roster_refs or None,
            "updated_utc": self._iso_utc(),
        }
        raw["payload"] = payload
        raw["updated_utc"] = self._iso_utc()
        try:
            path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            return

    def _mission_from_waiting_payload(self, payload: Dict[str, Any]) -> MissionState:
        mission_raw = payload.get("mission") if isinstance(payload.get("mission"), dict) else {}
        mission_id = str(payload.get("mission_id") or mission_raw.get("mission_id") or "")
        if not mission_id:
            raise ValueError("waiting_mission missing mission_id")
        intention = str(payload.get("intention") or mission_raw.get("intention") or "")
        if not intention:
            raise ValueError("waiting_mission missing intention")
        mode = str(payload.get("mode") or mission_raw.get("mode") or "auto")
        if mode not in {"auto", "dry"}:
            mode = "auto"

        envelope_obj = None
        env_raw = payload.get("envelope")
        if env_raw and MissionEnvelope is not None and hasattr(MissionEnvelope, "from_dict"):
            try:
                envelope_obj = MissionEnvelope.from_dict(env_raw)  # type: ignore[attr-defined]
            except Exception:
                envelope_obj = None

        mission = MissionState(
            intention=intention,
            mode=mode,  # type: ignore[arg-type]
            envelope=envelope_obj,
            mission_id=mission_id,
        )
        # Importante: cargar notes temprano, porque contiene opciones pendientes y contexto de WAITING_FOR_USER.
        try:
            notes_raw = mission_raw.get("notes")
            if isinstance(notes_raw, dict):
                mission.notes = dict(notes_raw)
        except Exception:
            pass
        try:
            mission.started_at = float(
                payload.get("started_at") or mission_raw.get("started_at") or mission.started_at
            )
        except Exception:
            pass
        try:
            mission.attempts = int(mission_raw.get("attempts") or 0)
        except Exception:
            pass
        try:
            raw_plan_attempts = mission_raw.get("plan_attempts")
            if raw_plan_attempts is None:
                raw_plan_attempts = mission_raw.get("attempts") or 0
            mission.plan_attempts = int(raw_plan_attempts or 0)
        except Exception:
            mission.plan_attempts = 0
        try:
            cm = mission_raw.get("cost_mode")
            mission.cost_mode = str(cm).strip().lower() if cm else None
        except Exception:
            mission.cost_mode = None
        try:
            mission.max_attempts = int(mission_raw.get("max_attempts") or mission.max_attempts)
        except Exception:
            pass
        try:
            mission.waiting_cycles = int(mission_raw.get("waiting_cycles") or 0)
        except Exception:
            mission.waiting_cycles = 0
        try:
            mission.feedback = mission_raw.get("feedback")
        except Exception:
            pass
        try:
            bex = mission_raw.get("brain_exclude") or []
            if isinstance(bex, list):
                mission.brain_exclude = set([str(x) for x in bex if str(x)])
        except Exception:
            pass
        try:
            batt = mission_raw.get("brain_attempts") or []
            if isinstance(batt, list):
                mission.brain_attempts = batt  # type: ignore[assignment]
        except Exception:
            pass
        try:
            mission.brain_cost_mode_override = mission_raw.get("brain_cost_mode_override")
        except Exception:
            mission.brain_cost_mode_override = None
        try:
            mission.brain_retry_level = int(mission_raw.get("brain_retry_level") or 0)
        except Exception:
            mission.brain_retry_level = 0
        try:
            rule = str(mission_raw.get("premium_rule") or "").strip().lower()
            if rule in {"never", "if_needed", "now"}:
                mission.premium_rule = rule  # type: ignore[assignment]
        except Exception:
            mission.premium_rule = "if_needed"
        try:
            mission.notes = mission_raw.get("notes") or {}
        except Exception:
            pass
        try:
            mission.progress_token = mission_raw.get("progress_token") or payload.get(
                "progress_token"
            )
        except Exception:
            mission.progress_token = None
        try:
            mission.progress_token_prev = mission_raw.get("progress_token_prev") or payload.get(
                "progress_token_prev"
            )
        except Exception:
            mission.progress_token_prev = None
        try:
            mission.progress_no_change_count = int(
                mission_raw.get("progress_no_change_count")
                or payload.get("progress_no_change_count")
                or 0
            )
        except Exception:
            mission.progress_no_change_count = 0
        try:
            raw_loop_guard = mission_raw.get("loop_guard")
            if raw_loop_guard is None:
                raw_loop_guard = payload.get("loop_guard")
            mission.loop_guard = bool(raw_loop_guard)
        except Exception:
            mission.loop_guard = False
        try:
            lab_outcome = payload.get("lab_outcome")
            if isinstance(lab_outcome, dict):
                if not isinstance(mission.notes, dict):
                    mission.notes = {}
                mission.notes["lab_outcome"] = lab_outcome
                if not mission.feedback:
                    outcome = lab_outcome.get("outcome") or ""
                    summary = lab_outcome.get("summary") or ""
                    refs = lab_outcome.get("evidence_refs") or []
                    refs_txt = ""
                    if isinstance(refs, list) and refs:
                        refs_txt = f" refs={refs[:3]}"
                    mission.feedback = (
                        f"[LAB_OUTCOME] outcome={outcome} summary={summary}{refs_txt}".strip()
                    )
        except Exception:
            pass
        try:
            options = (
                mission_raw.get("pending_user_options")
                or mission.notes.get("_pending_user_options")
                if isinstance(mission.notes, dict)
                else []
            )
            if isinstance(options, list):
                mission.pending_user_options = [opt for opt in options if isinstance(opt, dict)]
        except Exception:
            mission.pending_user_options = []
        try:
            options, info = self._apply_lab_stale_options(
                list(mission.pending_user_options or []),
                job_id=mission.lab_job_id,
            )
            mission.pending_user_options = options
            if isinstance(mission.notes, dict):
                mission.notes["_pending_user_options"] = options
                if info and info.get("is_stale"):
                    mission.notes["lab_stale_reason"] = info.get("reason")
                    mission.notes["lab_stale_detected_ts"] = self._iso_utc()
        except Exception:
            pass
        try:
            if isinstance(mission_raw.get("infra_issue"), dict):
                mission.infra_issue = MissionInfraIssue(**mission_raw.get("infra_issue"))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            if isinstance(mission_raw.get("council_signal"), dict):
                mission.council_signal = MissionCouncilSignal(**mission_raw.get("council_signal"))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            mission.needs_explicit_permission = bool(mission_raw.get("needs_explicit_permission"))
            mission.permission_granted = bool(mission_raw.get("permission_granted"))
            mission.permission_question = mission_raw.get("permission_question")
        except Exception:
            pass
        try:
            mission.app_launch_counts = mission_raw.get("app_launch_counts") or {}
        except Exception:
            pass
        try:
            mission.lab_job_id = mission_raw.get("lab_job_id")
        except Exception:
            mission.lab_job_id = None
        try:
            if isinstance(mission_raw.get("pending_plan"), dict):
                mission.pending_plan = AjaxPlan(**mission_raw.get("pending_plan"))  # type: ignore[arg-type]
        except Exception:
            mission.pending_plan = None
        try:
            if isinstance(mission_raw.get("last_plan"), dict):
                mission.last_plan = AjaxPlan(**mission_raw.get("last_plan"))  # type: ignore[arg-type]
        except Exception:
            mission.last_plan = None
        try:
            aur = payload.get("ask_user_request")
            if isinstance(aur, dict):
                mission.ask_user_request = AskUserRequest(
                    question=str(aur.get("question") or ""),
                    reason=aur.get("reason"),
                    timeout_seconds=int(aur.get("timeout_seconds") or 60),
                    on_timeout=str(aur.get("on_timeout") or "abort"),
                    alert_level=str(aur.get("alert_level") or "normal"),
                    escalation_trace=aur.get("escalation_trace"),
                )
        except Exception:
            mission.ask_user_request = None
        return mission

    def _waiting_payload_summary(
        self, payload: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        mission_id = str(payload.get("mission_id") or "").strip()
        question = str(
            payload.get("pending_question")
            or payload.get("question")
            or mission_id
            or "Pregunta pendiente"
        ).strip()
        question_lc = question.lower()
        mission_raw = payload.get("mission") if isinstance(payload.get("mission"), dict) else {}
        raw_opts = (
            mission_raw.get("pending_user_options") if isinstance(mission_raw, dict) else None
        )
        options: List[Dict[str, str]] = []
        if isinstance(raw_opts, list):
            for opt in raw_opts:
                if not isinstance(opt, dict):
                    continue
                opt_id = str(opt.get("id") or "").strip()
                if not opt_id:
                    continue
                options.append(
                    {
                        "id": opt_id,
                        "label": str(opt.get("label") or "").strip(),
                    }
                )
        ask_payload = (
            payload.get("ask_user_payload")
            if isinstance(payload.get("ask_user_payload"), dict)
            else None
        )
        default_option = None
        expects = None
        if isinstance(ask_payload, dict):
            default_option = ask_payload.get("default")
            expects = ask_payload.get("expects")
        else:
            default_option = payload.get("default_option") or payload.get("default_option_id")
        if not expects:
            expects = payload.get("expects")
        if not expects:
            expects = "menu_choice" if options else "user_answer"
        ctx = (ask_payload or {}).get("context") if isinstance(ask_payload, dict) else {}
        budget_exhausted = bool(ctx.get("budget_exhausted")) if isinstance(ctx, dict) else False
        if ("[budget_exhausted]" in question_lc) or ("budget_exhausted" in question_lc):
            budget_exhausted = True
        if budget_exhausted:
            expects = "user_answer"
            default_option = "open_incident"
            filtered: List[Dict[str, str]] = []
            seen_ids: set[str] = set()
            for opt in options:
                opt_id = str(opt.get("id") or "").strip().lower()
                if opt_id in {
                    "retry_escalate_brain",
                    "use_deterministic_recipe",
                    "use_premium_now",
                }:
                    continue
                if opt_id in {"open_incident", "close_manual_done", "retry_fresh"}:
                    if opt_id not in seen_ids:
                        filtered.append(opt)
                        seen_ids.add(opt_id)
            if "open_incident" not in seen_ids:
                filtered.append(
                    {
                        "id": "open_incident",
                        "label": "Abrir INCIDENT y derivar a LAB para triage automático",
                    }
                )
            if "close_manual_done" not in seen_ids:
                filtered.append(
                    {"id": "close_manual_done", "label": "Cerrar misión manualmente (ya realizado)"}
                )
            if "retry_fresh" not in seen_ids:
                filtered.append(
                    {
                        "id": "retry_fresh",
                        "label": "Reintentar como nueva misión (reset de presupuesto)",
                    }
                )
            options = filtered
        return {
            "mission_id": mission_id,
            "question": question or "Pregunta pendiente",
            "options": options,
            "default_option": default_option,
            "expects": expects,
            "context": ctx if isinstance(ctx, dict) else None,
        }

    def _lab_stale_option_pack(self) -> List[Dict[str, str]]:
        return [
            {"id": "cancel_job", "label": "Cancelar job LAB (terminal)"},
            {"id": "requeue_job", "label": "Reencolar job LAB con nuevo id"},
            {"id": "open_incident", "label": "Abrir INCIDENT para triage"},
        ]

    @staticmethod
    def _merge_option_lists(
        options: List[Dict[str, str]], extra: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        merged: List[Dict[str, str]] = []
        seen: set[str] = set()
        for opt in options + extra:
            if not isinstance(opt, dict):
                continue
            opt_id = str(opt.get("id") or "").strip().lower()
            if not opt_id or opt_id in seen:
                continue
            seen.add(opt_id)
            merged.append(opt)
        return merged

    def _load_lab_stale_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        store = getattr(self, "lab_control", None)
        if not store:
            return None
        try:
            job, path = store.load_job(job_id)
            return store.annotate_job_staleness(job, path)
        except Exception:
            return None

    def _apply_lab_stale_options(
        self,
        options: List[Dict[str, str]],
        *,
        job_id: Optional[str],
    ) -> tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]:
        if not job_id:
            return options, None
        info = self._load_lab_stale_info(str(job_id))
        if not info or not info.get("is_stale"):
            return options, info
        merged = self._merge_option_lists(options, self._lab_stale_option_pack())
        return merged, info

    def evaluate_waiting_user_reply(self, user_reply: str) -> Optional[Dict[str, Any]]:
        payload = self._load_waiting_mission_payload()
        summary = self._waiting_payload_summary(payload)
        if not summary:
            return None
        job_id = None
        try:
            if isinstance(payload, dict):
                mission_raw = (
                    payload.get("mission") if isinstance(payload.get("mission"), dict) else {}
                )
                job_id = mission_raw.get("lab_job_id") or payload.get("lab_job_id")
        except Exception:
            job_id = None
        try:
            options = summary.get("options") or []
            options, info = self._apply_lab_stale_options(options, job_id=job_id)
            summary["options"] = options
            if info and info.get("is_stale"):
                summary["lab_stale"] = {
                    "reason": info.get("reason"),
                    "job_age_s": info.get("job_age_s"),
                    "heartbeat_age_s": info.get("heartbeat_age_s"),
                }
        except Exception:
            pass
        normalized = self._normalize_waiting_reply_text(user_reply)
        raw = (user_reply or "").strip()
        options = summary.get("options") or []
        expects = str(summary.get("expects") or "").strip().lower() or "menu_choice"
        option_ids = {opt["id"].strip().lower(): opt for opt in options if opt.get("id")}
        if options:
            if normalized and normalized in option_ids:
                return {
                    "status": "resume",
                    "mission_id": summary["mission_id"],
                    "question": summary["question"],
                    "options": options,
                    "default_option": summary.get("default_option"),
                    "expects": expects,
                }
            if raw and expects == "user_answer":
                return {
                    "status": "resume",
                    "mission_id": summary["mission_id"],
                    "question": summary["question"],
                    "options": options,
                    "default_option": summary.get("default_option"),
                    "expects": expects,
                }
            return {
                "status": "invalid",
                "mission_id": summary["mission_id"],
                "question": summary["question"],
                "options": options,
                "default_option": summary.get("default_option"),
                "expects": expects,
            }
        if normalized or raw:
            return {
                "status": "resume",
                "mission_id": summary["mission_id"],
                "question": summary["question"],
                "options": [],
                "default_option": summary.get("default_option"),
                "expects": expects,
            }
        return {
            "status": "invalid",
            "mission_id": summary["mission_id"],
            "question": summary["question"],
            "options": [],
            "default_option": summary.get("default_option"),
            "expects": expects,
        }

    @staticmethod
    def _normalize_waiting_reply_text(user_reply: Optional[str]) -> str:
        if not user_reply:
            return ""
        txt = str(user_reply).strip().lower()
        match = re.fullmatch(r"\[(.+)\]", txt)
        if match:
            return match.group(1).strip().lower()
        return txt

    @staticmethod
    def _split_bracket_command(user_reply: Optional[str]) -> Tuple[Optional[str], str]:
        if not user_reply:
            return None, ""
        raw = str(user_reply)
        match = re.match(r"^\s*\[(?P<cmd>[^\]]+)\]\s*(?P<arg>.*)$", raw)
        if not match:
            return None, raw.strip()
        cmd = match.group("cmd").strip().lower()
        arg = (match.group("arg") or "").strip()
        return cmd, arg

    def _apply_user_reply_to_waiting_mission(
        self, mission: MissionState, *, user_reply: str, question: str
    ) -> None:
        cmd, cmd_arg = self._split_bracket_command(user_reply)
        answer_mode = "free_text"
        reply_text = user_reply
        if cmd in {"answer", "ack_user"}:
            reply_text = cmd_arg or ("ack" if cmd == "ack_user" else "")
            answer_mode = "bracket_command"
        mission.last_user_reply = reply_text

        # STEP_CONSENT: el reply NO es feedback para replanning; es un gate explícito para reanudar el plan.
        try:
            reason = mission.ask_user_request.reason if mission.ask_user_request else None
        except Exception:
            reason = None
        options: List[Dict[str, str]] = []
        try:
            options = list(mission.pending_user_options or [])
        except Exception:
            options = []
        if not options and isinstance(mission.notes, dict):
            pending_opts = mission.notes.get("_pending_user_options") or []
            if isinstance(pending_opts, list):
                options = [opt for opt in pending_opts if isinstance(opt, dict)]
                mission.pending_user_options = options
        normalized_reply = cmd or self._normalize_waiting_reply_text(reply_text)
        option_map = {
            str(opt.get("id") or "").strip().lower(): opt
            for opt in options
            if isinstance(opt, dict) and opt.get("id")
        }
        if normalized_reply in option_map:
            self._handle_user_option_selection(
                mission, option_map[normalized_reply], question=question, reason=reason
            )
            return
        if (reason == "step_consent_required") or ("[STEP_CONSENT]" in (question or "")):
            reply = (reply_text or "").strip().lower()
            reply = reply.replace("sí", "si")
            affirmative = reply in {
                "si",
                "s",
                "yes",
                "y",
                "ok",
                "okay",
                "vale",
                "confirmo",
                "confirmar",
                "confirm",
            }

            plan = mission.pending_plan or mission.last_plan
            pending = None
            try:
                if plan and isinstance(plan.metadata, dict):
                    pending = plan.metadata.get("pending_step_consent")
            except Exception:
                pending = None

            if (
                affirmative
                and plan
                and isinstance(plan.metadata, dict)
                and isinstance(pending, dict)
            ):
                step_id = pending.get("step_id")
                if isinstance(step_id, str) and step_id.strip():
                    cons = plan.metadata.get("step_consents")
                    if not isinstance(cons, dict):
                        cons = {}
                    cons[str(step_id)] = True
                    plan.metadata["step_consents"] = cons
                try:
                    if bool(plan.metadata.get("after_replan")):
                        plan.metadata["replan_consent_granted"] = True
                except Exception:
                    pass
                # Mantener el resume_state/resume_from_step_index generado por el runner.
                mission.feedback = None
            else:
                # Negación o instrucciones: replanear (no ejecutar el step).
                mission.pending_plan = None
                mission.feedback = reply_text
                try:
                    if plan and isinstance(plan.metadata, dict):
                        plan.metadata.pop("pending_step_consent", None)
                except Exception:
                    pass

            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            mission.council_signal = None
            mission.status = "IN_PROGRESS"
            mission.pending_user_options = []
            if isinstance(mission.notes, dict):
                mission.notes["_pending_user_options"] = []
            if mission.envelope:
                try:
                    mission.envelope.metadata["last_user_question"] = question
                    mission.envelope.metadata["last_user_reply"] = reply_text
                    mission.envelope.metadata["last_user_answer_mode"] = answer_mode
                except Exception:
                    pass
                try:
                    dlg = mission.envelope.metadata.get("dialog")
                    if not isinstance(dlg, list):
                        dlg = []
                    dlg.append(
                        {
                            "ts_utc": self._iso_utc(),
                            "question": question,
                            "reply": reply_text,
                            "mode": answer_mode,
                        }
                    )
                    mission.envelope.metadata["dialog"] = dlg[-20:]
                except Exception:
                    pass
            return

        mission.feedback = reply_text
        mission.await_user_input = False
        mission.permission_question = None
        mission.ask_user_request = None
        mission.council_signal = None
        lab_outcome = None
        try:
            if isinstance(mission.notes, dict):
                lab_outcome = mission.notes.get("lab_outcome")
        except Exception:
            lab_outcome = None
        unsupported = False
        try:
            if isinstance(lab_outcome, dict):
                failure_codes = lab_outcome.get("failure_codes")
                if isinstance(failure_codes, list):
                    for code in failure_codes:
                        code_txt = str(code or "")
                        if code_txt == "lab_job_unsupported" or code_txt.startswith("job_kind_"):
                            unsupported = True
                            break
        except Exception:
            unsupported = False
        if unsupported:
            mission.status = "FAILED"
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["lab_unsupported_kind"] = True
                    mission.notes["lab_unsupported_kind_skip_once"] = True
            except Exception:
                pass
            if MissionError:
                mission.last_mission_error = MissionError(
                    kind="plan_error", step_id=None, reason="LAB_UNSUPPORTED_KIND"
                )
            mission.last_result = AjaxExecutionResult(
                success=False,
                error="LAB_UNSUPPORTED_KIND",
                path="lab_handoff",
                detail={"mission_id": mission.mission_id, "lab_outcome": lab_outcome},
            )
        else:
            mission.status = "IN_PROGRESS"
        mission.pending_user_options = []
        if isinstance(mission.notes, dict):
            mission.notes["_pending_user_options"] = []
            mission.notes.pop("pending_question", None)
            mission.notes["last_user_answer_mode"] = answer_mode
            mission.notes["last_user_reply"] = reply_text
        if mission.envelope:
            try:
                mission.envelope.metadata["last_user_question"] = question
                mission.envelope.metadata["last_user_reply"] = reply_text
                mission.envelope.metadata["last_user_answer_mode"] = answer_mode
            except Exception:
                pass
            try:
                mission.envelope.metadata.pop("pending_question", None)
            except Exception:
                pass
            try:
                dlg = mission.envelope.metadata.get("dialog")
                if not isinstance(dlg, list):
                    dlg = []
                dlg.append(
                    {
                        "ts_utc": self._iso_utc(),
                        "question": question,
                        "reply": reply_text,
                        "mode": answer_mode,
                    }
                )
                mission.envelope.metadata["dialog"] = dlg[-20:]
            except Exception:
                pass

    def _handle_user_option_selection(
        self, mission: MissionState, option: Dict[str, Any], *, question: str, reason: Optional[str]
    ) -> None:
        option_id = str(option.get("id") or "").strip().lower()
        mission.pending_user_options = []
        if isinstance(mission.notes, dict):
            mission.notes["_pending_user_options"] = []
        # Al seleccionar una opción del menú, salimos explícitamente del estado WAITING antes de continuar.
        # Esto evita re-imprimir la misma pregunta por estado residual.
        mission.await_user_input = False
        mission.permission_question = None
        mission.ask_user_request = None
        if option_id in {"lab_cancel", "cancel", "cancel_job"}:
            mission.user_cancelled = True
            try:
                mission.notes["lab_cancelled"] = True
            except Exception:
                pass
            job_status = None
            job_action = "detached"
            cancel_reason = "user_cancel"
            if option_id == "cancel_job":
                cancel_reason = "lab_job_cancelled"
            if mission.lab_job_id and getattr(self, "lab_control", None):
                try:
                    job, _path = self.lab_control.load_job(mission.lab_job_id)
                    job_status = str(job.get("status") or "").upper()
                    if job_status in {"QUEUED", "RUNNING"}:
                        self.lab_control.cancel_job(mission.lab_job_id, reason=cancel_reason)
                        job_status = "CANCELLED"
                        job_action = "cancelled"
                except Exception:
                    job_status = None
                    job_action = "detached"
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["lab_job_cancelled"] = job_action == "cancelled"
                    if job_status:
                        mission.notes["lab_job_status"] = job_status
            except Exception:
                pass
            self._clear_waiting_mission(
                mission_id=mission.mission_id,
                cancelled=True,
                cancel_reason=cancel_reason,
            )
            mission.feedback = mission.last_user_reply or ""
            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            mission.council_signal = None
            mission.status = "CANCELLED"
            detail: Dict[str, Any] = {
                "mission_id": mission.mission_id,
                "lab_job_id": mission.lab_job_id,
                "lab_job_status": job_status,
                "lab_job_action": job_action,
                "cancel_reason": cancel_reason,
            }
            try:
                self._enrich_detail_with_router_summary(mission, detail)
            except Exception:
                pass
            mission.last_result = AjaxExecutionResult(
                success=False,
                error="CANCELLED",
                path="mission",
                detail=detail,
            )
            try:
                self._record_pending_mission_receipt(
                    mission_id=mission.mission_id,
                    choice=option_id,
                    transition=mission.status,
                )
            except Exception:
                pass
            return
        if option_id == "requeue_job":
            if mission.lab_job_id and getattr(self, "lab_control", None):
                old_job_id = mission.lab_job_id
                try:
                    record = self.lab_control.requeue_job(old_job_id, reason="user_requeue")
                except Exception:
                    record = {}
                new_job_id = record.get("job_id") if isinstance(record, dict) else None
                if new_job_id:
                    mission.lab_job_id = new_job_id
                    try:
                        if isinstance(mission.notes, dict):
                            mission.notes["lab_job_id"] = new_job_id
                            mission.notes["lab_job_path"] = record.get("job_path")
                            mission.notes["lab_requeue_from"] = old_job_id
                    except Exception:
                        pass
                    lab_kind = "experiment"
                    try:
                        if isinstance(mission.notes, dict):
                            lab_kind = mission.notes.get("lab_handoff_kind") or lab_kind
                    except Exception:
                        pass
                    payload = record.get("payload") if isinstance(record, dict) else {}
                    output_paths = (
                        list(payload.get("output_paths") or []) if isinstance(payload, dict) else []
                    )
                    artifacts = (
                        {"lab_job": record.get("job_path")} if record.get("job_path") else {}
                    )
                    derived = {"requeue_from": old_job_id}
                    self._finalize_lab_handoff_wait(
                        mission,
                        job_record=record,
                        derived=derived,
                        reason="lab_job_requeue",
                        lab_kind=lab_kind,
                        output_paths=output_paths,
                        artifacts=artifacts,
                    )
                    return
            mission.feedback = mission.last_user_reply or ""
            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            mission.council_signal = None
            mission.status = "IN_PROGRESS"
            return
        if option_id == "wait_lab":
            try:
                mission.notes["lab_wait_requested"] = True
            except Exception:
                pass
            mission.feedback = mission.last_user_reply or ""
            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            mission.council_signal = None
            mission.status = "IN_PROGRESS"
            try:
                self._record_pending_mission_receipt(
                    mission_id=mission.mission_id,
                    choice=option_id,
                    transition=mission.status,
                )
            except Exception:
                pass
            return
        if option_id in {"lab_resume", "resume"}:
            try:
                mission.notes["lab_resume_override"] = True
            except Exception:
                pass
            mission.feedback = mission.last_user_reply or ""
            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            mission.council_signal = None
            mission.status = "IN_PROGRESS"
            try:
                self._record_pending_mission_receipt(
                    mission_id=mission.mission_id,
                    choice=option_id,
                    transition=mission.status,
                )
            except Exception:
                pass
            return
        if option_id in {"close_manual_done", "close_manual"}:
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["completed_manual"] = True
                    mission.notes["manual_close_reason"] = "user_confirmed"
            except Exception:
                pass
            try:
                payload_dir = self.config.root_dir / "artifacts" / "waiting_for_user"
                payload_path = payload_dir / f"{mission.mission_id}.json"
                if payload_path.exists():
                    doc = json.loads(payload_path.read_text(encoding="utf-8"))
                else:
                    doc = {}
                if isinstance(doc, dict):
                    doc["completed_manual"] = True
                    doc["completed_manual_utc"] = self._iso_utc()
                    doc["completed_manual_reason"] = "user_confirmed"
                    payload_path.write_text(
                        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
            except Exception:
                pass
            self._clear_waiting_mission(mission_id=mission.mission_id, cancelled=False)
            mission.feedback = mission.last_user_reply or ""
            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            mission.council_signal = None
            if microfilm_enforce_verify_before_done is not None:
                verification_payload: Dict[str, Any] = {}
                try:
                    if (
                        mission.last_result
                        and isinstance(mission.last_result.detail, dict)
                        and isinstance(mission.last_result.detail.get("verification"), dict)
                    ):
                        verification_payload = dict(mission.last_result.detail.get("verification"))
                except Exception:
                    verification_payload = {}
                done_gate = microfilm_enforce_verify_before_done(
                    {"status": "DONE", "mission_status": "DONE", "manual_close": True},
                    verification_payload,
                )
                if not bool(done_gate.get("ok")):
                    mission.status = "BLOCKED"
                    mission.last_result = AjaxExecutionResult(
                        success=False,
                        error=str(done_gate.get("code") or "BLOCKED_VERIFY_REQUIRED"),
                        path="manual_close",
                        detail=done_gate,
                    )
                    try:
                        self._record_pending_mission_receipt(
                            mission_id=mission.mission_id,
                            choice=option_id,
                            transition=mission.status,
                        )
                    except Exception:
                        pass
                    return
            mission.status = "DONE"
            detail = {
                "mission_id": mission.mission_id,
                "manual_close": True,
                "reason": "user_confirmed",
            }
            try:
                self._enrich_detail_with_router_summary(mission, detail)
            except Exception:
                pass
            mission.last_result = AjaxExecutionResult(
                success=True,
                error=None,
                path="manual_close",
                detail=detail,
            )
            try:
                self._record_pending_mission_receipt(
                    mission_id=mission.mission_id,
                    choice=option_id,
                    transition=mission.status,
                )
            except Exception:
                pass
            return
        if option_id == "use_deterministic_recipe":
            habit = None
            try:
                if habits_mod:
                    habit = habits_mod.find_habit_for_intent(
                        mission.intention,
                        safety_profile=self._safety_profile(),
                        os_name=self._os_name(),
                        path=self.config.root_dir / "data" / "habits.json",
                    )
            except Exception:
                habit = None
            if habit:
                try:
                    if isinstance(mission.notes, dict):
                        mission.notes["recipe_match"] = True
                        mission.notes["recipe_kind"] = "habit"
                        mission.notes["recipe_id"] = habit.id
                except Exception:
                    pass
                plan_dict = self._build_plan_from_habit(habit)
                plan = AjaxPlan(
                    plan_id=plan_dict.get("plan_id"),
                    steps=plan_dict.get("steps") or [],
                    metadata=plan_dict.get("metadata") or {},
                )
                if plan_dict.get("success_contract"):
                    plan.success_spec = plan_dict.get("success_contract")
                mission.pending_plan = plan
                mission.await_user_input = False
                mission.permission_question = None
                mission.ask_user_request = None
                mission.council_signal = None
                mission.status = "IN_PROGRESS"
                return
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["recipe_match"] = False
                    mission.notes["recipe_reason"] = "no_deterministic_recipe"
            except Exception:
                pass
            note = "No hay receta determinista aplicable para este intent. Puedes abrir INCIDENT para crear una."
            mission.last_result = self._finalize_ask_user_wait(
                mission,
                note,
                source="deterministic_recipe",
                blocking_reason="no_deterministic_recipe",
                extra_context={"recipe_match": False, "recipe_reason": "no_deterministic_recipe"},
            )
            return
        if option_id == "open_incident":
            try:
                incident_id = self._open_provider_health_incident(
                    mission,
                    question=question,
                    reason=reason or option.get("label") or "",
                )
                mission.feedback = f"incident:{incident_id}"
            except Exception as exc:
                gap_path = self._emit_incident_gap(
                    mission, detail={"error": str(exc), "option": option_id}
                )
                mission.feedback = mission.last_user_reply or ""
                try:
                    print(
                        f"⚠️ No se pudo abrir INCIDENT automáticamente (gap={gap_path}). Continúo con tu respuesta."
                    )
                except Exception:
                    pass
            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            mission.council_signal = None
            mission.status = "IN_PROGRESS"
            return
        if option_id == "use_premium_now":
            mission.premium_rule = "now"  # type: ignore[assignment]
            mission.brain_cost_mode_override = "premium"
            mission.feedback = ""
            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            mission.council_signal = None
            mission.status = "IN_PROGRESS"
            return
        if option_id.startswith("retry_escalate"):
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["retry_escalate_requested"] = True
                    mission.notes["retry_escalate_ts"] = self._iso_utc()
            except Exception:
                pass
            self._apply_brain_retry_policy(mission)
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["retry_escalate_cost_mode"] = mission.cost_mode
                    sx_path = None
                    if mission.envelope and isinstance(
                        getattr(mission.envelope, "metadata", None), dict
                    ):
                        sx_path = mission.envelope.metadata.get("starting_xi_path")
                    mission.notes["retry_escalate_starting_xi_path"] = sx_path
            except Exception:
                pass
            self._record_waiting_payload_escalation(mission, option_id=option_id)
        mission.feedback = mission.last_user_reply or ""
        mission.await_user_input = False
        mission.permission_question = None
        mission.ask_user_request = None
        mission.council_signal = None
        mission.status = "IN_PROGRESS"

    def health_check(self, strict: bool = False) -> AjaxHealthReport:
        notes: Dict[str, str] = {}
        # Driver
        driver_ok = False
        try:
            if self._driver_status() == "down":
                notes["driver"] = "breaker_down"
            elif self.driver and hasattr(self.driver, "health"):
                info = self.driver.health()
                driver_ok = (
                    bool(getattr(info, "get", lambda *args, **kwargs: False)("ok", False))
                    if isinstance(info, dict)
                    else True
                )
                if not driver_ok:
                    notes["driver"] = f"health()={info!r}"
            elif self.driver:
                driver_ok = True
            else:
                notes["driver"] = "driver_none"
        except Exception as exc:
            driver_ok = False
            notes["driver"] = f"exception:{exc}"

        # RAG
        rag_ok = False
        if self.rag_client:
            try:
                if hasattr(self.rag_client, "health"):
                    self.rag_client.health()
                rag_ok = True
            except Exception as exc:
                rag_ok = False
                notes["rag"] = f"exception:{exc}"
        else:
            notes["rag"] = "rag_client_none"

        models_count = len(self.capabilities.get("models_all", []))
        vision_models_count = len(self.capabilities.get("models_vision", []))
        if models_count == 0:
            notes["models"] = "no models discovered"

        ok = driver_ok and (models_count > 0 or not strict)

        report = AjaxHealthReport(
            ok=ok,
            driver_ok=driver_ok,
            rag_ok=rag_ok,
            models_count=models_count,
            vision_models_count=vision_models_count,
            notes=notes,
        )
        self.health = report  # type: ignore[attr-defined]
        return report

    def _default_governance(self) -> Optional["GovernanceSpec"]:
        if GovernanceSpec is None:
            return None
        return GovernanceSpec(budget_tokens=0, budget_seconds=0, risk_level="medium", autonomy="L1")

    @staticmethod
    def _infer_risk_level_from_intent(intention: str) -> str:
        """
        Heurística mínima para quorum del Council (LAB+low => 1).
        Conservadora: solo marca como low intents claramente "media/playback".
        """
        text = (intention or "").strip().lower()
        if not text:
            return "medium"
        # Nota: el riesgo "low" es una clasificación de gobernanza (quorum) y NO implica que se pueda ejecutar sin EFE.
        # Debe ser agnóstico de proveedor/URL (CG-1): evita strings de casos concretos.
        try:
            if AjaxCore._infer_intent_class(intention) == "media_playback":
                return "low"
        except Exception:
            pass
        high_markers = [
            "borra",
            "elimina",
            "delete",
            "rm -rf",
            "transfer",
            "paga",
            "paypal",
            "tarjeta",
            "password",
            "contraseña",
            "banco",
            "configura",
            "configurar",
            "configuración",
            "configuracion",
            "settings",
        ]
        if any(m in text for m in high_markers):
            return "high"
        low_markers = ["reproduce", "reproduc", "play ", "música", "musica", "song", "standby"]
        if any(m in text for m in low_markers):
            return "low"
        return "medium"

    @staticmethod
    def _infer_intent_class(intention: str) -> str:
        """
        Clasifica la intención en una clase funcional (agnóstica de apps/URLs).
        """
        text = (intention or "").strip().lower()
        if not text:
            return "generic"
        if AjaxCore._is_os_micro_action(text):
            return "os_micro_action"
        # Clase: reproducción de media (audio/video) sin asumir plataforma.
        media_markers = [
            "reproduce",
            "reproduc",
            "pon ",
            "play",
            "música",
            "musica",
            "canción",
            "cancion",
            "song",
            "audio",
            "video",
            "vídeo",
            "podcast",
            "escucha",
            "listen",
            "watch",
        ]
        if any(tok in text for tok in media_markers):
            return "media_playback"
        return "generic"

    @staticmethod
    def _is_os_micro_action(intention: str) -> bool:
        """
        Heurística minimalista: acciones OS/ventana/app (agnóstico de app).
        Requiere verbo + objeto UI para evitar clasificar intents web complejos.
        """
        text = (intention or "").strip().lower()
        if not text:
            return False
        verbs = [
            "abre",
            "abrir",
            "open",
            "launch",
            "minimiza",
            "minimizar",
            "maximize",
            "maximiza",
            "restaura",
            "restaurar",
            "cierra",
            "cerrar",
            "focus",
            "enfoca",
            "cambia",
            "switch",
            "alternar",
        ]
        targets = [
            "ventana",
            "window",
            "app",
            "aplicación",
            "aplicacion",
            "programa",
            "escritorio",
            "desktop",
        ]
        if any(v in text for v in verbs) and any(t in text for t in targets):
            return True
        # Heurística corta: "abre X" sin conectores (no flujo multi-paso).
        if any(v in text for v in verbs):
            if any(tok in text for tok in ("http://", "https://", "www.")):
                return False
            if any(tok in text for tok in (" y ", " then ", " luego ", " después ")):
                return False
            if len(text.split()) <= 6:
                return True
        return False

    def _create_mission_envelope(self, intention: str, mode: str) -> Optional["MissionEnvelope"]:
        if MissionEnvelope is None:
            return None
        gid = f"mission-{uuid.uuid4().hex[:8]}"
        governance = self._default_governance()
        if governance is not None:
            try:
                governance.risk_level = self._infer_risk_level_from_intent(intention)  # type: ignore[assignment]
            except Exception:
                pass
        metadata = {
            "mode": mode,
            "intent_class": self._infer_intent_class(intention),
            # Constitucional: nunca reescribir la petición cruda del usuario. Se guarda tal cual
            # para trazabilidad/verificación externa.
            "user_request_raw": intention,
        }
        return MissionEnvelope(
            mission_id=gid,
            original_intent=intention,
            governance=governance if governance is not None else GovernanceSpec(),  # type: ignore[arg-type]
            metadata=metadata,
        )

    def _clone_envelope_for_retry(
        self, envelope: "MissionEnvelope", *, retry_of: str
    ) -> Optional["MissionEnvelope"]:
        if MissionEnvelope is None:
            return None
        try:
            payload = envelope.to_dict()
        except Exception:
            return None
        new_id = f"mission-{uuid.uuid4().hex[:8]}"
        payload["mission_id"] = new_id
        payload["created_at"] = time.time()
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata["retry_of"] = retry_of
        metadata["fresh_clone"] = True
        payload["metadata"] = metadata
        try:
            return MissionEnvelope.from_dict(payload)
        except Exception:
            return None

    def _normalize_success_contract(self, spec: Optional[Any]) -> Optional["SuccessContract"]:
        if SuccessContract is None:
            return None
        if isinstance(spec, SuccessContract):
            return spec
        if isinstance(spec, dict):
            # Nuevo contrato explícito
            if "primary_source" in spec or "primary_check" in spec:
                return SuccessContract(
                    primary_source=str(spec.get("primary_source") or "uia"),
                    primary_check=spec.get("primary_check") or {},
                    fallback_source=spec.get("fallback_source", "none"),
                    fallback_check=spec.get("fallback_check"),
                    conflict_resolution=str(spec.get("conflict_resolution") or "fail_safe"),
                )
            # Compatibilidad: success_spec legado como primary_check UIA
            return SuccessContract(primary_source="uia", primary_check=spec)
        # Sin criterio -> contrato mínimo que no bloquea
        return SuccessContract(
            primary_source="uia", primary_check={"type": "check_last_step_status"}
        )

    @staticmethod
    def _contract_has_two_signals(contract: Optional["SuccessContract"]) -> bool:
        if contract is None:
            return False
        try:
            primary_ok = bool(
                contract.primary_source and contract.primary_source != "none"
            ) and bool(contract.primary_check)
            fallback_ok = bool(
                contract.fallback_source and contract.fallback_source != "none"
            ) and bool(contract.fallback_check)
            return primary_ok and fallback_ok
        except Exception:
            return False

    def _default_media_playback_success_contract(
        self, intention: str
    ) -> Optional["SuccessContract"]:
        """
        SuccessContract fuerte y agnóstico para `intent_class=media_playback`.
        Mínimo: 2 señales (telemetría/UIA + visión) con fail_safe.
        """
        if SuccessContract is None:
            return None
        browser_candidates = ["brave.exe", "chrome.exe", "msedge.exe", "firefox.exe"]
        primary_check = {"type": "active_window_process_in", "processes": browser_candidates}
        fallback_check = {
            "type": "visual_audit",
            "description": (
                "La pantalla muestra reproducción de media activa (audio/vídeo en curso): "
                "un reproductor o pestaña de media está en estado 'playing' (p.ej. botón Pausa visible, indicador de reproducción). "
                f"Intención: {intention}"
            ),
        }
        return SuccessContract(
            primary_source="uia",
            primary_check=primary_check,
            fallback_source="vision",
            fallback_check=fallback_check,
            conflict_resolution="fail_safe",
        )

    def _classify_mission_error(
        self,
        plan: Optional["AjaxPlan"],
        result: Optional["AjaxExecutionResult"],
        evaluation: Optional[Dict[str, Any]],
    ) -> Optional["MissionError"]:
        if MissionError is None:
            return None
        safe_eval: Dict[str, Any] = evaluation or {}
        reason_parts: List[str] = []
        kind: Literal[
            "plan_error", "world_error", "sensor_error", "governance_error", "unknown"
        ] = "unknown"

        if result and result.error:
            kind = "plan_error"
            reason_parts.append(f"runner_error={result.error}")
        if not safe_eval.get("ok", True):
            eval_reason = str(safe_eval.get("reason") or "")
            reason_parts.append(f"success_eval={eval_reason}")
            primary_detail = safe_eval.get("primary") or {}
            if primary_detail.get("error"):
                kind = "sensor_error"
            elif (safe_eval.get("fallback") or {}).get("error"):
                kind = "sensor_error"
            elif kind == "unknown":
                kind = "world_error"
        if not reason_parts and result and not bool(result.success):
            reason_parts.append("execution_failed")
            kind = "plan_error" if kind == "unknown" else kind

        step_id = plan.plan_id if plan and plan.plan_id else (plan.id if plan else None)
        return MissionError(
            kind=kind, step_id=step_id, reason="; ".join([p for p in reason_parts if p])
        )

    def _append_execution_event(
        self,
        envelope: Optional["MissionEnvelope"],
        plan: Optional["AjaxPlan"],
        result: "AjaxExecutionResult",
        evaluation: Optional[Dict[str, Any]],
    ) -> None:
        if envelope is None or ExecutionEvent is None:
            return
        step_id = plan.plan_id if plan and plan.plan_id else plan.id if plan else "plan"
        action = "run_plan"
        safe_result = result.detail
        try:
            json.dumps(safe_result)
        except Exception:
            safe_result = str(result.detail)
        safe_eval = evaluation
        try:
            json.dumps(safe_eval)
        except Exception:
            safe_eval = str(evaluation)
        event = ExecutionEvent(
            ts=time.time(),
            step_id=str(step_id or "plan"),
            action=action,
            ok=bool(result.success),
            detail={"result": safe_result, "evaluation": safe_eval},
            error=result.error,
        )
        envelope.execution_log.append(event)

    def chat(self, message: str, user_id: Optional[str] = None) -> str:
        reply, _perf = self._chat_with_perf(message, user_id=user_id, perf=None)
        return reply

    def chat_with_perf(
        self, message: str, user_id: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        perf: Dict[str, Any] = {}
        reply, perf = self._chat_with_perf(message, user_id=user_id, perf=perf)
        return reply, perf

    def _chat_with_perf(
        self,
        message: str,
        user_id: Optional[str],
        perf: Optional[Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any]]:
        """
        Respuesta rápida (modo conversación).
        - Devuelve [ACTION_REQUIRED] solo cuando existe COMMIT confirmado (token de escalado para CLI).
        - En caso de intención accionable sin slots, produce un intent draft y pide confirmación.
        """
        perf = perf if perf is not None else {}
        msg_clean = (message or "").strip()
        if not msg_clean:
            perf["intent_gate"] = {
                "speech_act": "chat",
                "action_required": False,
                "confidence": 0.4,
                "candidates": [],
                "one_question": None,
            }
            perf["snippets_count"] = 0
            return "¿En qué te ayudo?", perf
        user_key = self._resolve_chat_user_id(user_id)
        user_label = self._chat_user_labels.get(user_key, user_key) or "Usuario"
        ux_profile = self._chat_ux_profile(user_key)
        self._chat_last_ux_profile = ux_profile
        perf["ux_profile"] = ux_profile
        decision = self._decide_chat_gate(msg_clean, user_key)
        self.log.info(
            "chat_gate user=%s mode=%s reason=%s pending=%s",
            user_label,
            decision.get("mode"),
            decision.get("reason"),
            decision.get("pending_intent"),
        )
        debug_enabled = (os.getenv("AJAX_CHAT_DEBUG") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        confidence = decision.get("confidence")
        ambiguity = bool(decision.get("ambiguity"))
        action_commit = decision.get("mode") == "actuate"
        action_intent = bool(
            decision.get("action_intent")
            or action_commit
            or decision.get("mode") == "pre_operational"
        )
        leann_mode = self._chat_leann_mode()
        use_profile = leann_mode != "off"
        use_snips, leann_reason, _mode = self._chat_leann_should_use(
            ambiguity=ambiguity,
            action_required=action_intent,
            confidence=confidence if isinstance(confidence, (int, float)) else None,
        )
        perf["leann_mode"] = leann_mode
        perf["leann_reason"] = leann_reason
        perf["leann_profile_always"] = bool(use_profile and leann_mode == "auto")
        if decision.get("mode") in {"actuate", "pre_operational"}:
            self._lab_preempt(
                reason="chat_preemption",
                metadata={
                    "user": user_label,
                    "mode": decision.get("mode"),
                    "pending_intent": decision.get("pending_intent"),
                },
            )
        user_role_label = f"Usuario[{user_label}]"

        if str(decision.get("speech_act") or "") == "cancel":
            reply = "Entendido, cancelado."
            self.chat_history.append((user_role_label, msg_clean))
            self.chat_history.append(("AJAX", reply))
            perf["intent_gate"] = {
                "speech_act": "cancel",
                "action_required": False,
                "confidence": 0.9,
                "candidates": [],
                "one_question": None,
            }
            perf["snippets_count"] = 0
            self._record_leann_chat_receipt(
                reason=leann_reason, mode=leann_mode, snippets_count=0, profile_used=False
            )
            return reply, perf

        if decision["mode"] == "actuate":
            intent = (decision.get("intent") or msg_clean).strip()
            self.last_action_intent = intent
            self.chat_history.append((user_role_label, msg_clean))
            self.chat_history.append(("AJAX", "[ACTION_REQUIRED]"))
            perf["action_required"] = True
            if decision.get("commit"):
                perf["commit"] = True
            perf["intent_gate"] = {
                "speech_act": str(decision.get("speech_act") or "commit"),
                "action_required": True,
                "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.9,
                "candidates": [intent],
                "one_question": None,
                "reason": decision.get("reason"),
            }
            perf["snippets_count"] = 0
            self._record_leann_chat_receipt(
                reason=leann_reason, mode=leann_mode, snippets_count=0, profile_used=False
            )
            return "[ACTION_REQUIRED]", perf

        if decision["mode"] == "pre_operational":
            pending = decision.get("pending_intent") or msg_clean
            profile_card: Optional[Dict[str, Any]] = None
            if use_profile:
                profile_card, t_leann_profile = self._chat_leann_profile_card(pending, user_key)
                if t_leann_profile is not None:
                    perf["t_leann"] = t_leann_profile
            one_question = decision.get("one_question")
            reply = self._pre_operational_prompt(
                user_label,
                pending,
                has_profile=bool(profile_card),
                explicit_default_yes=ux_profile == "human",
                one_question=one_question if isinstance(one_question, dict) else None,
            )
            self.chat_history.append((user_role_label, msg_clean))
            self.chat_history.append(("AJAX", reply))
            perf["intent_gate"] = {
                "speech_act": str(decision.get("speech_act") or "await_confirm"),
                "action_required": True,
                "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.6,
                "candidates": [str(pending).strip()],
                "one_question": one_question if isinstance(one_question, dict) else None,
            }
            perf["snippets_count"] = 0
            self._record_leann_chat_receipt(
                reason=leann_reason,
                mode=leann_mode,
                snippets_count=0,
                profile_used=bool(profile_card),
            )
            return reply, perf

        system_prompt = ((AJAX_METHOD_PACK + "\n\n") if AJAX_METHOD_PACK else "") + (
            "Eres AJAX, un agente de sistema operativo.\n\n"
            "OBJETIVO:\n"
            "- Primero conversación útil y breve.\n"
            "- Si detectas una intención accionable (el usuario quiere que hagas algo en el ordenador), NO pidas una 'instrucción exacta'. "
            "En su lugar, propone un intent draft ejecutable y pide COMMIT.\n\n"
            "FORMATO:\n"
            "- Si `action_required=false`: responde SOLO en lenguaje natural.\n"
            "- Si `action_required=true`: responde SOLO con un objeto JSON (sin fences) con claves:\n"
            "  speech_act (string), action_required (true), confidence (0-1), candidates (list[str]), one_question (object|null),\n"
            "  intent_draft (string), request_commit (true).\n\n"
            "REGLAS:\n"
            "0) intent_draft debe ser SOLO la intención del usuario (1-2 líneas). NO enumeres Step0..StepN ni listas de acciones.\n"
            '1) `one_question` SOLO si falta un slot. Debe ser un objeto: {"question": "...", "default": "..."}.\n'
            "2) Si incluyes `one_question`, `intent_draft` DEBE corresponder al plan usando el valor por defecto.\n"
            "3) Solo UNA pregunta (la de `one_question`).\n"
            "4) Nunca emitas el token [ACTION_REQUIRED]. Eso lo gestiona el runtime tras COMMIT.\n"
            "5) Nunca afirmes 'completado' sin receipt de verificación."
        )
        # Construye contexto corto con últimos turnos
        hist_lines = []
        for role, content in self.chat_history[-6:]:
            hist_lines.append(f"{role}: {content}")
        user_prompt = ""
        if hist_lines:
            user_prompt += "Historial breve:\n" + "\n".join(hist_lines) + "\n"
        user_prompt += "Usuario: " + msg_clean
        # Paso 0: UCP (LEANN profile + 0-2 episodios)
        ajax_snips: List[Dict[str, Any]] = []
        profile_card = None
        t_leann = None
        try:
            t_leann_profile = None
            if use_profile:
                profile_card, t_leann_profile = self._chat_leann_profile_card(msg_clean, user_key)
            if use_snips:
                ajax_snips, t_leann = self._chat_leann_episodic_snippets(msg_clean, top_k=2)
                total_leann = 0.0
                if t_leann_profile is not None:
                    total_leann += t_leann_profile
                if t_leann is not None:
                    total_leann += t_leann
                if total_leann:
                    perf["t_leann"] = round(total_leann, 2)
        except Exception:
            ajax_snips = []
            profile_card = None
        snippets_count = len(ajax_snips)
        perf["snippets_count"] = int(snippets_count)
        perf["intent_gate"] = {
            "speech_act": str(
                decision.get("speech_act") or ("action_hint" if action_intent else "chat")
            ),
            "action_required": bool(action_intent),
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.6,
            "candidates": [],
            "one_question": None,
        }
        self._record_leann_chat_receipt(
            reason=leann_reason,
            mode=leann_mode,
            snippets_count=snippets_count,
            profile_used=bool(profile_card),
        )
        # Inyecta snippets al user_prompt si existen
        profile_txt = self._chat_leann_profile_text(profile_card)
        if profile_txt:
            user_prompt += "\nPerfil del usuario (LEANN):\n" + profile_txt + "\n"
        if ajax_snips:
            snippet_txt = "\n".join(
                f"- {s.get('text', '')[:400]}" for s in ajax_snips if isinstance(s, dict)
            )
            user_prompt += "\nContexto LEANN (ajax_history):\n" + snippet_txt + "\n"

        brain_error: Optional[Exception] = None
        t_llm_start = time.monotonic()
        try:
            resp = self._chat_llm(system_prompt, user_prompt)
            reply = resp.strip() if isinstance(resp, str) else str(resp)
        except Exception as exc:
            brain_error = exc
            self.log.warning("chat fallback (error=%s)", exc)
            reply = ""
        finally:
            perf["t_llm"] = round((time.monotonic() - t_llm_start) * 1000, 2)

        intent_draft_payload = self._extract_intent_draft(reply)
        if not intent_draft_payload and action_intent:
            intent_draft_payload = {
                "speech_act": "intent_draft_fallback",
                "action_required": True,
                "confidence": 0.45,
                "candidates": [msg_clean],
                "one_question": None,
                "intent_draft": msg_clean,
                "request_commit": True,
            }
        if intent_draft_payload:
            draft = self._sanitize_intent_draft(
                str(intent_draft_payload.get("intent_draft") or msg_clean).strip()
            )
            profile = self._chat_intent_profiles.setdefault(
                user_key, {"state": "idle", "pending_intent": None, "confirmed_templates": []}
            )
            profile["state"] = "pre_operational"
            profile["pending_intent"] = draft
            one_question = (
                intent_draft_payload.get("one_question")
                if isinstance(intent_draft_payload, dict)
                else None
            )
            profile["pending_question"] = one_question if isinstance(one_question, dict) else None
            candidates = (
                intent_draft_payload.get("candidates")
                if isinstance(intent_draft_payload, dict)
                else None
            )
            profile["pending_candidates"] = candidates if isinstance(candidates, list) else None
            profile["commit_ready"] = True
            has_profile = False
            if use_profile:
                profile_card, t_leann_profile = self._chat_leann_profile_card(draft, user_key)
                if t_leann_profile is not None:
                    perf["t_leann"] = t_leann_profile
                has_profile = bool(profile_card)
            reply = self._pre_operational_prompt(
                user_label,
                draft,
                has_profile=has_profile,
                explicit_default_yes=ux_profile == "human",
                one_question=profile.get("pending_question")
                if isinstance(profile.get("pending_question"), dict)
                else None,
            )
            perf["commit_ready"] = True
            perf["intent_gate"] = {
                "speech_act": str(intent_draft_payload.get("speech_act") or "intent_draft"),
                "action_required": True,
                "confidence": float(intent_draft_payload.get("confidence") or 0.6)
                if isinstance(intent_draft_payload.get("confidence"), (int, float))
                else 0.6,
                "candidates": [str(x) for x in (candidates or []) if isinstance(x, str)][:3],
                "one_question": profile.get("pending_question")
                if isinstance(profile.get("pending_question"), dict)
                else None,
            }
            self.chat_history.append((user_role_label, msg_clean))
            self.chat_history.append(("AJAX", reply))
            self._record_leann_chat_receipt(
                reason=leann_reason, mode=leann_mode, snippets_count=0, profile_used=has_profile
            )
            return reply, perf

        if not reply:
            if brain_error:
                reply = "Ahora mismo no tengo un modelo disponible. Configura `GROQ_API_KEY` o arranca un servidor local (p.ej. LM Studio en `http://localhost:1235/v1`)."
            else:
                reply = "Entendido."
        elif self._reply_claims_completion(reply) and not perf.get("action_required"):
            reply = "Listo para ejecutar cuando confirmes."
        # Actualiza historial
        self.chat_history.append((user_role_label, msg_clean))
        self.chat_history.append(("AJAX", reply))
        return reply, perf

    def _resolve_chat_user_id(self, user_id: Optional[str]) -> str:
        candidate = (
            user_id or getattr(self.config, "user_id", None) or self.default_user_id or "primary"
        ).strip()
        if not candidate:
            candidate = "primary"
        safe = re.sub(r"[^a-z0-9_\\-]+", "_", candidate.lower()) or "primary"
        self._chat_user_labels.setdefault(safe, candidate)
        return safe

    def _chat_ux_profile(self, user_key: str) -> str:
        raw = (os.getenv("UX_PROFILE") or os.getenv("AJAX_CHAT_UX_PROFILE") or "").strip().lower()
        if raw in {"human", "ops"}:
            return raw
        legacy = (os.getenv("AJAX_UX_PROFILE") or "").strip().lower()
        if legacy in {"human", "ops"}:
            return legacy
        return "human" if user_key == "primary" else "ops"

    def _decide_chat_gate(self, message: str, user_id: str) -> Dict[str, Any]:
        profile = self._chat_intent_profiles.setdefault(
            user_id, {"state": "idle", "pending_intent": None, "confirmed_templates": []}
        )
        normalized = message.strip()
        compact = self._normalize_intent_text(normalized)
        lowered = normalized.lower()
        if not normalized:
            return {
                "mode": "chat",
                "speech_act": "chat",
                "action_intent": False,
                "reason": "empty_input",
                "ambiguity": False,
                "confidence": 0.4,
            }

        if profile.get("state") == "pre_operational":
            pending = profile.get("pending_intent")
            pending_norm = self._normalize_intent_text(pending or "")
            pending_question = (
                profile.get("pending_question")
                if isinstance(profile.get("pending_question"), dict)
                else None
            )
            cancel_tokens = {
                "no",
                "cancel",
                "cancela",
                "cancelar",
                "stop",
                "para",
                "parar",
                "aborta",
                "abort",
            }
            if lowered.strip() in cancel_tokens:
                profile["state"] = "idle"
                profile["pending_intent"] = None
                profile.pop("pending_question", None)
                profile.pop("pending_candidates", None)
                profile.pop("commit_ready", None)
                return {
                    "mode": "chat",
                    "speech_act": "cancel",
                    "action_intent": False,
                    "reason": "user_cancelled",
                    "ambiguity": False,
                    "confidence": 0.9,
                }
            if self._is_short_confirmation(lowered):
                intent = pending or normalized
                profile["state"] = "idle"
                profile["pending_intent"] = None
                profile.pop("pending_question", None)
                profile.pop("pending_candidates", None)
                commit_ready = bool(profile.pop("commit_ready", False))
                if pending_norm:
                    self._remember_confirmed_template(profile, pending_norm)
                return {
                    "mode": "actuate",
                    "speech_act": "confirm",
                    "action_intent": True,
                    "intent": intent,
                    "reason": "commit_confirmed"
                    if commit_ready
                    else "confirmation_pre_operational",
                    "ambiguity": False,
                    "confidence": 0.9,
                    "commit": commit_ready,
                }
            if pending_question and normalized:
                # Interpretar respuesta como filling del único slot y ejecutar.
                default_val = str(pending_question.get("default") or "").strip()
                answer = normalized.strip()
                new_intent = str(pending or "").strip() or answer
                if default_val and default_val in new_intent:
                    new_intent = new_intent.replace(default_val, answer, 1)
                elif answer and answer not in new_intent:
                    new_intent = f"{new_intent} ({answer})"
                profile["state"] = "idle"
                profile["pending_intent"] = None
                profile.pop("pending_question", None)
                profile.pop("pending_candidates", None)
                commit_ready = bool(profile.pop("commit_ready", False))
                if pending_norm:
                    self._remember_confirmed_template(
                        profile, self._normalize_intent_text(new_intent)
                    )
                return {
                    "mode": "actuate",
                    "speech_act": "slot_answer",
                    "action_intent": True,
                    "intent": new_intent,
                    "reason": "slot_answer_commit",
                    "ambiguity": False,
                    "confidence": 0.85,
                    "commit": commit_ready,
                }

            # Interpretar como ajuste del intent draft y volver a pedir confirmación.
            if normalized and normalized != pending:
                profile["pending_intent"] = normalized
                return {
                    "mode": "pre_operational",
                    "speech_act": "edit",
                    "action_intent": True,
                    "pending_intent": normalized,
                    "one_question": pending_question,
                    "reason": "edit_pending",
                    "ambiguity": True,
                    "confidence": 0.55,
                }
            return {
                "mode": "pre_operational",
                "speech_act": "await_confirm",
                "action_intent": True,
                "pending_intent": pending or normalized,
                "one_question": pending_question,
                "reason": "awaiting_confirm",
                "ambiguity": True,
                "confidence": 0.6,
            }

        if self._looks_like_direct_command(lowered):
            profile["state"] = "idle"
            profile["pending_intent"] = None
            profile.pop("pending_question", None)
            profile.pop("pending_candidates", None)
            self._remember_confirmed_template(profile, compact)
            return {
                "mode": "actuate",
                "speech_act": "command",
                "action_intent": True,
                "intent": normalized,
                "reason": "direct_command",
                "ambiguity": False,
                "confidence": 0.9,
            }

        if self._is_similar_to_history(compact, profile):
            profile["state"] = "idle"
            profile["pending_intent"] = None
            profile.pop("pending_question", None)
            profile.pop("pending_candidates", None)
            self._remember_confirmed_template(profile, compact)
            return {
                "mode": "actuate",
                "speech_act": "command",
                "action_intent": True,
                "intent": normalized,
                "reason": "known_operational_pattern",
                "ambiguity": False,
                "confidence": 0.85,
            }

        if self._looks_like_operational_hint(lowered):
            return {
                "mode": "chat",
                "speech_act": "action_hint",
                "action_intent": True,
                "reason": "operational_hint",
                "ambiguity": True,
                "confidence": 0.45,
            }

        profile["state"] = "idle"
        profile["pending_intent"] = None
        profile.pop("pending_question", None)
        profile.pop("pending_candidates", None)
        return {
            "mode": "chat",
            "speech_act": "chat",
            "action_intent": False,
            "reason": "conversational",
            "ambiguity": False,
            "confidence": 0.6,
        }

    def _lab_preempt(self, reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        store = getattr(self, "lab_control", None)
        if not store:
            return
        try:
            changed = store.pause_lab_org(reason=reason, metadata=metadata or {})
            if changed:
                self.log.info("LAB_ORG paused (%s).", reason)
        except Exception:
            pass

    def _current_rail(self) -> str:
        raw = os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"
        return str(raw).strip().lower()

    def _read_human_active_flag(self) -> Optional[bool]:
        paths = [
            self.config.root_dir / "state" / "human_active.flag",
            self.config.root_dir / "artifacts" / "state" / "human_active.flag",
            self.config.root_dir / "artifacts" / "policy" / "human_active.flag",
        ]
        for path in paths:
            try:
                if not path.exists():
                    continue
                raw = path.read_text(encoding="utf-8").strip()
                if not raw:
                    continue
                if raw.startswith("{"):
                    data = json.loads(raw)
                    if isinstance(data, dict) and "human_active" in data:
                        return bool(data.get("human_active"))
                lowered = raw.lower()
                if "true" in lowered:
                    return True
                if "false" in lowered:
                    return False
            except Exception:
                continue
        return None

    def _resolve_display_target_label(self, rail: str) -> str:
        env_target = str(os.getenv("AJAX_DISPLAY_TARGET") or "").strip().lower()
        if env_target:
            return env_target
        if load_display_map is not None:
            try:
                payload = load_display_map(self.config.root_dir)
                targets = (
                    payload.get("display_targets")
                    if isinstance(payload.get("display_targets"), dict)
                    else {}
                )
                target = targets.get("lab" if str(rail).strip().lower() == "lab" else "prod")
                if target is not None:
                    return "dummy" if str(rail).strip().lower() == "lab" else "primary"
            except Exception:
                pass
        return "dummy" if str(rail).strip().lower() == "lab" else "primary"

    @staticmethod
    def _normalize_reason_key(text: Optional[str]) -> str:
        if not text:
            return ""
        lowered = text.strip().lower()
        ascii_txt = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-z0-9]+", "_", ascii_txt).strip("_")

    def _should_launch_lab_probe(self, mission: MissionState, reason_key: str) -> bool:
        if not reason_key or reason_key not in LAB_PROBE_TRIGGER_KEYS:
            return False
        if self._current_rail() != "prod":
            return False
        if mission.status in {"DERIVED_TO_LAB_PROBE", "PAUSED_FOR_LAB"}:
            return False
        if mission.lab_job_id:
            return False
        return True

    def _pre_operational_prompt(
        self,
        user_label: str,
        pending: str,
        *,
        has_profile: bool = False,
        explicit_default_yes: bool = False,
        one_question: Optional[Dict[str, Any]] = None,
    ) -> str:
        snippet = pending.strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        prefix = f"{user_label}, " if user_label else ""
        q_text = ""
        q_default = ""
        if isinstance(one_question, dict):
            q_text = str(one_question.get("question") or "").strip()
            q_default = str(one_question.get("default") or "").strip()

        if q_text:
            lines = [f'{prefix}Propuesta: "{snippet}"', q_text.rstrip("?") + "?"]
            if q_default:
                lines[-1] = lines[-1] + f" (por defecto: {q_default})."
                lines.append(
                    'Responde "sí"/"default"/"1" para ejecutar el default, o escribe tu respuesta para ejecutarla.'
                )
            else:
                lines.append('Responde con tu respuesta (o "no" para cancelar).')
            return "\n".join(lines)

        confirm = "¿Confirmas?"
        if has_profile or explicit_default_yes:
            confirm += " (por defecto: sí)"
        lines = [
            f'{prefix}Voy a ejecutar: "{snippet}".',
            confirm,
            'Responde "sí"/"default"/"1" para ejecutar, o "no" para cancelar.',
        ]
        return "\n".join(lines)

    def _normalize_intent_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _remember_confirmed_template(self, profile: Dict[str, Any], normalized_text: str) -> None:
        if not normalized_text:
            return
        templates: List[str] = profile.setdefault("confirmed_templates", [])
        templates.append(normalized_text)
        if len(templates) > 5:
            del templates[:-5]

    def _text_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _is_similar_to_history(self, normalized_text: str, profile: Dict[str, Any]) -> bool:
        templates: List[str] = profile.get("confirmed_templates") or []
        for template in templates:
            if self._text_similarity(template, normalized_text) >= 0.9:
                return True
        return False

    def _looks_like_direct_command(self, text: str) -> bool:
        cleaned = re.sub(r"[¿?¡!]", " ", text).strip()
        if not cleaned:
            return False
        tokens = cleaned.split()
        if not tokens:
            return False
        first = tokens[0]
        imperative_prefixes = {
            "abre",
            "open",
            "lanza",
            "inicia",
            "ejecuta",
            "play",
            "pon",
            "reproduce",
            "busca",
            "verifica",
            "muestra",
            "cierra",
            "apaga",
            "enciende",
        }
        if first in imperative_prefixes:
            return True
        lowered = cleaned.lower()
        if lowered.startswith("por favor "):
            lowered = lowered[len("por favor ") :]
            tokens = lowered.split()
            if tokens and tokens[0] in imperative_prefixes:
                return True
        polite_starters = (
            "puedes ",
            "podrias ",
            "podrías ",
            "puede ",
            "podras ",
            "podrías por favor ",
            "puedes por favor ",
        )
        if any(lowered.startswith(ps) for ps in polite_starters):
            return any(word in lowered for word in self._CHAT_ACTION_WORDS)
        if (
            "necesito que" in lowered or "quiero que" in lowered or "me ayudas a" in lowered
        ) and any(word in lowered for word in self._CHAT_ACTION_WORDS):
            return True
        if "hazlo" in lowered or lowered.startswith("haz "):
            return True
        return False

    def _looks_like_operational_hint(self, text: str) -> bool:
        lowered = text.lower()
        if self._looks_like_direct_command(lowered):
            return False
        if any(hint in lowered for hint in self._CHAT_DESIRE_HINTS):
            return True
        return any(word in lowered for word in self._CHAT_ACTION_WORDS)

    def _is_short_confirmation(self, text: str) -> bool:
        cleaned = text.strip().lower()
        if not cleaned:
            return False
        if cleaned in self._CHAT_CONFIRMATION_TOKENS:
            return True
        if cleaned in {"1", "default", "por defecto"}:
            return True
        if "hazlo" in cleaned or "adelante" in cleaned:
            return True
        if "como otras veces" in cleaned or "como la otra vez" in cleaned:
            return True
        return False

    def _extract_intent_draft(self, reply: str) -> Optional[Dict[str, Any]]:
        raw = (reply or "").strip()
        if not raw.startswith("{") or not raw.endswith("}"):
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if not payload.get("intent_draft"):
            return None
        return payload

    def _sanitize_intent_draft(self, text: str) -> str:
        """
        Intent drafts must be 1-2 lines of user intent, not a narrated Step0..StepN plan.
        """
        raw = (text or "").strip()
        if not raw:
            return ""
        # Drop code fences / obvious step lists
        raw = raw.replace("```", "").strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        filtered: list[str] = []
        for ln in lines:
            low = ln.lower()
            if (
                low.startswith("step")
                or low.startswith("paso")
                or low.startswith("1)")
                or low.startswith("2)")
            ):
                continue
            if ln.startswith(("-", "*")):
                continue
            filtered.append(ln)
        if not filtered:
            filtered = lines[:1]
        # Keep at most 2 short lines.
        filtered = filtered[:2]
        out = " ".join(filtered).strip()
        out = re.sub(r"\s+", " ", out)
        if len(out) > 220:
            out = out[:217] + "..."
        return out

    def _compact_planner_intent(self, intention: str) -> str:
        """
        Planning input should be the user's intent only (1-2 lines).
        """
        return self._sanitize_intent_draft(intention)

    def _library_action_candidates(self, intention: str) -> List[LibrarySkillCandidate]:
        catalog = getattr(self, "actions_catalog", None)
        if not catalog or not hasattr(catalog, "list_actions"):
            return []
        text = str(intention or "").strip().lower()
        if not text:
            return []
        candidates: List[LibrarySkillCandidate] = []
        for spec in catalog.list_actions():
            raw_name = str(getattr(spec, "name", "") or "").strip()
            name = raw_name.lower()
            if not name:
                continue
            score = 0.0
            if name in text:
                score = 1.0
            else:
                spaced = name.replace(".", " ")
                if spaced in text:
                    score = 0.9
                else:
                    flat = name.replace(".", "")
                    if flat and flat in text:
                        score = 0.8
            if score < 0.8:
                continue
            candidates.append(
                LibrarySkillCandidate(
                    skill_id=f"action:{raw_name}",
                    kind="action",
                    label=str(getattr(spec, "description", "") or raw_name),
                    confidence=score,
                    action_name=raw_name,
                    args_schema=getattr(spec, "args_schema", None),
                )
            )
        return candidates

    def _library_habit_candidates(
        self,
        intention: str,
        safety_profile: str,
        os_name: str,
    ) -> Tuple[List[LibrarySkillCandidate], List[Dict[str, Any]]]:
        if executor is None:
            return [], []
        try:
            plans, issues = executor.propose_habit_plans(
                intention,
                safety_profile,
                os_name,
                path=str(self.config.root_dir / "data" / "habits.json"),
            )
        except Exception as exc:
            self.log.warning("No se pudo evaluar hábitos: %s", exc)
            return [], []
        candidates: List[LibrarySkillCandidate] = []
        for plan in plans:
            candidates.append(
                LibrarySkillCandidate(
                    skill_id=f"habit:{plan.habit_id}",
                    kind="habit",
                    label=f"habit:{plan.habit_id}",
                    confidence=plan.confidence,
                    plan_json=plan.plan_json,
                    habit_id=plan.habit_id,
                )
            )
        issues_payload = [
            {"habit_id": issue.habit_id, "reason": issue.reason, "detail": issue.detail}
            for issue in (issues or [])
        ]
        return candidates, issues_payload

    def _library_first_candidates(
        self, intention: str
    ) -> Tuple[List[LibrarySkillCandidate], List[Dict[str, Any]]]:
        safety_profile = os.getenv("AJAX_SAFETY_PROFILE", "normal").lower()
        os_name = "windows"
        habit_candidates, habit_issues = self._library_habit_candidates(
            intention, safety_profile, os_name
        )
        action_candidates = self._library_action_candidates(intention)
        candidates = habit_candidates + action_candidates
        return candidates, habit_issues

    def _map_intent_to_library_skill(
        self,
        intention: str,
        candidates: List[LibrarySkillCandidate],
    ) -> Optional[Dict[str, Any]]:
        if not candidates or mapper_system_prompt is None:
            return None
        try:
            provider_name, provider_cfg = self._select_brain_provider(exclude=None)
        except Exception:
            return None
        mapper_candidates = []
        for cand in candidates:
            mapper_candidates.append(
                {
                    "skill_id": cand.skill_id,
                    "kind": cand.kind,
                    "label": cand.label,
                    "args_schema": cand.args_schema or {},
                }
            )
        payload = {
            "intention": intention,
            "candidates": mapper_candidates,
        }
        system_prompt = mapper_system_prompt()
        user_prompt = json.dumps(payload, ensure_ascii=False)
        try:
            out = self._call_brain_provider(
                provider_name,
                provider_cfg,
                system_prompt,
                user_prompt,
                meta={"intention": intention, "planning": False, "purpose": "skill_mapper"},
            )
        except Exception:
            return None
        if not isinstance(out, dict):
            return None
        skill_id = out.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id.strip():
            return None
        if skill_id not in {c.skill_id for c in candidates}:
            return None
        slots = out.get("slots") if isinstance(out.get("slots"), dict) else {}
        confidence = (
            out.get("confidence") if isinstance(out.get("confidence"), (int, float)) else None
        )
        return {
            "skill_id": skill_id,
            "slots": slots,
            "confidence": confidence,
            "notes": out.get("notes"),
        }

    def _required_args_from_schema(self, schema: Optional[Dict[str, Any]]) -> List[str]:
        required: List[str] = []
        if not isinstance(schema, dict):
            return required
        for key, val in schema.items():
            text = str(val or "").lower()
            if "obligatorio" in text or "required" in text or "mandatory" in text:
                required.append(str(key))
        return required

    def _default_receipts_dir(self) -> str:
        override = os.getenv("AJAX_ARTIFACTS_DIR")
        if override:
            return str(Path(override) / "receipts")
        try:
            root = getattr(self, "config", None).root_dir  # type: ignore[union-attr]
            return str(Path(root) / "artifacts" / "receipts")
        except Exception:
            return "artifacts/receipts"

    def _is_missing_efe_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        if "empty_success_spec" in msg or "bad_success_spec" in msg:
            return True
        if "missing_fields" in msg and "success_spec" in msg:
            return True
        return False

    def _validate_brain_plan_with_efe_repair(
        self,
        plan_json: Dict[str, Any],
        *,
        intention: str,
        source: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        try:
            return self._validate_brain_plan(plan_json), None
        except Exception as exc:
            if not self._is_missing_efe_error(exc):
                raise
            repaired, receipt, reason = self._repair_missing_efe_for_brain_plan(
                intention=intention, plan_json=plan_json, source=source
            )
            if repaired:
                try:
                    validated = self._validate_brain_plan(repaired)
                except Exception as exc2:
                    return None, {
                        "receipt": receipt,
                        "reason": str(exc2),
                        "repaired": False,
                    }
                return validated, {"receipt": receipt, "repaired": True}
            return None, {"receipt": receipt, "reason": reason or str(exc), "repaired": False}

    def _repair_missing_efe_for_action_plan(
        self,
        *,
        intention: str,
        action_name: str,
        args: Dict[str, Any],
        evidence_required: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        if repair_plan_if_needed is None:
            return None, None, "repair_loop_unavailable"

        plan_stub = {
            "goal": f"Ejecutar {action_name}",
            "steps": [
                {
                    "id": "task-1",
                    "intent": f"Ejecutar {action_name} con parámetros derivados.",
                    "preconditions": {"expected_state": {}},
                    "action": action_name,
                    "args": args,
                    "evidence_required": evidence_required,
                    "success_spec": {"expected_state": {}},
                    "on_fail": "abort",
                }
            ],
            "metadata": {"intention": intention, "source": "library_action_stub"},
        }

        def _drafter_fn(prompt: str, original: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                provider_name, provider_cfg = self._select_brain_provider()
            except Exception:
                return None
            system_prompt = "Eres un reparador EFE. Responde SOLO JSON válido sin texto adicional."
            try:
                out = self._call_brain_provider(
                    provider_name,
                    provider_cfg,
                    system_prompt,
                    prompt,
                    meta={"intention": intention, "planning": True, "purpose": "efe_repair"},
                )
            except Exception:
                return None
            if not isinstance(out, dict):
                return None
            if isinstance(out.get("expected_state"), dict):
                repaired = dict(original)
                repaired["expected_state"] = out.get("expected_state")
                return repaired
            return out if isinstance(out.get("steps"), list) else None

        result = repair_plan_if_needed(
            plan_stub,
            drafter_fn=_drafter_fn,
            receipts_dir=self._default_receipts_dir(),
        )
        if result.success and isinstance(result.plan, dict):
            expected_state = result.plan.get("expected_state")
            if isinstance(expected_state, dict) and expected_state:
                return expected_state, result.receipt_path, None
        return None, result.receipt_path, result.reason or "missing_efe_final"

    def _repair_missing_efe_for_brain_plan(
        self,
        *,
        intention: str,
        plan_json: Dict[str, Any],
        source: str = "brain",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        if repair_plan_if_needed is None:
            return None, None, "repair_loop_unavailable"

        def _drafter_fn(prompt: str, original: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            try:
                provider_name, provider_cfg = self._select_brain_provider()
            except Exception:
                return None
            system_prompt = (
                "Eres un reparador EFE de planes AJAX. Responde SOLO JSON valido.\n"
                "No cambies acciones, args ni el orden de pasos. Solo completa success_spec.expected_state."
            )
            payload = {
                "intention": intention,
                "plan": original,
                "rules": [
                    "NO cambies ni reordenes steps",
                    "NO inventes acciones nuevas",
                    "Completa success_spec.expected_state con checks verificables",
                ],
            }
            user_prompt = json.dumps(payload, ensure_ascii=False)
            try:
                out = self._call_brain_provider(
                    provider_name,
                    provider_cfg,
                    system_prompt,
                    user_prompt,
                    meta={"intention": intention, "planning": True, "purpose": "efe_repair"},
                )
            except Exception:
                return None
            if not isinstance(out, dict):
                return None
            return out if isinstance(out.get("steps"), list) else None

        result = repair_plan_if_needed(
            plan_json,
            drafter_fn=_drafter_fn,
            receipts_dir=self._default_receipts_dir(),
        )
        if result.success and isinstance(result.plan, dict):
            return result.plan, result.receipt_path, None
        return None, result.receipt_path, result.reason or "missing_efe_final"

    def _build_missing_efe_plan(
        self,
        *,
        intention: str,
        source: str,
        receipt_path: Optional[str],
        reason: Optional[str],
        errors: Optional[List[str]] = None,
    ) -> AjaxPlan:
        plan_id = f"abort-missing-efe-{int(time.time())}"
        meta = {
            "intention": intention,
            "source": source,
            "planning_error": "missing_efe_final",
            "skip_council_review": True,
        }
        if receipt_path:
            meta["efe_repair_receipt"] = receipt_path
        if reason:
            meta["efe_repair_reason"] = reason
        if errors:
            meta["errors"] = list(errors)
        return AjaxPlan(
            id=plan_id,
            summary=f"Abort (missing EFE) for {intention}",
            steps=[],
            plan_id=plan_id,
            metadata=meta,
            success_spec={"type": "check_last_step_status"},
        )

    def _derive_action_success_spec(
        self, action_name: str, args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        expected: Dict[str, Any] = {}
        if action_name == "app.launch":
            process = args.get("process")
            if isinstance(process, str) and process.strip():
                expected["windows"] = [{"process_equals": process.strip(), "must_exist": True}]
            else:
                expected["meta"] = {"must_be_active": True}
        elif action_name == "window.focus":
            w = {}
            title = args.get("title_contains")
            process = args.get("process")
            if isinstance(title, str) and title.strip():
                w["title_contains"] = title.strip()
            if isinstance(process, str) and process.strip():
                w["process_equals"] = process.strip()
            if w:
                w["must_exist"] = True
                expected["windows"] = [w]
            expected.setdefault("meta", {})
            expected["meta"]["must_be_active"] = True
        elif action_name in {
            "keyboard.type",
            "keyboard.hotkey",
            "mouse.click",
            "mouse.move",
            "desktop.isolate_active_window",
        }:
            expected["meta"] = {"must_be_active": True}
        else:
            return None
        return {"expected_state": expected}

    def _build_action_plan_from_slots(
        self,
        *,
        intention: str,
        action_name: str,
        args_schema: Optional[Dict[str, Any]],
        slots: Dict[str, Any],
    ) -> Dict[str, Any]:
        required = self._required_args_from_schema(args_schema)
        args = {k: v for k, v in (slots or {}).items() if v is not None}
        missing = [k for k in required if k not in args]
        if missing:
            raise LibrarySelectionError(
                "missing_slots", detail={"action": action_name, "missing": missing}
            )
        requires_vision = False
        catalog = getattr(self, "actions_catalog", None)
        if catalog and hasattr(catalog, "requires_vision"):
            try:
                requires_vision = bool(catalog.requires_vision(action_name))
            except Exception:
                requires_vision = False
        evidence_required = ["driver.screenshot"] if requires_vision else ["driver.active_window"]
        success_spec = self._derive_action_success_spec(action_name, args)
        repair_receipt = None
        if not success_spec:
            repaired_es, repair_receipt, repair_reason = self._repair_missing_efe_for_action_plan(
                intention=intention,
                action_name=action_name,
                args=args,
                evidence_required=evidence_required,
            )
            if repaired_es:
                success_spec = {"expected_state": repaired_es}
            else:
                raise LibrarySelectionError(
                    "missing_efe",
                    detail={
                        "action": action_name,
                        "repair_receipt": repair_receipt,
                        "repair_reason": repair_reason,
                    },
                )
        step = {
            "id": "task-1",
            "intent": f"Ejecutar {action_name} con parámetros derivados.",
            "preconditions": {"expected_state": {}},
            "action": action_name,
            "args": args,
            "evidence_required": evidence_required,
            "success_spec": success_spec,
            "on_fail": "abort",
        }
        return {
            "plan_id": f"action:{action_name}-{int(time.time())}",
            "description": f"Plan directo para {action_name}",
            "steps": [step],
            "success_contract": {"type": "check_last_step_status"},
            "metadata": {
                "intention": intention,
                "source": "library_action",
                "efe_repair_receipt": repair_receipt,
            },
        }

    def _enrich_plan_steps(self, plan_json: Dict[str, Any]) -> None:
        steps = plan_json.get("steps")
        if not isinstance(steps, list):
            return
        meta = plan_json.get("metadata") or {}
        intention = meta.get("intention") or "Acción desde biblioteca"
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            # Asegurar campos obligatorios
            if not step.get("id"):
                step["id"] = f"step-{i + 1}"
            if not step.get("intent"):
                step["intent"] = intention
            if not step.get("on_fail"):
                step["on_fail"] = "abort"
            if not step.get("preconditions"):
                step["preconditions"] = {"expected_state": {}}

            action_name = step.get("action")
            args = step.get("args") or {}

            # Requisitos de evidencia
            if not step.get("evidence_required"):
                requires_vision = False
                catalog = getattr(self, "actions_catalog", None)
                if catalog and hasattr(catalog, "requires_vision"):
                    try:
                        requires_vision = bool(catalog.requires_vision(action_name))
                    except Exception:
                        requires_vision = False
                step["evidence_required"] = (
                    ["driver.screenshot"] if requires_vision else ["driver.active_window"]
                )

            # Success Spec (EFE)
            if not step.get("success_spec"):
                success_spec = self._derive_action_success_spec(action_name, args)
                if success_spec:
                    step["success_spec"] = success_spec

    def _plan_json_from_library_candidate(
        self,
        *,
        intention: str,
        candidate: LibrarySkillCandidate,
        slots: Dict[str, Any],
    ) -> Dict[str, Any]:
        if candidate.kind == "habit":
            if not candidate.plan_json:
                raise LibrarySelectionError("missing_efe", detail={"habit_id": candidate.habit_id})
            plan_json = dict(candidate.plan_json)
        else:
            plan_json = self._build_action_plan_from_slots(
                intention=intention,
                action_name=str(candidate.action_name or ""),
                args_schema=candidate.args_schema,
                slots=slots,
            )
        # Enriquecer pasos antes de devolver (especialmente importante para hábitos)
        self._enrich_plan_steps(plan_json)
        meta = plan_json.get("metadata") if isinstance(plan_json.get("metadata"), dict) else {}
        meta = dict(meta)
        meta["library_skill_id"] = candidate.skill_id
        meta["library_kind"] = candidate.kind
        if candidate.habit_id:
            meta["habit_id"] = candidate.habit_id
        meta["slots"] = slots or {}
        meta.setdefault("plan_source", "library")
        plan_json["metadata"] = meta
        return plan_json

    def _library_first_plan_json(
        self,
        intention: str,
        *,
        mission: Optional["MissionState"] = None,
    ) -> Optional[Dict[str, Any]]:
        candidates, habit_issues = self._library_first_candidates(intention)
        if not candidates:
            if habit_issues:
                raise LibrarySelectionError("missing_efe", detail={"habit_issues": habit_issues})
            return None
        slots: Dict[str, Any] = {}
        selected: Optional[LibrarySkillCandidate] = None
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            mapping = self._map_intent_to_library_skill(intention, candidates)
            if not mapping:
                raise LibrarySelectionError(
                    "mapper_failed", detail={"candidates": [c.skill_id for c in candidates]}
                )
            selected = next((c for c in candidates if c.skill_id == mapping.get("skill_id")), None)
            slots = mapping.get("slots") or {}
            if selected is None:
                raise LibrarySelectionError(
                    "mapper_failed", detail={"skill_id": mapping.get("skill_id")}
                )
        plan_json = self._plan_json_from_library_candidate(
            intention=intention, candidate=selected, slots=slots or {}
        )
        if mission is not None:
            try:
                mission.notes["library_skill_id"] = selected.skill_id
                mission.notes["library_kind"] = selected.kind
                mission.notes["library_trivial"] = bool(selected.confidence >= 0.9)
            except Exception:
                pass
        return plan_json

    def _reply_claims_completion(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(
            token in lowered for token in ("completad", "terminad", "hecho", "done", "completed")
        )

    # --- Ciclo OODA ---
    def perceive(self) -> AjaxObservation:
        fg = None
        if self.driver:
            if self._driver_status() == "down":
                raise RuntimeError("driver_unavailable")
            try:
                fg = self.driver.get_active_window()
                # reset CB si va bien
                self._driver_cb = {"status": "up", "failures": [], "down_since": None}
            except DriverTimeout as exc:
                self._register_driver_failure("timeout")
                raise RuntimeError("driver_timeout") from exc
            except DriverConnectionError as exc:
                self._register_driver_failure("connection_error")
                raise RuntimeError("driver_connection_error") from exc
            except WindowsDriverError as exc:
                self._register_driver_failure("driver_error")
                raise RuntimeError(f"driver_error:{exc}") from exc
            except Exception:
                self._register_driver_failure("driver_unknown")
                raise
        return AjaxObservation(timestamp=time.time(), foreground=fg, notes={})

    def plan(
        self,
        intention: str,
        observation: AjaxObservation,
        feedback: Optional[str] = None,
        envelope: Optional["MissionEnvelope"] = None,
        brain_exclude: Optional[set[str]] = None,
        mission: Optional["MissionState"] = None,
    ) -> AjaxPlan:
        norm = intention.lower()
        self.log.info("AJAX.plan: '%s'", intention)
        library_plan = self._library_first_plan_json(intention, mission=mission)
        if library_plan:
            return self._plan_from_json(library_plan)
        return self._plan_with_brain(
            intention,
            observation,
            feedback=feedback,
            envelope=envelope,
            brain_exclude=brain_exclude,
            mission=mission,
        )

    def _plan_from_json(self, plan_json: Dict[str, Any]) -> AjaxPlan:
        """
        Construye un AjaxPlan desde un dict (usado para hábitos u otras fuentes).
        """
        plan_id = plan_json.get("plan_id") or f"plan-{int(time.time())}"
        steps = plan_json.get("steps") or []
        raw_contract = plan_json.get("success_contract")
        plan_obj = AjaxPlan(
            id=plan_id,
            summary=plan_json.get("description") or f"Plan {plan_id}",
            steps=steps,
            plan_id=plan_id,
            metadata=plan_json.get("metadata") or {},
            success_spec=raw_contract,
        )
        if not plan_obj.success_spec:
            plan_obj.success_spec = {"type": "check_last_step_status"}
        if raw_contract:
            plan_obj.metadata["success_contract"] = raw_contract
        return plan_obj

    def _apply_rigor_to_plan(
        self, plan: AjaxPlan, rigor: Optional["agency.rigor_selector.RigorDecision"]
    ) -> AjaxPlan:
        if rigor:
            plan.metadata = plan.metadata or {}
            plan.metadata["rigor_decision"] = rigor.to_dict()
        return plan

    @staticmethod
    def _expected_state_has_checks(expected: Any) -> bool:
        if not isinstance(expected, dict):
            return False
        if expected.get("windows"):
            return True
        if expected.get("files"):
            return True
        if isinstance(expected.get("checks"), list) and expected.get("checks"):
            return True
        meta = expected.get("meta") or {}
        return bool(isinstance(meta, dict) and meta.get("must_be_active"))

    def _extract_expected_state_from_plan(self, plan: Optional[AjaxPlan]) -> Optional[Dict[str, Any]]:
        if not plan:
            return None
        try:
            if isinstance(plan.success_spec, dict):
                expected = plan.success_spec.get("expected_state")
                if self._expected_state_has_checks(expected):
                    return expected
        except Exception:
            pass
        try:
            steps = plan.steps or []
        except Exception:
            steps = []
        if isinstance(steps, list):
            for step in reversed(steps):
                if not isinstance(step, dict):
                    continue
                succ = step.get("success_spec")
                if not isinstance(succ, dict):
                    continue
                expected = succ.get("expected_state")
                if self._expected_state_has_checks(expected):
                    return expected
        return None

    @staticmethod
    def _plan_has_destructive_markers(plan: Optional[AjaxPlan]) -> bool:
        if not plan or not isinstance(plan.steps, list):
            return False
        destructive_actions = {
            "delete",
            "delete_file",
            "remove",
            "rm",
            "rm -rf",
            "wipe",
            "format",
            "truncate",
            "drop",
        }
        destructive_cmd_tokens = [
            "rm -rf",
            "del /f",
            "rd /s",
            "format ",
            "mkfs",
            "wipe",
            "truncate",
            "drop database",
            "drop table",
        ]
        for step in plan.steps:
            if not isinstance(step, dict):
                continue
            action = str(step.get("action") or "").strip().lower()
            if any(tok in action for tok in destructive_actions):
                return True
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            for key, value in args.items():
                if not isinstance(value, str):
                    continue
                lowered = value.lower()
                if any(tok in lowered for tok in destructive_cmd_tokens):
                    return True
        return False

    def _record_rigor_decision_trace(
        self,
        mission: MissionState,
        *,
        decision: Optional[Dict[str, Any]] = None,
        plan_id: Optional[str] = None,
        synthetic_approve: bool = False,
        note: Optional[str] = None,
    ) -> Optional[str]:
        try:
            root = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            root = None
        try:
            base = Path(root) if root else Path(__file__).resolve().parents[1]
            out_dir = base / "artifacts" / "rigor_decision"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            payload = {
                "schema": "ajax.rigor_decision.v1",
                "created_at": self._iso_utc(),
                "mission_id": mission.mission_id,
                "plan_id": plan_id,
                "decision": decision or {},
                "synthetic_approve": bool(synthetic_approve),
                "note": note,
            }
            path = out_dir / f"decision_{ts}_{mission.mission_id}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _find_recent_capability_gap(
        self,
        *,
        kind: str,
        mission_id: str,
        window_seconds: int,
    ) -> Optional[str]:
        try:
            root = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            root = None
        try:
            base = Path(root) if root else Path(__file__).resolve().parents[1]
            gap_dir = base / "artifacts" / "capability_gaps"
            if not gap_dir.exists():
                return None
            pattern = f"*_{kind}_{mission_id}.json"
            now = time.time()
            candidates = sorted(gap_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in candidates:
                try:
                    if now - path.stat().st_mtime <= int(window_seconds):
                        return str(path)
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _select_plan_repair_provider(
        self,
        *,
        exclude: Optional[set[str]] = None,
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        providers_cfg = (
            self.provider_configs.get("providers", {})
            if isinstance(self.provider_configs, dict)
            else {}
        )
        exclude = exclude or set()
        preferred = ["gemini_cli", "qwen_cloud", "groq", "codex_brain"]

        def _eligible(cfg: Dict[str, Any]) -> bool:
            if not isinstance(cfg, dict) or cfg.get("disabled"):
                return False
            roles = cfg.get("roles") or []
            if isinstance(roles, str):
                roles = [roles]
            roles_l = {str(r).lower() for r in roles if r}
            return bool(roles_l & {"brain", "council"})

        for name in preferred:
            cfg = providers_cfg.get(name)
            if name in exclude or not isinstance(cfg, dict):
                continue
            if _eligible(cfg):
                return name, cfg

        for name, cfg in providers_cfg.items():
            if name in exclude or not isinstance(cfg, dict):
                continue
            if _eligible(cfg):
                return name, cfg
        return None

    def _attempt_plan_repair(
        self,
        *,
        intention: str,
        brain_input: Dict[str, Any],
        draft_plan: Optional[Dict[str, Any]],
        errors: List[str],
        mission: Optional["MissionState"],
        exclude: Optional[set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        self._record_council_invocation(mission, invoked=True, reason="plan_repair")
        choice = self._select_plan_repair_provider(exclude=exclude)
        if not choice:
            return None
        provider_name, provider_cfg = choice
        try:
            from agency.method_pack import AJAX_METHOD_PACK  # type: ignore
        except Exception:
            AJAX_METHOD_PACK = ""

        system_prompt = (
            (AJAX_METHOD_PACK + "\n\n" if AJAX_METHOD_PACK else "")
            + "You are AJAX's Plan Repairer.\n"
            + "Fix and complete the plan JSON so it is VALID and EXECUTABLE.\n"
            + "Return ONLY a JSON object with: plan_id, steps, success_contract.\n"
            + "Every step MUST include: id, intent, preconditions, action, args, evidence_required, success_spec, on_fail.\n"
            + "Every step MUST set on_fail='abort'.\n"
            + "Types:\n"
            + "- preconditions MUST be an object with preconditions.expected_state as an object.\n"
            + "- evidence_required MUST be a list of strings.\n"
            + "- success_spec MUST be an object with success_spec.expected_state as an object.\n"
            + "success_spec.expected_state must contain at least one check (windows/files/meta must_be_active).\n"
            + "Do NOT invent actions or args outside actions_catalog.\n"
            + "If the draft plan is unusable or empty, generate a fresh plan from the intention.\n"
        )
        repair_payload = {
            "intention": intention,
            "errors": errors[-6:],
            "draft_plan": draft_plan or {},
            "actions_catalog": brain_input.get("actions_catalog"),
            "installed_apps": brain_input.get("installed_apps"),
        }
        user_prompt = json.dumps(repair_payload, ensure_ascii=False)
        try:
            plan_json = self._call_brain_provider(
                provider_name,
                provider_cfg,
                system_prompt,
                user_prompt,
                meta={"intention": intention, "planning": True, "purpose": "plan_repair"},
            )
        except Exception:
            return None
        if not isinstance(plan_json, dict):
            return None
        try:
            if normalize_plan is not None:
                allow_non_abort = False
                try:
                    allow_non_abort = bool(
                        (brain_input.get("constraints") or {}).get("allow_non_abort_on_fail")
                    )
                except Exception:
                    allow_non_abort = False
                plan_json, _ = normalize_plan(plan_json, intention, allow_non_abort=allow_non_abort)
            plan_json, repair_meta = self._validate_brain_plan_with_efe_repair(
                plan_json, intention=intention, source="plan_repair"
            )
            if plan_json is None:
                return None
            if repair_meta and isinstance(plan_json, dict):
                meta = plan_json.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                if repair_meta.get("receipt"):
                    meta.setdefault("efe_repair_receipt", repair_meta.get("receipt"))
                if repair_meta.get("reason"):
                    meta.setdefault("efe_repair_reason", repair_meta.get("reason"))
                plan_json["metadata"] = meta
            self._enforce_brain_plan_order(plan_json)
        except Exception:
            return None
        return plan_json

    def _maybe_exclude_provider_after_timeout(self, excluded: set[str]) -> bool:
        attempts = getattr(self, "_last_brain_attempts", None) or []
        if not attempts:
            return False
        last = attempts[-1] if isinstance(attempts, list) else None
        if not isinstance(last, dict):
            return False
        provider = str(last.get("provider") or last.get("id") or "").strip()
        if not provider:
            return False
        error_code = str(last.get("error_code") or "").strip().lower()
        detail = str(last.get("error_detail") or last.get("result") or "").lower()
        if error_code == "client_timeout":
            return False
        if error_code == "timeout" or "timeout" in detail or "timed out" in detail:
            excluded.add(provider)
            return True
        return False

    def _build_knowledge_context(
        self,
        intention: str,
        envelope: Optional["MissionEnvelope"] = None,
        risk_score: float = 0.5,
    ) -> Dict[str, Any]:
        knowledge_context: Dict[str, Any] = {}
        try:
            if leann_context:
                knowledge_context = (
                    leann_context.get_leann_context(intention, mode="persona+system") or {}
                )
        except Exception:
            knowledge_context = {"source": "leann_stub_error"}
        signals: Dict[str, Any] = {}
        try:
            if system_signals:
                signals = system_signals.collect_signals()
        except Exception:
            signals = {"driver_health": "unknown"}
        heartbeat_snapshot: Dict[str, Any] = {}
        try:
            if load_heartbeat_snapshot:
                heartbeat_snapshot = load_heartbeat_snapshot()
        except Exception:
            heartbeat_snapshot = {}
        tool_use_notes: Dict[str, Any] = {}
        try:
            tool_use_notes = self._load_tool_use_notes()
        except Exception:
            tool_use_notes = {}

        tool_plan = None
        if select_tool_plan and load_inventory:
            try:
                risk_flags: Dict[str, Any] = {}
                driver_health = signals.get("driver_health") if isinstance(signals, dict) else None
                if driver_health:
                    risk_flags["driver_health"] = driver_health
                risk_flags["high_risk_intent"] = bool(risk_score >= 0.7)
                tool_plan = select_tool_plan(
                    intention,
                    heartbeat_snapshot,
                    risk_flags,
                    self._load_tool_inventory(),
                    tool_use_notes=tool_use_notes,
                )
                self._apply_tool_plan_requirements(tool_plan, intention, knowledge_context)
            except Exception:
                pass

        if tool_plan is None:
            try:
                ajax_snippets = self._fetch_ajax_history_snippets(intention)
                if ajax_snippets:
                    knowledge_context.setdefault("ajax_history_snippets", ajax_snippets)
                    notes = knowledge_context.get("notes") or []
                    notes.append(f"ajax_history_v1_snippets={len(ajax_snippets)}")
                    knowledge_context["notes"] = notes
            except Exception:
                pass

        if not isinstance(knowledge_context, dict):
            knowledge_context = {"source": "leann_stub_raw", "raw": knowledge_context}
        knowledge_context.setdefault("signals", signals or {"driver_health": "unknown"})

        if self.rag_client and hasattr(self.rag_client, "query"):
            try:
                rag_hits = self.rag_client.query(intention, n_results=3)
                if rag_hits:
                    knowledge_context.setdefault("rag_snippets", rag_hits)
            except Exception:
                pass

        return {
            "knowledge_context": knowledge_context,
            "signals": signals,
            "heartbeat_snapshot": heartbeat_snapshot,
            "tool_use_notes": tool_use_notes,
            "tool_plan": tool_plan,
        }

    def _plan_with_brain(
        self,
        intention: str,
        observation: AjaxObservation,
        feedback: Optional[str] = None,
        envelope: Optional["MissionEnvelope"] = None,
        brain_exclude: Optional[set[str]] = None,
        mission: Optional["MissionState"] = None,
    ) -> AjaxPlan:
        self.log.info("AJAX.plan: usando fallback Brain (LLM-ready) para '%s'", intention)
        max_retries = 1
        errors: list[str] = []
        invalid_same_retry_done = False
        invalid_alt_retry_done = False
        excluded = set(brain_exclude) if brain_exclude is not None else set()
        if mission is not None:
            excluded.update(self._active_provider_cooldowns(mission))
        if mission and self._codex_budget_exceeded(mission):
            providers_cfg = (
                self.provider_configs.get("providers", {})
                if isinstance(self.provider_configs, dict)
                else {}
            )
            codex_providers = {name for name in providers_cfg if str(name).startswith("codex_")}
            excluded.update(codex_providers)
            try:
                self._select_brain_provider(exclude=excluded)
            except Exception:
                used = self._codex_calls_used(mission)
                limit = self._max_codex_calls_per_mission()
                try:
                    if isinstance(mission.notes, dict):
                        mission.notes["codex_budget_exceeded"] = True
                        mission.notes["codex_calls_used"] = used
                        mission.notes["codex_calls_limit"] = limit
                except Exception:
                    pass
                plan_id = f"abort-codex-budget-{int(time.time())}"
                meta = {
                    "intention": intention,
                    "source": "abort:codex_budget",
                    "planning_error": "codex_budget_exceeded",
                    "codex_calls_used": used,
                    "codex_calls_limit": limit,
                }
                return AjaxPlan(
                    id=plan_id,
                    summary=f"Abort (codex budget) for {intention}",
                    steps=[],
                    plan_id=plan_id,
                    metadata=meta,
                    success_spec={"type": "check_last_step_status"},
                )
        last_exc: Optional[Exception] = None
        providers_cfg = (
            self.provider_configs.get("providers", {})
            if isinstance(self.provider_configs, dict)
            else {}
        )
        try:
            from agency.skills.os_inventory import get_installed_apps  # type: ignore

            installed_apps = get_installed_apps()
        except Exception:
            installed_apps = []
        # Contexto LEANN (stub seguro + RAG) y señales de sistema
        knowledge_context: Dict[str, Any] = {}
        try:
            if leann_context:
                knowledge_context = (
                    leann_context.get_leann_context(intention, mode="persona+system") or {}
                )
        except Exception:
            knowledge_context = {"source": "leann_stub_error"}
        signals: Dict[str, Any] = {}
        try:
            if system_signals:
                signals = system_signals.collect_signals()
        except Exception:
            signals = {"driver_health": "unknown"}
        heartbeat_snapshot: Dict[str, Any] = {}
        try:
            if load_heartbeat_snapshot:
                heartbeat_snapshot = load_heartbeat_snapshot()
        except Exception:
            heartbeat_snapshot = {}
        tool_use_notes: Dict[str, Any] = {}
        try:
            tool_use_notes = self._load_tool_use_notes()
        except Exception:
            tool_use_notes = {}
        risk_score = 0.5
        try:
            gov = envelope.governance if envelope else None
            risk_level = str(getattr(gov, "risk_level", "medium") or "medium").strip().lower()
            if risk_level == "low":
                risk_score = 0.2
            elif risk_level == "high":
                risk_score = 0.8
        except Exception:
            risk_score = 0.5

        # Explicit Rigor Selector [NUEVO]
        rigor = None
        if decide_rigor:
            try:
                fail_count = mission.plan_attempts if mission else 0
                risk_level = str((envelope.governance.risk_level if envelope and envelope.governance else "medium") or "medium").lower()
                rigor = decide_rigor(
                    tier="balanced", # Default initial tier
                    risk=risk_level,
                    fail_count=fail_count,
                    cost_mode_override=mission.cost_mode if mission else None,
                    intent_class=(mission.envelope.metadata.get("intent_class") if mission and mission.envelope and isinstance(mission.envelope.metadata, dict) else None),
                )
                self.log.info("Rigor Selector decision: %s (%s)", rigor.strategy.value, rigor.reason)
                if mission and not mission.cost_mode:
                    mission.cost_mode = rigor.cost_mode
                if rigor.strategy == RigorStrategy.COUNCIL:
                    # Forzar modo premium si es COUNCIL
                    override_value = "premium"
            except Exception as exc:
                self.log.warning("Rigor selector failed: %s", exc)

        kctx_res = self._build_knowledge_context(
            intention, envelope=envelope, risk_score=risk_score
        )
        knowledge_context = kctx_res["knowledge_context"]
        signals = kctx_res["signals"]
        heartbeat_snapshot = kctx_res["heartbeat_snapshot"]
        tool_use_notes = kctx_res["tool_use_notes"]
        tool_plan = kctx_res["tool_plan"]

        recent_provider_failures: List[Dict[str, Any]] = []
        brain_failover_logged = False
        brain_failover_path = None

        def _capture_router_trace() -> None:
            trace = getattr(self, "_last_brain_selection_trace", None)
            attempts = getattr(self, "_last_brain_attempts", None)
            if mission:
                self._record_brain_decision_trace(mission, trace, attempts=attempts)
            self._last_brain_selection_trace = None

        for attempt in range(max_retries):
            # HUD de depuración: cuántos snippets aporta LEANN
            try:
                rag_snips = knowledge_context.get("rag_snippets") or []
                ajax_snips = knowledge_context.get("ajax_history_snippets") or []
                src = knowledge_context.get("source", "unknown")
                mode = ""
                if ajax_snips and isinstance(ajax_snips[0], dict):
                    mode = ajax_snips[0].get("source_mode") or ""
                mode_txt = f" (source_mode {mode})" if mode else ""
                print(
                    f"🧠 LEANN[{src}]: rag={len(rag_snips)} ajax_history={len(ajax_snips)} snippets adjuntos{mode_txt}."
                )
            except Exception:
                pass

            actions_payload = (
                self.actions_catalog.to_brain_payload()
                if getattr(self, "actions_catalog", None) is not None
                else {"actions": []}
            )
            envelope_info: Dict[str, Any] = {}
            if envelope:
                envelope_info["mission_id"] = envelope.mission_id
                if envelope.governance:
                    try:
                        envelope_info["governance"] = asdict(envelope.governance)
                    except Exception:
                        envelope_info["governance"] = None
                if envelope.last_error:
                    try:
                        envelope_info["last_error"] = asdict(envelope.last_error)
                    except Exception:
                        envelope_info["last_error"] = None
            compact_intent = self._compact_planner_intent(intention)
            browser_choice = self._choose_browser(installed_apps)
            brain_input = {
                "intention": compact_intent or intention,
                "intent_class": (
                    (
                        envelope.metadata.get("intent_class")
                        if envelope and isinstance(getattr(envelope, "metadata", None), dict)
                        else None
                    )
                    or self._infer_intent_class(intention)
                ),
                "observation": observation.__dict__,
                "capabilities": self.capabilities,
                "actions_catalog": actions_payload,
                "previous_errors": errors,
                "installed_apps": installed_apps,
                "knowledge_context": knowledge_context,
            }
            brain_input["constraints"] = {"browser": browser_choice, "prefer_lab_rehearsal": True}
            # Preflight: Starting XI por rol (si existe) para ruteo determinista y fallback.
            try:
                if envelope and isinstance(getattr(envelope, "metadata", None), dict):
                    sx = envelope.metadata.get("starting_xi")
                    if isinstance(sx, dict):
                        brain_input["starting_xi"] = sx
            except Exception:
                pass
            if tool_plan:
                try:
                    brain_input["tool_plan"] = tool_plan.to_dict()
                    brain_input["tool_affordances"] = tool_plan.affordances
                    brain_input["tool_use_notes"] = tool_plan.tool_use_notes
                except Exception:
                    brain_input["tool_plan"] = None
            if feedback:
                brain_input["feedback"] = feedback
            if envelope_info:
                brain_input["mission_envelope"] = envelope_info
            confidence = None
            try:
                meta_conf = None
                if envelope and isinstance(envelope.metadata, dict):
                    meta_conf = envelope.metadata.get("confidence")
                if isinstance(meta_conf, (int, float)):
                    confidence = float(meta_conf)
            except Exception:
                confidence = None
            brain_input["risk_score"] = risk_score
            brain_input["confidence"] = confidence
            brain_input["fail_count"] = len(errors)
            try:
                if mission and isinstance(mission.notes, dict):
                    pack_state = mission.notes.get("prompt_pack_state")
                    if isinstance(pack_state, dict):
                        brain_input["prompt_pack_state"] = pack_state
            except Exception:
                pass
            provider_name: Optional[str] = None
            premium_rule = getattr(mission, "premium_rule", "if_needed") if mission else "if_needed"
            override_value = self._effective_cost_mode(mission)
            if mission and premium_rule == "now":
                override_value = "premium"
            elif mission and override_value is None and premium_rule == "never":
                override_value = "balanced"

            def _retry_invalid_plan_with_provider(
                provider_name: str,
                provider_cfg: Dict[str, Any],
                *,
                source_label: str,
                prompt_input: Optional[Dict[str, Any]] = None,
            ) -> Optional[AjaxPlan]:
                retry_input = dict(prompt_input or brain_input)
                retry_input["prompt_directive"] = "invalid_plan_retry"
                try:
                    retry_input["fail_count"] = max(int(retry_input.get("fail_count") or 0), 1)
                except Exception:
                    retry_input["fail_count"] = 1
                prompts = build_brain_prompts(retry_input)
                system_prompt = prompts["system"]
                user_prompt = prompts["user"]
                try:
                    self._record_prompt_pack_receipt(
                        mission_id=envelope.mission_id if envelope else None,
                        intention=intention,
                        pack_id=str(prompts.get("pack_id") or "P2_OPS_SAFE"),
                        context_budget=int(prompts.get("context_budget") or 0) or None,
                        escalation_reason=str(prompts.get("escalation_reason") or "") or None,
                        decision=str(prompts.get("decision") or "") or None,
                        reason=str(prompts.get("reason") or "") or None,
                        filters=prompts.get("filters")
                        if isinstance(prompts.get("filters"), dict)
                        else None,
                        confidence=float(prompts.get("confidence"))
                        if isinstance(prompts.get("confidence"), (int, float))
                        else None,
                    )
                except Exception:
                    pass
                try:
                    if mission and isinstance(prompts.get("pack_state"), dict):
                        mission.notes["prompt_pack_state"] = prompts.get("pack_state")
                except Exception:
                    pass
                started = time.time()
                try:
                    plan_json = self._call_brain_provider(
                        provider_name,
                        provider_cfg,
                        system_prompt,
                        user_prompt,
                        meta={
                            "intention": intention,
                            "planning": True,
                            "purpose": source_label,
                            "invalid_plan_retry": True,
                        },
                    )
                except Exception as exc:
                    self._record_brain_attempt(provider_name, provider_cfg, str(exc), started)
                    errors.append(f"{source_label}_error:{exc}")
                    return None
                try:
                    stage_sequence: List[str] = ["parse"]
                    legacy_applied = False
                    if isinstance(plan_json, dict) and normalize_plan is not None:
                        allow_non_abort = False
                        try:
                            allow_non_abort = bool(
                                (brain_input.get("constraints") or {}).get(
                                    "allow_non_abort_on_fail"
                                )
                            )
                        except Exception:
                            allow_non_abort = False
                        if adapt_legacy_plan is not None:
                            plan_json, lmeta = adapt_legacy_plan(plan_json)
                            legacy_applied = bool(lmeta.get("legacy_adapt_applied"))
                            if legacy_applied:
                                stage_sequence.append("legacy_adapt")
                        stage_sequence.append("normalize")
                        plan_json, nmeta = normalize_plan(
                            plan_json, compact_intent or intention, allow_non_abort=allow_non_abort
                        )
                        try:
                            if nmeta.get("coerced_fields"):
                                if (
                                    isinstance(getattr(self, "_last_brain_attempts", None), list)
                                    and self._last_brain_attempts
                                ):
                                    last_attempt = self._last_brain_attempts[-1]
                                    if isinstance(last_attempt, dict):
                                        stage_sequence.append("coerce_safety")
                                        last_attempt["stage"] = "coerce_safety"
                                        last_attempt["coerced_fields"] = nmeta.get("coerced_fields")
                                        last_attempt["legacy_adapt_applied"] = legacy_applied
                                        last_attempt["stage_sequence"] = stage_sequence + [
                                            "validate"
                                        ]
                        except Exception:
                            pass
                    stage_sequence.append("validate")
                    plan_json, repair_meta = self._validate_brain_plan_with_efe_repair(
                        plan_json, intention=intention, source=f"brain_invalid_retry:{source_label}"
                    )
                    if plan_json is None:
                        return self._build_missing_efe_plan(
                            intention=intention,
                            source=f"brain_invalid_retry:{source_label}",
                            receipt_path=(repair_meta or {}).get("receipt"),
                            reason=(repair_meta or {}).get("reason"),
                            errors=errors[-6:] if isinstance(errors, list) else None,
                        )
                    if repair_meta and isinstance(plan_json, dict):
                        meta = plan_json.get("metadata")
                        if not isinstance(meta, dict):
                            meta = {}
                        if repair_meta.get("receipt"):
                            meta.setdefault("efe_repair_receipt", repair_meta.get("receipt"))
                        if repair_meta.get("reason"):
                            meta.setdefault("efe_repair_reason", repair_meta.get("reason"))
                        plan_json["metadata"] = meta
                    self._enforce_brain_plan_order(plan_json)
                except Exception as exc:
                    self._record_brain_attempt(
                        provider_name, provider_cfg, f"invalid_brain_plan:{exc}", started
                    )
                    errors.append(f"{source_label}_invalid:{exc}")
                    return None
                self._record_brain_attempt(provider_name, provider_cfg, "ok", started)
                plan_id = (
                    plan_json.get("plan_id") or f"plan-repair-{source_label}-{int(time.time())}"
                )
                steps = plan_json.get("steps", [])
                raw_contract = plan_json.get("success_contract")
                plan_obj = AjaxPlan(
                    id=plan_id,
                    summary=f"Plan Brain retry ({source_label}) para {intention}",
                    steps=steps,
                    plan_id=plan_id,
                    metadata={
                        "intention": intention,
                        "source": f"brain_invalid_retry:{source_label}",
                    },
                    success_spec=plan_json.get("success_spec") or raw_contract,
                )
                if not plan_obj.success_spec:
                    plan_obj.success_spec = {"type": "check_last_step_status"}
                if raw_contract:
                    plan_obj.metadata["success_contract"] = raw_contract
                if self.council and CouncilVerdict:
                    try:
                        catalog_payload = (
                            self.actions_catalog.to_brain_payload()
                            if getattr(self, "actions_catalog", None)
                            else {}
                        )
                        self._record_council_invocation(
                            mission, invoked=True, reason="plan_review_retry"
                        )
                        with self._override_cost_mode(override_value):
                            verdict = self.council.review_plan(
                                intention,
                                plan_json,
                                context=brain_input,
                                actions_catalog=catalog_payload,
                            )
                        verdict = self._normalize_council_verdict(verdict)
                        if verdict and not verdict.approved:
                            self.log.warning("Council rejected retry plan: %s", verdict.feedback)
                            errors.append(f"council_reject:{verdict.feedback}")
                            if envelope and MissionError:
                                try:
                                    envelope.last_error = MissionError(
                                        kind="plan_error",
                                        step_id=None,
                                        reason=f"council_reject:{verdict.feedback}",
                                    )
                                except Exception:
                                    pass
                            try:
                                plan_obj.metadata["planning_error"] = (
                                    f"council_reject:{verdict.feedback}"
                                )
                                plan_obj.metadata["council_verdict"] = verdict.__dict__
                                plan_obj.metadata["council_vetoed"] = True
                            except Exception:
                                pass
                            return self._apply_rigor_to_plan(plan_obj, rigor)
                        if verdict:
                            try:
                                plan_obj.metadata["council_verdict"] = verdict.__dict__
                            except Exception:
                                pass
                    except Exception as exc:
                        self._record_council_invocation(
                            mission, invoked=True, reason="plan_review_retry_failed"
                        )
                        self.log.warning("Council review (retry) failed: %s", exc)
                return self._apply_rigor_to_plan(plan_obj, rigor)

            try:
                # Mejor esfuerzo: capturar el nombre del provider actual para poder excluirlo
                # en caso de rate_limit/unavailable cuando _call_brain_llm está monkeypatcheado en tests.
                try:
                    provider_name, _ = self._select_brain_provider(exclude=excluded)
                except Exception:
                    provider_name = None
                with self._override_cost_mode(override_value):
                    brain_output = self._call_brain_llm(brain_input, exclude=excluded)
                last_brain_output = brain_output if isinstance(brain_output, dict) else None
                try:
                    if mission and isinstance(
                        getattr(self, "_last_prompt_pack_decision", None), dict
                    ):
                        pack_state = self._last_prompt_pack_decision.get("pack_state")
                        if isinstance(pack_state, dict):
                            mission.notes["prompt_pack_state"] = pack_state
                except Exception:
                    pass
                _capture_router_trace()
                if isinstance(brain_output, dict) and normalize_plan is not None:
                    allow_non_abort = False
                    try:
                        allow_non_abort = bool(
                            (brain_input.get("constraints") or {}).get("allow_non_abort_on_fail")
                        )
                    except Exception:
                        allow_non_abort = False
                    stage_sequence: List[str] = ["parse"]
                    legacy_applied = False
                    if adapt_legacy_plan is not None:
                        brain_output, lmeta = adapt_legacy_plan(brain_output)
                        legacy_applied = bool(lmeta.get("legacy_adapt_applied"))
                        if legacy_applied:
                            stage_sequence.append("legacy_adapt")
                    stage_sequence.append("normalize")
                    brain_output, nmeta = normalize_plan(
                        brain_output, compact_intent or intention, allow_non_abort=allow_non_abort
                    )
                    last_brain_output = brain_output
                    try:
                        if nmeta.get("coerced_fields"):
                            if (
                                isinstance(getattr(self, "_last_brain_attempts", None), list)
                                and self._last_brain_attempts
                            ):
                                last_attempt = self._last_brain_attempts[-1]
                                if isinstance(last_attempt, dict):
                                    stage_sequence.append("coerce_safety")
                                    last_attempt["stage"] = "coerce_safety"
                                    last_attempt["coerced_fields"] = nmeta.get("coerced_fields")
                                    last_attempt["legacy_adapt_applied"] = legacy_applied
                                    last_attempt["stage_sequence"] = stage_sequence
                    except Exception:
                        pass
                # If steps missing/empty => retry once with a short constraint.
                non_empty_retry_done = False
                if (
                    isinstance(brain_output, dict)
                    and not brain_output.get("steps")
                    and not non_empty_retry_done
                ):
                    retry_input = dict(brain_input)
                    prev = list(retry_input.get("previous_errors") or [])
                    prev.append("empty_plan")
                    retry_input["previous_errors"] = prev
                    retry_input["feedback"] = "MUST return >=1 step."
                    with self._override_cost_mode(override_value):
                        brain_output = self._call_brain_llm(retry_input, exclude=excluded)
                    last_brain_output = brain_output if isinstance(brain_output, dict) else None
                    if isinstance(brain_output, dict) and normalize_plan is not None:
                        allow_non_abort = False
                        try:
                            allow_non_abort = bool(
                                (brain_input.get("constraints") or {}).get(
                                    "allow_non_abort_on_fail"
                                )
                            )
                        except Exception:
                            allow_non_abort = False
                        stage_sequence = ["parse"]
                        legacy_applied = False
                        if adapt_legacy_plan is not None:
                            brain_output, lmeta = adapt_legacy_plan(brain_output)
                            legacy_applied = bool(lmeta.get("legacy_adapt_applied"))
                            if legacy_applied:
                                stage_sequence.append("legacy_adapt")
                        stage_sequence.append("normalize")
                        brain_output, nmeta = normalize_plan(
                            brain_output,
                            compact_intent or intention,
                            allow_non_abort=allow_non_abort,
                        )
                        last_brain_output = brain_output
                        try:
                            if nmeta.get("coerced_fields"):
                                if (
                                    isinstance(getattr(self, "_last_brain_attempts", None), list)
                                    and self._last_brain_attempts
                                ):
                                    last_attempt = self._last_brain_attempts[-1]
                                    if isinstance(last_attempt, dict):
                                        stage_sequence.append("coerce_safety")
                                        last_attempt["stage"] = "coerce_safety"
                                        last_attempt["coerced_fields"] = nmeta.get("coerced_fields")
                                        last_attempt["legacy_adapt_applied"] = legacy_applied
                                        last_attempt["stage_sequence"] = stage_sequence
                        except Exception:
                            pass
                    non_empty_retry_done = True
                try:
                    if (
                        isinstance(getattr(self, "_last_brain_attempts", None), list)
                        and self._last_brain_attempts
                    ):
                        last_attempt = self._last_brain_attempts[-1]
                        if isinstance(last_attempt, dict):
                            seq = last_attempt.get("stage_sequence")
                            if isinstance(seq, list):
                                seq2 = [str(s) for s in seq if isinstance(s, str) and s.strip()]
                                if "validate" not in seq2:
                                    seq2.append("validate")
                                last_attempt["stage_sequence"] = seq2
                except Exception:
                    pass
                plan_json, repair_meta = self._validate_brain_plan_with_efe_repair(
                    brain_output, intention=intention, source="brain"
                )
                if plan_json is None:
                    return self._build_missing_efe_plan(
                        intention=intention,
                        source="brain",
                        receipt_path=(repair_meta or {}).get("receipt"),
                        reason=(repair_meta or {}).get("reason"),
                        errors=errors[-6:] if isinstance(errors, list) else None,
                    )
                if repair_meta and isinstance(plan_json, dict):
                    meta = plan_json.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {}
                    if repair_meta.get("receipt"):
                        meta.setdefault("efe_repair_receipt", repair_meta.get("receipt"))
                    if repair_meta.get("reason"):
                        meta.setdefault("efe_repair_reason", repair_meta.get("reason"))
                    plan_json["metadata"] = meta
                self._enforce_brain_plan_order(plan_json)
                plan_id = plan_json.get("plan_id") or f"plan-brain-{int(time.time())}"
                steps = plan_json.get("steps", [])
                raw_contract = plan_json.get("success_contract")
                plan_obj = AjaxPlan(
                    id=plan_id,
                    summary=f"Plan Brain para {intention}",
                    steps=steps,
                    plan_id=plan_id,
                    metadata={"intention": intention, "source": "brain"},
                    success_spec=plan_json.get("success_spec") or raw_contract,
                )
                if not plan_obj.success_spec:
                    plan_obj.success_spec = {"type": "check_last_step_status"}
                if raw_contract:
                    plan_obj.metadata["success_contract"] = raw_contract
                if tool_plan:
                    try:
                        plan_obj.metadata["tool_plan"] = tool_plan.to_dict()
                        plan_obj.metadata["tool_plan_incomplete"] = bool(tool_plan.incomplete)
                    except Exception:
                        pass
                # Council review
                if self.council and CouncilVerdict:
                    try:
                        catalog_payload = (
                            self.actions_catalog.to_brain_payload()
                            if getattr(self, "actions_catalog", None)
                            else {}
                        )
                        self._record_council_invocation(mission, invoked=True, reason="plan_review")
                        with self._override_cost_mode(override_value):
                            verdict: CouncilVerdict = self.council.review_plan(
                                intention,
                                plan_json,
                                context=brain_input,
                                actions_catalog=catalog_payload,
                            )
                        verdict = self._normalize_council_verdict(verdict)
                        if verdict and not verdict.approved:
                            self.log.warning("Council rejected plan: %s", verdict.feedback)
                            errors.append(f"council_reject:{verdict.feedback}")
                            feedback = f"Council says: {verdict.feedback}"
                            if envelope and MissionError:
                                try:
                                    envelope.last_error = MissionError(
                                        kind="plan_error",
                                        step_id=None,
                                        reason=f"council_reject:{verdict.feedback}",
                                    )
                                except Exception:
                                    pass
                            # Persistencia consejo
                            try:
                                art = self.config.root_dir / "artifacts"
                                art.mkdir(parents=True, exist_ok=True)
                                (art / "council_last_session.md").write_text(
                                    f"# Council session\nIntention: {intention}\nPlan:\n{json.dumps(plan_json, ensure_ascii=False, indent=2)}\nFeedback: {verdict.feedback}",
                                    encoding="utf-8",
                                )
                            except Exception:
                                pass
                            # Conservar el plan propuesto para poder derivarlo a LAB (no-consenso != muerte).
                            try:
                                plan_obj.metadata["planning_error"] = (
                                    f"council_reject:{verdict.feedback}"
                                )
                                plan_obj.metadata["council_verdict"] = verdict.__dict__
                                plan_obj.metadata["council_vetoed"] = True
                            except Exception:
                                pass
                            return plan_obj
                        if verdict:
                            try:
                                plan_obj.metadata["council_verdict"] = verdict.__dict__
                            except Exception:
                                pass
                    except Exception as exc:
                        self._record_council_invocation(
                            mission, invoked=True, reason="plan_review_failed"
                        )
                        self.log.warning("Council review failed: %s", exc)
                recent_provider_failures = self._record_provider_failures_from_attempts(
                    mission,
                    stage="plan",
                    intention=intention,
                )
                return self._apply_rigor_to_plan(plan_obj, rigor)
            except ValueError as ve:
                _capture_router_trace()
                msg = str(ve)
                last_exc = ve
                if not msg.startswith("invalid_brain_plan:"):
                    break
                # Mark last attempt stage/excerpt for receipts.
                try:
                    if (
                        isinstance(getattr(self, "_last_brain_attempts", None), list)
                        and self._last_brain_attempts
                    ):
                        last_attempt = self._last_brain_attempts[-1]
                        if isinstance(last_attempt, dict):
                            last_attempt["stage"] = "validate"
                            if isinstance(last_brain_output, dict):
                                raw_ex = last_brain_output.get("_raw_plan_excerpt")
                                if isinstance(raw_ex, str) and raw_ex.strip():
                                    last_attempt["raw_plan_excerpt"] = raw_ex[:500]
                except Exception:
                    pass
                attempts_used = [
                    a
                    for a in (getattr(self, "_last_brain_attempts", []) or [])
                    if isinstance(a, dict)
                ]
                if len(attempts_used) >= 2:
                    errors.append(msg)
                    break
                # Deterministic one-pass repair only for missing_fields
                if "invalid_brain_plan:missing_fields" in msg and not invalid_same_retry_done:
                    last_attempt = None
                    for ent in reversed(attempts_used):
                        if isinstance(ent, dict) and (ent.get("provider") or ent.get("id")):
                            last_attempt = ent
                            break
                    provider = str(
                        (last_attempt or {}).get("provider") or (last_attempt or {}).get("id") or ""
                    ).strip()
                    if provider:
                        # Deterministic repair: re-ask SAME provider once with a strict-schema reminder.
                        # We restrict the router to the chosen provider by excluding all others.
                        try:
                            exclude_repair = set(providers_cfg.keys()) - {provider}
                        except Exception:
                            exclude_repair = None
                        retry_input = {
                            "intention": brain_input.get("intention"),
                            "intent_class": brain_input.get("intent_class"),
                            "constraints": brain_input.get("constraints"),
                            "actions_catalog": brain_input.get("actions_catalog"),
                            "installed_apps": brain_input.get("installed_apps"),
                            "previous_errors": [msg],
                            "prompt_directive": "repair_pass_ultrashort",
                        }
                        try:
                            plan_json = self._call_brain_llm(retry_input, exclude=exclude_repair)
                            plan_json, repair_meta = self._validate_brain_plan_with_efe_repair(
                                plan_json,
                                intention=intention,
                                source="brain_repair_ultrashort",
                            )
                            if plan_json is None:
                                errors.append(msg)
                                break
                            if repair_meta and isinstance(plan_json, dict):
                                meta = plan_json.get("metadata")
                                if not isinstance(meta, dict):
                                    meta = {}
                                if repair_meta.get("receipt"):
                                    meta.setdefault(
                                        "efe_repair_receipt", repair_meta.get("receipt")
                                    )
                                if repair_meta.get("reason"):
                                    meta.setdefault("efe_repair_reason", repair_meta.get("reason"))
                                plan_json["metadata"] = meta
                            self._enforce_brain_plan_order(plan_json)
                        except Exception:
                            errors.append(msg)
                            break
                        plan_id = (
                            plan_json.get("plan_id") or f"plan-repair-ultrashort-{int(time.time())}"
                        )
                        steps = plan_json.get("steps", [])
                        raw_contract = plan_json.get("success_contract")
                        plan_obj = AjaxPlan(
                            id=plan_id,
                            summary=f"Plan reparado (ultrashort) para {intention}",
                            steps=steps,
                            plan_id=plan_id,
                            metadata={"intention": intention, "source": "brain_repair_ultrashort"},
                            success_spec=plan_json.get("success_spec") or raw_contract,
                        )
                        if not plan_obj.success_spec:
                            plan_obj.success_spec = {"type": "check_last_step_status"}
                        if raw_contract:
                            plan_obj.metadata["success_contract"] = raw_contract
                        invalid_same_retry_done = True
                        return self._apply_rigor_to_plan(plan_obj, rigor)
                # One deterministic LLM repair pass (JSON-only) after normalization+validation fails.
                if not invalid_alt_retry_done and last_brain_output is not None:
                    repaired = self._attempt_plan_repair(
                        intention=intention,
                        brain_input=brain_input,
                        draft_plan=last_brain_output,
                        errors=errors + [msg],
                        mission=mission,
                        exclude=excluded,
                    )
                    if isinstance(repaired, dict):
                        if normalize_plan is not None:
                            repaired, _ = normalize_plan(repaired, compact_intent or intention)
                        try:
                            repaired, repair_meta = self._validate_brain_plan_with_efe_repair(
                                repaired, intention=intention, source="brain_repair"
                            )
                            if repaired is None:
                                errors.append(msg)
                                break
                            if repair_meta and isinstance(repaired, dict):
                                meta = repaired.get("metadata")
                                if not isinstance(meta, dict):
                                    meta = {}
                                if repair_meta.get("receipt"):
                                    meta.setdefault(
                                        "efe_repair_receipt", repair_meta.get("receipt")
                                    )
                                if repair_meta.get("reason"):
                                    meta.setdefault("efe_repair_reason", repair_meta.get("reason"))
                                repaired["metadata"] = meta
                            self._enforce_brain_plan_order(repaired)
                        except Exception:
                            errors.append(msg)
                            break
                        plan_id = repaired.get("plan_id") or f"plan-repair-{int(time.time())}"
                        steps = repaired.get("steps", [])
                        raw_contract = repaired.get("success_contract")
                        plan_obj = AjaxPlan(
                            id=plan_id,
                            summary=f"Plan reparado para {intention}",
                            steps=steps,
                            plan_id=plan_id,
                            metadata={"intention": intention, "source": "brain_repair"},
                            success_spec=repaired.get("success_spec") or raw_contract,
                        )
                        if not plan_obj.success_spec:
                            plan_obj.success_spec = {"type": "check_last_step_status"}
                        if raw_contract:
                            plan_obj.metadata["success_contract"] = raw_contract
                        invalid_alt_retry_done = True
                        return self._apply_rigor_to_plan(plan_obj, rigor)
                errors.append(msg)
                recent_provider_failures = self._record_provider_failures_from_attempts(
                    mission,
                    stage="plan",
                    intention=intention,
                )
                self._maybe_exclude_provider_after_timeout(excluded)
                plan_repair_attempted = True
                break
            except RuntimeError as exc:
                _capture_router_trace()
                last_exc = exc
                recent_provider_failures = self._record_provider_failures_from_attempts(
                    mission,
                    stage="plan",
                    intention=intention,
                )
                self._maybe_exclude_provider_after_timeout(excluded)
                msg = str(exc).lower()
                if "brain_failover_exhausted" in msg:
                    if not brain_failover_logged:
                        brain_failover_path = self._record_brain_failover_receipt(
                            mission=mission, intention=intention
                        )
                        brain_failover_logged = True
                    errors.append(msg)
                    break
                # Si es un rate limit / unavailable, intenta otro provider
                if any(tok in msg for tok in ["429", "rate_limit", "rate limit", "unavailable"]):
                    try:
                        if provider_name:
                            excluded.add(provider_name)
                    except Exception:
                        pass
                    errors.append(f"rate_limited_or_unavailable:{msg}")
                    continue
                break
            except Exception as exc:
                _capture_router_trace()
                last_exc = exc
                recent_provider_failures = self._record_provider_failures_from_attempts(
                    mission,
                    stage="plan",
                    intention=intention,
                )
                self._maybe_exclude_provider_after_timeout(excluded)
                msg = str(exc).lower()
                if "brain_failover_exhausted" in msg and AllBrainsFailed:
                    if not brain_failover_logged:
                        brain_failover_path = self._record_brain_failover_receipt(
                            mission=mission, intention=intention
                        )
                        brain_failover_logged = True
                    errors.append(msg)
                    break
                # Si es un rate limit / unavailable, intenta otro provider
                if any(tok in msg for tok in ["429", "rate_limit", "rate limit", "unavailable"]):
                    try:
                        if "provider_name" in locals():
                            excluded.add(provider_name)
                    except Exception:
                        pass
                    errors.append(f"rate_limited_or_unavailable:{msg}")
                    continue
                break

        if not recent_provider_failures:
            recent_provider_failures = self._record_provider_failures_from_attempts(
                mission,
                stage="plan",
                intention=intention,
            )
            self._maybe_exclude_provider_after_timeout(excluded)
        self._log_brain_failure(intention=intention, errors=errors, exc=last_exc)

        def _annotate_brain_failed_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
            meta = dict(meta or {})
            meta["planning_error"] = "brain_failed_no_plan"
            meta["brain_failed_no_plan"] = True
            meta["skip_council_review"] = True
            meta["errors"] = list(errors or [])
            try:
                if mission and isinstance(mission.notes, dict):
                    trace_path = mission.notes.get("router_trace_path")
                    if isinstance(trace_path, str) and trace_path:
                        meta["router_trace_path"] = trace_path
            except Exception:
                pass
            return meta

        if not brain_failover_logged:
            brain_failover_path = self._record_brain_failover_receipt(
                mission=mission, intention=intention
            )
            brain_failover_logged = True
        plan_id = f"abort-no-plan-{int(time.time())}"
        metadata = _annotate_brain_failed_meta(
            {
                "intention": intention,
                "source": "abort:no_plan",
                "errors": errors,
                "abort_reason": "no_valid_plan",
            }
        )
        if brain_failover_path:
            metadata["brains_failures_path"] = brain_failover_path
        provider_failures = recent_provider_failures or []
        if provider_failures:
            metadata["provider_failures"] = provider_failures
            try:
                severe_codes = {
                    "SANDBOX_LISTEN_EPERM",
                    "AUTH_REQUIRED",
                    "HTTP_403",
                    "ENV_BLOCKED",
                    "BINARY_MISSING",
                }
                codes = {
                    str(ent.get("code") or "").strip().upper()
                    for ent in provider_failures
                    if isinstance(ent, dict)
                }
                allow_force = True
                try:
                    if failure_force_ask_user_on_severe is not None:
                        allow_force = bool(
                            failure_force_ask_user_on_severe(
                                self._provider_failure_policy(), default=True
                            )
                        )
                except Exception:
                    allow_force = True
                if allow_force and codes.intersection(severe_codes):
                    metadata["force_ask_user"] = True
            except Exception:
                pass
        return self._apply_rigor_to_plan(
            AjaxPlan(
                id=plan_id,
                summary=f"Abort (no valid plan) for {intention}",
                steps=[],
                plan_id=plan_id,
                metadata=metadata,
                success_spec={"type": "check_last_step_status"},
            ),
            rigor,
        )

    def _choose_browser(self, installed_apps: Optional[list[str]] = None) -> str:
        if os_inventory:
            found = os_inventory.discover_browsers(self.root_dir)
            if found:
                # Si tenemos installed_apps, priorizar los que están ahí
                if installed_apps:
                    apps_l = {a.lower() for a in installed_apps}
                    for b in found:
                        if os.path.basename(b).lower() in apps_l or b.lower() in apps_l:
                            return b
                return found[0]
        
        # Fallback si os_inventory no está o no encuentra nada
        candidates = [
            "brave.exe",
            "chrome.exe",
            "msedge.exe",
            "firefox.exe",
        ]
        apps = [a.lower() for a in (installed_apps or [])]
        for c in candidates:
            if c.lower() in apps:
                return c
        return "brave.exe"

    def _log_brain_failure(
        self, intention: str, errors: list[str], exc: Optional[Exception]
    ) -> None:
        try:
            self.log.error(
                "Brain planning failed for '%s': errors=%s exc=%s", intention, errors, exc
            )
        except Exception:
            pass

    def _make_fallback_log_plan(self, intention: str, reason: str, errors: list[str]) -> AjaxPlan:
        plan_id = f"fallback-{int(time.time())}"
        meta = {
            "intention": intention,
            "reason": reason,
            "errors": errors,
            "source": "fallback_log",
        }
        return AjaxPlan(
            id=plan_id,
            summary=f"Log fallback for {intention}",
            steps=[],
            plan_id=plan_id,
            metadata=meta,
        )

    def _fetch_ajax_history_snippets(
        self, intention: str, top_k: int = 5, *, force: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Paso 0: consulta ajax_history_v1 si la intención suena a AJAX/HUB.
        Devuelve lista vacía en fallo.
        """
        intent_lower = intention.lower()
        force_env = os.getenv("AJAX_HISTORY_FORCE", "").lower() in {"1", "true", "yes", "always"}
        if force_env and not force:
            force = True
            try:
                print("⚙️  DEBUG: AJAX_HISTORY_FORCE activo (debug only).")
            except Exception:
                pass
        hints_raw = os.getenv("AJAX_HISTORY_HINTS", "ajax,hub,officebot,leann,rag,querydrop")
        hints = [h.strip().lower() for h in hints_raw.split(",") if h.strip()]
        if not force and hints:
            if not any(h in intent_lower for h in hints):
                return []
        collection = os.getenv("AJAX_HISTORY_COLLECTION", "ajax_history_v1.leann")
        try:
            return query_leann(collection, intention, top_k=top_k) or []
        except Exception:
            return []

    def _chat_leann_mode(self) -> str:
        raw = (os.getenv("AJAX_CHAT_LEANN") or "").strip().lower()
        if raw in {"0", "false", "off", "no"}:
            return "off"
        if raw in {"1", "true", "on", "yes"}:
            return "on"
        if raw in {"auto", ""}:
            return "auto"
        return "auto"

    def _chat_leann_confidence_threshold(self) -> float:
        raw = os.getenv("AJAX_CHAT_LEANN_CONFIDENCE_T")
        if raw is None:
            return 0.55
        try:
            return float(raw)
        except Exception:
            return 0.55

    def _chat_leann_should_use(
        self,
        *,
        ambiguity: bool,
        action_required: bool,
        confidence: Optional[float],
    ) -> tuple[bool, str, str]:
        mode = self._chat_leann_mode()
        if mode == "off":
            return False, "forced_off", mode
        if mode == "on":
            return True, "forced_on", mode
        if ambiguity:
            return True, "ambiguity", mode
        if action_required:
            return True, "action_required", mode
        thresh = self._chat_leann_confidence_threshold()
        if confidence is not None and confidence < thresh:
            return True, "low_confidence", mode
        return False, "auto_skip", mode

    def _chat_leann_profile_card(
        self, intention: str, user_key: str
    ) -> tuple[Optional[Dict[str, Any]], Optional[float]]:
        if leann_context is None:
            return None, None
        now = time.time()
        ttl_s = 21600.0
        cache = self._chat_leann_profile_cache or {}
        key = f"profile|{user_key}"
        if cache.get("key") == key and (now - float(cache.get("ts") or 0.0)) <= ttl_s:
            return cache.get("card"), 0.0
        t0 = time.monotonic()
        try:
            card = (
                leann_context.get_leann_context(intention, mode="persona", timeout_seconds=0.8)
                or {}
            )
        except Exception:
            card = {}
        t_leann = round((time.monotonic() - t0) * 1000, 2)
        self._chat_leann_profile_cache = {"ts": now, "key": key, "card": card}
        return card or None, t_leann

    def _chat_leann_profile_text(self, card: Optional[Dict[str, Any]]) -> str:
        if not isinstance(card, dict):
            return ""
        notes = card.get("notes") if isinstance(card.get("notes"), list) else []
        related = (
            card.get("related_projects") if isinstance(card.get("related_projects"), list) else []
        )
        snippets = card.get("rag_snippets") if isinstance(card.get("rag_snippets"), list) else []
        lines: List[str] = []
        for item in notes[:2]:
            if item:
                lines.append(f"- {str(item)[:200]}")
        for item in related[:2]:
            if item:
                lines.append(f"- proyecto: {str(item)[:120]}")
        for item in snippets[:1]:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or ""
            else:
                txt = str(item)
            if txt:
                lines.append(f"- {txt[:200]}")
        return "\n".join(lines).strip()

    def _chat_leann_episodic_snippets(
        self, intention: str, top_k: int = 2
    ) -> tuple[List[Dict[str, Any]], Optional[float]]:
        if not self._chat_has_history_index() and not bool(
            self._chat_leann_cache.get("had_snippets")
        ):
            return [], None
        now = time.time()
        key = f"{intention}|{top_k}"
        ttl_s = 600.0
        cache = self._chat_leann_cache or {}
        if cache.get("key") == key and (now - float(cache.get("ts") or 0.0)) <= ttl_s:
            return list(cache.get("snippets") or []), 0.0
        t0 = time.monotonic()
        snippets = self._fetch_ajax_history_snippets(intention, top_k=top_k)
        t_leann = round((time.monotonic() - t0) * 1000, 2)
        self._chat_leann_cache = {
            "ts": now,
            "key": key,
            "snippets": snippets,
            "had_snippets": bool(snippets),
        }
        return snippets, t_leann

    def _chat_has_history_index(self) -> bool:
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        candidates = [
            root_dir / "ajax_history_v1.leann.passages.idx",
            root_dir / "ajax_history_v1.leann.passages.jsonl",
            root_dir / "ajax_history_v1.leann.meta.json",
        ]
        for path in candidates:
            try:
                if path.exists() and path.stat().st_size > 0:
                    return True
            except Exception:
                continue
        return False

    def providers_preflight(self, *, requested_tier: Optional[str] = None) -> Dict[str, Any]:
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        providers_cfg = (self.provider_configs or {}).get("providers") or {}
        req_tier = (requested_tier or os.getenv("AJAX_COST_MODE") or "premium").strip().lower()
        # Fast preflight is used by interactive chat to avoid blocking on live provider probes
        # (some CLIs can take minutes due to auth or network).
        fast_mode = (os.getenv("AJAX_PREFLIGHT_FAST") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        receipt: Dict[str, Any] = {
            "schema": "ajax.providers_preflight.v1",
            "ts": time.time(),
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "requested_tier": req_tier,
        }

        if fast_mode:
            quota_gate = QuotaGate(root_dir=Path(root_dir))
            quota_state = quota_gate.load_quota_state()
            ignore_quota = (
                os.getenv("AJAX_IGNORE_QUOTA") or os.getenv("AJAX_FORCE_PREMIUM") or ""
            ).strip().lower() in {"1", "true", "yes"}

            if quota_gate.is_stale(quota_state) and not ignore_quota:
                # Intento de refresco corto para no bloquear chat
                quota_state = quota_gate.refresh_quota(timeout=2.0)

            status_path = Path(root_dir) / "artifacts" / "health" / "providers_status.json"
            ledger_doc: Dict[str, Any] = {}
            status_doc: Dict[str, Any] = {}
            scoreboard_doc: Dict[str, Any] = {}
            try:
                if status_path.exists():
                    status_doc = json.loads(status_path.read_text(encoding="utf-8"))
                    if not isinstance(status_doc, dict):
                        status_doc = {}
            except Exception:
                status_doc = {}
            try:
                from agency import provider_scoreboard  # type: ignore

                scoreboard_doc = provider_scoreboard.load_scoreboard(
                    Path(root_dir) / "artifacts" / "state" / "provider_scoreboard.json"
                )
            except Exception:
                scoreboard_doc = {}
            if ProviderLedger is not None:
                try:
                    ledger_doc = (
                        ProviderLedger(
                            root_dir=Path(root_dir), provider_configs=self.provider_configs
                        ).refresh(timeout_s=1.5)
                        or {}
                    )
                except Exception:
                    ledger_doc = {}

            pool: List[str] = []
            cloud_eligible = False
            for name, cfg in (self.provider_configs or {}).get("providers", {}).items():
                if not isinstance(cfg, dict) or cfg.get("disabled"):
                    continue

                # QuotaGate Check
                if not ignore_quota:
                    q_status, q_reason = quota_gate.get_provider_status(str(name), quota_state)
                    if q_status == "false":
                        self.log.info("AJAX.quota: %s excluido por %s", name, q_reason)
                        continue

                    # Si es un cloud provider y no es unknown/false, marcamos que hay cloud elegible
                    if q_status == "true" and str(name) not in {
                        "lmstudio",
                        "lmstudio_vision",
                        "static_local",
                    }:
                        cloud_eligible = True
                else:
                    cloud_eligible = True

                roles = cfg.get("roles") or []

                if isinstance(roles, str):
                    roles = [roles]
                if "brain" in {str(r).lower() for r in roles if r}:
                    pool.append(str(name))

            rail = (
                os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"
            )
            fallback_chain: List[str] = []
            if provider_ranker is not None:
                try:
                    fallback_chain = provider_ranker.rank_providers(
                        pool,
                        providers_cfg=providers_cfg,
                        status=status_doc,
                        scoreboard=scoreboard_doc,
                        prefer_tier=req_tier,
                        role="brain",
                        rail=rail,
                        risk_level="medium",
                    )
                except Exception:
                    fallback_chain = []
            chosen_provider = fallback_chain[0] if fallback_chain else None

            if not cloud_eligible and chosen_provider and "lmstudio" in chosen_provider:
                print("☁️  Sin presupuesto cloud elegible; paso a LMStudio.")

            # When fast_mode is enabled we do not attempt auto-repair (e.g., autostart LMStudio),
            # we only return the best-effort selection based on cached status + current ledger.
            receipt.update(
                {
                    "mode": "fast",
                    "rail": rail,
                    "chosen_tier": req_tier,
                    "chosen_provider": chosen_provider,
                    "fallback_chain": fallback_chain,
                    "providers_status_path": str(status_path),
                    "providers_status_updated_utc": status_doc.get("updated_utc")
                    if isinstance(status_doc, dict)
                    else None,
                    "provider_ledger_path": ledger_doc.get("path")
                    if isinstance(ledger_doc, dict)
                    else None,
                    "provider_ledger_updated_utc": ledger_doc.get("updated_utc")
                    if isinstance(ledger_doc, dict)
                    else None,
                }
            )
            try:
                doc_dir = (
                    Path(root_dir)
                    / "artifacts"
                    / "doctor"
                    / f"providers_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                )
                doc_dir.mkdir(parents=True, exist_ok=True)
                (doc_dir / "receipt.json").write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            except Exception:
                pass
            return receipt

        def _refresh_state() -> Dict[str, Any]:
            status_doc: Dict[str, Any] = {}
            if ProviderBreathingLoop is not None:
                try:
                    loop = ProviderBreathingLoop(
                        root_dir=Path(root_dir), provider_configs=self.provider_configs
                    )
                    status_doc = loop.run_once(roles=["brain", "council", "vision", "scout"])
                except Exception:
                    status_doc = {}
            ledger_doc: Dict[str, Any] = {}
            if ProviderLedger is not None:
                try:
                    ledger_doc = (
                        ProviderLedger(
                            root_dir=Path(root_dir), provider_configs=self.provider_configs
                        ).refresh(timeout_s=1.5)
                        or {}
                    )
                except Exception:
                    ledger_doc = {}
            return {"status": status_doc, "ledger": ledger_doc}

        def _write_ledger(doc: Dict[str, Any]) -> None:
            try:
                out_path = Path(root_dir) / "artifacts" / "provider_ledger" / "latest.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            except Exception:
                pass

        def _provider_available(
            entry: Dict[str, Any], ledger_row: Optional[Dict[str, Any]]
        ) -> bool:
            auth_state = str(entry.get("auth_state") or "").upper()
            if auth_state in {"MISSING", "EXPIRED"}:
                return False
            policy_text = str((entry.get("policy_state") or {}).get("text") or "").strip().lower()
            if policy_text in {"disallowed", "blocked", "deny"}:
                return False
            if ledger_row and str(ledger_row.get("status") or "") != "ok":
                return False
            breathing = entry.get("breathing") if isinstance(entry.get("breathing"), dict) else {}
            if str(breathing.get("status") or "").upper() != "UP":
                return False
            if str(breathing.get("contract_status") or "").upper() == "DOWN":
                return False
            return True

        def _rebuild_ledger_from_status(status_doc: Dict[str, Any]) -> Dict[str, Any]:
            providers_status = status_doc.get("providers") if isinstance(status_doc, dict) else {}
            if not isinstance(providers_status, dict):
                providers_status = {}
            rows: List[Dict[str, Any]] = []
            providers_cfg = (self.provider_configs or {}).get("providers", {})
            if isinstance(providers_cfg, dict) and providers_cfg:
                for name, cfg in providers_cfg.items():
                    if not isinstance(cfg, dict) or cfg.get("disabled"):
                        continue
                    roles = cfg.get("roles") or []
                    if isinstance(roles, str):
                        roles = [roles]
                    roles_l = [str(r).strip().lower() for r in roles if str(r).strip()]
                    if not roles_l:
                        continue
                    entry = (
                        providers_status.get(name) if isinstance(providers_status, dict) else None
                    )
                    if not isinstance(entry, dict):
                        continue
                    unavailable = entry.get("unavailable_reason")
                    available_recent = bool(entry.get("available_recent"))
                    status = (
                        "ok"
                        if available_recent and not unavailable
                        else str(unavailable or "unavailable")
                    )
                    reason = None if status == "ok" else str(unavailable or "unavailable")
                    latency = entry.get("latency_p95_ms")
                    timeout_rate = entry.get("timeout_rate_recent")
                    failure_rate = entry.get("failure_rate_recent")
                    model_id = cfg.get("default_model") or cfg.get("model")
                    for role in roles_l:
                        rows.append(
                            {
                                "provider": str(name),
                                "model": model_id,
                                "role": role,
                                "status": status,
                                "reason": reason,
                                "available_recent": available_recent,
                                "latency_p95_ms": latency,
                                "timeout_rate_recent": timeout_rate,
                                "failure_rate_recent": failure_rate,
                                "details": {"source": "providers_status"},
                            }
                        )
            else:
                for name, entry in providers_status.items():
                    if not isinstance(entry, dict):
                        continue
                    breathing = (
                        entry.get("breathing") if isinstance(entry.get("breathing"), dict) else {}
                    )
                    roles = (
                        breathing.get("roles") if isinstance(breathing.get("roles"), dict) else {}
                    )
                    roles_l = [str(r).strip().lower() for r in roles.keys() if str(r).strip()] or [
                        "brain"
                    ]
                    unavailable = entry.get("unavailable_reason")
                    available_recent = bool(entry.get("available_recent"))
                    status = (
                        "ok"
                        if available_recent and not unavailable
                        else str(unavailable or "unavailable")
                    )
                    reason = None if status == "ok" else str(unavailable or "unavailable")
                    latency = entry.get("latency_p95_ms")
                    timeout_rate = entry.get("timeout_rate_recent")
                    failure_rate = entry.get("failure_rate_recent")
                    for role in roles_l:
                        rows.append(
                            {
                                "provider": str(name),
                                "model": None,
                                "role": role,
                                "status": status,
                                "reason": reason,
                                "available_recent": available_recent,
                                "latency_p95_ms": latency,
                                "timeout_rate_recent": timeout_rate,
                                "failure_rate_recent": failure_rate,
                                "details": {"source": "providers_status"},
                            }
                        )
            return {
                "schema": "ajax.provider_ledger.v1",
                "updated_ts": time.time(),
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "path": str(Path(root_dir) / "artifacts" / "provider_ledger" / "latest.json"),
                "rows": rows,
            }

        def _update_unavailable_reasons(
            status_doc: Dict[str, Any], ledger_doc: Dict[str, Any]
        ) -> None:
            providers_status = status_doc.get("providers") if isinstance(status_doc, dict) else {}
            if not isinstance(providers_status, dict):
                return
            ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else []
            ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
            ledger_by_provider: Dict[str, Dict[str, Any]] = {}
            for row in ledger_rows:
                if not isinstance(row, dict):
                    continue
                prov = str(row.get("provider") or "").strip()
                if prov and prov not in ledger_by_provider:
                    ledger_by_provider[prov] = row
            for name, entry in providers_status.items():
                if not isinstance(entry, dict):
                    continue
                reason = ""
                policy_state = (
                    entry.get("policy_state") if isinstance(entry.get("policy_state"), dict) else {}
                )
                if str(policy_state.get("text") or "").strip().lower() in {
                    "disallowed",
                    "blocked",
                    "deny",
                }:
                    reason = "policy_disallowed"
                auth_state = str(entry.get("auth_state") or "").upper()
                if not reason and auth_state == "MISSING":
                    reason = "auth_missing"
                if not reason and auth_state == "EXPIRED":
                    reason = "auth_expired"
                breathing = (
                    entry.get("breathing") if isinstance(entry.get("breathing"), dict) else {}
                )
                if not reason and str(breathing.get("status") or "").upper() == "DOWN":
                    reason = "transport_down"
                if not reason and str(breathing.get("contract_status") or "").upper() == "DOWN":
                    reason = "contract_down"
                ledger_row = ledger_by_provider.get(name)
                if not reason and ledger_row:
                    status = str(ledger_row.get("status") or "").lower()
                    lreason = str(ledger_row.get("reason") or "").lower()
                    if status == "quota_exhausted" or "quota" in lreason or "429" in lreason:
                        reason = "quota_exhausted"
                    elif status == "auth_fail" or "auth" in lreason:
                        reason = "auth_fail"
                    elif status and status != "ok":
                        reason = status
                entry["unavailable_reason"] = reason or None
                if entry.get("latency_p95_ms") is None:
                    role_latency = None
                    roles = (
                        breathing.get("roles") if isinstance(breathing.get("roles"), dict) else {}
                    )
                    for probe in roles.values():
                        if isinstance(probe, dict) and isinstance(
                            probe.get("latency_ms"), (int, float)
                        ):
                            role_latency = probe.get("latency_ms")
                            break
                    if role_latency is not None:
                        entry["latency_p95_ms"] = int(role_latency)

        scoreboard_doc: Dict[str, Any] = {}
        try:
            from agency import provider_scoreboard  # type: ignore

            scoreboard_doc = provider_scoreboard.load_scoreboard(
                Path(root_dir) / "artifacts" / "state" / "provider_scoreboard.json"
            )
        except Exception:
            scoreboard_doc = {}

        def _select_brain_fallback_chain(status_doc: Dict[str, Any], prefer_tier: str) -> List[str]:
            if provider_ranker is None:
                return []
            pool: List[str] = []
            for name, cfg in (self.provider_configs or {}).get("providers", {}).items():
                if not isinstance(cfg, dict) or cfg.get("disabled"):
                    continue
                roles = cfg.get("roles") or []
                if isinstance(roles, str):
                    roles = [roles]
                if "brain" in {str(r).lower() for r in roles if r}:
                    pool.append(str(name))
            ranked = provider_ranker.rank_providers(
                pool,
                providers_cfg=providers_cfg,
                status=status_doc,
                scoreboard=scoreboard_doc,
                prefer_tier=prefer_tier,
                role="brain",
                rail=os.getenv("AJAX_RAIL")
                or os.getenv("AJAX_ENV")
                or os.getenv("AJAX_MODE")
                or "lab",
                risk_level="medium",
            )
            return ranked

        def _write_status(doc: Dict[str, Any]) -> None:
            try:
                out_path = Path(root_dir) / "artifacts" / "health" / "providers_status.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            except Exception:
                pass

        def _parse_port_from_cfg(cfg: Dict[str, Any], default_port: int) -> int:
            base_url = str(cfg.get("base_url") or "").strip()
            if base_url:
                try:
                    parsed = urllib.parse.urlparse(base_url)
                    if parsed.port:
                        return int(parsed.port)
                except Exception:
                    return default_port
            return default_port

        def _lmstudio_soft_fail_timeout(ledger_doc: Dict[str, Any]) -> Optional[str]:
            rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else []
            rows = rows if isinstance(rows, list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                provider = str(row.get("provider") or "").strip()
                status = str(row.get("status") or "").strip().lower()
                reason = str(row.get("reason") or "").strip().lower()
                if provider == "lmstudio_vision" and status == "soft_fail" and reason == "timeout":
                    role = str(row.get("role") or "vision").strip().lower() or "vision"
                    return role
            return None

        def _clear_ledger_cooldown(ledger_doc: Dict[str, Any], provider: str, role: str) -> bool:
            rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else []
            rows = rows if isinstance(rows, list) else []
            updated = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("provider") or "").strip() != provider:
                    continue
                if str(row.get("role") or "").strip().lower() != role:
                    continue
                if row.get("cooldown_until_ts") or row.get("cooldown_until"):
                    row["cooldown_until_ts"] = None
                    row["cooldown_until"] = None
                    updated = True
            return updated

        def _ledger_status(ledger_doc: Dict[str, Any], provider: str, role: str) -> str:
            rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else []
            rows = rows if isinstance(rows, list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("provider") or "").strip() != provider:
                    continue
                if str(row.get("role") or "").strip().lower() != role:
                    continue
                return str(row.get("status") or "").strip().lower()
            return ""

        def _run_lmstudio_autostart(port: int) -> Dict[str, Any]:
            script_path = Path(root_dir) / "scripts" / "ops" / "ensure_lmstudio_server.ps1"
            if not script_path.exists():
                return {"ok": False, "error": "script_missing"}
            # In WSL we generally need Windows PowerShell to access Task Scheduler and Windows-local CLIs.
            # Prefer `powershell.exe` when available, otherwise fall back to pwsh/powershell.
            ps = (
                shutil.which("powershell.exe") or shutil.which("pwsh") or shutil.which("powershell")
            )
            if not ps:
                return {"ok": False, "error": "powershell_missing"}
            script_arg = str(script_path)
            # If invoking a Windows binary from WSL, convert /mnt/c/... to C:\... so PowerShell can find the file.
            try:
                if ps.lower().endswith("powershell.exe") and script_arg.startswith("/mnt/"):
                    parts = script_arg.split("/")
                    if len(parts) >= 4:
                        drive = parts[2].upper()
                        rest = "\\".join(parts[3:])
                        script_arg = f"{drive}:\\{rest}"
            except Exception:
                script_arg = str(script_path)
            cmd = [
                ps,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script_arg,
                "-Port",
                str(port),
            ]
            started = time.monotonic()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
            except Exception as exc:
                return {"ok": False, "error": str(exc)[:200]}
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return {
                "ok": proc.returncode == 0,
                "elapsed_ms": elapsed_ms,
                "stdout": (proc.stdout or "")[-400:],
                "stderr": (proc.stderr or "")[-400:],
            }

        refresh = _refresh_state()
        status_doc = refresh.get("status") or {}
        ledger_doc = refresh.get("ledger") or {}
        _update_unavailable_reasons(status_doc, ledger_doc)
        _write_status(status_doc)
        ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else None
        ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
        if not ledger_rows:
            ledger_doc = _rebuild_ledger_from_status(status_doc)
            _write_ledger(ledger_doc)

        lmstudio_role = _lmstudio_soft_fail_timeout(ledger_doc)
        if lmstudio_role:
            lm_cfg = (
                providers_cfg.get("lmstudio_vision", {}) if isinstance(providers_cfg, dict) else {}
            )
            port = _parse_port_from_cfg(lm_cfg if isinstance(lm_cfg, dict) else {}, 1235)
            receipt["lmstudio_autostart"] = {
                "attempted": True,
                "provider": "lmstudio_vision",
                "role": lmstudio_role,
                "port": port,
            }
            autostart_result = _run_lmstudio_autostart(port)
            receipt["lmstudio_autostart"].update(autostart_result)

            # Verificar estado detallado del servidor LM Studio
            # Importar desde broker para evitar duplicación de código
            try:
                from agency.broker import _lmstudio_health_detailed

                health = _lmstudio_health_detailed(f"http://127.0.0.1:{port}/models", timeout=2.0)
                receipt["lmstudio_health"] = {
                    "server_up": health.server_up,
                    "model_ready": health.model_ready,
                    "status_code": health.status_code,
                    "error": health.error,
                }

                # Si el servidor está up pero el modelo no está listo (404), marcar en el ledger
                if health.server_up and not health.model_ready and health.status_code == 404:
                    receipt["lmstudio_autostart"]["warning"] = "Server UP but model not ready (404)"
                    # Actualizar ledger para reflejar que el modelo no está listo
                    for row in ledger_rows:
                        if isinstance(row, dict) and row.get("provider") == "lmstudio_vision":
                            row["model_ready"] = False
                            row["status"] = "model_not_ready"
                            row["last_error"] = "HTTP 404: Model not loaded"
                    _write_ledger(ledger_doc)
            except Exception as health_exc:
                receipt["lmstudio_health"] = {"error": str(health_exc)[:200]}

            if _clear_ledger_cooldown(ledger_doc, "lmstudio_vision", lmstudio_role):
                _write_ledger(ledger_doc)
            refresh = _refresh_state()
            status_doc = refresh.get("status") or {}
            ledger_doc = refresh.get("ledger") or {}
            _update_unavailable_reasons(status_doc, ledger_doc)
            _write_status(status_doc)
            ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else None
            ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
            if not ledger_rows:
                ledger_doc = _rebuild_ledger_from_status(status_doc)
                _write_ledger(ledger_doc)
            if _ledger_status(ledger_doc, "lmstudio_vision", lmstudio_role) == "ok":
                if _clear_ledger_cooldown(ledger_doc, "lmstudio_vision", lmstudio_role):
                    _write_ledger(ledger_doc)

        has_primary = False
        providers_status = status_doc.get("providers") if isinstance(status_doc, dict) else {}
        if not isinstance(providers_status, dict):
            providers_status = {}
        ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else []
        ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
        ledger_by_provider = {
            str(row.get("provider")): row for row in ledger_rows if isinstance(row, dict)
        }
        for name, cfg in providers_cfg.items():
            if not isinstance(cfg, dict) or cfg.get("disabled"):
                continue
            tier = str(cfg.get("tier") or "balanced").lower()
            if tier != "premium":
                continue
            entry = providers_status.get(name) if isinstance(providers_status, dict) else None
            if not isinstance(entry, dict):
                continue
            if _provider_available(entry, ledger_by_provider.get(name)):
                has_primary = True
                break

        repair_attempted = False
        if not has_primary:
            repair_attempted = True
            try:
                self.provider_configs = self._load_provider_configs()
            except Exception:
                pass
            refresh = _refresh_state()
            status_doc = refresh.get("status") or {}
            ledger_doc = refresh.get("ledger") or {}
            _update_unavailable_reasons(status_doc, ledger_doc)
            _write_status(status_doc)
            ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else None
            ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
            if not ledger_rows:
                ledger_doc = _rebuild_ledger_from_status(status_doc)
                _write_ledger(ledger_doc)
            providers_status = status_doc.get("providers") if isinstance(status_doc, dict) else {}
            if not isinstance(providers_status, dict):
                providers_status = {}
            ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else []
            ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
            ledger_by_provider = {
                str(row.get("provider")): row for row in ledger_rows if isinstance(row, dict)
            }
            for name, cfg in (self.provider_configs or {}).get("providers", {}).items():
                if not isinstance(cfg, dict) or cfg.get("disabled"):
                    continue
                tier = str(cfg.get("tier") or "balanced").lower()
                if tier != "premium":
                    continue
                entry = providers_status.get(name) if isinstance(providers_status, dict) else None
                if not isinstance(entry, dict):
                    continue
                if _provider_available(entry, ledger_by_provider.get(name)):
                    has_primary = True
                    break

        chosen_tier = req_tier
        if req_tier == "premium" and not has_primary:
            chosen_tier = "balanced"
            try:
                if isinstance(self.state, AjaxStateSnapshot):
                    self.state.notes["system_state"] = "DEGRADED"
            except Exception:
                pass
        fallback_chain = _select_brain_fallback_chain(status_doc, chosen_tier)
        if not fallback_chain:
            providers_status = status_doc.get("providers") if isinstance(status_doc, dict) else {}
            if isinstance(providers_status, dict):
                fallback_chain = [
                    name
                    for name, entry in providers_status.items()
                    if isinstance(entry, dict)
                    and entry.get("available_recent") is True
                    and not entry.get("unavailable_reason")
                ]
        chosen_provider = fallback_chain[0] if fallback_chain else None

        receipt.update(
            {
                "has_primary": bool(has_primary),
                "repair_attempted": bool(repair_attempted),
                "chosen_tier": chosen_tier,
                "chosen_provider": chosen_provider,
                "fallback_chain": fallback_chain,
                "providers_status_path": str(
                    Path(root_dir) / "artifacts" / "health" / "providers_status.json"
                ),
                "provider_ledger_path": ledger_doc.get("path")
                if isinstance(ledger_doc, dict)
                else None,
            }
        )
        try:
            doc_dir = (
                Path(root_dir)
                / "artifacts"
                / "doctor"
                / f"providers_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            )
            doc_dir.mkdir(parents=True, exist_ok=True)
            (doc_dir / "receipt.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass
        return receipt

    def _load_tool_inventory(self) -> List[Any]:
        if getattr(self, "_tool_inventory", None) is None:
            try:
                self._tool_inventory = load_inventory() if load_inventory else []
            except Exception:
                self._tool_inventory = []
        return self._tool_inventory or []

    def _load_tool_use_notes(self) -> Dict[str, Any]:
        if not load_tool_use_notes:
            return {}
        try:
            return load_tool_use_notes()
        except Exception:
            return {}

    def _auto_crystallize_flag_path(self) -> Path:
        return self.state_dir / "auto_crystallize.flag"

    def _load_auto_crystallize_flag(self) -> bool:
        env = os.getenv("AJAX_AUTO_CRYSTALLIZE")
        default_env = False
        if env is not None:
            default_env = env.strip().lower() not in {"0", "false", "off", ""}
        flag_path = self._auto_crystallize_flag_path()
        if flag_path.exists():
            try:
                raw = flag_path.read_text(encoding="utf-8").strip().lower()
                return raw in {"1", "true", "on", "yes"}
            except Exception:
                return default_env
        return default_env

    def _emit_episode_and_receipt(
        self,
        mission: MissionState,
        *,
        waiting_for_user: bool,
    ) -> Optional[str]:
        ts = time.time()
        ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
        root_dir = self.config.root_dir
        receipt_dir = Path(root_dir) / "artifacts" / "receipts"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"episode_{ts_label}_{mission.mission_id}.json"
        payload: Dict[str, Any] = {
            "schema": "ajax.episode_receipt.v1",
            "ts": ts,
            "ts_utc": self._iso_utc(ts),
            "mission_id": mission.mission_id,
            "waiting_for_user": bool(waiting_for_user),
            "skill_id": (
                mission.notes.get("library_skill_id") if isinstance(mission.notes, dict) else None
            ),
            "plan_id": mission.last_plan.plan_id if mission.last_plan else None,
            "plan_source": (
                mission.last_plan.metadata.get("plan_source")
                if mission.last_plan and mission.last_plan.metadata
                else None
            ),
        }
        if CrystallizationEngine is None:
            payload["ok"] = False
            payload["error"] = "crystallization_unavailable"
            receipt_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(receipt_path)
        try:
            engine = CrystallizationEngine(root_dir)
            res = engine.crystallize_mission(mission.mission_id)
            payload.update(
                {
                    "ok": True,
                    "episode_path": res.get("episode_path") if isinstance(res, dict) else None,
                    "recipe_path": res.get("recipe_path") if isinstance(res, dict) else None,
                    "episode_id": res.get("episode_id") if isinstance(res, dict) else None,
                }
            )
        except Exception as exc:
            payload["ok"] = False
            payload["error"] = str(exc)[:200]
            try:
                gap_path = None
                self._emit_crystallize_failed_gap(mission.mission_id, str(exc))
                payload["gap_path"] = gap_path
            except Exception:
                pass
        receipt_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        try:
            if isinstance(mission.notes, dict):
                mission.notes["last_episode_receipt"] = str(receipt_path)
        except Exception:
            pass
        return str(receipt_path)

    def _maybe_auto_crystallize(self, mission: MissionState, *, waiting_for_user: bool) -> None:
        if waiting_for_user or not self.auto_crystallize_enabled or CrystallizationEngine is None:
            return
        try:
            engine = CrystallizationEngine(self.config.root_dir)
            engine.crystallize_mission(mission.mission_id)
        except Exception as exc:
            self._emit_crystallize_failed_gap(mission.mission_id, str(exc))

    def _emit_crystallize_failed_gap(self, mission_id: str, error: str) -> None:
        try:
            gaps_dir = self.config.root_dir / "artifacts" / "capability_gaps"
            gaps_dir.mkdir(parents=True, exist_ok=True)
            path = gaps_dir / f"{int(time.time())}_crystallize_failed_{mission_id}.json"
            payload = {
                "schema": "ajax.capability_gap.crystallize_failed.v1",
                "mission_id": mission_id,
                "error": error,
                "ts": time.time(),
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            try:
                self.log.warning("No se pudo escribir gap crystallize_failed", exc_info=True)
            except Exception:
                pass

    def _apply_tool_plan_requirements(
        self,
        tool_plan: Optional["PolicyToolPlan"],
        intention: str,
        knowledge_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        ajax_snippets: List[Dict[str, Any]] = []
        if not tool_plan:
            return ajax_snippets
        required = set(tool_plan.required or [])
        if "memory.leann_history" in required:
            top_k = 3
            try:
                top_k = int(tool_plan.budget.get("memory.leann_history", {}).get("top_k", 3))
            except Exception:
                top_k = 3
            ajax_snippets = self._fetch_ajax_history_snippets(intention, top_k=top_k)
            base_satisfied = tool_plan.satisfied.get("memory.leann_history")
            if base_satisfied is False:
                tool_plan.satisfied["memory.leann_history"] = False
            else:
                tool_plan.satisfied["memory.leann_history"] = True
            if ajax_snippets:
                knowledge_context.setdefault("ajax_history_snippets", ajax_snippets)
                notes = knowledge_context.get("notes") or []
                notes.append(f"ajax_history_v1_snippets={len(ajax_snippets)}")
                knowledge_context["notes"] = notes
        return ajax_snippets

    def _log_tool_plan(self, tool_plan: Optional["PolicyToolPlan"], mode: str = "mission") -> None:
        if not tool_plan:
            return
        try:
            print(f"🧰 TOOLS[{mode}]: selected={tool_plan.selected} required={tool_plan.required}")
        except Exception:
            pass

    def _write_tool_plan_artifact(
        self,
        tool_plan: Optional["PolicyToolPlan"],
        heartbeat_snapshot: Optional[Dict[str, Any]],
        *,
        mission_id: Optional[str],
        tool_use_notes: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not tool_plan:
            return
        try:
            run_id = mission_id or f"plan-{int(time.time())}"
            run_dir = self.config.root_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = tool_plan.to_dict() if hasattr(tool_plan, "to_dict") else {}
            payload["heartbeat_snapshot"] = heartbeat_snapshot or {}
            try:
                payload["inventory"] = [spec.to_dict() for spec in self._load_tool_inventory()]
            except Exception:
                payload["inventory"] = []
            if tool_use_notes:
                payload["tool_use_notes_raw"] = tool_use_notes
            payload["timestamp"] = time.time()
            target = run_dir / "tool_selection.json"
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _provider_cooldown_seconds(self) -> int:
        try:
            ttl = int(os.getenv("AJAX_PROVIDER_COOLDOWN_SEC", "").strip() or 0)
        except Exception:
            ttl = 0
        if ttl <= 0:
            policy = {}
            try:
                policy = self._provider_failure_policy()
            except Exception:
                policy = {}
            if failure_cooldown_seconds_default is not None:
                try:
                    ttl = int(failure_cooldown_seconds_default(policy, default=90))
                except Exception:
                    ttl = 90
            else:
                ttl = 90
        return max(30, min(int(ttl), 600))

    def _provider_failure_policy(self) -> Dict[str, Any]:
        cached = getattr(self, "_provider_failure_policy_cache", None)
        if isinstance(cached, dict):
            return cached
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        doc: Dict[str, Any] = {}
        if load_provider_failure_policy is not None:
            try:
                doc = load_provider_failure_policy(Path(root_dir))
            except Exception:
                doc = {}
        self._provider_failure_policy_cache = doc
        return doc

    def _provider_timeouts_policy(self) -> tuple[Dict[str, Any], Optional[str]]:
        cached = getattr(self, "_provider_timeouts_policy_cache", None)
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached  # type: ignore[return-value]
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        doc: Dict[str, Any] = {}
        digest = None
        if load_provider_timeouts_policy is not None:
            try:
                doc, digest = load_provider_timeouts_policy(Path(root_dir))
            except Exception:
                doc, digest = {}, None
        self._provider_timeouts_policy_cache = (doc, digest)
        return doc, digest

    def _provider_timeout_bench_p95(self, provider_name: str) -> Dict[str, Optional[int]]:
        """
        Bench p95 derivado de providers_status (total + ttft).
        """
        out: Dict[str, Optional[int]] = {"total_p95_ms": None, "ttft_p95_ms": None}
        try:
            if provider_ranker is None:
                return out
            status = provider_ranker.load_status(
                self.config.root_dir / "artifacts" / "health" / "providers_status.json"
            )
            entry = ((status or {}).get("providers") or {}).get(provider_name)
            if not isinstance(entry, dict):
                return out
            total = entry.get("total_p95_ms")
            if not isinstance(total, (int, float)) or total <= 0:
                total = entry.get("latency_p95_ms")
            ttft = entry.get("ttft_p95_ms")
            if isinstance(total, (int, float)) and total > 0:
                out["total_p95_ms"] = int(total)
            if isinstance(ttft, (int, float)) and ttft > 0:
                out["ttft_p95_ms"] = int(ttft)
        except Exception:
            return out
        return out

    def _recent_timeout_kind_hint(self, provider_name: str) -> Optional[str]:
        """
        Devuelve el último timeout_kind observado para el provider si es reciente.
        """
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        base = Path(root_dir) / "artifacts" / "provider_timeouts"
        if not base.exists():
            return None
        try:
            dirs = sorted(
                [p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True
            )
        except Exception:
            return None
        now = time.time()
        window_s = int(os.getenv("AJAX_TIMEOUT_HINT_WINDOW_SEC", "3600") or 3600)
        for p in dirs[:30]:
            receipt = p / "receipt.json"
            if not receipt.exists():
                continue
            try:
                doc = json.loads(receipt.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            provider = str(doc.get("provider") or "").strip()
            if provider != provider_name:
                continue
            ts = doc.get("ts")
            if not isinstance(ts, (int, float)) or ts <= 0:
                continue
            if (now - float(ts)) > window_s:
                continue
            return str(doc.get("timeout_kind") or "").strip().upper() or None
        return None

    def _resolve_provider_timeouts(
        self,
        provider_name: str,
        provider_cfg: Dict[str, Any],
        *,
        rail: str,
        intent_class: Optional[str],
        planning: bool,
    ) -> Dict[str, Any]:
        policy, digest = self._provider_timeouts_policy()
        defaults = (policy.get("defaults") if isinstance(policy, dict) else {}) or {}
        rail_cfg = {}
        if isinstance(policy, dict):
            rails = policy.get("rails")
            if isinstance(rails, dict):
                rail_cfg = rails.get(rail, {}) if isinstance(rail, str) else {}
        rail_defaults = rail_cfg.get("defaults") if isinstance(rail_cfg, dict) else {}
        provider_overrides = {}
        if isinstance(policy, dict):
            providers = policy.get("providers")
            if isinstance(providers, dict):
                provider_overrides = providers.get(provider_name, {}) or {}
        stage_key = "planning" if planning else "default"
        stage_overrides = (
            provider_overrides.get(stage_key) if isinstance(provider_overrides, dict) else None
        )
        intent_overrides = None
        if isinstance(provider_overrides, dict):
            intent_map = provider_overrides.get("intent_classes")
            if isinstance(intent_map, dict) and isinstance(intent_class, str) and intent_class:
                intent_entry = intent_map.get(intent_class)
                if isinstance(intent_entry, dict):
                    intent_overrides = intent_entry.get(stage_key) or intent_entry

        def _pick(key: str) -> Optional[int]:
            for src in (intent_overrides, stage_overrides, rail_defaults, defaults):
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

        total_ms = _pick("total_timeout_ms")
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
        connect_ms = _pick("connect_timeout_ms")
        first_output_ms = _pick("first_output_timeout_ms")
        stall_ms = _pick("stall_timeout_ms")
        policy_ver = None
        if timeouts_policy_version is not None:
            try:
                policy_ver = timeouts_policy_version(policy)
            except Exception:
                policy_ver = None
        timeout_cfg = {
            "connect_timeout_ms": connect_ms,
            "first_output_timeout_ms": first_output_ms,
            "stall_timeout_ms": stall_ms,
            "total_timeout_ms": total_ms,
            "policy_version": policy_ver,
            "policy_hash": digest,
        }
        # Dynamic overrides from bench p95 (total/ttft) + recent TTFT hint.
        try:
            bench = self._provider_timeout_bench_p95(provider_name)
            mult = float(os.getenv("AJAX_TIMEOUT_P95_MULT", "1.35") or 1.35)
            if mult < 1.0:
                mult = 1.0
            if isinstance(bench.get("total_p95_ms"), int):
                bench_total = int(bench["total_p95_ms"] * mult) + 1000
                if not isinstance(
                    timeout_cfg.get("total_timeout_ms"), (int, float)
                ) or bench_total > int(timeout_cfg.get("total_timeout_ms") or 0):
                    timeout_cfg["total_timeout_ms"] = bench_total
            if isinstance(bench.get("ttft_p95_ms"), int):
                bench_ttft = int(bench["ttft_p95_ms"] * mult) + 500
                if not isinstance(
                    timeout_cfg.get("first_output_timeout_ms"), (int, float)
                ) or bench_ttft > int(timeout_cfg.get("first_output_timeout_ms") or 0):
                    timeout_cfg["first_output_timeout_ms"] = bench_ttft
            hint = self._recent_timeout_kind_hint(provider_name)
            if hint == "TTFT" and isinstance(
                timeout_cfg.get("first_output_timeout_ms"), (int, float)
            ):
                boost = float(os.getenv("AJAX_TIMEOUT_TTFT_BOOST_MULT", "1.5") or 1.5)
                timeout_cfg["first_output_timeout_ms"] = int(
                    float(timeout_cfg["first_output_timeout_ms"]) * boost
                )
        except Exception:
            pass
        return timeout_cfg

    def _next_cost_mode(self, current: Optional[str]) -> str:
        order = ["save_codex", "balanced", "premium", "emergency"]
        base = (current or os.getenv("AJAX_COST_MODE", "premium") or "premium").strip().lower()
        if base in {"cheap", "economy"}:
            return "balanced"
        if base in {"premium_fast", "premium-first", "premium_first"}:
            base = "premium"
        idx = order.index(base) if base in order else len(order) - 1
        if idx < len(order) - 1:
            idx += 1
        return order[idx]

    def _next_brain_cost_mode(self, current: Optional[str]) -> str:
        return self._next_cost_mode(current)

    @contextlib.contextmanager
    def _override_cost_mode(self, override: Optional[str]):
        if not override:
            yield
            return
        backup = os.getenv("AJAX_COST_MODE")
        os.environ["AJAX_COST_MODE"] = override
        try:
            yield
        finally:
            if backup is None:
                try:
                    os.environ.pop("AJAX_COST_MODE", None)
                except Exception:
                    pass
            else:
                os.environ["AJAX_COST_MODE"] = backup

    def _effective_cost_mode(self, mission: Optional[MissionState]) -> Optional[str]:
        if mission is None:
            return None
        if mission.cost_mode:
            return mission.cost_mode
        return mission.brain_cost_mode_override

    def _max_codex_calls_per_mission(self) -> int:
        raw = (os.getenv("MAX_CODEX_CALLS_PER_MISSION") or "1").strip()
        try:
            return max(0, int(raw))
        except Exception:
            return 1

    @staticmethod
    def _codex_calls_used(mission: MissionState) -> int:
        attempts = getattr(mission, "brain_attempts", None) or []
        count = 0
        if isinstance(attempts, list):
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                provider = str(attempt.get("provider") or attempt.get("id") or "").strip()
                if provider.startswith("codex_"):
                    count += 1
        return count

    def _codex_budget_exceeded(self, mission: MissionState) -> bool:
        limit = self._max_codex_calls_per_mission()
        if limit <= 0:
            return True
        return self._codex_calls_used(mission) >= limit

    def _rebuild_starting_xi_for_mission(
        self,
        mission: MissionState,
        *,
        cost_mode: str,
        vision_required: bool = False,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if build_starting_xi is None:
            return None
        rail = os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"
        risk_level = "medium"
        try:
            gov = mission.envelope.governance if mission.envelope else None  # type: ignore[union-attr]
            risk_level = str(getattr(gov, "risk_level", "medium") or "medium")
        except Exception:
            risk_level = "medium"
        prev_force_inv = os.environ.get("AJAX_STARTING_XI_FORCE_INVENTORY_REFRESH")
        prev_force_breath = os.environ.get("AJAX_STARTING_XI_FORCE_BREATHING_REFRESH")
        if force_refresh:
            os.environ["AJAX_STARTING_XI_FORCE_INVENTORY_REFRESH"] = "1"
            os.environ["AJAX_STARTING_XI_FORCE_BREATHING_REFRESH"] = "1"
        try:
            sx = build_starting_xi(
                root_dir=self.config.root_dir,
                provider_configs=self.provider_configs,
                rail=rail,
                risk_level=risk_level,
                cost_mode=cost_mode,
                vision_required=bool(vision_required),
            )
        except Exception:
            sx = None
        finally:
            if prev_force_inv is None:
                os.environ.pop("AJAX_STARTING_XI_FORCE_INVENTORY_REFRESH", None)
            else:
                os.environ["AJAX_STARTING_XI_FORCE_INVENTORY_REFRESH"] = prev_force_inv
            if prev_force_breath is None:
                os.environ.pop("AJAX_STARTING_XI_FORCE_BREATHING_REFRESH", None)
            else:
                os.environ["AJAX_STARTING_XI_FORCE_BREATHING_REFRESH"] = prev_force_breath
        if sx is None:
            return None
        try:
            mission.notes["starting_xi"] = sx
            mission.notes["cost_mode"] = cost_mode
        except Exception:
            pass
        if mission.envelope:
            try:
                mission.envelope.metadata["starting_xi_path"] = sx.get("path")
            except Exception:
                pass
            try:
                mission.envelope.metadata["starting_xi"] = sx
            except Exception:
                pass
        return sx

    @staticmethod
    def _starting_xi_role_primary(
        sx: Optional[Dict[str, Any]], role: str
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(sx, dict):
            return None
        key = (role or "").strip().lower()
        if key == "vision":
            ent = sx.get("vision")
            ent = ent if isinstance(ent, dict) else {}
            prim = ent.get("primary")
            return prim if isinstance(prim, dict) else None
        if key == "brain":
            ent = sx.get("brain")
            ent = ent if isinstance(ent, dict) else {}
            prim = ent.get("primary")
            return prim if isinstance(prim, dict) else None
        if key.startswith("council"):
            council = sx.get("council") if isinstance(sx.get("council"), dict) else {}
            role_key = "role1"
            if key in {"council.role2", "role2"}:
                role_key = "role2"
            role_ent = (council or {}).get(role_key) if isinstance(council, dict) else {}
            role_ent = role_ent if isinstance(role_ent, dict) else {}
            prim = role_ent.get("primary")
            return prim if isinstance(prim, dict) else None
        return None

    def _plan_requires_vision_player(self, plan: AjaxPlan) -> bool:
        for st in plan.steps or []:
            if not isinstance(st, dict):
                continue
            action = str(st.get("action") or "").strip()
            if not action:
                continue
            catalog = getattr(self, "actions_catalog", None)
            if catalog is not None and hasattr(catalog, "requires_vision"):
                try:
                    if bool(catalog.requires_vision(action)):
                        return True
                except Exception:
                    if action.startswith("vision."):
                        return True
            elif action.startswith("vision."):
                return True
        return False

    @staticmethod
    def _plan_has_ui_actions(plan: AjaxPlan) -> bool:
        ui_prefixes = ("app.", "window.", "keyboard.", "mouse.", "desktop.")
        for st in plan.steps or []:
            if not isinstance(st, dict):
                continue
            action = str(st.get("action") or "").strip().lower()
            if not action:
                continue
            if action.startswith(ui_prefixes):
                return True
        return False

    def _missing_vision_fix_steps(self, sx: Optional[Dict[str, Any]]) -> List[str]:
        slots_missing = []
        try:
            slots_missing = self._slots_missing_entries()
        except Exception:
            slots_missing = []
        fix = [
            "Reintenta con refresh: AJAX_STARTING_XI_FORCE_BREATHING_REFRESH=1 y AJAX_STARTING_XI_FORCE_INVENTORY_REFRESH=1.",
            "Diagnóstico: `ajaxctl doctor providers --roles vision --explain`.",
            "Revisa `artifacts/health/providers_status.json` (auth_state/quota_state) y `artifacts/provider_ledger/latest.json` (cooldown).",
        ]
        if slots_missing:
            fix.append(
                "Corrige slots/inventario: revisa `config/model_slots.json` vs `config/model_inventory_cloud.json`."
            )
        # Hint desde starting_xi (auth/quota/timeout) si existe.
        try:
            hints = str((sx or {}).get("fix_hints") or "").strip()
            if hints:
                fix.append(f"Fix hints: {hints} (auth/quota/timeout).")
        except Exception:
            pass
        return fix

    def _keyboard_only_media_playback_plan(
        self, intention: str, *, degraded_reason: Optional[str] = None
    ) -> AjaxPlan:
        try:
            from agency.skills.os_inventory import get_installed_apps  # type: ignore

            installed_apps = get_installed_apps()
        except Exception:
            installed_apps = []
        browser = self._choose_browser(installed_apps)
        browser_exe = str(browser or "").strip()
        if (
            browser_exe
            and "\\" not in browser_exe
            and "/" not in browser_exe
            and not browser_exe.lower().endswith(".exe")
        ):
            browser_exe = f"{browser_exe}.exe"
        if not browser_exe:
            browser_exe = "brave.exe"
        expected_browser_active = {
            "windows": [{"process_equals": browser_exe, "must_exist": True}],
            "meta": {"must_be_active": True},
        }
        steps = [
            {
                "id": "task-1",
                "intent": "Abrir un navegador (sin asumir proveedor) para poder reproducir media.",
                "preconditions": {"expected_state": {}},
                "action": "app.launch",
                "args": {"process": browser_exe},
                "evidence_required": ["driver.screenshot"],
                "success_spec": {"expected_state": expected_browser_active},
                "on_fail": "abort",
            },
            {
                "id": "task-2",
                "intent": "Enfocar el navegador para interactuar con la barra de direcciones.",
                "preconditions": {
                    "expected_state": {
                        "windows": [{"process_equals": browser_exe, "must_exist": True}]
                    }
                },
                "action": "window.focus",
                "args": {"process": browser_exe},
                "evidence_required": ["driver.active_window"],
                "success_spec": {"expected_state": expected_browser_active},
                "on_fail": "abort",
            },
            {
                "id": "task-3",
                "intent": "Abrir la barra de direcciones (ctrl+l) para introducir la búsqueda o URL.",
                "preconditions": {"expected_state": expected_browser_active},
                "action": "keyboard.hotkey",
                "args": {"keys": ["ctrl", "l"]},
                "evidence_required": ["driver.active_window"],
                "success_spec": {"expected_state": expected_browser_active},
                "on_fail": "abort",
            },
            {
                "id": "task-4",
                "intent": "Introducir la consulta y navegar con Enter.",
                "preconditions": {"expected_state": expected_browser_active},
                "action": "keyboard.type",
                "args": {"text": intention, "submit": True},
                "evidence_required": ["driver.screenshot"],
                "success_spec": {"expected_state": expected_browser_active},
                "on_fail": "abort",
            },
        ]
        plan_id = f"media-playback-keyboard-only-{int(time.time())}"
        meta: Dict[str, Any] = {
            "intention": intention,
            "source": "degraded:keyboard_only",
            "intent_class": "media_playback",
        }
        if degraded_reason:
            meta["degraded_reason"] = degraded_reason
        return AjaxPlan(
            id=plan_id,
            summary="Degraded (sin vision): abrir navegador y buscar para reproducción de media",
            steps=steps,
            plan_id=plan_id,
            metadata=meta,
            success_spec=(
                self._default_media_playback_success_contract(intention)
                if hasattr(self, "_default_media_playback_success_contract")
                else {"type": "check_last_step_status"}
            ),
        )

    def _apply_brain_retry_policy(self, mission: MissionState) -> None:
        providers = set()
        try:
            for attempt in mission.brain_attempts or []:
                if not isinstance(attempt, dict):
                    continue
                provider = str(attempt.get("id") or "").strip()
                if provider:
                    providers.add(provider)
        except Exception:
            providers = set()
        cooldown_until = time.time() + self._provider_cooldown_seconds()
        for provider in providers:
            try:
                mission.brain_exclude.add(provider)
                mission.provider_cooldowns[provider] = max(
                    mission.provider_cooldowns.get(provider, 0.0), cooldown_until
                )
            except Exception:
                pass
        try:
            mission.brain_retry_level = int(mission.brain_retry_level or 0) + 1
        except Exception:
            mission.brain_retry_level = 1
        next_mode = self._next_cost_mode(mission.cost_mode or mission.brain_cost_mode_override)
        mission.cost_mode = next_mode
        mission.brain_cost_mode_override = next_mode
        self._rebuild_starting_xi_for_mission(mission, cost_mode=next_mode)

    def _record_brain_decision_trace(
        self,
        mission: Optional[MissionState],
        trace: Optional[Dict[str, Any]],
        *,
        attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        if mission is None:
            return None
        try:
            payload = json.loads(json.dumps(trace)) if isinstance(trace, dict) else {}
            if not isinstance(payload, dict):
                payload = {}
            if attempts is None:
                attempts = getattr(self, "_last_brain_attempts", None)
            normalized_attempts = self._normalize_router_attempts(attempts)
            try:
                if not normalized_attempts and isinstance(mission.notes, dict):
                    normalized_attempts = self._normalize_router_attempts(
                        mission.notes.get("router_attempts")
                    )
            except Exception:
                pass
            target = self.config.root_dir / "artifacts" / "router"
            target.mkdir(parents=True, exist_ok=True)
            path = target / f"decision_{mission.mission_id}.json"
            existing_payload: Optional[Dict[str, Any]] = None
            if not normalized_attempts and path.exists():
                try:
                    existing_raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(existing_raw, dict):
                        existing_payload = existing_raw
                        normalized_attempts = self._normalize_router_attempts(
                            existing_raw.get("attempts")
                        )
                except Exception:
                    existing_payload = None
            if existing_payload and not payload:
                payload = existing_payload
            payload["attempts"] = normalized_attempts
            failures = [a for a in normalized_attempts if not a.get("ok")]
            any_ok = any(a.get("ok") for a in normalized_attempts)
            error_counts: Dict[str, int] = {}
            for att in failures:
                code = str(att.get("error_code") or "unknown")
                error_counts[code] = error_counts.get(code, 0) + 1
            top_causes = [
                k for k, _ in sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
            ]
            all_failed_reason = None
            if normalized_attempts and not any_ok:
                if top_causes:
                    all_failed_reason = "all_failed:" + ", ".join(top_causes)
                else:
                    all_failed_reason = "all_failed"
            premium_attempt_seen = any(
                str(a.get("tier") or "").strip().lower() == "premium" for a in normalized_attempts
            )
            payload["summary"] = {
                "all_failed_reason": all_failed_reason,
                "premium_attempt_seen": bool(premium_attempt_seen),
                "top_causes": top_causes,
            }
            policy = payload.setdefault("policy", {})
            policy["premium_rule"] = getattr(mission, "premium_rule", "if_needed")
            if mission.cost_mode:
                policy["cost_mode"] = mission.cost_mode
            payload["mission_id"] = mission.mission_id
            payload["ts"] = time.time()
            payload["ts_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                rel = os.path.relpath(path, self.config.root_dir)
            except Exception:
                rel = str(path)
            try:
                mission.notes["router_trace_path"] = rel
                if normalized_attempts:
                    mission.notes["router_attempts"] = normalized_attempts
            except Exception:
                pass
            return str(path)
        except Exception:
            try:
                self.log.warning("No se pudo registrar decision trace", exc_info=True)
            except Exception:
                pass
        return None

    def _record_chat_selection_receipt(
        self,
        *,
        provider: str,
        cfg: Dict[str, Any],
        trace: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        try:
            payload = json.loads(json.dumps(trace)) if isinstance(trace, dict) else {}
            if not isinstance(payload, dict):
                payload = {}
            candidates = payload.get("candidates") if isinstance(payload, dict) else None
            reason_codes: Dict[str, List[str]] = {}
            ttls: Dict[str, Dict[str, Any]] = {}
            if isinstance(candidates, list):
                for entry in candidates:
                    if not isinstance(entry, dict):
                        continue
                    name = str(entry.get("provider") or "").strip()
                    if not name:
                        continue
                    rejects = entry.get("reject_codes")
                    if isinstance(rejects, list) and rejects:
                        reason_codes[name] = [str(x) for x in rejects if x]
                    cooldown_ts = entry.get("cooldown_until_ts")
                    cooldown_utc = entry.get("cooldown_until")
                    if cooldown_ts or cooldown_utc:
                        ttls[name] = {
                            "cooldown_until": cooldown_utc,
                            "cooldown_until_ts": cooldown_ts,
                        }
            selected = {
                "provider": provider,
                "model": cfg.get("_selected_model") or cfg.get("default_model") or cfg.get("model"),
                "tier": cfg.get("tier"),
            }
            payload.update(
                {
                    "schema": "ajax.chat_selection.v1",
                    "ts": time.time(),
                    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "selected": selected,
                    "fallback_chain": payload.get("fallback_chain")
                    or payload.get("ranked_candidates")
                    or [],
                    "reason_codes": reason_codes,
                    "ttls": ttls,
                }
            )
            target = self.config.root_dir / "artifacts" / "health"
            target.mkdir(parents=True, exist_ok=True)
            path = target / "chat_selection.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(path)
        except Exception:
            return None

    def _infer_brain_error_code(self, raw: Any) -> str:
        msg = str(raw or "").strip().lower()
        if not msg:
            return "unknown"
        if msg.startswith("brain_http_"):
            return msg.split(":", 1)[0].replace("brain_", "")
        if msg.startswith("http_"):
            return msg.split(":", 1)[0]
        if msg.startswith("cli_rc_"):
            return "cli_rc"
        if msg.startswith("cli_failed"):
            return "cli_failed"
        if "auth_missing" in msg:
            return "auth_missing"
        if "auth_expired" in msg:
            return "auth_expired"
        if "client_timeout" in msg:
            return "client_timeout"
        if "timeout" in msg or "timed out" in msg or "deadline" in msg:
            return "timeout"
        if "json" in msg or "parse" in msg:
            return "parse_error"
        if "empty_plan" in msg:
            return "empty_plan"
        if "empty_reply" in msg:
            return "empty_reply"
        return "unknown"

    def _normalize_router_attempts(
        self, attempts: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if not attempts or not isinstance(attempts, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            role = attempt.get("role") or "brain"
            provider = attempt.get("provider") or attempt.get("id")
            if not provider:
                continue
            ok = attempt.get("ok")
            if ok is None:
                ok = str(attempt.get("result") or "").strip().lower() == "ok"
            error_code = attempt.get("error_code")
            if not ok and not error_code:
                error_code = self._infer_brain_error_code(attempt.get("result"))
            error_detail = attempt.get("error_detail")
            if not ok and not error_detail:
                error_detail = attempt.get("result")
            if isinstance(error_detail, str) and len(error_detail) > 200:
                error_detail = error_detail[:200]
            ts = attempt.get("ts")
            if not isinstance(ts, (int, float)) or ts <= 0:
                ts = time.time()
            payload = {
                "role": role,
                "provider": provider,
                "model": attempt.get("model"),
                "tier": attempt.get("tier"),
                "ok": bool(ok),
                "error_code": error_code if not ok else None,
                "error_detail": error_detail if not ok else None,
                "stderr_excerpt": attempt.get("stderr_excerpt"),
                "timeout_s": attempt.get("timeout_s"),
                "parse_error": attempt.get("parse_error"),
                "latency_ms": attempt.get("latency_ms"),
                "t_start": attempt.get("t_start"),
                "t_connect_ok": attempt.get("t_connect_ok"),
                "t_first_output": attempt.get("t_first_output"),
                "t_last_output": attempt.get("t_last_output"),
                "t_end": attempt.get("t_end"),
                "timeout_kind": attempt.get("timeout_kind"),
                "client_abort": attempt.get("client_abort"),
                "exit_code": attempt.get("exit_code"),
                "stderr_tail": attempt.get("stderr_tail"),
                "bytes_rx": attempt.get("bytes_rx"),
                "tokens_rx": attempt.get("tokens_rx"),
                "result": attempt.get("result"),
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts))),
            }
            normalized.append(payload)
        return normalized

    def _active_provider_cooldowns(self, mission: MissionState) -> set[str]:
        active: set[str] = set()
        try:
            cooldowns = mission.provider_cooldowns
        except Exception:
            cooldowns = {}
        now = time.time()
        expired: List[str] = []
        for provider, until in list(cooldowns.items()):
            try:
                if until <= now:
                    expired.append(provider)
                else:
                    active.add(provider)
            except Exception:
                expired.append(provider)
        for provider in expired:
            cooldowns.pop(provider, None)
            try:
                mission.brain_exclude.discard(provider)
            except Exception:
                pass
        mission.brain_exclude.update(active)
        return active

    def _providers_on_timeout_cooldown(self) -> set[str]:
        """
        Cross-mission cooldown based on persisted timeout receipts.
        Deterministic: reads latest receipts under artifacts/provider_timeouts/*/receipt.json.
        """
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        base = Path(root_dir) / "artifacts" / "provider_timeouts"
        if not base.exists():
            return set()
        try:
            dirs = sorted(
                [p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True
            )
        except Exception:
            return set()
        now = time.time()
        active: set[str] = set()
        # Limit scan to avoid overhead.
        for p in dirs[:30]:
            receipt = p / "receipt.json"
            if not receipt.exists():
                continue
            try:
                doc = json.loads(receipt.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            provider = str(doc.get("provider") or "").strip()
            if not provider:
                continue
            ts = doc.get("ts")
            cooldown = doc.get("cooldown_seconds")
            if not isinstance(ts, (int, float)) or ts <= 0:
                continue
            if not isinstance(cooldown, (int, float)) or cooldown <= 0:
                continue
            if float(ts) + float(cooldown) > now:
                active.add(provider)
        return active

    def _codex_timeout_cooldown_seconds(self) -> int:
        raw = os.getenv("AJAX_CODEX_TIMEOUT_COOLDOWN_SEC")
        if raw:
            try:
                val = int(raw)
                if val > 0:
                    return val
            except Exception:
                pass
        return 600

    def _register_provider_failure(
        self,
        mission: MissionState,
        failure: "ProviderFailure",
        *,
        cooldown_override_seconds: Optional[int] = None,
    ) -> None:
        ttl = int(cooldown_override_seconds or self._provider_cooldown_seconds())
        try:
            mission.provider_cooldowns[failure.provider] = time.time() + ttl
            mission.brain_exclude.add(failure.provider)
        except Exception:
            pass
        entry = failure.to_dict()
        try:
            entry.setdefault("context", {})
            if isinstance(entry.get("context"), dict):
                entry["context"].setdefault("cooldown_seconds", ttl)
        except Exception:
            pass
        try:
            receipt_path = self._record_provider_failure_receipt(
                mission=mission, failure=failure, cooldown_seconds=ttl
            )
            if receipt_path:
                entry.setdefault("context", {})
                if isinstance(entry.get("context"), dict):
                    entry["context"]["receipt_path"] = receipt_path
        except Exception:
            pass
        try:
            mission.provider_failures.append(entry)
        except Exception:
            mission.provider_failures = [entry]
        try:
            notes = mission.notes if isinstance(mission.notes, dict) else {}
            notes.setdefault("provider_failures", []).append(entry)
            mission.notes = notes
        except Exception:
            pass

    def _record_provider_timeout_receipt(
        self,
        *,
        mission: Optional[MissionState],
        failure: "ProviderFailure",
        attempt: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "provider_timeouts" / ts_label
            out_dir.mkdir(parents=True, exist_ok=True)
            policy, policy_hash = self._provider_timeouts_policy()
            policy_ver = None
            if timeouts_policy_version is not None:
                try:
                    policy_ver = timeouts_policy_version(policy)
                except Exception:
                    policy_ver = None
            payload: Dict[str, Any] = {
                "schema": "ajax.provider_timeout.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "provider": failure.provider,
                "stage": failure.stage,
                "message": failure.message,
                "mission_id": mission.mission_id if mission else None,
                "intention": mission.intention if mission else None,
                "cooldown_seconds": (failure.context or {}).get("cooldown_seconds"),
                "timeout_s": (attempt or {}).get("timeout_s"),
                "latency_ms": (attempt or {}).get("latency_ms"),
                "timeout_kind": (attempt or {}).get("timeout_kind") or "NONE",
                "client_abort": (attempt or {}).get("client_abort"),
                "exit_code": (attempt or {}).get("exit_code"),
                "stderr_tail": (attempt or {}).get("stderr_tail"),
                "bytes_rx": (attempt or {}).get("bytes_rx"),
                "tokens_rx": (attempt or {}).get("tokens_rx"),
                "t_start": (attempt or {}).get("t_start"),
                "t_connect_ok": (attempt or {}).get("t_connect_ok"),
                "t_first_output": (attempt or {}).get("t_first_output"),
                "t_last_output": (attempt or {}).get("t_last_output"),
                "t_end": (attempt or {}).get("t_end"),
                "policy_version": policy_ver,
                "policy_hash": policy_hash,
            }
            path = out_dir / "receipt.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                self._update_provider_timeouts_rollup(payload)
            except Exception:
                pass
            return str(path)
        except Exception:
            return None

    def _update_provider_timeouts_rollup(self, payload: Dict[str, Any]) -> None:
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        path = Path(root_dir) / "artifacts" / "metrics" / "provider_timeouts_rollup.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {}
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        months = data.get("months") if isinstance(data.get("months"), dict) else {}
        if not isinstance(months, dict):
            months = {}
        ts = payload.get("ts")
        if not isinstance(ts, (int, float)):
            ts = time.time()
        month_key = time.strftime("%Y-%m", time.gmtime(float(ts)))
        month_entry = months.get(month_key)
        if not isinstance(month_entry, dict):
            month_entry = {"providers": {}}
        providers = month_entry.get("providers")
        if not isinstance(providers, dict):
            providers = {}
        provider = str(payload.get("provider") or "").strip() or "unknown"
        prov_entry = providers.get(provider)
        if not isinstance(prov_entry, dict):
            prov_entry = {"total": 0, "timeout_kinds": {}, "last_ts": None}
        prov_entry["total"] = int(prov_entry.get("total") or 0) + 1
        timeout_kind = str(payload.get("timeout_kind") or "NONE")
        tk = prov_entry.get("timeout_kinds")
        if not isinstance(tk, dict):
            tk = {}
        tk[timeout_kind] = int(tk.get(timeout_kind) or 0) + 1
        prov_entry["timeout_kinds"] = tk
        prov_entry["last_ts"] = float(ts)
        providers[provider] = prov_entry
        month_entry["providers"] = providers
        months[month_key] = month_entry
        data["schema"] = "ajax.provider_timeouts_rollup.v1"
        data["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))
        data["months"] = months
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _record_pipeline_fail_receipt(
        self,
        *,
        mission: Optional[MissionState],
        provider: str,
        stage: str,
        reason: str,
        attempt: Dict[str, Any],
    ) -> Optional[str]:
        try:
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "receipts"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "ajax.pipeline_fail.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "provider": provider,
                "stage": stage,
                "reason": reason,
                "mission_id": mission.mission_id if mission else None,
                "intention": mission.intention if mission else None,
                "attempt": {
                    "error_code": attempt.get("error_code"),
                    "error_detail": attempt.get("error_detail"),
                    "parse_error": attempt.get("parse_error"),
                    "raw_plan_excerpt": attempt.get("raw_plan_excerpt"),
                },
            }
            path = out_dir / f"pipeline_fail_{ts_label}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _emit_infra_bridge_gap(
        self,
        *,
        mission: Optional[MissionState],
        provider: str,
        message: str,
        attempt: Dict[str, Any],
    ) -> Optional[str]:
        try:
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        except Exception:
            root_dir = Path(".")
        try:
            gap_dir = Path(root_dir) / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            mission_id = mission.mission_id if mission else "unknown"
            gap_id = f"{ts}_infra_bridge_error_{mission_id}"
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "infra_bridge_error",
                "mission_id": mission_id,
                "intent": mission.intention if mission else None,
                "provider": provider,
                "reason": "infra_bridge_error",
                "message": message[:400] if message else None,
                "attempt": {
                    "error_code": attempt.get("error_code"),
                    "error_detail": attempt.get("error_detail"),
                    "stderr_tail": attempt.get("stderr_tail"),
                },
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(out_path)
        except Exception:
            return None

    def _record_provider_failure_receipt(
        self,
        *,
        mission: Optional[MissionState],
        failure: "ProviderFailure",
        cooldown_seconds: Optional[int] = None,
    ) -> Optional[str]:
        try:
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "receipts"
            out_dir.mkdir(parents=True, exist_ok=True)
            policy = {}
            try:
                policy = self._provider_failure_policy()
            except Exception:
                policy = {}
            required_fields: List[str] = []
            if failure_receipt_required_fields is not None:
                try:
                    required_fields = list(failure_receipt_required_fields(policy))
                except Exception:
                    required_fields = []
            reason = None
            try:
                ctx = failure.context if isinstance(failure.context, dict) else {}
                reason = ctx.get("reason") or ctx.get("cooldown_reason")
            except Exception:
                reason = None
            payload: Dict[str, Any] = {
                "schema": "ajax.provider_failure_receipt.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "ok": False,
                "reason": str(reason or failure.code or "UNKNOWN"),
                "provider": failure.provider,
                "error_code": failure.code,
                "stage": failure.stage,
                "message": (failure.message or "")[:800],
                "mission_id": mission.mission_id if mission else None,
                "intention": mission.intention if mission else None,
                "cooldown_seconds": int(cooldown_seconds)
                if isinstance(cooldown_seconds, int)
                else None,
                "context": failure.context if isinstance(failure.context, dict) else {},
                "remediation": list(failure.remediation)
                if isinstance(failure.remediation, list)
                else [],
            }
            for key in required_fields:
                payload.setdefault(key, None)
            path = out_dir / f"provider_failure_{ts_label}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _record_prompt_pack_receipt(
        self,
        *,
        mission_id: Optional[str],
        intention: Optional[str],
        pack_id: str,
        context_budget: Optional[int],
        escalation_reason: Optional[str],
        decision: Optional[str],
        reason: Optional[str],
        filters: Optional[Dict[str, Any]],
        confidence: Optional[float],
    ) -> Optional[str]:
        try:
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "metrics"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "ajax.prompt_compile.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "mission_id": mission_id,
                "intention": intention,
                "pack_id": pack_id,
                "context_budget": context_budget,
                "budget": context_budget,
                "escalation_reason": escalation_reason,
                "decision": decision,
                "reason": reason,
                "filters": filters or {"mode": "none"},
                "confidence": confidence,
            }
            path = out_dir / f"prompt_compile_{ts_label}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _record_pending_mission_receipt(
        self,
        *,
        mission_id: str,
        choice: str,
        transition: str,
    ) -> Optional[str]:
        try:
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "metrics"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "ajax.pending_mission.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "mission_id": mission_id,
                "choice": choice,
                "transition": transition,
            }
            path = out_dir / f"pending_mission_{ts_label}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _record_lab_queue_receipt(
        self,
        *,
        mission_id: str,
        job_kind: str,
        validated: bool,
        reason: str,
    ) -> Optional[str]:
        try:
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "lab" / "queue_receipts"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "ajax.lab.queue_receipt.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "mission_id": mission_id,
                "job_kind": job_kind,
                "validated": bool(validated),
                "reason": reason,
            }
            path = out_dir / f"{ts_label}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _loop_guard_threshold(self) -> int:
        raw = (os.getenv("AJAX_LOOP_GUARD_THRESHOLD") or "2").strip()
        try:
            return max(1, int(raw))
        except Exception:
            return 2

    def _loop_guard_verbose(self) -> bool:
        return str(os.getenv("AJAX_LOOP_GUARD_VERBOSE") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _hash_file(self, path: Optional[Path]) -> Optional[str]:
        if not path:
            return None
        try:
            if not path.exists():
                return None
            return self._hash_bytes(path.read_bytes())
        except Exception:
            return None

    @staticmethod
    def _hash_dict(doc: Dict[str, Any]) -> str:
        raw = json.dumps(doc, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _tool_state_snapshot(self, mission: Optional[MissionState]) -> Dict[str, Any]:
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        vision_status = None
        try:
            hb = root_dir / "artifacts" / "health" / "ajax_heartbeat.json"
            if hb.exists():
                data = json.loads(hb.read_text(encoding="utf-8"))
                vision_status = (data.get("subsystems") or {}).get("vision", {}).get("status")
        except Exception:
            vision_status = None
        lab_status = None
        try:
            store = LabStateStore(root_dir)
            lab_status = (store.state.get("lab_org") or {}).get("status")
        except Exception:
            lab_status = None
        return {
            "vision_status": vision_status,
            "lab_status": lab_status,
            "lab_job_id": getattr(mission, "lab_job_id", None) if mission else None,
        }

    def _compute_progress_token(
        self,
        mission: MissionState,
        *,
        result: Optional[AjaxExecutionResult] = None,
        pending_question: Optional[str] = None,
    ) -> str:
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        last_provider = None
        last_model = None
        attempts = getattr(mission, "brain_attempts", None) or []
        if isinstance(attempts, list) and attempts:
            last = attempts[-1] if isinstance(attempts[-1], dict) else None
            if last:
                last_provider = last.get("provider") or last.get("id")
                last_model = last.get("model")
        question = pending_question
        if not question and result and isinstance(result.detail, dict):
            question = result.detail.get("question")
        budget_exhausted = False
        if isinstance(question, str) and "[budget_exhausted]" in question.lower():
            budget_exhausted = True
        if isinstance(mission.notes, dict) and mission.notes.get("budget_exhausted"):
            budget_exhausted = True
        waiting_path = self._waiting_mission_path()
        per_path = root_dir / "artifacts" / "waiting_for_user" / f"{mission.mission_id}.json"
        last_receipt_summary = None
        if result is not None:
            last_receipt_summary = {
                "mission_id": mission.mission_id,
                "path": result.path,
                "error": result.error,
                "success": bool(result.success),
            }
        data = {
            "mission_id": mission.mission_id,
            "mission_state": mission.status,
            "pending_question": question,
            "last_provider": last_provider,
            "last_model": last_model,
            "budget_state": {
                "attempts": mission.attempts,
                "plan_attempts": mission.plan_attempts,
                "budget_exhausted": budget_exhausted,
            },
            "key_artifacts_hashes": [
                self._hash_file(waiting_path),
                self._hash_file(per_path),
                self._hash_dict(last_receipt_summary) if last_receipt_summary else None,
            ],
            "tool_state": self._tool_state_snapshot(mission),
        }
        raw = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8", errors="surrogatepass"
        )
        return self._hash_bytes(raw)

    def _update_progress_tracking(
        self,
        mission: MissionState,
        *,
        result: Optional[AjaxExecutionResult] = None,
        pending_question: Optional[str] = None,
        source: str = "exec",
    ) -> Dict[str, Any]:
        token = self._compute_progress_token(
            mission, result=result, pending_question=pending_question
        )
        prev = mission.progress_token
        count = int(mission.progress_no_change_count or 0)
        if prev and token == prev:
            count += 1
        else:
            count = 0
        mission.progress_token_prev = prev
        mission.progress_token = token
        mission.progress_no_change_count = count
        try:
            if isinstance(mission.notes, dict):
                mission.notes["progress_token_source"] = source
        except Exception:
            pass
        loop_guard_triggered = False
        if count >= self._loop_guard_threshold():
            if not mission.loop_guard:
                loop_guard_triggered = True
            mission.loop_guard = True
        return {
            "progress_token": token,
            "progress_token_prev": prev,
            "progress_no_change_count": count,
            "loop_guard_triggered": loop_guard_triggered,
        }

    @staticmethod
    def _provider_failures_dead(failures: Optional[List[Dict[str, Any]]]) -> bool:
        if not failures or not isinstance(failures, list):
            return False
        for row in failures:
            if not isinstance(row, dict):
                continue
            code = str(row.get("error_code") or row.get("status") or "").lower()
            detail = str(row.get("error_detail") or row.get("detail") or "").lower()
            text = f"{code} {detail}"
            if any(
                tok in text
                for tok in {
                    "timeout",
                    "no_viable_provider",
                    "auth_missing",
                    "remote_http_disabled",
                    "no_provider",
                }
            ):
                return True
        return False

    def _record_exec_receipt(
        self,
        *,
        mission: Optional[MissionState] = None,
        mission_id: Optional[str],
        plan_id: Optional[str],
        result: AjaxExecutionResult,
        verify_ok: bool,
    ) -> Optional[str]:
        def _provider_errors_from_attempts(attempts: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(attempts, list):
                return None

            def _extract_http_status(code: Any, detail: Any) -> Optional[int]:
                for raw in (code, detail):
                    if not raw:
                        continue
                    text = str(raw).lower()
                    if "http_" in text:
                        idx = text.find("http_") + 5
                        digits = ""
                        for ch in text[idx:]:
                            if ch.isdigit():
                                digits += ch
                            else:
                                break
                        if digits:
                            return int(digits)
                return None

            by_provider: Dict[str, Any] = {}
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                provider = str(attempt.get("provider") or attempt.get("id") or "").strip()
                if not provider:
                    continue
                if attempt.get("ok") is True:
                    continue
                code = attempt.get("error_code") or "unknown"
                detail = attempt.get("error_detail") or attempt.get("result")
                by_provider[provider] = {
                    "timeout": bool(code == "timeout" or "timeout" in str(detail or "").lower()),
                    "http_status": _extract_http_status(code, detail),
                    "status": code,
                    "last_error": detail,
                    "stage": attempt.get("stage"),
                    "raw_plan_excerpt": (
                        attempt.get("raw_plan_excerpt")[:500]
                        if isinstance(attempt.get("raw_plan_excerpt"), str)
                        else None
                    ),
                    "coerced_fields": attempt.get("coerced_fields")
                    if isinstance(attempt.get("coerced_fields"), list)
                    else None,
                    "legacy_adapt_applied": bool(attempt.get("legacy_adapt_applied"))
                    if attempt.get("legacy_adapt_applied") is not None
                    else None,
                    "stage_sequence": attempt.get("stage_sequence")
                    if isinstance(attempt.get("stage_sequence"), list)
                    else None,
                }
            return by_provider or None

        def _last_raw_excerpt(attempts: Any) -> Optional[str]:
            if isinstance(attempts, list):
                for attempt in reversed(attempts):
                    if not isinstance(attempt, dict):
                        continue
                    excerpt = attempt.get("raw_plan_excerpt")
                    if isinstance(excerpt, str) and excerpt.strip():
                        return excerpt[:500]
            val = getattr(self, "_last_raw_plan_excerpt", None)
            if isinstance(val, str) and val.strip():
                return val[:500]
            return None

        try:
            if not mission_id and isinstance(result.detail, dict):
                raw_id = result.detail.get("mission_id")
                if isinstance(raw_id, str) and raw_id.strip():
                    mission_id = raw_id.strip()
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "receipts"
            out_dir.mkdir(parents=True, exist_ok=True)
            progress_payload = {}
            if mission is not None:
                skip_update = False
                try:
                    if result.error == "await_user_input":
                        src = (
                            mission.notes.get("progress_token_source")
                            if isinstance(mission.notes, dict)
                            else None
                        )
                        skip_update = src == "waiting"
                except Exception:
                    skip_update = False
                if skip_update:
                    progress_payload = {
                        "progress_token": mission.progress_token,
                        "progress_token_prev": mission.progress_token_prev,
                        "progress_no_change_count": mission.progress_no_change_count,
                        "loop_guard": bool(mission.loop_guard),
                        "loop_guard_triggered": False,
                    }
                else:
                    progress_payload = self._update_progress_tracking(
                        mission, result=result, source="exec"
                    )
                    progress_payload["loop_guard"] = bool(mission.loop_guard)
            payload = {
                "schema": "ajax.exec_receipt.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "mission_id": mission_id,
                "plan_id": plan_id,
                "verify_ok": bool(verify_ok),
                "success": bool(verify_ok),
                "path": result.path,
                "error": result.error,
                "detail": result.detail if isinstance(result.detail, dict) else None,
                "provider_errors": _provider_errors_from_attempts(
                    getattr(self, "_last_brain_attempts", None)
                ),
                "last_raw_plan_excerpt": _last_raw_excerpt(
                    getattr(self, "_last_brain_attempts", None)
                ),
            }
            payload.update(progress_payload)
            path = out_dir / f"exec_{ts_label}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                if mission is not None and isinstance(mission.notes, dict):
                    mission.notes["last_exec_receipt"] = str(path)
            except Exception:
                pass
            try:
                if hasattr(self, "state") and isinstance(self.state, AjaxStateSnapshot):
                    self.state.notes["last_exec_receipt"] = str(path)
            except Exception:
                pass
            return str(path)
        except Exception:
            return None

    def _record_waiting_receipt(
        self,
        *,
        mission: MissionState,
        question: str,
        payload: AskUserPayload,
        loop_info: Optional[Dict[str, Any]] = None,
        waiting_path: Optional[str] = None,
        payload_path: Optional[str] = None,
    ) -> Optional[str]:
        try:
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "receipts"
            out_dir.mkdir(parents=True, exist_ok=True)
            loop_info = loop_info or {}
            payload_doc = {
                "schema": "ajax.waiting_receipt.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "mission_id": mission.mission_id,
                "question": question,
                "options": payload.options,
                "default_option": payload.default,
                "expects": payload.expects,
                "progress_token": mission.progress_token,
                "progress_token_prev": mission.progress_token_prev,
                "progress_no_change_count": mission.progress_no_change_count,
                "loop_guard": bool(mission.loop_guard),
                "loop_guard_triggered": bool(loop_info.get("loop_guard_triggered")),
                "waiting_mission_path": waiting_path,
                "waiting_payload_path": payload_path,
            }
            path = out_dir / f"waiting_{ts_label}.json"
            path.write_text(
                json.dumps(payload_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _record_brain_failover_receipt(
        self,
        *,
        mission: Optional[MissionState],
        intention: str,
    ) -> Optional[str]:
        attempts = getattr(self, "_last_brain_attempts", None) or []
        if not attempts or not isinstance(attempts, list):
            return None

        def _extract_http_status(
            error_code: Optional[str], error_detail: Optional[str]
        ) -> Optional[int]:
            for raw in (error_code, error_detail):
                if not raw:
                    continue
                text = str(raw).lower()
                if "http_" in text:
                    idx = text.find("http_") + 5
                    digits = ""
                    for ch in text[idx:]:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    if digits:
                        return int(digits)
                if "http " in text:
                    idx = text.find("http ") + 5
                    digits = ""
                    for ch in text[idx:]:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    if digits:
                        return int(digits)
            return None

        by_provider: Dict[str, Dict[str, Any]] = {}
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            provider = str(attempt.get("provider") or attempt.get("id") or "").strip()
            if not provider:
                continue
            attempt_ts = attempt.get("ts")
            prev = by_provider.get(provider)
            if (
                prev
                and isinstance(prev.get("_ts"), (int, float))
                and isinstance(attempt_ts, (int, float))
            ):
                if attempt_ts <= prev["_ts"]:
                    continue
            error_code = str(attempt.get("error_code") or "").strip().lower()
            error_detail = str(attempt.get("error_detail") or attempt.get("result") or "").strip()
            timeout = bool(
                error_code == "timeout"
                or "timeout" in error_detail.lower()
                or "timed out" in error_detail.lower()
            )
            http_status = _extract_http_status(error_code, error_detail)
            parse_error = bool(attempt.get("parse_error")) or error_code == "parse_error"
            status = error_code or "unknown"
            stage = attempt.get("stage")
            raw_excerpt = attempt.get("raw_plan_excerpt")
            coerced_fields = attempt.get("coerced_fields")
            legacy_adapt_applied = attempt.get("legacy_adapt_applied")
            stage_sequence = attempt.get("stage_sequence")
            by_provider[provider] = {
                "timeout": timeout,
                "http_status": http_status,
                "parse_error": parse_error,
                "status": status,
                "last_error": error_detail,
                "stage": stage,
                "raw_plan_excerpt": raw_excerpt[:500]
                if isinstance(raw_excerpt, str) and raw_excerpt.strip()
                else None,
                "coerced_fields": coerced_fields if isinstance(coerced_fields, list) else None,
                "legacy_adapt_applied": bool(legacy_adapt_applied)
                if legacy_adapt_applied is not None
                else None,
                "stage_sequence": stage_sequence if isinstance(stage_sequence, list) else None,
                "_ts": attempt_ts if isinstance(attempt_ts, (int, float)) else None,
            }

        if not by_provider:
            return None

        try:
            ts = time.time()
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            out_dir = Path(root_dir) / "artifacts" / "brains_failures"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "ajax.brain_failover.v1",
                "ts": ts,
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "mission_id": mission.mission_id if mission else None,
                "intention": intention,
                "providers": {
                    name: {
                        "timeout": info.get("timeout"),
                        "http": info.get("http_status"),
                        "http_status": info.get("http_status"),
                        "parse_error": info.get("parse_error"),
                        "status": info.get("status"),
                        "last_error": info.get("last_error"),
                        "stage": info.get("stage"),
                        "raw_plan_excerpt": info.get("raw_plan_excerpt"),
                        "coerced_fields": info.get("coerced_fields"),
                        "legacy_adapt_applied": info.get("legacy_adapt_applied"),
                        "stage_sequence": info.get("stage_sequence"),
                    }
                    for name, info in by_provider.items()
                },
            }
            path = out_dir / f"{ts_label}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _record_provider_failures_from_attempts(
        self,
        mission: Optional[MissionState],
        *,
        stage: str,
        intention: str,
    ) -> List[Dict[str, Any]]:
        if (
            mission is None
            or not getattr(self, "_last_brain_attempts", None)
            or classify_provider_failure is None
        ):
            return []
        failures: List[Dict[str, Any]] = []
        attempts = getattr(self, "_last_brain_attempts", []) or []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            result = attempt.get("result")
            if not result or str(result).strip().lower() == "ok":
                continue
            provider = str(attempt.get("id") or "").strip()
            if not provider:
                continue
            message = str(result)
            message_lower = message.lower()
            error_code = str(attempt.get("error_code") or "").strip().lower()
            parse_error = bool(attempt.get("parse_error")) or error_code == "parse_error"
            empty_plan = error_code == "empty_plan" or "empty_plan" in message_lower
            latency_only = False
            if "latency" in message_lower:
                bad_tokens = (
                    "timeout",
                    "timed out",
                    "deadline",
                    "error",
                    "fail",
                    "unavailable",
                    "auth",
                    "eperm",
                    "http",
                    "429",
                    "403",
                    "401",
                )
                latency_only = not any(tok in message_lower for tok in bad_tokens)
            if latency_only:
                continue
            if parse_error or empty_plan:
                try:
                    self._record_pipeline_fail_receipt(
                        mission=mission,
                        provider=provider,
                        stage=stage,
                        reason="parse_error" if parse_error else "empty_plan",
                        attempt=attempt,
                    )
                except Exception:
                    pass
                continue

            provider_cfg = {}
            try:
                providers_cfg = (
                    self.provider_configs.get("providers", {})
                    if isinstance(self.provider_configs, dict)
                    else {}
                )
                provider_cfg = (
                    providers_cfg.get(provider) if isinstance(providers_cfg, dict) else {}
                )
            except Exception:
                provider_cfg = {}
            kind = str((provider_cfg or {}).get("kind") or "").strip().lower()
            cmd = provider_cfg.get("command") if isinstance(provider_cfg, dict) else None
            cmd_list = cmd if isinstance(cmd, list) else []
            is_bridge = bool(
                kind == "cli" and any("provider_cli_bridge.py" in str(x) for x in cmd_list)
            )
            infra_bridge_error = False
            if error_code == "infra_bridge_error":
                infra_bridge_error = True
            elif is_bridge and (
                error_code.startswith("cli_rc_") or error_code in {"cli_failed", "bridge_error"}
            ):
                infra_bridge_error = True
            if infra_bridge_error:
                try:
                    self._emit_infra_bridge_gap(
                        mission=mission,
                        provider=provider,
                        message=message,
                        attempt=attempt,
                    )
                except Exception:
                    pass
                try:
                    policy = self._provider_failure_policy()
                    cooldown = None
                    if failure_cooldown_seconds_for_reason is not None:
                        cooldown = failure_cooldown_seconds_for_reason(
                            policy, reason="infra_bridge_error", default=120
                        )
                    if cooldown is None:
                        cooldown = 120
                    mission.provider_cooldowns[provider] = time.time() + float(cooldown)
                    mission.brain_exclude.add(provider)
                except Exception:
                    pass
                try:
                    root_dir = getattr(getattr(self, "config", None), "root_dir", None)
                    failure = classify_provider_failure(
                        provider,
                        message,
                        stage=stage,
                        context={"error_code": "infra_bridge_error", "intention": intention},
                        root_dir=root_dir,
                    )
                    failures.append(failure.to_dict())
                except Exception:
                    pass
                continue

            timeout_kind = str(attempt.get("timeout_kind") or "").strip().upper()
            client_abort = bool(attempt.get("client_abort"))
            client_timeout = error_code == "client_timeout" or (
                client_abort and timeout_kind in {"TTFT", "STALL", "TOTAL"}
            )

            context = {
                "intention": intention,
                "latency_ms": attempt.get("latency_ms"),
                "stage": stage,
                "timeout_kind": timeout_kind or None,
                "client_abort": client_abort,
            }
            if client_timeout:
                context["error_code"] = "client_timeout"
            try:
                root_dir = getattr(getattr(self, "config", None), "root_dir", None)
                failure = classify_provider_failure(
                    provider, message, stage=stage, context=context, root_dir=root_dir
                )
            except Exception:
                continue
            cooldown_override = None
            if ProviderFailureCode is not None and failure.code == ProviderFailureCode.TIMEOUT:
                cooldown_override = self._codex_timeout_cooldown_seconds()
                try:
                    failure.context = dict(failure.context or {})
                    failure.context["cooldown_reason"] = "timeout"
                except Exception:
                    pass
            if (
                ProviderFailureCode is not None
                and failure.code == ProviderFailureCode.CLIENT_TIMEOUT
            ):
                try:
                    policy = self._provider_failure_policy()
                    if failure_cooldown_seconds_for_reason is not None:
                        reason_key = "client_timeout"
                        if timeout_kind == "TTFT":
                            reason_key = "timeout_ttft"
                        elif timeout_kind == "STALL":
                            reason_key = "timeout_stall"
                        cooldown_override = failure_cooldown_seconds_for_reason(
                            policy, reason=reason_key, default=30
                        )
                except Exception:
                    cooldown_override = 30
                try:
                    failure.context = dict(failure.context or {})
                    failure.context["cooldown_reason"] = (
                        "timeout_ttft"
                        if timeout_kind == "TTFT"
                        else ("timeout_stall" if timeout_kind == "STALL" else "client_timeout")
                    )
                    failure.context["cooldown_seconds"] = cooldown_override
                except Exception:
                    pass
            if ProviderFailureCode is not None and failure.code in {
                ProviderFailureCode.TIMEOUT,
                ProviderFailureCode.CLIENT_TIMEOUT,
            }:
                if cooldown_override is not None:
                    try:
                        failure.context = dict(failure.context or {})
                        failure.context["cooldown_seconds"] = cooldown_override
                    except Exception:
                        pass
                self._record_provider_timeout_receipt(
                    mission=mission, failure=failure, attempt=attempt
                )
            self._register_provider_failure(
                mission, failure, cooldown_override_seconds=cooldown_override
            )
            failures.append(failure.to_dict())
        return failures

    def _format_provider_failure_question(
        self, mission: Optional[MissionState], failures: List[Dict[str, Any]]
    ) -> str:
        debug_enabled = (os.getenv("AJAX_CHAT_DEBUG") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if (
            mission
            and isinstance(mission.notes, dict)
            and mission.notes.get("library_trivial")
            and not debug_enabled
        ):
            base_intent = mission.intention if mission else "la intención actual"
            return (
                f"No pude completar '{base_intent}' con el skill disponible.\n"
                "¿Deseas reintentar o derivar a LAB?"
            )
        if not failures:
            base_intent = mission.intention if mission else "la intención actual"
            base = f"No hay proveedores Brain operativos para {base_intent}."
            extra, ref = self._summarize_router_failure_causes(mission)
            if extra:
                base += f"\nCausas principales: {'; '.join(extra)}"
            if ref:
                base += f"\nRefs: {ref}"
            return f"{base}\n¿Cómo quieres proceder?"
        snippet_lines: List[str] = []
        for entry in failures[:4]:
            provider = entry.get("provider") or "provider"
            code = entry.get("code") or "UNKNOWN"
            detail = entry.get("message") or ""
            if detail:
                detail = detail[:80]
            snippet_lines.append(f"- {provider}: {code}{(' · ' + detail) if detail else ''}")
        summary = "\n".join(snippet_lines)
        base_intent = mission.intention if mission else "la intención actual"
        extra, ref = self._summarize_router_failure_causes(mission)
        extra_block = ""
        if extra:
            extra_block += f"\nCausas principales: {'; '.join(extra)}"
        if ref:
            extra_block += f"\nRefs: {ref}"
        return (
            f"Todos los proveedores Brain fallaron para '{base_intent}'.\n{summary}{extra_block}\n"
            "¿Deseas reintentar con override, abrir un INCIDENT o derivar a LAB?"
        )

    def _summarize_router_failure_causes(
        self, mission: Optional[MissionState]
    ) -> tuple[List[str], Optional[str]]:
        if mission is None or not isinstance(mission.notes, dict):
            return [], None
        attempts = mission.notes.get("router_attempts") or []
        if not isinstance(attempts, list):
            attempts = []
        causes: List[str] = []
        seen: set[str] = set()
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            ok = attempt.get("ok")
            if ok is True:
                continue
            provider = attempt.get("provider") or attempt.get("id") or "provider"
            code = attempt.get("error_code") or self._infer_brain_error_code(attempt.get("result"))
            detail = attempt.get("error_detail") or attempt.get("result") or ""
            detail = str(detail).strip()
            if detail:
                detail = detail[:80]
            key = f"{provider}:{code}"
            if key in seen:
                continue
            seen.add(key)
            if detail:
                causes.append(f"{provider}:{code} · {detail}")
            else:
                causes.append(f"{provider}:{code}")
            if len(causes) >= 2:
                break
        ref = mission.notes.get("router_trace_path") if isinstance(mission.notes, dict) else None
        return causes, str(ref) if ref else None

    def _load_router_trace_summary(
        self,
        mission: Optional[MissionState],
    ) -> tuple[List[str], Optional[str], Optional[str]]:
        if mission is None:
            return [], None, None
        trace_path = None
        try:
            if isinstance(mission.notes, dict):
                trace_path = mission.notes.get("router_trace_path")
        except Exception:
            trace_path = None
        if not trace_path and mission.mission_id:
            candidate = (
                self.config.root_dir
                / "artifacts"
                / "router"
                / f"decision_{mission.mission_id}.json"
            )
            if candidate.exists():
                try:
                    trace_path = os.path.relpath(candidate, self.config.root_dir)
                except Exception:
                    trace_path = str(candidate)
        if not trace_path:
            return [], None, None
        try:
            path = Path(trace_path)
            if not path.is_absolute():
                path = self.config.root_dir / trace_path
            if not path.exists():
                return [], trace_path, None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return [], trace_path, None
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            top_causes = (
                summary.get("top_causes") if isinstance(summary.get("top_causes"), list) else []
            )
            all_failed_reason = summary.get("all_failed_reason")
            return (
                [str(x) for x in top_causes if x],
                trace_path,
                str(all_failed_reason) if all_failed_reason else None,
            )
        except Exception:
            return [], trace_path, None

    def _enrich_detail_with_router_summary(
        self, mission: Optional[MissionState], detail: Dict[str, Any]
    ) -> None:
        if mission is None or not isinstance(detail, dict):
            return
        top_causes, trace_path, all_failed_reason = self._load_router_trace_summary(mission)
        if top_causes and not detail.get("router_top_causes"):
            detail["router_top_causes"] = top_causes
        if trace_path and not detail.get("router_trace_path"):
            detail["router_trace_path"] = trace_path
        if all_failed_reason and not detail.get("router_all_failed_reason"):
            detail["router_all_failed_reason"] = all_failed_reason

    def _open_provider_health_incident(
        self, mission: MissionState, *, question: str, reason: Optional[str]
    ) -> str:
        if IncidentReporter is None:
            raise RuntimeError("IncidentReporter no disponible")
        try:
            from agency.provider_health import run_provider_doctor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("provider_health no disponible") from exc
        reporter = IncidentReporter(self.config.root_dir)
        doctor_path, payload = run_provider_doctor(self.config.root_dir, roles=["brain"])
        attachments = [str(doctor_path)]
        env_snapshot = (
            payload.get("doctor_env", {}).get("snapshot_path")
            if isinstance(payload.get("doctor_env"), dict)
            else None
        )
        if env_snapshot:
            attachments.append(str(env_snapshot))
        context = {
            "mission_id": mission.mission_id,
            "intention": mission.intention,
            "question": question,
            "reason": reason,
            "provider_failures": payload.get("providers"),
        }
        remediation = [
            "Revisar el snapshot de doctor y seguir el runbook de salud de providers.",
            "Asignar responsables para cada provider degradado/bloqueado.",
        ]
        summary = f"Incidente de salud de providers para misión {mission.mission_id}"
        dedupe_key = self._hash_dict(
            {
                "kind": "provider_health",
                "mission_id": mission.mission_id,
                "reason": reason,
                "question": question,
            }
        )
        incident_id = reporter.open_incident(
            kind="provider_health",
            summary=summary,
            context=context,
            remediation=remediation,
            attachments=attachments,
            dedupe_key=dedupe_key,
        )
        try:
            print(f"🆘 INCIDENT creado: {incident_id} (adjuntos={len(attachments)})")
        except Exception:
            pass
        return incident_id

    def _open_council_contract_invalid_incident(
        self,
        mission: MissionState,
        *,
        reason: str,
        council_verdict: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if IncidentReporter is None:
            return None
        reporter = IncidentReporter(self.config.root_dir)
        attachments: List[str] = []
        try:
            last_session = self.config.root_dir / "artifacts" / "council_last_session.md"
            if last_session.exists():
                attachments.append(str(last_session))
        except Exception:
            pass
        context = {
            "mission_id": mission.mission_id,
            "intention": mission.intention,
            "reason": reason,
            "council_verdict": council_verdict or None,
        }
        remediation = [
            "Revisar el output del Council (validación JSON/contrato).",
            "Actualizar prompts/modelos del Council o ajustar el roster Starting XI.",
        ]
        summary = f"Council inválido para misión {mission.mission_id}"
        incident_id = reporter.open_incident(
            kind="council_contract_invalid",
            summary=summary,
            context=context,
            remediation=remediation,
            attachments=attachments,
        )
        try:
            mission.notes["incident_id"] = incident_id
        except Exception:
            pass
        return str(incident_id)

    def _emit_incident_gap(self, mission: MissionState, detail: Dict[str, Any]) -> Optional[str]:
        try:
            gaps_dir = self.config.root_dir / "artifacts" / "capability_gaps"
            gaps_dir.mkdir(parents=True, exist_ok=True)
            path = gaps_dir / f"{int(time.time())}_incident_failure_{mission.mission_id}.json"
            payload = {
                "schema": "ajax.capability_gap.incident_failure.v1",
                "mission_id": mission.mission_id,
                "intention": mission.intention,
                "detail": detail,
                "ts": time.time(),
                "ts_utc": self._iso_utc(),
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _emit_library_selection_gap(
        self,
        mission: MissionState,
        *,
        reason: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            gaps_dir = self.config.root_dir / "artifacts" / "capability_gaps"
            gaps_dir.mkdir(parents=True, exist_ok=True)
            path = gaps_dir / f"{int(time.time())}_library_selection_{mission.mission_id}.json"
            payload = {
                "schema": "ajax.capability_gap.library_selection.v1",
                "mission_id": mission.mission_id,
                "intention": mission.intention,
                "reason": reason,
                "detail": detail or {},
                "ts": time.time(),
                "ts_utc": self._iso_utc(),
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _emit_library_failure_gap(
        self,
        mission: MissionState,
        *,
        final_result: AjaxExecutionResult,
        episode_receipt_path: Optional[str],
    ) -> Optional[str]:
        try:
            gaps_dir = self.config.root_dir / "artifacts" / "capability_gaps"
            gaps_dir.mkdir(parents=True, exist_ok=True)
            path = gaps_dir / f"{int(time.time())}_library_failure_{mission.mission_id}.json"
            evidence_refs: List[str] = []
            try:
                last_exec = (
                    mission.notes.get("last_exec_receipt")
                    if isinstance(mission.notes, dict)
                    else None
                )
            except Exception:
                last_exec = None
            if isinstance(last_exec, str) and last_exec:
                evidence_refs.append(last_exec)
            if episode_receipt_path:
                evidence_refs.append(episode_receipt_path)
            try:
                log_path = (
                    mission.envelope.metadata.get("last_mission_log")
                    if mission.envelope and isinstance(mission.envelope.metadata, dict)
                    else None
                )
                if isinstance(log_path, str) and log_path:
                    evidence_refs.append(log_path)
            except Exception:
                pass
            payload = {
                "schema": "ajax.capability_gap.library_failure.v1",
                "mission_id": mission.mission_id,
                "intention": mission.intention,
                "skill_id": (
                    mission.notes.get("library_skill_id")
                    if isinstance(mission.notes, dict)
                    else None
                ),
                "error": final_result.error,
                "path": final_result.path,
                "detail": final_result.detail if isinstance(final_result.detail, dict) else None,
                "evidence_refs": evidence_refs,
                "ts": time.time(),
                "ts_utc": self._iso_utc(),
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(path)
        except Exception:
            return None

    def _brain_expert_providers(self) -> List[tuple[str, Dict[str, Any]]]:
        """
        Selecciona proveedores Brain de mayor tier para escalados (smart > balanced > fast).
        """
        cfg = (
            self.provider_configs.get("providers", {})
            if isinstance(self.provider_configs, dict)
            else {}
        )
        if not cfg:
            return []
        tier_score = {"smart": 3, "balanced": 2, "fast": 1}
        candidates: List[tuple[str, Dict[str, Any], int]] = []
        for name, pcfg in cfg.items():
            if not isinstance(pcfg, dict):
                continue
            roles = pcfg.get("roles") or []
            if not (isinstance(roles, list) and any(r.lower() == "brain" for r in roles)):
                continue
            tier = str(pcfg.get("tier") or "").lower()
            score = tier_score.get(tier, 0)
            candidates.append((name, pcfg, score))
        if not candidates:
            return []
        max_score = max(sc for _, _, sc in candidates)
        top = [(n, c) for n, c, sc in candidates if sc == max_score and max_score > 0]
        if not top:
            top = [(n, c) for n, c, _ in candidates]
        return top

    def _escalate_ask_user(
        self, mission: MissionState, question: str
    ) -> tuple[Optional[AjaxPlan], Optional[AskUserRequest]]:
        alert_level = str(getattr(self, "alert_level", None) or "normal").lower()
        reason = question or mission.permission_question or mission.feedback or "ask_user_requested"
        escalation_attempts: List[Dict[str, Any]] = []
        plan_obj: Optional[AjaxPlan] = None
        expert_providers = self._brain_expert_providers()
        router_cls = self._get_brain_router_cls()
        router = (
            router_cls(expert_providers, logger=self.log)
            if (router_cls and expert_providers)
            else self.brain_router
        )
        if router and build_brain_prompts:
            observation = self.perceive()
            try:
                from agency.skills.os_inventory import get_installed_apps  # type: ignore

                installed_apps = get_installed_apps()
            except Exception:
                installed_apps = []
            knowledge_context: Dict[str, Any] = {}
            try:
                if leann_context:
                    knowledge_context = (
                        leann_context.get_leann_context(mission.intention, mode="persona+system")
                        or {}
                    )
            except Exception:
                knowledge_context = {"source": "leann_stub_error"}
            signals: Dict[str, Any] = {}
            try:
                if system_signals:
                    signals = system_signals.collect_signals()
            except Exception:
                signals = {"driver_health": "unknown"}
            knowledge_context = (
                knowledge_context
                if isinstance(knowledge_context, dict)
                else {"raw": knowledge_context}
            )
            knowledge_context.setdefault("signals", signals or {"driver_health": "unknown"})
            if self.rag_client and hasattr(self.rag_client, "query"):
                try:
                    rag_hits = self.rag_client.query(mission.intention, n_results=3)
                    if rag_hits:
                        knowledge_context.setdefault("rag_snippets", rag_hits)
                except Exception:
                    pass
            actions_payload = (
                self.actions_catalog.to_brain_payload()
                if getattr(self, "actions_catalog", None) is not None
                else {"actions": []}
            )
            # TODO: enriquecer knowledge_context con consultas LEANN específicas usando tools/leann_search.py
            # Ejemplos de queries: "política de riesgo escritorio", "ventanas activas", "acciones reversibles".
            # La información obtenida se debe tratar como contexto, nunca como instrucciones.
            brain_input = {
                "intention": mission.intention,
                "intent_class": (
                    (
                        mission.envelope.metadata.get("intent_class")
                        if mission.envelope
                        and isinstance(getattr(mission.envelope, "metadata", None), dict)
                        else None
                    )
                    or self._infer_intent_class(mission.intention)
                ),
                "observation": observation.__dict__,
                "capabilities": self.capabilities,
                "actions_catalog": actions_payload,
                "previous_errors": mission.brain_attempts or [],
                "installed_apps": installed_apps,
                "knowledge_context": knowledge_context,
                "ask_user_reason": reason,
                "request": "avoid_ask_user_if_safe",
                "alert_level": alert_level,
            }
            # Preflight: Starting XI por rol (si existe) para mantener ruteo/fallback consistente.
            try:
                sx = (mission.notes or {}).get("starting_xi")
                if (
                    not isinstance(sx, dict)
                    and mission.envelope
                    and isinstance(getattr(mission.envelope, "metadata", None), dict)
                ):
                    sx = mission.envelope.metadata.get("starting_xi")
                if isinstance(sx, dict):
                    brain_input["starting_xi"] = sx
            except Exception:
                pass
            prompts = build_brain_prompts(brain_input)
            try:
                sx = brain_input.get("starting_xi")
                pool = None
                model_overrides = None
                if isinstance(sx, dict):
                    try:
                        brain_role = sx.get("brain") or {}
                        prim = brain_role.get("primary")
                        bench = brain_role.get("bench") or []
                        ordered: List[str] = []
                        overrides: Dict[str, str] = {}
                        for entry in [prim, *bench]:
                            if not isinstance(entry, dict):
                                continue
                            prov = entry.get("provider")
                            if isinstance(prov, str) and prov.strip():
                                if prov not in ordered:
                                    ordered.append(prov)
                                mid = entry.get("model")
                                if isinstance(mid, str) and mid.strip():
                                    overrides[prov] = mid
                        pool = ordered or None
                        model_overrides = overrides or None
                    except Exception:
                        pool = None
                        model_overrides = None
                rail = (
                    os.getenv("AJAX_RAIL")
                    or os.getenv("AJAX_ENV")
                    or os.getenv("AJAX_MODE")
                    or "lab"
                )
                risk_level = "medium"
                try:
                    gov = mission.envelope.governance if mission.envelope else None  # type: ignore[union-attr]
                    rl = (
                        str(getattr(gov, "risk_level", "") or "").strip().lower()
                        if gov is not None
                        else ""
                    )
                    if rl in {"low", "medium", "high"}:
                        risk_level = rl
                except Exception:
                    pass
                with self._override_cost_mode(self._effective_cost_mode(mission)):
                    plan_json = router.plan_with_failover(
                        prompt_system=prompts["system"],
                        prompt_user=prompts["user"],
                        meta={
                            "intention": mission.intention,
                            "ask_user_reason": reason,
                            "rail": rail,
                            "risk_level": risk_level,
                            "model_overrides": model_overrides,
                            "intent_class": brain_input.get("intent_class"),
                        },
                        exclude=None,
                        pool=pool,
                        caller=self._call_brain_provider,
                        attempt_collector=escalation_attempts,
                    )
                plan_json, repair_meta = self._validate_brain_plan_with_efe_repair(
                    plan_json, intention=mission.intention, source="brain_escalation"
                )
                if plan_json is None:
                    return self._build_missing_efe_plan(
                        intention=mission.intention,
                        source="brain_escalation",
                        receipt_path=(repair_meta or {}).get("receipt"),
                        reason=(repair_meta or {}).get("reason"),
                    )
                if repair_meta and isinstance(plan_json, dict):
                    meta = plan_json.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {}
                    if repair_meta.get("receipt"):
                        meta.setdefault("efe_repair_receipt", repair_meta.get("receipt"))
                    if repair_meta.get("reason"):
                        meta.setdefault("efe_repair_reason", repair_meta.get("reason"))
                    plan_json["metadata"] = meta
                self._enforce_brain_plan_order(plan_json)
                plan_id = plan_json.get("plan_id") or f"plan-escalated-{int(time.time())}"
                steps = plan_json.get("steps", [])
                raw_contract = plan_json.get("success_contract")
                plan_obj = AjaxPlan(
                    id=plan_id,
                    summary=f"Plan escalado para {mission.intention}",
                    steps=steps,
                    plan_id=plan_id,
                    metadata={
                        "intention": mission.intention,
                        "source": "brain_escalation",
                        "ask_user_reason": reason,
                        "alert_level": alert_level,
                        "escalation_attempts": escalation_attempts,
                    },
                    success_spec=plan_json.get("success_spec")
                    or raw_contract
                    or {"type": "check_last_step_status"},
                )
                if self.council and CouncilVerdict:
                    try:
                        catalog_payload = (
                            self.actions_catalog.to_brain_payload()
                            if getattr(self, "actions_catalog", None)
                            else {}
                        )
                        self._record_council_invocation(
                            mission, invoked=True, reason="plan_review_escalation"
                        )
                        with self._override_cost_mode(self._effective_cost_mode(mission)):
                            verdict: CouncilVerdict = self.council.review_plan(
                                mission.intention,
                                plan_json,
                                context=brain_input,
                                actions_catalog=catalog_payload,
                            )
                        verdict = self._normalize_council_verdict(verdict)
                        if verdict and not verdict.approved:
                            plan_obj = None
                            escalation_attempts.append(
                                {"id": "council", "result": f"veto:{verdict.feedback}"}
                            )
                        elif verdict:
                            try:
                                plan_obj.metadata["council_verdict"] = verdict.__dict__
                            except Exception:
                                pass
                    except Exception as exc:
                        escalation_attempts.append({"id": "council", "result": f"error:{exc}"})
            except Exception as exc:
                escalation_attempts.append({"id": "escalation_router", "result": str(exc)})

        try:
            if escalation_attempts:
                mission.brain_attempts.extend(escalation_attempts)
        except Exception:
            pass

        if plan_obj:
            return plan_obj, None

        timeout_sec = int(self.security_policy.get("ask_user_timeout_seconds", 60))
        on_timeout = str(self.security_policy.get("ask_user_on_timeout", "abort"))
        ask_req = AskUserRequest(
            question=question or reason or "Necesito confirmación antes de continuar.",
            reason=reason,
            timeout_seconds=timeout_sec,
            on_timeout=on_timeout,
            alert_level=alert_level,
            escalation_trace=escalation_attempts or None,
        )
        return None, ask_req

    def _record_brain_attempt(
        self, provider_name: str, provider_cfg: Dict[str, Any], result: str, started: float
    ) -> None:
        try:
            ok = str(result).strip().lower() == "ok"
            error_code = None if ok else self._infer_brain_error_code(result)
            self._last_brain_attempts.append(
                {
                    "role": "brain",
                    "id": provider_name,
                    "provider": provider_name,
                    "model": provider_cfg.get("_selected_model")
                    or provider_cfg.get("default_model")
                    or provider_cfg.get("model"),
                    "tier": provider_cfg.get("tier"),
                    "result": result,
                    "ok": ok,
                    "error_code": error_code,
                    "error_detail": None if ok else result,
                    "raw_plan_excerpt": getattr(self, "_last_raw_plan_excerpt", None),
                    "raw_plan_provider": getattr(self, "_last_raw_plan_provider", None),
                    "latency_ms": int((time.time() - started) * 1000),
                    "ts": started,
                    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                }
            )
            if provider_ranker is not None:
                try:
                    provider_ranker.record_attempt(
                        self.config.root_dir / "artifacts" / "health" / "providers_status.json",
                        provider=provider_name,
                        ok=ok,
                        latency_ms=int((time.time() - started) * 1000),
                        total_ms=int((time.time() - started) * 1000),
                        outcome=str(result),
                        error=None if ok else str(result),
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _record_provider_ranker_attempts(self, attempts: List[Dict[str, Any]]) -> None:
        if provider_ranker is None:
            return
        status_path = self.config.root_dir / "artifacts" / "health" / "providers_status.json"
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            provider = str(attempt.get("provider") or attempt.get("id") or "").strip()
            if not provider:
                continue
            ok = attempt.get("ok")
            if ok is None:
                ok = str(attempt.get("result") or "").strip().lower() == "ok"
            latency_ms = attempt.get("latency_ms")
            total_ms = None
            t_start = attempt.get("t_start")
            t_end = attempt.get("t_end")
            if (
                isinstance(t_start, (int, float))
                and isinstance(t_end, (int, float))
                and t_end > t_start
            ):
                total_ms = int((float(t_end) - float(t_start)) * 1000)
            if total_ms is None and isinstance(latency_ms, (int, float)) and latency_ms > 0:
                total_ms = int(latency_ms)
            ttft_ms = None
            t_first = attempt.get("t_first_output")
            if (
                isinstance(t_start, (int, float))
                and isinstance(t_first, (int, float))
                and t_first > t_start
            ):
                ttft_ms = int((float(t_first) - float(t_start)) * 1000)
            try:
                provider_ranker.record_attempt(
                    status_path,
                    provider=provider,
                    ok=bool(ok),
                    latency_ms=int(latency_ms) if isinstance(latency_ms, (int, float)) else None,
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    outcome=str(attempt.get("result") or ""),
                    error=None
                    if ok
                    else str(attempt.get("error_detail") or attempt.get("result") or "")[:200],
                )
            except Exception:
                continue

    def _run_cli_with_timeouts(
        self,
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
            q.put((name, b""))  # Signal stream end

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
            # If process finished, we still want to drain the queue
            if rc is not None and out_q.empty():
                break

            try:
                name, chunk = out_q.get(timeout=0.05)
                if not chunk:
                    # Stream ended, nothing more to do for this record
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

    def _call_brain_provider(
        self,
        provider_name: str,
        provider_cfg: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kind = (provider_cfg.get("kind") or "").lower()
        selected_model = provider_cfg.get("_selected_model") or provider_cfg.get("default_model")
        # Override de modelo (p.ej. Starting XI): meta.model_overrides[provider]=model
        try:
            overrides = (meta or {}).get("model_overrides") or (meta or {}).get(
                "starting_xi_model_overrides"
            )
            if isinstance(overrides, dict):
                ov = overrides.get(provider_name)
                if isinstance(ov, str) and ov.strip():
                    selected_model = ov.strip()
        except Exception:
            pass
        rail = str(
            (meta or {}).get("rail")
            or os.getenv("AJAX_RAIL")
            or os.getenv("AJAX_ENV")
            or os.getenv("AJAX_MODE")
            or "lab"
        )
        risk_level = str((meta or {}).get("risk_level") or "medium").strip().lower()
        timeout_override: Optional[int] = None
        planning_timeout_override: Optional[int] = None
        intent_class = (meta or {}).get("intent_class")
        auth_source: Optional[str] = None
        provider_features: List[str] = []
        if isinstance(provider_cfg, dict):
            provider_features = provider_cfg.get("features", [])
            if isinstance(provider_features, str):
                provider_features = [provider_features]

        if (meta or {}).get("planning"):
            raw_planning = provider_cfg.get("planning_timeout_seconds")
            if raw_planning is None and "auto_retry_on_stall" in provider_features:
                raw_planning = os.getenv("AJAX_CODEX_PLANNING_TIMEOUT_SEC", "60")
            if raw_planning is not None:
                try:
                    planning_timeout_override = int(raw_planning)
                except Exception:
                    planning_timeout_override = None

        def _raise_provider_error(
            message: str,
            *,
            error_code: Optional[str] = None,
            error_detail: Optional[str] = None,
            stderr_excerpt: Optional[str] = None,
            timeout_s: Optional[int] = None,
            parse_error: bool = False,
            timings: Optional[Dict[str, Any]] = None,
        ) -> None:
            if BrainProviderError is not None:
                raise BrainProviderError(
                    message,
                    error_code=error_code,
                    error_detail=error_detail,
                    stderr_excerpt=stderr_excerpt,
                    timeout_s=timeout_s,
                    parse_error=parse_error,
                    timings=timings,
                )
            raise RuntimeError(message)

        def _normalize_plan_shape(plan_obj: Any) -> Any:
            if not isinstance(plan_obj, dict):
                return plan_obj
            if "steps" not in plan_obj and isinstance(plan_obj.get("plan"), list):
                plan_obj = dict(plan_obj)
                plan_obj["steps"] = plan_obj.get("plan")
                plan_obj.pop("plan", None)
            if (
                not plan_obj.get("plan_id")
                and isinstance(plan_obj.get("id"), str)
                and plan_obj.get("id")
            ):
                plan_obj = dict(plan_obj)
                plan_obj["plan_id"] = plan_obj.get("id")
            # Attach raw excerpt for receipts/diagnostics (ignored by schema validation).
            try:
                excerpt = getattr(self, "_last_raw_plan_excerpt", None)
                provider = getattr(self, "_last_raw_plan_provider", None)
                if isinstance(excerpt, str) and excerpt.strip():
                    plan_obj = dict(plan_obj)
                    plan_obj["_raw_plan_excerpt"] = excerpt[:500]
                if isinstance(provider, str) and provider.strip():
                    plan_obj = dict(plan_obj)
                    plan_obj["_raw_plan_provider"] = provider.strip()
            except Exception:
                pass
            return plan_obj

        def _retry_json_repair_if_possible(
            raw_output: Optional[str] = None,
        ) -> Optional[Dict[str, Any]]:
            if not (meta or {}).get("planning"):
                return None
            if (meta or {}).get("json_repair_attempted"):
                return None
            repair_meta = dict(meta or {})
            repair_meta["json_repair_attempted"] = True
            repair_meta["json_repair"] = True
            raw = str(raw_output or "").strip()
            if raw:
                raw = raw[:8000]
                repair_user = (
                    "Return ONLY valid JSON that matches the required schema. No prose.\n"
                    "Repair the INVALID_OUTPUT into valid JSON without changing intent.\n\n"
                    f"INVALID_OUTPUT:\n{raw}\n"
                )
            else:
                repair_user = "Return ONLY valid JSON that matches the required schema. No prose."
            try:
                return self._call_brain_provider(
                    provider_name,
                    provider_cfg,
                    system_prompt,
                    repair_user,
                    meta=repair_meta,
                )
            except Exception:
                return None

        try:
            from agency.auth_manager import AuthManager  # type: ignore
        except Exception:
            AuthManager = None  # type: ignore
        if AuthManager is not None:
            auth = AuthManager(root_dir=self.config.root_dir)
            astate = auth.auth_state(provider_name, provider_cfg)
            auth.persist_auth_state(provider_name, astate)
            auth.ensure_auth_gap(provider_name, astate)
            try:
                auth_source = auth.auth_source(provider_name, provider_cfg)
            except Exception:
                auth_source = None
            if astate.state in {"MISSING", "EXPIRED"} and AuthManager.is_web_auth_required(
                provider_cfg
            ):
                code = f"auth_{astate.state.lower()}"
                _raise_provider_error(
                    f"{code}:{astate.reason}",
                    error_code=code,
                    error_detail=astate.reason,
                    timeout_s=timeout_override,
                )
        try:
            from agency import provider_ranker  # type: ignore

            status = provider_ranker.load_status(
                self.config.root_dir / "artifacts" / "health" / "providers_status.json"
            )
            timeout_override = int(
                provider_ranker.recommended_timeout_seconds(
                    provider_name=provider_name,
                    provider_cfg=provider_cfg,
                    status=status,
                    role="brain",
                    rail=rail,
                    risk_level=risk_level,
                )
            )
        except Exception:
            timeout_override = None
        try:
            self.log.info(
                "AJAX.mission: Brain provider=%s model=%s tier=%s",
                provider_name,
                selected_model or "-",
                provider_cfg.get("tier", "unknown"),
            )
        except Exception:
            pass
        timeout_cfg = self._resolve_provider_timeouts(
            provider_name,
            provider_cfg,
            rail=rail,
            intent_class=str(intent_class) if intent_class else None,
            planning=bool((meta or {}).get("planning")),
        )
        timeout_cfg_override = (meta or {}).get("timeout_cfg_override")
        if isinstance(timeout_cfg_override, dict):
            for key in (
                "connect_timeout_ms",
                "first_output_timeout_ms",
                "stall_timeout_ms",
                "total_timeout_ms",
            ):
                if timeout_cfg_override.get(key) is not None:
                    timeout_cfg[key] = timeout_cfg_override.get(key)

        endpoint = None
        if kind == "http_openai":
            endpoint = (provider_cfg.get("base_url") or "").rstrip("/") or None
        elif kind == "cli":
            cmd_template = provider_cfg.get("command") or []
            first = str(cmd_template[0]) if isinstance(cmd_template, list) and cmd_template else ""
            endpoint = f"cli:{first}" if first else "cli"
        elif kind == "codex_cli_jsonl":
            endpoint = "codex_cli_jsonl"
        try:
            self.log.info(
                "provider_call_signature provider=%s model=%s endpoint=%s auth_source=%s ttft_timeout_ms=%s total_timeout_ms=%s",
                provider_name,
                selected_model or "-",
                endpoint or "-",
                auth_source or "-",
                timeout_cfg.get("first_output_timeout_ms"),
                timeout_cfg.get("total_timeout_ms"),
            )
        except Exception:
            pass

        if kind == "cli":
            prompt = system_prompt.rstrip() + "\n\n" + user_prompt
            prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            cmd_template = provider_cfg.get("command") or []
            cmd = []
            use_stdin = False
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
            timeout_s = int(timeout_override or provider_cfg.get("timeout_seconds") or 40)
            if planning_timeout_override is not None:
                timeout_s = int(planning_timeout_override)
            if timeout_cfg.get("total_timeout_ms") is None:
                timeout_cfg["total_timeout_ms"] = int(timeout_s * 1000)
            else:
                try:
                    timeout_s = int(float(timeout_cfg.get("total_timeout_ms")) / 1000.0)
                except Exception:
                    pass
            result = self._run_cli_with_timeouts(
                cmd,
                input_text=prompt if use_stdin else None,
                timeout_cfg=timeout_cfg,
            )
            stderr_tail = (
                (result.get("stderr") or "")[-4096:]
                if isinstance(result.get("stderr"), str)
                else None
            )
            timings = dict(result.get("timings") or {})
            timings.update(
                {
                    "timeout_kind": result.get("timeout_kind") or "NONE",
                    "client_abort": bool(result.get("client_abort")),
                    "exit_code": result.get("returncode"),
                    "stderr_tail": stderr_tail,
                    "bytes_rx": int(result.get("bytes_stdout") or 0)
                    + int(result.get("bytes_stderr") or 0),
                    "tokens_rx": None,
                }
            )
            if result.get("client_abort") and result.get("timeout_kind") in {
                "TTFT",
                "STALL",
                "TOTAL",
            }:
                _raise_provider_error(
                    f"cli_timeout:{result.get('timeout_kind')}",
                    error_code="client_timeout",
                    error_detail=f"timeout_kind:{result.get('timeout_kind')}",
                    stderr_excerpt=stderr_tail,
                    timeout_s=timeout_s,
                    timings=timings,
                )
            out = (result.get("stdout") or "").strip()
            rc = result.get("returncode")
            if rc != 0:
                err = (result.get("stderr") or "").strip() or out or f"cli_rc_{rc}"
                is_bridge = "provider_cli_bridge.py" in " ".join(str(x) for x in cmd)
                err_code = "infra_bridge_error" if is_bridge else f"cli_rc_{rc}"
                _raise_provider_error(
                    f"cli_rc_{rc}:{err[:200]}",
                    error_code=err_code,
                    error_detail=err[:200],
                    stderr_excerpt=stderr_tail,
                    timeout_s=timeout_s,
                    timings=timings,
                )
            if out.startswith("```"):
                out = out.strip("`")
                if out.lower().startswith("json"):
                    out = out[len("json") :].strip()
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            candidate = out
            for ln in reversed(lines):
                if ln.startswith("{") and ln.endswith("}"):
                    candidate = ln
                    break
            try:
                self._last_raw_plan_provider = str(provider_name)
                self._last_raw_plan_excerpt = str(candidate)[:500]
            except Exception:
                pass
            try:
                plan_out = _normalize_plan_shape(json.loads(candidate))
                if isinstance(plan_out, dict):
                    plan_out = dict(plan_out)
                    plan_out["_timings"] = timings
                return plan_out
            except Exception:
                try:
                    start = candidate.find("{")
                    end = candidate.rfind("}")
                    if start != -1 and end != -1:
                        plan_out = _normalize_plan_shape(json.loads(candidate[start : end + 1]))
                        if isinstance(plan_out, dict):
                            plan_out = dict(plan_out)
                            plan_out["_timings"] = timings
                        return plan_out
                except Exception:
                    pass
                repaired = _retry_json_repair_if_possible(candidate)
                if isinstance(repaired, dict):
                    repaired = dict(repaired)
                    repaired["_timings"] = timings
                    return repaired
                _raise_provider_error(
                    f"Brain devolvió JSON inválido desde CLI: {candidate[:200]!r}",
                    error_code="parse_error",
                    error_detail=candidate[:200],
                    parse_error=True,
                    timeout_s=timeout_s,
                    stderr_excerpt=stderr_tail,
                    timings=timings,
                )
        # end cli branch

        if kind == "codex_cli_jsonl":
            cfg = dict(provider_cfg or {})
            if selected_model:
                cfg["default_model"] = selected_model
            if timeout_override is not None:
                cfg["timeout_seconds"] = int(timeout_override)
            timeout_s = int(cfg.get("timeout_seconds") or 60)
            env_override: Dict[str, str] = {}
            if (meta or {}).get("planning"):
                if planning_timeout_override is not None:
                    timeout_s = int(planning_timeout_override)
                env_override["CODEX_REASONING_EFFORT"] = os.getenv(
                    "AJAX_CODEX_PLANNING_EFFORT", "medium"
                )
            if timeout_cfg.get("total_timeout_ms") is None:
                timeout_cfg["total_timeout_ms"] = int(timeout_s * 1000)
            else:
                try:
                    timeout_s = int(float(timeout_cfg.get("total_timeout_ms")) / 1000.0)
                except Exception:
                    pass
            cfg["timeout_seconds"] = timeout_s
            try:
                timeout_cfg.setdefault("total_timeout_ms", int(timeout_s * 1000))
                result = self._call_codex_cli(
                    cfg,
                    system_prompt,
                    user_prompt,
                    parse_json=True,
                    env_override=env_override or None,
                    timeout_cfg=timeout_cfg,
                    return_timings=True,
                )
                if isinstance(result, tuple):
                    plan_obj, timings = result
                    if isinstance(plan_obj, dict):
                        plan_obj = dict(plan_obj)
                        plan_obj["_timings"] = timings
                    return plan_obj
                return result
            except Exception as exc:
                if BrainProviderError is not None and isinstance(exc, BrainProviderError):
                    if exc.parse_error:
                        repaired = _retry_json_repair_if_possible()
                        if isinstance(repaired, dict):
                            return repaired
                    raise
                _raise_provider_error(
                    f"codex_error:{exc}",
                    error_code="codex_error",
                    error_detail=str(exc),
                    timeout_s=timeout_s,
                )

        base_url = (provider_cfg.get("base_url") or "").rstrip("/")
        api_key_env = provider_cfg.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        if not base_url:
            _raise_provider_error(
                "Brain provider sin base_url",
                error_code="config_missing",
                error_detail="base_url_missing",
            )
        if api_key_env and not api_key:
            _raise_provider_error(
                f"API key no configurada para Brain ({api_key_env})",
                error_code="auth_missing",
                error_detail=f"missing_api_key:{api_key_env}",
            )
        model = (
            selected_model
            or provider_cfg.get("default_model")
            or provider_cfg.get("model")
            or "llama-3.1-8b-instant"
        )
        timeout = int(timeout_override or provider_cfg.get("timeout_seconds") or 20)
        if planning_timeout_override is not None:
            timeout = int(planning_timeout_override)
        max_tokens = 512
        if (meta or {}).get("planning"):
            try:
                max_tokens = int(
                    provider_cfg.get("planning_max_tokens")
                    or os.getenv("AJAX_PLANNING_MAX_TOKENS", "1400")
                    or 1400
                )
            except Exception:
                max_tokens = 1400

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if timeout_cfg.get("total_timeout_ms") is None:
            timeout_cfg["total_timeout_ms"] = int(timeout * 1000)
        else:
            try:
                timeout = int(float(timeout_cfg.get("total_timeout_ms")) / 1000.0)
            except Exception:
                pass
        t_start = time.monotonic()
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except Exception as exc:
            timings = {
                "t_start": t_start,
                "t_connect_ok": None,
                "t_first_output": None,
                "t_last_output": None,
                "t_end": time.monotonic(),
                "timeout_kind": "TOTAL" if "timeout" in str(exc).lower() else "NONE",
                "client_abort": True if "timeout" in str(exc).lower() else False,
                "exit_code": None,
                "stderr_tail": str(exc)[:200],
                "bytes_rx": None,
                "tokens_rx": None,
            }
            _raise_provider_error(
                f"brain_http_error:{exc}",
                error_code="client_timeout" if "timeout" in str(exc).lower() else "http_error",
                error_detail=str(exc)[:200],
                timeout_s=timeout,
                timings=timings,
            )
        t_end = time.monotonic()
        timings = {
            "t_start": t_start,
            "t_connect_ok": None,
            "t_first_output": None,
            "t_last_output": None,
            "t_end": t_end,
            "timeout_kind": "NONE",
            "client_abort": False,
            "exit_code": None,
            "stderr_tail": None,
            "bytes_rx": len(resp.content) if hasattr(resp, "content") else None,
            "tokens_rx": None,
        }
        if resp.status_code >= 400:
            _raise_provider_error(
                f"brain_http_{resp.status_code}: {resp.text}",
                error_code=f"http_{resp.status_code}",
                error_detail=resp.text[:200],
                timeout_s=timeout,
                stderr_excerpt=resp.text[:200],
                timings=timings,
            )
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        content = content.strip()
        try:
            self._last_raw_plan_provider = str(provider_name)
            self._last_raw_plan_excerpt = content[:500]
        except Exception:
            pass
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[len("json") :].strip()
        try:
            plan_out = _normalize_plan_shape(json.loads(content))
            if isinstance(plan_out, dict):
                plan_out = dict(plan_out)
                plan_out["_timings"] = timings
            return plan_out
        except json.JSONDecodeError as exc:
            repaired = _retry_json_repair_if_possible(content)
            if isinstance(repaired, dict):
                repaired = dict(repaired)
                repaired["_timings"] = timings
                return repaired
            _raise_provider_error(
                f"Brain devolvió JSON inválido: {exc}: {content[:200]!r}",
                error_code="parse_error",
                error_detail=content[:200],
                parse_error=True,
                timeout_s=timeout,
                timings=timings,
            )

    def _codex_timeout_retry_enabled(self) -> bool:
        raw = (os.getenv("AJAX_CODEX_TIMEOUT_RETRY") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _attempt_timeout_kind(self, attempt: Dict[str, Any]) -> Optional[str]:
        if not isinstance(attempt, dict):
            return None
        val = attempt.get("timeout_kind")
        if isinstance(val, str) and val.strip():
            return val.strip().upper()
        detail = str(attempt.get("error_detail") or attempt.get("result") or "").upper()
        if "TIMEOUT_KIND:TTFT" in detail:
            return "TTFT"
        if "TIMEOUT_KIND:STALL" in detail:
            return "STALL"
        if "TIMEOUT_KIND:TOTAL" in detail:
            return "TOTAL"
        if "CLI_TIMEOUT:TTFT" in detail:
            return "TTFT"
        if "CLI_TIMEOUT:STALL" in detail:
            return "STALL"
        if "CLI_TIMEOUT:TOTAL" in detail:
            return "TOTAL"
        return None

    def _should_retry_on_timeout(self, attempts: List[Dict[str, Any]]) -> bool:
        if not attempts:
            return False
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            provider_name = str(attempt.get("provider") or attempt.get("id") or "").strip()
            if not provider_name:
                continue
            
            # Solo reintentar si el provider tiene la feature 'auto_retry_on_stall'
            cfg = (self.provider_configs.get("providers", {}) or {}).get(provider_name, {})
            features = cfg.get("features", []) if isinstance(cfg, dict) else []
            if "auto_retry_on_stall" not in features:
                continue
                
            if attempt.get("ok") is True:
                continue
            kind = self._attempt_timeout_kind(attempt)
            if kind in {"TTFT", "STALL"}:
                return True
        return False

    def _provider_timeout_retry_cfg(self, *, provider_name: str, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
        bench = self._provider_timeout_bench_p95(provider_name)
        total_base = base_cfg.get("total_timeout_ms")
        ttft_base = base_cfg.get("first_output_timeout_ms")
        total_p95 = bench.get("total_p95_ms")
        ttft_p95 = bench.get("ttft_p95_ms")
        total_target = max(
            int((float(total_p95) * 1.5)) if isinstance(total_p95, (int, float)) else 0,
            int(total_base) if isinstance(total_base, (int, float)) else 0,
            120000,
        )
        ttft_target = max(
            int((float(ttft_p95) * 1.5)) if isinstance(ttft_p95, (int, float)) else 0,
            int(ttft_base) if isinstance(ttft_base, (int, float)) else 0,
            30000,
        )
        override = {
            "total_timeout_ms": total_target,
            "first_output_timeout_ms": ttft_target,
        }
        stall_base = base_cfg.get("stall_timeout_ms")
        if isinstance(stall_base, (int, float)):
            override["stall_timeout_ms"] = int(stall_base)
        return override

    def _call_brain_llm(
        self, brain_input: Dict[str, Any], exclude: Optional[set[str]] = None
    ) -> Dict[str, Any]:
        """
        Hook al LLM de planificación con failover (BrainRouter).
        """
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        # Cross-mission cooldown based on real-time ledger status.
        if self.ledger:
            try:
                ledger_doc = self.ledger.load_latest()
                now = time.time()
                for row in ledger_doc.get("rows", []):
                    if row.get("status") in {"soft_fail", "hard_fail"}:
                        cd = row.get("cooldown_until_ts")
                        if cd is None or now < float(cd):
                            exclude.add(row.get("provider"))
            except Exception as exc:
                self.log.warning("Failed to filter providers from ledger: %s", exc)
        if build_brain_prompts is None:
            raise RuntimeError("build_brain_prompts no disponible")
        prompts = build_brain_prompts(brain_input)
        system_prompt = prompts["system"]
        user_prompt = prompts["user"]
        pack_id = str(prompts.get("pack_id") or "P2_OPS_SAFE")
        context_budget = prompts.get("context_budget")
        escalation_reason = prompts.get("escalation_reason")
        decision = prompts.get("decision")
        decision_reason = prompts.get("reason")
        filters = prompts.get("filters")
        confidence = prompts.get("confidence")
        pack_state = prompts.get("pack_state")
        mission_id = None
        try:
            env = brain_input.get("mission_envelope") or {}
            if isinstance(env, dict):
                mission_id = env.get("mission_id")
        except Exception:
            mission_id = None
        self._record_prompt_pack_receipt(
            mission_id=str(mission_id) if mission_id else None,
            intention=str(brain_input.get("intention") or ""),
            pack_id=pack_id,
            context_budget=int(context_budget)
            if isinstance(context_budget, (int, float))
            else None,
            escalation_reason=str(escalation_reason) if escalation_reason else None,
            decision=str(decision) if decision else None,
            reason=str(decision_reason) if decision_reason else None,
            filters=filters if isinstance(filters, dict) else None,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        )
        try:
            self._last_prompt_pack_decision = {
                "pack_id": pack_id,
                "context_budget": context_budget,
                "escalation_reason": escalation_reason,
                "decision": decision,
                "reason": decision_reason,
                "filters": filters,
                "confidence": confidence,
                "pack_state": pack_state,
            }
        except Exception:
            self._last_prompt_pack_decision = None
        self._last_brain_attempts = []
        rail = os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"
        risk_level = "medium"
        try:
            env = (brain_input.get("mission_envelope") or {}).get("governance") or {}
            rl = str(env.get("risk_level") or "").strip().lower()
            if rl in {"low", "medium", "high"}:
                risk_level = rl
        except Exception:
            pass

        if self.brain_router:
            try:
                sx = brain_input.get("starting_xi")
                pool = None
                model_overrides = None
                if isinstance(sx, dict):
                    try:
                        brain_role = sx.get("brain") or {}
                        prim = brain_role.get("primary")
                        bench = brain_role.get("bench") or []
                        ordered: List[str] = []
                        overrides: Dict[str, str] = {}
                        for entry in [prim, *bench]:
                            if not isinstance(entry, dict):
                                continue
                            prov = entry.get("provider")
                            if isinstance(prov, str) and prov.strip():
                                if prov not in ordered:
                                    ordered.append(prov)
                                mid = entry.get("model")
                                if isinstance(mid, str) and mid.strip():
                                    overrides[prov] = mid
                        pool = ordered or None
                        model_overrides = overrides or None
                    except Exception:
                        pool = None
                        model_overrides = None
                if pool is None:
                    providers_cfg = (
                        self.provider_configs.get("providers", {})
                        if isinstance(self.provider_configs, dict)
                        else {}
                    )
                    status_doc = {}
                    scoreboard_doc: Dict[str, Any] = {}
                    try:
                        if provider_ranker is not None:
                            status_doc = provider_ranker.load_status(
                                self.config.root_dir
                                / "artifacts"
                                / "health"
                                / "providers_status.json"
                            )
                    except Exception:
                        status_doc = {}
                    try:
                        from agency import provider_scoreboard  # type: ignore

                        scoreboard_doc = provider_scoreboard.load_scoreboard(
                            self.config.root_dir
                            / "artifacts"
                            / "state"
                            / "provider_scoreboard.json"
                        )
                    except Exception:
                        scoreboard_doc = {}
                    providers_status = (
                        (status_doc or {}).get("providers") if isinstance(status_doc, dict) else {}
                    )
                    if not isinstance(providers_status, dict):
                        providers_status = {}
                    if os.getenv("AJAX_ALLOW_DEGRADED_PLANNING") is not None:
                        allow_degraded = (
                            os.getenv("AJAX_ALLOW_DEGRADED_PLANNING") or ""
                        ).strip().lower() in {"1", "true", "yes", "on"}
                    else:
                        allow_degraded = False
                        try:
                            if failure_allow_degraded_planning is not None:
                                allow_degraded = bool(
                                    failure_allow_degraded_planning(
                                        self._provider_failure_policy(), default=False
                                    )
                                )
                        except Exception:
                            allow_degraded = False

                    base_pref = ["codex_brain", "groq", "qwen_cloud", "gemini_cli"]
                    pref_rank = {name: idx for idx, name in enumerate(base_pref)}
                    scored: List[tuple[tuple, str]] = []
                    try:
                        raw_p95 = os.getenv("AJAX_CODEX_ESCALATE_P95_MS", "20000")
                        codex_p95_escalate_ms = int(raw_p95)
                    except Exception:
                        codex_p95_escalate_ms = 20000
                    for name, cfg in providers_cfg.items():
                        if not isinstance(cfg, dict):
                            continue
                        if cfg.get("disabled"):
                            continue
                        roles = cfg.get("roles") or []
                        if not isinstance(roles, list) or "brain" not in [r.lower() for r in roles]:
                            continue
                        if exclude and name in exclude:
                            continue
                        st = (
                            providers_status.get(name)
                            if isinstance(providers_status, dict)
                            else None
                        )
                        availability = (
                            str((st or {}).get("availability") or "").strip().lower()
                            if isinstance(st, dict)
                            else ""
                        )
                        if availability == "degraded" and not allow_degraded:
                            continue
                        availability_rank = {"up": 0, "ok": 0, "degraded": 1, "down": 2}.get(
                            availability, 1
                        )
                        p95 = st.get("latency_p95_ms") if isinstance(st, dict) else None
                        p95_val = float(p95) if isinstance(p95, (int, float)) else 99999.0
                        
                        # Si el provider tiene auto_retry_on_stall (ej: Codex) y está lento, 
                        # saltarlo para favorecer algo más rápido (Groq/Gemini) si no estamos ya escalando.
                        p_features = cfg.get("features", []) if isinstance(cfg, dict) else []
                        if (
                            "auto_retry_on_stall" in p_features
                            and p95_val >= float(codex_p95_escalate_ms)
                            and not escalation_reason
                        ):
                            continue
                        timeout_rate = (
                            st.get("timeout_rate_recent") if isinstance(st, dict) else None
                        )
                        timeout_val = (
                            float(timeout_rate) if isinstance(timeout_rate, (int, float)) else 1.0
                        )
                        pref = pref_rank.get(name, len(pref_rank) + 10)
                        scoreboard_rank = 0.0
                        if isinstance(scoreboard_doc, dict) and scoreboard_doc:
                            try:
                                from agency import provider_scoreboard  # type: ignore

                                model_id = (
                                    cfg.get("_selected_model")
                                    or cfg.get("default_model")
                                    or cfg.get("model")
                                )
                                min_samples = int(
                                    os.getenv("AJAX_SCOREBOARD_MIN_SAMPLES", "3") or 3
                                )
                                cooldown_minutes = int(
                                    os.getenv("AJAX_SCOREBOARD_COOLDOWN_MIN", "15") or 15
                                )
                                state = provider_scoreboard.promotion_state(
                                    scoreboard_doc,
                                    provider=name,
                                    model=model_id,
                                    min_samples=min_samples,
                                    cooldown_minutes=cooldown_minutes,
                                )
                                if state.get("eligible") is False:
                                    scoreboard_rank = 1.0
                                elif state.get("reorder_allowed"):
                                    sb_score = provider_scoreboard.score_for(
                                        scoreboard_doc, provider=name, model=model_id
                                    )
                                    if isinstance(sb_score, (int, float)):
                                        scoreboard_rank = -float(sb_score)
                            except Exception:
                                scoreboard_rank = 0.0
                        score = (availability_rank, timeout_val, p95_val, scoreboard_rank, pref)
                        scored.append((score, name))
                    scored.sort(key=lambda item: item[0])
                    ranked_pool = [name for _, name in scored]
                    pool = ranked_pool[:2] if ranked_pool else None
                plan_attempts = 2
                try:
                    if failure_planning_max_attempts is not None:
                        plan_attempts = int(
                            failure_planning_max_attempts(
                                self._provider_failure_policy(), default=2
                            )
                        )
                except Exception:
                    plan_attempts = 2
                plan_attempts = max(1, int(plan_attempts or 1))
                base_meta = {
                    "intention": brain_input.get("intention"),
                    "rail": rail,
                    "risk_level": risk_level,
                    "model_overrides": model_overrides,
                    "planning": True,
                    "intent_class": brain_input.get("intent_class"),
                }
                plan = self.brain_router.plan_with_failover(
                    prompt_system=system_prompt,
                    prompt_user=user_prompt,
                    meta=base_meta,
                    exclude=exclude,
                    pool=pool,
                    max_attempts=plan_attempts,
                    caller=self._call_brain_provider,
                    attempt_collector=self._last_brain_attempts,
                )

                # Feedback real-time para el usuario
                if self._last_brain_attempts:
                    for att in self._last_brain_attempts:
                        if not att.get("ok") and att.get("transient"):
                            prov = att.get("provider", "unknown")
                            err = att.get("error_code", "error")
                            print(
                                f"⚠️  [WATCHDOG] Proveedor '{prov}' degradado ({err}). Aplicando cooldown..."
                            )

                return plan
            except Exception as exc:
                if self._codex_timeout_retry_enabled() and not getattr(
                    self, "_codex_timeout_retry_guard", False
                ):
                    attempts = getattr(self, "_last_brain_attempts", []) or []
                    if self._should_retry_on_timeout(attempts):
                        self._codex_timeout_retry_guard = True
                        try:
                            # Identificar cuál fue el provider que falló por timeout y tiene la feature
                            target_provider = None
                            for a in reversed(attempts):
                                if not a.get("ok") and self._attempt_timeout_kind(a) in {"TTFT", "STALL"}:
                                    p_name = str(a.get("provider") or "").strip()
                                    p_cfg = (self.provider_configs.get("providers", {}) or {}).get(p_name, {})
                                    if "auto_retry_on_stall" in (p_cfg.get("features", []) if isinstance(p_cfg, dict) else []):
                                        target_provider = p_name
                                        break
                            
                            if target_provider:
                                provider_cfg = (
                                    self.provider_configs.get("providers", {}).get(target_provider, {})
                                    if isinstance(self.provider_configs, dict)
                                    else {}
                                )
                                timeout_base = self._resolve_provider_timeouts(
                                    target_provider,
                                    provider_cfg if isinstance(provider_cfg, dict) else {},
                                    rail=rail,
                                    intent_class=str(brain_input.get("intent_class") or ""),
                                    planning=True,
                                )
                                timeout_override = self._provider_timeout_retry_cfg(
                                    provider_name=target_provider, 
                                    base_cfg=timeout_base
                                )
                                retry_meta = dict(base_meta)
                                retry_meta["timeout_cfg_override"] = timeout_override
                                retry_meta["retry_reason"] = f"{target_provider}_timeout_escalate"
                                retry_plan = self.brain_router.plan_with_failover(
                                    prompt_system=system_prompt,
                                    prompt_user=user_prompt,
                                    meta=retry_meta,
                                    exclude=exclude,
                                    pool=[target_provider],
                                    max_attempts=1,
                                    caller=self._call_brain_provider,
                                    attempt_collector=self._last_brain_attempts,
                                )
                                return retry_plan
                        except Exception:
                            pass
                        try:
                            tried = {
                                str(a.get("provider") or a.get("id") or "").strip()
                                for a in attempts
                                if isinstance(a, dict)
                            }
                            if "gemini_cli" not in tried:
                                fallback_meta = dict(base_meta)
                                fallback_meta["retry_reason"] = "timeout_fallback_gemini"
                                return self.brain_router.plan_with_failover(
                                    prompt_system=system_prompt,
                                    prompt_user=user_prompt,
                                    meta=fallback_meta,
                                    exclude=exclude,
                                    pool=["gemini_cli"],
                                    max_attempts=1,
                                    caller=self._call_brain_provider,
                                    attempt_collector=self._last_brain_attempts,
                                )
                        except Exception:
                            pass
                        finally:
                            try:
                                self._codex_timeout_retry_guard = False
                            except Exception:
                                pass
                # BrainRouter ya anotó todos los fallos; propaga con prefijo
                try:
                    if self.ledger:
                        for a in getattr(self, "_last_brain_attempts", []) or []:
                            if not a.get("ok"):
                                p_failed = a.get("provider")
                                r_failed = a.get("reason") or a.get("error_code") or "http_error"
                                d_failed = a.get("error_detail") or str(exc)
                                if p_failed:
                                    self.ledger.record_failure(p_failed, "brain", r_failed, detail=d_failed)
                except Exception:
                    pass
                raise RuntimeError(f"brain_failover_exhausted:{exc}") from exc
            finally:
                try:
                    self._record_provider_ranker_attempts(
                        getattr(self, "_last_brain_attempts", []) or []
                    )
                except Exception:
                    pass

        provider_name, provider_cfg = self._select_brain_provider(exclude=exclude)
        started = time.time()
        attempt_result = "error"
        try:
            plan = self._call_brain_provider(
                provider_name,
                provider_cfg,
                system_prompt,
                user_prompt,
                meta={
                    "intention": brain_input.get("intention"),
                    "rail": rail,
                    "risk_level": risk_level,
                    "planning": True,
                    "intent_class": brain_input.get("intent_class"),
                },
            )
            attempt_result = "ok"
            return plan
        finally:
            self._record_brain_attempt(provider_name, provider_cfg, attempt_result, started)

    def _chat_llm(self, system_prompt: str, user_prompt: str) -> str:
        providers_cfg = (
            self.provider_configs.get("providers", {})
            if isinstance(self.provider_configs, dict)
            else {}
        )
        errors: List[str] = []
        fallback_chain: List[Dict[str, Any]] = []
        cost_mode = os.getenv("AJAX_COST_MODE", "premium")
        root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
        auth_mgr = None
        try:
            if AuthManager is not None:
                auth_mgr = AuthManager(root_dir=Path(root_dir))
        except Exception:
            auth_mgr = None

        status_path = Path(root_dir) / "artifacts" / "health" / "providers_status.json"
        ledger_path = Path(root_dir) / "artifacts" / "provider_ledger" / "latest.json"

        def _load_json_doc(path: Path) -> Dict[str, Any]:
            try:
                if not path.exists():
                    return {}
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        status_doc = _load_json_doc(status_path)
        ledger_doc = _load_json_doc(ledger_path)
        status_updated_utc = (
            str(status_doc.get("updated_utc") or "") if isinstance(status_doc, dict) else ""
        )
        ledger_updated_utc = (
            str(ledger_doc.get("updated_utc") or "") if isinstance(ledger_doc, dict) else ""
        )
        providers_status = status_doc.get("providers") if isinstance(status_doc, dict) else {}
        providers_status = providers_status if isinstance(providers_status, dict) else {}
        ledger_rows = ledger_doc.get("rows") if isinstance(ledger_doc, dict) else []
        ledger_rows = ledger_rows if isinstance(ledger_rows, list) else []
        ledger_by_provider: Dict[str, Dict[str, Any]] = {}
        for row in ledger_rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("role") or "").strip().lower() != "brain":
                continue
            prov = str(row.get("provider") or "").strip()
            if prov and prov not in ledger_by_provider:
                ledger_by_provider[prov] = row

        def _provider_is_down(provider: str) -> bool:
            entry = providers_status.get(provider)
            if not isinstance(entry, dict):
                return False
            transport = entry.get("transport") if isinstance(entry.get("transport"), dict) else {}
            if str(transport.get("status") or "").strip().upper() == "DOWN":
                return True
            breathing = entry.get("breathing") if isinstance(entry.get("breathing"), dict) else {}
            if str(breathing.get("status") or "").strip().upper() == "DOWN":
                return True
            roles = breathing.get("roles") if isinstance(breathing.get("roles"), dict) else {}
            probe = roles.get("brain") if isinstance(roles.get("brain"), dict) else {}
            if str(probe.get("status") or "").strip().upper() == "DOWN":
                return True
            return False

        def _provider_latency_p95_ms(provider: str) -> Optional[int]:
            entry = providers_status.get(provider)
            if not isinstance(entry, dict):
                return None
            p95 = entry.get("latency_p95_ms")
            try:
                return int(p95) if isinstance(p95, (int, float)) else None
            except Exception:
                return None

        def _tier_allows(tier: str, cm: str) -> bool:
            tier_n = (tier or "balanced").strip().lower()
            cm_n = (cm or "premium").strip().lower()
            if cm_n == "emergency":
                return tier_n in {"cheap", "balanced", "premium"}
            if cm_n == "balanced":
                return tier_n in {"cheap", "balanced"}
            return tier_n in {"cheap", "balanced", "premium"}

        def _build_chat_ladder() -> tuple[List[Tuple[str, Dict[str, Any]]], List[Dict[str, Any]]]:
            allow_local_override = (os.getenv("AJAX_ALLOW_LOCAL_TEXT") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            } or str(cost_mode or "").strip().lower() == "emergency"
            chat_policy = (
                (os.getenv("AJAX_CHAT_PROVIDER_POLICY") or "premium_first").strip().lower()
            )
            preferred = (
                ["codex_brain", "gemini_cli", "qwen_cloud", "groq"]
                if chat_policy == "premium_first"
                else []
            )
            order: List[str] = []
            for name in preferred:
                if name in providers_cfg and name not in order:
                    order.append(name)
            for name in sorted(providers_cfg.keys()):
                if name not in order:
                    order.append(name)

            decision_trace: Optional[Dict[str, Any]] = None
            trace_entries: Dict[str, Dict[str, Any]] = {}
            model_by_provider: Dict[str, Any] = {}
            ranked: List[str] = []
            ladder: List[Tuple[str, Dict[str, Any]]] = []
            try:
                if pick_model:
                    prov, cfg = pick_model(
                        "brain_chat",
                        providers_cfg=dict(providers_cfg),
                        cost_mode=cost_mode,
                        intent=None,
                        slot="brain.chat",
                    )
                    decision_trace = cfg.get("_decision_trace") if isinstance(cfg, dict) else None
                    if isinstance(decision_trace, dict):
                        ranked_raw = (
                            decision_trace.get("ranked_candidates")
                            or decision_trace.get("fallback_chain")
                            or []
                        )
                        ranked = [str(x) for x in ranked_raw if isinstance(x, str)]
                        for ent in decision_trace.get("candidates") or []:
                            if not isinstance(ent, dict):
                                continue
                            name = str(ent.get("provider") or "").strip()
                            if not name:
                                continue
                            trace_entries[name] = ent
                            if ent.get("model") is not None:
                                model_by_provider[name] = ent.get("model")
            except Exception:
                decision_trace = None
                trace_entries = {}
                model_by_provider = {}
                ranked = []

            def _cfg_is_local(cfg: Dict[str, Any]) -> bool:
                kind = str(cfg.get("kind") or "").strip().lower()
                if kind == "static":
                    return True
                base_url = str(cfg.get("base_url") or "").strip().lower()
                return "localhost" in base_url or "127.0.0.1" in base_url

            def _ledger_eligible(provider: str) -> tuple[bool, Optional[str]]:
                row = ledger_by_provider.get(provider)
                if not isinstance(row, dict):
                    return True, None
                status = str(row.get("status") or "").strip().lower()
                if not status:
                    return True, None
                if status == "ok":
                    return True, None
                if status == "soft_fail":
                    cd_ts = row.get("cooldown_until_ts")
                    if isinstance(cd_ts, (int, float)) and cd_ts > time.time():
                        return False, "cooldown"
                    return False, "policy"
                return False, "policy"

            def _fallback_ladder() -> List[Tuple[str, Dict[str, Any]]]:
                out: List[Tuple[str, Dict[str, Any]]] = []
                for name in order:
                    cfg_any = providers_cfg.get(name)
                    if not isinstance(cfg_any, dict) or cfg_any.get("disabled"):
                        continue
                    if _provider_is_down(name):
                        continue
                    if not allow_local_override and _cfg_is_local(cfg_any):
                        continue
                    tier = str(cfg_any.get("tier") or "balanced")
                    if not _tier_allows(tier, cost_mode):
                        continue
                    roles = cfg_any.get("roles") or []
                    if not isinstance(roles, list) or "brain" not in [r.lower() for r in roles]:
                        continue
                    modes = cfg_any.get("modes")
                    if modes:
                        modes_l = [m.lower() for m in modes] if isinstance(modes, list) else []
                        if "chatter" not in modes_l:
                            continue
                    eligible, _why = _ledger_eligible(name)
                    if not eligible:
                        continue
                    model_id = cfg_any.get("default_model") or cfg_any.get("model")
                    out.append((name, {**cfg_any, "_selected_model": model_id}))
                return out

            if ranked:
                for name in ranked:
                    cfg_any = providers_cfg.get(name)
                    if not isinstance(cfg_any, dict) or cfg_any.get("disabled"):
                        continue
                    if _provider_is_down(name):
                        continue
                    if not allow_local_override and _cfg_is_local(cfg_any):
                        continue
                    tier = str(cfg_any.get("tier") or "balanced")
                    if not _tier_allows(tier, cost_mode):
                        continue
                    roles = cfg_any.get("roles") or []
                    if not isinstance(roles, list) or "brain" not in [r.lower() for r in roles]:
                        continue
                    modes = cfg_any.get("modes")
                    if modes:
                        modes_l = [m.lower() for m in modes] if isinstance(modes, list) else []
                        if "chatter" not in modes_l:
                            continue
                    eligible, _why = _ledger_eligible(name)
                    if not eligible:
                        continue
                    model_id = (
                        model_by_provider.get(name)
                        or cfg_any.get("default_model")
                        or cfg_any.get("model")
                    )
                    cfg2: Dict[str, Any] = {**cfg_any, "_selected_model": model_id}
                    if decision_trace and not ladder:
                        cfg2["_decision_trace"] = decision_trace
                    ladder.append((name, cfg2))
            if not ladder:
                ladder = _fallback_ladder()

            candidates_view: List[Dict[str, Any]] = []
            for name in order:
                cfg_any = providers_cfg.get(name) if isinstance(providers_cfg, dict) else None
                ent = trace_entries.get(name)
                eligible = True
                reasons: List[str] = []
                cooldown_until = None
                cooldown_ts = None
                if isinstance(ent, dict):
                    eligible = bool(ent.get("eligible"))
                    cooldown_until = ent.get("cooldown_until")
                    cooldown_ts = ent.get("cooldown_until_ts")
                    reject = (
                        ent.get("reject_codes") if isinstance(ent.get("reject_codes"), list) else []
                    )
                    if any(str(c).upper() == "COOLDOWN_ACTIVE" for c in reject):
                        eligible = False
                        reasons.append("cooldown")
                    elif any("QUOTA" in str(c).upper() for c in reject):
                        eligible = False
                        reasons.append("cooldown")
                    elif reject:
                        eligible = False
                        reasons.append("policy")
                if not isinstance(cfg_any, dict):
                    eligible = False
                    reasons.append("policy")
                else:
                    if cfg_any.get("disabled"):
                        eligible = False
                        reasons.append("policy")
                    if not allow_local_override and _cfg_is_local(cfg_any):
                        eligible = False
                        reasons.append("policy")
                    tier = str(cfg_any.get("tier") or "balanced")
                    if not _tier_allows(tier, cost_mode):
                        eligible = False
                        reasons.append("policy")
                    roles = cfg_any.get("roles") or []
                    if not isinstance(roles, list) or "brain" not in [r.lower() for r in roles]:
                        eligible = False
                        reasons.append("policy")
                    modes = cfg_any.get("modes")
                    if modes:
                        modes_l = [m.lower() for m in modes] if isinstance(modes, list) else []
                        if "chatter" not in modes_l:
                            eligible = False
                            reasons.append("policy")
                    led_ok, led_reason = _ledger_eligible(name)
                    if not led_ok:
                        eligible = False
                        reasons.append(led_reason or "policy")
                        row = ledger_by_provider.get(name)
                        if isinstance(row, dict):
                            cooldown_until = cooldown_until or row.get("cooldown_until")
                            cooldown_ts = cooldown_ts or row.get("cooldown_until_ts")
                            if led_reason == "cooldown":
                                reasons.append("cooldown")
                if _provider_is_down(name):
                    eligible = False
                    reasons.append("policy:down")
                p95 = _provider_latency_p95_ms(name)
                if isinstance(p95, int) and p95 > 0:
                    reasons.append(f"latency={p95}ms")
                source_ts = (
                    f"ledger={ledger_updated_utc or 'n/a'} status={status_updated_utc or 'n/a'}"
                )
                reasons_u: List[str] = []
                for r in reasons:
                    if r and r not in reasons_u:
                        reasons_u.append(r)
                if cooldown_until and "cooldown" in reasons_u:
                    reasons_u = [
                        r
                        for r in reasons_u
                        if r != "cooldown" and not str(r).startswith("cooldown_until=")
                    ]
                    reasons_u.insert(0, f"cooldown_until={cooldown_until}")
                reason_str = " + ".join(reasons_u) if reasons_u else "ok"
                candidates_view.append(
                    {
                        "provider": name,
                        "role": "brain",
                        "eligible": bool(eligible),
                        "reason": reason_str,
                        "source_ts": source_ts,
                    }
                )
            return ladder, candidates_view

        def _fmt_ms(ms: Any) -> Optional[str]:
            try:
                n = int(ms)
            except Exception:
                return None
            if n < 0:
                return None
            if n < 1000:
                return f"{n}ms"
            sec = n / 1000.0
            if sec < 10:
                return f"{sec:.1f}s"
            return f"{int(round(sec))}s"

        def _fmt_player(prov: str, cfg: Dict[str, Any]) -> str:
            mid = cfg.get("_selected_model") or cfg.get("default_model") or cfg.get("model")
            return f"{prov}:{mid}" if mid else prov

        def _chat_reason(prov: str, cfg: Dict[str, Any]) -> str:
            # reason ultra-corto: auth_state + latency_p95 + tier
            tier = str(cfg.get("tier") or "balanced").strip().lower()
            auth_token = "auth_unknown"
            p95_token = None
            try:
                root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
                status_path = Path(root_dir) / "artifacts" / "health" / "providers_status.json"
                if status_path.exists():
                    doc = json.loads(status_path.read_text(encoding="utf-8"))
                    entry = (
                        ((doc.get("providers") or {}).get(prov) or {})
                        if isinstance(doc, dict)
                        else {}
                    )
                    if isinstance(entry, dict):
                        astate = str(entry.get("auth_state") or "").strip().upper()
                        if astate == "OK":
                            auth_token = "auth_ok"
                        elif astate == "MISSING":
                            auth_token = "auth_missing"
                        elif astate == "EXPIRED":
                            auth_token = "auth_expired"
                        p95 = entry.get("latency_p95_ms")
                        p95_f = _fmt_ms(p95)
                        if p95_f:
                            p95_token = f"p95={p95_f}"
            except Exception:
                pass
            parts = [auth_token]
            if p95_token:
                parts.append(p95_token)
            parts.append(f"tier={tier}")
            return " + ".join(parts)

        def _provider_priority(cfg: Dict[str, Any]) -> int:
            if not isinstance(cfg, dict):
                return 1
            kind = str(cfg.get("kind") or "").lower()
            transport = str(cfg.get("transport") or "").lower()
            tokens = f"{kind} {transport}"
            cloud_tokens = ("http", "https", "cloud", "groq", "openai", "vertex", "anthropic")
            return 0 if any(token in tokens for token in cloud_tokens) else 1

        def _emit_chat_cli_provider_error_gap(
            *,
            provider: str,
            provider_cfg: Dict[str, Any],
            error_kind: str,
            fix_hint: str,
            error: Optional[str],
            returncode: Optional[int] = None,
        ) -> Optional[str]:
            """
            Gap accionable para fallos de providers kind=cli durante chat.
            - No guarda el prompt (solo command template y metadatos).
            - Nombre estable (append via occurrences) para evitar explosión.
            """
            try:
                gap_dir = Path(root_dir) / "artifacts" / "capability_gaps"
                gap_dir.mkdir(parents=True, exist_ok=True)
                safe_kind = (
                    re.sub(r"[^a-z0-9_]+", "_", (error_kind or "unknown").lower()).strip("_")
                    or "unknown"
                )
                out_path = gap_dir / f"chat_cli_provider_error_{provider}_{safe_kind}.json"
                prev: Dict[str, Any] = {}
                if out_path.exists():
                    try:
                        prev = json.loads(out_path.read_text(encoding="utf-8"))
                        if not isinstance(prev, dict):
                            prev = {}
                    except Exception:
                        prev = {}
                occ = 0
                try:
                    occ = int(prev.get("occurrences") or 0)
                except Exception:
                    occ = 0
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))
                model_id = (
                    provider_cfg.get("_selected_model")
                    or provider_cfg.get("default_model")
                    or provider_cfg.get("model")
                    or None
                )
                payload: Dict[str, Any] = {
                    "gap_id": f"CHAT_CLI_PROVIDER_ERROR_{provider}_{safe_kind}".upper(),
                    "capability_family": f"chat_cli_provider_error_{safe_kind}",
                    "provider": provider,
                    "tool": str(((provider_cfg.get("command") or [None])[0]) or ""),
                    "model": model_id,
                    "error_kind": safe_kind,
                    "returncode": returncode,
                    "error": (error or "")[:400],
                    "fix_hint": (fix_hint or "")[:400],
                    "timeout_seconds": provider_cfg.get("timeout_seconds"),
                    "command_template": provider_cfg.get("command"),
                    "occurrences": occ + 1,
                    "created_at": str(prev.get("created_at") or now_iso),
                    "updated_at": now_iso,
                }
                out_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                return str(out_path)
            except Exception:
                return None

        def _maybe_print_chat_candidates(
            ladder: List[Tuple[str, Dict[str, Any]]], candidates: List[Dict[str, Any]]
        ) -> None:
            mode = (os.getenv("AJAX_CHAT_STARTING_XI_PRINT") or "debug").strip().lower()
            if mode in {"0", "false", "off", "never", "no"}:
                return
            if mode == "debug":
                dbg = (os.getenv("AJAX_CHAT_DEBUG") or "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if not dbg:
                    return
            if not candidates:
                return
            print("Chat candidates:")
            print("provider\trole\teligible\treason\tsource_ts")
            for row in candidates:
                if not isinstance(row, dict):
                    continue
                print(
                    "\t".join(
                        [
                            str(row.get("provider") or ""),
                            str(row.get("role") or ""),
                            "yes" if row.get("eligible") else "no",
                            str(row.get("reason") or ""),
                            str(row.get("source_ts") or ""),
                        ]
                    )
                )
            if ladder:
                max_items = 3
                try:
                    max_items = max(1, int(os.getenv("AJAX_CHAT_STARTING_XI_LADDER_MAX", "3") or 3))
                except Exception:
                    max_items = 3
                chain = " -> ".join(_fmt_player(p, c) for p, c in ladder[:max_items])
                if len(ladder) > max_items:
                    chain += " -> …"
                print(f"Chat ladder: {chain}")

        ladder, candidates_view = _build_chat_ladder()
        _maybe_print_chat_candidates(ladder, candidates_view)

        if not ladder:
            raise RuntimeError("chat_all_brains_failed: no_viable_provider")

        response_text: Optional[str] = None
        chosen_provider: Optional[str] = None
        chosen_cfg: Optional[Dict[str, Any]] = None
        for provider_name, provider_cfg in ladder:
            kind = (provider_cfg.get("kind") or "").lower()
            model = (
                provider_cfg.get("_selected_model")
                or provider_cfg.get("default_model")
                or provider_cfg.get("model")
                or "llama-3.1-8b-instant"
            )
            timeout = int(provider_cfg.get("timeout_seconds") or 15)
            lite_timeout_env = os.getenv("AJAX_CHAT_LLM_TIMEOUT_SEC")
            if getattr(self, "_chat_lite", False):
                if lite_timeout_env is None:
                    ux_profile = (
                        str(getattr(self, "_chat_last_ux_profile", "") or "").strip().lower()
                    )
                    lite_timeout = 12 if ux_profile == "human" else 4
                else:
                    try:
                        lite_timeout = int(lite_timeout_env)
                    except Exception:
                        lite_timeout = 4
                if lite_timeout > 0:
                    timeout = min(timeout, lite_timeout)

            try:
                if kind == "codex_cli_jsonl":
                    data = self._call_codex_cli(
                        provider_cfg, system_prompt, user_prompt, parse_json=False
                    )
                    return str(data)
                if kind == "cli":
                    prompt = system_prompt.rstrip() + "\n\n" + user_prompt
                    prompt = prompt.encode("utf-8", errors="ignore").decode(
                        "utf-8", errors="ignore"
                    )
                    cmd_template = provider_cfg.get("command") or []
                    cmd: List[str] = []
                    needs_stdin = True
                    for token in cmd_template:
                        if token == "{prompt}":
                            cmd.append(prompt)
                            needs_stdin = False
                        elif token == "{model}":
                            cmd.append(str(model))
                        else:
                            cmd.append(str(token))
                    if not cmd:
                        raise RuntimeError("cli_missing_command")
                    env = os.environ.copy()
                    extra_env = provider_cfg.get("env")
                    if isinstance(extra_env, dict):
                        for k, v in extra_env.items():
                            env[str(k)] = str(v)
                    cwd = provider_cfg.get("workdir")
                    cwd_path = None
                    if isinstance(cwd, str) and cwd.strip():
                        try:
                            base = (
                                getattr(self, "config", None).root_dir
                                if getattr(self, "config", None)
                                else None
                            )
                            cwd_path = str((Path(base) / cwd).resolve()) if base else cwd
                        except Exception:
                            cwd_path = cwd
                    try:
                        proc = subprocess.run(
                            cmd,
                            input=prompt if needs_stdin else None,
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                            check=False,
                            env=env,
                            cwd=cwd_path,
                        )
                    except FileNotFoundError as exc:
                        tool = str(cmd[0]) if cmd else "cli"
                        fix = f"binary_missing: instala `{tool}` y asegúrate de que está en PATH."
                        gap_path = _emit_chat_cli_provider_error_gap(
                            provider=provider_name,
                            provider_cfg=provider_cfg,
                            error_kind="binary_missing",
                            fix_hint=fix,
                            error=str(exc),
                        )
                        raise RuntimeError(f"cli_binary_missing:{tool} (gap={gap_path})") from exc
                    except subprocess.TimeoutExpired as exc:
                        fix = "timeout: sube `timeout_seconds` o revisa conectividad."
                        gap_path = _emit_chat_cli_provider_error_gap(
                            provider=provider_name,
                            provider_cfg=provider_cfg,
                            error_kind="timeout",
                            fix_hint=fix,
                            error=str(exc),
                        )
                        raise RuntimeError(f"cli_timeout (gap={gap_path})") from exc
                    if proc.returncode != 0:
                        err = (
                            (proc.stderr or "").strip()
                            or (proc.stdout or "").strip()
                            or f"cli_rc_{proc.returncode}"
                        )
                        akind = "nonzero_exit"
                        fix = "nonzero_exit: ejecuta el comando manualmente y revisa stderr; verifica PATH/auth."
                        if auth_mgr is not None:
                            try:
                                astate = auth_mgr.auth_state(provider_name, provider_cfg)
                                if astate.state in {"MISSING", "EXPIRED"}:
                                    akind = (
                                        "auth_missing"
                                        if astate.state == "MISSING"
                                        else "auth_expired"
                                    )
                                    fix = astate.instructions or fix
                                    try:
                                        auth_mgr.persist_auth_state(provider_name, astate)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        gap_path = _emit_chat_cli_provider_error_gap(
                            provider=provider_name,
                            provider_cfg=provider_cfg,
                            error_kind=akind,
                            fix_hint=fix,
                            error=err,
                            returncode=int(proc.returncode),
                        )
                        raise RuntimeError(f"cli_{akind}:rc={proc.returncode} (gap={gap_path})")
                    out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
                    if not out:
                        gap_path = _emit_chat_cli_provider_error_gap(
                            provider=provider_name,
                            provider_cfg=provider_cfg,
                            error_kind="nonzero_exit",
                            fix_hint="empty_reply: el CLI devolvió salida vacía; revisa configuración/versión.",
                            error="cli_empty_reply",
                            returncode=int(proc.returncode),
                        )
                        raise RuntimeError(f"cli_empty_reply (gap={gap_path})")
                    response_text = out
                    chosen_provider = provider_name
                    chosen_cfg = provider_cfg
                    break

                base_url = (provider_cfg.get("base_url") or "").rstrip("/")
                api_key_env = provider_cfg.get("api_key_env")
                api_key = os.getenv(api_key_env) if api_key_env else None
                if not base_url:
                    raise RuntimeError("Brain provider sin base_url")
                if api_key_env and not api_key:
                    raise RuntimeError(f"API key no configurada ({api_key_env})")

                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 128,
                }
                resp = requests.post(
                    f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"brain_http_{resp.status_code}: {resp.text}")
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
                content = content.strip()
                if not content:
                    raise RuntimeError("brain_empty_reply")
                response_text = content
                chosen_provider = provider_name
                chosen_cfg = provider_cfg
                break
            except Exception as exc:
                errors.append(f"{provider_name}:{exc}")
                fallback_chain.append({"provider": provider_name, "error": str(exc)[:200]})
                continue

        if response_text is None:
            raise RuntimeError("chat_all_brains_failed: " + " | ".join(errors[-4:]))
        if fallback_chain:
            try:
                last_reason = fallback_chain[-1].get("error")
                if last_reason:
                    print(f"Chat fallback_reason: {last_reason}")
            except Exception:
                pass
        try:
            receipt_dir = Path(root_dir) / "artifacts" / "health"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            ladder_chain = []
            for prov, cfg in ladder:
                ladder_chain.append(
                    {
                        "provider": prov,
                        "model": cfg.get("_selected_model")
                        or cfg.get("default_model")
                        or cfg.get("model"),
                    }
                )
            receipt = {
                "schema": "ajax.chat.selection.v0",
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "tier": cost_mode,
                "ladder": ladder_chain,
                "chosen": {
                    "provider": chosen_provider,
                    "model": (chosen_cfg or {}).get("_selected_model") if chosen_cfg else None,
                },
                "fallback_chain": fallback_chain,
            }
            (receipt_dir / "chat_selection.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        return response_text

    def _call_codex_cli(
        self,
        provider_cfg: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        parse_json: bool = True,
        *,
        env_override: Optional[Dict[str, str]] = None,
        timeout_cfg: Optional[Dict[str, Any]] = None,
        return_timings: bool = False,
    ) -> Any:
        """
        Ejecuta Codex CLI en modo JSONL.
        - Si parse_json=True, intenta parsear la respuesta como JSON (para planes).
        - Si parse_json=False, devuelve el texto concatenado.
        """
        cmd_template = provider_cfg.get("command") or [
            "codex",
            "exec",
            "--model",
            "{model}",
            "--json",
        ]
        model_id = (
            provider_cfg.get("default_model") or provider_cfg.get("model") or "gpt-5.1-codex-max"
        )
        timeout = int(provider_cfg.get("timeout_seconds") or 60)

        cmd: List[str] = []
        for token in cmd_template:
            if token == "{model}":
                cmd.append(model_id)
            else:
                cmd.append(token)
        if "--skip-git-repo-check" not in cmd:
            cmd.append("--skip-git-repo-check")

        full_prompt = system_prompt.rstrip() + "\n\n" + user_prompt
        full_prompt = full_prompt.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        env = os.environ.copy()
        extra_env = provider_cfg.get("env")
        if isinstance(extra_env, dict):
            for k, v in extra_env.items():
                env[str(k)] = str(v)
        if isinstance(env_override, dict):
            for k, v in env_override.items():
                env[str(k)] = str(v)
        timeout_cfg = dict(timeout_cfg or {})
        timeout_cfg.setdefault("total_timeout_ms", int(timeout * 1000))
        result = self._run_cli_with_timeouts(
            cmd,
            input_text=full_prompt,
            timeout_cfg=timeout_cfg,
            env=env,
        )
        stderr_tail = (
            (result.get("stderr") or "")[-4096:] if isinstance(result.get("stderr"), str) else None
        )
        timings = dict(result.get("timings") or {})
        timings.update(
            {
                "timeout_kind": result.get("timeout_kind") or "NONE",
                "client_abort": bool(result.get("client_abort")),
                "exit_code": result.get("returncode"),
                "stderr_tail": stderr_tail,
                "bytes_rx": int(result.get("bytes_stdout") or 0)
                + int(result.get("bytes_stderr") or 0),
                "tokens_rx": None,
            }
        )
        if result.get("client_abort") and result.get("timeout_kind") in {"TTFT", "STALL", "TOTAL"}:
            msg = f"Codex CLI timeout ({result.get('timeout_kind')})"
            if BrainProviderError is not None:
                raise BrainProviderError(
                    msg,
                    error_code="client_timeout",
                    error_detail=msg,
                    stderr_excerpt=stderr_tail,
                    timeout_s=timeout,
                    timings=timings,
                )
            raise RuntimeError(msg)
        if result.get("returncode") != 0:
            rc = result.get("returncode")
            err = (
                (result.get("stderr") or "").strip()
                or (result.get("stdout") or "").strip()
                or f"codex_rc_{rc}"
            )
            if BrainProviderError is not None:
                raise BrainProviderError(
                    f"Codex CLI falló (rc={rc}): {err[:200]!r}",
                    error_code=f"codex_rc_{rc}",
                    error_detail=err[:200],
                    stderr_excerpt=stderr_tail,
                    timeout_s=timeout,
                    timings=timings,
                )
            raise RuntimeError(f"Codex CLI falló (rc={rc}): {err[:200]!r}")

        final_chunks: List[str] = []
        stdout_text = str(result.get("stdout") or "")
        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        final_chunks.append(text)

        if not final_chunks:
            msg = f"Codex CLI no devolvió mensajes de agente. stdout={stdout_text[:200]!r}"
            if BrainProviderError is not None:
                raise BrainProviderError(
                    msg,
                    error_code="codex_empty_reply",
                    error_detail=msg,
                    stderr_excerpt=stderr_tail,
                    timeout_s=timeout,
                    timings=timings,
                )
            raise ValueError(msg)

        full_text = "\n".join(final_chunks).strip()
        try:
            self._last_raw_plan_provider = "codex_brain"
            self._last_raw_plan_excerpt = full_text[:500]
        except Exception:
            pass
        if not parse_json:
            return (full_text, timings) if return_timings else full_text

        if full_text.startswith("```"):
            full_text = full_text.strip("`")
            if full_text.startswith("json"):
                full_text = full_text[len("json") :].strip()

        try:
            plan = json.loads(full_text)
        except json.JSONDecodeError as exc:
            msg = f"Plan JSON inválido desde Codex: {exc}: {full_text[:200]!r}"
            if BrainProviderError is not None:
                raise BrainProviderError(
                    msg,
                    error_code="parse_error",
                    error_detail=full_text[:200],
                    parse_error=True,
                    timeout_s=timeout,
                    timings=timings,
                )
            raise ValueError(msg)
        return (plan, timings) if return_timings else plan

    def _select_brain_provider(
        self, exclude: Optional[set[str]] = None
    ) -> tuple[str, Dict[str, Any]]:
        providers_cfg = (
            self.provider_configs.get("providers", {})
            if isinstance(self.provider_configs, dict)
            else {}
        )
        cost_mode = os.getenv("AJAX_COST_MODE", "premium")
        if pick_model:
            filtered = {k: v for k, v in providers_cfg.items() if not exclude or k not in exclude}
            prov, cfg = pick_model(
                "brain_plan",
                providers_cfg=filtered,
                cost_mode=cost_mode,
                intent=None,
                slot="brain.main",
            )
            if cfg.get("_slot_missing"):
                try:
                    self.log.warning(
                        "Slot brain.main faltante en inventario (%s), usando fallback %s",
                        cfg.get("_slot_requested"),
                        cfg.get("_slot_selected") or prov,
                    )
                except Exception:
                    pass
            trace = cfg.pop("_decision_trace", None)
            self._last_brain_selection_trace = trace
            self._record_chat_selection_receipt(provider=prov, cfg=cfg, trace=trace)
            return prov, cfg
        self._last_brain_selection_trace = None
        # fallback si pick_model no está disponible
        for name, cfg in providers_cfg.items():
            roles = cfg.get("roles") or []
            if cfg.get("disabled"):
                continue
            if isinstance(roles, list) and "brain" in [r.lower() for r in roles]:
                return name, cfg
        raise RuntimeError("No hay provider con rol 'brain' configurado")

    def _validate_brain_plan(self, brain_output: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(brain_output, dict):
            raise ValueError("brain_output no es un dict")
        steps = brain_output.get("steps")
        if not isinstance(steps, list):
            raise ValueError("brain_output.steps no es una lista")
        required = {
            "id",
            "intent",
            "preconditions",
            "action",
            "args",
            "evidence_required",
            "success_spec",
            "on_fail",
        }

        def _expected_state_has_checks(expected: Any) -> bool:
            if not isinstance(expected, dict):
                return False
            if expected.get("windows"):
                return True
            if expected.get("files"):
                return True
            if isinstance(expected.get("checks"), list) and expected.get("checks"):
                return True
            meta = expected.get("meta") or {}
            return bool(isinstance(meta, dict) and meta.get("must_be_active"))

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"invalid_brain_plan:step_not_dict_at_{i + 1}")
            missing = [k for k in sorted(required) if k not in step]
            if missing:
                raise ValueError(
                    f"invalid_brain_plan:missing_fields_at_{i + 1}:" + ",".join(missing)
                )

            if not isinstance(step.get("id"), str) or not str(step.get("id")).strip():
                raise ValueError(f"invalid_brain_plan:bad_id_at_{i + 1}")
            if not isinstance(step.get("intent"), str) or not str(step.get("intent")).strip():
                raise ValueError(f"invalid_brain_plan:bad_intent_at_{i + 1}")
            if step.get("on_fail") != "abort":
                raise ValueError(f"invalid_brain_plan:on_fail_must_be_abort_at_{i + 1}")

            action = step.get("action")
            if not isinstance(action, str) or not action.strip():
                raise ValueError(f"invalid_brain_plan:bad_action_at_{i + 1}")
            if getattr(self, "actions_catalog", None) is not None:
                if not self.actions_catalog.is_allowed(action):
                    raise ValueError(f"invalid_brain_plan:action_not_allowed_at_{i + 1}:{action}")

            args = step.get("args")
            if not isinstance(args, dict):
                raise ValueError(f"invalid_brain_plan:args_not_object_at_{i + 1}")

            pre = step.get("preconditions")
            if not isinstance(pre, dict) or not isinstance(pre.get("expected_state"), dict):
                raise ValueError(f"invalid_brain_plan:bad_preconditions_at_{i + 1}")

            succ = step.get("success_spec")
            if not isinstance(succ, dict) or not isinstance(succ.get("expected_state"), dict):
                raise ValueError(f"invalid_brain_plan:bad_success_spec_at_{i + 1}")
            # Allow await_user_input to pause without requiring EFE checks.
            if action != "await_user_input":
                if not _expected_state_has_checks(succ.get("expected_state")):
                    raise ValueError(f"invalid_brain_plan:empty_success_spec_at_{i + 1}")

            ev = step.get("evidence_required")
            if not isinstance(ev, list) or any(not isinstance(x, str) or not x.strip() for x in ev):
                raise ValueError(f"invalid_brain_plan:bad_evidence_required_at_{i + 1}")

        if "success_spec" in brain_output and not isinstance(
            brain_output.get("success_spec"), dict
        ):
            brain_output["success_spec"] = None
        if "success_contract" in brain_output and not isinstance(
            brain_output.get("success_contract"), dict
        ):
            brain_output["success_contract"] = None
        return brain_output

    def _enforce_brain_plan_order(self, plan_json: Dict[str, Any]) -> None:
        """
        Validación estricta del plan Brain antes de entregarlo al runner.
        Reglas:
        1) keyboard.type debe tener ui.inspect y window.focus previos (y solo un focus antes de teclear).
        2) text no puede contener Enter ni secuencias \\n, \\r, ↵.
        3) Orden obligatorio: app.launch (opcional) -> ui.inspect -> window.focus -> keyboard.type -> keyboard.hotkey.
           Si hay hotkey, debe ir después de keyboard.type.
        Plan inválido -> lanza ValueError("invalid_brain_plan:<motivo>")
        """
        steps: List[Dict[str, Any]] = plan_json.get("steps") or []
        if not steps:
            raise ValueError("invalid_brain_plan:empty_steps")

        seen_focus = False
        focus_count = 0
        last_action: Optional[str] = None
        seen_inspect = False

        order_map = {"app.launch": 0, "window.focus": 1, "keyboard.type": 2, "keyboard.hotkey": 3}
        last_order = -1

        for idx, step in enumerate(steps):
            action = step.get("action")
            args = step.get("args") or {}
            if not isinstance(action, str):
                continue

            # Regla 2: text no debe contener enter embebido
            if action == "keyboard.type":
                text = args.get("text") or ""
                if any(tok in str(text) for tok in ["Enter", "\n", "\r", "↵"]):
                    raise ValueError(f"invalid_brain_plan:text_contains_enter_at_step_{idx + 1}")

            # Regla 4: orden obligatorio
            if action in order_map:
                current_order = order_map[action]
                if current_order < last_order:
                    # Permitimos keyboard.type después de un hotkey (ej. ctrl+l) aunque el orden map sea menor
                    if not (action == "keyboard.type" and last_action == "keyboard.hotkey"):
                        raise ValueError(f"invalid_brain_plan:bad_order_{action}_at_step_{idx + 1}")
                last_order = max(last_order, current_order)
            last_action = action

            if action == "ui.inspect":
                seen_inspect = True
            if action == "window.focus":
                seen_focus = True
                focus_count += 1
            if action == "keyboard.type":
                # inspect no obligatorio; focus recomendado pero no bloqueante
                if focus_count > 1:
                    raise ValueError(
                        f"invalid_brain_plan:focus_count_{focus_count}_before_keyboard_type_at_step_{idx + 1}"
                    )
            if action == "keyboard.hotkey":
                if last_order < order_map["keyboard.type"]:
                    raise ValueError(f"invalid_brain_plan:hotkey_before_type_at_step_{idx + 1}")

    def _enforce_degraded_mode(self, plan: AjaxPlan) -> Optional[AjaxExecutionResult]:
        """
        COUNCIL degraded es informativo (fallback/timeout); no debe bloquear ejecución.
        El bloqueo duro se expresa vía escalation_hint/blocked o unsafe.
        """
        council_meta = (plan.metadata or {}).get("council_verdict") if plan.metadata else None
        degraded = (
            bool(council_meta.get("council_degraded")) if isinstance(council_meta, dict) else False
        )
        if not degraded:
            return None
        reason = (
            (council_meta.get("council_degraded_reason") or council_meta.get("reason"))
            if isinstance(council_meta, dict)
            else None
        )
        tried = council_meta.get("providers_tried") if isinstance(council_meta, dict) else None
        try:
            self.log.warning("Council degraded (no bloqueante): %s (%s)", reason, tried)
        except Exception:
            pass
        return None

    def _record_leann_chat_receipt(
        self,
        *,
        reason: str,
        mode: str,
        snippets_count: int,
        profile_used: bool,
    ) -> Optional[str]:
        try:
            root_dir = getattr(getattr(self, "config", None), "root_dir", None) or Path(".")
            receipt_dir = Path(root_dir) / "artifacts" / "metrics"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            payload = {
                "schema": "ajax.leann_chat.v1",
                "ts": time.time(),
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": mode,
                "reason": reason,
                "snippets_count": int(snippets_count),
                "profile_used": bool(profile_used),
            }
            out_path = receipt_dir / f"leann_chat_{ts}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return str(out_path)
        except Exception:
            return None

    def act(self, plan: AjaxPlan) -> AjaxExecutionResult:
        if not plan.steps:
            return AjaxExecutionResult(
                success=False, error="plan_empty", path="act", plan_id=plan.plan_id
            )
        job_path = plan.metadata.get("job_path") if plan.metadata else None
        guard_result = self._enforce_degraded_mode(plan)
        if guard_result:
            return guard_result
        if job_path and AgencyJob and run_job_plan:
            try:
                job_obj = AgencyJob.load(Path(job_path))
                res = run_job_plan(job_obj, actuator=self.actuator)
                return AjaxExecutionResult(
                    success=bool(res.get("ok")),
                    detail=res,
                    path="plan_runner",
                    plan_id=plan.plan_id or plan.id,
                    artifacts=res.get("artifacts") if isinstance(res, dict) else None,
                )
            except Exception as exc:
                return AjaxExecutionResult(
                    success=False, error=str(exc), path="plan_runner", plan_id=plan.plan_id
                )
        elif AgencyJob and run_job_plan and plan.steps:
            try:
                meta = dict(plan.metadata or {})
                meta["steps"] = plan.steps
                job_obj = AgencyJob(
                    job_id=plan.plan_id or plan.id or "plan",
                    goal=plan.summary or meta.get("intention") or "plan_runner_job",
                    metadata=meta,
                )
                res = run_job_plan(job_obj, actuator=self.actuator)
                return AjaxExecutionResult(
                    success=bool(res.get("ok")),
                    detail=res,
                    path="plan_runner",
                    plan_id=plan.plan_id or plan.id,
                    artifacts=res.get("artifacts") if isinstance(res, dict) else None,
                )
            except Exception as exc:
                return AjaxExecutionResult(
                    success=False, error=str(exc), path="plan_runner", plan_id=plan.plan_id
                )
        return AjaxExecutionResult(
            success=True, detail={"executed": plan.steps}, path="act_stub", plan_id=plan.plan_id
        )

    def verify(
        self,
        result: AjaxExecutionResult,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> AjaxExecutionResult:
        detail: Dict[str, Any]
        if isinstance(result.detail, dict):
            detail = dict(result.detail)
        else:
            detail = {"raw_detail": result.detail} if result.detail is not None else {}

        verification = (
            detail.get("verification") if isinstance(detail.get("verification"), dict) else {}
        )
        verification = dict(verification)
        verify_ok = bool(verification.get("ok")) if "ok" in verification else bool(result.success)
        verification["ok"] = verify_ok
        if "delta" not in verification:
            verification["delta"] = 0 if verify_ok else 1
        elif verify_ok:
            verification["delta"] = 0
        verification_mode = str(
            verification.get("verification_mode")
            or (context or {}).get("verification_mode")
            or ("real" if result.path == "plan_runner" else "synthetic")
        ).strip().lower()
        verification["verification_mode"] = verification_mode
        if "driver_online" not in verification:
            verification["driver_online"] = bool(
                (context or {}).get("driver_online", self._driver_online())
            )

        if microfilm_enforce_evidence_tiers is not None:
            try:
                verification = microfilm_enforce_evidence_tiers(
                    {
                        "driver_online": verification.get("driver_online"),
                        "verification_mode": verification_mode,
                    },
                    verification,
                )
            except Exception:
                pass
        else:
            verification["evidence_tier"] = (
                "real_online"
                if verify_ok and verification_mode == "real" and verification.get("driver_online")
                else "synthetic_or_offline"
            )
            verification["promote_trust"] = bool(
                verify_ok and verification_mode == "real" and verification.get("driver_online")
            )

        detail["verification"] = verification
        result.detail = detail
        return result

    # --- Misión (bucle unificado) ---
    def _emit_council_block_gap(self, mission: MissionState, cv: "CouncilVerdict") -> None:
        """
        Registra un capability_gap cuando el Council bloquea por motivos técnicos
        (p.ej. invalid_reviewer_output / no quorum por providers caídos).
        """
        try:
            root = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            root = None
        try:
            base = Path(root) if root else Path(__file__).resolve().parents[1]
            gap_dir = base / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_council_blocked_{mission.mission_id}"
            evidence_paths: List[str] = []
            try:
                sx_path = (
                    (mission.envelope.metadata or {}).get("starting_xi_path")
                    if mission.envelope
                    else None
                )
                if isinstance(sx_path, str) and sx_path:
                    evidence_paths.append(sx_path)
            except Exception:
                pass
            try:
                sx = (
                    (mission.envelope.metadata or {}).get("starting_xi")
                    if mission.envelope
                    else None
                )
                if isinstance(sx, dict):
                    inputs = sx.get("inputs")
                    if isinstance(inputs, dict):
                        lp = inputs.get("provider_ledger_path")
                        if isinstance(lp, str) and lp:
                            evidence_paths.append(lp)
            except Exception:
                pass
            try:
                probe = (
                    (mission.notes or {}).get("probe_driver")
                    if isinstance(mission.notes, dict)
                    else None
                )
                if isinstance(probe, dict):
                    rp = probe.get("report_path")
                    if isinstance(rp, str) and rp:
                        evidence_paths.append(rp)
            except Exception:
                pass
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "council.blocked",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "reason": getattr(cv, "reason", None),
                "escalation_hint": getattr(cv, "escalation_hint", None),
                "providers_tried": getattr(cv, "providers_tried", None)
                or getattr(cv, "debug_notes", None),
                "evidence_paths": evidence_paths or None,
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["council_gap_path"] = str(out_path)
            except Exception:
                pass
        except Exception:
            return

    def _emit_brain_failed_no_plan_gap(
        self,
        mission: MissionState,
        *,
        errors: Optional[List[str]] = None,
        plan_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Registra un capability_gap cuando el BrainRouter no produce un plan válido.
        Incluye refs al router trace para diagnóstico.
        """
        try:
            existing = None
            if isinstance(mission.notes, dict):
                existing = mission.notes.get("brain_failed_no_plan_gap_path")
            if isinstance(existing, str) and existing:
                if Path(existing).exists():
                    return existing
        except Exception:
            pass
        try:
            base = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            base = None
        try:
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            gap_dir = root / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_brain_failed_no_plan_{mission.mission_id}"
            router_trace = None
            router_attempts = None
            try:
                if isinstance(mission.notes, dict):
                    router_trace = mission.notes.get("router_trace_path")
                    router_attempts = mission.notes.get("router_attempts")
            except Exception:
                router_trace = None
                router_attempts = None
            sx_path = None
            try:
                if mission.envelope and isinstance(
                    getattr(mission.envelope, "metadata", None), dict
                ):
                    sx_path = mission.envelope.metadata.get("starting_xi_path")
            except Exception:
                sx_path = None
            plan_meta = plan_meta or {}
            refs: List[str] = []
            if isinstance(router_trace, str) and router_trace:
                refs.append(router_trace)
            if isinstance(sx_path, str) and sx_path:
                refs.append(sx_path)
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "brain_failed_no_plan",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "reason": "brain_failed_no_plan",
                "errors": errors or plan_meta.get("errors") or [],
                "provider_failures": plan_meta.get("provider_failures"),
                "router_trace_path": router_trace,
                "router_attempts": router_attempts,
                "starting_xi_path": sx_path,
                "refs": refs or None,
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["brain_failed_no_plan_gap_path"] = str(out_path)
            except Exception:
                pass
            return str(out_path)
        except Exception:
            return None

    def _emit_missing_efe_gap(
        self,
        mission: MissionState,
        *,
        receipt_path: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        """
        Registra un capability_gap cuando falla la reparación de EFE.
        Constitutional Fail-Closed: missing_efe_final.
        """
        try:
            base = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            base = None
        try:
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            gap_dir = root / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_missing_efe_final_{mission.mission_id}"

            payload = {
                "gap_id": gap_id,
                "capability_family": "constitution.efe",
                "symptoms": ["missing_expected_state", "repair_failed"],
                "mission_id": mission.mission_id,
                "intention": mission.intention,
                "reason": reason or "Agotados intentos de reparación de EFE",
                "created_at": self._iso_utc(),
                "evidence_refs": [],
                "next_actions": [
                    "Director: Revisar por qué el modelo no genera expected_state válido",
                    "Scout: Ajustar prompts de efe_repair si es un patrón recurrente"
                ]
            }
            if receipt_path:
                payload["evidence_refs"].append(receipt_path)

            fpath = gap_dir / f"{gap_id}.json"
            fpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return str(fpath)
        except Exception as exc:
            try:
                self.log.warning("No se pudo emitir gap (missing_efe): %s", exc)
            except Exception:
                pass
            return None

    def _emit_missing_efe_derived_gap(
        self,
        mission: MissionState,
        *,
        reason: Optional[str] = None,
        plan_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Registra un capability_gap cuando se intenta derivar a LAB sin EFE válido.
        Constitutional Fail-Closed: missing_efe_derived.
        """
        try:
            base = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            base = None
        try:
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            gap_dir = root / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_missing_efe_derived_{mission.mission_id}"
            payload = {
                "gap_id": gap_id,
                "capability_family": "constitution.efe",
                "symptoms": ["missing_expected_state", "derived_efe_blocked"],
                "mission_id": mission.mission_id,
                "intention": mission.intention,
                "plan_id": plan_id,
                "reason": reason or "Derived EFE inválido (missing expected_state)",
                "created_at": self._iso_utc(),
                "evidence_refs": [],
                "next_actions": [
                    "Director: Asegurar que los planes incluyan success_spec.expected_state",
                    "Scout: Ajustar prompts/normalización para EFE en derivaciones",
                ],
            }
            fpath = gap_dir / f"{gap_id}.json"
            fpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return str(fpath)
        except Exception as exc:
            try:
                self.log.warning("No se pudo emitir gap (missing_efe_derived): %s", exc)
            except Exception:
                pass
            return None

    def _emit_missing_players_gap(
        self, mission: MissionState, starting_xi: Dict[str, Any]
    ) -> Optional[str]:
        """
        Registra un capability_gap cuando el preflight no puede construir un Starting XI por rol.
        Soft-block: no es unsafe; es falta de jugadores (auth/quota/timeout).
        """
        try:
            base = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            base = None
        try:
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            gap_dir = root / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_missing_players_{mission.mission_id}"
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "preflight.missing_players",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "rail": starting_xi.get("rail"),
                "risk_level": starting_xi.get("risk_level"),
                "quorum": starting_xi.get("quorum"),
                "missing_players": starting_xi.get("missing_players"),
                "slots_missing": starting_xi.get("slots_missing"),
                "fix_hints": starting_xi.get("fix_hints"),
                "starting_xi_path": starting_xi.get("path"),
                "inputs": starting_xi.get("inputs"),
                "preflight": starting_xi.get("preflight"),
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["missing_players_gap_path"] = str(out_path)
            except Exception:
                pass
            return str(out_path)
        except Exception:
            return None

    def _emit_council_quorum_degraded_gap(
        self,
        mission: MissionState,
        *,
        starting_xi: Dict[str, Any],
        quorum_from: int,
        quorum_to: int,
        reason: str,
    ) -> Optional[str]:
        """
        Registra un capability_gap cuando se baja quorum del Council por falta de jugadores (ledger).
        No es unsafe: es degradación explícita con evidencia.
        """
        try:
            base = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            base = None
        try:
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            gap_dir = root / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_council_quorum_degraded_{mission.mission_id}"
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "council.quorum_degraded",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "rail": starting_xi.get("rail"),
                "risk_level": starting_xi.get("risk_level"),
                "quorum_from": int(quorum_from),
                "quorum_to": int(quorum_to),
                "reason": str(reason or ""),
                "starting_xi_path": starting_xi.get("path"),
                "inputs": starting_xi.get("inputs"),
                "preflight": starting_xi.get("preflight"),
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["council_quorum_degraded_gap_path"] = str(out_path)
            except Exception:
                pass
            return str(out_path)
        except Exception:
            return None

    def _emit_lab_display_unavailable_gap(
        self,
        mission: MissionState,
        *,
        rail: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Registra un capability_gap cuando rail=lab no puede actuar físicamente por falta de display/sesión visible.
        Política provisional mientras no hay HDMI dummy: fail-closed.
        """
        try:
            base = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            base = None
        try:
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            gap_dir = root / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            deduped = self._find_recent_capability_gap(
                kind="lab_display_unavailable",
                mission_id=mission.mission_id,
                window_seconds=300,
            )
            if deduped:
                try:
                    mission.notes["lab_display_gap_path"] = deduped
                    mission.notes["lab_display_gap_deduped"] = True
                except Exception:
                    pass
                return deduped
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_lab_display_unavailable_{mission.mission_id}"
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "lab_display_unavailable",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "rail": str(rail or "lab"),
                "detail": detail or {},
                "fix_hint": "Open an RDP session for LAB or set AJAX_LAB_DISPLAY_READY=1; then resume the same mission.",
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["lab_display_gap_path"] = str(out_path)
            except Exception:
                pass
            return str(out_path)
        except Exception:
            return None

    def _emit_lab_job_kind_unsupported_gap(
        self,
        mission: MissionState,
        *,
        lab_kind: str,
        reason: str,
    ) -> Optional[str]:
        try:
            base = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            base = None
        try:
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            gap_dir = root / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            deduped = self._find_recent_capability_gap(
                kind="lab_job_kind_unsupported",
                mission_id=mission.mission_id,
                window_seconds=300,
            )
            if deduped:
                try:
                    mission.notes["lab_job_kind_gap_path"] = deduped
                    mission.notes["lab_job_kind_gap_deduped"] = True
                    mission.notes["lab_job_kind_blocked_until"] = time.time() + 300
                except Exception:
                    pass
                return deduped
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_lab_job_kind_unsupported_{mission.mission_id}"
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "lab_job_kind_unsupported",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "lab_kind": str(lab_kind or ""),
                "reason": str(reason or ""),
                "fix_hint": "Mapear lab_kind a job_kind soportado por LAB worker o derivar a ASK_USER/manual.",
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["lab_job_kind_gap_path"] = str(out_path)
                mission.notes["lab_job_kind_blocked_until"] = time.time() + 300
            except Exception:
                pass
            return str(out_path)
        except Exception:
            return None

    def _emit_council_veto_timeout_gap(
        self, mission: MissionState, *, timeout_seconds: int
    ) -> Optional[str]:
        """
        Registra un capability_gap cuando la misión agota el budget temporal con veto(s) del Council activos/repetidos.
        Esto es un bloqueo gobernado (no un fallo silencioso).
        """
        try:
            base = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            base = None
        try:
            root = Path(base) if base else Path(__file__).resolve().parents[1]
            gap_dir = root / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_blocked_by_council_veto_{mission.mission_id}"
            vetoes = []
            try:
                vetoes = (mission.notes or {}).get("council_vetoes") or []
            except Exception:
                vetoes = []
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "council.veto_timeout",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "timeout_seconds": int(timeout_seconds),
                "last_status": mission.status,
                "council_vetoes": vetoes[-5:] if isinstance(vetoes, list) else vetoes,
                "fix_hint": "If veto is due to incomplete intent: transition to WAITING_FOR_USER (ASK_USER) instead of timing out; otherwise increase budget or improve planning/council prompts.",
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["council_veto_timeout_gap_path"] = str(out_path)
            except Exception:
                pass
            return str(out_path)
        except Exception:
            return None

    def _emit_needs_lab_proof_gap(
        self,
        mission: MissionState,
        *,
        reason: str,
        experiment_envelope_path: str,
        council_verdict: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Registra un capability_gap cuando una misión no puede ejecutarse por no-consenso del Council.
        La propuesta no se descarta: se deriva a LAB como experimento pendiente.
        """
        try:
            root = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            root = None
        try:
            base = Path(root) if root else Path(__file__).resolve().parents[1]
            gap_dir = base / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_needs_lab_proof_{mission.mission_id}"
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "needs_lab_proof",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "reason": reason,
                "fix_hint": "Run in LAB with a strong SuccessContract (>=2 independent signals for dynamic states) and durable evidence.",
                "experiment_envelope_path": experiment_envelope_path,
                "council_verdict": council_verdict or None,
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["needs_lab_proof_gap_path"] = str(out_path)
            except Exception:
                pass
            return str(out_path)
        except Exception:
            return None

    def _emit_budget_exhausted_gap(
        self,
        mission: MissionState,
        *,
        reason: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            root = getattr(self, "root_dir", None) or getattr(self, "config", None).root_dir  # type: ignore[union-attr]
        except Exception:
            root = None
        try:
            base = Path(root) if root else Path(__file__).resolve().parents[1]
            gap_dir = base / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            gap_id = f"{ts}_budget_exhausted_{mission.mission_id}"
            payload = {
                "capability_gap_id": gap_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
                "capability_family": "budget_exhausted",
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "reason": reason,
                "plan_attempts": int(getattr(mission, "plan_attempts", 0) or 0),
                "max_plan_attempts": int(getattr(mission, "max_attempts", 0) or 0),
                "detail": detail or {},
                "next_action": "derive_to_lab",
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                mission.notes["budget_exhausted_gap_path"] = str(out_path)
            except Exception:
                pass
            return str(out_path)
        except Exception:
            return None

    def _summarize_plan_steps(self, plan: Optional[AjaxPlan]) -> List[Dict[str, Any]]:
        if not plan or not isinstance(plan.steps, list):
            return []
        summarized: List[Dict[str, Any]] = []
        for step in plan.steps:
            if not isinstance(step, dict):
                continue
            item: Dict[str, Any] = {}
            if step.get("id"):
                item["id"] = step.get("id")
            if step.get("action"):
                item["action"] = step.get("action")
            summary = step.get("summary") or step.get("label")
            if summary:
                item["summary"] = summary
            if item:
                summarized.append(item)
        return summarized

    def _extract_expected_evidence(self, plan: Optional[AjaxPlan]) -> List[Any]:
        if not plan:
            return []
        spec = plan.success_spec if isinstance(plan.success_spec, dict) else None
        if not spec and isinstance(plan.metadata, dict):
            spec = (
                plan.metadata.get("success_contract")
                if isinstance(plan.metadata.get("success_contract"), dict)
                else None
            )
        if isinstance(spec, dict):
            for key in ("assertions", "expected_evidence", "evidence"):
                val = spec.get(key)
                if isinstance(val, list):
                    return val
        return []

    def _lab_job_priority(self, mission: MissionState) -> tuple[int, str]:
        try:
            if str(os.getenv("AJAX_INTERACTIVE_CHAT") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return 90, "user_waiting"
        except Exception:
            pass
        try:
            if mission.await_user_input or mission.status in {"WAITING_FOR_USER", "PAUSED_FOR_LAB"}:
                return 90, "user_waiting"
        except Exception:
            pass
        return 50, "default"

    def _map_lab_job_kind(self, *, lab_kind: str, reason: str) -> Optional[str]:
        kind = str(lab_kind or "").strip().lower()
        if kind == "probe":
            return "probe_ui"
        if kind in {"experiment", "lab_experiment"}:
            return "experiment"
        if kind in {"snap_lab", "probe_ui", "probe_notepad", "lab_notepad_smoke"}:
            return kind
        return None

    def _create_lab_job(
        self,
        mission: MissionState,
        *,
        objective: str,
        planned_steps: List[Dict[str, Any]],
        evidence_expected: List[Any],
        output_paths: List[str],
        job_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        store = getattr(self, "lab_control", None)
        if not store:
            return {}
        job_kind = str(job_kind or "").strip().lower()
        if not job_kind:
            return {}
        incident_id = None
        try:
            incident_id = (
                mission.notes.get("incident_id") if isinstance(mission.notes, dict) else None
            )
        except Exception:
            incident_id = None
        priority, priority_reason = self._lab_job_priority(mission)
        risk_level = "medium"
        try:
            gov = mission.envelope.governance if mission.envelope else None
            if gov and getattr(gov, "risk_level", None):
                risk_level = str(getattr(gov, "risk_level", "medium") or "medium").strip().lower()
        except Exception:
            risk_level = "medium"
        if mission.needs_explicit_permission:
            risk_level = "high"
        payload = {
            "mission_id": mission.mission_id,
            "incident_id": incident_id,
            "status": "QUEUED",
            "job_kind": job_kind,
            "objective": objective,
            "planned_steps": planned_steps,
            "evidence_expected": evidence_expected,
            "output_paths": output_paths,
            "priority": priority,
            "priority_reason": priority_reason,
            "risk_level": risk_level,
            "requires_ack": bool(mission.needs_explicit_permission),
        }
        record = store.enqueue_job(payload)
        try:
            mission.lab_job_id = record.get("job_id")
            if isinstance(mission.notes, dict):
                mission.notes["lab_job_id"] = record.get("job_id")
                mission.notes["lab_job_path"] = record.get("job_path")
        except Exception:
            pass
        try:
            self.state.notes["lab_job_id"] = record.get("job_id")
        except Exception:
            pass
        return record

    def _finalize_lab_handoff_wait(
        self,
        mission: MissionState,
        *,
        job_record: Dict[str, Any],
        derived: Dict[str, Any],
        reason: str,
        lab_kind: str,
        output_paths: List[str],
        artifacts: Dict[str, Any],
    ) -> AjaxExecutionResult:
        job_id = job_record.get("job_id")
        job_label = job_id or "job_desconocido"
        status = None
        payload = job_record.get("payload") if isinstance(job_record, dict) else None
        if isinstance(payload, dict):
            status = payload.get("status")
        prompt = (
            f"Derivado a LAB (job {job_label}). Espera el outcome o responde "
            f"[wait_lab] para esperar en LAB, [resume] para reintentar sin LAB, o [cancel] para cancelar."
        )
        options = [
            {"id": "wait_lab", "label": "Esperar en LAB y reanudar automaticamente"},
            {"id": "resume", "label": "Reanudar la mision sin LAB (override manual)"},
            {"id": "cancel", "label": "Cancelar la mision"},
        ]
        default_opt = "resume"
        try:
            env_default = (os.getenv("AJAX_LAB_DEFAULT_OPTION") or "").strip().lower()
            if env_default in {"wait_lab", "resume", "cancel"}:
                default_opt = env_default
            elif (os.getenv("AJAX_LAB_WAIT_DEFAULT") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                default_opt = "wait_lab"
        except Exception:
            default_opt = "resume"
        try:
            mission.waiting_cycles = int(getattr(mission, "waiting_cycles", 0) or 0) + 1
        except Exception:
            mission.waiting_cycles = 1
        mission.status = "PAUSED_FOR_LAB"
        mission.await_user_input = True
        mission.permission_question = prompt
        try:
            mission.pending_user_options = options
            if isinstance(mission.notes, dict):
                mission.notes["_pending_user_options"] = options
                mission.notes["lab_handoff_reason"] = reason
                mission.notes["lab_handoff_kind"] = lab_kind
        except Exception:
            pass
        try:
            if isinstance(mission.notes, dict):
                invoked = mission.notes.get("council_invoked")
                if invoked is None:
                    mission.notes["council_invoked"] = False
                    invoked = False
                if invoked is False and not mission.notes.get("council_invoked_reason"):
                    mission.notes["council_invoked_reason"] = f"lab_handoff:{reason}"
                    mission.notes["council_invoked_ts"] = self._iso_utc()
        except Exception:
            pass
        payload_ctx = {
            "mission_id": mission.mission_id,
            "intent": mission.intention,
            "lab_job_id": job_id,
            "lab_kind": lab_kind,
            "reason": reason,
        }
        user_payload = {
            "question": prompt,
            "context": payload_ctx,
            "options": options,
            "default": default_opt,
            "expects": "menu_choice",
        }
        waiting_path = self._persist_waiting_mission(
            mission, question=prompt, user_payload=user_payload
        )
        payload_path = (
            mission.notes.get("_waiting_payload_path") if isinstance(mission.notes, dict) else None
        )
        detail: Dict[str, Any] = {
            "mission_id": mission.mission_id,
            "lab_job_id": job_id,
            "lab_job_path": job_record.get("job_path"),
            "status": status,
            "objective": mission.intention,
            "lab_kind": lab_kind,
            "reason": reason,
            "output_paths": output_paths,
            "derived": derived,
            "question": prompt,
            "options": options,
        }
        try:
            if isinstance(mission.notes, dict):
                detail["council_invoked"] = mission.notes.get("council_invoked")
                detail["council_invoked_reason"] = mission.notes.get("council_invoked_reason")
        except Exception:
            pass
        try:
            self._enrich_detail_with_router_summary(mission, detail)
        except Exception:
            pass
        if waiting_path:
            detail["waiting_mission_path"] = waiting_path
        if payload_path:
            detail["waiting_payload_path"] = payload_path
        mission.last_result = AjaxExecutionResult(
            success=False,
            error="PAUSED_FOR_LAB",
            path="lab_handoff",
            detail=detail,
            artifacts=artifacts or None,
        )
        return mission.last_result

    def _derive_to_lab_experiment(
        self,
        mission: MissionState,
        *,
        plan: Optional[AjaxPlan],
        reason: str,
        council_verdict: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Deriva una propuesta no ejecutable (no-consenso Council) a un experimento pendiente en LAB.
        Escribe:
        - artifacts/experiments/pending/<...>.json (ExperimentEnvelope)
        - artifacts/experiments/pending/<...>_efe.json (skeleton EFE)
        - artifacts/capability_gaps/<...>_needs_lab_proof_<mission_id>.json (gap)
        """
        pending_dir = self.config.root_dir / "artifacts" / "experiments" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        ts_compact = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
        exp_id = f"exp_{ts_compact}_{mission.mission_id}"
        efe_name = f"{exp_id}_efe.json"
        env_name = f"{exp_id}.json"
        efe_path = pending_dir / efe_name
        env_path = pending_dir / env_name
        created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))

        plan_payload: Any = None
        if plan is not None:
            plan_payload = self._safe_serialize(plan)

        expected_state = self._extract_expected_state_from_plan(plan or mission.last_plan)
        if not self._expected_state_has_checks(expected_state):
            gap_path = self._emit_missing_efe_derived_gap(
                mission,
                reason="missing_expected_state_for_derived_experiment",
                plan_id=(plan.plan_id if plan else None),
            )
            return {
                "blocked": True,
                "gap_path": gap_path,
            }

        description = "Derived EFE from plan success_spec.expected_state."
        if description.strip().lower().startswith("todo"):
            gap_path = self._emit_missing_efe_derived_gap(
                mission,
                reason="derived_efe_description_todo",
                plan_id=(plan.plan_id if plan else None),
            )
            return {
                "blocked": True,
                "gap_path": gap_path,
            }

        efe_skeleton = {
            "schema_version": "0.1",
            "envelope_id": exp_id,
            "created_utc": created_utc,
            "description": description,
            "assertions": [{"expected_state": expected_state}],
            "success_spec": {"expected_state": expected_state},
        }
        try:
            efe_path.write_text(
                json.dumps(efe_skeleton, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

        envelope = {
            "schema_version": "ajax.experiment_envelope.v1",
            "id": exp_id,
            "rail": "lab",
            "kind": "experiment",
            "title": "Derived experiment (no-consensus Council)",
            "objective": mission.intention,
            "efe_path": efe_name,
            "spl": "LOW",
            "derived_from": {
                "mission_id": mission.mission_id,
                "reason": reason,
                "council_verdict": council_verdict or None,
            },
            "plan_candidate": plan_payload,
        }
        try:
            env_path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

        gap_path = self._emit_needs_lab_proof_gap(
            mission,
            reason=reason,
            experiment_envelope_path=str(env_path),
            council_verdict=council_verdict,
        )
        return {
            "experiment_id": exp_id,
            "experiment_envelope_path": str(env_path),
            "efe_path": str(efe_path),
            "gap_path": gap_path,
        }

    def _derive_to_lab_probe(
        self,
        mission: MissionState,
        *,
        reason: str,
        detail: Optional[Dict[str, Any]] = None,
        lab_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        store = getattr(self, "lab_control", None)
        if not store:
            return {}
        plan_payload = self._safe_serialize(mission.last_plan)
        success_contract = None
        if mission.last_plan and getattr(mission.last_plan, "success_spec", None):
            success_contract = self._safe_serialize(mission.last_plan.success_spec)
        mission_log = None
        try:
            mission_log = mission.notes.get("last_mission_log")
        except Exception:
            mission_log = None
        payload = {
            "status": "pending",
            "ttl_seconds": DEFAULT_PROBE_TTL_SECONDS,
            "origin": {
                "mission_id": mission.mission_id,
                "intent": mission.intention,
                "rail": self._current_rail(),
                "reason": reason,
                "detail": detail or {},
                "lab_job_id": lab_job_id,
                "plan_snapshot": plan_payload,
                "success_contract": success_contract,
                "mission_log": mission_log,
            },
        }
        record = store.create_probe(payload, ttl_seconds=DEFAULT_PROBE_TTL_SECONDS)
        try:
            mission.notes["lab_probe_path"] = record.get("probe_path")
            mission.notes["lab_probe_id"] = record.get("probe_id")
        except Exception:
            pass
        return record

    def _handoff_to_lab(
        self,
        mission: MissionState,
        *,
        lab_kind: str,
        reason: str,
        plan: Optional[AjaxPlan] = None,
        council_verdict: Optional[Dict[str, Any]] = None,
        detail: Optional[Dict[str, Any]] = None,
        extra_output_paths: Optional[List[str]] = None,
        extra_artifacts: Optional[Dict[str, Any]] = None,
    ) -> AjaxExecutionResult:
        try:
            if isinstance(mission.notes, dict) and mission.notes.get(
                "lab_unsupported_kind_skip_once"
            ):
                mission.notes["lab_unsupported_kind_skip_once"] = False
                return AjaxExecutionResult(
                    success=False,
                    error="LAB_UNSUPPORTED_KIND",
                    path="lab_handoff",
                    detail={
                        "mission_id": mission.mission_id,
                        "reason": "lab_unsupported_kind_skip_once",
                    },
                )
        except Exception:
            pass
        try:
            blocked_until = 0.0
            if isinstance(mission.notes, dict):
                blocked_until = float(mission.notes.get("lab_job_kind_blocked_until") or 0)
            if blocked_until and time.time() < blocked_until:
                mission.status = "PAUSED_BY_USER"
                mission.last_result = AjaxExecutionResult(
                    success=False,
                    error="LAB_UNSUPPORTED_KIND_THROTTLED",
                    path="lab_handoff",
                    detail={
                        "mission_id": mission.mission_id,
                        "blocked_until": blocked_until,
                        "gap_path": mission.notes.get("lab_job_kind_gap_path")
                        if isinstance(mission.notes, dict)
                        else None,
                    },
                )
                return mission.last_result
        except Exception:
            pass
        planned_steps = self._summarize_plan_steps(plan or mission.last_plan)
        evidence_expected = self._extract_expected_evidence(plan or mission.last_plan)
        output_paths: List[str] = []
        artifacts: Dict[str, Any] = {}
        derived: Dict[str, Any] = {}
        job_record: Dict[str, Any] = {}
        store = getattr(self, "lab_control", None)
        job_kind = self._map_lab_job_kind(lab_kind=lab_kind, reason=reason)
        if not job_kind:
            gap_path = self._emit_lab_job_kind_unsupported_gap(
                mission, lab_kind=lab_kind, reason=reason
            )
            try:
                self._record_lab_queue_receipt(
                    mission_id=mission.mission_id,
                    job_kind="",
                    validated=False,
                    reason="job_kind_unmapped",
                )
            except Exception:
                pass
            mission.status = "FAILED"
            if MissionError:
                mission.last_mission_error = MissionError(
                    kind="plan_error", step_id=None, reason="LAB_UNSUPPORTED_KIND"
                )
            mission.last_result = AjaxExecutionResult(
                success=False,
                error="LAB_UNSUPPORTED_KIND",
                path="lab_handoff",
                detail={
                    "mission_id": mission.mission_id,
                    "lab_kind": lab_kind,
                    "reason": reason,
                    "gap_path": gap_path,
                },
            )
            return mission.last_result
        else:
            try:
                self._record_lab_queue_receipt(
                    mission_id=mission.mission_id,
                    job_kind=job_kind,
                    validated=True,
                    reason="job_kind_mapped",
                )
            except Exception:
                pass

        if store and mission.lab_job_id:
            try:
                existing_job, job_path = store.load_job(mission.lab_job_id)
                status = str(existing_job.get("status") or "").upper()
                if status in {"QUEUED", "RUNNING"}:
                    output_paths = list(existing_job.get("output_paths") or [])
                    if extra_output_paths:
                        try:
                            store.update_job_status(
                                mission.lab_job_id,
                                status=status,
                                output_paths=extra_output_paths,
                            )
                            for path in extra_output_paths:
                                if path and path not in output_paths:
                                    output_paths.append(path)
                        except Exception:
                            pass
                    job_record = {
                        "job_id": existing_job.get("job_id") or mission.lab_job_id,
                        "job_path": str(job_path),
                        "payload": existing_job,
                    }
                    artifacts["lab_job"] = str(job_path)
                    if output_paths:
                        artifacts["lab_outputs"] = output_paths
                    if extra_artifacts:
                        for key, value in extra_artifacts.items():
                            if key and value is not None:
                                artifacts[key] = value
                    mission.pending_plan = None
                    return self._finalize_lab_handoff_wait(
                        mission,
                        job_record=job_record,
                        derived=derived,
                        reason=reason,
                        lab_kind=lab_kind,
                        output_paths=output_paths,
                        artifacts=artifacts,
                    )
            except Exception:
                pass

        if lab_kind == "probe":
            job_record = self._create_lab_job(
                mission,
                objective=mission.intention,
                planned_steps=planned_steps,
                evidence_expected=evidence_expected,
                output_paths=[],
                job_kind=job_kind,
            )
            job_id = job_record.get("job_id")
            derived = self._derive_to_lab_probe(
                mission,
                reason=reason,
                detail=detail,
                lab_job_id=job_id,
            )
            if derived.get("probe_path"):
                output_paths.append(str(derived.get("probe_path")))
                artifacts["lab_probe"] = derived.get("probe_path")
            if derived.get("probe_id"):
                artifacts["lab_probe_id"] = derived.get("probe_id")
            if job_id and store and output_paths:
                try:
                    store.update_job_status(job_id, status="QUEUED", output_paths=output_paths)
                except Exception:
                    pass
        else:
            derived = self._derive_to_lab_experiment(
                mission,
                plan=plan,
                reason=reason,
                council_verdict=council_verdict,
            )
            if derived.get("blocked"):
                gap_path = derived.get("gap_path")
                mission.status = "BLOCKED"
                mission.last_result = AjaxExecutionResult(
                    success=False,
                    error="missing_efe_derived",
                    path="lab_handoff",
                    detail={
                        "mission_id": mission.mission_id,
                        "reason": reason,
                        "gap_path": gap_path,
                    },
                    artifacts={"capability_gap": gap_path} if gap_path else None,
                )
                return mission.last_result
            for key in ("experiment_envelope_path", "efe_path", "gap_path"):
                if derived.get(key):
                    output_paths.append(str(derived.get(key)))
            job_record = self._create_lab_job(
                mission,
                objective=mission.intention,
                planned_steps=planned_steps,
                evidence_expected=evidence_expected,
                output_paths=output_paths,
                job_kind=job_kind,
            )
            if derived.get("experiment_envelope_path"):
                artifacts["experiment_envelope"] = derived.get("experiment_envelope_path")
            if derived.get("efe_path"):
                artifacts["efe"] = derived.get("efe_path")
            if derived.get("gap_path"):
                artifacts["capability_gap"] = derived.get("gap_path")

        if job_record.get("job_path"):
            artifacts["lab_job"] = job_record.get("job_path")
        if extra_output_paths:
            for path in extra_output_paths:
                if path and path not in output_paths:
                    output_paths.append(path)
        if extra_artifacts:
            for key, value in extra_artifacts.items():
                if key and value is not None:
                    artifacts[key] = value
        if output_paths:
            artifacts["lab_outputs"] = output_paths
        mission.pending_plan = None
        return self._finalize_lab_handoff_wait(
            mission,
            job_record=job_record,
            derived=derived,
            reason=reason,
            lab_kind=lab_kind,
            output_paths=output_paths,
            artifacts=artifacts,
        )

    def run_lab_probe_recipe(
        self, probe_payload: Dict[str, Any], *, dry_run: bool = False
    ) -> AjaxExecutionResult:
        outputs = probe_payload.get("outputs") or {}
        recipe = outputs.get("recipe") or {}
        plan_dict = recipe.get("plan")
        if not isinstance(plan_dict, dict):
            raise ValueError("Lab probe recipe must include a plan JSON object.")
        plan = self._plan_from_json(plan_dict)
        plan.metadata = plan.metadata or {}
        plan.metadata["source"] = "lab_probe"
        plan.metadata["lab_probe_id"] = probe_payload.get("probe_id")
        if "success_contract" in recipe and not getattr(plan, "success_spec", None):
            plan.success_spec = recipe.get("success_contract")
        intent = (probe_payload.get("origin") or {}).get("intent") or "LAB_PROBE_RECIPE"
        mode = "dry" if dry_run else "auto"
        return self.do(intent, mode=mode, override_plan=plan)

    def _preflight_starting_xi(
        self, mission: MissionState
    ) -> Tuple[Optional[Dict[str, Any]], Optional[AjaxExecutionResult]]:
        """
        Paso 0 (Starting XI): antes de plan/act, verificar providers+inventario y elegir titulares+banquillo por rol.
        Si falta quorum -> soft abort + gap accionable.
        """
        rail = os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"
        if validate_policy_contract is None:
            res = AjaxExecutionResult(
                success=False,
                error="BLOCKED_BY_POLICY_CONTRACT_UNAVAILABLE",
                path="preflight",
                detail={
                    "reason": "policy_contract_validator_unavailable",
                    "terminal_status": "BLOCKED",
                },
            )
            return None, res
        try:
            policy_contract = validate_policy_contract(
                self.config.root_dir,
                sync_json=True,
                write_receipt=True,
            )
        except Exception as exc:
            res = AjaxExecutionResult(
                success=False,
                error="BLOCKED_BY_POLICY_CONTRACT_FAILED",
                path="preflight",
                detail={
                    "reason": "policy_contract_exception",
                    "error": str(exc)[:200],
                    "terminal_status": "BLOCKED",
                },
            )
            return None, res
        if isinstance(mission.notes, dict):
            mission.notes["policy_contract"] = policy_contract.to_dict()
        if not policy_contract.ok:
            res = AjaxExecutionResult(
                success=False,
                error="BLOCKED_BY_POLICY_CONFIG_MISSING",
                path="preflight",
                detail={
                    "reason": policy_contract.reason,
                    "policy_contract": policy_contract.to_dict(),
                    "terminal_status": "BLOCKED",
                },
                artifacts={"policy_contract_receipt": policy_contract.receipt_path}
                if policy_contract.receipt_path
                else None,
            )
            return None, res
        if run_anchor_preflight is None:
            res = AjaxExecutionResult(
                success=False,
                error="BLOCKED_BY_ANCHOR_GATE_UNAVAILABLE",
                path="preflight",
                detail={
                    "reason": "anchor_preflight_unavailable",
                    "terminal_status": "BLOCKED",
                },
            )
            return None, res
        try:
            anchor_gate = run_anchor_preflight(
                root_dir=self.config.root_dir,
                rail=rail,
                write_receipt=True,
            )
        except Exception as exc:
            anchor_gate = {
                "ok": False,
                "status": "BLOCKED",
                "reason": f"anchor_preflight_exception:{str(exc)[:120]}",
            }
        if isinstance(mission.notes, dict):
            mission.notes["anchor_preflight"] = anchor_gate
        if not bool(anchor_gate.get("ok")):
            res = AjaxExecutionResult(
                success=False,
                error="BLOCKED_BY_ANCHOR_MISMATCH",
                path="preflight",
                detail={
                    "reason": anchor_gate.get("reason") or "anchor_mismatch",
                    "anchor_preflight": anchor_gate,
                    "terminal_status": "BLOCKED",
                },
                artifacts={"anchor_receipt": anchor_gate.get("receipt_path")}
                if anchor_gate.get("receipt_path")
                else None,
            )
            return None, res
        if microfilm_enforce_lab_prod_separation is not None:
            try:
                microfilm_ctx = {
                    "rail": rail,
                    "display_target": self._resolve_display_target_label(str(rail)),
                    "human_active": self._read_human_active_flag(),
                    "require_display_target": True,
                    "anchor_mismatches": [
                        *((anchor_gate.get("mismatches") or []) if isinstance(anchor_gate, dict) else []),
                        *((anchor_gate.get("warnings") or []) if isinstance(anchor_gate, dict) else []),
                    ],
                }
                rail_gate = microfilm_enforce_lab_prod_separation(microfilm_ctx)
            except Exception as exc:
                rail_gate = {
                    "ok": False,
                    "status": "BLOCKED",
                    "code": "BLOCKED_RAIL_MISMATCH",
                    "actionable_hint": f"microfilm_guard_exception:{str(exc)[:120]}",
                }
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["microfilm_lab_prod"] = rail_gate
            except Exception:
                pass
            if not bool(rail_gate.get("ok")):
                res = AjaxExecutionResult(
                    success=False,
                    error=str(rail_gate.get("code") or "BLOCKED_RAIL_MISMATCH"),
                    path="preflight",
                    detail={
                        "reason": "microfilm_lab_prod_separation_failed",
                        "microfilm_guard": rail_gate,
                        "terminal_status": "BLOCKED",
                    },
                )
                return None, res
        if build_starting_xi is None:
            res = AjaxExecutionResult(
                success=False,
                error="BLOCKED_BY_MISSING_PLAYERS",
                path="preflight",
                detail={"reason": "starting_xi_module_missing"},
            )
            return None, res
        risk_level = "medium"
        try:
            gov = mission.envelope.governance if mission.envelope else None  # type: ignore[union-attr]
            risk_level = str(getattr(gov, "risk_level", "medium") or "medium")
        except Exception:
            risk_level = "medium"
        cost_mode = mission.cost_mode or os.getenv("AJAX_COST_MODE", "premium")
        if self.ledger:
            try:
                self.ledger.refresh()
            except Exception as exc:
                self.log.warning("Ledger refresh failed in preflight: %s", exc)
        sx = build_starting_xi(
            root_dir=self.config.root_dir,
            provider_configs=self.provider_configs,
            rail=rail,
            risk_level=risk_level,
            cost_mode=cost_mode,
            vision_required=False,
        )
        try:
            if not mission.cost_mode:
                mission.cost_mode = str(cost_mode).strip().lower()
        except Exception:
            pass
        try:
            mission.notes["starting_xi"] = sx
        except Exception:
            pass
        if mission.envelope:
            try:
                mission.envelope.metadata["starting_xi_path"] = sx.get("path")
            except Exception:
                pass
            try:
                mission.envelope.metadata["starting_xi"] = sx
            except Exception:
                pass

        # Consola (4 líneas): por defecto siempre; configurable con AJAX_STARTING_XI_PRINT=always|degraded|never
        print_mode = (os.getenv("AJAX_STARTING_XI_PRINT") or "always").strip().lower()
        if print_mode not in {"always", "degraded", "never"}:
            print_mode = "always"
        should_print = print_mode == "always"
        if print_mode == "degraded":
            if sx.get("missing_players"):
                should_print = True
            else:
                try:
                    pre = sx.get("preflight") or {}
                    should_print = not bool((pre.get("breathing") or {}).get("ok")) or not bool(
                        (pre.get("inventory") or {}).get("ok")
                    )
                except Exception:
                    should_print = True
        if should_print and format_console_lines is not None:
            try:
                for ln in format_console_lines(sx):
                    print(ln)
            except Exception:
                pass

        slots_missing = []
        try:
            slots_missing = self._slots_missing_entries()
            sx["slots_missing"] = slots_missing
        except Exception:
            slots_missing = []

        missing = sx.get("missing_players") or []
        if isinstance(missing, list) and missing:
            try:
                print("missing_players=" + json.dumps(missing, ensure_ascii=False))
            except Exception:
                pass
            if slots_missing:
                try:
                    print("slots_missing=" + json.dumps(slots_missing, ensure_ascii=False))
                except Exception:
                    pass
            # Degraded Mode (constitucional): si SOLO falta Council.role2 por quorum=2,
            # permitir quorum efectivo=1 en LAB o intents low-risk (o override explícito).
            try:
                rail_n = str(sx.get("rail") or rail).strip().lower()
                risk_n = str(sx.get("risk_level") or risk_level).strip().lower()
                required = int(((sx.get("quorum") or {}).get("council_required")) or 0)
            except Exception:
                rail_n = str(rail or "lab").strip().lower()
                risk_n = str(risk_level or "medium").strip().lower()
                required = 0

            only_role2 = True
            for m in missing:
                if not isinstance(m, dict):
                    continue
                if str(m.get("role") or "") != "council.role2":
                    only_role2 = False
                    break
            allow_quorum_degrade = (
                (rail_n not in {"prod", "production", "live"})
                or (risk_n == "low")
                or (os.getenv("AJAX_ALLOW_COUNCIL_QUORUM_DEGRADE") or "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            if only_role2 and allow_quorum_degrade and required >= 2:
                try:
                    sx["missing_players"] = []
                    sx.setdefault("quorum", {})
                    sx["quorum"]["council_effective"] = 1
                    sx.setdefault("policy", {})
                    sx["policy"]["council_quorum_degraded"] = True
                    sx["policy"]["council_quorum_from"] = required
                    sx["policy"]["council_quorum_to"] = 1
                    # Persistir el Starting XI actualizado (best-effort) para auditoría.
                    sx_path = sx.get("path")
                    if isinstance(sx_path, str) and sx_path:
                        Path(sx_path).write_text(
                            json.dumps(sx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                        )
                except Exception:
                    pass
                try:
                    if mission.envelope:
                        mission.envelope.metadata["council_quorum_degraded"] = True
                        mission.envelope.metadata["council_quorum_effective"] = 1
                except Exception:
                    pass
                self._emit_council_quorum_degraded_gap(
                    mission,
                    starting_xi=sx,
                    quorum_from=required or 2,
                    quorum_to=1,
                    reason="missing_council_role2_quorum",
                )
                return sx, None

            gap_path = self._emit_missing_players_gap(mission, sx) or sx.get("path")
            detail = {
                "missing_players": missing,
                "slots_missing": slots_missing,
                "slots_missing_count": len(slots_missing),
                "fix_hints": sx.get("fix_hints"),
                "starting_xi_path": sx.get("path"),
                "gap_path": gap_path,
            }
            # ProbePlan local (no LLM) para dejar evidencia de infra aunque falten jugadores cloud.
            try:
                probe_res = self._run_probe_driver_plan(mission, reason="preflight_missing_players")
                if probe_res:
                    detail["probe_driver"] = (mission.notes or {}).get("probe_driver") or probe_res
            except Exception:
                pass
            res = AjaxExecutionResult(
                success=False,
                error="BLOCKED_BY_MISSING_PLAYERS",
                path="preflight",
                detail=detail,
                artifacts={
                    "starting_xi": sx.get("path"),
                    "capability_gap": gap_path,
                    "probe_driver": ((mission.notes or {}).get("probe_driver") or {}).get(
                        "report_path"
                    )
                    if isinstance(mission.notes, dict)
                    else None,
                }
                if gap_path
                else {
                    "starting_xi": sx.get("path"),
                    "probe_driver": ((mission.notes or {}).get("probe_driver") or {}).get(
                        "report_path"
                    )
                    if isinstance(mission.notes, dict)
                    else None,
                },
            )
            return sx, res
        return sx, None

    def _run_probe_driver_plan(
        self, mission: MissionState, *, reason: str
    ) -> Optional[Dict[str, Any]]:
        """
        ProbePlan local sin LLM: ejecuta action `probe_driver` via plan_runner y deja evidencia durable.
        """
        if AgencyJob is None or run_job_plan is None:
            return None
        try:
            out_dir = (
                Path(self.config.root_dir) / "artifacts" / "driver_probe" / str(mission.mission_id)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            out_dir = Path("artifacts") / "driver_probe" / str(mission.mission_id)
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        report_path = out_dir / "probe_report.json"
        step = {
            "id": "probe-driver",
            "intent": f"Probe driver ({reason})",
            "preconditions": {"expected_state": {}},
            "action": "probe_driver",
            "args": {"out_dir": str(out_dir)},
            "evidence_required": ["driver.ui_inspect", "driver.screenshot"],
            "success_spec": {
                "expected_state": {
                    "files": [{"path": str(report_path), "must_exist": True}],
                }
            },
            "on_fail": "abort",
        }
        job = AgencyJob(
            job_id=f"probe_driver_{mission.mission_id}",
            goal="PROBE_DRIVER",
            metadata={
                "mission_id": mission.mission_id,
                "attempt": "probe_driver",
                "reason": reason,
                "steps": [step],
            },
        )
        try:
            res = run_job_plan(job_obj=job, actuator=self.actuator)  # type: ignore[call-arg]
        except TypeError:
            res = run_job_plan(job, actuator=self.actuator)
        except Exception as exc:
            res = {
                "ok": False,
                "error": f"probe_driver_exception:{str(exc)[:200]}",
                "job_id": job.job_id,
                "goal": job.goal,
            }
        try:
            mission.notes["probe_driver"] = {
                "reason": reason,
                "out_dir": str(out_dir),
                "report_path": str(report_path),
                "result": res,
            }
        except Exception:
            pass
        return res if isinstance(res, dict) else None

    def _normalize_council_verdict(
        self, verdict: Optional["CouncilVerdict"]
    ) -> Optional["CouncilVerdict"]:
        if verdict is None:
            return None
        try:
            reason = str(getattr(verdict, "reason", "") or "").strip()
            if not verdict.approved and not reason:
                verdict.reason = "BLOCKED_BY_COUNCIL_INVALID_REVIEW:empty_reason"
                verdict.escalation_hint = "blocked"
        except Exception:
            return verdict
        return verdict

    def _record_council_invocation(
        self, mission: Optional[MissionState], *, invoked: bool, reason: str
    ) -> None:
        if mission is None:
            return
        try:
            notes = mission.notes if isinstance(mission.notes, dict) else {}
            notes["council_invoked"] = bool(invoked)
            notes["council_invoked_reason"] = str(reason or "unknown")
            notes["council_invoked_ts"] = self._iso_utc()
            mission.notes = notes
        except Exception:
            pass
        if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
            try:
                mission.envelope.metadata["council_invoked"] = bool(invoked)
                mission.envelope.metadata["council_invoked_reason"] = str(reason or "unknown")
                mission.envelope.metadata["council_invoked_ts"] = self._iso_utc()
            except Exception:
                pass

    def _step_from_council_signal(self, mission: MissionState) -> Optional[MissionStep]:
        if not mission.council_signal:
            return None
        try:
            cv = CouncilVerdict(
                verdict=mission.council_signal.verdict or "reject",
                reason=mission.council_signal.reason or "",
                suggested_fix=mission.council_signal.suggested_fix,
                escalation_hint=mission.council_signal.escalation_hint,
                debug_notes=None,
            )
        except Exception:
            return None
        cv = self._normalize_council_verdict(cv)
        hint = (cv.escalation_hint or "").strip()
        if cv.approved:
            return None
        if hint == "await_user_input":
            question = (
                cv.suggested_fix
                or cv.reason
                or "Necesito que confirmes exactamente qué quieres antes de continuar."
            )
            return MissionStep(kind="ASK_USER", question=question)
        if hint == "try_stronger_model":
            return MissionStep(kind="UPGRADE_MODEL", next_model=self._pick_stronger_model(mission))
        if hint == "blocked":
            # Política: no-consenso -> LAB, salvo falta de quorum real (ledger/providers caídos).
            council_verdict = None
            try:
                council_verdict = (
                    mission.last_plan.metadata.get("council_verdict")
                    if mission.last_plan and isinstance(mission.last_plan.metadata, dict)
                    else None
                )
            except Exception:
                council_verdict = None
            # Si es un bloqueo técnico (quorum/providers), no derivar a experimento por defecto.
            reason_txt = str(cv.reason or "")
            invalid_review = reason_txt.startswith("BLOCKED_BY_COUNCIL_INVALID_REVIEW")
            is_quorum_block = reason_txt.startswith("BLOCKED_BY_COUNCIL_QUORUM")
            try:
                if isinstance(council_verdict, dict):
                    dr = str(council_verdict.get("council_degraded_reason") or "")
                    if dr.startswith("ledger_"):
                        is_quorum_block = True
            except Exception:
                pass
            if invalid_review:
                incident_id = self._open_council_contract_invalid_incident(
                    mission,
                    reason=reason_txt,
                    council_verdict=council_verdict
                    if isinstance(council_verdict, dict)
                    else cv.__dict__,
                )
                mission.last_result = self._handoff_to_lab(
                    mission,
                    lab_kind="experiment",
                    reason="council_contract_invalid",
                    plan=mission.last_plan,
                    council_verdict=council_verdict
                    if isinstance(council_verdict, dict)
                    else cv.__dict__,
                    detail={
                        "incident_id": incident_id,
                        "reason": reason_txt,
                        "council_verdict": council_verdict
                        if isinstance(council_verdict, dict)
                        else cv.__dict__,
                    },
                )
                return MissionStep(kind="GIVE_UP", final_status="PAUSED_FOR_LAB")
            if is_quorum_block:
                # ProbePlan local (no LLM) para producir EFE/evidencia antes de pedir intervención humana.
                probe_report_path = None
                try:
                    self._run_probe_driver_plan(mission, reason="council_quorum_block")
                    probe_note = (
                        (mission.notes or {}).get("probe_driver")
                        if isinstance(mission.notes, dict)
                        else None
                    )
                    if isinstance(probe_note, dict):
                        probe_report_path = probe_note.get("report_path")
                except Exception:
                    probe_report_path = None

                # GAP accionable + evidencia (starting_xi / provider_ledger / probe_driver)
                try:
                    cv_full = cv
                    if isinstance(council_verdict, dict):
                        try:
                            cv_full = CouncilVerdict(**council_verdict)
                        except Exception:
                            cv_full = cv
                    self._emit_council_block_gap(mission, cv_full)
                except Exception:
                    pass
                sx_path = None
                ledger_path = None
                try:
                    sx_path = (
                        (mission.envelope.metadata or {}).get("starting_xi_path")
                        if mission.envelope
                        else None
                    )
                except Exception:
                    sx_path = None
                try:
                    sx = (
                        (mission.envelope.metadata or {}).get("starting_xi")
                        if mission.envelope
                        else None
                    )
                    if isinstance(sx, dict):
                        ledger_path = (
                            ((sx.get("inputs") or {}).get("provider_ledger_path"))
                            if isinstance(sx.get("inputs"), dict)
                            else None
                        )
                except Exception:
                    ledger_path = None
                question = (
                    "[COUNCIL_QUORUM] No hay quorum real del Council (providers en cooldown/caídos).\n"
                    "Acciones posibles:\n"
                    "1) Esperar/reintentar cuando pase el cooldown (ver ledger).\n"
                    "2) Reparar credenciales/CLI del provider (auth/bridge).\n"
                    "3) (Opcional) Autorizar quorum=1 exportando `AJAX_ALLOW_COUNCIL_QUORUM_DEGRADE=1` y reintentar.\n"
                )
                if ledger_path:
                    question += f"\n- ledger: {ledger_path}"
                if sx_path:
                    question += f"\n- starting_xi: {sx_path}"
                if isinstance(probe_report_path, str) and probe_report_path:
                    question += f"\n- probe_driver: {probe_report_path}"
                return MissionStep(kind="ASK_USER", question=question)
            normalized_reason = self._normalize_reason_key(cv.reason or "")
            if self._should_launch_lab_probe(mission, normalized_reason):
                mission.last_result = self._handoff_to_lab(
                    mission,
                    lab_kind="probe",
                    reason=normalized_reason or "council_blocked",
                    plan=mission.last_plan,
                    detail={"council_verdict": council_verdict or cv.__dict__},
                )
                return MissionStep(kind="GIVE_UP", final_status="PAUSED_FOR_LAB")
            mission.last_result = self._handoff_to_lab(
                mission,
                lab_kind="experiment",
                reason=cv.reason or "council_blocked",
                plan=mission.last_plan,
                council_verdict=council_verdict
                if isinstance(council_verdict, dict)
                else cv.__dict__,
            )
            return MissionStep(kind="GIVE_UP", final_status="PAUSED_FOR_LAB")
        if hint == "unsafe":
            mission.status = "IRRECOVERABLY_UNSAFE"
            return MissionStep(kind="GIVE_UP", final_status="IRRECOVERABLY_UNSAFE")
        return None

    def _budget_exhausted_step(self, mission: MissionState) -> Optional[MissionStep]:
        retry_requested = False
        try:
            if isinstance(mission.notes, dict):
                retry_requested = bool(mission.notes.get("retry_escalate_requested"))
        except Exception:
            retry_requested = False
        question = (
            "[BUDGET_EXHAUSTED] Presupuesto de planificación agotado.\n"
            "Indica cómo quieres proceder (reintento manual, abrir INCIDENT o derivar a LAB)."
        )
        if retry_requested and getattr(self, "lab_control", None):
            gap_path = self._emit_budget_exhausted_gap(
                mission,
                reason="retry_escalate_brain",
                detail={
                    "plan_attempts": mission.plan_attempts,
                    "max_attempts": mission.max_attempts,
                },
            )
            extra_outputs = [gap_path] if gap_path else []
            extra_artifacts = {"budget_exhausted_gap": gap_path} if gap_path else {}
            mission.last_result = self._handoff_to_lab(
                mission,
                lab_kind="experiment",
                reason="budget_exhausted_retry",
                plan=mission.last_plan,
                detail={
                    "budget_exhausted": True,
                    "plan_attempts": mission.plan_attempts,
                    "max_attempts": mission.max_attempts,
                },
                extra_output_paths=extra_outputs,
                extra_artifacts=extra_artifacts,
            )
            try:
                if isinstance(mission.notes, dict):
                    mission.notes.pop("retry_escalate_requested", None)
            except Exception:
                pass
            return MissionStep(kind="GIVE_UP", final_status="PAUSED_FOR_LAB")
        try:
            if isinstance(mission.notes, dict):
                mission.notes["budget_exhausted"] = True
        except Exception:
            pass
        mission.await_user_input = True
        mission.permission_question = question
        return MissionStep(kind="ASK_USER", question=question)

    def choose_next_step(self, mission: MissionState) -> MissionStep:
        """
        Oráculo de siguiente paso. Mientras exista una opción segura para avanzar,
        no devuelve GIVE_UP.
        """
        now = time.time()
        if mission.retry_after and mission.retry_after > now:
            wait_seconds = int(mission.retry_after - now) or 1
            return MissionStep(kind="WAIT_AND_RETRY", seconds=wait_seconds)

        if mission.await_user_input:
            return MissionStep(kind="ASK_USER", question=mission.permission_question)

        council_step = self._step_from_council_signal(mission)
        if council_step:
            return council_step

        if mission.infra_issue and mission.infra_issue.recoverable:
            return MissionStep(
                kind="FIX_INFRA",
                component=mission.infra_issue.component,
                retry_in=mission.infra_issue.retry_in,
            )

        next_action = mission.next_pending_action()
        if next_action is not None:
            return MissionStep(kind="EXECUTE_ACTION", action=next_action)
        if mission.plan_attempts < mission.max_attempts:
            return MissionStep(kind="EXECUTE_ACTION", action=next_action)
        budget_step = self._budget_exhausted_step(mission)
        if budget_step:
            return budget_step

        if mission.needs_explicit_permission and not mission.permission_granted:
            q = mission.permission_question or "¿Autorizas continuar?"
            return MissionStep(kind="REQUEST_PERMISSION", question=q)

        if mission.user_cancelled:
            return MissionStep(kind="GIVE_UP", final_status="ABORTED_BY_USER")

        if self._technically_impossible(mission):
            return MissionStep(kind="GIVE_UP", final_status="TECHNICALLY_IMPOSSIBLE")

        return MissionStep(kind="GIVE_UP", final_status="TECHNICALLY_IMPOSSIBLE")

    def pursue_intent(
        self,
        mission: MissionState,
        preselected_plan: Optional[Dict[str, Any]] = None,
        plan_source: Optional[str] = None,
        plan_scores: Optional[Dict[str, float]] = None,
    ) -> AjaxExecutionResult:
        """
        Núcleo del mayordomo: mientras exista una opción segura, la misión no muere.
        """
        start_ts = time.time()
        plan_timeout_start: Optional[float] = None
        # Timeout total por misión (configurable). Si es demasiado agresivo, corta el "pursuit" antes
        # de que el sistema pueda replanear/fallback.
        mission_timeout = 0
        try:
            mission_timeout = int(os.getenv("AJAX_MISSION_TIMEOUT_SEC", "0") or 0)
        except Exception:
            mission_timeout = 0
        if mission_timeout <= 0:
            try:
                mission_timeout = int(getattr(self, "mission_retry_budget_seconds", 0) or 0)
            except Exception:
                mission_timeout = 0
        if mission_timeout <= 0:
            # Preferir budget_seconds si viene en governance.
            budget_seconds = 0
            try:
                gov = mission.envelope.governance if mission.envelope else None  # type: ignore[union-attr]
                budget_seconds = (
                    int(getattr(gov, "budget_seconds", 0) or 0) if gov is not None else 0
                )
            except Exception:
                budget_seconds = 0
            mission_timeout = budget_seconds if budget_seconds > 0 else 120
            # PROD/high pueden necesitar más tiempo por quorum + providers lentos.
            try:
                rail = (
                    (
                        os.getenv("AJAX_RAIL")
                        or os.getenv("AJAX_ENV")
                        or os.getenv("AJAX_MODE")
                        or "lab"
                    )
                    .strip()
                    .lower()
                )
                risk = ""
                try:
                    gov = mission.envelope.governance if mission.envelope else None  # type: ignore[union-attr]
                    risk = (
                        str(getattr(gov, "risk_level", "") or "").strip().lower()
                        if gov is not None
                        else ""
                    )
                except Exception:
                    risk = ""
                if rail in {"prod", "production", "live"} or risk == "high":
                    mission_timeout = max(mission_timeout, 180)
                elif risk == "medium":
                    mission_timeout = max(mission_timeout, 150)
            except Exception:
                pass
        mission_timeout = max(30, min(int(mission_timeout), 900))
        # Si hay plan preseleccionado, usarlo para el primer intento
        if preselected_plan:
            try:
                mission.last_plan = self._plan_from_json(preselected_plan)
                mission.pending_plan = mission.last_plan
                if mission.last_plan.metadata is not None:
                    mission.last_plan.metadata["plan_source"] = plan_source or "preselected"
                    if plan_scores:
                        mission.last_plan.metadata["plan_scores"] = plan_scores
                if mission.last_plan and mission.last_plan.steps and plan_timeout_start is None:
                    plan_timeout_start = time.time()
            except Exception as exc:
                self.log.warning("No se pudo construir plan preseleccionado: %s", exc)

        while mission.status not in TERMINAL_STATES:
            step_origin: Optional[str] = None
            step_reason: Optional[str] = None
            step: Optional[MissionStep] = None
            if mission.council_signal:
                forced_step = self._step_from_council_signal(mission)
                step_reason = mission.council_signal.reason if mission.council_signal else None
                mission.council_signal = None
                if forced_step:
                    step = forced_step
                    step_origin = "council"
            if plan_timeout_start is None and mission.last_plan and mission.last_plan.steps:
                plan_timeout_start = time.time()
            if (
                step is None
                and plan_timeout_start is not None
                and time.time() - plan_timeout_start > mission_timeout
            ):
                vetoes = None
                try:
                    vetoes = (mission.notes or {}).get("council_vetoes")
                except Exception:
                    vetoes = None
                if isinstance(vetoes, list) and vetoes:
                    gap_path = None
                    try:
                        gap_path = self._emit_council_veto_timeout_gap(
                            mission, timeout_seconds=mission_timeout
                        )
                    except Exception:
                        gap_path = None
                    mission.status = "BLOCKED_BY_COUNCIL_VETO"
                    mission.last_result = AjaxExecutionResult(
                        success=False,
                        error="BLOCKED_BY_COUNCIL_VETO",
                        path="council",
                        detail={
                            "timeout_seconds": mission_timeout,
                            "council_vetoes": vetoes[-3:],
                            "gap_path": gap_path,
                        },
                        artifacts={"capability_gap": gap_path} if gap_path else None,
                    )
                else:
                    mission.status = "TIMED_OUT"
                    mission.last_result = AjaxExecutionResult(
                        success=False,
                        error="mission_timeout",
                        path="mission",
                        detail={"timeout_seconds": mission_timeout},
                    )
                break
            if step is None:
                step = self.choose_next_step(mission)
            match step.kind:
                case "EXECUTE_ACTION":
                    err = self._execute_action_step(mission, step)
                    if err == _AWAIT_USER_SENTINEL or mission.status == "WAITING_FOR_USER":
                        return mission.last_result or AjaxExecutionResult(
                            success=False, error="await_user_input", path="waiting_for_user"
                        )
                    if err:
                        self._register_error(mission, err)
                        if mission.status in TERMINAL_STATES:
                            break
                        continue
                case "ASK_USER":
                    decision = self._handle_ask_user_step(mission, step.question)
                    if decision == "plan":
                        continue
                    return self._finalize_ask_user_wait(
                        mission,
                        step.question,
                        source=step_origin,
                        blocking_reason=step_reason,
                    )
                case "UPGRADE_MODEL":
                    self._switch_brain_model(mission, step.next_model)
                    mission.council_signal = None
                case "FIX_INFRA":
                    self._attempt_infra_fix(mission, step)
                case "WAIT_AND_RETRY":
                    time.sleep(step.seconds or step.retry_in or 2)
                case "REQUEST_PERMISSION":
                    self._request_explicit_permission(mission, step)
                case "LEARN_FROM_USER":
                    self._update_memory_from_user(mission, step)
                case "GIVE_UP":
                    mission.status = step.final_status or "TECHNICALLY_IMPOSSIBLE"
                    if not mission.last_result:
                        mission.last_result = AjaxExecutionResult(
                            success=mission.status == "COMPLETED",
                            error=None if mission.status == "COMPLETED" else mission.status,
                            path="mission",
                        )
                    break

        return mission.last_result or AjaxExecutionResult(
            success=False, error="no_result", path="mission"
        )

    def _has_stronger_model(self, mission: MissionState) -> bool:
        providers = (
            self.provider_configs.get("providers", {})
            if isinstance(self.provider_configs, dict)
            else {}
        )
        if not providers:
            return False
        return len(providers) > 1 or any(
            cfg.get("tier") == "smart" for cfg in providers.values() if isinstance(cfg, dict)
        )

    def _pick_stronger_model(self, mission: MissionState) -> Optional[str]:
        providers = (
            self.provider_configs.get("providers", {})
            if isinstance(self.provider_configs, dict)
            else {}
        )
        tier_order = {"smart": 3, "balanced": 2, "fast": 1}
        best = None
        best_score = -1
        for name, cfg in providers.items():
            if not isinstance(cfg, dict):
                continue
            tier = str(cfg.get("tier") or "").lower()
            score = tier_order.get(tier, 0)
            if score > best_score:
                best = name
                best_score = score
        return best

    def _execute_action_step(self, mission: MissionState, step: MissionStep) -> Optional[str]:
        mission.attempts += 1
        attempt = mission.attempts
        plan_attempt_increased = False
        obs = self.perceive()
        prev_plan = mission.last_plan
        new_plan_generated = False
        try:
            if step.action and isinstance(step.action, AjaxPlan):
                plan = step.action
            elif step.action and isinstance(step.action, dict) and step.action.get("steps"):
                plan = self._plan_from_json(step.action)
                new_plan_generated = True
            else:
                plan = self.plan(
                    mission.intention,
                    obs,
                    feedback=mission.feedback,
                    envelope=mission.envelope,
                    brain_exclude=mission.brain_exclude,
                    mission=mission,
                )
                new_plan_generated = True
        except RuntimeError as exc:
            if isinstance(exc, LibrarySelectionError):
                gap_path = None
                try:
                    gap_path = self._emit_library_selection_gap(
                        mission, reason=exc.reason, detail=exc.detail
                    )
                except Exception:
                    gap_path = None
                mission.status = "GAP_LOGGED"
                mission.last_result = AjaxExecutionResult(
                    success=False,
                    error=exc.reason,
                    path="library",
                    detail={"reason": exc.reason, "gap_path": gap_path, "detail": exc.detail},
                    artifacts={"capability_gap": gap_path} if gap_path else None,
                )
                try:
                    self._record_exec_receipt(
                        mission=mission,
                        mission_id=mission.mission_id,
                        plan_id=None,
                        result=mission.last_result,
                        verify_ok=False,
                    )
                except Exception:
                    pass
                return None
            mission.last_result = AjaxExecutionResult(success=False, error=str(exc), path="plan")
            mission.last_mission_error = (
                MissionError(kind="world_error", step_id=None, reason=str(exc))
                if MissionError
                else None
            )
            return str(exc)

        if new_plan_generated:
            try:
                mission.plan_attempts = int(mission.plan_attempts or 0) + 1
            except Exception:
                mission.plan_attempts = 1
            plan_attempt_increased = True
            try:
                if isinstance(mission.notes, dict):
                    mission.notes.pop("retry_escalate_requested", None)
            except Exception:
                pass

        def _rollback_plan_attempt() -> None:
            if not plan_attempt_increased:
                return
            try:
                mission.plan_attempts = max(0, int(mission.plan_attempts or 0) - 1)
            except Exception:
                mission.plan_attempts = 0

        # Plan step: await_user_input pauses deterministically (no UI actuation).
        try:
            if plan and isinstance(plan.steps, list):
                for s in plan.steps:
                    if not isinstance(s, dict):
                        continue
                    if str(s.get("action") or "").strip() != "await_user_input":
                        continue
                    prompt = None
                    try:
                        args = s.get("args") if isinstance(s.get("args"), dict) else {}
                        raw = args.get("prompt")
                        if isinstance(raw, str) and raw.strip():
                            prompt = raw.strip()
                    except Exception:
                        prompt = None
                    if not prompt:
                        prompt = str(
                            s.get("intent") or "Necesito confirmación del usuario para continuar."
                        ).strip()
                    mission.await_user_input = True
                    mission.permission_question = prompt
                    mission.last_result = self._finalize_ask_user_wait(
                        mission,
                        prompt,
                        source="plan",
                        blocking_reason="await_user_input_step",
                    )
                    return _AWAIT_USER_SENTINEL
        except Exception:
            pass

        # Marcar replans: per-step consent puede exigir confirmación antes de actuar físicamente tras un replan.
        try:
            plan.metadata = plan.metadata or {}
            if new_plan_generated:
                # Limpiar cualquier estado de resume/consent anterior si este plan es nuevo.
                for k in (
                    "resume_from_step_index",
                    "resume_state",
                    "pending_step_consent",
                    "step_consents",
                ):
                    try:
                        plan.metadata.pop(k, None)
                    except Exception:
                        pass
                if prev_plan is not None:
                    plan.metadata["after_replan"] = True
                    plan.metadata["replan_parent_plan_id"] = prev_plan.plan_id or prev_plan.id
                    plan.metadata.setdefault("replan_consent_granted", False)
                else:
                    plan.metadata.setdefault("after_replan", False)
        except Exception:
            pass

        try:
            attempts = getattr(self, "_last_brain_attempts", [])
            if attempts:
                mission.brain_attempts.extend(attempts)
                plan.metadata = plan.metadata or {}
                plan.metadata["brain_attempts"] = attempts
        except Exception:
            pass

        mission.last_plan = plan
        mission.pending_plan = plan
        mission.feedback = None
        mission.council_signal = None

        # Preflight hard-gate (antes de Council + act): validar jugadores requeridos por las acciones del plan.
        # Nota: vision puede ser opcional en Starting XI inicial; aquí se decide según el plan real.
        sx = (mission.notes or {}).get("starting_xi") if isinstance(mission.notes, dict) else None
        vision_primary = self._starting_xi_role_primary(sx, "vision")
        vision_available = bool(vision_primary)
        plan_requires_vision = self._plan_requires_vision_player(plan)
        plan_has_ui = self._plan_has_ui_actions(plan)
        intent_class = None
        try:
            if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
                intent_class = mission.envelope.metadata.get("intent_class")
        except Exception:
            intent_class = None
        intent_class = (
            str(intent_class or self._infer_intent_class(mission.intention)).strip().lower()
        )

        if plan_requires_vision and not vision_available:
            # Auto-repair: intentar refrescar inventario/breathing + rebuild Starting XI con vision_required=true.
            repaired = None
            try:
                repaired = self._rebuild_starting_xi_for_mission(
                    mission,
                    cost_mode=str(mission.cost_mode or os.getenv("AJAX_COST_MODE", "premium")),
                    vision_required=True,
                    force_refresh=True,
                )
            except Exception:
                repaired = None
            if repaired:
                sx = repaired
                vision_primary = self._starting_xi_role_primary(sx, "vision")
                vision_available = bool(vision_primary)

        if plan_requires_vision and not vision_available:
            # Degradación permitida: media_playback tiene ruta keyboard-only.
            if intent_class == "media_playback":
                degraded = {
                    "reason": "missing_vision",
                    "fix_steps": self._missing_vision_fix_steps(sx),
                }
                try:
                    mission.notes["preflight_degraded"] = degraded
                except Exception:
                    pass
                # Forzar plan keyboard-only (sin vision.*) y continuar por Council + act.
                plan = self._keyboard_only_media_playback_plan(
                    mission.intention, degraded_reason="missing_vision"
                )
                # Reinstalar en misión para que Council revise el plan real.
                mission.last_plan = plan
                mission.pending_plan = plan
                try:
                    plan.metadata = plan.metadata or {}
                    plan.metadata.pop("council_verdict", None)
                except Exception:
                    pass
                plan_requires_vision = self._plan_requires_vision_player(plan)
                plan_has_ui = self._plan_has_ui_actions(plan)
            else:
                # Abort BEFORE act: receta + WAITING_FOR_USER (no BLOCKED).
                _rollback_plan_attempt()
                mission.attempts = max(0, mission.attempts - 1)
                fix_steps = self._missing_vision_fix_steps(sx)
                question = "DEGRADED: missing Vision.\n" + "\n".join(f"- {s}" for s in fix_steps)
                self._finalize_ask_user_wait(
                    mission,
                    question,
                    source="preflight",
                    blocking_reason="missing_vision",
                    extra_context={
                        "degraded": True,
                        "degraded_reason": "missing_vision",
                        "missing_players": [],
                        "optional_missing_players": (sx or {}).get("optional_missing_players")
                        if isinstance(sx, dict)
                        else None,
                        "slots_missing": self._slots_missing_entries(),
                        "fix_steps": fix_steps,
                    },
                )
                return _AWAIT_USER_SENTINEL

        # Si hay UI y falta vision (opcional), marcar degradación pero permitir ejecución keyboard-only.
        if plan_has_ui and not vision_available:
            try:
                if isinstance(mission.notes, dict) and "preflight_degraded" not in mission.notes:
                    mission.notes["preflight_degraded"] = {
                        "reason": "missing_vision",
                        "fix_steps": self._missing_vision_fix_steps(sx),
                    }
            except Exception:
                pass

        meta = plan.metadata if isinstance(plan.metadata, dict) else {}
        has_steps = bool(plan.steps)
        planning_error = str(meta.get("planning_error") or "").strip().lower()
        brain_failed_no_plan = bool(
            meta.get("brain_failed_no_plan") or planning_error == "brain_failed_no_plan"
        )
        missing_efe_final = bool(planning_error == "missing_efe_final")
        skip_council_review = bool(meta.get("skip_council_review"))

        if missing_efe_final:
            # Constitutional Fail-Closed: No EFE = BLOCKED
            self._record_council_invocation(mission, invoked=False, reason="missing_efe_final")
            gap_path = None
            try:
                gap_path = self._emit_missing_efe_gap(
                    mission,
                    receipt_path=meta.get("efe_repair_receipt"),
                    reason=meta.get("efe_repair_reason"),
                )
            except Exception:
                gap_path = None
            mission.status = "BLOCKED"
            mission.last_result = AjaxExecutionResult(
                success=False,
                error="missing_efe_final",
                path="plan",
                detail={
                    "planning_error": "missing_efe_final",
                    "receipt": meta.get("efe_repair_receipt"),
                    "reason": meta.get("efe_repair_reason"),
                    "gap_path": gap_path,
                },
                artifacts={"capability_gap": gap_path} if gap_path else None,
            )
            try:
                self._record_exec_receipt(
                    mission=mission,
                    mission_id=mission.mission_id,
                    plan_id=plan.plan_id or plan.id,
                    result=mission.last_result,
                    verify_ok=False,
                )
            except Exception:
                pass
            return None

        if brain_failed_no_plan or skip_council_review:
            self._record_council_invocation(mission, invoked=False, reason="brain_failed_no_plan")
            brains_failures_path = (
                meta.get("brains_failures_path") if isinstance(meta, dict) else None
            )
            provider_failures = (
                meta.get("provider_failures")
                if isinstance(meta.get("provider_failures"), list)
                else []
            )
            question = self._format_provider_failure_question(mission, provider_failures)
            extra_context = {
                "planning_error": "brain_failed_no_plan",
                "brains_failures_path": brains_failures_path,
                "provider_failures": provider_failures,
                "errors": meta.get("errors") if isinstance(meta.get("errors"), list) else None,
            }
            mission.pending_plan = None
            terminal = "WAITING_FOR_USER"
            try:
                if failure_on_no_plan_terminal is not None:
                    terminal = str(
                        failure_on_no_plan_terminal(
                            self._provider_failure_policy(), default="WAITING_FOR_USER"
                        )
                    )
            except Exception:
                terminal = "WAITING_FOR_USER"
            if terminal == "GAP_LOGGED":
                gap_path = None
                try:
                    gap_errors = (
                        extra_context.get("errors")
                        if isinstance(extra_context.get("errors"), list)
                        else None
                    )
                    gap_path = self._emit_brain_failed_no_plan_gap(
                        mission,
                        errors=gap_errors,
                        plan_meta=meta if isinstance(meta, dict) else None,
                    )
                except Exception:
                    gap_path = None
                mission.status = "GAP_LOGGED"
                mission.last_result = AjaxExecutionResult(
                    success=False,
                    error="NO_PLAN",
                    path="plan",
                    detail={
                        "planning_error": "brain_failed_no_plan",
                        "context": extra_context,
                        "gap_path": gap_path,
                    },
                    artifacts={"capability_gap": gap_path} if gap_path else None,
                )
            try:
                self._record_exec_receipt(
                    mission=mission,
                    mission_id=mission.mission_id,
                    plan_id=plan.plan_id or plan.id,
                    result=mission.last_result,
                    verify_ok=False,
                )
            except Exception:
                pass
            return None
            self._finalize_ask_user_wait(
                mission,
                question,
                source="plan",
                blocking_reason="brain_failed_no_plan",
                extra_context=extra_context,
            )
            return _AWAIT_USER_SENTINEL

        # Council-first: asegurar que todo plan (incluyendo fastpaths/hábitos) tiene verdict antes de ejecutar.
        try:
            has_verdict = bool(
                plan.metadata
                and isinstance(plan.metadata, dict)
                and isinstance(plan.metadata.get("council_verdict"), dict)
            )
        except Exception:
            has_verdict = False
        if has_verdict:
            self._record_council_invocation(mission, invoked=True, reason="verdict_precomputed")
        elif not self.council or not CouncilVerdict:
            self._record_council_invocation(mission, invoked=False, reason="council_disabled")
        
        # Guardrail: operaciones destructivas o riesgo alto MUST -> COUNCIL
        try:
            risk_level = "medium"
            gov = mission.envelope.governance if mission.envelope else None
            if gov and getattr(gov, "risk_level", None):
                risk_level = str(getattr(gov, "risk_level") or "medium").strip().lower()
            if mission.needs_explicit_permission:
                risk_level = "high"
        except Exception:
            risk_level = "medium"
        force_council = False
        try:
            if risk_level in {"high", "critical"}:
                force_council = True
            if self._plan_has_destructive_markers(plan):
                force_council = True
        except Exception:
            force_council = False
        if force_council:
            plan.metadata = plan.metadata or {}
            plan.metadata["rigor_decision"] = {
                "strategy": "COUNCIL",
                "reason": "forced_destructive_or_high_risk",
                "cost_mode": "premium",
                "use_council": True,
                "signals": {
                    "risk_level": risk_level,
                    "destructive_markers": self._plan_has_destructive_markers(plan),
                },
                "forced": True,
            }

        # Rigor Selector: FAST strategy bypasses Council [NUEVO]
        rigor_decision = (plan.metadata or {}).get("rigor_decision")
        if not has_verdict and rigor_decision and rigor_decision.get("strategy") == "FAST":
            has_verdict = True
            plan.metadata = plan.metadata or {}
            plan.metadata["council_verdict"] = {
                "verdict": "approve", 
                "reason": "rigor_selector:FAST",
                "approved": True
            }
            try:
                trace = self._record_rigor_decision_trace(
                    mission,
                    decision=rigor_decision if isinstance(rigor_decision, dict) else {},
                    plan_id=plan.plan_id or plan.id,
                    synthetic_approve=True,
                    note="rigor_selector_FAST_bypass",
                )
                if trace:
                    plan.metadata["rigor_decision_trace"] = trace
            except Exception:
                pass
            self._record_council_invocation(mission, invoked=False, reason="rigor_selector:FAST_bypass")

        if has_steps and not has_verdict and self.council and CouncilVerdict:
            try:
                plan_json = {
                    "plan_id": plan.plan_id or plan.id or f"plan-{int(time.time())}",
                    "steps": plan.steps or [],
                }
                if (
                    plan.metadata
                    and isinstance(plan.metadata, dict)
                    and plan.metadata.get("success_contract")
                ):
                    plan_json["success_contract"] = plan.metadata.get("success_contract")
                elif plan.success_spec:
                    plan_json["success_spec"] = plan.success_spec
                catalog_payload = (
                    self.actions_catalog.to_brain_payload()
                    if getattr(self, "actions_catalog", None)
                    else {}
                )
                kctx_res = self._build_knowledge_context(
                    mission.intention, envelope=mission.envelope
                )
                tp = kctx_res["tool_plan"]
                ctx = {
                    "knowledge_context": kctx_res["knowledge_context"],
                    "signals": kctx_res["signals"],
                    "observation": obs.__dict__,
                    "tool_plan": tp.to_dict() if hasattr(tp, "to_dict") else tp,
                    "starting_xi": (mission.notes or {}).get("starting_xi"),
                }
                self._record_council_invocation(mission, invoked=True, reason="plan_review_post")
                with self._override_cost_mode(self._effective_cost_mode(mission)):
                    verdict = self.council.review_plan(
                        mission.intention, plan_json, context=ctx, actions_catalog=catalog_payload
                    )
                verdict = self._normalize_council_verdict(verdict)
                plan.metadata = plan.metadata or {}
                plan.metadata["council_verdict"] = verdict.__dict__ if verdict else None
            except Exception as exc:
                self._record_council_invocation(
                    mission, invoked=True, reason="plan_review_post_failed"
                )
                try:
                    self.log.warning("Council review (post-plan) failed: %s", exc)
                except Exception:
                    pass

        council_meta = (plan.metadata or {}).get("council_verdict") if plan.metadata else None
        if council_meta:
            mission.council_signal = MissionCouncilSignal(
                verdict=council_meta.get("verdict"),
                reason=council_meta.get("reason"),
                suggested_fix=council_meta.get("suggested_fix"),
                escalation_hint=council_meta.get("escalation_hint"),
            )
            # Registrar veto(s) para evitar finales "mission_timeout" ambiguos bajo gobernanza.
            try:
                verdict_txt = str(council_meta.get("verdict") or "").strip().lower()
                if verdict_txt and verdict_txt != "approve":
                    entry = {
                        "ts_utc": self._iso_utc(),
                        "verdict": council_meta.get("verdict"),
                        "reason": council_meta.get("reason"),
                        "escalation_hint": council_meta.get("escalation_hint"),
                        "suggested_fix": council_meta.get("suggested_fix"),
                    }
                    vetoes = []
                    try:
                        vetoes = (
                            mission.notes.get("council_vetoes")
                            if isinstance(mission.notes, dict)
                            else []
                        )
                    except Exception:
                        vetoes = []
                    if not isinstance(vetoes, list):
                        vetoes = []
                    vetoes.append(entry)
                    mission.notes["council_vetoes"] = vetoes[-10:]
                    mission.notes["council_vetoed"] = True
            except Exception:
                pass

        # No-consenso -> LAB: nunca ejecutar un plan sin quorum/consenso del Council.
        if plan.steps:
            if self.contract_enforcer is not None:
                try:
                    dec = self.contract_enforcer.enforce_council_consensus(council_meta)
                except Exception:
                    dec = None
                if dec is not None:
                    if dec.action == "defer":
                        if getattr(dec, "clear_pending_plan", False):
                            mission.pending_plan = None
                        mission.feedback = (
                            mission.council_signal.reason or mission.feedback
                            if mission.council_signal
                            else mission.feedback
                        )
                        return None
                    if dec.action == "derive_to_lab":
                        mission.last_result = self._handoff_to_lab(
                            mission,
                            lab_kind="experiment",
                            reason=dec.reason,
                            plan=plan,
                            council_verdict=council_meta
                            if isinstance(council_meta, dict)
                            else None,
                        )
                        return None
            # Fallback defensivo si el enforcer no está disponible.
            if council_meta and isinstance(council_meta, dict):
                verdict_txt = str(council_meta.get("verdict") or "").strip().lower()
                hint = str(council_meta.get("escalation_hint") or "").strip().lower()
                if verdict_txt and verdict_txt != "approve":
                    # Council pide input humano / unsafe / escalación: delegar al oráculo (ASK_USER / UPGRADE / GIVE_UP).
                    if hint in {"await_user_input", "unsafe", "try_stronger_model"}:
                        mission.pending_plan = None
                        mission.feedback = (
                            mission.council_signal.reason or mission.feedback
                            if mission.council_signal
                            else mission.feedback
                        )
                        return None
                    mission.last_result = self._handoff_to_lab(
                        mission,
                        lab_kind="experiment",
                        reason=str(council_meta.get("reason") or "council_no_consensus").strip(),
                        plan=plan,
                        council_verdict=council_meta,
                    )
                    return None
            else:
                mission.last_result = self._handoff_to_lab(
                    mission,
                    lab_kind="experiment",
                    reason="council_verdict_missing",
                    plan=plan,
                    council_verdict=None,
                )
                return None

        planning_error = plan.metadata.get("planning_error") if plan.metadata else None
        if not plan.steps:
            meta = plan.metadata or {}
            brains_failures_path = None
            if isinstance(meta, dict):
                raw_path = meta.get("brains_failures_path")
                if isinstance(raw_path, str) and raw_path.strip():
                    brains_failures_path = raw_path
            terminal = "WAITING_FOR_USER"
            try:
                if failure_on_no_plan_terminal is not None:
                    terminal = str(
                        failure_on_no_plan_terminal(
                            self._provider_failure_policy(), default="WAITING_FOR_USER"
                        )
                    )
            except Exception:
                terminal = "WAITING_FOR_USER"
            if terminal == "GAP_LOGGED":
                gap_path = None
                try:
                    gap_errors = (
                        meta.get("errors") if isinstance(meta.get("errors"), list) else None
                    )
                    gap_path = self._emit_brain_failed_no_plan_gap(
                        mission,
                        errors=gap_errors,
                        plan_meta=meta if isinstance(meta, dict) else None,
                    )
                except Exception:
                    gap_path = None
                mission.status = "GAP_LOGGED"
                mission.last_result = AjaxExecutionResult(
                    success=False,
                    error="NO_PLAN",
                    path="plan",
                    detail={
                        "planning_error": planning_error or "planning_error:empty_plan",
                        "brains_failures_path": brains_failures_path,
                        "mission_id": mission.mission_id,
                        "gap_path": gap_path,
                    },
                    artifacts={"capability_gap": gap_path} if gap_path else None,
                )
                try:
                    self._record_exec_receipt(
                        mission=mission,
                        mission_id=mission.mission_id,
                        plan_id=plan.plan_id or plan.id,
                        result=mission.last_result,
                        verify_ok=False,
                    )
                except Exception:
                    pass
                return None
            payload_obj: AskUserPayload
            if planning_error == "codex_budget_exceeded":
                used = meta.get("codex_calls_used") if isinstance(meta, dict) else None
                limit = meta.get("codex_calls_limit") if isinstance(meta, dict) else None
                q = (
                    "Presupuesto Codex agotado para esta misión. "
                    f"Usado={used} límite={limit}. "
                    "Puedes usar [park] / :park para aparcar, o abrir un INCIDENT para triage."
                )
                payload_obj = AskUserPayload(
                    question=q,
                    context={
                        "planning_error": "codex_budget_exceeded",
                        "codex_calls_used": used,
                        "codex_calls_limit": limit,
                    },
                    options=[
                        {
                            "id": "open_incident",
                            "label": "Abrir INCIDENT y derivar a LAB para triage automático",
                        },
                        {
                            "id": "close_manual_done",
                            "label": "Cerrar misión manualmente (ya realizado)",
                        },
                    ],
                    default="open_incident",
                    expects="user_answer",
                )
            else:
                q = (
                    "No pude generar un plan válido (NO_PLAN). ¿Cómo quieres proceder?\n"
                    "- Proporciona un enlace exacto o más detalle (p.ej. el vídeo exacto).\n"
                    "- O responde 'retry' para reintentar planificación.\n"
                    "- O responde 'lab' para derivar a rehearsal en LAB antes de ejecutar."
                )
                # Preserve NO_PLAN error while still persisting a waiting mission.
                payload_obj = self._build_ask_user_payload(
                    mission,
                    q,
                    blocking_reason="NO_PLAN",
                    source="plan",
                    extra_context={
                        "planning_error": planning_error or "planning_error:empty_plan",
                        "brains_failures_path": brains_failures_path,
                    },
                )
            mission.status = "WAITING_FOR_USER"
            mission.await_user_input = True
            mission.permission_question = payload_obj.question
            try:
                mission.pending_user_options = payload_obj.options
            except Exception:
                pass
            waiting_path = self._persist_waiting_mission(
                mission, question=payload_obj.question, user_payload=payload_obj.to_dict()
            )
            detail: Dict[str, Any] = {
                "planning_error": planning_error or "planning_error:empty_plan",
                "brains_failures_path": brains_failures_path,
                "mission_id": mission.mission_id,
                "question": payload_obj.question,
                "options": payload_obj.options,
                "default_option": payload_obj.default,
                "context": payload_obj.context,
                "waiting_mission_path": waiting_path,
            }
            if planning_error == "codex_budget_exceeded" and isinstance(meta, dict):
                detail["codex_calls_used"] = meta.get("codex_calls_used")
                detail["codex_calls_limit"] = meta.get("codex_calls_limit")
            mission.last_result = AjaxExecutionResult(
                success=False,
                error="NO_PLAN",
                path="plan",
                detail=detail,
                artifacts={
                    **({"brains_failures": brains_failures_path} if brains_failures_path else {}),
                    **({"waiting_mission": waiting_path} if waiting_path else {}),
                }
                or None,
            )
            try:
                self._record_exec_receipt(
                    mission=mission,
                    mission_id=mission.mission_id,
                    plan_id=plan.plan_id or plan.id,
                    result=mission.last_result,
                    verify_ok=False,
                )
            except Exception:
                pass
            return _AWAIT_USER_SENTINEL

        plan_requires_physical = self._plan_requires_physical_actions(plan)
        rail = os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"

        # SSC snapshot0: obligatorio cuando hay actuation real.
        if mission.mode != "dry" and plan_requires_physical and mission.envelope:
            try:
                if not isinstance(getattr(mission.envelope, "metadata", None), dict):
                    mission.envelope.metadata = {}
                snap0 = mission.envelope.metadata.get("snapshot0")
                if not isinstance(snap0, dict) or not snap0:
                    mission.envelope.metadata["snapshot0"] = self._capture_snapshot0(mission.envelope)
            except Exception:
                pass
        if mission.mode != "dry" and microfilm_enforce_ssc is not None:
            snap0 = None
            try:
                if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
                    snap0 = mission.envelope.metadata.get("snapshot0")
            except Exception:
                snap0 = None
            ssc_gate = microfilm_enforce_ssc(
                {"actuation": plan_requires_physical, "snapshot0": snap0}
            )
            try:
                mission.notes["microfilm_ssc"] = ssc_gate
            except Exception:
                pass
            if not bool(ssc_gate.get("ok")):
                mission.status = "BLOCKED"
                mission.last_result = AjaxExecutionResult(
                    success=False,
                    error=str(ssc_gate.get("code") or "BLOCKED_SSC_MISSING"),
                    path="microfilm_guard",
                    detail=ssc_gate,
                )
                try:
                    self._record_exec_receipt(
                        mission=mission,
                        mission_id=mission.mission_id,
                        plan_id=plan.plan_id or plan.id,
                        result=mission.last_result,
                        verify_ok=False,
                    )
                except Exception:
                    pass
                return None

        # Gate LAB-actuation (sin HDMI dummy): requiere display visible/activo en rail=lab.
        if mission.mode != "dry" and self.contract_enforcer is not None:
            try:
                dec = self.contract_enforcer.enforce_lab_display(
                    rail=str(rail),
                    plan_requires_physical_actions=plan_requires_physical,
                )
                if dec.action == "allow" and isinstance(dec.detail, dict) and dec.detail.get("notice"):
                    print(f"\n[NOTICE] {dec.detail.get('notice')}\n")

                if dec.action == "wait_user":
                    # No consumir budget de intentos por un bloqueo de entorno.
                    _rollback_plan_attempt()
                    mission.attempts = max(0, mission.attempts - 1)
                    mission.await_user_input = True
                    mission.permission_question = (
                        dec.question
                        or "Necesito que prepares el entorno de LAB antes de continuar."
                    )
                    try:
                        mission.notes["lab_display_status"] = dec.detail
                    except Exception:
                        pass
                    try:
                        if not (mission.notes or {}).get("lab_display_gap_path"):
                            self._emit_lab_display_unavailable_gap(
                                mission,
                                rail=str(rail),
                                detail=dec.detail if isinstance(dec.detail, dict) else None,
                            )
                    except Exception:
                        pass
                    return None
            except Exception:
                pass

        # Pixel-safety (sin HDMI dummy): gate humano TTL antes de ejecutar acciones físicas.
        if mission.mode != "dry" and self.contract_enforcer is not None:
            try:
                dec = self.contract_enforcer.enforce_human_permission(
                    plan_requires_physical_actions=plan_requires_physical,
                )
                if dec.action == "wait_user":
                    # No consumir budget de intentos por un bloqueo de permiso humano.
                    _rollback_plan_attempt()
                    mission.attempts = max(0, mission.attempts - 1)
                    mission.await_user_input = True
                    mission.permission_question = (
                        dec.question or "Necesito permiso humano temporal antes de continuar."
                    )
                    try:
                        perm = (
                            dec.detail.get("permission")
                            if isinstance(dec.detail, dict)
                            else dec.detail
                        )
                        mission.notes["human_permission_status"] = perm
                    except Exception:
                        pass
                    return None
            except Exception:
                pass

        if mission.envelope:
            plan.metadata = plan.metadata or {}
            plan.metadata["mission_id"] = mission.mission_id
            plan.metadata["attempt"] = attempt
        contract = (
            self._normalize_success_contract(plan.success_spec)
            if hasattr(plan, "success_spec")
            else None
        )
        # Enforce intent-class contracts (CG-1): no dominio; mínimo de señales según clase funcional.
        try:
            intent_class = None
            if mission.envelope and isinstance(getattr(mission.envelope, "metadata", None), dict):
                intent_class = mission.envelope.metadata.get("intent_class")
            if intent_class == "media_playback" and not self._contract_has_two_signals(contract):
                upgraded = self._default_media_playback_success_contract(mission.intention)
                if upgraded is not None:
                    contract = upgraded
                    try:
                        plan.metadata = plan.metadata or {}
                        plan.metadata["intent_class"] = intent_class
                        plan.metadata["success_contract_upgraded"] = True
                    except Exception:
                        pass
        except Exception:
            pass
        if mission.envelope and Hypothesis:
            try:
                mission.envelope.hypothesis = Hypothesis(
                    plan=plan,
                    success_contract=contract
                    if contract is not None
                    else self._normalize_success_contract({"type": "check_last_step_status"}),
                )
            except Exception:
                pass
        try:
            self.log.info(
                "AJAX.mission: [%s#a%d] Plan listo (steps=%d, success=%s)",
                mission.mission_id,
                attempt,
                len(plan.steps or []),
                contract.primary_source if contract else "none",
            )
        except Exception:
            pass

        # Cleanup/orden mínimo: snapshot de PIDs existentes antes de ejecutar (para matar solo lo creado por la misión).
        if mission.mode != "dry":
            try:
                plan.metadata = plan.metadata or {}
                self._tag_reversible_actions(plan)
                if "preexisting_pids" not in plan.metadata:
                    plan.metadata["preexisting_pids"] = self._collect_preexisting_pids(plan)
            except Exception:
                pass

        # Transaction PREPARE: persist undo-plan + tx record (best-effort, no bloqueante).
        if mission.mode != "dry":
            try:
                self._tx_prepare(mission=mission, plan=plan, attempt=attempt)
            except Exception:
                pass
            if microfilm_enforce_undo_for_reversible is not None:
                undo_gate = microfilm_enforce_undo_for_reversible(plan)
                try:
                    mission.notes["microfilm_undo"] = undo_gate
                except Exception:
                    pass
                if not bool(undo_gate.get("ok")):
                    mission.status = "BLOCKED"
                    mission.last_result = AjaxExecutionResult(
                        success=False,
                        error=str(undo_gate.get("code") or "BLOCKED_UNDO_REQUIRED"),
                        path="microfilm_guard",
                        detail=undo_gate,
                    )
                    try:
                        self._record_exec_receipt(
                            mission=mission,
                            mission_id=mission.mission_id,
                            plan_id=plan.plan_id or plan.id,
                            result=mission.last_result,
                            verify_ok=False,
                        )
                    except Exception:
                        pass
                    return None

        result = (
            self.act(plan)
            if mission.mode != "dry"
            else AjaxExecutionResult(success=True, detail="dry_run", path="dry")
        )
        # Attach degraded preflight info (if any) to result.detail for receipts/console.
        try:
            degraded = (
                (mission.notes or {}).get("preflight_degraded")
                if isinstance(mission.notes, dict)
                else None
            )
            if isinstance(degraded, dict):
                if not isinstance(result.detail, dict):
                    result.detail = {}
                if isinstance(result.detail, dict):
                    result.detail.setdefault("degraded", True)
                    result.detail.setdefault("degraded_reason", degraded.get("reason") or "unknown")
                    result.detail.setdefault("fix_steps", degraded.get("fix_steps"))
                    try:
                        result.detail.setdefault(
                            "optional_missing_players",
                            (sx or {}).get("optional_missing_players")
                            if isinstance(sx, dict)
                            else None,
                        )
                    except Exception:
                        pass
                    try:
                        result.detail.setdefault("slots_missing", self._slots_missing_entries())
                    except Exception:
                        pass
        except Exception:
            pass
        result = self.verify(
            result,
            context={
                "driver_online": self._driver_online(),
                "verification_mode": (
                    "real" if mission.mode != "dry" and result.path == "plan_runner" else "synthetic"
                ),
            },
        )
        try:
            self._record_exec_receipt(
                mission=mission,
                mission_id=mission.mission_id,
                plan_id=plan.plan_id or plan.id,
                result=result,
                verify_ok=bool(result.success),
            )
        except Exception:
            pass
        # Si plan_runner devolvió detalle con error explícito, propagarlo
        if result.path == "plan_runner" and isinstance(result.detail, dict):
            if not result.success and result.detail.get("error"):
                result.error = str(result.detail.get("error"))
            # También propagar el primer step fallido si existe
            steps_detail = (
                result.detail.get("steps") if isinstance(result.detail.get("steps"), list) else []
            )
            for st in steps_detail:
                if isinstance(st, dict) and not st.get("ok"):
                    result.error = result.error or st.get("error") or "step_failed"
                    break

            # Gates del runner que deben pausar (fail-closed) sin cleanup ni consumo de budget.
            status = str(result.detail.get("status") or "").strip()
            if status == "human_permission_required":
                _rollback_plan_attempt()
                mission.attempts = max(0, mission.attempts - 1)
                mission.await_user_input = True
                mission.permission_question = (
                    "[HUMAN_PERMISSION] Pixel-safety activo: permiso TTL expirado o no presente.\n"
                    "Ejecuta: ajaxctl permit 120\n"
                    "Luego reintenta la misma misión para reanudar."
                )
                try:
                    mission.notes["human_permission_status"] = (
                        result.detail.get("permission") or result.detail
                    )
                except Exception:
                    pass
                return None

            if status == "step_consent_required":
                _rollback_plan_attempt()
                mission.attempts = max(0, mission.attempts - 1)
                consent = (
                    result.detail.get("consent")
                    if isinstance(result.detail.get("consent"), dict)
                    else {}
                )
                resume = (
                    result.detail.get("resume")
                    if isinstance(result.detail.get("resume"), dict)
                    else {}
                )
                try:
                    plan.metadata = plan.metadata or {}
                    plan.metadata["pending_step_consent"] = consent
                    if isinstance(resume.get("resume_from_step_index"), int):
                        plan.metadata["resume_from_step_index"] = int(
                            resume.get("resume_from_step_index")
                        )
                    elif isinstance(consent.get("step_index"), int):
                        plan.metadata["resume_from_step_index"] = int(consent.get("step_index"))
                    plan.metadata["resume_state"] = resume
                except Exception:
                    pass
                mission.pending_plan = plan
                mission.await_user_input = True
                try:
                    idx = consent.get("step_index")
                    tot = consent.get("total_steps")
                    sid = consent.get("step_id")
                    act_name = consent.get("action")
                    risk = consent.get("risk_level")
                    reasons = (
                        consent.get("reasons") if isinstance(consent.get("reasons"), list) else []
                    )
                    reasons_txt = ", ".join([str(r) for r in reasons if str(r)]) or "policy"
                    mission.permission_question = (
                        "[STEP_CONSENT] Confirmación por paso requerida.\n"
                        f"Paso {idx}/{tot} · id={sid} · action={act_name} · risk={risk}\n"
                        f"Motivos: {reasons_txt}\n"
                        "Responde 'sí' para ejecutar este paso. Cualquier otra respuesta → replanear."
                    )
                except Exception:
                    mission.permission_question = "[STEP_CONSENT] Confirmación por paso requerida. Responde 'sí' para continuar."
                try:
                    mission.notes["step_consent_pending"] = consent
                except Exception:
                    pass
                return None

            if status == "deference_human_active":
                _rollback_plan_attempt()
                mission.attempts = max(0, mission.attempts - 1)
                try:
                    resume = (
                        result.detail.get("resume")
                        if isinstance(result.detail.get("resume"), dict)
                        else {}
                    )
                    if isinstance(resume, dict):
                        plan.metadata = plan.metadata or {}
                        if isinstance(resume.get("resume_from_step_index"), int):
                            plan.metadata["resume_from_step_index"] = int(
                                resume.get("resume_from_step_index")
                            )
                        plan.metadata["resume_state"] = resume
                except Exception:
                    pass
                mission.pending_plan = plan
                mission.await_user_input = True
                question = None
                try:
                    question = result.detail.get("question")
                except Exception:
                    question = None
                mission.permission_question = (
                    str(question).strip()
                    if isinstance(question, str) and question.strip()
                    else "[DEFERENCE] Detectada actividad humana (teclado/ratón). Pausado en PROD para no pelear por el control. Responde para reanudar."
                )
                try:
                    mission.notes["deference"] = result.detail
                except Exception:
                    pass
                return None
        # Persistir siempre el detalle en la misión para el registro
        mission.last_result = result
        overall_success = bool(result.success)

        evaluation: Dict[str, Any] = {"ok": True, "reason": "no_contract", "score": 1.0}
        if contract and mission.mode != "dry" and evaluate_success:
            try:
                evaluation = evaluate_success(
                    self.driver,
                    contract,
                    intention=mission.intention,
                    last_result=result.detail if isinstance(result.detail, dict) else None,
                    mission_id=mission.mission_id,
                    attempt=attempt,
                )
                success_check = bool(evaluation.get("ok", False))
            except Exception as exc:
                success_check = False
                evaluation = {"ok": False, "reason": f"Success check crashed: {exc}", "score": 0.0}
                mission.feedback = evaluation["reason"]
            overall_success = overall_success and success_check
            if not overall_success and not mission.feedback:
                expected = None
                if isinstance(contract, SuccessContract):
                    expected = contract.primary_check.get("text")
                elif isinstance(plan.success_spec, dict):
                    expected = plan.success_spec.get("text")
                try:
                    fg = self.driver.get_active_window() if self.driver else {}
                    actual_title = str((fg or {}).get("title") or "")
                except Exception:
                    actual_title = "unknown"
                fallback_reason = evaluation.get("reason") or ""
                mission.feedback = (
                    f"Plan executed, but success condition failed. Expected title containing '{expected}', "
                    f"found '{actual_title}'. {fallback_reason}".strip()
                )

        if overall_success and microfilm_enforce_verify_before_done is not None:
            verification_payload = {}
            if isinstance(result.detail, dict) and isinstance(result.detail.get("verification"), dict):
                verification_payload = dict(result.detail.get("verification"))
            done_gate = microfilm_enforce_verify_before_done(
                {"status": "DONE", "mission_status": "DONE", "detail": result.detail},
                verification_payload,
            )
            if isinstance(result.detail, dict):
                result.detail["microfilm_done_gate"] = done_gate
            if not bool(done_gate.get("ok")):
                overall_success = False
                result.error = str(done_gate.get("code") or "BLOCKED_VERIFY_REQUIRED")
                if not isinstance(result.detail, dict):
                    result.detail = {}
                if isinstance(result.detail, dict):
                    result.detail.setdefault("terminal_status", "BLOCKED")
                    result.detail.setdefault(
                        "actionable_hint",
                        done_gate.get("actionable_hint") or "verification_required_before_done",
                    )

        result.success = overall_success
        if not overall_success and mission.mode != "dry":
            try:
                if isinstance(result.detail, dict) and bool((plan.metadata or {}).get("has_reversible_actions")):
                    result.detail.setdefault("undo_flow", "triggered")
            except Exception:
                pass
            try:
                self._cleanup_after_failure(mission, plan, result)
            except Exception:
                pass

        # Transaction VERIFY/UNDO: finalizar tx record con apply/verify/undo (best-effort, no bloqueante).
        if mission.mode != "dry":
            try:
                self._tx_finalize(
                    mission=mission, plan=plan, attempt=attempt, result=result, evaluation=evaluation
                )
            except Exception:
                pass

        mission.pending_plan = None
        self._append_execution_event(mission.envelope, plan, result, evaluation)
        mission_error = self._classify_mission_error(plan, result, evaluation)
        mission.last_mission_error = mission_error
        if mission.envelope:
            mission.envelope.last_error = mission_error
        log_path = self._save_mission_attempt(
            mission.envelope, attempt, plan, contract, result, evaluation, mission_error
        )

        try:
            self.log.info(
                "AJAX.mission: [%s#a%d] Exec ok=%s success_eval=%s log=%s",
                mission.mission_id,
                attempt,
                overall_success,
                evaluation.get("reason") if isinstance(evaluation, dict) else evaluation,
                log_path or "-",
            )
        except Exception:
            pass

        # Actualizar estadísticas del hábito si aplica
        try:
            if overall_success and mission.last_plan and mission.last_plan.metadata:
                habit_id = mission.last_plan.metadata.get("habit_id")
                if habit_id and habits_mod:
                    habits_mod.update_habit_usage(habit_id, True)
        except Exception:
            pass

        if overall_success:
            mission.status = "COMPLETED"
            if log_path:
                result.artifacts = result.artifacts or {}
                if isinstance(result.artifacts, dict):
                    result.artifacts["mission_log"] = log_path
            return None

        if not self._driver_health():
            mission.infra_issue = MissionInfraIssue(
                component="driver", recoverable=True, retry_in=8, detail="driver_unhealthy"
            )

        if mission_error and not mission.feedback:
            mission.feedback = mission_error.reason or mission.feedback
        if log_path:
            result.artifacts = result.artifacts or {}
            if isinstance(result.artifacts, dict):
                result.artifacts["mission_log"] = log_path
        if str(result.error or "").strip().upper().startswith("BLOCKED_"):
            mission.status = "BLOCKED"
            return None
        # Mantener la misión viva para pursuit/replan (hasta agotar mission.max_attempts).
        mission.status = "IN_PROGRESS"
        return None

    def _handle_ask_user_step(self, mission: MissionState, question: Optional[str]) -> str:
        """
        Antes de preguntar al usuario, intenta escalar a modelos superiores.
        Devuelve "plan" si se obtuvo plan autónomo, "ask" si se debe consultar.
        """
        q = question or mission.permission_question or "Necesito confirmación antes de continuar."
        if "[HUMAN_PERMISSION]" in q:
            mission.ask_user_request = AskUserRequest(
                question=q,
                reason="human_permission_required",
                timeout_seconds=600,
                on_timeout="abort",
                alert_level="normal",
                escalation_trace=[],
            )
            mission.await_user_input = True
            mission.permission_question = q
            return "ask"
        if "[LAB_DISPLAY]" in q:
            mission.ask_user_request = AskUserRequest(
                question=q,
                reason="lab_display_unavailable",
                timeout_seconds=600,
                on_timeout="abort",
                alert_level="normal",
                escalation_trace=[],
            )
            mission.await_user_input = True
            mission.permission_question = q
            return "ask"
        if "[STEP_CONSENT]" in q:
            mission.ask_user_request = AskUserRequest(
                question=q,
                reason="step_consent_required",
                timeout_seconds=600,
                on_timeout="abort",
                alert_level="normal",
                escalation_trace=[],
            )
            mission.await_user_input = True
            mission.permission_question = q
            return "ask"
        if "[BUDGET_EXHAUSTED]" in q:
            mission.ask_user_request = AskUserRequest(
                question=q,
                reason="budget_exhausted",
                timeout_seconds=600,
                on_timeout="abort",
                alert_level="normal",
                escalation_trace=[],
            )
            mission.await_user_input = True
            mission.permission_question = q
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["budget_exhausted"] = True
            except Exception:
                pass
            return "ask"
        plan, ask_req = self._escalate_ask_user(mission, q)
        if plan:
            mission.pending_plan = plan
            mission.await_user_input = False
            mission.permission_question = None
            mission.ask_user_request = None
            return "plan"
        mission.ask_user_request = ask_req
        mission.await_user_input = True
        mission.permission_question = q
        return "ask"

    def _build_ask_user_payload(
        self,
        mission: MissionState,
        base_question: Optional[str],
        *,
        blocking_reason: Optional[str],
        source: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> AskUserPayload:
        reason_txt = (blocking_reason or source or "").strip().lower()
        prompt = (base_question or "").strip()
        ctx: Dict[str, Any] = {
            "mission_id": mission.mission_id,
            "intent": mission.intention,
            "premium_rule": getattr(mission, "premium_rule", "if_needed"),
        }
        budget_exhausted = (
            "budget_exhausted" in reason_txt or "[budget_exhausted]" in prompt.lower()
        )
        no_recipe = "no_deterministic_recipe" in reason_txt
        if source:
            ctx["source"] = source
        if blocking_reason:
            ctx["blocking_reason"] = blocking_reason
        if extra_context:
            ctx.update(extra_context)
        if budget_exhausted:
            ctx["budget_exhausted"] = True
        if no_recipe:
            ctx["recipe_match"] = False
        if not prompt or reason_txt in {"consensus_blocked", "consensus_block"}:
            prompt = (
                "El Council bloqueó la maniobra por falta de consenso. "
                "Indica cómo quieres proceder."
            )
        elif source == "council" and not blocking_reason:
            prompt = "El Council no aportó motivo textual para el veto. Decide cómo proceder."
            ctx["blocking_reason"] = "council_no_reason"
        if budget_exhausted:
            options = [
                {
                    "id": "open_incident",
                    "label": "Abrir INCIDENT y derivar a LAB para triage automático",
                },
                {"id": "close_manual_done", "label": "Cerrar misión manualmente (ya realizado)"},
            ]
        else:
            options = [
                {
                    "id": "retry_escalate_brain",
                    "label": "Reintentar con Brain más capaz (coste mayor)",
                },
                {
                    "id": "use_deterministic_recipe",
                    "label": "Usar receta determinista segura para esta clase de intent (si aplica)",
                },
                {
                    "id": "open_incident",
                    "label": "Abrir INCIDENT y derivar a LAB para triage automático",
                },
                {"id": "close_manual_done", "label": "Cerrar misión manualmente (ya realizado)"},
            ]
        if getattr(mission, "loop_guard", False):
            if not prompt:
                prompt = "Loop guard activo: necesito un detalle nuevo y concreto para avanzar."
            else:
                prompt = f"Loop guard activo: necesito un detalle nuevo para avanzar.\n{prompt}"
            if budget_exhausted:
                options = [
                    {
                        "id": "retry_fresh",
                        "label": "Reintentar como nueva misión (reset de presupuesto)",
                    },
                    {
                        "id": "open_incident",
                        "label": "Abrir INCIDENT y derivar a LAB para triage automático",
                    },
                    {
                        "id": "close_manual_done",
                        "label": "Cerrar misión manualmente (ya realizado)",
                    },
                ]
            else:
                options = [
                    {
                        "id": "use_deterministic_recipe",
                        "label": "Usar receta determinista segura (si aplica)",
                    },
                    {
                        "id": "open_incident",
                        "label": "Abrir INCIDENT y derivar a LAB para triage automático",
                    },
                    {
                        "id": "close_manual_done",
                        "label": "Cerrar misión manualmente (ya realizado)",
                    },
                ]
        allow_premium_override = getattr(mission, "premium_rule", "if_needed") != "now"
        has_provider_failures = bool(extra_context and extra_context.get("provider_failures"))
        if not budget_exhausted and allow_premium_override and has_provider_failures:
            options.append(
                {
                    "id": "use_premium_now",
                    "label": "Permitir premium inmediatamente (mayor coste)",
                }
            )
        if budget_exhausted:
            default_opt = "open_incident"
        elif no_recipe:
            default_opt = "open_incident"
        else:
            default_opt = "retry_escalate_brain"
        if getattr(mission, "loop_guard", False):
            if budget_exhausted:
                default_opt = "retry_fresh"
            elif "use_deterministic_recipe" in {opt.get("id") for opt in options}:
                default_opt = "use_deterministic_recipe"
        return AskUserPayload(
            question=prompt,
            context=ctx,
            options=options,
            default=default_opt,
            expects="user_answer",
        )

    def _finalize_ask_user_wait(
        self,
        mission: MissionState,
        question: Optional[str],
        *,
        source: Optional[str] = None,
        blocking_reason: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> AjaxExecutionResult:
        base_question = (
            mission.permission_question
            or question
            or "Necesito que confirmes exactamente qué quieres antes de continuar."
        )
        loop_info = self._update_progress_tracking(
            mission,
            pending_question=base_question,
            source="waiting",
        )
        if loop_info.get("loop_guard_triggered"):
            try:
                if isinstance(mission.notes, dict):
                    mission.notes["loop_guard_triggered_utc"] = self._iso_utc()
            except Exception:
                pass
        parked_by_loop_guard = False
        provider_failures = (
            extra_context.get("provider_failures") if isinstance(extra_context, dict) else None
        )
        if mission.loop_guard and self._provider_failures_dead(provider_failures):
            try:
                reporter = IncidentReporter(self.config.root_dir)
                dedupe_key = self._hash_dict(
                    {
                        "signal": "loop_guard_provider_dead",
                        "mission_id": mission.mission_id,
                        "failures": provider_failures,
                    }
                )
                incident_id = reporter.open_incident(
                    kind="provider_health",
                    summary="Loop guard: providers sin progreso (timeout/no providers).",
                    context={"mission_id": mission.mission_id, "failures": provider_failures},
                    remediation=[
                        "Verifica auth/providers_status y reinicia providers locales si aplica."
                    ],
                    dedupe_key=dedupe_key,
                )
                mission.notes["loop_guard_incident_id"] = incident_id
            except Exception:
                pass
            mission.status = "PAUSED_BY_USER"
            try:
                mission.notes["parked_by_user"] = True
                mission.notes["paused_by_user"] = True
            except Exception:
                pass
            base_question = (
                "Loop guard: INCIDENT abierto y misión aparcada. Vuelve a chat o aporta nueva info."
            )
            parked_by_loop_guard = True
        payload_obj = self._build_ask_user_payload(
            mission,
            base_question,
            blocking_reason=blocking_reason or question,
            source=source,
            extra_context=extra_context,
        )
        prompt = payload_obj.question
        try:
            mission.waiting_cycles = int(getattr(mission, "waiting_cycles", 0) or 0) + 1
        except Exception:
            mission.waiting_cycles = 1
        mission.status = "PAUSED_BY_USER" if parked_by_loop_guard else "WAITING_FOR_USER"
        try:
            mission.pending_user_options = payload_obj.options
            if isinstance(mission.notes, dict):
                mission.notes["_pending_user_options"] = payload_obj.options
        except Exception:
            pass
        try:
            if isinstance(mission.notes, dict):
                invoked = mission.notes.get("council_invoked")
                if invoked is None:
                    mission.notes["council_invoked"] = False
                    invoked = False
                if invoked is False and not mission.notes.get("council_invoked_reason"):
                    fallback_reason = blocking_reason or source or "ask_user"
                    mission.notes["council_invoked_reason"] = f"ask_user:{fallback_reason}"
                    mission.notes["council_invoked_ts"] = self._iso_utc()
        except Exception:
            pass
        if source == "council":
            try:
                mission.notes["council_user_notified"] = True
            except Exception:
                pass
        try:
            print(f"[ASK_USER] {prompt}")
        except Exception:
            pass
        waiting_path = self._persist_waiting_mission(
            mission, question=prompt, user_payload=payload_obj.to_dict()
        )
        payload_path = (
            mission.notes.get("_waiting_payload_path") if isinstance(mission.notes, dict) else None
        )
        detail: Dict[str, Any] = {
            "question": prompt,
            "mission_id": mission.mission_id,
            "options": payload_obj.options,
            "default_option": payload_obj.default,
            "expects": payload_obj.expects,
            "context": payload_obj.context,
            "loop_guard": bool(mission.loop_guard),
            "loop_guard_triggered": bool(loop_info.get("loop_guard_triggered")),
        }
        if parked_by_loop_guard:
            detail["parked_by_loop_guard"] = True
        try:
            if isinstance(mission.notes, dict):
                detail["council_invoked"] = mission.notes.get("council_invoked")
                detail["council_invoked_reason"] = mission.notes.get("council_invoked_reason")
        except Exception:
            pass
        try:
            self._enrich_detail_with_router_summary(mission, detail)
        except Exception:
            pass
        if mission.ask_user_request:
            detail["ask_user_request"] = mission.ask_user_request.to_dict()
        if waiting_path:
            detail["waiting_mission_path"] = waiting_path
        if payload_path:
            detail["waiting_payload_path"] = payload_path
        self._record_waiting_receipt(
            mission=mission,
            question=prompt,
            payload=payload_obj,
            loop_info=loop_info,
            waiting_path=waiting_path,
            payload_path=payload_path,
        )
        mission.last_result = AjaxExecutionResult(
            success=False,
            error="await_user_input",
            detail=detail,
            path="waiting_for_user",
            artifacts={"waiting_mission": waiting_path} if waiting_path else None,
        )
        return mission.last_result

    @staticmethod
    def _plan_requires_physical_actions(plan: AjaxPlan) -> bool:
        physical = {
            "app.launch",
            "keyboard.type",
            "keyboard.hotkey",
            "window.focus",
            "vision.llm_click",
            "desktop.isolate_active_window",
            "mouse.click",
            "mouse.move",
        }
        for st in plan.steps or []:
            if not isinstance(st, dict):
                continue
            action = str(st.get("action") or "").strip()
            if action in physical:
                return True
        return False

    @staticmethod
    def _is_reversible_action(action: str) -> bool:
        reversible = {
            "app.launch",
            "window.focus",
            "window.minimize",
            "window.maximize",
            "window.restore",
            "window.move",
            "window.resize",
            "desktop.isolate_active_window",
            "keyboard.type",
            "keyboard.hotkey",
            "mouse.click",
            "mouse.move",
        }
        return str(action or "").strip() in reversible

    def _tag_reversible_actions(self, plan: AjaxPlan) -> bool:
        has_reversible = False
        for st in plan.steps or []:
            if not isinstance(st, dict):
                continue
            action = str(st.get("action") or "").strip()
            reversible = self._is_reversible_action(action)
            if reversible:
                has_reversible = True
            st["_microfilm_reversible"] = bool(reversible)
        plan.metadata = plan.metadata or {}
        plan.metadata["has_reversible_actions"] = has_reversible
        if has_reversible and "reversible" not in plan.metadata:
            plan.metadata["reversible"] = True
        return has_reversible

    def _ask_user_and_wait(self, mission: MissionState, question: Optional[str]) -> None:
        mission.await_user_input = True
        mission.permission_question = question or mission.permission_question
        mission.retry_after = time.time() + 5
        try:
            if mission.permission_question:
                print(f"[ASK_USER] {mission.permission_question}")
        except Exception:
            pass
        try:
            self.state.notes["last_user_question"] = mission.permission_question
        except Exception:
            pass

    def _switch_brain_model(self, mission: MissionState, next_model: Optional[str]) -> None:
        # Marca intención de usar un modelo más fuerte; detalle simple para ruteo.
        os.environ["AJAX_COST_MODE"] = os.getenv("AJAX_COST_MODE", "premium")
        if next_model:
            try:
                self.state.notes["brain_upgrade"] = next_model
            except Exception:
                pass

    def _attempt_infra_fix(self, mission: MissionState, step: MissionStep) -> None:
        ok = False
        if step.component == "driver":
            ok = self._start_driver()
        if ok:
            mission.infra_issue = None
            mission.retry_after = 0
        else:
            mission.retry_after = time.time() + (step.retry_in or 8)

    def _request_explicit_permission(self, mission: MissionState, step: MissionStep) -> None:
        mission.permission_question = step.question or mission.permission_question
        mission.retry_after = time.time() + 5

    def _update_memory_from_user(self, mission: MissionState, step: MissionStep) -> None:
        if not mission.last_user_reply:
            return
        if self.rag_client and hasattr(self.rag_client, "add_memory"):
            try:
                self.rag_client.add_memory(mission.last_user_reply)
            except Exception:
                pass

    def _technically_impossible(self, mission: MissionState) -> bool:
        if mission.plan_attempts < mission.max_attempts:
            return False
        if mission.infra_issue and mission.infra_issue.recoverable:
            return False
        if mission.council_signal and mission.council_signal.escalation_hint in {
            "await_user_input",
            "try_stronger_model",
        }:
            return False
        return True

    def _register_error(self, mission: MissionState, err: str) -> None:
        try:
            self.log.warning("AJAX.mission: [%s] step error: %s", mission.mission_id, err)
        except Exception:
            pass
        if not mission.last_result:
            mission.last_result = AjaxExecutionResult(success=False, error=err, path="mission")
        mission.last_mission_error = (
            MissionError(kind="plan_error", step_id=None, reason=err) if MissionError else None
        )
        reason_key = self._normalize_reason_key(err)
        if self._should_launch_lab_probe(mission, reason_key):
            mission.last_result = self._handoff_to_lab(
                mission,
                lab_kind="probe",
                reason=reason_key,
                plan=mission.last_plan,
                detail={"error": err},
            )

    def do(
        self,
        intention: str,
        mode: Literal["auto", "dry"] = "auto",
        override_plan: Optional[Any] = None,
        envelope_override: Optional["MissionEnvelope"] = None,
        ignore_waiting: Optional[bool] = None,
    ) -> AjaxExecutionResult:
        self.log.info("AJAX.do: Procesando intención '%s' (modo=%s)", intention, mode)
        if self._current_rail() == "prod":
            self._lab_preempt("prod_mission_start", {"intention": intention, "mode": mode})
        input_text = intention
        resuming = False
        resumed_question = ""

        if ignore_waiting is None:
            ignore_waiting_env = (
                os.getenv("AJAX_IGNORE_WAITING_MISSION") or ""
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            ignore_waiting_env = ignore_waiting

        mission: MissionState
        envelope: Optional["MissionEnvelope"]
        mission_id: str

        if not ignore_waiting_env:
            waiting = self._load_waiting_mission_payload()
            if waiting:
                try:
                    mission = self._mission_from_waiting_payload(waiting)
                    resumed_question = (
                        str(waiting.get("question") or "")
                        or "Necesito confirmación antes de continuar."
                    )
                    self._apply_user_reply_to_waiting_mission(
                        mission,
                        user_reply=input_text,
                        question=resumed_question,
                    )
                    resuming = True
                    envelope = mission.envelope
                    mission_id = mission.mission_id
                    intention = mission.intention
                    mode = mission.mode  # type: ignore[assignment]
                    self._clear_waiting_mission(mission_id=mission_id)
                    try:
                        print(
                            f"📌 Reanudando misión {mission_id} en WAITING_FOR_USER con tu respuesta…"
                        )
                        print(f"❓ Pregunta pendiente: {resumed_question}")
                    except Exception:
                        pass
                except Exception:
                    # Si no se puede rehidratar, no bloquear la nueva misión: limpiar y seguir.
                    try:
                        self._clear_waiting_mission(mission_id=str(waiting.get("mission_id")))
                    except Exception:
                        self._clear_waiting_mission()

        intent_hash: Optional[str] = None
        if self.mission_breaker and not resuming:
            intent_hash = self.mission_breaker.hash_intention(intention, mode)
            if self.mission_breaker.mission_should_block(
                intent_hash, time.time(), intent_text=intention
            ):
                detail = {
                    "intent_hash": intent_hash,
                    "threshold": self.mission_breaker.threshold,
                    "window_seconds": self.mission_breaker.window_seconds,
                    "reason": "mission breaker tripped for repeated failures",
                }
                speak_instability_alert("mission")
                return AjaxExecutionResult(
                    success=False,
                    error="mission_breaker_tripped",
                    detail=detail,
                    path="circuit_breaker",
                    plan_id=None,
                )
        if (os.getenv("AJAX_PROVIDERS_PREFLIGHT_DONE") or "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                preflight = self.providers_preflight(requested_tier=os.getenv("AJAX_COST_MODE"))
                if isinstance(preflight, dict):
                    chosen_tier = preflight.get("chosen_tier")
                    requested_tier = preflight.get("requested_tier")
                    if chosen_tier and chosen_tier != requested_tier:
                        os.environ["AJAX_COST_MODE"] = str(chosen_tier)
            except Exception:
                pass
            os.environ["AJAX_PROVIDERS_PREFLIGHT_DONE"] = "1"
        try:
            self._ensure_driver_running()
        except Exception:
            pass

        if not resuming:
            envelope = envelope_override or self._create_mission_envelope(intention, mode)
            mission_id = envelope.mission_id if envelope else f"mission-{int(time.time())}"
            mission = MissionState(
                intention=intention, mode=mode, envelope=envelope, mission_id=mission_id
            )
        else:
            # Recomputar hash para registro final (sin bloquear inicio).
            if self.mission_breaker:
                try:
                    intent_hash = self.mission_breaker.hash_intention(intention, mode)
                except Exception:
                    intent_hash = None
            if mission.max_attempts <= 0:
                mission.max_attempts = 3

        # Exponer el mission_id vigente a herramientas externas (CLI reporters/verifiers).
        try:
            self._last_mission_id = mission_id  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if not resuming:
                mission.max_attempts = max(1, int(getattr(self, "mission_attempt_budget", 3) or 3))
            else:
                mission.max_attempts = max(1, int(mission.max_attempts or 1))
        except Exception:
            mission.max_attempts = 3
        if override_plan is not None:
            try:
                if isinstance(override_plan, AjaxPlan):
                    preset_plan = override_plan
                else:
                    preset_plan = self._plan_from_json(override_plan)
                mission.pending_plan = preset_plan
                mission.last_plan = preset_plan
                mission.notes["plan_source"] = "override"
            except Exception:
                pass
        if intent_hash:
            mission.notes["intent_hash"] = intent_hash
        try:
            mission.notes["attempt_budget"] = mission.max_attempts
            mission.notes["retry_budget_sec"] = int(
                getattr(self, "mission_retry_budget_seconds", 30) or 30
            )
        except Exception:
            pass
        scores_log: Dict[str, float] = {}
        selected_plan_json: Optional[Dict[str, Any]] = None
        selected_origin: Optional[str] = None
        final_result: Optional[AjaxExecutionResult] = None

        if self._driver_status() == "down":
            detail = {
                "error": "driver_down",
                "message": "Circuit breaker: driver en down state",
                "recovery": "restart_driver",
                "down_since": self._driver_cb.get("down_since"),
                "cooldown_seconds": getattr(self, "driver_recovery_cooldown", None),
                "failure_count": len(self._driver_cb.get("failures", [])),
            }
            mission.status = "DRIVER_DOWN"
            mission.infra_issue = MissionInfraIssue(
                component="driver",
                recoverable=True,
                retry_in=int(getattr(self, "driver_recovery_cooldown", 20) or 20),
                detail="driver_down",
            )
            final_result = AjaxExecutionResult(
                success=False, error="driver_down", detail=detail, path="driver"
            )
            mission.last_result = final_result
            mission.last_mission_error = (
                MissionError(kind="infra_error", step_id=None, reason="driver_down")
                if MissionError
                else None
            )

        if final_result is None and not self._driver_health():
            mission.infra_issue = MissionInfraIssue(
                component="driver", recoverable=True, retry_in=8, detail="driver_unhealthy"
            )

        # Preflight obligatorio: Starting XI por rol (providers/inventory/quorum) antes de cualquier plan/act.
        if final_result is None and not resuming:
            try:
                _, preflight_res = self._preflight_starting_xi(mission)
            except Exception as exc:
                preflight_res = AjaxExecutionResult(
                    success=False,
                    error="BLOCKED_BY_MISSING_PLAYERS",
                    path="preflight",
                    detail={"reason": "starting_xi_failed", "error": str(exc)[:200]},
                )
            if preflight_res is not None:
                status_override = None
                try:
                    if isinstance(preflight_res.detail, dict):
                        raw_status = preflight_res.detail.get("terminal_status")
                        if isinstance(raw_status, str) and raw_status.strip():
                            status_override = raw_status.strip().upper()
                except Exception:
                    status_override = None
                if status_override in TERMINAL_STATES:
                    mission.status = status_override
                elif str(preflight_res.error or "").strip().upper().startswith("BLOCKED"):
                    mission.status = "BLOCKED"
                else:
                    mission.status = "BLOCKED_BY_MISSING_PLAYERS"
                mission.last_result = preflight_res
                mission.last_mission_error = (
                    MissionError(
                        kind="governance_error",
                        step_id=None,
                        reason=preflight_res.error or "missing_players",
                    )
                    if MissionError
                    else None
                )
                final_result = preflight_res

        # Snapshot-0 (best-effort) antes de cualquier acción de escritorio.
        if final_result is None and mission.envelope and not resuming:
            try:
                snap0 = self._capture_snapshot0(mission.envelope)
                if isinstance(mission.envelope.metadata, dict):
                    mission.envelope.metadata["snapshot0"] = snap0
            except Exception:
                pass

        # Library-first (ActionCatalog + Habits)
        candidates: list = []
        skip_explorer = False
        if final_result is None and not resuming:
            try:
                library_plan = self._library_first_plan_json(intention, mission=mission)
                if library_plan:
                    selected_plan_json = library_plan
                    selected_origin = "library"
                    scores_log = {"library": 1.0}
                    skip_explorer = True
                    try:
                        meta = (
                            library_plan.get("metadata") if isinstance(library_plan, dict) else {}
                        )
                        habit_id = meta.get("habit_id") if isinstance(meta, dict) else None
                        if habit_id and mission.envelope:
                            mission.envelope.metadata["used_habit_id"] = habit_id
                        if habit_id:
                            print(f"⚡ Hábito activado: {habit_id}")
                    except Exception:
                        pass
                    if self._is_os_micro_action(intention):
                        try:
                            if isinstance(mission.notes, dict):
                                mission.notes["recipe_fast_path"] = True
                                mission.notes["recipe_reason"] = "os_micro_action"
                        except Exception:
                            pass
            except LibrarySelectionError as exc:
                gap_path = None
                try:
                    gap_path = self._emit_library_selection_gap(
                        mission, reason=exc.reason, detail=exc.detail
                    )
                except Exception:
                    gap_path = None
                final_result = AjaxExecutionResult(
                    success=False,
                    error=exc.reason,
                    path="library",
                    detail={"reason": exc.reason, "gap_path": gap_path, "detail": exc.detail},
                    artifacts={"capability_gap": gap_path} if gap_path else None,
                )
                mission.status = "GAP_LOGGED"
                mission.last_result = final_result

        # Explorador (Brain+Council) como candidato
        if (
            final_result is None
            and not resuming
            and not skip_explorer
            and explorer
            and PlanCandidate
        ):
            try:
                exp = explorer.Explorer(self).propose_plan(mission)
                if exp:
                    candidates.append(exp)
            except Exception as exc:
                self.log.warning("Explorer falló: %s", exc)

        if choose_best and candidates:
            best, scores = choose_best(candidates)
            scores_log = scores
            if best:
                selected_plan_json = best.plan_json
                selected_origin = best.origin
                try:
                    print(f"⚖ Árbitro: scores={scores} -> elijo {best.origin}")
                except Exception:
                    pass
        elif candidates:
            # Fallback: primer candidato
            selected_plan_json = candidates[0].plan_json
            selected_origin = (
                candidates[0].origin if hasattr(candidates[0], "origin") else "unknown"
            )

        self.log.info("AJAX.mission: [%s] Intent: %s", mission_id, intention)
        ran_plan_act = False
        if final_result is None and mission.status in TERMINAL_STATES:
            final_result = mission.last_result or AjaxExecutionResult(
                success=False,
                error=str(mission.status),
                path="mission",
            )
        else:
            try:
                self.log.info("AJAX.mission: [%s] calling run_plan/act", mission_id)
            except Exception:
                pass
            if final_result is None:
                final_result = self.pursue_intent(
                    mission,
                    preselected_plan=selected_plan_json,
                    plan_source=selected_origin,
                    plan_scores=scores_log,
                )
            ran_plan_act = True
        if ran_plan_act:
            try:
                self.log.info(
                    "AJAX.mission: [%s] finished run_plan/act result=%s",
                    mission_id,
                    getattr(final_result, "error", None),
                )
            except Exception:
                pass
        if final_result:
            try:
                self._annotate_plan_runner_result(mission, final_result)
            except Exception:
                self.log.debug("No se pudo anotar detalle del plan_runner", exc_info=True)
            try:
                if final_result.error in {"PAUSED_FOR_LAB", "CANCELLED", "ABORTED_BY_USER"}:
                    if final_result.detail is None:
                        final_result.detail = {}
                    if isinstance(final_result.detail, dict):
                        final_result.detail.setdefault("mission_id", mission.mission_id)
                        self._enrich_detail_with_router_summary(mission, final_result.detail)
            except Exception:
                pass

        self.state.last_intention = intention
        self.state.last_plan_id = mission.last_plan.plan_id if mission.last_plan else None
        waiting_for_user = bool(
            mission.status == "WAITING_FOR_USER"
            or final_result.path == "waiting_for_user"
            or (final_result.error or "").strip() == "await_user_input"
        )
        if waiting_for_user:
            self.state.last_result_summary = "waiting_for_user"
            try:
                self.state.notes["waiting_mission_id"] = mission.mission_id
                q_note = mission.permission_question
                if isinstance(final_result.detail, dict) and final_result.detail.get("question"):
                    q_note = str(final_result.detail.get("question"))
                self.state.notes["waiting_question"] = q_note
            except Exception:
                pass
        else:
            self.state.last_result_summary = "success" if final_result.success else "failure"

        probe_note: Optional[Dict[str, Any]] = None
        if final_result.path == "plan_runner" and isinstance(final_result.detail, dict):
            for step in final_result.detail.get("steps", []):
                if not isinstance(step, dict):
                    continue
                detail = step.get("detail")
                action_name = detail.get("action") if isinstance(detail, dict) else None
                if action_name == "os.probe_standby":
                    report = detail.get("report") if isinstance(detail, dict) else None
                    probe_note = {
                        "ok": bool(step.get("ok")),
                        "error": step.get("error")
                        or (report.get("error") if isinstance(report, dict) else None),
                        "report": report,
                    }
                    break
        if probe_note:
            self.state.notes["last_os_probe"] = probe_note
            if not probe_note.get("ok"):
                self.state.last_result_summary = "failure:os_probe"

        if mission.feedback and not final_result.success:
            self.state.notes["last_failure_feedback"] = mission.feedback

        if envelope:
            try:
                self.state.notes["last_mission_envelope"] = envelope.to_dict()
            except Exception:
                pass
            try:
                last_log = envelope.metadata.get("last_mission_log")
                if last_log:
                    final_result.artifacts = final_result.artifacts or {}
                    if isinstance(final_result.artifacts, dict):
                        final_result.artifacts["mission_log"] = last_log
            except Exception:
                pass

        try:
            if final_result.success and self.rag_client and hasattr(self.rag_client, "add_memory"):
                mem_text = f"Intention: {intention}\nResult: success\nDetail: {final_result.detail}"
                self.rag_client.add_memory(mem_text)
        except Exception:
            pass

        try:
            end_ts = time.time()
            if self.mission_history:
                providers_tried = (
                    mission.brain_attempts if hasattr(mission, "brain_attempts") else []
                )
                tags = ["brain_router"] if providers_tried else []
                if waiting_for_user:
                    tags.append("waiting_for_user")
                else:
                    tags.append("success" if final_result.success else "failure")
                if mission.last_mission_error and mission.last_mission_error.kind:
                    tags.append(mission.last_mission_error.kind)
                final_error = final_result.error or (
                    mission.last_mission_error.reason if mission.last_mission_error else None
                )
                if final_error and str(final_error).startswith("driver_"):
                    tags.append("driver_error")
                metadata = {
                    "plan_source": selected_origin,
                    "plan_scores": scores_log,
                    "path": final_result.path,
                }
                if mission.ask_user_request:
                    metadata["ask_user_request"] = mission.ask_user_request.to_dict()
                if isinstance(final_result.detail, dict):
                    metadata["result_detail"] = final_result.detail
                self.mission_history.log_mission(
                    mission_id=mission.mission_id,
                    intent_text=intention,
                    mode=mode,
                    timestamp_start=mission.started_at,
                    timestamp_end=end_ts,
                    providers_tried=providers_tried,
                    final_status="waiting"
                    if waiting_for_user
                    else ("success" if final_result.success else "fail"),
                    final_error=final_error,
                    tags=tags,
                    metadata=metadata,
                )
        except Exception:
            try:
                self.log.warning("No se pudo registrar historial de misión", exc_info=True)
            except Exception:
                pass

        episode_receipt_path = None
        try:
            episode_receipt_path = self._emit_episode_and_receipt(
                mission, waiting_for_user=waiting_for_user
            )
        except Exception:
            try:
                self.log.warning("Episode/receipt emit failed", exc_info=True)
            except Exception:
                pass

        if final_result is not None and not final_result.success and not waiting_for_user:
            try:
                skill_id = (
                    mission.notes.get("library_skill_id")
                    if isinstance(mission.notes, dict)
                    else None
                )
            except Exception:
                skill_id = None
            if skill_id:
                try:
                    gap_path = self._emit_library_failure_gap(
                        mission,
                        final_result=final_result,
                        episode_receipt_path=episode_receipt_path,
                    )
                    if gap_path:
                        final_result.artifacts = final_result.artifacts or {}
                        if isinstance(final_result.artifacts, dict):
                            final_result.artifacts.setdefault("capability_gap", gap_path)
                except Exception:
                    pass

        try:
            self._maybe_auto_crystallize(mission, waiting_for_user=waiting_for_user)
        except Exception:
            try:
                self.log.warning("Auto-crystallize falló", exc_info=True)
            except Exception:
                pass

        if self.mission_breaker and intent_hash and final_result is not None:
            err_txt = final_result.error or (
                mission.last_mission_error.reason if mission.last_mission_error else None
            )
            if not final_result.success and not waiting_for_user:
                self.mission_breaker.mission_register_failure(
                    intent_hash,
                    time.time(),
                    intent_text=intention,
                    last_error=str(err_txt) if err_txt else None,
                )

        try:
            if isinstance(mission.notes, dict):
                self.state.notes["council_invoked"] = mission.notes.get("council_invoked")
                self.state.notes["council_invoked_reason"] = mission.notes.get(
                    "council_invoked_reason"
                )
        except Exception:
            pass

        self.snapshot()
        return final_result


# Singleton
_core_instance: Optional[AjaxCore] = None
_chat_lite_instance: Optional["AjaxChatLite"] = None


class AjaxChatLite(AjaxCore):
    def __init__(self, config: Optional[AjaxConfig] = None):
        self.log = logging.getLogger("ajax.chat_lite")
        self.config = config or AjaxConfig.from_env()
        self.root_dir = self.config.root_dir
        self.state_dir = self.config.state_dir
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.auto_crystallize_enabled = False
        self.lab_control = None
        self.health = None
        self.provider_configs = self._load_provider_configs()
        self._last_brain_selection_trace = None
        self.chat_history = []
        self.last_action_intent = None
        self._chat_leann_cache = {"ts": 0.0, "key": None, "snippets": [], "had_snippets": False}
        self._chat_leann_profile_cache = {"ts": 0.0, "key": None, "card": None}
        self.default_user_id = (
            getattr(self.config, "user_id", None)
            or os.getenv("AJAX_DEFAULT_USER")
            or os.getenv("AJAX_USER_ID")
            or "primary"
        )
        self._chat_intent_profiles = {}
        self._chat_user_labels = {}
        self.council = None
        self._tool_inventory = None
        self.security_policy = {}
        self.alert_level = "normal"
        self.mission_breaker = None
        self.constitution = ""
        self.orchestrator_prompt = MISSION_ORCHESTRATOR_PROMPT
        self.actions_catalog = {}
        self.driver = None
        self.actuator = None
        self.contract_enforcer = None
        self.rag_client = None
        self.capabilities = {}
        self.state = None
        self._last_brain_attempts = []
        self._chat_lite = True
        router_cls = self._get_brain_router_cls()
        self.brain_router = (
            router_cls.from_config(self.provider_configs, logger=self.log) if router_cls else None
        )


def wake_up() -> AjaxCore:
    global _core_instance
    if _core_instance is None:
        _core_instance = AjaxCore()
    return _core_instance


def wake_up_chat_lite() -> AjaxChatLite:
    global _chat_lite_instance
    if _chat_lite_instance is None:
        _chat_lite_instance = AjaxChatLite()
    return _chat_lite_instance
