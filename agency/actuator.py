# agency/actuator.py
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import logging

from .vision import Vision, VisionError, ElementNotFoundError
from .motor import Motor
from .feedback import Feedback, FeedbackResult, FeedbackError
from .motor_memory import MotorRoute, find_route_for_goal, mark_route_result
try:
    from .skill_fabricator import fabricate_skill  # type: ignore
except Exception:  # pragma: no cover
    fabricate_skill = None  # type: ignore

log = logging.getLogger(__name__)

ACTUATOR_ARTIFACTS_ROOT = Path("artifacts") / "actuator"


def _normalize_driver_base_url(raw: str) -> str:
    if not raw:
        return "http://127.0.0.1:5010"
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if parsed.netloc:
        return f"{parsed.scheme or 'http'}://{parsed.netloc}"
    return raw


def _resolve_windows_driver_base_url(explicit: str | None = None) -> str:
    if explicit:
        return _normalize_driver_base_url(explicit)
    env_url = os.getenv("OS_DRIVER_URL")
    if env_url:
        return _normalize_driver_base_url(env_url)
    env_host = os.getenv("OS_DRIVER_HOST")
    if env_host:
        env_port = os.getenv("OS_DRIVER_PORT") or "5010"
        return _normalize_driver_base_url(f"{env_host}:{env_port}")
    return "http://127.0.0.1:5010"


class Strategy(str, Enum):
    AUTO = "auto"
    FORCE_BALLISTIC = "force_ballistic"
    FORCE_COGNITIVE = "force_cognitive"


class ActuatorError(Exception):
    """Base error for actuator subsystem."""


class SafetyVetoError(ActuatorError):
    """Raised when the Safety Officer vetoes an action."""


@dataclass
class ActuatorContext:
    """Context for a single actuator execution."""
    goal: str
    strategy: Strategy = Strategy.AUTO
    run_id: Optional[str] = None  # could be a UUID
    artifacts_dir: Path = ACTUATOR_ARTIFACTS_ROOT
    capability: Optional[dict] = None


class Actuator:
    """
    Motor Cortex de AJAX.

    Alto nivel: decide ruta (Fast/Cognitive), coordina visión, motor y feedback,
    y escribe artifacts para self-audit.
    """

    def __init__(
        self,
        vision: Vision,
        motor: Motor,
        feedback: Feedback,
        safety_callback=None,
        capability_resolver=None,
    ) -> None:
        """
        safety_callback: callable opcional para pasar por el Safety Officer
        (p.ej. Gemini) antes de acciones sensibles.
        capability_resolver: callable opcional que devuelva info de capacidades
        (p.ej. governance_intel.capability_lookup).
        """
        self._vision = vision
        self._motor = motor
        self._feedback = feedback
        self._safety_callback = safety_callback
        self._capability_resolver = capability_resolver

    def execute(
        self,
        goal: str,
        strategy: Strategy = Strategy.AUTO,
    ) -> FeedbackResult:
        """
        Punto de entrada principal.

        goal: "Haz click en 'Archivo'", "Abre Spotify", etc.
        strategy:
          - AUTO: intenta primero Fast Path, si falla → Cognitive Path
          - FORCE_BALLISTIC: solo Fast Path (error si no puede)
          - FORCE_COGNITIVE: solo Cognitive Path
        """
        ctx = ActuatorContext(goal=goal, strategy=strategy)
        log.info("Actuator.execute start goal=%r strategy=%s", goal, strategy.value)

        # 0. Autoconocimiento: qué capacidades tenemos para este goal (no bloqueante)
        ctx.capability = self._resolve_capability(goal)

        # 1. Safety gate (placeholder: no hacemos nada todavía)
        self._maybe_run_safety_check(ctx)

        # 2. Selección de ruta
        if strategy == Strategy.FORCE_BALLISTIC:
            return self._run_ballistic(ctx, allow_fallback=False)
        if strategy == Strategy.FORCE_COGNITIVE:
            return self._run_cognitive(ctx)

        # AUTO: heurística simple por ahora (future: motor memory)
        try:
            return self._run_ballistic(ctx, allow_fallback=True)
        except (ActuatorError, VisionError, FeedbackError) as e:
            log.warning("Fast path failed for goal=%r: %s → falling back to cognitive", goal, e)
            return self._run_cognitive(ctx)
        except Exception as exc:
            # Intentar fabricar skill y reintentar una vez
            if fabricate_skill is not None:
                fab = fabricate_skill(goal, {"stage": "auto_fastpath", "error": str(exc)})
                if fab.get("ok"):
                    try:
                        return self._run_ballistic(ctx, allow_fallback=True)
                    except Exception:
                        pass
            raise

    # -------------------------------
    # Paths
    # -------------------------------
    def _run_ballistic(self, ctx: ActuatorContext, allow_fallback: bool) -> FeedbackResult:
        """
        Fast Path: solo heurísticas locales (sin LLM vision).
        Aquí solo definimos el esqueleto.
        """
        log.debug("Actuator._run_ballistic goal=%r", ctx.goal)

        # 0) Intentar ruta memorizada (MotorMemory)
        route = self._find_route(ctx.goal)
        if route:
            log.info("MotorMemory hit for goal=%r route=%s", ctx.goal, route.name)
            before_img = self._vision.capture()
            try:
                self._execute_route(route)
                after_img = self._vision.capture()
                result = self._feedback.compare(before_img, after_img)
                if result.changed:
                    self._feedback.save_artifacts(ctx, before_img, after_img, result)
                    mark_route_result(route, True)
                    return result
                mark_route_result(route, False)
                if not allow_fallback:
                    raise ActuatorError("Motor route produced no visible change")
                log.info("Motor route produced no change; fallback to vision fast path")
            except Exception as exc:
                mark_route_result(route, False)
                if not allow_fallback:
                    raise ActuatorError(f"Motor route failed: {exc}") from exc
                log.warning("Motor route failed (%s); fallback to vision fast path", exc)

        before_img = self._vision.capture()
        coords = None

        # Sonar semántico (accesibilidad) antes del OCR rápido
        if hasattr(self._vision, "find_semantic"):
            try:
                coords = self._vision.find_semantic(ctx.goal)  # type: ignore[attr-defined]
            except Exception as exc:
                log.debug("find_semantic failed for goal=%r: %s", ctx.goal, exc)

        if coords is None:
            # TODO: implementar motor memory / patrones conocidos
            coords = self._vision.find_fast(ctx.goal, before_img)

        # TODO: propiocepción diferencial (visual hash local antes de actuar)
        # hash_ok = self._feedback.check_visual_hash(...)
        # if not hash_ok: raise ActuatorError("Visual hash mismatch")

        self._motor.move(coords)
        self._motor.click()

        after_img = self._vision.capture()
        result = self._feedback.compare(before_img, after_img)

        if not result.changed:
            if allow_fallback:
                raise ActuatorError("Fast path produced no visible change")
            raise ActuatorError("Fast path failed with no visible change")

        self._feedback.save_artifacts(ctx, before_img, after_img, result)
        return result

    def _run_cognitive(self, ctx: ActuatorContext) -> FeedbackResult:
        """
        Cognitive Path: puede llamar a LLMs de visión vía Vision.
        """
        log.debug("Actuator._run_cognitive goal=%r", ctx.goal)

        before_img = self._vision.capture()
        coords = self._vision.find_cognitive(ctx.goal, before_img)

        self._motor.move(coords)
        self._motor.click()

        after_img = self._vision.capture()
        result = self._feedback.compare(before_img, after_img)
        self._feedback.save_artifacts(ctx, before_img, after_img, result)

        return result

    # -------------------------------
    # Safety
    # -------------------------------
    def _maybe_run_safety_check(self, ctx: ActuatorContext) -> None:
        """
        Hook de seguridad: delega en Gemini/Qwen/etc si la acción es peligrosa.
        De momento solo está el gancho.
        """
        if self._safety_callback is None:
            return
        veto = self._safety_callback(ctx.goal)
        if veto:
            raise SafetyVetoError(f"Safety veto for goal: {ctx.goal!r}")

    def _find_route(self, goal: str) -> MotorRoute | None:
        try:
            return find_route_for_goal(goal)
        except Exception:
            return None

    def _execute_route(self, route: MotorRoute) -> None:
        for step in route.steps:
            op = step.op
            args = step.args or {}
            if op == "mouse_move":
                x = int(args.get("x", 0))
                y = int(args.get("y", 0))
                self._motor.move((x, y))
            elif op == "mouse_click":
                # Permitir coords opcionales
                x = args.get("x")
                y = args.get("y")
                if x is not None and y is not None:
                    self._motor.move((int(x), int(y)))
                button = args.get("button", "left")
                self._motor.click(button=button)
            elif op == "keyboard_type":
                text = str(args.get("text", ""))
                submit = bool(args.get("submit", False))
                self._motor.type(text)
                if submit:
                    self._motor.type("\n")
            else:
                raise ActuatorError(f"Unknown motor step op={op!r}")

    def _resolve_capability(self, goal: str) -> Optional[dict]:
        if self._capability_resolver is None:
            return None
        try:
            return self._capability_resolver(goal)
        except Exception as exc:
            log.debug("capability_resolver failed for goal %r: %s", goal, exc)
            return None


# -----------------------------------
# Helpers
# -----------------------------------
def create_default_actuator(
    artifacts_dir: Path = ACTUATOR_ARTIFACTS_ROOT,
    windows_base_url: str | None = None,
    capability_resolver=None,
) -> "Actuator":
    """
    Construye un Actuator con backend apropiado según el sistema operativo.
    - win32 → usa driver FastAPI (drivers/os_driver.py)
    - otros → usa impl local (safe_pyautogui + Xvfb)
    """
    if sys.platform == "win32":
        try:
            from .vision_windows import VisionWindows, VisionWindowsConfig  # type: ignore
            from .motor_windows import MotorWindows, MotorWindowsConfig  # type: ignore
            base_url = _resolve_windows_driver_base_url(windows_base_url)
            v_cfg = VisionWindowsConfig(base_url=base_url)
            m_cfg = MotorWindowsConfig(base_url=base_url)
            vision = VisionWindows(v_cfg)
            motor = MotorWindows(m_cfg)
            log.info("Actuator factory: using Windows driver backend at %s", v_cfg.base_url)
        except Exception as exc:
            log.warning("Actuator factory: Windows backend unavailable (%s), falling back to Linux backend", exc)
            from .vision import VisionImpl
            from .motor import Motor
            vision = VisionImpl()
            motor = Motor()
    else:
        from .vision import VisionImpl
        from .motor import Motor
        vision = VisionImpl()
        motor = Motor()

    feedback = Feedback(artifacts_dir)
    return Actuator(vision, motor, feedback, capability_resolver=capability_resolver)
