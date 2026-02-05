from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agency.vision_llm import call_vision_llm  # type: ignore


@dataclass
class VisualAuditResult:
    success: bool
    reason: str


class VisualAuditor:
    """
    Auditor híbrido: usa telemetría básica primero y luego verificación visual
    mediante el VLM configurado.
    """

    def __init__(self, driver, backend: str = "local"):
        self.driver = driver
        self.backend = backend

    def verify(self, goal: str, description: str) -> VisualAuditResult:
        desc = description or goal
        # 1) Telemetría básica
        if self._telemetry_success(desc):
            return VisualAuditResult(True, "telemetry_condition_met")

        # 2) Screenshot + visión
        try:
            snap = self.driver.screenshot()
            prompt = (
                f"User intent: '{goal}'. Screenshot attached. "
                f"Does the image visually confirm the intent was successful? "
                f"Expected description: '{description}'. "
                f"Reply JSON {{\"success\": bool, \"reason\": \"...\"}}"
            )
            resp = call_vision_llm(
                image_path=str(snap),
                user_prompt=prompt,
                backend=self.backend,
            )
            content = resp.get("text") or resp.get("response") or json.dumps(resp)
            try:
                data = json.loads(content) if isinstance(content, str) else resp
            except Exception:
                data = resp
            success = bool(data.get("success", False) or data.get("satisfied", False))
            reason = data.get("reason") or "visual_audit"
            return VisualAuditResult(success, reason)
        except Exception as exc:
            return VisualAuditResult(False, f"visual_audit_error:{exc}")

    def _telemetry_success(self, description: str) -> bool:
        if not self.driver:
            return False
        desc = description.lower()
        try:
            fg = self.driver.get_active_window()
        except Exception:
            fg = None
        title = str((fg or {}).get("title") or "").lower()
        if any(token in desc for token in ("minimiza", "minimizado", "sin ventanas", "desktop", "escritorio")):
            if not title or "desktop" in title or "escritorio" in title:
                return True
        return False
