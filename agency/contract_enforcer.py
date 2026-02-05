from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

from agency.human_permission import human_permission_gate_enabled, read_human_permission_status


EnforcementAction = Literal["allow", "wait_user", "defer", "derive_to_lab"]


@dataclass(frozen=True)
class EnforcementDecision:
    """
    Fachada fina (ContractEnforcer): NO duplica lógica del sistema; solo orquesta gates existentes
    y devuelve una decisión estructurada para que el caller (p.ej. AjaxCore) aplique efectos.
    """

    action: EnforcementAction
    reason: str
    question: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    clear_pending_plan: bool = False


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_rail(raw: Optional[str]) -> str:
    val = (raw or "").strip().lower()
    if val in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def _detect_rdp_active(timeout_sec: int = 2) -> Dict[str, Any]:
    """
    Best-effort: intenta detectar una sesión RDP activa en Windows.
    No debe lanzar; devuelve dict con evidencia minimal.
    """
    candidates = [
        ("query_session", ["query.exe", "session"]),
        ("qwinsta", ["qwinsta.exe"]),
        ("quser", ["quser.exe"]),
    ]
    for name, cmd in candidates:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_sec)),
                check=False,
            )
        except FileNotFoundError:
            continue
        except Exception as exc:
            return {"ok": False, "detector": name, "error": str(exc)[:200]}

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        text = f"{stdout}\n{stderr}".lower()
        active = ("rdp-tcp" in text) and ("active" in text)
        return {
            "ok": bool(active),
            "detector": name,
            "rc": proc.returncode,
            "stdout": stdout[:2000],
            "stderr": stderr[:500],
        }
    return {"ok": False, "error": "no_windows_rdp_detector"}


class ContractEnforcer:
    def __init__(
        self,
        *,
        root_dir: Path,
        rdp_detector: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.root_dir = root_dir
        self._rdp_detector = rdp_detector or _detect_rdp_active

    def enforce_council_consensus(self, council_meta: Any) -> EnforcementDecision:
        """
        Consenso → allow.
        No-consenso:
          - await_user_input / unsafe / try_stronger_model → defer (pursuit/ASK_USER/GIVE_UP)
          - resto → derive_to_lab
        """
        if not isinstance(council_meta, dict):
            return EnforcementDecision(action="derive_to_lab", reason="council_verdict_missing")
        verdict = str(council_meta.get("verdict") or "").strip().lower()
        if not verdict:
            return EnforcementDecision(action="derive_to_lab", reason="council_verdict_missing")
        if verdict == "approve":
            return EnforcementDecision(action="allow", reason="council_approved")

        hint = str(council_meta.get("escalation_hint") or "").strip().lower()
        if hint in {"await_user_input", "unsafe", "try_stronger_model"}:
            return EnforcementDecision(
                action="defer",
                reason=f"council_{hint}",
                detail={
                    "verdict": council_meta.get("verdict"),
                    "escalation_hint": council_meta.get("escalation_hint"),
                    "reason": council_meta.get("reason"),
                },
                clear_pending_plan=True,
            )

        # Quorum/provider outage: no derivar a LAB por defecto; degradación explícita (ASK_USER / retry).
        reason_txt = str(council_meta.get("reason") or "")
        degraded_reason = str(council_meta.get("council_degraded_reason") or "")
        if reason_txt.startswith("BLOCKED_BY_COUNCIL_QUORUM") or reason_txt.startswith("BLOCKED_BY_COUNCIL_INVALID_REVIEW") or degraded_reason.startswith("ledger_"):
            return EnforcementDecision(
                action="defer",
                reason="council_quorum_unavailable",
                detail={
                    "verdict": council_meta.get("verdict"),
                    "escalation_hint": council_meta.get("escalation_hint"),
                    "reason": council_meta.get("reason"),
                    "council_degraded_reason": council_meta.get("council_degraded_reason"),
                },
                clear_pending_plan=False,
            )

        return EnforcementDecision(
            action="derive_to_lab",
            reason=str(council_meta.get("reason") or "council_no_consensus").strip() or "council_no_consensus",
            detail={
                "verdict": council_meta.get("verdict"),
                "escalation_hint": council_meta.get("escalation_hint"),
            },
        )

    def enforce_lab_display(self, *, rail: str, plan_requires_physical_actions: bool) -> EnforcementDecision:
        """
        Gate LAB-actuation (sin HDMI dummy):
        - Solo aplica a rail=lab.
        - Si hay HDMI dummy (AJAX_HDMI_DUMMY_PRESENT=1) ⇒ allow.
        - Si `AJAX_LAB_DISPLAY_READY=1` ⇒ allow (override explícito).
        - Si hay RDP activa ⇒ allow (best-effort).
        - Si no ⇒ wait_user (fail-closed).
        """
        if not plan_requires_physical_actions:
            return EnforcementDecision(action="allow", reason="no_physical_actions")

        if _normalize_rail(rail) != "lab":
            return EnforcementDecision(action="allow", reason="not_lab")

        if _env_truthy("AJAX_HDMI_DUMMY_PRESENT"):
            return EnforcementDecision(action="allow", reason="hdmi_dummy_present")

        if _env_truthy("AJAX_LAB_DISPLAY_READY"):
            return EnforcementDecision(
                action="allow",
                reason="lab_display_ready_env",
                detail={"source": "AJAX_LAB_DISPLAY_READY"},
            )

        det = self._rdp_detector()
        if bool(det.get("ok")):
            return EnforcementDecision(action="allow", reason="lab_display_ready_rdp", detail=det)

        q = (
            "[LAB_DISPLAY] LAB-actuation bloqueada: no hay señal de display activo para rail=lab (sin HDMI dummy).\n"
            "Abre una sesión RDP a la cuenta/sesión de LAB, o exporta `AJAX_LAB_DISPLAY_READY=1`, y reintenta la misma misión."
        )
        return EnforcementDecision(action="wait_user", reason="lab_display_unavailable", question=q, detail=det)

    def enforce_human_permission(self, *, plan_requires_physical_actions: bool) -> EnforcementDecision:
        """
        Pixel-safety: requiere permiso humano TTL para acciones físicas cuando no hay HDMI dummy.
        """
        if not plan_requires_physical_actions:
            return EnforcementDecision(action="allow", reason="no_physical_actions")
        if not human_permission_gate_enabled():
            return EnforcementDecision(action="allow", reason="human_permission_gate_disabled")

        perm = read_human_permission_status(self.root_dir)
        if bool(perm.get("ok", False)):
            return EnforcementDecision(action="allow", reason="human_permission_ok", detail=perm)

        q = (
            "[HUMAN_PERMISSION] Pixel-safety activo: para ejecutar acciones físicas (click/type/launch) "
            "necesito permiso humano temporal.\n"
            "Ejecuta: ajaxctl permit 120\n"
            "Luego reintenta la misma misión (ajaxctl do <cualquier texto>) para reanudar."
        )
        return EnforcementDecision(action="wait_user", reason="human_permission_required", question=q, detail={"permission": perm})
