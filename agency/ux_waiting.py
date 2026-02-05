from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _normalize_options(options: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(options, list):
        return out
    for opt in options:
        if not isinstance(opt, dict):
            continue
        opt_id = str(opt.get("id") or "").strip()
        label = str(opt.get("label") or "").strip()
        if not opt_id:
            continue
        out.append({"id": opt_id, "label": label})
    return out


def _filter_budget_exhausted(options: List[Dict[str, str]]) -> List[Dict[str, str]]:
    allow = {"open_incident", "close_manual_done"}
    filtered: List[Dict[str, str]] = []
    seen = set()
    for opt in options:
        opt_id = str(opt.get("id") or "").strip().lower()
        if opt_id not in allow:
            continue
        if opt_id in seen:
            continue
        seen.add(opt_id)
        filtered.append(opt)
    if "open_incident" not in seen:
        filtered.append({"id": "open_incident", "label": "Abrir INCIDENT y derivar a LAB para triage automático"})
    if "close_manual_done" not in seen:
        filtered.append({"id": "close_manual_done", "label": "Cerrar misión manualmente (ya realizado)"})
    return filtered


def _filter_loop_guard(options: List[Dict[str, str]]) -> List[Dict[str, str]]:
    allow_ids = {"retry_fresh", "open_incident", "close_manual_done", "use_deterministic_recipe"}
    out: List[Dict[str, str]] = []
    for opt in options:
        opt_id = str(opt.get("id") or "").strip().lower()
        if opt_id in allow_ids:
            out.append(opt)
    return out


def render_waiting_panel(
    detail: Dict[str, Any],
    root_dir: Path,
    *,
    debug: bool = False,
    compact: bool = False,
    has_lab_job: bool = False,
) -> List[str]:
    question = str(detail.get("question") or "Esperando confirmación humana.")
    expects = str(detail.get("expects") or "menu_choice").strip().lower()
    budget_exhausted = "[budget_exhausted]" in question.lower()
    loop_guard = bool(detail.get("loop_guard") or detail.get("loop_guard_triggered"))
    options = _normalize_options(detail.get("options") or [])
    if budget_exhausted and not loop_guard:
        options = _filter_budget_exhausted(options)
        expects = "user_answer"
    if loop_guard:
        options = _filter_loop_guard(options)
    default_opt = detail.get("default_option")
    if budget_exhausted:
        default_opt = "open_incident"
    if loop_guard and budget_exhausted:
        default_opt = "retry_fresh"

    lines: List[str] = []
    if loop_guard:
        lines.append(f"⏳ Loop guard activo: {question}")
    else:
        lines.append(f"⏳ Esperando tu decisión: {question}")

    max_opts = 3 if compact else None
    for idx, opt in enumerate(options):
        if max_opts is not None and idx >= max_opts:
            break
        opt_id = opt.get("id") or "opción"
        label = opt.get("label") or ""
        lines.append(f"   - [{opt_id}] {label}".rstrip())
    if max_opts is not None and len(options) > max_opts:
        lines.append(f"   - (+{len(options) - max_opts} opciones más)")

    if expects == "user_answer":
        lines.append("   - Responde escribiendo texto, o usa [answer] <texto>")

    if compact:
        shortcuts = ["[status]", "[ack_user]", "[close_manual_done]", "[snap]"]
        if loop_guard:
            shortcuts.insert(1, "[park]")
        if has_lab_job:
            shortcuts.append("[snap_lab]")
        lines.append("   Comandos: " + " ".join(shortcuts))
    else:
        if loop_guard:
            lines.append("   - [park] Aparcar misión y volver a chat")
            lines.append("   - O usa :park para aparcar la misión y volver a chat")
        lines.append("   - [status] Ver estado de LAB (si aplica)")
        lines.append("   - [ack_user] Confirmar acción manual (sin LAB)")
        if not any(str(opt.get("id") or "").strip().lower() == "close_manual_done" for opt in options):
            lines.append("   - [close_manual_done] Cerrar misión manualmente (sin LAB)")
        if loop_guard and "retry_fresh" not in {str(opt.get("id") or "").strip().lower() for opt in options}:
            lines.append("   - [retry_fresh] Reintentar como nueva misión (reset de presupuesto)")
        lines.append("   - [snap] Capturar screenshot del escritorio")
        lines.append("   - [snap+vision] Capturar y describir con vision (opcional)")
        if has_lab_job:
            lines.append("   - [snap_lab] Capturar screenshot del escritorio LAB")
            lines.append("   - [snap_lab+vision] Capturar y describir LAB con vision (opcional)")

    if default_opt:
        lines.append(f"   (Opción por defecto: {default_opt})")

    payload_path = detail.get("waiting_payload_path") or detail.get("waiting_mission_path")
    if payload_path and debug:
        try:
            rel = str(Path(payload_path))
            if root_dir:
                rel = str(Path(payload_path)).replace(str(root_dir), "").lstrip("/\\")
        except Exception:
            rel = str(payload_path)
        lines.append(f"   → Detalle estructurado: {rel}")

    return lines


def resolve_enter_default(
    *,
    stripped: str,
    default_option: Optional[str],
    pending_default_confirm: bool,
    enter_behavior: str,
) -> Tuple[Optional[str], bool, str]:
    """
    Returns (action, new_pending_confirm, message)
    action: None | "confirm" | "execute"
    """
    if stripped:
        return None, pending_default_confirm, ""
    if enter_behavior == "default" and default_option:
        if not pending_default_confirm:
            return "confirm", True, f"Pulsa Enter otra vez para confirmar default: [{default_option}]"
        return "execute", False, f"[{default_option}]"
    return None, pending_default_confirm, "Escribe tu respuesta o usa un comando entre corchetes."
