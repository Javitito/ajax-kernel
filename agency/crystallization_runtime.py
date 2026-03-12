from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from agency.crystallization import (
        CrystallizationEngine,
        emit_crystallization_receipt,
    )
except ImportError:  # pragma: no cover
    CrystallizationEngine = None  # type: ignore
    emit_crystallization_receipt = None  # type: ignore

LAB_RAILS = {"lab"}
FALSEY_VALUES = {"0", "false", "off", "no", ""}


def _normalized_rail(rail: Optional[str]) -> str:
    return str(rail or "lab").strip().lower() or "lab"


def load_auto_crystallize_flag(state_dir: Path | str) -> bool:
    env = os.getenv("AJAX_AUTO_CRYSTALLIZE")
    default_enabled = True
    if env is not None:
        default_enabled = env.strip().lower() not in FALSEY_VALUES
    flag_path = Path(state_dir) / "auto_crystallize.flag"
    if flag_path.exists():
        try:
            raw = flag_path.read_text(encoding="utf-8").strip().lower()
        except Exception:
            return default_enabled
        return raw not in FALSEY_VALUES
    return default_enabled


def emit_crystallization_considered_receipt(
    root_dir: Path | str,
    *,
    mission_id: str,
    trigger: str,
    rail: Optional[str],
    auto_crystallize_enabled: bool,
    waiting_for_user: bool,
    plan_id: Optional[str] = None,
    plan_source: Optional[str] = None,
    skill_id: Optional[str] = None,
) -> Optional[str]:
    if emit_crystallization_receipt is None:
        return None
    return emit_crystallization_receipt(
        root_dir,
        event="crystallization_considered",
        mission_id=mission_id,
        trigger=trigger,
        rail=_normalized_rail(rail),
        auto_crystallize_enabled=bool(auto_crystallize_enabled),
        decision="considered",
        detail={
            "waiting_for_user": bool(waiting_for_user),
            "plan_id": plan_id,
            "plan_source": plan_source,
            "skill_id": skill_id,
        },
    )


def _emit_skip_flow_receipts(
    root_dir: Path | str,
    *,
    mission_id: str,
    trigger: str,
    rail: Optional[str],
    auto_crystallize_enabled: bool,
    reason: str,
    detail: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if emit_crystallization_receipt is None:
        return []
    base = {
        "root_dir": root_dir,
        "mission_id": mission_id,
        "trigger": trigger,
        "rail": _normalized_rail(rail),
        "auto_crystallize_enabled": bool(auto_crystallize_enabled),
        "reason": reason,
        "detail": detail or {},
    }
    paths = [
        emit_crystallization_receipt(
            base["root_dir"],
            event="episode_skipped",
            mission_id=base["mission_id"],
            trigger=base["trigger"],
            rail=base["rail"],
            auto_crystallize_enabled=base["auto_crystallize_enabled"],
            decision="skipped",
            reason=base["reason"],
            detail=base["detail"],
        ),
        emit_crystallization_receipt(
            base["root_dir"],
            event="candidate_recipe_skipped",
            mission_id=base["mission_id"],
            trigger=base["trigger"],
            rail=base["rail"],
            auto_crystallize_enabled=base["auto_crystallize_enabled"],
            decision="skipped",
            reason=base["reason"],
            detail=base["detail"],
        ),
        emit_crystallization_receipt(
            base["root_dir"],
            event="validation_result",
            mission_id=base["mission_id"],
            trigger=base["trigger"],
            rail=base["rail"],
            auto_crystallize_enabled=base["auto_crystallize_enabled"],
            decision="refused",
            reason=base["reason"],
            detail={"validation_status": "refused", **base["detail"]},
        ),
    ]
    return [path for path in paths if path]


def maybe_auto_crystallize(
    root_dir: Path | str,
    *,
    mission_id: str,
    rail: Optional[str],
    waiting_for_user: bool,
    auto_crystallize_enabled: bool,
    emit_gap: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    normalized_rail = _normalized_rail(rail)
    summary: Dict[str, Any] = {
        "triggered": False,
        "mission_id": mission_id,
        "rail": normalized_rail,
        "auto_crystallize_enabled": bool(auto_crystallize_enabled),
        "receipt_paths": [],
    }
    detail = {"waiting_for_user": bool(waiting_for_user)}
    if waiting_for_user:
        summary["skip_reason"] = "waiting_boundary_open"
        summary["receipt_paths"] = _emit_skip_flow_receipts(
            root_dir,
            mission_id=mission_id,
            trigger="auto",
            rail=normalized_rail,
            auto_crystallize_enabled=auto_crystallize_enabled,
            reason="waiting_boundary_open",
            detail=detail,
        )
        return summary
    if not auto_crystallize_enabled:
        summary["skip_reason"] = "auto_crystallize_disabled"
        summary["receipt_paths"] = _emit_skip_flow_receipts(
            root_dir,
            mission_id=mission_id,
            trigger="auto",
            rail=normalized_rail,
            auto_crystallize_enabled=auto_crystallize_enabled,
            reason="auto_crystallize_disabled",
            detail=detail,
        )
        return summary
    if normalized_rail not in LAB_RAILS:
        summary["skip_reason"] = "rail_not_lab"
        summary["receipt_paths"] = _emit_skip_flow_receipts(
            root_dir,
            mission_id=mission_id,
            trigger="auto",
            rail=normalized_rail,
            auto_crystallize_enabled=auto_crystallize_enabled,
            reason="rail_not_lab",
            detail=detail,
        )
        return summary
    if CrystallizationEngine is None:
        summary["skip_reason"] = "crystallization_unavailable"
        summary["receipt_paths"] = _emit_skip_flow_receipts(
            root_dir,
            mission_id=mission_id,
            trigger="auto",
            rail=normalized_rail,
            auto_crystallize_enabled=auto_crystallize_enabled,
            reason="crystallization_unavailable",
            detail=detail,
        )
        return summary
    try:
        engine = CrystallizationEngine(root_dir)
        result = engine.crystallize_mission(mission_id, trigger="auto", require_lab=True)
    except Exception as exc:
        error = str(exc)[:400]
        summary["skip_reason"] = "crystallization_failed"
        summary["error"] = error
        if emit_gap is not None:
            try:
                emit_gap(error)
            except Exception:
                pass
        summary["receipt_paths"] = _emit_skip_flow_receipts(
            root_dir,
            mission_id=mission_id,
            trigger="auto",
            rail=normalized_rail,
            auto_crystallize_enabled=auto_crystallize_enabled,
            reason="crystallization_failed",
            detail={"error": error},
        )
        return summary
    summary.update(result if isinstance(result, dict) else {})
    summary["triggered"] = True
    receipt_paths = result.get("receipt_paths") if isinstance(result, dict) else None
    if isinstance(receipt_paths, list):
        summary["receipt_paths"] = [path for path in receipt_paths if path]
    return summary


__all__ = [
    "emit_crystallization_considered_receipt",
    "load_auto_crystallize_flag",
    "maybe_auto_crystallize",
]
