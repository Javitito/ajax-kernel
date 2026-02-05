from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JitPermitDecision:
    eligible: bool
    requires_confirmation: bool
    auto_permit: bool
    suppress_action_required: bool
    reason: str


def evaluate_jit_permit(
    *,
    action_required: bool,
    user_id: str,
    rail: str,
    risk_level: str,
    interactive: bool,
) -> JitPermitDecision:
    if not action_required:
        return JitPermitDecision(
            eligible=False,
            requires_confirmation=False,
            auto_permit=False,
            suppress_action_required=False,
            reason="action_not_required",
        )

    if not interactive:
        return JitPermitDecision(
            eligible=False,
            requires_confirmation=False,
            auto_permit=False,
            suppress_action_required=False,
            reason="non_interactive",
        )

    if (user_id or "").strip().lower() != "primary":
        return JitPermitDecision(
            eligible=False,
            requires_confirmation=False,
            auto_permit=False,
            suppress_action_required=False,
            reason="non_primary_user",
        )

    rail_norm = (rail or "lab").strip().lower()
    if rail_norm == "lab":
        return JitPermitDecision(
            eligible=False,
            requires_confirmation=False,
            auto_permit=False,
            suppress_action_required=False,
            reason="rail_lab",
        )

    risk_norm = (risk_level or "medium").strip().lower()
    if risk_norm == "high":
        return JitPermitDecision(
            eligible=True,
            requires_confirmation=True,
            auto_permit=False,
            suppress_action_required=False,
            reason="high_risk_requires_confirmation",
        )

    return JitPermitDecision(
        eligible=True,
        requires_confirmation=False,
        auto_permit=True,
        suppress_action_required=True,
        reason="auto_permit_low_medium",
    )


def confirm_jit_permit(decision: JitPermitDecision, confirmed: bool) -> JitPermitDecision:
    if not decision.eligible:
        return decision
    if not decision.requires_confirmation:
        return decision
    if confirmed:
        return JitPermitDecision(
            eligible=True,
            requires_confirmation=False,
            auto_permit=True,
            suppress_action_required=True,
            reason="confirmed",
        )
    return JitPermitDecision(
        eligible=False,
        requires_confirmation=False,
        auto_permit=False,
        suppress_action_required=False,
        reason="confirmation_denied",
    )
