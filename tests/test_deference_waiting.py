from __future__ import annotations

import types

import agency.ajax_core as ajax_core_mod
from agency.ajax_core import (
    AjaxCore,
    AjaxExecutionResult,
    AjaxObservation,
    AjaxPlan,
    MissionState,
    MissionStep,
    _AWAIT_USER_SENTINEL,
)


class _LogStub:
    def info(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def warning(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def debug(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


def _build_core() -> AjaxCore:
    core = AjaxCore.__new__(AjaxCore)
    core.log = _LogStub()
    core.perceive = types.MethodType(
        lambda self: AjaxObservation(timestamp=0.0, foreground=None, notes={}),
        core,
    )
    core._starting_xi_role_primary = types.MethodType(lambda self, sx, role: None, core)
    core._plan_requires_vision_player = types.MethodType(lambda self, plan: False, core)
    core._plan_has_ui_actions = types.MethodType(lambda self, plan: False, core)
    core._infer_intent_class = types.MethodType(lambda self, intent: "general", core)
    core._record_council_invocation = types.MethodType(lambda self, mission, **kwargs: None, core)
    core._record_exec_receipt = types.MethodType(lambda self, **kwargs: None, core)
    core._provider_failure_policy = types.MethodType(lambda self: {}, core)
    core._emit_brain_failed_no_plan_gap = types.MethodType(
        lambda self, mission, **kwargs: "artifacts/capability_gaps/gap_test.json",
        core,
    )

    def _ask_wait(
        self: AjaxCore,
        mission: MissionState,
        question: str,
        *,
        source: str,
        blocking_reason: str | None = None,
        extra_context: dict | None = None,
    ):
        _ = (question, source, blocking_reason, extra_context)
        mission.status = "WAITING_FOR_USER"
        mission.last_result = AjaxExecutionResult(
            success=False,
            error="await_user_input",
            path="waiting_for_user",
        )
        return mission.last_result

    core._finalize_ask_user_wait = types.MethodType(_ask_wait, core)
    return core


def _brain_failed_no_plan() -> AjaxPlan:
    return AjaxPlan(
        id="p1",
        plan_id="p1",
        summary="no-plan",
        steps=[],
        metadata={
            "planning_error": "brain_failed_no_plan",
            "brain_failed_no_plan": True,
            "provider_failures": [{"error_code": "quota_exhausted"}],
            "errors": ["quota_exhausted"],
        },
    )


def test_deference_human_present_forces_waiting(monkeypatch) -> None:
    core = _build_core()
    core._human_present_now = types.MethodType(lambda self: True, core)
    monkeypatch.setattr(
        ajax_core_mod,
        "failure_on_no_plan_terminal",
        lambda policy, default="WAITING_FOR_USER": "GAP_LOGGED",
        raising=True,
    )

    mission = MissionState(intention="resolver proveedor", mode="auto")
    out = core._execute_action_step(
        mission,
        MissionStep(kind="EXECUTE_ACTION", action=_brain_failed_no_plan()),
    )

    assert out == _AWAIT_USER_SENTINEL
    assert mission.status == "WAITING_FOR_USER"
    assert mission.last_result is not None
    assert mission.last_result.path == "waiting_for_user"


def test_deference_without_human_keeps_gap_logged(monkeypatch) -> None:
    core = _build_core()
    core._human_present_now = types.MethodType(lambda self: False, core)
    monkeypatch.setattr(
        ajax_core_mod,
        "failure_on_no_plan_terminal",
        lambda policy, default="WAITING_FOR_USER": "GAP_LOGGED",
        raising=True,
    )

    mission = MissionState(intention="resolver proveedor", mode="auto")
    out = core._execute_action_step(
        mission,
        MissionStep(kind="EXECUTE_ACTION", action=_brain_failed_no_plan()),
    )

    assert out is None
    assert mission.status == "GAP_LOGGED"
    assert mission.last_result is not None
    assert mission.last_result.error == "NO_PLAN"
