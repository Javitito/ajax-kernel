from __future__ import annotations

import types

from agency.ajax_core import AjaxCore, AjaxExecutionResult, MissionState, MissionStep


def test_pursue_intent_waiting_for_user_even_without_sentinel() -> None:
    core = AjaxCore.__new__(AjaxCore)

    def choose_step(self: AjaxCore, mission: MissionState) -> MissionStep:
        return MissionStep(kind="EXECUTE_ACTION", action=None)

    def execute_step(self: AjaxCore, mission: MissionState, step: MissionStep):
        mission.status = "WAITING_FOR_USER"
        mission.last_result = AjaxExecutionResult(
            success=False,
            error="await_user_input",
            path="waiting_for_user",
        )
        return None

    core.choose_next_step = types.MethodType(choose_step, core)
    core._execute_action_step = types.MethodType(execute_step, core)

    mission = MissionState(intention="necesito confirmacion", mode="auto", envelope=None)
    result = core.pursue_intent(mission)

    assert result.error == "await_user_input"
    assert result.path == "waiting_for_user"
    assert mission.status == "WAITING_FOR_USER"
