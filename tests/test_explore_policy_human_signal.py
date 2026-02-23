from agency.explore_policy import compute_human_active


def test_compute_human_active_ignores_fields_when_signal_not_ok() -> None:
    active, reason = compute_human_active(
        {
            "ok": False,
            "last_input_age_sec": 0,
            "session_unlocked": True,
            "error": "stub_fail_closed",
        },
        threshold_s=90,
        unknown_as_human=False,
    )
    assert active is False
    assert reason == "signal_not_ok"


def test_compute_human_active_respects_policy_when_signal_not_ok() -> None:
    active, reason = compute_human_active(
        {"ok": False, "last_input_age_sec": 0, "session_unlocked": True},
        threshold_s=90,
        unknown_as_human=True,
    )
    assert active is True
    assert reason == "signal_not_ok"
