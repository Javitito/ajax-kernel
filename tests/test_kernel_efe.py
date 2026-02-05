from agency.expected_state import verify_efe


def test_efe_verify_file_expectation_ok():
    ok, delta = verify_efe(
        {"files": [{"path": "pyproject.toml", "must_exist": True}]},
        driver=None,
        timeout_s=0.1,
        poll_interval_s=0.01,
    )
    assert ok is True
    assert delta is None
