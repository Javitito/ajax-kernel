import json

from agency.subcall import _extract_codex_jsonl, _parse_json_object_or_event_wrapped


def test_extract_codex_jsonl_parses_jsonl_item_completed() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "assistant_message", "text": '{"ok":true,"errors":[]}'},
                }
            ),
        ]
    )
    text, err = _extract_codex_jsonl(raw)
    assert err is None
    assert text == '{"ok":true,"errors":[]}'


def test_extract_codex_jsonl_parses_single_json_array_of_events() -> None:
    raw = json.dumps(
        [
            {"type": "system", "subtype": "init", "session_id": "abc"},
            {
                "type": "item.completed",
                "item": {"type": "assistant_message", "text": '{"ok":true,"errors":[]}'},
            },
        ]
    )
    text, err = _extract_codex_jsonl(raw)
    assert err is None
    assert text == '{"ok":true,"errors":[]}'


def test_extract_codex_jsonl_parses_content_parts() -> None:
    raw = json.dumps(
        [
            {
                "type": "item.completed",
                "item": {
                    "type": "assistant_message",
                    "content": [
                        {"type": "output_text", "text": "{\n"},
                        {"type": "output_text", "text": '  "ok": true,\n  "errors": []\n}'},
                    ],
                },
            }
        ]
    )
    text, err = _extract_codex_jsonl(raw)
    assert err is None
    assert '"ok": true' in (text or "")
    assert '"errors": []' in (text or "")


def test_parse_json_object_or_event_wrapped_parses_qwen_style_event_array() -> None:
    raw = json.dumps(
        [
            {"type": "system", "subtype": "init"},
            {
                "type": "item.completed",
                "item": {"type": "assistant_message", "text": '{"ok": true, "errors": []}'},
            },
        ]
    )
    parsed = _parse_json_object_or_event_wrapped(raw)
    assert parsed == {"ok": True, "errors": []}


def test_parse_json_object_or_event_wrapped_parses_qwen_assistant_result_events() -> None:
    fenced = '```json\n{"ok": true, "errors": []}\n```'
    raw = json.dumps(
        [
            {
                "type": "assistant",
                "message": {
                    "type": "message",
                    "content": [{"type": "text", "text": fenced}],
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "result": fenced,
            },
        ]
    )
    parsed = _parse_json_object_or_event_wrapped(raw)
    assert parsed == {"ok": True, "errors": []}
