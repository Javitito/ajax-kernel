from __future__ import annotations

from agency.vision_llm import _score_text_match, _ocr_match_marks


def test_score_text_match():
    assert _score_text_match("Enviar", "Enviar") == 1.0
    assert _score_text_match("Enviar", "Enviar ahora") >= 0.75
    assert _score_text_match("Enviar ahora", "Enviar") >= 0.75
    assert _score_text_match("ABC", "xyz") == 0.0


def test_ocr_match_marks():
    marks = [
        {"id": "A1", "text": "Enviar", "ocr_confidence": 0.9},
        {"id": "B1", "text": "Cancelar", "ocr_confidence": 0.9},
    ]
    m, score = _ocr_match_marks("Enviar", marks, min_score=0.8)
    assert m is not None and m.get("id") == "A1"
    assert score >= 0.8

