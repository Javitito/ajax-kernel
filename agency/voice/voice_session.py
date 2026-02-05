#!/usr/bin/env python3
"""
Sesión de voz simple para AJAX: usa VoiceIO (ASR/TTS) y el mismo core/chat.
Bloquea si el heartbeat no está en verde.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ajax
from agency.voice.vibevoice_client import VoiceIO

ROOT = Path(__file__).resolve().parents[2]
HEARTBEAT_PATH = ROOT / "artifacts" / "health" / "ajax_heartbeat.json"
LOG_PATH = ROOT / "logs" / "voice_session.log"


def load_heartbeat_status() -> Dict[str, Any]:
    if not HEARTBEAT_PATH.exists():
        return {"status": "unknown", "detail": "heartbeat_missing"}
    try:
        data = json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("heartbeat not dict")
        return data
    except Exception as exc:  # pragma: no cover - best effort
        return {"status": "unknown", "detail": f"heartbeat_parse_error:{exc}"}


def log_line(text: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except Exception:
        pass


def run_voice_session() -> int:
    core = ajax.wake_up()
    voice = VoiceIO()
    hb = load_heartbeat_status()
    hb_status = str(hb.get("status", "unknown")).lower()
    if hb_status != "green":
        msg = f"Heartbeat={hb_status}. Sesión de voz no iniciada."
        voice.speak(msg)
        log_line(msg)
        return 1

    voice.speak("Sesión de voz de AJAX iniciada. Di 'salir' para terminar.")
    while True:
        hb = load_heartbeat_status()
        hb_status = str(hb.get("status", "unknown")).lower()
        if hb_status != "green":
            voice.speak(f"Heartbeat {hb_status}. Pausando sesión.")
            log_line(f"heartbeat={hb_status} -> exit")
            break
        user_text = voice.listen_once()
        if voice.should_exit(user_text):
            voice.speak("Sesión de voz terminada.")
            break
        if not user_text:
            continue
        log_line(f">>> {user_text}")
        try:
            reply = core.chat(user_text)
        except Exception as exc:
            reply = f"Error procesando la petición: {exc}"
        log_line(f"<<< {reply}")
        voice.speak(reply)
        time.sleep(0.1)
    return 0


def main() -> None:
    sys.exit(run_voice_session())


if __name__ == "__main__":
    main()
