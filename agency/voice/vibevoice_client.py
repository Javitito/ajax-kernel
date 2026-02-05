from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class VoiceIO:
    """
    Envoltura sencilla para ASR/TTS. Por defecto usa fallback textual
    para no depender de librerías de audio. Sustituye los métodos
    _asr_listen y _tts_speak cuando tengas VibeVoice real.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).resolve().parents[2] / "config" / "ajax_voice.json"
        self.cfg = self._load_config(self.config_path)
        self.exit_words = [w.lower() for w in self.cfg.get("exit_words", [])]
        self.input_prompt = self.cfg.get("input_prompt") or "🎤 "

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _asr_listen(self) -> Optional[str]:
        """
        Stub de ASR: usa input() como fallback.
        Sustituir por llamada real a VibeVoice cuando esté disponible.
        """
        try:
            text = input(self.input_prompt)
            return text.strip()
        except EOFError:
            return None
        except Exception:
            return None

    def _tts_speak(self, text: str) -> None:
        """
        Stub de TTS: imprime en consola.
        Sustituir por llamada real a VibeVoice cuando esté disponible.
        """
        print(f"🔊 {text}")

    def listen_once(self) -> Optional[str]:
        text = self._asr_listen()
        if text:
            return text.strip()
        return None

    def speak(self, text: str) -> None:
        if not text:
            return
        self._tts_speak(text)

    def should_exit(self, text: Optional[str]) -> bool:
        if not text:
            return False
        low = text.strip().lower()
        return low in self.exit_words if self.exit_words else False

    def status(self) -> Dict[str, Any]:
        return {
            "asr_mode": self.cfg.get("asr_mode", "text_fallback"),
            "tts_mode": self.cfg.get("tts_mode", "text_fallback"),
            "language": self.cfg.get("language", "unknown"),
            "exit_words": self.exit_words,
        }
