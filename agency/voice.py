from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional


class Voice:
    """
    Sintetiza texto a voz con edge-tts y reproduce el audio.
    Pensado para uso rápido en CLI/WSL; intenta reproductores CLI antes de fallback PowerShell.
    """

    def __init__(self, voice: str = "es-ES-AlvaroNeural"):
        self.voice = voice

    async def _synthesize(self, text: str, out_path: Path) -> None:
        try:
            import edge_tts  # type: ignore
        except ImportError:
            raise RuntimeError("edge-tts no instalado; instala con `pip install edge-tts`")

        tts = edge_tts.Communicate(text, voice=self.voice)
        await tts.save(str(out_path))

    def _play(self, path: Path) -> None:
        players = [
            ["mpv", "--no-terminal", "--really-quiet", str(path)],
            ["mpg123", "-q", str(path)],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        ]
        for cmd in players:
            if shutil.which(cmd[0]):
                os.spawnlp(os.P_NOWAIT, cmd[0], *cmd)
                return
        # Fallback a PowerShell en Windows (si disponible)
        if shutil.which("powershell.exe") or shutil.which("powershell"):
            ps = shutil.which("powershell.exe") or shutil.which("powershell")
            cmd = [
                ps,
                "-NoProfile",
                "-Command",
                f"Add-Type -AssemblyName presentationcore; $player = New-Object System.Media.SoundPlayer '{path}'; $player.Play();",
            ]
            os.spawnvp(os.P_NOWAIT, cmd[0], cmd)

    def speak(self, text: str) -> None:
        """
        Sintetiza y reproduce; no bloquea la llamada principal (lanza task).
        """
        if not text:
            return
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def _runner():
            with tempfile.TemporaryDirectory() as tmpd:
                out = Path(tmpd) / "voice.mp3"
                asyncio.run(self._synthesize(text, out))
                self._play(out)

        if loop and loop.is_running():
            loop.create_task(self._async_wrapper(text))
        else:
            _runner()

    async def _async_wrapper(self, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmpd:
            out = Path(tmpd) / "voice.mp3"
            await self._synthesize(text, out)
            self._play(out)
