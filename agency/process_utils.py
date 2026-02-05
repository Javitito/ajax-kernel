from __future__ import annotations

import os
import subprocess
from typing import Optional


def is_wsl() -> bool:
    """
    Best-effort detection for WSL.

    We keep this env-based (no /proc reads) so it is predictable in tests.
    """
    return bool(
        os.environ.get("WSL_INTEROP")
        or os.environ.get("WSL_DISTRO_NAME")
        or os.environ.get("WSLENV")
    )


def _pid_running_windows(pid: int) -> bool:
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None  # type: ignore
    if psutil is not None:
        try:
            return bool(psutil.pid_exists(pid))
        except Exception:
            pass
    try:
        proc = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "").strip()
        if not out:
            return False
        if "No tasks are running" in out:
            return False
        return f'"{pid}"' in out
    except Exception:
        return False


def _pid_running_wsl(pid: int) -> bool:
    """
    In WSL, Windows-side processes are not visible to os.kill(pid, 0).
    Check via Windows tooling instead.
    """
    # Prefer tasklist.exe: cheap and doesn't require PowerShell parsing.
    try:
        proc = subprocess.run(
            ["tasklist.exe", "/FO", "CSV", "/NH", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (proc.stdout or "").strip()
        if out:
            if "No tasks are running" in out:
                return False
            return f'"{pid}"' in out
    except FileNotFoundError:
        pass
    except Exception:
        # Fall through to PowerShell.
        pass

    # Fallback: PowerShell Get-Process.
    try:
        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"try {{ Get-Process -Id {pid} -ErrorAction Stop | Out-Null; 'OK' }} catch {{ 'NO' }}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return (proc.stdout or "").strip().upper().endswith("OK")
    except Exception:
        return False


def pid_running(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if is_wsl():
        return _pid_running_wsl(pid)
    if os.name == "nt":
        return _pid_running_windows(pid)
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False
