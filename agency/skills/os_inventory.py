from __future__ import annotations

import subprocess
from typing import List


def get_installed_apps() -> List[str]:
    """
    Devuelve una lista de apps detectadas en el sistema Windows usando PowerShell Get-StartApps.
    Filtra duplicados y normaliza a nombres simples.
    """
    ps = ["powershell", "-NoProfile", "-Command", "Get-StartApps | Select-Object -ExpandProperty Name"]
    try:
        proc = subprocess.run(ps, capture_output=True, text=True, timeout=10, check=False)
        names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        names = []
    # fallback a algunos candidatos frecuentes
    candidates = [
        "brave.exe",
        "chrome.exe",
        "msedge.exe",
        "spotify.exe",
        "vlc.exe",
        "notepad.exe",
        "notepad++.exe",
        "youtube music",
        "cmd.exe",
        "powershell.exe",
    ]
    all_names = names + candidates
    seen = set()
    uniq = []
    for n in all_names:
        key = n.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(n)
    return uniq
