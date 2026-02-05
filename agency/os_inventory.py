from __future__ import annotations

import shutil
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import yaml
except ImportError:
    yaml = None

log = logging.getLogger(__name__)

def discover_browsers(root_dir: Optional[Path] = None) -> List[str]:
    """
    Detecta navegadores instalados en el sistema.
    Usa shutil.which y un fallback declarativo en config/os_fallback.yaml.
    """
    candidates = [
        "brave.exe",
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
        "brave",
        "chrome",
        "msedge",
        "firefox",
    ]
    
    found = []
    for c in candidates:
        if shutil.which(c):
            found.append(c)
            
    # Fallback declarativo
    if root_dir:
        fallback_path = root_dir / "config" / "os_fallback.yaml"
        if fallback_path.exists() and yaml:
            try:
                with open(fallback_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    extra = (data or {}).get("browsers", [])
                    for b in extra:
                        if b not in found:
                            # Si es un path absoluto, verificar existencia
                            if os.path.isabs(b):
                                if os.path.exists(b):
                                    found.append(b)
                            else:
                                if shutil.which(b):
                                    found.append(b)
            except Exception as exc:
                log.warning("No se pudo leer os_fallback.yaml: %s", exc)
                
    return found

def get_default_browser(root_dir: Optional[Path] = None) -> str:
    """
    Devuelve el navegador por defecto (el primero encontrado o fallback).
    """
    found = discover_browsers(root_dir)
    if found:
        return found[0]
    return "brave.exe" # Hard-fallback final si todo falla
