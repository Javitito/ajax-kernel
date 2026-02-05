
import json
import time
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

class QuotaGate:
    """
    Gestiona el acceso a proveedores basado en cuotas persistidas.
    Implementa cache con TTL y lógica de tri-estado.
    """
    def __init__(self, root_dir: Path, ttl_seconds: int = 3600):
        self.root_dir = root_dir
        self.quota_path = root_dir / "artifacts" / "providers_quota.json"
        self.ttl_seconds = ttl_seconds

    def load_quota_state(self) -> Dict[str, Any]:
        if not self.quota_path.exists():
            return {"checked_at": 0, "providers": {}}
        try:
            return json.loads(self.quota_path.read_text(encoding="utf-8"))
        except Exception:
            return {"checked_at": 0, "providers": {}}

    def is_stale(self, state: Dict[str, Any]) -> bool:
        checked_at = state.get("checked_at", 0)
        return (time.time() - checked_at) > self.ttl_seconds

    def refresh_quota(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Intenta actualizar las cuotas ejecutando el script externo con timeout."""
        script_path = self.root_dir / "scripts" / "quota_providers.py"
        try:
            # Ejecutamos sin bloquear demasiado el flujo principal
            subprocess.run(
                ["python3", str(script_path)],
                timeout=timeout,
                capture_output=True,
                check=False
            )
        except subprocess.TimeoutExpired:
            pass # Si tarda demasiado, seguimos con lo que hay
        except Exception:
            pass
        return self.load_quota_state()

    def get_provider_status(self, provider_name: str, state: Dict[str, Any]) -> Tuple[str, str]:
        """
        Retorna (status, reason)
        status: "true" | "false" | "unknown"
        """
        pdata = state.get("providers", {}).get(provider_name)
        if not pdata:
            return "unknown", "no_data"
        
        # available en el JSON original es bool, lo convertimos a tri-estado Ajax
        is_avail = pdata.get("ok_to_use")
        if is_avail is False:
            return "false", pdata.get("reason", "quota_exhausted")
        
        if is_avail is True:
            # Verificación extra de margen
            margin = pdata.get("margin", {})
            if margin.get("rpm", 0) <= 0 and margin.get("rpd", 0) <= 0:
                return "false", "hard_limit_reached"
            return "true", "ok"
            
        return "unknown", "insufficient_signal"

    def filter_roster(self, providers: List[str]) -> List[str]:
        """Filtra la lista de proveedores eliminando los 'false'."""
        state = self.load_quota_state()
        eligible = []
        for p in providers:
            status, _ = self.get_provider_status(p, state)
            if status != "false":
                eligible.append(p)
        return eligible
