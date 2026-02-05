"""
EvidenceRef - Tipos de evidencia para Proof-Carrying Output.

Meta: Hacer imposible que un modelo cierre "confirmado" sin pruebas verificables.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass(frozen=True)
class EvidenceRef:
    """
    Referencia tipada e inmutable a evidencia verificable.

    Args:
        kind: Tipo de evidencia (log, receipt, verify_result, efe, etc.)
        path: Path real al artifact (debe existir)
        scope_id: ID de scope/misión si aplica
        meta: Metadata adicional (líneas, hash, timestamp, etc.)
    """

    kind: str
    path: str
    scope_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def validate_exists(self) -> bool:
        """Verificar que el path existe y es legible."""
        try:
            return Path(self.path).exists()
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializar a dict para JSON."""
        return {"kind": self.kind, "path": self.path, "scope_id": self.scope_id, "meta": self.meta}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRef":
        """Deserializar desde dict."""
        return cls(
            kind=data["kind"],
            path=data["path"],
            scope_id=data.get("scope_id"),
            meta=data.get("meta"),
        )
