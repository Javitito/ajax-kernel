from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def run_named_audit(
    *,
    name: str,
    root_dir: Path,
    run_id: Optional[str] = None,
    last: int = 1,
) -> Dict[str, Any]:
    audit_name = str(name or "").strip().lower()
    if audit_name == "tx":
        from agency.audits.tx_integrity_audit import run_tx_integrity_audit  # type: ignore

        return run_tx_integrity_audit(root_dir, run_id=run_id, last=last)
    raise ValueError(f"unknown_audit:{audit_name or 'empty'}")


__all__ = ["run_named_audit"]

