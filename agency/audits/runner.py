from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


AuditRunnerFn = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class AuditRegistration:
    name: str
    help: str
    runner: AuditRunnerFn


def _run_tx(*, root_dir: Path, run_id: Optional[str] = None, last: int = 1) -> Dict[str, Any]:
    from agency.audits.tx_integrity_audit import run_tx_integrity_audit  # type: ignore

    return run_tx_integrity_audit(root_dir, run_id=run_id, last=last)


_AUDIT_REGISTRY: Dict[str, AuditRegistration] = {
    "tx": AuditRegistration(
        name="tx",
        help="Transactional Integrity + Proof-Carrying Claims Audit (read-only sobre runs).",
        runner=_run_tx,
    ),
}


def list_registered_audits() -> Mapping[str, AuditRegistration]:
    return dict(_AUDIT_REGISTRY)


def run_named_audit(
    *,
    name: str,
    root_dir: Path,
    run_id: Optional[str] = None,
    last: int = 1,
) -> Dict[str, Any]:
    audit_name = str(name or "").strip().lower()
    reg = _AUDIT_REGISTRY.get(audit_name)
    if reg is None:
        raise ValueError(f"unknown_audit:{audit_name or 'empty'}")
    return reg.runner(root_dir=root_dir, run_id=run_id, last=last)


__all__ = ["AuditRegistration", "list_registered_audits", "run_named_audit"]
