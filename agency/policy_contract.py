from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


REQUIRED_POLICY_RELATIVE = (
    "config/provider_policy.yaml",
    "config/provider_failure_policy.yaml",
)
PROVIDER_POLICY_DERIVED_JSON = "config/provider_policy.json"


@dataclass(frozen=True)
class PolicyContractResult:
    ok: bool
    status: str
    reason: str
    required_files: List[str]
    missing_files: List[str]
    invalid_files: List[str]
    derived_files: List[str]
    receipt_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "reason": self.reason,
            "required_files": list(self.required_files),
            "missing_files": list(self.missing_files),
            "invalid_files": list(self.invalid_files),
            "derived_files": list(self.derived_files),
            "receipt_path": self.receipt_path,
        }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ts_label() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _load_yaml_dict(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if yaml is None:
        return None, "pyyaml_unavailable"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return None, f"yaml_parse_error:{str(exc)[:120]}"
    if not isinstance(payload, dict):
        return None, "yaml_not_object"
    return payload, None


def _write_policy_receipt(root_dir: Path, payload: Dict[str, Any]) -> Optional[str]:
    try:
        receipt_dir = Path(root_dir) / "artifacts" / "receipts"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"policy_contract_{_ts_label()}.json"
        receipt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(receipt_path)
    except Exception:
        return None


def validate_policy_contract(
    root_dir: Path,
    *,
    sync_json: bool = True,
    write_receipt: bool = True,
) -> PolicyContractResult:
    root = Path(root_dir)
    required_abs = [root / rel for rel in REQUIRED_POLICY_RELATIVE]
    missing_files: List[str] = []
    invalid_files: List[str] = []
    derived_files: List[str] = []
    loaded_docs: Dict[str, Dict[str, Any]] = {}

    for abs_path in required_abs:
        rel = str(abs_path.relative_to(root))
        if not abs_path.exists():
            missing_files.append(rel)
            continue
        payload, err = _load_yaml_dict(abs_path)
        if err:
            invalid_files.append(f"{rel}:{err}")
            continue
        loaded_docs[rel] = payload or {}

    if sync_json and "config/provider_policy.yaml" in loaded_docs:
        try:
            json_path = root / PROVIDER_POLICY_DERIVED_JSON
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(loaded_docs["config/provider_policy.yaml"], ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            derived_files.append(str(json_path))
        except Exception as exc:
            invalid_files.append(f"{PROVIDER_POLICY_DERIVED_JSON}:json_sync_failed:{str(exc)[:120]}")

    ok = not missing_files and not invalid_files
    status = "READY" if ok else "BLOCKED"
    reason = "policy_contract_ok"
    if not ok:
        if missing_files:
            reason = "missing_policy_files"
        else:
            reason = "invalid_policy_files"

    receipt_payload: Dict[str, Any] = {
        "schema": "ajax.policy_contract.v1",
        "ts_utc": _utc_now(),
        "ok": ok,
        "status": status,
        "reason": reason,
        "required_files": [str(path.relative_to(root)) for path in required_abs],
        "missing_files": missing_files,
        "invalid_files": invalid_files,
        "derived_files": derived_files,
    }
    receipt_path = _write_policy_receipt(root, receipt_payload) if write_receipt else None

    return PolicyContractResult(
        ok=ok,
        status=status,
        reason=reason,
        required_files=[str(path.relative_to(root)) for path in required_abs],
        missing_files=missing_files,
        invalid_files=invalid_files,
        derived_files=derived_files,
        receipt_path=receipt_path,
    )
