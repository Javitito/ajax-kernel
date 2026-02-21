from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


_REGISTRY_REL = Path("config") / "experiments_registry.yaml"
_CANCELLED_BUCKET_REL = Path("artifacts") / "capability_gaps" / "cancelled"
_RECEIPTS_REL = Path("artifacts") / "receipts"
_EXPERIMENT_ID_RE = re.compile(r"\bUI-\d{3}(?:-[A-Z0-9]+)*\b", re.IGNORECASE)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _registry_path(root_dir: Path) -> Path:
    return Path(root_dir) / _REGISTRY_REL


def _load_registry_blob(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema": "ajax.experiments_registry.v1", "experiments": {}}
    if yaml is not None:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                return raw
        except Exception:
            pass
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema": "ajax.experiments_registry.v1", "experiments": {}}
    return raw if isinstance(raw, dict) else {"schema": "ajax.experiments_registry.v1", "experiments": {}}


def _normalize_registry(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    experiments = raw.get("experiments")
    if isinstance(experiments, dict):
        for exp_id, payload in experiments.items():
            key = str(exp_id or "").strip().upper()
            if not key:
                continue
            row = dict(payload) if isinstance(payload, dict) else {}
            row.setdefault("id", key)
            out[key] = row
    return out


def get_experiment_record(root_dir: Path, experiment_id: str) -> Optional[Dict[str, Any]]:
    key = str(experiment_id or "").strip().upper()
    if not key:
        return None
    raw = _load_registry_blob(_registry_path(Path(root_dir)))
    row = _normalize_registry(raw).get(key)
    return dict(row) if isinstance(row, dict) else None


def is_experiment_cancelled(root_dir: Path, experiment_id: str) -> bool:
    row = get_experiment_record(Path(root_dir), experiment_id)
    if not row:
        return False
    return str(row.get("status") or "").strip().upper() == "CANCELLED"


def _iter_candidate_texts(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key:
                yield from _iter_candidate_texts(key)
            yield from _iter_candidate_texts(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_candidate_texts(item)
        return
    yield str(value)


def resolve_experiment_id(candidates: Iterable[Any]) -> Optional[str]:
    for value in candidates:
        for text in _iter_candidate_texts(value):
            match = _EXPERIMENT_ID_RE.search(text)
            if match:
                return match.group(0).upper()
    return None


def guard_gap_write_for_cancelled_experiment(
    root_dir: Path,
    *,
    source: str,
    proposed_path: Path,
    gap_payload: Dict[str, Any],
    candidates: Iterable[Any],
) -> Tuple[Path, Optional[Path], Optional[str], bool]:
    root = Path(root_dir)
    experiment_id = resolve_experiment_id(candidates)
    if not experiment_id:
        return proposed_path, None, None, False

    row = get_experiment_record(root, experiment_id)
    if not row or str(row.get("status") or "").strip().upper() != "CANCELLED":
        return proposed_path, None, experiment_id, False

    target_dir = root / _CANCELLED_BUCKET_REL / experiment_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(proposed_path).name

    payload = dict(gap_payload)
    payload["cancelled_experiment"] = {
        "id": experiment_id,
        "status": "CANCELLED",
        "reason": row.get("reason"),
        "label": "CANCELLED",
    }
    payload["guard_action"] = "redirect_cancelled"
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    receipt_dir = root / _RECEIPTS_REL
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_compact()
    nonce = int(time.time() * 1000) % 1000
    receipt_path = receipt_dir / f"gap_guard_{ts}_{experiment_id}_{nonce:03d}.json"
    receipt = {
        "schema": "ajax.gap_cancelled_guard_receipt.v1",
        "ts_utc": _utc_now(),
        "source": source,
        "skipped_reason": "experiment_cancelled",
        "experiment_id": experiment_id,
        "proposed_path": str(proposed_path),
        "redirected_path": str(target_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target_path, receipt_path, experiment_id, True
