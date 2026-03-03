from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CANDIDATE_SCHEMA = "ajax.verify.efe_candidate.v0"
RECEIPT_SCHEMA = "ajax.receipt.efe_autogen.v0"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        if val in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        out = value.strip()
        return out or None
    return None


def _normalize_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        num = int(value)
        return num
    except Exception:
        return None


def _dedupe_sorted(items: List[str]) -> List[str]:
    return sorted({item for item in items if isinstance(item, str) and item.strip()})


def _collect_paths(params: Dict[str, Any]) -> List[str]:
    raw_paths: List[str] = []
    for key in ("paths", "files"):
        val = params.get(key)
        if isinstance(val, list):
            for item in val:
                text = _normalize_text(item)
                if text:
                    raw_paths.append(text)

    for key in (
        "path",
        "file",
        "filepath",
        "target",
        "target_path",
        "output",
        "output_path",
        "destination",
        "dest",
    ):
        text = _normalize_text(params.get(key))
        if text:
            raw_paths.append(text)

    return _dedupe_sorted(raw_paths)


def _descriptor_from_step(action: Any, args: Any) -> Optional[Dict[str, Any]]:
    action_name = str(action or "").strip()
    if not action_name:
        return None
    args_map = args if isinstance(args, dict) else {}

    if isinstance(args_map.get("kind"), str):
        return {
            "kind": str(args_map.get("kind")).strip().lower(),
            "params": dict(args_map.get("params") or {}),
            "source": "step.args",
        }

    action_l = action_name.lower()

    has_port = any(k in args_map for k in ("port", "local_port", "bind_port"))
    has_process = any(k in args_map for k in ("process", "process_name", "name", "pid", "process_pid"))
    has_paths = bool(_collect_paths(args_map))

    if has_port or any(tok in action_l for tok in ("port", "listen", "server")):
        port = args_map.get("port", args_map.get("local_port", args_map.get("bind_port")))
        host = args_map.get("host", args_map.get("bind_host", "127.0.0.1"))
        is_open = not any(tok in action_l for tok in ("stop", "close", "kill", "shutdown"))
        return {
            "kind": "port",
            "params": {
                "host": host,
                "port": port,
                "open": is_open,
            },
            "source": "step.infer",
            "action": action_name,
        }

    if has_process or any(tok in action_l for tok in ("process", "launch", "spawn", "kill", "terminate", "start", "stop")):
        running = not any(tok in action_l for tok in ("kill", "terminate", "stop", "shutdown"))
        return {
            "kind": "process",
            "params": {
                "name": args_map.get("process") or args_map.get("process_name") or args_map.get("name"),
                "pid": args_map.get("pid") or args_map.get("process_pid"),
                "running": running,
            },
            "source": "step.infer",
            "action": action_name,
        }

    if has_paths or any(tok in action_l for tok in ("file", "write", "save", "create", "delete", "copy", "move", "mkdir", "touch")):
        exists = not any(tok in action_l for tok in ("delete", "remove", "unlink"))
        return {
            "kind": "fs",
            "params": {
                "paths": _collect_paths(args_map),
                "exists": exists,
            },
            "source": "step.infer",
            "action": action_name,
        }

    return None


def extract_action_descriptor(source_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(source_doc, dict):
        return None

    if isinstance(source_doc.get("kind"), str):
        return {
            "kind": str(source_doc.get("kind") or "").strip().lower(),
            "params": dict(source_doc.get("params") or {}),
            "source": "root",
        }

    ad = source_doc.get("action_descriptor")
    if isinstance(ad, dict) and isinstance(ad.get("kind"), str):
        return {
            "kind": str(ad.get("kind") or "").strip().lower(),
            "params": dict(ad.get("params") or {}),
            "source": "action_descriptor",
        }

    ads = source_doc.get("action_descriptors")
    if isinstance(ads, list):
        for item in ads:
            if isinstance(item, dict) and isinstance(item.get("kind"), str):
                return {
                    "kind": str(item.get("kind") or "").strip().lower(),
                    "params": dict(item.get("params") or {}),
                    "source": "action_descriptors[0]",
                }

    if isinstance(source_doc.get("action"), str):
        inferred = _descriptor_from_step(source_doc.get("action"), source_doc.get("args"))
        if inferred:
            return inferred

    steps = source_doc.get("steps")
    if isinstance(steps, list):
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            inferred = _descriptor_from_step(step.get("action"), step.get("args"))
            if inferred:
                inferred["source"] = f"steps[{idx}]"
                step_id = _normalize_text(step.get("id"))
                if step_id:
                    inferred["step_id"] = step_id
                return inferred

    return None


def generate_expected_state(
    descriptor: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[str], Optional[str]]:
    kind = str(descriptor.get("kind") or "").strip().lower()
    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}

    if kind == "fs":
        paths = _collect_paths(params)
        exists = _normalize_bool(params.get("exists"), default=True)
        if not paths:
            return None, [], "unsupported_action_kind", "FS descriptor requires at least one path"

        files = [{"path": p, "must_exist": exists} for p in paths]
        checks = [
            {
                "kind": "fs",
                "path": p,
                "exists": exists,
                "mtime": {"required": bool(exists)},
                "size": {"required": bool(exists)},
                "sha256": {"required": bool(exists)},
            }
            for p in paths
        ]
        explain = [
            f"FS deterministic checks generated for {len(paths)} path(s)",
            "Each FS check captures exists/mtime/size/sha256 observables",
        ]
        return {"files": files, "checks": checks}, explain, None, None

    if kind == "process":
        running = _normalize_bool(params.get("running"), default=True)
        name = _normalize_text(params.get("name") or params.get("process"))
        pid = _normalize_int(params.get("pid"))
        if not name and pid is None:
            return (
                None,
                [],
                "unsupported_action_kind",
                "Process descriptor requires name or pid",
            )

        check: Dict[str, Any] = {
            "kind": "process",
            "running": running,
        }
        if name:
            check["name"] = name
        if pid is not None:
            check["pid"] = pid
        explain = [
            "Process deterministic check generated",
            "Process state check tracks running/not-running with optional identity",
        ]
        return {"checks": [check]}, explain, None, None

    if kind == "port":
        host = _normalize_text(params.get("host")) or "127.0.0.1"
        if host == "localhost":
            host = "127.0.0.1"
        port = _normalize_int(params.get("port"))
        is_open = _normalize_bool(params.get("open"), default=True)
        if port is None or port <= 0 or port > 65535:
            return None, [], "unsupported_action_kind", "Port descriptor requires valid TCP port"
        if host not in {"127.0.0.1", "::1"}:
            return None, [], "unsupported_action_kind", "Only localhost ports are supported"

        check = {
            "kind": "port",
            "host": host,
            "port": port,
            "open": is_open,
        }
        explain = [
            "Port deterministic check generated",
            "Port state check tracks open/closed on localhost",
        ]
        return {"checks": [check]}, explain, None, None

    return None, [], "unsupported_action_kind", f"Unsupported descriptor kind: {kind or 'unknown'}"


def autogen_efe_candidate(
    *,
    source_doc: Dict[str, Any],
    out_path: Path,
    source_path: Optional[Path] = None,
    receipts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    descriptor = extract_action_descriptor(source_doc)

    unsupported_kind: Optional[str] = None
    unsupported_hint: Optional[str] = None
    explain: List[str] = []
    expected_state: Dict[str, Any] = {}

    if descriptor is None:
        unsupported_kind = "unsupported_action_kind"
        unsupported_hint = "No action descriptor found (kind+params | action_descriptor | steps[])."
        explain = [
            "No supported descriptor found in source JSON",
            "Provide kind+params or a plan with actionable steps",
        ]
    else:
        generated, explain, unsupported_kind, unsupported_hint = generate_expected_state(descriptor)
        if isinstance(generated, dict):
            expected_state = generated

    payload: Dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "version": "v0",
        "created_at": _utc_now(),
        "source_path": str(source_path) if isinstance(source_path, Path) else None,
        "descriptor": descriptor,
        "expected_state": expected_state,
        "explain": explain,
        "ok": bool(expected_state),
    }
    if unsupported_kind:
        payload["unsupported_action_kind"] = unsupported_kind
    if unsupported_hint:
        payload["hint"] = unsupported_hint

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    receipt_path: Optional[Path] = None
    if receipts_dir is not None:
        receipts_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipts_dir / f"efe_autogen_{_utc_stamp()}.json"
        receipt_payload = {
            "schema": RECEIPT_SCHEMA,
            "version": "v0",
            "created_at": _utc_now(),
            "ok": bool(payload.get("ok")),
            "source_path": str(source_path) if isinstance(source_path, Path) else None,
            "efe_candidate_path": str(out_path),
            "descriptor_kind": (descriptor or {}).get("kind") if isinstance(descriptor, dict) else None,
            "unsupported_action_kind": unsupported_kind,
            "hint": unsupported_hint,
        }
        receipt_path.write_text(
            json.dumps(receipt_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "ok": bool(payload.get("ok")),
        "efe_candidate_path": str(out_path),
        "receipt_path": str(receipt_path) if receipt_path else None,
        "unsupported_action_kind": unsupported_kind,
        "hint": unsupported_hint,
        "descriptor": descriptor,
        "expected_state": expected_state,
        "explain": explain,
    }


def autogen_efe_candidate_from_file(
    *,
    source_path: Path,
    out_path: Path,
    receipts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--from must point to a JSON object")
    return autogen_efe_candidate(
        source_doc=raw,
        out_path=out_path,
        source_path=source_path,
        receipts_dir=receipts_dir,
    )
