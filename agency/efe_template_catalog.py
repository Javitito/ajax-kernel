from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agency.verify.efe_autogen import extract_action_descriptor


MISSION_FAMILY_ARTIFACT = "artifact_generation"
MISSION_FAMILY_ANALYSIS = "analysis_only"
MISSION_FAMILY_OBSERVATION = "observation_only"
MISSION_FAMILY_REPO_PATCH = "repo_patch"
MISSION_FAMILY_DESKTOP = "desktop_action"


@dataclass(frozen=True)
class EFETemplateSpec:
    template_id: str
    mission_families: Tuple[str, ...]
    verify_shape: Dict[str, Any]
    required_fields: Dict[str, Dict[str, Any]]
    rollback_hint: Optional[str] = None


@dataclass(frozen=True)
class TemplateResolution:
    template_id: str
    mission_family: str
    fields: Dict[str, Any]
    explain: List[str]


def _normalize_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _normalize_paths(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
    elif isinstance(value, list):
        for item in value:
            text = _normalize_text(item)
            if text:
                out.append(text)
    deduped: List[str] = []
    seen = set()
    for item in out:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _collect_candidate_paths(source_doc: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
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
        "receipt_path",
        "artifact_path",
        "candidate_path",
    ):
        text = _normalize_text(source_doc.get(key))
        if text:
            paths.append(text)

    for key in ("paths", "files", "artifacts"):
        paths.extend(_normalize_paths(source_doc.get(key)))

    args = source_doc.get("args")
    if isinstance(args, dict):
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
            "receipt_path",
            "artifact_path",
        ):
            text = _normalize_text(args.get(key))
            if text:
                paths.append(text)
        for key in ("paths", "files", "artifacts"):
            paths.extend(_normalize_paths(args.get(key)))

    meta = source_doc.get("metadata")
    if isinstance(meta, dict):
        for key in ("receipt_path", "subcall_output_path", "artifact_path", "output_path"):
            text = _normalize_text(meta.get(key))
            if text:
                paths.append(text)
        paths.extend(_normalize_paths(meta.get("artifacts")))

    deduped: List[str] = []
    seen = set()
    for item in paths:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _receipt_schema_from_doc(source_doc: Dict[str, Any], receipt_path: Optional[str]) -> Optional[str]:
    explicit = _normalize_text(source_doc.get("receipt_schema"))
    if explicit:
        return explicit
    args = source_doc.get("args")
    if isinstance(args, dict):
        explicit = _normalize_text(args.get("receipt_schema"))
        if explicit:
            return explicit
    meta = source_doc.get("metadata")
    if isinstance(meta, dict):
        explicit = _normalize_text(meta.get("receipt_schema"))
        if explicit:
            return explicit
    if receipt_path:
        try:
            payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            inferred = _normalize_text(payload.get("schema"))
            if inferred:
                return inferred
    return None


def classify_mission_family(source_doc: Dict[str, Any]) -> str:
    descriptor = extract_action_descriptor(source_doc)
    kind = str((descriptor or {}).get("kind") or "").strip().lower()
    action = _normalize_text(source_doc.get("action")) or ""
    action_l = action.lower()

    if action_l.startswith(("app.", "window.", "keyboard.", "desktop.")):
        return MISSION_FAMILY_DESKTOP

    paths = _collect_candidate_paths(source_doc)
    if any("\\artifacts\\receipts\\" in p.lower() or "/artifacts/receipts/" in p.lower() for p in paths):
        return MISSION_FAMILY_ANALYSIS
    if any("\\artifacts\\subcalls\\" in p.lower() or "/artifacts/subcalls/" in p.lower() for p in paths):
        return MISSION_FAMILY_OBSERVATION

    if kind == "fs":
        if not paths and isinstance(descriptor, dict):
            params = descriptor.get("params")
            if isinstance(params, dict):
                paths = _normalize_paths(params.get("paths"))
        if paths and all(
            "\\artifacts\\" in p.lower() or "/artifacts/" in p.lower() or p.lower().startswith("artifacts/")
            for p in paths
        ):
            return MISSION_FAMILY_ARTIFACT
        return MISSION_FAMILY_REPO_PATCH

    if kind in {"port", "process"}:
        return MISSION_FAMILY_OBSERVATION

    return MISSION_FAMILY_ANALYSIS


TEMPLATE_CATALOG: Dict[str, EFETemplateSpec] = {
    "efe.fs_path_materialized.v0": EFETemplateSpec(
        template_id="efe.fs_path_materialized.v0",
        mission_families=(MISSION_FAMILY_ARTIFACT, MISSION_FAMILY_REPO_PATCH),
        verify_shape={
            "files": [{"path": "<path>", "must_exist": True}],
            "checks": [{"kind": "fs", "path": "<path>", "exists": True}],
        },
        required_fields={
            "paths": {"type": "array[string]", "min_items": 1},
            "must_exist": {"type": "boolean"},
        },
        rollback_hint="Delete generated artifact paths if execution later fails and the path is disposable.",
    ),
    "efe.receipt_schema_valid.v0": EFETemplateSpec(
        template_id="efe.receipt_schema_valid.v0",
        mission_families=(MISSION_FAMILY_ANALYSIS, MISSION_FAMILY_OBSERVATION),
        verify_shape={
            "files": [{"path": "<receipt_path>", "must_exist": True}],
            "checks": [{"kind": "receipt_schema", "path": "<receipt_path>", "schema": "<schema_id>"}],
        },
        required_fields={
            "receipt_path": {"type": "string"},
            "receipt_schema": {"type": "string"},
        },
        rollback_hint=None,
    ),
    "efe.subcall_structured_output.v0": EFETemplateSpec(
        template_id="efe.subcall_structured_output.v0",
        mission_families=(MISSION_FAMILY_OBSERVATION, MISSION_FAMILY_ANALYSIS),
        verify_shape={
            "files": [{"path": "<output_path>", "must_exist": True}],
            "checks": [
                {
                    "kind": "structured_output",
                    "path": "<output_path>",
                    "format": "json",
                    "root_type": "object",
                    "required_keys": ["schema", "role", "provider_selected", "result", "reason_code"],
                }
            ],
        },
        required_fields={
            "output_path": {"type": "string"},
            "required_keys": {"type": "array[string]", "min_items": 1},
        },
        rollback_hint=None,
    ),
}


def get_template_catalog() -> Dict[str, EFETemplateSpec]:
    return dict(TEMPLATE_CATALOG)


def validate_template_fields(template_id: str, fields: Dict[str, Any]) -> List[str]:
    spec = TEMPLATE_CATALOG.get(template_id)
    if spec is None:
        return ["unknown_template"]
    problems: List[str] = []

    if template_id == "efe.fs_path_materialized.v0":
        paths = _normalize_paths(fields.get("paths"))
        if not paths:
            problems.append("paths_required")
    elif template_id == "efe.receipt_schema_valid.v0":
        if not _normalize_text(fields.get("receipt_path")):
            problems.append("receipt_path_required")
        if not _normalize_text(fields.get("receipt_schema")):
            problems.append("receipt_schema_required")
    elif template_id == "efe.subcall_structured_output.v0":
        if not _normalize_text(fields.get("output_path")):
            problems.append("output_path_required")
        keys = fields.get("required_keys")
        if not isinstance(keys, list) or not any(_normalize_text(item) for item in keys):
            problems.append("required_keys_required")
    return problems


def materialize_template_expected_state(template_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    problems = validate_template_fields(template_id, fields)
    if problems:
        raise ValueError(",".join(problems))

    if template_id == "efe.fs_path_materialized.v0":
        must_exist = bool(fields.get("must_exist", True))
        paths = _normalize_paths(fields.get("paths"))
        return {
            "files": [{"path": path, "must_exist": must_exist} for path in paths],
            "checks": [
                {
                    "kind": "fs",
                    "path": path,
                    "exists": must_exist,
                    "mtime": {"required": bool(must_exist)},
                    "size": {"required": bool(must_exist)},
                    "sha256": {"required": bool(must_exist)},
                }
                for path in paths
            ],
        }

    if template_id == "efe.receipt_schema_valid.v0":
        receipt_path = str(fields["receipt_path"])
        receipt_schema = str(fields["receipt_schema"])
        return {
            "files": [{"path": receipt_path, "must_exist": True}],
            "checks": [
                {
                    "kind": "receipt_schema",
                    "path": receipt_path,
                    "schema": receipt_schema,
                }
            ],
        }

    if template_id == "efe.subcall_structured_output.v0":
        output_path = str(fields["output_path"])
        required_keys = [str(item) for item in fields.get("required_keys") or [] if _normalize_text(item)]
        return {
            "files": [{"path": output_path, "must_exist": True}],
            "checks": [
                {
                    "kind": "structured_output",
                    "path": output_path,
                    "format": "json",
                    "root_type": "object",
                    "required_keys": required_keys,
                }
            ],
        }

    raise ValueError(f"unknown_template:{template_id}")


def resolve_template(source_doc: Dict[str, Any]) -> Optional[TemplateResolution]:
    if not isinstance(source_doc, dict):
        return None

    explicit_template = _normalize_text(source_doc.get("efe_template_id"))
    if explicit_template and explicit_template in TEMPLATE_CATALOG:
        mission_family = classify_mission_family(source_doc)
        fields = dict(source_doc.get("efe_template_fields") or {})
        explain = ["template_hint_explicit"]
        if validate_template_fields(explicit_template, fields):
            return None
        return TemplateResolution(
            template_id=explicit_template,
            mission_family=mission_family,
            fields=fields,
            explain=explain,
        )

    descriptor = extract_action_descriptor(source_doc)
    mission_family = classify_mission_family(source_doc)
    if isinstance(descriptor, dict) and str(descriptor.get("kind") or "").strip().lower() == "fs":
        params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
        paths = _normalize_paths(params.get("paths"))
        if not paths:
            paths = _collect_candidate_paths(source_doc)
        if paths:
            must_exist = bool(params.get("exists", True))
            return TemplateResolution(
                template_id="efe.fs_path_materialized.v0",
                mission_family=mission_family,
                fields={"paths": paths, "must_exist": must_exist},
                explain=["descriptor_kind=fs", f"mission_family={mission_family}"],
            )

    paths = _collect_candidate_paths(source_doc)
    receipt_path = next(
        (
            path
            for path in paths
            if "\\artifacts\\receipts\\" in path.lower()
            or "/artifacts/receipts/" in path.lower()
            or path.lower().startswith("artifacts/receipts/")
        ),
        None,
    )
    receipt_schema = _receipt_schema_from_doc(source_doc, receipt_path)
    if receipt_path and receipt_schema:
        return TemplateResolution(
            template_id="efe.receipt_schema_valid.v0",
            mission_family=mission_family,
            fields={"receipt_path": receipt_path, "receipt_schema": receipt_schema},
            explain=["receipt_path_detected", f"receipt_schema={receipt_schema}"],
        )

    output_path = next(
        (
            path
            for path in paths
            if "\\artifacts\\subcalls\\" in path.lower()
            or "/artifacts/subcalls/" in path.lower()
            or path.lower().startswith("artifacts/subcalls/")
        ),
        None,
    )
    if output_path:
        required_keys = source_doc.get("required_keys")
        if not isinstance(required_keys, list) or not required_keys:
            required_keys = ["schema", "role", "provider_selected", "result", "reason_code"]
        return TemplateResolution(
            template_id="efe.subcall_structured_output.v0",
            mission_family=mission_family,
            fields={"output_path": output_path, "required_keys": required_keys},
            explain=["subcall_artifact_path_detected"],
        )

    return None
