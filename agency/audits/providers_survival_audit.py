"""
OPS Providers Survival Audit (read-only).

Audita coherencia de supervivencia operativa entre:
- config/provider_policy.{yaml,json}
- artifacts/health/providers_status.json
- artifacts/provider_ledger/latest.json
- config/provider_timeouts_policy.json
- config/provider_failure_policy.yaml (opcional; warn si falta)

Objetivo: detectar riesgos de bloqueo/quorum/ladder/cooldowns/timeouts sin ejecutar probes vivos.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


AUDIT_SCHEMA = "ajax.audit.providers_survival.v0"
RECEIPT_SCHEMA = "ajax.receipt.providers_survival_audit.v1"

VALID_LEDGER_STATUS = {"ok", "soft_fail", "hard_fail"}
CORE_ROLES = {"brain", "council", "scout"}
TIMEOUT_DEFAULT_KEYS = {
    "connect_timeout_ms",
    "first_output_timeout_ms",
    "stall_timeout_ms",
    "total_timeout_ms",
}


@dataclass
class Finding:
    id: str
    severity: str  # info|warn|error|critical
    title: str
    code: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    invariant: str = ""
    recommendation: str = ""
    next_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InputSnapshot:
    key: str
    required: bool
    path: Optional[Path] = None
    exists: bool = False
    parsed: bool = False
    parse_error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    raw_sha256: Optional[str] = None
    raw_bytes: int = 0
    source_kind: str = "json"  # json|yaml|none

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "required": self.required,
            "path": str(self.path) if self.path else None,
            "exists": self.exists,
            "parsed": self.parsed,
            "parse_error": self.parse_error,
            "raw_sha256": self.raw_sha256,
            "raw_bytes": self.raw_bytes,
            "source_kind": self.source_kind,
        }


@dataclass
class AuditResult:
    ok: bool
    schema: str = AUDIT_SCHEMA
    timestamp: float = 0.0
    timestamp_iso: str = ""
    mode: str = "fast"
    read_only: bool = True
    live_probes_invoked: bool = False
    findings: List[Finding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[Dict[str, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "schema": self.schema,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "mode": self.mode,
            "read_only": self.read_only,
            "live_probes_invoked": self.live_probes_invoked,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "inputs": self.inputs,
            "recommended_actions": self.recommended_actions,
            "meta": self.meta,
        }


def _now_ts() -> float:
    return time.time()


def _iso_utc(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_now_ts() if ts is None else ts))


def _ts_label(ts: float) -> str:
    return time.strftime("%Y%m%d-%H%M%SZ", time.gmtime(ts))


def _sha256_text(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json_doc(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], int]:
    if not path.exists():
        return None, "file_not_found", None, 0
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, f"read_error:{exc}", None, 0
    try:
        data = json.loads(raw)
    except Exception as exc:
        return None, f"json_parse_error:{exc}", _sha256_text(raw), len(raw.encode('utf-8'))
    if not isinstance(data, dict):
        return None, "json_not_object", _sha256_text(raw), len(raw.encode("utf-8"))
    return data, None, _sha256_text(raw), len(raw.encode("utf-8"))


def _read_yaml_doc(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], int]:
    if not path.exists():
        return None, "file_not_found", None, 0
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, f"read_error:{exc}", None, 0
    if yaml is None:
        return None, "yaml_unavailable", _sha256_text(raw), len(raw.encode("utf-8"))
    try:
        data = yaml.safe_load(raw) or {}
    except Exception as exc:
        return None, f"yaml_parse_error:{exc}", _sha256_text(raw), len(raw.encode("utf-8"))
    if not isinstance(data, dict):
        return None, "yaml_not_object", _sha256_text(raw), len(raw.encode("utf-8"))
    return data, None, _sha256_text(raw), len(raw.encode("utf-8"))


def _load_provider_policy(root_dir: Path) -> InputSnapshot:
    yaml_path = root_dir / "config" / "provider_policy.yaml"
    json_path = root_dir / "config" / "provider_policy.json"
    if yaml_path.exists():
        data, err, digest, size = _read_yaml_doc(yaml_path)
        return InputSnapshot(
            key="provider_policy",
            required=True,
            path=yaml_path,
            exists=True,
            parsed=data is not None and err is None,
            parse_error=err,
            data=data,
            raw_sha256=digest,
            raw_bytes=size,
            source_kind="yaml",
        )
    data, err, digest, size = _read_json_doc(json_path)
    return InputSnapshot(
        key="provider_policy",
        required=True,
        path=json_path if json_path.exists() else yaml_path,
        exists=json_path.exists(),
        parsed=data is not None and err is None,
        parse_error=err,
        data=data,
        raw_sha256=digest,
        raw_bytes=size,
        source_kind="json" if json_path.exists() else "none",
    )


def _load_json_input(root_dir: Path, *, key: str, rel_path: str, required: bool) -> InputSnapshot:
    path = root_dir / rel_path
    data, err, digest, size = _read_json_doc(path)
    return InputSnapshot(
        key=key,
        required=required,
        path=path,
        exists=path.exists(),
        parsed=data is not None and err is None,
        parse_error=err,
        data=data,
        raw_sha256=digest,
        raw_bytes=size,
        source_kind="json" if path.exists() else "none",
    )


def _load_yaml_input(root_dir: Path, *, key: str, rel_path: str, required: bool) -> InputSnapshot:
    path = root_dir / rel_path
    data, err, digest, size = _read_yaml_doc(path)
    return InputSnapshot(
        key=key,
        required=required,
        path=path,
        exists=path.exists(),
        parsed=data is not None and err is None,
        parse_error=err,
        data=data,
        raw_sha256=digest,
        raw_bytes=size,
        source_kind="yaml" if path.exists() else "none",
    )


def _parse_iso_utc_to_ts(raw: Any) -> Tuple[Optional[float], Optional[str]]:
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    text = str(raw or "").strip()
    if not text:
        return None, None
    try:
        # Accept Z and offsets.
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp()), None
    except Exception as exc:
        return None, f"iso_parse_error:{exc}"


def _safe_int(raw: Any) -> Optional[int]:
    try:
        if raw is None:
            return None
        return int(raw)
    except Exception:
        return None


def _safe_float(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        return float(raw)
    except Exception:
        return None


def _iter_policy_preferences(policy: Mapping[str, Any]) -> Iterable[Tuple[str, str, List[str]]]:
    rails = policy.get("rails")
    if not isinstance(rails, dict):
        return []
    rows: List[Tuple[str, str, List[str]]] = []
    for rail, rail_doc in rails.items():
        if not isinstance(rail_doc, dict):
            continue
        roles = rail_doc.get("roles")
        if not isinstance(roles, dict):
            continue
        for role, role_doc in roles.items():
            if not isinstance(role_doc, dict):
                continue
            pref = role_doc.get("preference")
            if pref is None:
                pref = role_doc.get("preferred")
            if isinstance(pref, str):
                pref = [pref]
            if not isinstance(pref, list):
                pref = []
            clean: List[str] = []
            for item in pref:
                if isinstance(item, str) and item.strip():
                    clean.append(item.strip())
            rows.append((str(rail).strip().lower(), str(role).strip().lower(), clean))
    return rows


def _is_manual_or_opaque_provider(policy: Mapping[str, Any], provider_id: str) -> bool:
    providers = policy.get("providers")
    if not isinstance(providers, dict):
        return False
    ent = providers.get(provider_id)
    if not isinstance(ent, dict):
        return False
    for key in ("status", "source", "kind"):
        val = ent.get(key)
        if isinstance(val, str) and val.strip().lower() in {"manual", "opaque", "known_opaque_provider", "manual_policy"}:
            return True
    rate_limits = ent.get("rate_limits")
    if isinstance(rate_limits, dict):
        src = str(rate_limits.get("source") or "").strip().lower()
        status = str(rate_limits.get("status") or "").strip().lower()
        if src in {"manual_policy", "manual", "opaque"}:
            return True
        if status in {"known_opaque_provider", "manual", "opaque"}:
            return True
    return False


def _provider_status_p95(status_entry: Mapping[str, Any]) -> bool:
    for key in ("latency_p95_ms", "total_p95_ms", "ttft_p95_ms"):
        val = status_entry.get(key)
        if isinstance(val, (int, float)) and float(val) > 0:
            return True
    return False


def _unique_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _row_is_effectively_available(row: Mapping[str, Any], *, now_ts: float) -> bool:
    if str(row.get("status") or "").strip().lower() != "ok":
        return False
    cooldown_ts = _safe_float(row.get("cooldown_until_ts"))
    if cooldown_ts is not None and cooldown_ts > now_ts:
        return False
    return True


def _write_artifacts(root_dir: Path, result: AuditResult, ts: float) -> Tuple[Path, Path, Path]:
    label = _ts_label(ts)
    audit_dir = root_dir / "artifacts" / "audit" / f"providers_{label}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_json_path = audit_dir / "providers_survival_audit.json"
    audit_json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md_path = audit_dir / "providers_survival_audit.md"
    summary = result.summary if isinstance(result.summary, dict) else {}
    lines: List[str] = [
        "# OPS Providers Survival Audit",
        "",
        f"- schema: {result.schema}",
        f"- timestamp: {result.timestamp_iso}",
        f"- mode: {result.mode}",
        f"- ok: {result.ok}",
        f"- read_only: {result.read_only}",
        f"- live_probes_invoked: {result.live_probes_invoked}",
        "",
        "## Summary",
        "",
        f"- critical: {summary.get('critical', 0)}",
        f"- error: {summary.get('error', 0)}",
        f"- warn: {summary.get('warn', 0)}",
        f"- info: {summary.get('info', 0)}",
        f"- rails_roles_checked: {summary.get('rails_roles_checked', 0)}",
        f"- quorum_impossible_roles: {summary.get('quorum_impossible_roles', 0)}",
        "",
        "## Inputs",
        "",
    ]
    inputs = result.inputs.get("files") if isinstance(result.inputs, dict) else None
    if isinstance(inputs, dict):
        for key, meta in inputs.items():
            if not isinstance(meta, dict):
                continue
            lines.append(
                "- {key}: exists={exists} parsed={parsed} path={path}".format(
                    key=key,
                    exists=meta.get("exists"),
                    parsed=meta.get("parsed"),
                    path=meta.get("path"),
                )
            )
    lines.extend(["", "## Findings", ""])
    if result.findings:
        for idx, finding in enumerate(result.findings[:18], start=1):
            lines.append(
                f"{idx}. [{finding.severity.upper()}] {finding.code} - {finding.title}"
            )
            if finding.next_action:
                lines.append(f"   next_action: `{finding.next_action}`")
    else:
        lines.append("1. No findings.")
    lines.extend(["", "## Recommended Actions", ""])
    if result.recommended_actions:
        for idx, action in enumerate(result.recommended_actions[:12], start=1):
            lines.append(
                f"{idx}. ({action.get('severity')}/{action.get('code')}) `{action.get('command')}`"
            )
    else:
        lines.append("1. No action required.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipts_dir = root_dir / "artifacts" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"providers_survival_audit_{label}.json"
    inputs_files = result.inputs.get("files") if isinstance(result.inputs, dict) else {}
    hashes = {
        key: meta.get("raw_sha256")
        for key, meta in (inputs_files.items() if isinstance(inputs_files, dict) else [])
        if isinstance(meta, dict)
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "ts_utc": _iso_utc(ts),
        "root": str(root_dir),
        "ok": result.ok,
        "summary_counts": {
            "critical": int(summary.get("critical") or 0),
            "error": int(summary.get("error") or 0),
            "warn": int(summary.get("warn") or 0),
            "info": int(summary.get("info") or 0),
        },
        "paths_written": [str(audit_json_path), str(md_path), str(receipt_path)],
        "schema_version": AUDIT_SCHEMA,
        "hashes": hashes,
        "read_only": True,
        "live_probes_invoked": False,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit_json_path, md_path, receipt_path


def _build_findings_for_inputs(snapshots: Mapping[str, InputSnapshot]) -> List[Finding]:
    findings: List[Finding] = []
    for key, snap in snapshots.items():
        if snap.required and (not snap.exists):
            sev = "error"
            if key == "timeouts_policy":
                sev = "warn"
            findings.append(
                Finding(
                    id=f"INPUT-MISSING-{key.upper()}",
                    severity=sev,
                    title=f"Required input missing: {key}",
                    code=f"{key}_missing",
                    evidence=[{"path": str(snap.path) if snap.path else None}],
                    invariant="AGENTS.md §4 / no silent degradation; config+artifacts must be auditable",
                    recommendation="Restore the canonical input file or regenerate it before trusting provider routing.",
                    next_action=(
                        "python bin/ajaxctl doctor providers"
                        if key in {"providers_status", "ledger", "timeouts_policy"}
                        else None
                    ),
                )
            )
            continue
        if snap.exists and not snap.parsed:
            sev = "error"
            if key in {"timeouts_policy", "failure_policy"}:
                sev = "warn"
            findings.append(
                Finding(
                    id=f"INPUT-PARSE-{key.upper()}",
                    severity=sev,
                    title=f"Input parse failure: {key}",
                    code=f"{key}_parse_error",
                    evidence=[
                        {
                            "path": str(snap.path) if snap.path else None,
                            "parse_error": snap.parse_error,
                            "source_kind": snap.source_kind,
                        }
                    ],
                    invariant="AGENTS.md §X Proof-Carrying: inputs must be parseable to support claims",
                    recommendation="Fix the malformed config/artifact and rerun the audit.",
                    next_action=(
                        f"python -m json.tool {snap.path}" if snap.source_kind == "json" and snap.path else None
                    ),
                )
            )
    if "failure_policy" in snapshots:
        snap = snapshots["failure_policy"]
        if not snap.exists:
            findings.append(
                Finding(
                    id="FAILURE-POLICY-OPTIONAL-MISSING",
                    severity="warn",
                    title="Optional provider_failure_policy missing",
                    code="failure_policy_missing_optional",
                    evidence=[{"path": str(snap.path) if snap.path else None}],
                    invariant="AGENTS.md §8.1 ladder/timeouts should be config-governed",
                    recommendation="Add config/provider_failure_policy.yaml to codify cooldown and receipt requirements.",
                    next_action="python bin/ajaxctl doctor providers",
                )
            )
    return findings


def _status_provider_map(status_doc: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    providers = status_doc.get("providers")
    if not isinstance(providers, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in providers.items():
        if isinstance(key, str) and key.strip() and isinstance(val, dict):
            out[key.strip()] = val
    return out


def _ledger_rows(ledger_doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = ledger_doc.get("rows")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict):
            out.append(item)
    return out


def _append_quorum_findings(
    findings: List[Finding],
    *,
    policy: Mapping[str, Any],
    providers_status: Mapping[str, Dict[str, Any]],
    ledger_rows: List[Dict[str, Any]],
    timeouts_policy: Optional[Mapping[str, Any]],
    now_ts: float,
) -> Tuple[int, int]:
    rows_by_provider_role: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in ledger_rows:
        provider = str(row.get("provider") or "").strip()
        role = str(row.get("role") or "").strip().lower()
        if not provider or not role:
            continue
        rows_by_provider_role.setdefault((provider, role), []).append(row)

    rails_roles_checked = 0
    quorum_impossible_roles = 0
    ladder_providers_seen: List[str] = []

    for rail, role, pref in _iter_policy_preferences(policy):
        rails_roles_checked += 1
        if role in CORE_ROLES and not pref:
            findings.append(
                Finding(
                    id=f"LADDER-EMPTY-{rail}-{role}",
                    severity="error",
                    title=f"Empty provider ladder for {rail}.{role}",
                    code="policy_ladder_empty",
                    evidence=[{"rail": rail, "role": role, "preference": pref}],
                    invariant="AGENTS.md §8.1 ladder policy must be config-defined and usable",
                    recommendation="Populate rails.<rail>.roles.<role>.preference with at least one provider.",
                    next_action="python bin/ajaxctl doctor providers",
                )
            )
        if len(pref) != len(_unique_keep_order(pref)):
            findings.append(
                Finding(
                    id=f"LADDER-DUP-{rail}-{role}",
                    severity="warn",
                    title=f"Duplicate providers in ladder {rail}.{role}",
                    code="policy_ladder_duplicates",
                    evidence=[{"rail": rail, "role": role, "preference": pref}],
                    invariant="AGENTS.md §4 abstraction/reusability; deterministic ladders",
                    recommendation="Deduplicate provider preferences to avoid misleading quorum/availability counts.",
                )
            )

        for provider_id in _unique_keep_order(pref):
            ladder_providers_seen.append(provider_id)
            if provider_id not in providers_status and not _is_manual_or_opaque_provider(policy, provider_id):
                findings.append(
                    Finding(
                        id=f"POLICY-STATUS-MISSING-{rail}-{role}-{provider_id}",
                        severity="error",
                        title=f"Provider in ladder missing from providers_status: {provider_id}",
                        code="policy_provider_missing_in_status",
                        evidence=[{"rail": rail, "role": role, "provider": provider_id}],
                        invariant="AGENTS.md §10.3 no silent degradation (fallback/availability must be explicit)",
                        recommendation="Refresh providers_status or mark the provider explicitly as manual/opaque in policy.",
                        next_action="python bin/ajaxctl doctor providers",
                    )
                )

        pref_unique = _unique_keep_order(pref)
        raw_present = 0
        effective_ok = 0
        blocked: List[Dict[str, Any]] = []
        for provider_id in pref_unique:
            rows = rows_by_provider_role.get((provider_id, role), [])
            if rows:
                raw_present += 1
            row_available = False
            row_states: List[str] = []
            for row in rows:
                status = str(row.get("status") or "").strip().lower() or "unknown"
                row_states.append(status)
                if _row_is_effectively_available(row, now_ts=now_ts):
                    row_available = True
            if row_available:
                effective_ok += 1
            elif rows:
                blocked.append({"provider": provider_id, "statuses": sorted(set(row_states))})

        if role in CORE_ROLES and pref_unique:
            if effective_ok <= 0:
                quorum_impossible_roles += 1
                sev = "critical" if role in {"brain", "council"} and rail == "prod" else "error"
                findings.append(
                    Finding(
                        id=f"QUORUM-IMPOSSIBLE-{rail}-{role}",
                        severity=sev,
                        title=f"No effective provider available for {rail}.{role}",
                        code="quorum_effective_impossible",
                        evidence=[
                            {
                                "rail": rail,
                                "role": role,
                                "preference": pref_unique,
                                "ledger_raw_present_count": raw_present,
                                "ledger_effective_ok_count": effective_ok,
                                "blocked": blocked,
                            }
                        ],
                        invariant="AGENTS.md §10.3 / status!=ok no cuenta para quorum",
                        recommendation="Recover at least one provider in ledger status=ok (or adjust ladder/config with evidence).",
                        next_action="python bin/ajaxctl doctor providers",
                    )
                )
            elif role == "council" and len(pref_unique) >= 2 and effective_ok < 2:
                findings.append(
                    Finding(
                        id=f"COUNCIL-QUORUM-RISK-{rail}",
                        severity="warn",
                        title=f"Council quorum degradation risk in {rail}",
                        code="council_quorum_risk",
                        evidence=[
                            {
                                "rail": rail,
                                "role": role,
                                "preference": pref_unique,
                                "ledger_effective_ok_count": effective_ok,
                                "ledger_raw_present_count": raw_present,
                            }
                        ],
                        invariant="AGENTS.md §10/§11 quorum and escalation governance",
                        recommendation="Ensure at least two council-capable providers are effectively available or document quorum degrade policy.",
                        next_action="python bin/ajaxctl doctor providers",
                    )
                )

        if role in CORE_ROLES and raw_present > 0 and effective_ok == 0:
            findings.append(
                Finding(
                    id=f"QUORUM-COUNT-MISMATCH-{rail}-{role}",
                    severity="warn",
                    title=f"Naive quorum count would overestimate availability for {rail}.{role}",
                    code="quorum_naive_count_mismatch",
                    evidence=[
                        {
                            "rail": rail,
                            "role": role,
                            "ledger_raw_present_count": raw_present,
                            "ledger_effective_ok_count": effective_ok,
                        }
                    ],
                    invariant="AGENTS.md §10.3 status!=ok must not count as available",
                    recommendation="Count only ledger status=ok (and not cooling down) when computing effective quorum.",
                )
            )

    # Timeout bench / p95 presence in providers_status for ladder providers (warn-only).
    ladder_providers = _unique_keep_order(ladder_providers_seen)
    missing_p95: List[str] = []
    if providers_status:
        for provider_id in ladder_providers:
            status_entry = providers_status.get(provider_id)
            if not isinstance(status_entry, dict):
                continue
            if not _provider_status_p95(status_entry):
                missing_p95.append(provider_id)
    if missing_p95:
        findings.append(
            Finding(
                id="TIMEOUTS-NO-P95-BASE",
                severity="warn",
                title="Providers in ladders without observed p95 baseline in providers_status",
                code="timeouts_missing_p95_base",
                evidence=[{"providers": missing_p95, "count": len(missing_p95)}],
                invariant="PSEUDOCODE_MAP/00 fast-mode provider preflight should use observed signals when available",
                recommendation="Refresh providers_status and bench-derived metrics before trusting timeout tuning decisions.",
                next_action="python bin/ajaxctl doctor providers",
            )
        )

    if isinstance(timeouts_policy, dict):
        defaults = timeouts_policy.get("defaults")
        if not isinstance(defaults, dict):
            findings.append(
                Finding(
                    id="TIMEOUTS-DEFAULTS-MISSING",
                    severity="warn",
                    title="provider_timeouts_policy missing defaults block",
                    code="timeouts_policy_defaults_missing",
                    evidence=[{"path": "config/provider_timeouts_policy.json"}],
                    invariant="AGENTS.md §8.1 timeouts should be config-governed",
                    recommendation="Add config/provider_timeouts_policy.json defaults for connect/ttft/stall/total timeouts.",
                    next_action="python bin/ajaxctl doctor providers",
                )
            )
        else:
            missing_keys = sorted(TIMEOUT_DEFAULT_KEYS.difference(defaults.keys()))
            if missing_keys:
                findings.append(
                    Finding(
                        id="TIMEOUTS-DEFAULT-KEYS-MISSING",
                        severity="warn",
                        title="provider_timeouts_policy defaults missing required keys",
                        code="timeouts_policy_defaults_keys_missing",
                        evidence=[{"missing_keys": missing_keys, "defaults_keys": sorted(defaults.keys())}],
                        invariant="AGENTS.md §8.1 timeouts should be config-governed",
                        recommendation="Populate missing timeout defaults to avoid implicit runtime fallbacks.",
                        next_action="python bin/ajaxctl doctor providers",
                    )
                )
    return rails_roles_checked, quorum_impossible_roles


def _append_ledger_findings(
    findings: List[Finding],
    *,
    ledger_rows: List[Dict[str, Any]],
    policy: Mapping[str, Any],
    now_ts: float,
) -> None:
    if not ledger_rows:
        findings.append(
            Finding(
                id="LEDGER-EMPTY",
                severity="error",
                title="Provider ledger has no rows",
                code="ledger_rows_empty",
                evidence=[{"path": "artifacts/provider_ledger/latest.json", "rows": 0}],
                invariant="AGENTS.md §10.3 availability/quorum require durable ledger evidence",
                recommendation="Refresh ledger before provider routing decisions.",
                next_action="python bin/ajaxctl doctor providers",
            )
        )
        return

    roles_present = {
        str(row.get("role") or "").strip().lower()
        for row in ledger_rows
        if isinstance(row, dict)
    }
    for rail, role, pref in _iter_policy_preferences(policy):
        if not pref:
            continue
        if role in CORE_ROLES and role not in roles_present:
            findings.append(
                Finding(
                    id=f"LEDGER-MISSING-ROLE-{rail}-{role}",
                    severity="error",
                    title=f"Ledger missing rows for required role {role}",
                    code="ledger_missing_required_role",
                    evidence=[{"rail": rail, "role": role, "roles_present": sorted(roles_present)}],
                    invariant="AGENTS.md §10 quorum requires ledger rows for role accounting",
                    recommendation="Refresh provider ledger and ensure probes/status mapping emits rows for required roles.",
                    next_action="python bin/ajaxctl doctor providers",
                )
            )

    for idx, row in enumerate(ledger_rows):
        status = str(row.get("status") or "").strip().lower()
        provider = str(row.get("provider") or "").strip()
        role = str(row.get("role") or "").strip().lower()
        if status not in VALID_LEDGER_STATUS:
            findings.append(
                Finding(
                    id=f"LEDGER-STATUS-INVALID-{idx}",
                    severity="error",
                    title="Ledger row has invalid status",
                    code="ledger_status_invalid",
                    evidence=[{"index": idx, "provider": provider, "role": role, "status": row.get("status")}],
                    invariant="AGENTS.md §10.3 status must be explicit for availability accounting",
                    recommendation="Restrict ledger statuses to ok|soft_fail|hard_fail.",
                    next_action="python bin/ajaxctl doctor providers",
                )
            )

        cooldown_txt = row.get("cooldown_until")
        cooldown_ts = _safe_float(row.get("cooldown_until_ts"))
        parsed_cooldown_ts, cooldown_parse_err = _parse_iso_utc_to_ts(cooldown_txt)
        if cooldown_txt is not None and cooldown_parse_err:
            findings.append(
                Finding(
                    id=f"LEDGER-COOLDOWN-ISO-INVALID-{idx}",
                    severity="warn",
                    title="Ledger row cooldown_until is not parseable ISO UTC",
                    code="ledger_cooldown_until_unparseable",
                    evidence=[{"index": idx, "provider": provider, "role": role, "cooldown_until": cooldown_txt}],
                    invariant="AGENTS.md §X proof-carrying timestamps must be parseable",
                    recommendation="Write cooldown_until as ISO-8601 UTC (e.g., 2026-02-23T21:54:04Z) or null.",
                    next_action="python bin/ajaxctl doctor providers",
                )
            )
        if parsed_cooldown_ts is not None and cooldown_ts is not None:
            if abs(parsed_cooldown_ts - cooldown_ts) > 2.0:
                findings.append(
                    Finding(
                        id=f"LEDGER-COOLDOWN-NONMONOTONIC-{idx}",
                        severity="warn",
                        title="Ledger cooldown_until and cooldown_until_ts are inconsistent",
                        code="ledger_cooldown_nonmonotonic",
                        evidence=[
                            {
                                "index": idx,
                                "provider": provider,
                                "role": role,
                                "cooldown_until": cooldown_txt,
                                "cooldown_until_ts": cooldown_ts,
                                "parsed_cooldown_until_ts": parsed_cooldown_ts,
                            }
                        ],
                        invariant="AGENTS.md §X proof-carrying timestamps must agree across representations",
                        recommendation="Persist cooldown_until and cooldown_until_ts from the same source timestamp.",
                        next_action="python bin/ajaxctl doctor providers",
                    )
                )

        if status == "hard_fail" and cooldown_ts is not None and cooldown_ts < now_ts:
            findings.append(
                Finding(
                    id=f"LEDGER-HARDFAIL-STALE-COOLDOWN-{idx}",
                    severity="warn",
                    title="Cooldown expired but ledger row remains hard_fail",
                    code="ledger_hard_fail_cooldown_expired",
                    evidence=[
                        {
                            "index": idx,
                            "provider": provider,
                            "role": role,
                            "cooldown_until_ts": cooldown_ts,
                            "now_ts": now_ts,
                            "status": status,
                        }
                    ],
                    invariant="AGENTS.md §10.2 availability fallback should use ledger+cooldown coherently",
                    recommendation="Refresh provider status/ledger; stale hard_fail may block valid fallback/quorum.",
                    next_action="python bin/ajaxctl doctor providers",
                )
            )


def _append_receipt_policy_findings(
    findings: List[Finding],
    *,
    failure_policy: Optional[Mapping[str, Any]],
) -> None:
    if not isinstance(failure_policy, dict):
        return
    receipts = failure_policy.get("receipts")
    if not isinstance(receipts, dict):
        findings.append(
            Finding(
                id="FAILURE-POLICY-RECEIPTS-MISSING",
                severity="warn",
                title="provider_failure_policy lacks receipts policy block",
                code="failure_policy_receipts_policy_missing",
                evidence=[{"path": "config/provider_failure_policy.yaml"}],
                invariant="AGENTS.md §10.3/§12 no degradation without receipt metadata",
                recommendation="Define receipts.required_fields in provider_failure_policy to enforce degradation traceability.",
            )
        )
        return
    req = receipts.get("required_fields")
    if not isinstance(req, list) or not any(isinstance(x, str) and x.strip() for x in req):
        findings.append(
            Finding(
                id="FAILURE-POLICY-RECEIPT-FIELDS-EMPTY",
                severity="warn",
                title="provider_failure_policy receipts.required_fields is empty or invalid",
                code="failure_policy_receipt_required_fields_invalid",
                evidence=[{"path": "config/provider_failure_policy.yaml", "required_fields": req}],
                invariant="AGENTS.md §10.3 fallback receipts must be structured",
                recommendation="Populate receipts.required_fields with a minimal required schema for degradation receipts.",
            )
        )


def run_providers_survival_audit(root_dir: Path, *, mode: str = "fast") -> Dict[str, Any]:
    """
    Run a read-only providers survival audit.

    The audit never performs live probes or network calls. It only reads local files/config and
    writes a report + receipt in artifacts/audit and artifacts/receipts.
    """
    ts = _now_ts()
    mode_n = str(mode or "fast").strip().lower() or "fast"
    root = Path(root_dir)

    snapshots: Dict[str, InputSnapshot] = {
        "provider_policy": _load_provider_policy(root),
        "providers_status": _load_json_input(
            root, key="providers_status", rel_path="artifacts/health/providers_status.json", required=True
        ),
        "ledger": _load_json_input(
            root, key="ledger", rel_path="artifacts/provider_ledger/latest.json", required=True
        ),
        "timeouts_policy": _load_json_input(
            root, key="timeouts_policy", rel_path="config/provider_timeouts_policy.json", required=True
        ),
        "failure_policy": _load_yaml_input(
            root, key="failure_policy", rel_path="config/provider_failure_policy.yaml", required=False
        ),
    }

    findings: List[Finding] = []
    findings.extend(_build_findings_for_inputs(snapshots))

    policy_doc = snapshots["provider_policy"].data if snapshots["provider_policy"].parsed else {}
    status_doc = snapshots["providers_status"].data if snapshots["providers_status"].parsed else {}
    ledger_doc = snapshots["ledger"].data if snapshots["ledger"].parsed else {}
    timeouts_doc = snapshots["timeouts_policy"].data if snapshots["timeouts_policy"].parsed else {}
    failure_doc = snapshots["failure_policy"].data if snapshots["failure_policy"].parsed else None

    providers_status = _status_provider_map(status_doc or {})
    ledger_rows = _ledger_rows(ledger_doc or {})

    if snapshots["providers_status"].parsed and not providers_status:
        findings.append(
            Finding(
                id="STATUS-PROVIDERS-EMPTY",
                severity="error",
                title="providers_status has no providers map",
                code="providers_status_empty",
                evidence=[{"path": str(snapshots['providers_status'].path), "providers": []}],
                invariant="AGENTS.md §10 availability signals must be explicit",
                recommendation="Regenerate providers_status.json with provider entries.",
                next_action="python bin/ajaxctl doctor providers",
            )
        )

    _append_ledger_findings(findings, ledger_rows=ledger_rows, policy=policy_doc or {}, now_ts=ts)
    rails_roles_checked, quorum_impossible_roles = _append_quorum_findings(
        findings,
        policy=policy_doc or {},
        providers_status=providers_status,
        ledger_rows=ledger_rows,
        timeouts_policy=timeouts_doc or {},
        now_ts=ts,
    )
    _append_receipt_policy_findings(findings, failure_policy=failure_doc)

    summary = {
        "critical": sum(1 for f in findings if f.severity == "critical"),
        "error": sum(1 for f in findings if f.severity == "error"),
        "warn": sum(1 for f in findings if f.severity == "warn"),
        "info": sum(1 for f in findings if f.severity == "info"),
        "rails_roles_checked": rails_roles_checked,
        "quorum_impossible_roles": quorum_impossible_roles,
        "providers_status_count": len(providers_status),
        "ledger_rows_count": len(ledger_rows),
    }
    ok = int(summary["critical"]) == 0 and int(summary["error"]) == 0

    recommended_actions: List[Dict[str, str]] = []
    seen_actions: set[tuple[str, str]] = set()
    for finding in findings:
        if finding.severity not in {"critical", "error"}:
            continue
        cmd = (finding.next_action or "").strip()
        if not cmd:
            continue
        key = (finding.code, cmd)
        if key in seen_actions:
            continue
        seen_actions.add(key)
        recommended_actions.append(
            {"severity": finding.severity, "code": finding.code, "command": cmd}
        )

    result = AuditResult(
        ok=ok,
        schema=AUDIT_SCHEMA,
        timestamp=ts,
        timestamp_iso=_iso_utc(ts),
        mode=mode_n,
        read_only=True,
        live_probes_invoked=False,
        findings=findings,
        summary=summary,
        inputs={
            "files": {key: snap.to_dict() for key, snap in snapshots.items()},
            "hashes": {key: snap.raw_sha256 for key, snap in snapshots.items()},
        },
        recommended_actions=recommended_actions,
        meta={
            "root_dir": str(root),
            "mode": mode_n,
            "fast_mode_read_only_contract": True,
            "live_probes_invoked": False,
        },
    )

    audit_path, md_path, receipt_path = _write_artifacts(root, result, ts)
    out = result.to_dict()
    out["artifact_path"] = str(audit_path)
    out["artifact_md_path"] = str(md_path)
    out["summary_md_path"] = str(md_path)
    out["receipt_path"] = str(receipt_path)
    out["paths_written"] = [str(audit_path), str(md_path), str(receipt_path)]
    return out


__all__ = [
    "AUDIT_SCHEMA",
    "RECEIPT_SCHEMA",
    "Finding",
    "AuditResult",
    "run_providers_survival_audit",
]
