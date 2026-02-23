"""
Transactional Integrity Audit (CAPA 2) - Read-only audit for T2+ compliance.

According to AGENTS.md §2.1 (Ley de Integridad Transaccional):
- Ninguna acción que modifique el estado físico del sistema es válida si no sigue
  obligatoriamente el ciclo: PREPARE → APPLY → VERIFY → UNDO (si falla)
- La capacidad de deshacer precede a la capacidad de hacer.

According to AGENTS.md §X (Proof-Carrying Output):
- Ninguna afirmación "confirmada/culpable/resuelto" es válida sin EvidenceRefs tipadas
- Sin evidence → degradar a HIPÓTESIS + comandos de verificación

This audit is READ-ONLY: no mutations to runs, state, ledger, or providers.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from agency.infraguard import classify_action as _infraguard_classify_action
except Exception:  # pragma: no cover - optional import
    _infraguard_classify_action = None  # type: ignore

# Schema identifiers
AUDIT_SCHEMA = "ajax.audit.tx_integrity.v0"
RECEIPT_SCHEMA = "ajax.receipt.tx_integrity_audit.v1"

# Valid claim types from agency/types/minimums.py
VALID_CLAIM_TYPES = {"fixed", "root_cause", "available_green", "diagnosed", "verified", "confirmed"}

# Evidence minimums per claim type (from PSEUDOCODE_MAP/01_INTERFACES.pseudo.md)
CLAIM_EVIDENCE_MINIMUMS: Dict[str, List[str]] = {
    "fixed": ["verify_result", "efe"],
    "root_cause": ["receipt", "log"],
    "available_green": ["providers_quota.json", "providers_status.json"],
    "diagnosed": ["snapshot", "log"],
    "verified": ["verify_result"],
    "confirmed": ["verify_result", "efe"],
}

# Alternative evidence combinations (OR logic)
ALTERNATIVE_EVIDENCES: Dict[str, List[List[str]]] = {
    "root_cause": [
        ["receipt", "log"],
        ["receipt", "state_before", "state_after"],
    ]
}

# Actions that are considered "mutating" and require TransactionalStep
# Based on agency/infraguard.py CRITICAL_ACTIONS and classification logic
MUTATING_ACTION_PATTERNS = {
    # Critical actions (from infraguard.py)
    "powershell.run",
    "powershell.exec",
    "cmd.exec",
    "shell.exec",
    "file.delete",
    "app.kill",
    "process.kill",
    "system.stop",
    # Additional patterns that indicate mutation
    "file.write",
    "file.move",
    "file.copy",
    "file.create",
    "registry.set",
    "registry.delete",
    "service.start",
    "service.stop",
    "service.restart",
    "process.start",
    "process.terminate",
    "network.block",
    "network.allow",
    "user.create",
    "user.delete",
    "user.modify",
    "install.",
    "uninstall.",
    "update.",
}

# Actions that are safe (read-only, observational)
SAFE_ACTION_PATTERNS = {
    "file.read",
    "file.list",
    "file.stat",
    "registry.get",
    "registry.list",
    "process.list",
    "process.info",
    "network.status",
    "network.list",
    "user.list",
    "user.info",
    "screenshot",
    "snap",
    "probe",
    "doctor.",
    "check.",
    "verify.",
    "audit.",
    "query",
    "search",
    "get",
    "list",
    "show",
    "describe",
}


@dataclass
class Finding:
    """
    A single audit finding.

    Args:
        id: Unique identifier for the finding
        severity: info|warn|error|critical
        title: Short title describing the issue
        evidence: List of paths/snippets that support the finding
        invariant: The invariant being violated (reference to AGENTS.md)
        recommendation: Suggested fix
        affected_run_id: Run ID where the issue was found
        affected_step_id: Step ID if applicable
    """

    id: str
    severity: str  # info|warn|error|critical
    title: str
    code: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    invariant: str = ""
    recommendation: str = ""
    affected_run_id: str = ""
    affected_step_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditResult:
    """
    Result of the transactional integrity audit.

    Args:
        ok: True if no critical/error findings
        schema: Schema identifier
        timestamp: Unix timestamp
        timestamp_iso: ISO 8601 UTC timestamp
        runs_audited: Number of runs audited
        runs_with_findings: Number of runs with findings
        findings: List of findings
        summary: Summary counts by severity
        meta: Metadata about the audit execution
    """

    ok: bool
    schema: str = AUDIT_SCHEMA
    timestamp: float = 0.0
    timestamp_iso: str = ""
    runs_audited: int = 0
    runs_with_findings: int = 0
    findings: List[Finding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    read_only: bool = True
    live_probes_invoked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "schema": self.schema,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "read_only": self.read_only,
            "live_probes_invoked": self.live_probes_invoked,
            "runs_audited": self.runs_audited,
            "runs_with_findings": self.runs_with_findings,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "meta": self.meta,
        }


def _now_ts() -> float:
    return time.time()


def _iso_utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _ts_label(ts: float) -> str:
    return time.strftime("%Y%m%d-%H%M%SZ", time.gmtime(ts))


def _sha256_text(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Safely read a JSON file."""
    try:
        if not path.exists():
            return None, "file_not_found"
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"json_parse_error: {e}"
    except Exception as e:
        return None, str(e)


def _is_mutating_action(action_name: str, args: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Determine if an action is mutating (requires TransactionalStep).

    Returns:
        (is_mutating, reason)
    """
    if not action_name:
        return False, "empty_action_name"

    lower = action_name.lower().strip()

    if lower == "await_user_input":
        return False, "explicit_non_mutating:await_user_input"

    # Check safe patterns first
    for safe_pattern in SAFE_ACTION_PATTERNS:
        if lower.startswith(safe_pattern) or lower == safe_pattern:
            return False, f"safe_action_pattern:{safe_pattern}"

    # Check mutating patterns
    for mut_pattern in MUTATING_ACTION_PATTERNS:
        if lower.startswith(mut_pattern) or lower == mut_pattern:
            return True, f"mutating_action_pattern:{mut_pattern}"

    # Heuristic: actions with certain keywords
    mutating_keywords = {
        "delete",
        "kill",
        "stop",
        "write",
        "create",
        "modify",
        "update",
        "install",
        "uninstall",
        "exec",
        "run",
        "start",
    }
    for kw in mutating_keywords:
        if kw in lower:
            return True, f"mutating_keyword:{kw}"

    # Prefer runtime classifier when available, but only after explicit safe/mutating patterns.
    # `infraguard.classify_action` defaults many unknowns to "moderate", so we preserve the
    # conservative/explicit semantics of this audit helper for truly unknown actions.
    if _infraguard_classify_action is not None:
        try:
            classification, cls_reason = _infraguard_classify_action({"tool": action_name, "args": args or {}})
            if classification == "critical":
                return True, f"infraguard:critical:{cls_reason}"
            if classification == "safe":
                return False, f"infraguard:safe:{cls_reason}"
            if classification == "moderate" and cls_reason != "default":
                return True, f"infraguard:moderate:{cls_reason}"
        except Exception:
            pass

    # Default: unknown actions are treated as potentially mutating
    # This is conservative per AGENTS.md principle
    return True, "unknown_action_conservative"


def _infer_evidence_kind(ev: Dict[str, Any]) -> Optional[str]:
    kind = ev.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    # Compat path-only parsing (tests/fixtures may omit `kind`)
    path_val = ev.get("path")
    if not isinstance(path_val, str) or not path_val.strip():
        return None
    p = path_val.strip().lower().replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if name in {"verification.json", "verify.json", "verify_result.json"}:
        return "verify_result"
    if "verify_result" in name:
        return "verify_result"
    if "efe" in name:
        return "efe"
    if "receipt" in name:
        return "receipt"
    if name.endswith(".log") or "log" in name:
        return "log"
    if "snapshot" in name:
        return "snapshot"
    if "state_before" in name:
        return "state_before"
    if "state_after" in name:
        return "state_after"
    if name == "providers_quota.json":
        return "providers_quota.json"
    if name == "providers_status.json":
        return "providers_status.json"
    return None


def _validate_claim_evidence(claim: Dict[str, Any]) -> Tuple[bool, List[str], str]:
    """
    Validate that a claim has minimum evidence.

    Returns:
        (is_valid, missing_evidence, reason)
    """
    claim_type = claim.get("type", "")
    if claim_type not in VALID_CLAIM_TYPES:
        return False, [], f"invalid_claim_type:{claim_type}"

    evidence_refs = claim.get("evidence_refs", [])
    if not isinstance(evidence_refs, list):
        evidence_refs = []

    evidence_kinds = set()
    for ev in evidence_refs:
        if isinstance(ev, dict):
            inferred = _infer_evidence_kind(ev)
            if inferred:
                evidence_kinds.add(inferred)

    # Check alternatives first
    if claim_type in ALTERNATIVE_EVIDENCES:
        for alternative in ALTERNATIVE_EVIDENCES[claim_type]:
            if all(req in evidence_kinds for req in alternative):
                return True, [], "alternative_evidence_satisfied"

    # Check standard minimums
    required = set(CLAIM_EVIDENCE_MINIMUMS.get(claim_type, []))
    if not required:
        # No minimums defined - any evidence is valid
        return len(evidence_kinds) > 0, [], "no_minimums_defined"

    missing = list(required - evidence_kinds)
    if missing:
        return False, missing, f"missing_evidence:{','.join(missing)}"

    return True, [], "evidence_minimums_satisfied"


def _check_step_transactional_integrity(
    step: Dict[str, Any],
    run_id: str,
    step_idx: int,
) -> Tuple[List[Finding], Dict[str, int]]:
    """
    Check if a step that is mutating follows TransactionalStep pattern.

    According to PSEUDOCODE_MAP/01_INTERFACES.pseudo.md:
    type TransactionalStep:
      prepare: {snapshot: bool, undo_script_gen: bool}
      apply: TaskStep
      verify: {doctor_check: bool, efe_check: bool}
      undo: {rollback_best_effort: bool}
    """
    findings: List[Finding] = []
    counters = {"transactional_wrappers_seen": 0}

    # Extract action name
    action = step.get("action", step.get("tool", ""))
    step_id = step.get("id", f"step_{step_idx}")

    if not action:
        return findings, counters

    is_mutating, reason = _is_mutating_action(str(action), step.get("args") if isinstance(step.get("args"), dict) else None)

    if not is_mutating:
        return findings, counters

    # Check for TransactionalStep structure
    tx = step.get("transactional", step.get("tx", {}))
    if isinstance(tx, dict) and tx:
        counters["transactional_wrappers_seen"] += 1

    # Check prepare phase
    prepare = tx.get("prepare", {})
    has_snapshot = prepare.get("snapshot", False) if isinstance(prepare, dict) else False
    has_undo_script = prepare.get("undo_script_gen", False) if isinstance(prepare, dict) else False

    if not has_snapshot:
        findings.append(
            Finding(
                id=f"TX-PREPARE-SNAPSHOT-{run_id}-{step_id}",
                code="tx_prepare_snapshot_missing",
                severity="error",
                title="Mutating step without PREPARE snapshot",
                evidence=[{"step": step_id, "action": action, "prepare": prepare}],
                invariant="AGENTS.md:§2.1 - MUST snapshot + script de reversión en PREPARE",
                recommendation="Add prepare.snapshot=true before executing mutating action",
                affected_run_id=run_id,
                affected_step_id=step_id,
            )
        )

    if not has_undo_script:
        findings.append(
            Finding(
                id=f"TX-PREPARE-UNDO-{run_id}-{step_id}",
                code="tx_prepare_undo_script_missing",
                severity="critical",
                title="Mutating step without UNDO script generation",
                evidence=[{"step": step_id, "action": action, "prepare": prepare}],
                invariant="AGENTS.md:§2.1 - MUST generar script de reversión en PREPARE",
                recommendation="Add prepare.undo_script_gen=true to enable rollback",
                affected_run_id=run_id,
                affected_step_id=step_id,
            )
        )

    # Check verify phase
    verify = tx.get("verify", {})
    has_doctor_check = verify.get("doctor_check", False) if isinstance(verify, dict) else False
    has_efe_check = verify.get("efe_check", False) if isinstance(verify, dict) else False

    # Also check for success_spec.expected_state as EFE indicator
    success_spec = step.get("success_spec", {})
    expected_state = (
        success_spec.get("expected_state", {}) if isinstance(success_spec, dict) else {}
    )
    has_expected_state = bool(expected_state)

    if not has_doctor_check:
        findings.append(
            Finding(
                id=f"TX-VERIFY-DOCTOR-{run_id}-{step_id}",
                code="tx_verify_doctor_check_missing",
                severity="warn",
                title="TransactionalStep verify missing doctor_check",
                evidence=[{"step": step_id, "action": action, "verify": verify}],
                invariant="PSEUDOCODE_MAP/01_INTERFACES - verify.doctor_check should be present",
                recommendation="Add verify.doctor_check=true for system health verification",
                affected_run_id=run_id,
                affected_step_id=step_id,
            )
        )

    if not has_efe_check and has_expected_state:
        findings.append(
            Finding(
                id=f"TX-VERIFY-EFE-{run_id}-{step_id}",
                code="tx_verify_efe_check_missing",
                severity="warn",
                title="EFE check missing where ExpectedState exists",
                evidence=[{"step": step_id, "action": action, "expected_state": expected_state}],
                invariant="AGENTS.md:§2 - EFE es precondición constitucional",
                recommendation="Add verify.efe_check=true when expected_state is defined",
                affected_run_id=run_id,
                affected_step_id=step_id,
            )
        )

    # Check undo phase
    undo = tx.get("undo", {})
    has_rollback = undo.get("rollback_best_effort", False) if isinstance(undo, dict) else False

    if not has_rollback:
        findings.append(
            Finding(
                id=f"TX-UNDO-{run_id}-{step_id}",
                code="tx_undo_missing",
                severity="critical",
                title="Mutating step without UNDO capability",
                evidence=[{"step": step_id, "action": action, "undo": undo}],
                invariant="AGENTS.md:§2.1 - MUST automático UNDO si VERIFY ≠ OK",
                recommendation="Add undo.rollback_best_effort=true to enable automatic rollback",
                affected_run_id=run_id,
                affected_step_id=step_id,
            )
        )

    # Check if step is wrapped in TransactionalStep at all
    if not tx and is_mutating:
        findings.append(
            Finding(
                id=f"TX-NOT-WRAPPED-{run_id}-{step_id}",
                code="mutating_taskstep_without_transactional_wrapper",
                severity="error",
                title="Mutating step not wrapped in TransactionalStep",
                evidence=[{"step": step_id, "action": action, "classification_reason": reason}],
                invariant="AGENTS.md:§2.1 - Tareas T2+ deben seguir PREPARE→APPLY→VERIFY→UNDO",
                recommendation="Wrap step in TransactionalStep with prepare/verify/undo phases",
                affected_run_id=run_id,
                affected_step_id=step_id,
            )
        )

    return findings, counters


def _check_claims_in_result(
    result_data: Dict[str, Any],
    run_id: str,
) -> Tuple[List[Finding], Dict[str, int]]:
    """
    Check if claims in result have proper evidence or are degraded to hypothesis.

    According to AGENTS.md §X:
    - Claim sin evidence → MUST degradar a HIPÓTESIS + comandos de verificación
    """
    findings: List[Finding] = []
    counters = {
        "proof_claims_valid": 0,
        "proof_hypotheses_valid": 0,
    }

    # Check for claims array
    claims = result_data.get("claims", [])
    if not isinstance(claims, list):
        claims = []

    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue

        claim_id = claim.get("id", f"claim_{idx}")
        claim_type = claim.get("type", "")

        is_valid, missing, reason = _validate_claim_evidence(claim)

        if not is_valid:
            # Check if there's a hypothesis fallback
            hypothesis = result_data.get("hypothesis", "")
            verification_commands = result_data.get("verification_commands", [])

            if not hypothesis and not verification_commands:
                findings.append(
                    Finding(
                        id=f"CLAIM-NO-EVIDENCE-{run_id}-{claim_id}",
                        code="proof_claim_missing_minimum_evidence",
                        severity="warn",
                        title="Claim without EvidenceRef and no degradation to hypothesis",
                        evidence=[
                            {
                                "claim": claim,
                                "missing_evidence": missing,
                                "validation_reason": reason,
                            }
                        ],
                        invariant="AGENTS.md:§X.2 - Claim sin evidence → degradar a HIPÓTESIS",
                        recommendation="Either add required evidence_refs or degrade to hypothesis with verification_commands",
                        affected_run_id=run_id,
                    )
                )
        else:
            counters["proof_claims_valid"] += 1

    # Check for hypothesis without verification_commands
    hypothesis = result_data.get("hypothesis", "")
    verification_commands = result_data.get("verification_commands", [])

    if hypothesis and not verification_commands:
        findings.append(
            Finding(
                id=f"HYPOTHESIS-NO-CMDS-{run_id}",
                code="proof_hypothesis_missing_commands",
                severity="warn",
                title="Hypothesis without verification commands",
                evidence=[{"hypothesis": hypothesis[:200]}],
                invariant="AGENTS.md:§X.2 - HIPÓTESIS debe incluir 1-3 comandos de verificación",
                recommendation="Add 1-3 verification_commands to the hypothesis",
                affected_run_id=run_id,
            )
        )

    # Check degraded hypothesis includes explicit reason (anti-optimism vocabulary)
    if hypothesis and verification_commands:
        hyp_text = str(hypothesis).strip()
        reason_markers = ("falta", "missing", "because", "porque", "evidence", "incompleta", "incomplete", ":")
        has_reason = bool(hyp_text) and any(m in hyp_text.lower() for m in reason_markers)
        if not has_reason:
            findings.append(
                Finding(
                    id=f"HYPOTHESIS-NO-REASON-{run_id}",
                    code="proof_hypothesis_missing_reason",
                    severity="warn",
                    title="Hypothesis degradation missing explicit reason",
                    evidence=[{"hypothesis": hyp_text[:200], "verification_commands": verification_commands[:3]}],
                    invariant="AGENTS.md:§X.2 - Sin proof => HIPÓTESIS con razón + comandos de verificación",
                    recommendation="Include explicit reason in hypothesis (e.g. 'HIPÓTESIS (evidence incompleta): ...')",
                    affected_run_id=run_id,
                )
            )
        else:
            counters["proof_hypotheses_valid"] += 1

    return findings, counters


def _audit_single_run(run_dir: Path) -> Tuple[List[Finding], Dict[str, Any]]:
    """
    Audit a single run directory.

    Returns:
        (findings, meta) where meta contains info about files read
    """
    findings = []
    meta: Dict[str, Any] = {
        "run_id": run_dir.name,
        "files_read": {},
        "steps_count": 0,
        "claims_count": 0,
        "transactional_wrappers_seen": 0,
        "proof_claims_valid": 0,
        "proof_hypotheses_valid": 0,
    }

    run_id = run_dir.name

    # Read plan.json
    plan_path = run_dir / "plan.json"
    plan_data, plan_err = _safe_read_json(plan_path)
    meta["files_read"]["plan.json"] = {
        "exists": plan_data is not None,
        "error": plan_err,
    }

    if plan_data:
        # Check steps for transactional integrity
        steps = plan_data.get("steps", [])
        if not isinstance(steps, list):
            steps = []

        meta["steps_count"] = len(steps)

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            step_findings, step_counters = _check_step_transactional_integrity(step, run_id, idx)
            meta["transactional_wrappers_seen"] += int(step_counters.get("transactional_wrappers_seen", 0))
            findings.extend(step_findings)

    # Read result.json
    result_path = run_dir / "result.json"
    result_data, result_err = _safe_read_json(result_path)
    meta["files_read"]["result.json"] = {
        "exists": result_data is not None,
        "error": result_err,
    }

    if result_data:
        # Check claims for evidence
        claims_findings, claim_counters = _check_claims_in_result(result_data, run_id)
        findings.extend(claims_findings)
        meta["proof_claims_valid"] += int(claim_counters.get("proof_claims_valid", 0))
        meta["proof_hypotheses_valid"] += int(claim_counters.get("proof_hypotheses_valid", 0))

        claims = result_data.get("claims", [])
        meta["claims_count"] = len(claims) if isinstance(claims, list) else 0

    # Read verification.json (optional)
    verification_path = run_dir / "verification.json"
    verification_data, verification_err = _safe_read_json(verification_path)
    meta["files_read"]["verification.json"] = {
        "exists": verification_data is not None,
        "error": verification_err,
    }

    # Read audit_log.json (optional)
    audit_log_path = run_dir / "audit_log.json"
    audit_log_data, audit_log_err = _safe_read_json(audit_log_path)
    meta["files_read"]["audit_log.json"] = {
        "exists": audit_log_data is not None,
        "error": audit_log_err,
    }

    primary_exists = any(
        bool(meta["files_read"].get(name, {}).get("exists"))
        for name in ("plan.json", "result.json", "verification.json", "audit_log.json")
    )
    if not primary_exists:
        findings.append(
            Finding(
                id=f"RUN-NO-PRIMARY-ARTIFACTS-{run_id}",
                code="run_missing_primary_artifacts",
                severity="error",
                title="Run missing primary audit artifacts (plan/result/verification/audit_log)",
                evidence=[{"run_dir": str(run_dir), "files_read": meta["files_read"]}],
                invariant="PSEUDOCODE_MAP/02 - runs/<run_id>/ should persist plan/result/verification/audit_log",
                recommendation="Re-run broker flow or point audit to a run with canonical artifacts.",
                affected_run_id=run_id,
            )
        )

    return findings, meta


def _discover_runs(root_dir: Path, run_id: Optional[str] = None, last: int = 1) -> List[Path]:
    """
    Discover run directories to audit.

    Args:
        root_dir: Root directory containing runs/
        run_id: Specific run ID to audit (optional)
        last: Number of most recent runs to audit if run_id not specified

    Returns:
        List of run directory paths
    """
    runs_dir = root_dir / "runs"

    if not runs_dir.exists():
        return []

    if run_id:
        specific_run = runs_dir / run_id
        if specific_run.exists() and specific_run.is_dir():
            return [specific_run]
        return []

    # Get all run directories sorted by modification time (most recent first)
    # Skip symlinks and hidden directories to avoid Windows access issues
    run_dirs = []
    try:
        for d in runs_dir.iterdir():
            try:
                # Skip symlinks (like _latest_started) and hidden dirs
                if d.is_symlink():
                    continue
                if not d.is_dir():
                    continue
                if d.name.startswith("."):
                    continue
                run_dirs.append(d)
            except (OSError, PermissionError):
                # Skip entries we can't access
                continue
    except (OSError, PermissionError):
        return []

    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    return run_dirs[:last]


def _write_artifacts(
    root_dir: Path,
    result: AuditResult,
    ts: float,
) -> Tuple[Path, Path, Path]:
    """
    Write audit artifacts.

    Returns:
        (audit_json_path, audit_md_path, receipt_path)
    """
    ts_label = _ts_label(ts)

    # Audit JSON
    audit_dir = root_dir / "artifacts" / "audit" / f"tx_{ts_label}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_json_path = audit_dir / "tx_integrity_audit.json"
    audit_json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Audit MD (brief summary)
    audit_md_path = audit_dir / "tx_integrity_audit.md"
    md_lines = [
        f"# Transactional Integrity Audit",
        f"",
        f"- **Schema:** {result.schema}",
        f"- **Timestamp:** {result.timestamp_iso}",
        f"- **OK:** {result.ok}",
        f"- **Runs Audited:** {result.runs_audited}",
        f"- **Runs with Findings:** {result.runs_with_findings}",
        f"",
        f"## Summary by Severity",
        f"",
    ]
    for sev in ["critical", "error", "warn", "info"]:
        count = result.summary.get(sev, 0)
        md_lines.append(f"- **{sev.upper()}:** {count}")
    for key in ["transactional_wrappers_seen", "proof_claims_valid", "proof_hypotheses_valid"]:
        md_lines.append(f"- **{key}:** {result.summary.get(key, 0)}")

    if result.findings:
        md_lines.extend(
            [
                f"",
                f"## Findings ({len(result.findings)})",
                f"",
            ]
        )
        for f in result.findings[:20]:  # Limit to first 20
            md_lines.append(f"### [{f.severity.upper()}] {f.title}")
            md_lines.append(f"- **ID:** {f.id}")
            md_lines.append(f"- **Run:** {f.affected_run_id}")
            if f.affected_step_id:
                md_lines.append(f"- **Step:** {f.affected_step_id}")
            md_lines.append(f"- **Invariant:** {f.invariant}")
            md_lines.append(f"- **Recommendation:** {f.recommendation}")
            md_lines.append("")

    audit_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Receipt (append-only)
    receipts_dir = root_dir / "artifacts" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"tx_integrity_audit_{ts_label}.json"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "timestamp": ts,
        "timestamp_iso": _iso_utc(ts),
        "ok": result.ok,
        "runs_audited": result.runs_audited,
        "findings_count": len(result.findings),
        "summary": result.summary,
        "audit_path": str(audit_json_path),
        "read_only": True,
        "live_probes_invoked": False,
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return audit_json_path, audit_md_path, receipt_path


def run_tx_integrity_audit(
    root_dir: Path,
    *,
    run_id: Optional[str] = None,
    last: int = 1,
) -> Dict[str, Any]:
    """
    Run the Transactional Integrity Audit.

    This is a READ-ONLY audit that:
    1) Verifies that mutating steps follow TransactionalStep pattern
    2) Verifies that claims have proper evidence or are degraded to hypothesis

    Args:
        root_dir: Root directory of the AJAX repo
        run_id: Specific run ID to audit (optional)
        last: Number of most recent runs to audit if run_id not specified

    Returns:
        Dict with audit result including:
        - ok: True if no critical/error findings
        - schema: Schema identifier
        - findings: List of findings
        - artifact_path: Path to the audit JSON
        - receipt_path: Path to the receipt
    """
    ts = _now_ts()

    # Discover runs
    run_dirs = _discover_runs(root_dir, run_id=run_id, last=last)

    all_findings: List[Finding] = []
    runs_meta: List[Dict[str, Any]] = []
    runs_with_findings = 0

    if not run_dirs:
        # No runs found - fail-closed for audit invocation contract.
        all_findings.append(
            Finding(
                id="NO-RUNS-FOUND",
                code="no_runs_found",
                severity="error",
                title="No runs found to audit",
                evidence=[{"root_dir": str(root_dir), "run_id": run_id, "last": last}],
                invariant="PSEUDOCODE_MAP/02 - runs/<run_id>/ contains run artifacts",
                recommendation="Ensure runs have been executed, or specify a valid run_id",
                affected_run_id="",
            )
        )
    else:
        for run_dir in run_dirs:
            findings, meta = _audit_single_run(run_dir)
            runs_meta.append(meta)
            if findings:
                runs_with_findings += 1
            all_findings.extend(findings)

    # Compute summary
    summary = {
        "critical": sum(1 for f in all_findings if f.severity == "critical"),
        "error": sum(1 for f in all_findings if f.severity == "error"),
        "warn": sum(1 for f in all_findings if f.severity == "warn"),
        "info": sum(1 for f in all_findings if f.severity == "info"),
        "transactional_wrappers_seen": sum(int(m.get("transactional_wrappers_seen", 0)) for m in runs_meta),
        "proof_claims_valid": sum(int(m.get("proof_claims_valid", 0)) for m in runs_meta),
        "proof_hypotheses_valid": sum(int(m.get("proof_hypotheses_valid", 0)) for m in runs_meta),
    }

    # Determine OK
    # OK if no critical/error findings, and fail-closed for proof claims without minimum evidence.
    proof_fail_closed = any(f.code == "proof_claim_missing_minimum_evidence" for f in all_findings)
    ok = summary["critical"] == 0 and summary["error"] == 0 and not proof_fail_closed

    result = AuditResult(
        ok=ok,
        schema=AUDIT_SCHEMA,
        timestamp=ts,
        timestamp_iso=_iso_utc(ts),
        runs_audited=len(run_dirs),
        runs_with_findings=runs_with_findings,
        findings=all_findings,
        summary=summary,
        meta={
            "root_dir": str(root_dir),
            "run_id": run_id,
            "last": last,
            "runs": runs_meta,
        },
    )

    # Write artifacts
    audit_path, md_path, receipt_path = _write_artifacts(root_dir, result, ts)

    # Return result with paths
    result_dict = result.to_dict()
    result_dict["artifact_path"] = str(audit_path)
    result_dict["artifact_md_path"] = str(md_path)
    # Alias for newer callers/tests that expect this key.
    result_dict["summary_md_path"] = str(md_path)
    result_dict["receipt_path"] = str(receipt_path)
    result_dict["paths_written"] = [str(audit_path), str(md_path), str(receipt_path)]

    return result_dict


__all__ = [
    "run_tx_integrity_audit",
    "AuditResult",
    "Finding",
    "AUDIT_SCHEMA",
    "RECEIPT_SCHEMA",
]
