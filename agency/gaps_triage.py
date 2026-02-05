from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SPL_SCORES = {
    "LOW": 0,
    "MED": 1,
    "MEDIUM": 1,
    "HIGH": 2,
}

W_SPL = 2.0
W_RECENCY = 1.0
W_OCCURRENCES = 0.5

PROD_ENV_KEYS = ("AJAX_RAIL", "AJAX_ENV", "AJAX_MODE")
PROD_ENV_VALUES = {"prod", "production", "live"}


@dataclass
class FeedbackEntry:
    envelope_id: str
    created_ts: float
    created_utc: str
    spl: str
    spl_score: int
    learning_signals: List[str]
    evidence_refs: List[str]
    path: Path


@dataclass
class GapEntry:
    gap_id: str
    envelope_id: Optional[str]
    summary: str
    occurrences: int
    created_ts: float
    created_utc: str
    path: Path
    raw: Dict[str, Any]


class TriageError(RuntimeError):
    pass


def _utc_now(now: Optional[datetime] = None) -> str:
    value = now or datetime.now(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(now: Optional[datetime] = None) -> str:
    value = now or datetime.now(timezone.utc)
    return value.strftime("%Y%m%d-%H%M")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = value
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except Exception:
        return None


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def env_indicates_prod(env: Dict[str, str]) -> bool:
    for key in PROD_ENV_KEYS:
        raw = env.get(key, "")
        if raw and raw.strip().lower() in PROD_ENV_VALUES:
            return True
    return False


def _collect_feedback(runs_dir: Path) -> List[FeedbackEntry]:
    entries: List[FeedbackEntry] = []
    for path in sorted(runs_dir.rglob("epistemic_feedback.json")):
        try:
            data = _read_json(path)
        except Exception:
            continue
        envelope_id = data.get("envelope_id")
        if not envelope_id:
            continue
        created = _parse_utc(data.get("created_utc"))
        created_ts = created.timestamp() if created else _file_mtime(path)
        created_utc = created.strftime("%Y-%m-%dT%H:%M:%SZ") if created else datetime.fromtimestamp(
            created_ts, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        spl_raw = str(data.get("spl", "LOW")).upper()
        spl_score = SPL_SCORES.get(spl_raw, 0)
        learning_signals = list(data.get("learning_signals") or [])
        evidence_refs = list(data.get("evidence_refs") or [])
        entries.append(
            FeedbackEntry(
                envelope_id=envelope_id,
                created_ts=created_ts,
                created_utc=created_utc,
                spl=spl_raw,
                spl_score=spl_score,
                learning_signals=learning_signals,
                evidence_refs=evidence_refs,
                path=path,
            )
        )
    return entries


def _collect_gaps(gaps_dir: Path) -> List[GapEntry]:
    gaps: List[GapEntry] = []
    if not gaps_dir.exists():
        return gaps
    for path in sorted(gaps_dir.glob("*.json")):
        try:
            data = _read_json(path)
        except Exception:
            continue
        gap_id = data.get("gap_id") or path.stem
        envelope_id = data.get("envelope_id")
        summary = data.get("summary") or data.get("title") or ""
        occurrences = _safe_int(
            data.get("occurrences") or data.get("occurrence_count") or 1,
            default=1,
        )
        created = _parse_utc(data.get("created_utc"))
        created_ts = created.timestamp() if created else _file_mtime(path)
        created_utc = created.strftime("%Y-%m-%dT%H:%M:%SZ") if created else datetime.fromtimestamp(
            created_ts, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        gaps.append(
            GapEntry(
                gap_id=gap_id,
                envelope_id=envelope_id,
                summary=summary,
                occurrences=occurrences,
                created_ts=created_ts,
                created_utc=created_utc,
                path=path,
                raw=data,
            )
        )
    return gaps


def _best_feedback(entries: Iterable[FeedbackEntry]) -> Optional[FeedbackEntry]:
    best = None
    for entry in entries:
        if best is None:
            best = entry
            continue
        if entry.spl_score > best.spl_score:
            best = entry
            continue
        if entry.spl_score == best.spl_score and entry.created_ts > best.created_ts:
            best = entry
            continue
        if entry.spl_score == best.spl_score and entry.created_ts == best.created_ts:
            if str(entry.path) < str(best.path):
                best = entry
    return best


def _learning_histogram(entries: Iterable[FeedbackEntry]) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    for entry in entries:
        for signal in entry.learning_signals:
            counter[str(signal)] += 1
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [{"signal": sig, "count": count} for sig, count in items]


def _derive_invariants(signals: List[str]) -> List[str]:
    invariants: List[str] = []
    if "probe_failed" in signals:
        invariants.append("execution_trace.observations.probe_ok == true")
    if "latency_too_high" in signals:
        invariants.append("execution_trace.observations.latency_ms <= 100")
    if not invariants:
        invariants.append("state_delta.verification.ok == true")
    while len(invariants) < 2:
        invariants.append("execution_trace.status != 'blocked'")
    return invariants


def _build_probes(gap: GapEntry, signals: List[str], evidence_refs: List[str]) -> List[Dict[str, Any]]:
    invariants = _derive_invariants(signals)
    base_refs = evidence_refs[:2]
    probes = []
    templates = [
        ("reproduce", "Reproduce the failure under a controlled baseline."),
        ("isolate", "Isolate the strongest signal with a single-variable change."),
        ("control", "Run a known-good control to confirm signal boundaries."),
    ]
    for idx, (tag, summary) in enumerate(templates, start=1):
        probe_id = f"{gap.gap_id}_probe_{idx}"
        hypothesis = f"{summary}"
        mini_efe = [
            invariants[0],
            invariants[1] if len(invariants) > 1 else invariants[0],
        ]
        probes.append(
            {
                "probe_id": probe_id,
                "hypothesis": hypothesis,
                "mini_efe": mini_efe,
                "suggested_sensors": ["execution_trace", "state_delta"],
                "uses_evidence_refs": list(base_refs),
            }
        )
    return probes


def triage_gaps(
    root_dir: Path,
    top_n: int = 5,
    rail: str = "lab",
    env: Optional[Dict[str, str]] = None,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], Path, Path]:
    rail_value = (rail or "").strip().lower()
    if rail_value == "prod":
        raise TriageError("LAB-only: rail=prod is not allowed for gaps triage")
    env_data = env or dict(os.environ)
    if env_indicates_prod(env_data):
        raise TriageError("LAB-only: environment indicates PROD")

    gaps_dir = root_dir / "artifacts" / "capability_gaps" / "open"
    runs_dir = root_dir / "experiments" / "runs"

    gaps = _collect_gaps(gaps_dir)
    feedback = _collect_feedback(runs_dir)

    feedback_by_envelope: Dict[str, List[FeedbackEntry]] = {}
    for entry in feedback:
        feedback_by_envelope.setdefault(entry.envelope_id, []).append(entry)

    recency_candidates: List[float] = []
    gap_rows: List[Dict[str, Any]] = []

    for gap in gaps:
        related = feedback_by_envelope.get(gap.envelope_id or "", [])
        best = _best_feedback(related)
        if best:
            recency_ts = best.created_ts
            last_seen_utc = best.created_utc
            spl_score = best.spl_score
            spl_label = best.spl
            evidence_refs = list(best.evidence_refs)
        else:
            recency_ts = gap.created_ts
            last_seen_utc = gap.created_utc
            spl_score = 0
            spl_label = "LOW"
            evidence_refs = list(gap.raw.get("evidence_refs") or [])
        recency_candidates.append(recency_ts)
        gap_rows.append(
            {
                "gap": gap,
                "related": related,
                "last_seen_utc": last_seen_utc,
                "recency_ts": recency_ts,
                "spl_score": spl_score,
                "spl_label": spl_label,
                "evidence_refs": evidence_refs,
            }
        )

    max_ts = max(recency_candidates) if recency_candidates else 0.0
    triaged: List[Dict[str, Any]] = []

    for row in gap_rows:
        gap = row["gap"]
        recency_score = (row["recency_ts"] / max_ts) if max_ts > 0 else 0.0
        score = (W_SPL * row["spl_score"]) + (W_RECENCY * recency_score) + (W_OCCURRENCES * gap.occurrences)
        learning_hist = _learning_histogram(row["related"])
        signals = [item["signal"] for item in learning_hist]
        probes = _build_probes(gap, signals, row["evidence_refs"])
        triaged.append(
            {
                "gap_id": gap.gap_id,
                "summary": gap.summary,
                "occurrences": gap.occurrences,
                "last_seen_utc": row["last_seen_utc"],
                "spl": row["spl_label"],
                "score": round(score, 6),
                "learning_signals": learning_hist,
                "evidence_refs": row["evidence_refs"],
                "probes": probes,
            }
        )

    triaged.sort(key=lambda item: (-item["score"], item["last_seen_utc"], item["gap_id"]))
    top_n = max(0, int(top_n))
    triaged = triaged[:top_n] if top_n else []

    out_dir = root_dir / "artifacts" / "gaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(now)

    payload = {
        "schema_version": "0.1",
        "created_utc": _utc_now(now),
        "rail": "lab",
        "weights": {
            "spl": W_SPL,
            "recency": W_RECENCY,
            "occurrences": W_OCCURRENCES,
        },
        "sources": {
            "gaps_dir": str(gaps_dir),
            "runs_dir": str(runs_dir),
        },
        "counts": {
            "gaps": len(gaps),
            "feedback": len(feedback),
            "triaged": len(triaged),
        },
        "gaps": triaged,
    }

    json_path = out_dir / f"triage_{stamp}.json"
    md_path = out_dir / f"triage_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(_format_md(payload), encoding="utf-8")

    return payload, json_path, md_path


def _format_md(payload: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Research Triage (LAB-only)")
    lines.append("")
    lines.append(f"Generated: {payload.get('created_utc')}")
    lines.append("")
    weights = payload.get("weights", {})
    lines.append(f"Weights: spl={weights.get('spl')} recency={weights.get('recency')} occurrences={weights.get('occurrences')}")
    lines.append("")

    gaps = payload.get("gaps", [])
    if not gaps:
        lines.append("No gaps found.")
        return "\n".join(lines)

    lines.append("| Rank | Gap ID | Score | SPL | Occurrences | Last Seen | Summary |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for idx, gap in enumerate(gaps, start=1):
        summary = gap.get("summary", "")
        lines.append(
            f"| {idx} | {gap.get('gap_id')} | {gap.get('score')} | {gap.get('spl')} | {gap.get('occurrences')} | {gap.get('last_seen_utc')} | {summary} |"
        )

    for gap in gaps:
        lines.append("")
        lines.append(f"## Gap {gap.get('gap_id')}")
        lines.append("")
        lines.append(f"Summary: {gap.get('summary', '')}")
        lines.append("")
        signals = gap.get("learning_signals") or []
        if signals:
            formatted = ", ".join([f"{item['signal']} ({item['count']})" for item in signals])
        else:
            formatted = "(none)"
        lines.append(f"Learning signals: {formatted}")
        lines.append("")
        evidence_refs = gap.get("evidence_refs") or []
        lines.append("Evidence refs:")
        if evidence_refs:
            for ref in evidence_refs:
                lines.append(f"- {ref}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("Suggested probes:")
        for probe in gap.get("probes", []):
            lines.append(f"- {probe.get('probe_id')}: {probe.get('hypothesis')}")
            for inv in probe.get("mini_efe", []):
                lines.append(f"  - {inv}")
            sensors = ", ".join(probe.get("suggested_sensors", []))
            lines.append(f"  - sensors: {sensors}")
            uses_refs = probe.get("uses_evidence_refs", [])
            if uses_refs:
                lines.append(f"  - uses: {', '.join(uses_refs)}")

    return "\n".join(lines)
