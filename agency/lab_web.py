from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agency.provider_policy import env_rail

try:
    from ddgs import DDGS  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DDGS = None  # type: ignore
try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore


DEFAULT_MAX_RESULTS = 8


def _slugify(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not cleaned:
        cleaned = "topic"
    return cleaned[:max_len]


def _timelimit_from_days(days: Optional[int]) -> Optional[str]:
    if days is None:
        return None
    try:
        d = int(days)
    except Exception:
        return None
    if d <= 0:
        return None
    if d <= 1:
        return "d"
    if d <= 7:
        return "w"
    if d <= 31:
        return "m"
    return "y"


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc or ""
    except Exception:
        return ""


def _search_ddg(query: str, *, max_results: int, timelimit: Optional[str]) -> List[Dict[str, Any]]:
    if DDGS is None:
        raise RuntimeError("ddgs_not_available")
    results: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            max_results=max_results,
            timelimit=timelimit,
            region="us-en",
            safesearch="moderate",
        )
    if results:
        return results
    with DDGS() as ddgs:
        results = ddgs.text(
            query,
            max_results=max_results,
            timelimit=timelimit,
            region="us-en",
            safesearch="moderate",
            backend="lite",
        )
    return results or []


def _is_timeout_error(exc: BaseException) -> bool:
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return True
    name = exc.__class__.__name__.lower()
    return "timeout" in name


def _normalize_allowlist(domains: Optional[List[str]]) -> List[str]:
    if not domains:
        return []
    cleaned = []
    for item in domains:
        if not item:
            continue
        dom = str(item).strip().lower()
        if dom.startswith("."):
            dom = dom[1:]
        if dom and dom not in cleaned:
            cleaned.append(dom)
    return cleaned


def _is_allowed_domain(domain: str, allowlist: List[str]) -> bool:
    if not domain:
        return False
    dom = domain.lower()
    for allowed in allowlist:
        if dom == allowed or dom.endswith("." + allowed):
            return True
    return False


def _derive_hypotheses(topic: str, sources: List[Dict[str, Any]]) -> List[str]:
    hypotheses: List[str] = []
    lower = topic.lower()
    if any(tok in lower for tok in ("rate", "limit", "quota", "cuota")):
        hypotheses.append("Rate limits are documented per provider; verify tiers and limits in official docs.")
    if any(tok in lower for tok in ("model", "modelo", "models")):
        hypotheses.append("Model list endpoints are documented in provider API references.")
    if any(tok in lower for tok in ("deprec", "cambio", "changes", "changelog")):
        hypotheses.append("Deprecations are published in release notes or changelogs per provider.")
    if not hypotheses:
        if sources:
            domains = sorted({s.get("domain") for s in sources if s.get("domain")})
            if domains:
                hypotheses.append(f"Relevant official documentation likely exists on: {', '.join(domains[:4])}.")
        if not hypotheses:
            hypotheses.append("Relevant official documentation likely exists; prioritize provider docs.")
    return hypotheses


def _suggest_actions(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for src in sources[:6]:
        title = src.get("title")
        url = src.get("url")
        if not url:
            continue
        actions.append(
            {
                "type": "review_source",
                "title": title,
                "url": url,
                "domain": src.get("domain"),
            }
        )
    return actions


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# LAB-WEB Report\n",
        f"- Topic: {payload.get('topic')}\n",
        f"- Query: {payload.get('query')}\n",
        f"- Results: {payload.get('results_count')}\n",
        f"- Strict: {payload.get('strict')}\n",
        f"- Days: {payload.get('days')}\n",
        f"- Generated (UTC): {payload.get('generated_at')}\n",
    ]
    if payload.get("error"):
        lines.append(f"- Error: {payload.get('error')}\n")
    lines.append("\n## Sources\n")
    sources = payload.get("sources") or []
    if sources:
        for idx, src in enumerate(sources, 1):
            title = src.get("title") or "untitled"
            url = src.get("url") or ""
            domain = src.get("domain") or ""
            lines.append(f"{idx}. {title} ({domain}) - {url}\n")
    else:
        lines.append("- No sources found.\n")
    lines.append("\n## Hypotheses\n")
    for item in payload.get("hypotheses") or []:
        lines.append(f"- {item}\n")
    lines.append("\n## Suggested actions (LAB only)\n")
    for item in payload.get("suggested_actions") or []:
        title = item.get("title") or "source"
        url = item.get("url") or ""
        lines.append(f"- review_source: {title} - {url}\n")
    lines.append("\n## Notes\n- No LLM used. LAB-only web discovery.\n")
    path.write_text("".join(lines), encoding="utf-8")


def run_lab_web(
    topic: str,
    *,
    days: Optional[int] = None,
    strict: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
    allowlist_domains: Optional[List[str]] = None,
    strict_sources: bool = False,
    root_dir: Optional[Path] = None,
    allow_prod: bool = False,
) -> Dict[str, Any]:
    if not topic or not str(topic).strip():
        raise ValueError("topic_required")
    rail = env_rail()
    if rail == "prod" and not allow_prod:
        raise RuntimeError("lab_web_blocked_in_prod")

    base = (root_dir or Path(__file__).resolve().parents[1]) / "artifacts" / "lab_web"
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    slug = _slugify(topic)
    run_dir = base / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    timelimit = _timelimit_from_days(days)
    query = str(topic).strip()
    try:
        raw_results = _search_ddg(query, max_results=max_results, timelimit=timelimit)
    except Exception as exc:
        if _is_timeout_error(exc):
            ok = False
            warnings = ["search_timeout"]
            allowlist = _normalize_allowlist(allowlist_domains)
            payload = {
                "schema": "ajax.lab_web.v1",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rail": rail,
                "topic": topic,
                "query": query,
                "days": days,
                "strict": bool(strict),
                "results_count": 0,
                "sources": [],
                "allowlist_domains": allowlist or None,
                "strict_sources": bool(strict_sources),
                "hypotheses": [],
                "suggested_actions": [],
                "warnings": warnings,
                "error": "search_timeout",
                "ok": bool(ok),
            }
            sources_path = run_dir / "sources.json"
            sources_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            report_path = run_dir / "report.md"
            _write_report(report_path, payload)
            return {
                "ok": bool(ok),
                "topic": topic,
                "query": query,
                "days": days,
                "strict": bool(strict),
                "results_count": 0,
                "run_dir": str(run_dir),
                "report_path": str(report_path),
                "sources_path": str(sources_path),
                "hypotheses": [],
                "suggested_actions": [],
                "allowlist_domains": allowlist or None,
                "strict_sources": bool(strict_sources),
                "warnings": warnings,
                "error": "search_timeout",
                "rail": rail,
            }
        raise

    sources: List[Dict[str, Any]] = []
    for item in raw_results or []:
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = str(item.get("body") or item.get("snippet") or "").strip()
        if not title and not url:
            continue
        sources.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": _extract_domain(url) if url else "",
            }
        )

    allowlist = _normalize_allowlist(allowlist_domains)
    if allowlist:
        sources = [s for s in sources if _is_allowed_domain(str(s.get("domain") or ""), allowlist)]

    hypotheses = _derive_hypotheses(topic, sources)
    suggested_actions = _suggest_actions(sources)
    ok = bool(sources) if (strict or strict_sources) else True
    warnings = []
    if not sources:
        warnings.append("no_sources_found")

    payload = {
        "schema": "ajax.lab_web.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rail": rail,
        "topic": topic,
        "query": query,
        "days": days,
        "strict": bool(strict),
        "results_count": len(sources),
        "sources": sources,
        "allowlist_domains": allowlist or None,
        "strict_sources": bool(strict_sources),
        "hypotheses": hypotheses,
        "suggested_actions": suggested_actions,
        "warnings": warnings,
        "ok": bool(ok),
    }

    sources_path = run_dir / "sources.json"
    sources_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = run_dir / "report.md"
    _write_report(report_path, payload)

    payload_summary = {
        "ok": bool(ok),
        "topic": topic,
        "query": query,
        "days": days,
        "strict": bool(strict),
        "results_count": len(sources),
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "sources_path": str(sources_path),
        "hypotheses": hypotheses,
        "suggested_actions": suggested_actions,
        "allowlist_domains": allowlist or None,
        "strict_sources": bool(strict_sources),
        "warnings": warnings,
        "rail": rail,
    }
    return payload_summary
