from __future__ import annotations

"""
The Scout: Órgano de Investigación de AJAX.

Responsabilidad:
- Dado un topic + context, buscar soluciones externas (GitHub/Web)
  usando wrappers CLI (Qwen/Gemini/lo que tengas).
- Filtrar resultados por dominios permitidos y señales de calidad.
- Evaluar riesgos básicos (inyección, código peligroso, proyectos abandonados).
- Generar un informe Markdown en artifacts/scout_sandbox/research/YYYY-MM-DD_topic/.

Nota: Este módulo es un esqueleto. Las llamadas reales a CLIs/HTTP
deben implementarse en los TODOs según las herramientas de tu entorno.
"""

import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

log = logging.getLogger("ajax.scout")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)


# ---------- Sandbox Helpers ----------


def get_scout_sandbox_dir(root: Optional[Path] = None) -> Path:
    base = root or Path(__file__).resolve().parents[1]
    path = Path(base) / "artifacts" / "scout_sandbox"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_scout_path(relative_path: Path, root: Optional[Path] = None) -> Path:
    sandbox = get_scout_sandbox_dir(root)
    rel = relative_path
    if rel.is_absolute():
        raise ValueError("Absolute paths not allowed for scout sandbox writes")
    if ".." in rel.parts:
        raise ValueError("Parent traversal not allowed in scout sandbox writes")
    target = (sandbox / rel).resolve()
    sandbox_resolved = sandbox.resolve()
    if sandbox_resolved not in target.parents and target != sandbox_resolved:
        raise ValueError("Target escapes scout sandbox")
    return target


def write_scout_file(relative_path: Path | str, content: str | bytes, *, root: Optional[Path] = None) -> Path:
    rel_path = Path(relative_path)
    target = _resolve_scout_path(rel_path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")
    return target


# ---------- Datos básicos ----------


@dataclass
class ScoutConfig:
    root_dir: Path
    research_dir: Path
    allowed_domains: List[str]
    min_stars: int = 100
    max_last_commit_age_days: int = 180  # ~6 meses

    @classmethod
    def default(cls, root: Optional[Path] = None) -> "ScoutConfig":
        root = root or Path(__file__).resolve().parents[1]
        sandbox = get_scout_sandbox_dir(root)
        research_dir = sandbox / "research"
        return cls(
            root_dir=root,
            research_dir=research_dir,
            allowed_domains=[
                "github.com",
                "pypi.org",
                "stackoverflow.com",
                "docs.python.org",
                "learn.microsoft.com",
                "developer.mozilla.org",
            ],
        )


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # p.ej. "gemini_cli", "qwen_cli"
    extra: Dict[str, Any]


@dataclass
class ProposedSkill:
    name: str
    description: str
    args: List[Dict[str, Any]]
    deps: List[str]
    source_file: Optional[str] = None


@dataclass
class RiskAssessment:
    domain_ok: bool
    stars_ok: bool
    recent_ok: bool
    code_risk_flags: List[str]
    overall_risk: str  # "low" | "medium" | "high"


@dataclass
class ResearchProposal:
    topic: str
    context: str
    selected: List[SearchResult]
    risks: Dict[str, RiskAssessment]
    suggestions: List[str]
    code_example: Optional[str]
    harvested_path: Optional[str]
    proposed_skills: List[ProposedSkill]


# ---------- Scout principal ----------


class Scout:
    """
    Órgano de Investigación de AJAX (The Scout).

    Uso típico:
        scout = Scout()
        report_path = scout.investigate("ocr rápido en Windows", context="Fallo al detectar botón en Spotify")
    """

    def __init__(self, config: Optional[ScoutConfig] = None) -> None:
        self.config = config or ScoutConfig.default()
        # Fuerza sandbox para todas las escrituras (descargas, informes, propuestas)
        self.sandbox_dir = get_scout_sandbox_dir(self.config.root_dir)
        self.config.research_dir = self.sandbox_dir / "research"
        self.config.research_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_sandbox_readme()
        log.info("Scout inicializado (sandbox=%s)", self.sandbox_dir)

    def _ensure_sandbox_readme(self) -> None:
        content = "\n".join(
            [
                "# Scout Sandbox",
                "",
                "- Todas las descargas y propuestas del Scout se guardan aquí (artifacts/scout_sandbox).",
                "- Las propuestas de código se escriben como archivos *.py.proposal dentro de proposals/.",
                "- Estos archivos **no se importan ni se ejecutan automáticamente**.",
                "- Flujo humano recomendado:",
                "  1) Revisar el archivo .py.proposal.",
                "  2) Revisar el security_report correspondiente.",
                "  3) Si se aprueba, renombrar manualmente a .py y mover al destino final.",
                "- Seguridad: los reportes se guardan como security_report_<nombre>.json.",
            ]
        )
        try:
            write_scout_file("SCOUT_SANDBOX.md", content, root=self.config.root_dir)
        except Exception:
            pass

    # --- Punto de entrada público ---

    def investigate(self, topic: str, context: str) -> Path:
        """
        Ejecuta el pipeline de investigación:
        - Search
        - Gatekeeper
        - Propuesta
        - Informe Markdown

        Devuelve la ruta del informe generado.
        """
        log.info("Scout.investigate: topic=%r", topic)

        # 1) Buscar (CLI wrappers)
        raw_results = self._search_external(topic, context)

        # 2) Filtrado por dominios permitidos
        filtered = [r for r in raw_results if self._is_domain_allowed(r.url)]
        if not filtered:
            log.warning("Sin resultados en dominios permitidos; usando todos los resultados.")
            filtered = raw_results

        # 3) Gatekeeper: evaluación de riesgos / calidad
        risks = {r.url: self._assess_risk(r) for r in filtered}

        # 4) Selección simple de candidatos (los que tienen riesgo bajo/medio)
        selected = [
            r for r in filtered if risks[r.url].overall_risk in ("low", "medium")
        ] or filtered[:3]

        harvested_path: Optional[Path] = None
        proposed_skills: List[ProposedSkill] = []

        if selected:
            top_repo = selected[0]
            try:
                harvested_path = self.harvest_repo(top_repo.url)
            except Exception as exc:
                log.warning("harvest_repo failed for %s: %s", top_repo.url, exc)
            if harvested_path:
                try:
                    proposed_skills = self.analyze_codebase(harvested_path)
                except Exception as exc:
                    log.warning("analyze_codebase failed: %s", exc)

        # 5) Generar propuesta (resumen + código ejemplo + análisis)
        proposal = self._generate_proposal(topic, context, selected, risks, harvested_path, proposed_skills)

        # 6) Escribir informe Markdown
        report_path = self._write_report_markdown(proposal)

        log.info("Informe de investigación generado en %s", report_path)
        return report_path

    # --- Propuestas y análisis de seguridad ---

    def _write_security_report(self, name: str, report: Dict[str, Any]) -> Path:
        safe = self._slugify(name)[:80] or "proposal"
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        return write_scout_file(f"security_report_{safe}.json", payload, root=self.config.root_dir)

    def _run_security_analysis(self, code: str, name: str, proposal_path: Path, gap_id: Optional[str] = None) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "analysis_mode": None,
            "flags": [],
            "errors": [],
            "proposal_path": str(proposal_path),
            "gap_id": gap_id,
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
        }
        bandit_path = shutil.which("bandit")
        if bandit_path:
            try:
                with tempfile.TemporaryDirectory() as tmpd:
                    tmp_file = Path(tmpd) / "candidate.py"
                    tmp_file.write_text(code, encoding="utf-8")
                    proc = subprocess.run(
                        [bandit_path, "-q", "-r", str(tmp_file), "-f", "json"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    report["analysis_mode"] = "bandit"
                    if proc.stdout:
                        try:
                            data = json.loads(proc.stdout)
                            report["bandit_raw"] = data
                            issues = data.get("results") or []
                            if issues:
                                report["flags"] = [
                                    f"{it.get('test_id') or ''}:{it.get('issue_text') or ''}" for it in issues
                                ]
                        except Exception as exc:
                            report["errors"].append(f"bandit_parse:{exc}")
                    if proc.returncode not in (0, 1):
                        report["errors"].append(f"bandit_exit_{proc.returncode}")
                    return report
            except Exception as exc:
                report["errors"].append(f"bandit_failed:{exc}")

        # Fallback heurístico
        report["analysis_mode"] = "heuristic"
        flags = self._scan_code_for_risks(code)
        if "subprocess" in code:
            flags.append("subprocess_usage")
        if "os.system" in code:
            flags.append("os_system")
        report["flags"] = sorted(set(flags))
        return report

    def _render_skill_stub(self, skill: ProposedSkill) -> str:
        args = [a.get("name") for a in (skill.args or []) if isinstance(a, dict) and a.get("name")]
        arg_list = ", ".join(args) if args else "input_data"
        description = skill.description or "Propuesta generada por Scout. Revisar antes de usar."
        fn_name = re.sub(r"[^A-Za-z0-9_]", "_", skill.name or "proposed_skill") or "proposed_skill"
        if fn_name[0].isdigit():
            fn_name = f"skill_{fn_name}"
        stub = textwrap.dedent(
            f'''
            """
            Proposal stub generated by Scout. Review and adapt before enabling.
            """

            def {fn_name}({arg_list}):
                """
                {description}
                """
                # TODO: implement de forma segura.
                raise NotImplementedError("Proposal stub – review and move to production code manually.")
            '''
        ).strip() + "\n"
        return stub

    def _write_skill_proposals(self, skills: List[ProposedSkill]) -> None:
        for skill in skills:
            try:
                code = self._render_skill_stub(skill)
                path = self.write_code_proposal(skill.name, code)
                skill.source_file = str(path)
            except Exception as exc:
                log.warning("No se pudo escribir propuesta para %s: %s", skill.name, exc)

    def write_code_proposal(self, name: str, code: str, gap_id: Optional[str] = None) -> Path:
        safe_name = self._slugify(name) or "proposal"
        filename = f"{safe_name}.py.proposal"
        code_out = code if code.endswith("\n") else code + "\n"
        proposal_path = write_scout_file(Path("proposals") / filename, code_out, root=self.config.root_dir)
        report = self._run_security_analysis(code_out, safe_name, proposal_path, gap_id=gap_id)
        try:
            self._write_security_report(safe_name, report)
        except Exception:
            pass
        return proposal_path

    # --- 1) Search Engine (wrappers CLI) ---

    def _read_token(self) -> str | None:
        tok = os.getenv("GITHUB_TOKEN")
        if tok:
            return tok.strip()
        cred = Path.home() / ".leann" / "creds" / "github_token"
        try:
            if cred.exists():
                return cred.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return None

    def _search_external(self, topic: str, context: str) -> List[SearchResult]:
        """
        Búsqueda real en GitHub usando la API (sorted by stars).
        """
        log.info("Scout._search_external: buscando '%s'", topic)
        token = self._read_token()
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "ajax-scout"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        query = f"{topic} in:name,description,readme language:python"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 10}
        try:
            resp = requests.get("https://api.github.com/search/repositories", headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("GitHub search failed: %s", exc)
            return []

        results: List[SearchResult] = []
        for item in data.get("items", []):
            full = item.get("full_name") or ""
            url = item.get("html_url") or f"https://github.com/{full}"
            results.append(
                SearchResult(
                    title=full,
                    url=url,
                    snippet=item.get("description") or "",
                    source="github_api",
                    extra={
                        "stars": item.get("stargazers_count"),
                        "last_commit_iso": item.get("pushed_at"),
                        "default_branch": item.get("default_branch"),
                        "topics": item.get("topics") or [],
                        "license": (item.get("license") or {}).get("spdx_id"),
                    },
                )
            )
        return results

    def _is_domain_allowed(self, url: str) -> bool:
        return any(domain in url for domain in self.config.allowed_domains)

    # --- Harvest (descarga ligera) ---

    def _shallow_clone(self, repo: str, branch: Optional[str], workdir: Path) -> str:
        candidates = []
        if branch:
            candidates.append(branch)
        candidates += ["main", "master"]
        seen = set()
        candidates = [b for b in candidates if (b not in seen and not seen.add(b))]

        if shutil.which("git"):
            last_err = None
            for b in candidates:
                try:
                    subprocess.run(["git", "clone", "--depth", "1", "--branch", b, repo, str(workdir)], check=True, timeout=45)
                    return b
                except Exception as exc:
                    last_err = exc
                    continue
            if last_err:
                raise last_err

        # Fallback: zip
        for b in candidates:
            try:
                url = repo.rstrip("/") + f"/archive/refs/heads/{b}.zip"
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with tempfile.TemporaryDirectory() as tmpz:
                    zpath = Path(tmpz) / "repo.zip"
                    zpath.write_bytes(resp.content)
                    shutil.unpack_archive(str(zpath), str(workdir))
                    roots = [p for p in workdir.iterdir() if p.is_dir()]
                    if roots:
                        tmp = workdir / "_root"
                        roots[0].rename(tmp)
                        for c in tmp.iterdir():
                            c.rename(workdir / c.name)
                        tmp.rmdir()
                    return b
            except Exception:
                continue
        raise RuntimeError("clone_failed")

    def _parse_repo_url(self, url: str) -> Tuple[str, str, str]:
        clean = url.strip()
        if clean.startswith("http"):
            m = re.match(r"https?://github.com/([^/]+)/([^/]+)", clean, flags=re.I)
            if m:
                return f"https://github.com/{m.group(1)}/{m.group(2)}", m.group(1), m.group(2)
        if "/" in clean:
            parts = clean.split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]
                base = f"https://github.com/{owner}/{repo}"
                return base, owner, repo
        raise ValueError(f"repo_url_invalid:{url}")

    def harvest_repo(self, url: str) -> Path:
        """
        Descarga/clona un repo de GitHub de forma ligera en artifacts/research/<today>_<slug>/repo
        """
        base, owner, repo = self._parse_repo_url(url)
        today = datetime.date.today().isoformat()
        slug = self._slugify(repo)[:40] or repo
        out_dir = self.config.research_dir / f"{today}_{slug}" / "repo"
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd) / "repo"
            used_branch = self._shallow_clone(base, None, tmp)
            # limpieza básica
            for bad in (".git", "node_modules", "dist", "build", ".venv", "__pycache__"):
                shutil.rmtree(tmp / bad, ignore_errors=True)
            # copiar
            for child in tmp.iterdir():
                target = out_dir / child.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        target.unlink()
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)

        meta = {"repo": base, "branch": used_branch}
        (out_dir.parent / "harvest_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_dir

    # --- Análisis de código (skills) ---

    def _read_py_snippets(self, root: Path, max_files: int = 5, max_chars_per_file: int = 2000) -> List[Tuple[str, str]]:
        candidates = []
        preferred = {"api.py", "client.py", "core.py", "main.py"}
        for p in root.rglob("*.py"):
            if any(seg in p.parts for seg in (".git", "tests", "venv", ".venv", "__pycache__", "node_modules")):
                continue
            candidates.append(p)
        candidates = sorted(candidates, key=lambda p: (0 if p.name in preferred else 1, -p.stat().st_size if p.exists() else 0))
        snippets: List[Tuple[str, str]] = []
        for p in candidates[:max_files]:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                snippets.append((str(p.relative_to(root)), content[:max_chars_per_file]))
            except Exception:
                continue
        return snippets

    def _load_provider_configs(self) -> Dict[str, Any]:
        if yaml is None:
            return {}
        path = self.config.root_dir / "config" / "model_providers.yaml"
        if not path.exists():
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _select_brain_provider(self, providers_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        for name, cfg in (providers_cfg.get("providers") or {}).items():
            roles = cfg.get("roles") or []
            if isinstance(roles, list) and "brain" in [r.lower() for r in roles]:
                return name, cfg
        # fallback: first provider
        items = list((providers_cfg.get("providers") or {}).items())
        if items:
            return items[0]
        raise RuntimeError("brain_provider_not_found")

    def _call_brain(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        providers_cfg = self._load_provider_configs()
        name, cfg = self._select_brain_provider(providers_cfg)
        kind = (cfg.get("kind") or "").lower()
        model = cfg.get("default_model") or cfg.get("model") or "gpt-4o-mini"
        timeout = int(cfg.get("timeout_seconds") or 30)

        if kind == "codex_cli_jsonl":
            cmd = cfg.get("command") or ["codex", "exec", "--model", model, "--json"]
            cmd = [str(c).replace("{model}", model) for c in cmd]
            full_prompt = system_prompt.rstrip() + "\n\n" + user_prompt
            proc = subprocess.run(cmd, input=full_prompt, text=True, capture_output=True, timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError(f"codex_cli_failed:{proc.stderr[:200]}")
            content = proc.stdout.strip()
            try:
                return json.loads(content)
            except Exception:
                raise RuntimeError("codex_cli_invalid_json")

        base_url = (cfg.get("base_url") or "").rstrip("/")
        api_key_env = cfg.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        if not base_url:
            raise RuntimeError("brain_base_url_missing")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"brain_http_{resp.status_code}:{resp.text[:200]}")
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[len("json") :].strip()
        return json.loads(content)

    def analyze_codebase(self, path: Path) -> List[ProposedSkill]:
        snippets = self._read_py_snippets(path)
        if not snippets:
            return []
        merged = "\n\n".join([f"# File: {p}\n{code}" for p, code in snippets])
        system_prompt = (
            "Analiza este código Python. Identifica funciones o clases que puedan exponerse como herramientas 'skills' autónomas. "
            "Devuelve SOLO JSON: {\"skills\":[{\"name\":\"...\",\"description\":\"...\",\"args\":[{\"name\":\"\",\"type\":\"\",\"required\":true}],\"deps\":[\"...\"],\"source_file\":\"...\"}]}"
        )
        user_prompt = merged[:8000]  # proteger tokens
        try:
            brain_resp = self._call_brain(system_prompt, user_prompt)
        except Exception as exc:
            log.warning("Brain analysis failed: %s", exc)
            return []
        skills_data = brain_resp.get("skills") if isinstance(brain_resp, dict) else None
        out: List[ProposedSkill] = []
        if isinstance(skills_data, list):
            for item in skills_data:
                if not isinstance(item, dict):
                    continue
                out.append(
                    ProposedSkill(
                        name=str(item.get("name") or "skill").strip(),
                        description=str(item.get("description") or "").strip(),
                        args=item.get("args") or [],
                        deps=item.get("deps") or [],
                        source_file=item.get("source_file"),
                    )
                )
        if out:
            self._write_skill_proposals(out)
        return out
    # --- 2) Gatekeeper (seguridad / calidad) ---

    def _assess_risk(self, result: SearchResult) -> RiskAssessment:
        """
        Evalúa riesgos básicos y calidad del recurso:
        - Dominio permitido
        - Estrellas / recencia (si están disponibles)
        - Patrones de código potencialmente peligrosos
        """
        domain_ok = self._is_domain_allowed(result.url)

        stars = int(result.extra.get("stars", 0))
        stars_ok = stars >= self.config.min_stars

        recent_ok = True
        last_commit_iso = result.extra.get("last_commit_iso")
        if last_commit_iso:
            try:
                last_dt = datetime.datetime.fromisoformat(last_commit_iso.replace("Z", "+00:00"))
                age = (datetime.datetime.now(datetime.timezone.utc) - last_dt).days
                recent_ok = age <= self.config.max_last_commit_age_days
            except Exception:
                recent_ok = True

        code = result.extra.get("sample_code", "") or ""
        code_risk_flags = self._scan_code_for_risks(code)

        if not domain_ok:
            overall = "high"
        elif code_risk_flags:
            overall = "high"
        elif not stars_ok or not recent_ok:
            overall = "medium"
        else:
            overall = "low"

        return RiskAssessment(
            domain_ok=domain_ok,
            stars_ok=stars_ok,
            recent_ok=recent_ok,
            code_risk_flags=code_risk_flags,
            overall_risk=overall,
        )

    def _scan_code_for_risks(self, code: str) -> List[str]:
        """
        Busca patrones obvios de peligro en fragmentos de código.
        """
        flags: List[str] = []
        patterns = {
            "shell_rm": r"rm\s+-rf\s+/",
            "subprocess_shell": r"subprocess\.run\([^)]*shell=True",
            "os_system": r"os\.system\(",
            "eval_exec": r"\b(eval|exec)\(",
        }
        for name, pat in patterns.items():
            if re.search(pat, code):
                flags.append(name)
        return flags

    # --- 3) Proposal Generator ---

    def _generate_proposal(
        self,
        topic: str,
        context: str,
        selected: List[SearchResult],
        risks: Dict[str, RiskAssessment],
        harvested_path: Optional[Path],
        proposed_skills: List[ProposedSkill],
    ) -> ResearchProposal:
        suggestions: List[str] = []
        code_example: Optional[str] = None

        for r in selected:
            ra = risks[r.url]
            suggestions.append(
                f"- [{r.title}]({r.url}) — riesgo: {ra.overall_risk}, stars_ok={ra.stars_ok}, recent_ok={ra.recent_ok}"
            )
            if code_example is None and r.extra.get("sample_code"):
                code_example = textwrap.dedent(r.extra["sample_code"])

        return ResearchProposal(
            topic=topic,
            context=context,
            selected=selected,
            risks=risks,
            suggestions=suggestions,
            code_example=code_example,
            harvested_path=str(harvested_path) if harvested_path else None,
            proposed_skills=proposed_skills,
        )

    # --- 4) Informe Markdown ---

    def _write_report_markdown(self, proposal: ResearchProposal) -> Path:
        """
        Crea una carpeta de investigación y escribe un informe Markdown.
        Ruta: artifacts/research/YYYY-MM-DD_topic-slug/report.md
        """
        today = datetime.date.today().isoformat()
        slug = self._slugify(proposal.topic)[:60] or "topic"
        base_dir = self.config.research_dir / f"{today}_{slug}"
        base_dir.mkdir(parents=True, exist_ok=True)

        report_path = base_dir / "report.md"
        lines: List[str] = []
        lines.append(f"# Research Report — {proposal.topic}\n")
        lines.append(f"**Contexto:** {proposal.context}\n")
        if proposal.harvested_path:
            lines.append(f"**Código descargado en:** `{proposal.harvested_path}`\n")
        lines.append("## Opciones encontradas\n")

        for r in proposal.selected:
            ra = proposal.risks[r.url]
            lines.append(f"### {r.title}")
            lines.append(f"- URL: {r.url}")
            lines.append(f"- Fuente: `{r.source}`")
            lines.append(f"- Riesgo: **{ra.overall_risk}**")
            lines.append(f"- Dominio permitido: {ra.domain_ok}")
            lines.append(f"- Stars suficientes: {ra.stars_ok}")
            lines.append(f"- Reciente: {ra.recent_ok}")
            if ra.code_risk_flags:
                lines.append(f"- Flags de riesgo en código: `{', '.join(ra.code_risk_flags)}`")
            lines.append("")
            if r.snippet:
                lines.append("Snippet:")
                lines.append("")
                lines.append(textwrap.indent(r.snippet, "> "))
                lines.append("")

        if proposal.code_example:
            lines.append("## Código de ejemplo (ADAPTAR A AJAX)")
            lines.append("")
            lines.append("```python")
            lines.append(proposal.code_example.rstrip())
            lines.append("```")
            lines.append("")

        if proposal.proposed_skills:
            lines.append("## Habilidades candidatas (Scout 2.0)")
            lines.append("")
            for sk in proposal.proposed_skills:
                lines.append(f"- **{sk.name}** ({sk.source_file or 'desconocido'}): {sk.description}")
                if sk.args:
                    lines.append(f"  - args: {json.dumps(sk.args, ensure_ascii=False)}")
                if sk.deps:
                    lines.append(f"  - deps: {', '.join(sk.deps)}")
            lines.append("")

        lines.append("## Análisis de riesgo agregado\n")
        for s in proposal.suggestions:
            lines.append(s)
        lines.append("")

        try:
            rel = report_path.relative_to(self.sandbox_dir)
        except Exception:
            rel = Path("research") / f"{today}_{slug}" / "report.md"
        report_path = write_scout_file(rel, "\n".join(lines), root=self.config.root_dir)

        # Indexar en LEANN (best-effort)
        try:
            from agency.leann_client import LeannClient  # type: ignore

            client = LeannClient()
            client.index_path(report_path)
        except Exception:
            pass

        return report_path

    # --- Utilidades ---

    def _slugify(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        return text.strip("-")
