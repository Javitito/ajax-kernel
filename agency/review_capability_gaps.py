"""
Utilidades para revisar capability_gaps y actualizar el backlog de investigación.

Pipeline básico:
1) scan_gaps: agrupa artifacts/capability_gaps/*.json por capability_family
   y devuelve un dict serializable para el Director_I+D_AJAX.
2) apply_decisions: toma la respuesta del Director y fusiona en
   LEANN_CAP_GAPS/research_backlog.yaml.

Esquemas esperados (versión 1, tolerante con formatos previos):

- capability_gap (JSON en artifacts/capability_gaps/):
  {
    "mission_id": "...",
    "job_id": "...",
    "capability_family": "desktop.editor_close",
    "symptoms": ["efe_mismatch", "editor_window_still_open"],
    "efe_delta": {...},
    "skills_involved": ["editor.close_safe"],
    "created_at": "ISO8601",
    "evidence_paths": [...]
  }

- Entrada en research_backlog.yaml (lista de entradas):
  - capability_family: desktop.editor_close
    decision: prioridad_alta | prioridad_baja | workaround_aceptado | no_viable_ahora
    rationale: "texto breve"
    evidence: [paths a capability_gaps...]
    project:
      id: desktop_editor_lifecycle_v1
      goal: "..."
      status: proposed|in_progress|done|dropped
      suggested_steps: [...]

- director_decisions.json (decisiones del Director I+D):
  {
    "families": [
      {
        "capability_family": "desktop.editor_close",
        "decision": "prioridad_alta",
        "rationale": "...",
        "evidence": ["artifacts/capability_gaps/...json"],
        "project": { ... }
      },
      ...
    ]
  }
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore
import logging

log = logging.getLogger(__name__)

DIRECTOR_DECISIONS_PATH = Path("director_decisions.json")

# ---------- Modelos internos ----------
@dataclass
class CapabilityGap:
    path: Path
    capability_family: str
    symptom: Optional[str] = None
    job_id: Optional[str] = None
    mission_id: Optional[str] = None
    efe_delta: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityFamilySummary:
    capability_family: str
    gaps: List[CapabilityGap] = field(default_factory=list)


# ---------- Paso 1: escanear gaps ----------
def scan_gaps(
    gaps_dir: str = "artifacts/capability_gaps",
    backlog_path: str = "LEANN_CAP_GAPS/research_backlog.yaml",
) -> Dict[str, Any]:
    """
    Escanea artifacts/capability_gaps/*.json y agrupa por capability_family.
    Devuelve un dict serializable para pasar al Director_I+D_AJAX.
    """
    root = Path(gaps_dir)
    families: Dict[str, CapabilityFamilySummary] = {}

    if not root.exists():
        log.info("SCAN_GAPS: ningún gap detectado (directorio %s no existe)", gaps_dir)
        return {"families": {}, "total_gaps": 0, "backlog_path": backlog_path}

    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # gap corrupto; lo ignoramos
            continue

        fam = data.get("capability_family")
        if not fam:
            continue

        gap = CapabilityGap(
            path=path,
            capability_family=str(fam),
            symptom=data.get("symptom"),
            job_id=data.get("job_id"),
            mission_id=data.get("mission_id"),
            efe_delta=data.get("efe_delta"),
            raw=data,
        )

        if fam not in families:
            families[fam] = CapabilityFamilySummary(capability_family=str(fam))
        families[fam].gaps.append(gap)

    total_gaps = sum(len(fam.gaps) for fam in families.values())
    mapping = {
        fam.capability_family: [g.raw.get("mission_id") or g.raw.get("job_id") or g.path.stem for g in fam.gaps]
        for fam in families.values()
    }

    result = {
        "families": mapping,
        "total_gaps": total_gaps,
        "backlog_path": backlog_path,
    }

    summary = ", ".join(f"{name} ({len(ids)} gaps)" for name, ids in mapping.items())
    log.info("SCAN_GAPS: found families: %s (total_gaps=%d)", summary or "none", total_gaps)

    return result


# ---------- Paso 2: aplicar decisiones del Director ----------
def _load_backlog(backlog_path: Path) -> List[Dict[str, Any]]:
    if not backlog_path.exists():
        return []
    data = yaml.safe_load(backlog_path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError(f"Backlog YAML must be a list, got: {type(data)}")
    return data


def _save_backlog(backlog_path: Path, entries: List[Dict[str, Any]]) -> None:
    backlog_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True)
    backlog_path.write_text(text, encoding="utf-8")


def apply_decisions(
    director_decisions: Dict[str, Any],
    backlog_path: str = "LEANN_CAP_GAPS/research_backlog.yaml",
) -> Dict[str, Any]:
    """
    Aplica las decisiones del Director_I+D_AJAX al backlog.
    director_decisions formato esperado:
    {
      "families": [
        {
          "capability_family": "...",
          "decision": "prioridad_alta",
          "rationale": "...",
          "evidence": [...],
          "project": {...}
        }, ...
      ]
    }
    """
    backlog_file = Path(backlog_path)
    current = _load_backlog(backlog_file)
    by_family: Dict[str, Dict[str, Any]] = {
        e.get("capability_family"): e for e in current if isinstance(e, dict) and e.get("capability_family")
    }

    families_decisions = director_decisions.get("families", []) or []
    if isinstance(families_decisions, dict):
        # soportar formato {"family": {...}}
        families_decisions = [
            {"capability_family": fam, **val} if isinstance(val, dict) else {"capability_family": fam}
            for fam, val in families_decisions.items()
        ]

    applied: List[str] = []
    for fam_dec in families_decisions:
        fam_name = fam_dec.get("capability_family")
        if not fam_name:
            continue

        if fam_name not in by_family:
            log.warning("APPLY_DECISIONS: familia desconocida en backlog, se ignora: %s", fam_name)
            continue

        existing = by_family.get(fam_name, {"capability_family": fam_name})

        # Fusionar evidencia
        existing_evidence = existing.get("evidence", []) or []
        new_evidence = fam_dec.get("evidence", []) or []
        merged_evidence = sorted(set(existing_evidence + new_evidence))

        existing["decision"] = fam_dec.get("decision", existing.get("decision"))
        existing["rationale"] = fam_dec.get("rationale", existing.get("rationale"))
        existing["evidence"] = merged_evidence

        # Project: si ya hay id, se respeta; si no, se crea
        if fam_dec.get("project"):
            proj_in = fam_dec["project"]
            if existing.get("project") and existing["project"].get("id"):
                existing_proj = existing["project"]
                for key, value in proj_in.items():
                    if key == "id":
                        continue
                    if value is not None:
                        existing_proj[key] = value
            else:
                existing["project"] = proj_in

        by_family[fam_name] = existing
        applied.append(fam_name)

    if families_decisions and not applied:
        log.info("APPLY_DECISIONS: nada que aplicar (familias desconocidas o decisiones vacías)")

    updated = list(by_family.values())
    _save_backlog(backlog_file, updated)
    for fam in applied:
        log.info("APPLY_DECISIONS: aplicada decision para familia=%s", fam)
    return {"updated_families": applied or list(by_family.keys())}


# ---------- Paso 3: planificar decisiones (manual/auto) ----------
def call_models_brain_for_decisions(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoca al modelo (Director_I+D_AJAX) para decidir sobre capability gaps.
    Entrada esperada: resultado de scan_gaps() con claves:
      - families: dict[capability_family -> list[gap_ids]]
      - total_gaps: int
    Devuelve un dict con forma {"families": [ {...}, ... ]} compatible con apply_decisions.
    """
    families_map = scan_result.get("families") or {}
    gaps_dir = Path("artifacts") / "capability_gaps"

    def _load_gaps_for_family(fam: str, ids: List[str]) -> List[Dict[str, Any]]:
        gaps: List[Dict[str, Any]] = []
        if not gaps_dir.exists():
            return gaps
        for path in gaps_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("capability_family") != fam:
                continue
            gap_id = data.get("gap_id") or data.get("mission_id") or path.stem
            if ids and gap_id not in ids and data.get("mission_id") not in ids:
                continue
            gaps.append(
                {
                    "gap_id": gap_id,
                    "mission_id": data.get("mission_id"),
                    "symptoms": data.get("symptoms"),
                    "efe_delta": data.get("efe_delta"),
                    "evidence_paths": data.get("evidence_paths"),
                }
            )
        return gaps

    def _backlog_snapshot() -> Dict[str, Any]:
        snap: Dict[str, Any] = {}
        try:
            backlog_path = Path("LEANN_CAP_GAPS") / "research_backlog.yaml"
            if not backlog_path.exists():
                return snap
            entries = yaml.safe_load(backlog_path.read_text(encoding="utf-8")) or []
            if isinstance(entries, list):
                for entry in entries:
                    fam = (entry or {}).get("capability_family")
                    if not fam:
                        continue
                    snap[fam] = {
                        "project_id": (entry.get("project") or {}).get("id"),
                        "status": entry.get("decision") or entry.get("status"),
                        "existing_evidence": entry.get("evidence"),
                    }
        except Exception as exc:
            log.warning("call_models_brain_for_decisions: no se pudo leer backlog: %s", exc)
        return snap

    gaps_by_family = {
        fam: _load_gaps_for_family(fam, ids if isinstance(ids, list) else [])
        for fam, ids in families_map.items()
    }
    payload = {
        "families": list(families_map.keys()),
        "gaps_by_family": gaps_by_family,
        "backlog_snapshot": _backlog_snapshot(),
    }

    log.info(
        "BRAIN: sending payload for families=%s total_gaps=%s",
        list(families_map.keys()),
        scan_result.get("total_gaps"),
    )

    cmd = [
        sys.executable,
        "bin/qwen_task.py",
        "director_capability_gaps",
        "--input-json",
        json.dumps(payload, ensure_ascii=False),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:
        log.warning("BRAIN: fallo al invocar modelo (%s)", exc)
        return {"families": []}

    if proc.returncode != 0:
        log.warning("BRAIN: modelo devolvió rc=%s stderr=%s", proc.returncode, proc.stderr.strip())
        return {"families": []}

    try:
        decisions = json.loads(proc.stdout or "{}")
    except Exception as exc:
        log.warning("BRAIN: no se pudo parsear salida JSON (%s)", exc)
        return {"families": []}

    fams_out = decisions.get("families")
    if not isinstance(fams_out, list):
        log.warning("BRAIN: salida sin 'families' lista; fallback vacío")
        return {"families": []}

    log.info("BRAIN: received decisions for families=%s", [f.get("capability_family") for f in fams_out if isinstance(f, dict)])
    return decisions


def plan_decisions(mode: str = "manual") -> Dict[str, Any]:
    """
    Planifica decisiones del Director I+D para capability gaps.

    - mode=\"manual\": respeta director_decisions.json si existe y tiene families; si no, crea stub {\"families\": []}.
    - mode=\"auto\": si no hay families en director_decisions.json, invoca call_models_brain_for_decisions(scan_gaps()) y escribe el resultado.
    Devuelve un dict con información básica.
    """
    used_existing = False
    decisions_path = DIRECTOR_DECISIONS_PATH

    # Si ya existe y tiene families no vacías, respetar
    if decisions_path.exists():
        try:
            data = json.loads(decisions_path.read_text(encoding="utf-8") or "{}")
            fams = data.get("families") or []
            if fams:
                used_existing = True
                log.info("plan_decisions: reutilizando director_decisions.json existente con %d familias", len(fams))
                return {
                    "mode": mode,
                    "used_existing_file": True,
                    "families_planned": [
                        f.get("capability_family") for f in fams if isinstance(f, dict)
                    ],
                }
        except Exception as exc:
            log.warning("plan_decisions: no se pudo leer director_decisions.json (%s), se recreará", exc)

    if mode == "manual":
        decisions_path.write_text(json.dumps({"families": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("plan_decisions: modo manual, stub creado en %s", decisions_path)
        return {"mode": mode, "used_existing_file": used_existing, "families_planned": []}

    # mode auto: intentar generar decisions si no hay
    scan = scan_gaps()
    if not scan.get("families"):
        decisions_path.write_text(json.dumps({"families": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("plan_decisions: auto pero sin gaps; stub vacío creado")
        return {"mode": mode, "used_existing_file": used_existing, "families_planned": []}

    decisions = call_models_brain_for_decisions(scan)
    decisions_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    fams_out = [f.get("capability_family") for f in decisions.get("families", []) if isinstance(f, dict)]
    log.info("plan_decisions: auto, families generadas=%s", fams_out)
    return {"mode": mode, "used_existing_file": used_existing, "families_planned": fams_out}
