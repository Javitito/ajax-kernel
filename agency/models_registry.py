from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests  # type: ignore
import yaml  # type: ignore


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "model_providers.yaml"


@dataclass
class ModelInfo:
    id: str
    provider: str
    source: str
    modalities: List[str]
    raw: Any


def _load_config() -> Dict[str, Any]:
    try:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _env(key: str | None) -> Optional[str]:
    return os.getenv(key) if key else None


def _detect_modalities(model_id: str, cfg: Dict[str, Any], explicit: Optional[List[str]] = None) -> List[str]:
    if explicit:
        return explicit
    modes = ["text"]
    wl = cfg.get("vision_whitelist") or []
    hints = [h.lower() for h in (cfg.get("vision_hints") or [])]
    if model_id in wl:
        modes.append("vision")
    else:
        low = model_id.lower()
        if any(h in low for h in hints):
            modes.append("vision")
    return list(dict.fromkeys(modes))


def _from_openai_http(provider: str, cfg: Dict[str, Any]) -> List[ModelInfo]:
    base = (cfg.get("base_url") or "").rstrip("/")
    key = _env(cfg.get("api_key_env"))
    out: List[ModelInfo] = []
    if not base or (cfg.get("api_key_env") and not key):
        # sin credenciales/base, intenta static_models si existen
        out.extend(_from_static(provider, cfg))
        return out
    try:
        resp = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data") or []
    except Exception:
        # si falla, usar static_models como fallback
        out.extend(_from_static(provider, cfg))
        return out
    for item in data:
        mid = item.get("id") or item.get("name")
        if not mid:
            continue
        out.append(ModelInfo(id=mid, provider=provider, source="http_openai", modalities=_detect_modalities(mid, cfg), raw=item))
    # incluir static_models si hay (etiquetados como static para diferenciar)
    out.extend(_from_static(provider, cfg))
    return out


def _from_http_gemini(provider: str, cfg: Dict[str, Any]) -> List[ModelInfo]:
    key = _env(cfg.get("api_key_env"))
    if not key:
        return []
    base = (cfg.get("base_url") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    url = f"{base}/models?key={key}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("models") or []
    except Exception:
        return []
    out: List[ModelInfo] = []
    for item in data:
        mid = item.get("name")
        if not mid:
            continue
        explicit = item.get("modalities") or item.get("supported_generation_methods")
        out.append(ModelInfo(id=mid, provider=provider, source="http_gemini", modalities=_detect_modalities(mid, cfg, explicit if isinstance(explicit, list) else None), raw=item))
    return out


def _from_cli(provider: str, cfg: Dict[str, Any]) -> List[ModelInfo]:
    if cfg.get("kind") == "codex_cli_jsonl":
        return _from_static(provider, cfg)
    list_cmd = cfg.get("list_command")
    list_fmt = (cfg.get("list_format") or cfg.get("format") or "text").lower()
    if isinstance(list_cmd, str):
        list_cmd = [list_cmd]
    if list_cmd and isinstance(list_cmd, list):
        env = os.environ.copy()
        env.update(cfg.get("env") or {})
        workdir = cfg.get("workdir") or "."
        try:
            res = subprocess.run(list_cmd, cwd=workdir, env=env, check=True, capture_output=True, text=True, timeout=30)
            parsed = _parse_cli_output(res.stdout, list_fmt, provider, cfg)
            if parsed:
                return parsed
        except Exception:
            pass
    # Si hay static_models declarados, priorizar eso y evitar llamar comandos no listables
    if cfg.get("static_models") or cfg.get("models"):
        return _from_static(provider, cfg)
    cmd = cfg.get("command")
    fmt = (cfg.get("format") or "text").lower()
    if not cmd:
        return []
    env = os.environ.copy()
    env.update(cfg.get("env") or {})
    workdir = cfg.get("workdir") or "."
    try:
        res = subprocess.run(cmd, cwd=workdir, env=env, check=True, capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    return _parse_cli_output(res.stdout, fmt, provider, cfg)


def _parse_cli_output(stdout: str, fmt: str, provider: str, cfg: Dict[str, Any]) -> List[ModelInfo]:
    out: List[ModelInfo] = []
    if fmt == "json":
        try:
            data = json.loads(stdout)
        except Exception:
            return []
        items = data.get("data") if isinstance(data, dict) and "data" in data else data
        if not isinstance(items, list):
            return []
        for item in items:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("name")
                explicit = item.get("modalities")
            else:
                mid = str(item)
                explicit = None
            if not mid:
                continue
            out.append(
                ModelInfo(
                    id=mid,
                    provider=provider,
                    source="cli",
                    modalities=_detect_modalities(mid, cfg, explicit),
                    raw=item,
                )
            )
    else:
        lines = stdout.strip().splitlines()
        for ln in lines:
            if not ln.strip():
                continue
            # saltar cabecera si parece encabezado (ej. empieza por MODEL o NAME)
            first_tok = ln.strip().split()[0]
            if first_tok.lower().startswith(("model", "name", "id")) and len(ln.split()) > 1:
                continue
            mid = first_tok
            if not mid:
                continue
            out.append(
                ModelInfo(
                    id=mid,
                    provider=provider,
                    source="cli",
                    modalities=_detect_modalities(mid, cfg),
                    raw=ln,
                )
            )
    return out


def _from_static(provider: str, cfg: Dict[str, Any]) -> List[ModelInfo]:
    out: List[ModelInfo] = []
    models_src = cfg.get("static_models") or cfg.get("models") or []
    for item in models_src:
        if isinstance(item, str):
            mid = item
            explicit = None
            raw = item
        else:
            mid = item.get("id")
            explicit = item.get("modalities")
            raw = item
        if not mid:
            continue
        out.append(ModelInfo(id=mid, provider=provider, source="static", modalities=_detect_modalities(mid, cfg, explicit), raw=raw))
    return out


def _from_web_catalog(provider: str) -> List[ModelInfo]:
    catalog_path = CONFIG_PATH.parent.parent / "artifacts" / "providers_catalog_web.json"
    if not catalog_path.exists():
        return []
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        p_data = data.get("providers", {}).get(provider) or data.get("alignment", {}).get(provider)
        if not p_data:
            return []
        
        models_list = p_data.get("models") or []
        # Soporte para el formato de alineación rápida
        if not models_list and "available" in p_data:
            models_list = [{"id": mid} for mid in p_data["available"]]
            
        out = []
        for m in models_list:
            mid = m.get("id") if isinstance(m, dict) else str(m)
            if not mid: continue
            out.append(ModelInfo(
                id=mid,
                provider=provider,
                source="web_catalog",
                modalities=m.get("modalities", ["text"]) if isinstance(m, dict) else ["text"],
                raw=m
            ))
        return out
    except Exception:
        return []

def discover_models(provider: Optional[str] = None) -> List[ModelInfo]:
    cfg = _load_config().get("providers", {}) or {}
    providers = [provider] if provider else list(cfg.keys())
    out: List[ModelInfo] = []
    for name in providers:
        # 1. Intentar Catálogo Web primero (Suelo de verdad actualizado)
        web_models = _from_web_catalog(name)
        
        pcfg = cfg.get(name) or {}
        kind = (pcfg.get("kind") or "").lower()
        
        live_models = []
        if kind == "http_openai":
            live_models = _from_openai_http(name, pcfg)
        elif kind == "http_gemini":
            live_models = _from_http_gemini(name, pcfg)
        elif kind in {"cli", "codex_cli_jsonl"}:
            live_models = _from_cli(name, pcfg)
        elif kind == "static":
            live_models = _from_static(name, pcfg)
            
        # Unificar evitando duplicados, priorizando el descubrimiento vivo pero manteniendo
        # los modelos del catálogo web que no hayan aparecido en la API.
        seen_ids = {m.id for m in live_models}
        out.extend(live_models)
        for wm in web_models:
            if wm.id not in seen_ids:
                out.append(wm)
                seen_ids.add(wm.id)
                
    return out


def list_vision_models(provider: Optional[str] = None) -> List[ModelInfo]:
    models = discover_models(provider)
    return [m for m in models if "vision" in m.modalities]


def _print_table(models: List[ModelInfo]) -> None:
    if not models:
        print("no models found")
        return
    for m in models:
        print(f"{m.provider}\t{m.source}\t{m.id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Descubre modelos desde los proveedores configurados.")
    ap.add_argument("--vision-only", action="store_true", help="Mostrar solo modelos de visión")
    ap.add_argument("--provider", help="Filtrar por proveedor concreto", default=None)
    args = ap.parse_args()
    mods = list_vision_models(args.provider) if args.vision_only else discover_models(args.provider)
    _print_table(mods)


if __name__ == "__main__":
    main()
