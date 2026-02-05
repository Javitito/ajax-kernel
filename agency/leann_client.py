from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class LeannClient:
    """
    Cliente ligero para LEANN RAG.
    - query(text, k): devuelve texto concatenado de pasajes relevantes.
    - index_path(path): stub opcional para ingesta directa (best-effort).
    """

    def __init__(
        self,
        base_url_web: str = "http://127.0.0.1:5002",
        base_url_rag: str = "http://127.0.0.1:8000",
        web_key_path: Optional[Path] = None,
        rag_token_path: Optional[Path] = None,
    ) -> None:
        self.base_url_web = base_url_web.rstrip("/")
        self.base_url_rag = base_url_rag.rstrip("/")
        self.web_api_key = self._read_key(web_key_path or Path.home() / ".leann" / "web_api_key")
        self.rag_token = os.getenv("LEANN_RAG_AUTH_TOKEN") or os.getenv("AUTH_TOKEN") or self._read_key(
            rag_token_path or Path.home() / ".leann" / "rag_auth_token"
        )

    def _read_key(self, path: Path) -> Optional[str]:
        try:
            if path.exists():
                key = path.read_text(encoding="utf-8").strip()
                return key or None
        except Exception:
            pass
        return None

    def query(self, text: str, n_results: int = 3) -> str:
        text = text.strip()
        if not text:
            return ""
        # Prefer web proxy if key available
        if self.web_api_key:
            try:
                headers = {"X-API-Key": self.web_api_key, "Content-Type": "application/json"}
                payload = {"q": text, "k": n_results}
                resp = requests.post(f"{self.base_url_web}/api/rag/query", headers=headers, json=payload, timeout=8)
                if resp.status_code < 400:
                    data = resp.json()
                    passages = data.get("results") or data.get("passages") or []
                    snippets = []
                    for p in passages[:n_results]:
                        snippets.append(p.get("text") or p.get("content") or "")
                    return "\n\n".join([s for s in snippets if s])
            except Exception:
                pass
        # Fallback directo al RAG 8000 si hay token
        if self.rag_token:
            try:
                headers = {"Authorization": f"Bearer {self.rag_token}", "Content-Type": "application/json"}
                payload = {"query": text, "k": n_results}
                resp = requests.post(f"{self.base_url_rag}/rag/search", headers=headers, json=payload, timeout=8)
                if resp.status_code < 400:
                    data = resp.json()
                    passages = data.get("results") or data.get("passages") or []
                    snippets = []
                    for p in passages[:n_results]:
                        snippets.append(p.get("text") or p.get("content") or "")
                    return "\n\n".join([s for s in snippets if s])
            except Exception:
                pass
        return ""

    def index_path(self, path: Path) -> bool:
        """
        Best-effort: si hay API expuesta para ingesta vía web proxy, se puede implementar aquí.
        Por ahora, stub que retorna False si no hay endpoint conocido.
        """
        # TODO: implementar si existe endpoint /api/rag/ingest
        return False
