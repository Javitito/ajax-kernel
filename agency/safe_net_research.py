from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class SafeNetResearch:
    """
    Búsqueda web con saneado defensivo contra prompt-injection.
    Devuelve contenido marcado como información externa, nunca instrucciones.
    """

    def __init__(self, http_client: Any) -> None:
        self.http = http_client

    def _extract_text(self, item: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(item, str):
            return item, {}
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "")
            meta = {k: v for k, v in item.items() if k not in {"text", "content"}}
            return text, meta
        return str(item), {}

    def _sanitize(self, text: str) -> str:
        markers = [
            "ignore previous",
            "disregard previous",
            "you are an ai assistant",
            "system prompt",
            "execute the following",
            "run this code",
            "change your instructions",
            "shut down",
            "format the disk",
        ]
        filtered_lines: List[str] = []
        for line in text.splitlines():
            lower = line.lower()
            if any(marker in lower for marker in markers):
                continue
            filtered_lines.append(line)
        sanitized = "\n".join(filtered_lines).strip()
        return sanitized

    def search(self, query: str, max_docs: int = 5) -> List[Dict[str, Any]]:
        raw_docs = []
        try:
            if hasattr(self.http, "search"):
                raw_docs = self.http.search(query, max_docs=max_docs) or []
        except Exception:
            raw_docs = []
        cleaned: List[Dict[str, Any]] = []
        for item in raw_docs:
            text, meta = self._extract_text(item)
            sanitized = self._sanitize(text)
            if not sanitized:
                continue
            cleaned.append(
                {
                    "kind": "external_info",
                    "text": sanitized,
                    "source": meta.get("source") or meta.get("url") or "web",
                    "meta": meta,
                }
            )
        return cleaned
