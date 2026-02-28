from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TextIO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_BASE = ROOT / ".leann" / "indexes" / "antigravity_skills_safe" / "documents.leann"
DEFAULT_RECEIPTS_DIR = ROOT / "artifacts" / "receipts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _compact(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass
class LeannBackend:
    index_base: Path
    receipts_dir: Path

    @classmethod
    def from_env(cls) -> "LeannBackend":
        index_raw = os.getenv("LEANN_MCP_INDEX_BASE") or str(DEFAULT_INDEX_BASE)
        receipts_raw = os.getenv("LEANN_MCP_RECEIPTS_DIR") or str(DEFAULT_RECEIPTS_DIR)
        return cls(index_base=Path(index_raw), receipts_dir=Path(receipts_raw))

    def _meta_path(self) -> Path:
        return Path(f"{self.index_base}.meta.json")

    def _passages_path(self) -> Path:
        return Path(f"{self.index_base}.passages.jsonl")

    def _capability_missing(self, operation: str, *, detail: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ok": False,
            "error": "capability_missing",
            "operation": operation,
            "index_base": str(self.index_base),
            "how_to_fix": [
                "Ensure LEANN is installed and the index exists.",
                "Run: leann build antigravity_skills_safe --docs .leann/sources/skills/antigravity --include-hidden --file-types .md --force",
            ],
        }
        if detail:
            payload["detail"] = detail
        return payload

    def _index_available(self) -> bool:
        return self._meta_path().exists() and self._passages_path().exists()

    def search(self, query: str, k: int = 5, filters: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        text = str(query or "").strip()
        if not text:
            return {"ok": False, "error": "invalid_params", "detail": "query_required"}
        if not self._index_available():
            return self._capability_missing("leann.search")
        try:
            from agency.leann_query_client import query_leann

            hits = query_leann(str(self.index_base), text, top_k=max(1, int(k)), fallback_grep=True)
        except Exception as exc:
            return self._capability_missing("leann.search", detail=str(exc)[:200])

        results: List[Dict[str, Any]] = []
        for hit in hits or []:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
            results.append(
                {
                    "doc_id": str(hit.get("id")) if hit.get("id") is not None else "",
                    "score": hit.get("score"),
                    "snippet": str(hit.get("text") or "")[:400],
                    "source_mode": hit.get("source_mode"),
                    "metadata": meta,
                }
            )
        return {
            "ok": True,
            "query": text,
            "k": max(1, int(k)),
            "filters": dict(filters or {}),
            "results": results,
        }

    def read(
        self,
        doc_id: str,
        *,
        span: Optional[Mapping[str, Any]] = None,
        range_: Optional[Any] = None,
    ) -> Dict[str, Any]:
        target_id = str(doc_id or "").strip()
        if not target_id:
            return {"ok": False, "error": "invalid_params", "detail": "doc_id_required"}
        if not self._index_available():
            return self._capability_missing("leann.read")

        record: Optional[Dict[str, Any]] = None
        passages = self._passages_path()
        try:
            with passages.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if str(data.get("id")) == target_id:
                        record = data
                        break
        except Exception as exc:
            return {"ok": False, "error": "read_failed", "detail": str(exc)[:200]}

        if record is None:
            return {"ok": False, "error": "not_found", "doc_id": target_id}

        text = str(record.get("text") or "")
        start = 0
        end = len(text)
        if isinstance(span, Mapping):
            try:
                start = int(span.get("start", start))
                end = int(span.get("end", end))
            except Exception:
                start, end = 0, len(text)
        elif isinstance(range_, Mapping):
            try:
                start = int(range_.get("start", start))
                end = int(range_.get("end", end))
            except Exception:
                start, end = 0, len(text)
        elif isinstance(range_, (list, tuple)) and len(range_) >= 2:
            try:
                start = int(range_[0])
                end = int(range_[1])
            except Exception:
                start, end = 0, len(text)
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        content = text[start:end]

        return {
            "ok": True,
            "doc_id": target_id,
            "content": content,
            "span": {"start": start, "end": end},
            "metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
        }

    def receipts_latest(self, k: int = 20) -> Dict[str, Any]:
        n = max(1, int(k))
        if not self.receipts_dir.exists():
            return {"ok": True, "k": n, "results": []}
        items = sorted(
            [p for p in self.receipts_dir.glob("*.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:n]
        results: List[Dict[str, Any]] = []
        for path in items:
            payload = _safe_json_load(path)
            results.append(
                {
                    "path": str(path),
                    "schema": payload.get("schema"),
                    "status": payload.get("status"),
                    "trigger": payload.get("trigger"),
                    "created_utc": payload.get("created_utc") or payload.get("ts_utc"),
                }
            )
        return {"ok": True, "k": n, "results": results}


class LeannMCPServer:
    def __init__(
        self,
        *,
        backend: Optional[LeannBackend] = None,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
    ) -> None:
        self.backend = backend or LeannBackend.from_env()
        self.stdin = stdin
        self.stdout = stdout

    @staticmethod
    def _result(resp_id: Any, result: Mapping[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": resp_id, "result": dict(result)}

    @staticmethod
    def _error(resp_id: Any, code: int, message: str, data: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": resp_id, "error": {"code": int(code), "message": message}}
        if data:
            payload["error"]["data"] = dict(data)
        return payload

    def handle_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            return self._error(req_id, -32600, "invalid_request", {"reason": "jsonrpc_must_be_2.0"})
        method = request.get("method")
        if not isinstance(method, str) or not method.strip():
            return self._error(req_id, -32600, "invalid_request", {"reason": "method_required"})
        params = request.get("params") if isinstance(request.get("params"), Mapping) else {}

        try:
            if method == "leann.search":
                query = str(params.get("query") or "")
                k = int(params.get("k", 5))
                filters = params.get("filters") if isinstance(params.get("filters"), Mapping) else {}
                return self._result(req_id, self.backend.search(query, k=k, filters=filters))
            if method == "leann.read":
                doc_id = str(params.get("doc_id") or "")
                span = params.get("span") if isinstance(params.get("span"), Mapping) else None
                range_ = params.get("range")
                return self._result(req_id, self.backend.read(doc_id, span=span, range_=range_))
            if method == "leann.receipts.latest":
                k = int(params.get("k", 20))
                return self._result(req_id, self.backend.receipts_latest(k=k))
            if method == "tools/list":
                return self._result(
                    req_id,
                    {
                        "ok": True,
                        "tools": [
                            {"name": "leann.search", "description": "Search LEANN index (read-only)."},
                            {"name": "leann.read", "description": "Read a LEANN doc chunk by doc_id (read-only)."},
                            {
                                "name": "leann.receipts.latest",
                                "description": "List latest local receipts metadata (read-only).",
                            },
                        ],
                    },
                )
            return self._error(req_id, -32601, "method_not_found", {"method": method})
        except Exception as exc:
            return self._error(req_id, -32000, "internal_error", {"detail": str(exc)[:200]})

    def run_forever(self) -> int:
        for raw in self.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except Exception:
                response = self._error(None, -32700, "parse_error", {"received": line[:120]})
                self.stdout.write(_compact(response) + "\n")
                self.stdout.flush()
                continue
            if not isinstance(request, Mapping):
                response = self._error(None, -32600, "invalid_request", {"reason": "request_must_be_object"})
            else:
                response = self.handle_request(request)
            self.stdout.write(_compact(response) + "\n")
            self.stdout.flush()
        return 0


def main() -> int:
    server = LeannMCPServer()
    hello = {
        "jsonrpc": "2.0",
        "method": "server.ready",
        "params": {"name": "leann-mcp", "version": "1.0", "ts_utc": _utc_now()},
    }
    sys.stdout.write(_compact(hello) + "\n")
    sys.stdout.flush()
    return server.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
