from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from agency.mcp.leann_server import LeannMCPServer


class FakeBackend:
    def __init__(self) -> None:
        self.last_search: Optional[Dict[str, Any]] = None
        self.last_read: Optional[Dict[str, Any]] = None
        self.last_receipts: Optional[int] = None

    def search(self, query: str, k: int = 5, filters: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        self.last_search = {"query": query, "k": k, "filters": dict(filters or {})}
        return {
            "ok": True,
            "query": query,
            "k": k,
            "filters": dict(filters or {}),
            "results": [{"doc_id": "d1", "score": 0.9, "snippet": "abc", "metadata": {}}],
        }

    def read(self, doc_id: str, *, span: Optional[Mapping[str, Any]] = None, range_: Any = None) -> Dict[str, Any]:
        self.last_read = {"doc_id": doc_id, "span": span, "range": range_}
        return {"ok": True, "doc_id": doc_id, "content": "hello", "span": {"start": 0, "end": 5}, "metadata": {}}

    def receipts_latest(self, k: int = 20) -> Dict[str, Any]:
        self.last_receipts = k
        return {"ok": True, "k": k, "results": [{"path": "artifacts/receipts/x.json"}]}


class MissingBackend:
    def search(self, query: str, k: int = 5, filters: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        _ = (query, k, filters)
        return {"ok": False, "error": "capability_missing", "how_to_fix": ["build index"]}

    def read(self, doc_id: str, *, span: Optional[Mapping[str, Any]] = None, range_: Any = None) -> Dict[str, Any]:
        _ = (doc_id, span, range_)
        return {"ok": False, "error": "capability_missing", "how_to_fix": ["build index"]}

    def receipts_latest(self, k: int = 20) -> Dict[str, Any]:
        _ = k
        return {"ok": True, "k": 0, "results": []}


def test_leann_mcp_search_returns_stable_schema() -> None:
    backend = FakeBackend()
    server = LeannMCPServer(backend=backend)  # type: ignore[arg-type]
    req = {"jsonrpc": "2.0", "id": "1", "method": "leann.search", "params": {"query": "x", "k": 3, "filters": {}}}
    resp = server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "1"
    result = resp["result"]
    assert result["ok"] is True
    assert result["query"] == "x"
    assert result["k"] == 3
    assert isinstance(result["results"], list)
    assert result["results"][0]["doc_id"] == "d1"


def test_leann_mcp_read_returns_content() -> None:
    backend = FakeBackend()
    server = LeannMCPServer(backend=backend)  # type: ignore[arg-type]
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "leann.read",
        "params": {"doc_id": "d1", "span": {"start": 0, "end": 5}},
    }
    resp = server.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 7
    result = resp["result"]
    assert result["ok"] is True
    assert result["doc_id"] == "d1"
    assert result["content"] == "hello"


def test_leann_mcp_capability_missing_when_backend_unavailable() -> None:
    server = LeannMCPServer(backend=MissingBackend())  # type: ignore[arg-type]
    req = {"jsonrpc": "2.0", "id": "x", "method": "leann.search", "params": {"query": "any", "k": 2}}
    resp = server.handle_request(req)
    result = resp["result"]
    assert result["ok"] is False
    assert result["error"] == "capability_missing"
    assert isinstance(result["how_to_fix"], list)
