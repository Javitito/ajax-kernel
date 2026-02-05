from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from leann.api import LeannSearcher
except ModuleNotFoundError:  # pragma: no cover
    import sys as _sys
    root = Path(__file__).resolve().parents[1]
    vendor = root / "packages" / "leann-core" / "src"
    if str(vendor) not in _sys.path:
        _sys.path.insert(0, str(vendor))
    from leann.api import LeannSearcher


def _to_snippets(hits: Any, source_mode: str) -> List[Dict[str, Any]]:
    snippets: List[Dict[str, Any]] = []
    for hit in hits or []:
        score_val = getattr(hit, "score", None)
        try:
            score_val = float(score_val) if score_val is not None else None
        except Exception:
            score_val = None
        snippets.append(
            {
                "text": getattr(hit, "text", ""),
                "score": score_val,
                "metadata": getattr(hit, "metadata", {}) or {},
                "id": getattr(hit, "id", None),
                "source_mode": source_mode,
            }
        )
    return snippets


def _grep_fallback(idx_base: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Grep fallback usando el archivo de pasajes declarado en el meta (o derivado).
    """
    meta_path = Path(f"{idx_base}.meta.json")
    passages_path = Path(f"{idx_base}.passages.jsonl")
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sources = meta.get("passage_sources") or []
            if sources:
                p = sources[0].get("path") or sources[0].get("path_relative")
                if p:
                    passages_path = Path(p)
                    if not passages_path.is_absolute():
                        passages_path = meta_path.parent / passages_path
    except Exception:
        pass

    if not passages_path.exists():
        return []

    matches: List[Dict[str, Any]] = []
    try:
        import re

        q_lower = query.lower()
        tokens = [t for t in re.split(r"\W+", q_lower) if len(t) >= 3]
        tokens = tokens or [q_lower]
        with passages_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line_l = line.lower()
                if not any(tok in line_l for tok in tokens):
                    continue
                try:
                    data = json.loads(line.strip())
                except Exception:
                    continue
                text = data.get("text", "")
                score = float(sum(text.lower().count(tok) for tok in tokens))
                matches.append(
                    {
                        "text": text,
                        "score": score,
                        "metadata": data.get("metadata", {}) or {},
                        "id": data.get("id"),
                        "source_mode": "grep",
                    }
                )
                if len(matches) >= top_k:
                    break
    except Exception:
        return []
    matches.sort(key=lambda x: x.get("score") or 0, reverse=True)
    return matches[:top_k]


def query_leann(
    collection: str,
    query: str,
    top_k: int = 5,
    *,
    fallback_grep: bool = True,
) -> List[Dict[str, Any]]:
    """
    Consulta un índice LEANN y devuelve snippets con metadata.
    Intenta primero vectorial; si no hay resultados y fallback_grep=True,
    repite con grep y marca source_mode en cada snippet.
    No revienta si el índice no existe; devuelve lista vacía.
    """
    idx_base = collection.replace(".meta.json", "").replace(".passages.jsonl", "")
    try:
        searcher = LeannSearcher(idx_base, enable_warmup=False)
    except Exception:
        return []

    # Vectorial primero
    vector_hits: List[Any] = []
    try:
        vector_hits = searcher.search(query, top_k=top_k, recompute_embeddings=False)
    except Exception:
        vector_hits = []

    if vector_hits:
        return _to_snippets(vector_hits, "vector")

    # Fallback a grep si procede
    if fallback_grep:
        grep_snippets = _grep_fallback(idx_base, query, top_k)
        if grep_snippets:
            return grep_snippets

    return []


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Consulta rápida a LEANN.")
    parser.add_argument("--collection", default="ajax_history_v1.leann", help="Base path del índice LEANN.")
    parser.add_argument("--query", required=True, help="Texto de consulta.")
    parser.add_argument("--top-k", type=int, default=5, help="Número de snippets a devolver.")
    parser.add_argument(
        "--fallback-grep",
        dest="fallback_grep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usa grep si la búsqueda vectorial devuelve 0 resultados (por defecto: True).",
    )
    args = parser.parse_args()

    results = query_leann(args.collection, args.query, top_k=args.top_k, fallback_grep=args.fallback_grep)
    mode = results[0].get("source_mode") if results else "none"
    if mode == "grep":
        try:
            import sys

            print("[FALLBACK: grep]", file=sys.stderr)
        except Exception:
            pass

    payload = {
        "collection": args.collection,
        "query": args.query,
        "top_k": args.top_k,
        "fallback_grep": args.fallback_grep,
        "source_mode": mode,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
