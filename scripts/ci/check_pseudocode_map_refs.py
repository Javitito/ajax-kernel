#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set
import re

REF_PATTERN = re.compile(r"PSEUDOCODE_MAP/[A-Za-z0-9_.-]+\.pseudo\.md")
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "artifacts",
    ".leann",
    "tools/third_party",
}


def _git_tracked_files(root: Path) -> List[Path]:
    cmd = ["git", "-C", str(root), "ls-files"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    out: List[Path] = []
    for line in (proc.stdout or "").splitlines():
        text = line.strip()
        if text:
            out.append(root / text)
    return out


def _walk_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS):
            continue
        files.append(path)
    return files


def _candidate_files(root: Path) -> List[Path]:
    tracked = _git_tracked_files(root)
    if tracked:
        return [p for p in tracked if p.is_file()]
    return _walk_files(root)


def _extract_refs(path: Path) -> Set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    return set(REF_PATTERN.findall(text))


def check_refs(root: Path) -> dict:
    refs: Set[str] = set()
    for file_path in _candidate_files(root):
        refs.update(_extract_refs(file_path))

    missing = sorted(ref for ref in refs if not (root / ref).exists())
    return {
        "schema": "ajax.ci.pseudocode_map_refs.v1",
        "root": str(root.resolve()),
        "refs_count": len(refs),
        "missing_count": len(missing),
        "refs": sorted(refs),
        "missing": missing,
        "ok": len(missing) == 0,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate PSEUDOCODE_MAP/*.pseudo.md references exist")
    parser.add_argument("--root", default=".", help="Repository root (default: .)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root).resolve()
    result = check_refs(root)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
