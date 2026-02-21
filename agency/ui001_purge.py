from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _discover_matches(root: Path) -> List[Path]:
    matches: Set[Path] = set()
    attic_root = root / "artifacts" / "attic"

    def _find_files(base: Path, pattern: str) -> List[Path]:
        if not base.exists():
            return []
        try:
            proc = subprocess.run(
                ["find", str(base), "-type", "f", "-name", pattern, "-print0"],
                check=False,
                capture_output=True,
            )
            if proc.returncode != 0:
                return []
            out = proc.stdout.decode("utf-8", errors="ignore")
            rows = [row for row in out.split("\x00") if row]
            return [Path(row) for row in rows]
        except Exception:
            return []

    for path in _find_files(root / "artifacts" / "gaps" / "UI-001", "*"):
        if path.is_file():
            matches.add(path.resolve())

    for path in _find_files(root / "artifacts", "*UI-001*"):
        if not path.is_file():
            continue
        if attic_root in path.parents:
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel.startswith("artifacts/reports/purge_UI-001_"):
            continue
        if rel.startswith("artifacts/receipts/purge_UI-001_"):
            continue
        matches.add(path.resolve())

    for path in _find_files(root / "gaps" / "UI-001", "*"):
        if path.is_file():
            matches.add(path.resolve())

    return sorted(matches, key=lambda p: str(p))


def purge_ui001_gaps(root_dir: Path, *, timestamp: Optional[str] = None, with_hash: bool = True) -> Dict[str, Any]:
    root = Path(root_dir).resolve()
    ts = timestamp or _utc_compact()
    attic_rel = Path("artifacts") / "attic" / f"purge_UI-001_{ts}"
    attic_root = root / attic_rel
    report_rel = Path("artifacts") / "reports" / f"purge_UI-001_{ts}.md"
    report_path = root / report_rel
    receipt_rel = Path("artifacts") / "receipts" / f"purge_UI-001_{ts}.json"
    receipt_path = root / receipt_rel

    files = _discover_matches(root)
    attic_root.mkdir(parents=True, exist_ok=True)

    moved_rows: List[Dict[str, Any]] = []
    total_bytes = 0
    for src in files:
        if not src.exists() or not src.is_file():
            continue
        try:
            rel_src = src.relative_to(root)
        except ValueError:
            continue
        size_bytes = int(src.stat().st_size)
        sha256_hex = _sha256_file(src) if with_hash else None
        dst = attic_root / rel_src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        total_bytes += size_bytes
        row = {
            "source_path": str(rel_src).replace("\\", "/"),
            "attic_path": str(dst.relative_to(root)).replace("\\", "/"),
            "size_bytes": size_bytes,
        }
        if sha256_hex:
            row["sha256"] = sha256_hex
        moved_rows.append(row)

    reindexed_only = False
    if not moved_rows and attic_root.exists():
        existing_rows: List[Dict[str, Any]] = []
        for attic_file in sorted(attic_root.rglob("*"), key=lambda p: str(p)):
            if not attic_file.is_file():
                continue
            if attic_file.name == "INDEX.json":
                continue
            rel_attic = attic_file.relative_to(root)
            rel_inside = attic_file.relative_to(attic_root)
            source_path = str(rel_inside).replace("\\", "/")
            size_bytes = int(attic_file.stat().st_size)
            row = {
                "source_path": source_path,
                "attic_path": str(rel_attic).replace("\\", "/"),
                "size_bytes": size_bytes,
            }
            if with_hash:
                row["sha256"] = _sha256_file(attic_file)
            existing_rows.append(row)
        if existing_rows:
            moved_rows = existing_rows
            total_bytes = sum(int(row.get("size_bytes") or 0) for row in moved_rows)
            reindexed_only = True

    index_payload: Dict[str, Any] = {
        "schema": "ajax.ui001_purge.index.v1",
        "ts_utc": _utc_now(),
        "root_dir": str(root),
        "attic_dir": str(attic_rel).replace("\\", "/"),
        "source_patterns": [
            "artifacts/gaps/UI-001/**",
            "artifacts/**/UI-001*",
            "gaps/UI-001/**",
        ],
        "reindexed_only": reindexed_only,
        "total_count": len(moved_rows),
        "total_bytes": total_bytes,
        "files": moved_rows,
    }
    index_path = attic_root / "INDEX.json"
    index_path.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# UI-001 purge report",
        "",
        f"- Timestamp (UTC): {_utc_now()}",
        f"- Root: `{root}`",
        f"- Attic: `{attic_rel}`",
        f"- Total moved files: **{len(moved_rows)}**",
        f"- Total bytes moved: **{total_bytes}**",
        "",
    ]
    if reindexed_only:
        lines.append("- Mode: reindexed existing attic content (no new moves in this invocation).")
        lines.append("")
    if moved_rows:
        lines.extend(
            [
                "## Sample",
                "",
                "| Source | Attic | Bytes |",
                "|---|---|---:|",
            ]
        )
        for row in moved_rows[:25]:
            lines.append(
                f"| `{row['source_path']}` | `{row['attic_path']}` | {row['size_bytes']} |"
            )
    else:
        lines.extend(
            [
                "No files matched UI-001 patterns. This run is a no-op.",
                "",
                "Idempotence note: re-running stays safe and will continue to return zero if no new UI-001 files exist.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_payload = {
        "schema": "ajax.ui001_purge.receipt.v1",
        "ts_utc": _utc_now(),
        "ok": True,
        "reindexed_only": reindexed_only,
        "moved_count": len(moved_rows),
        "total_bytes": total_bytes,
        "attic_dir": str(attic_rel).replace("\\", "/"),
        "index_path": str(index_path.relative_to(root)).replace("\\", "/"),
        "report_path": str(report_rel).replace("\\", "/"),
    }
    receipt_path.write_text(json.dumps(receipt_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "ok": True,
        "reindexed_only": reindexed_only,
        "moved_count": len(moved_rows),
        "total_bytes": total_bytes,
        "attic_dir": str(attic_rel).replace("\\", "/"),
        "index_path": str(index_path.relative_to(root)).replace("\\", "/"),
        "report_path": str(report_rel).replace("\\", "/"),
        "receipt_path": str(receipt_rel).replace("\\", "/"),
    }
