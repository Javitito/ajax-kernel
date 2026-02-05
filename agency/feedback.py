# agency/feedback.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging
import time
from PIL import Image

log = logging.getLogger(__name__)

ImageLike = Image.Image  # luego se sustituye


class FeedbackError(Exception):
    """Base error for feedback subsystem."""


@dataclass
class FeedbackResult:
    changed: bool
    detail: str | None = None
    metadata: dict[str, Any] | None = None


class Feedback:
    """
    Propiocepción: compara estados visuales y guarda artifacts para self-audit.
    Incluye hueco para hashing diferencial (Motor Memory).
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    # --------- Comparación visual básica --------- 
    def compare(self, before: ImageLike, after: ImageLike) -> FeedbackResult:
        """
        MVP: stub donde luego Codex implementará pixel diff/histograma/etc.
        """
        log.info("Feedback.compare called (stub)")
        # TODO: implementar diff real; por ahora finge "cambio" para no bloquear pruebas
        return FeedbackResult(changed=True, detail="stub-compare", metadata={})

    # --------- Visual Hash (Propiocepción diferencial) --------- 
    def hash_region(self, image: ImageLike, region: tuple[int, int, int, int]) -> str:
        """
        Calcula hash rápido (p.ej. perceptual hash) de una región.
        - region: (x, y, width, height)
        """
        log.debug("Feedback.hash_region region=%s", region)
        # TODO: recortar región, convertir a pHash/aHash/dHash, devolver hex
        raise NotImplementedError("Feedback.hash_region not implemented yet")

    def check_visual_hash(self, expected_hash: str, image: ImageLike, region: tuple[int, int, int, int]) -> bool:
        current_hash = self.hash_region(image, region)
        # TODO: quizá tolerancia de Hamming
        ok = (current_hash == expected_hash)
        log.info("Feedback.check_visual_hash expected=%s current=%s ok=%s", expected_hash, current_hash, ok)
        return ok

    # --------- Artifacts (Santuario) --------- 
    def save_artifacts(
        self,
        ctx,
        before: ImageLike,
        after: ImageLike,
        result: FeedbackResult,
    ) -> None:
        """
        Guarda capturas y metadatos en artifacts/actuator/YYYYMMDD-HHMMSS_...
        """
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = (ctx.artifacts_dir / f"{ts}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info("Saving actuator artifacts to %s", run_dir)

        before.save(run_dir / "before.png")
        after.save(run_dir / "after.png")

        # TODO:
        # - volcar metadata/result a JSON
        # Stub temporal:
        (run_dir / "result.txt").write_text(f"goal={ctx.goal}\nchanged={result.changed}\n{result.detail}\n")
