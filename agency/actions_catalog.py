from __future__ import annotations

import json
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ActionSpec:
    name: str
    description: str
    args_schema: Dict[str, Any]
    risk_level: str
    category: Optional[str] = None
    vision_required: bool = False
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionSpec":
        name = data["name"]
        category = data.get("category")
        raw_tags = data.get("tags") or []
        tags: List[str] = []
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                if isinstance(tag, str) and tag.strip():
                    tags.append(tag.strip().lower())
        vr = data.get("vision_required")
        vision_required: Optional[bool]
        if isinstance(vr, bool):
            vision_required = vr
        elif isinstance(vr, str) and vr.strip().lower() in {"1", "true", "yes", "on"}:
            vision_required = True
        elif isinstance(vr, str) and vr.strip().lower() in {"0", "false", "no", "off"}:
            vision_required = False
        else:
            cat_n = str(category or "").strip().lower()
            low = str(name).strip().lower()
            if low.startswith("vision.") or cat_n == "vision":
                vision_required = True
            elif low.startswith("keyboard.") or low in {"app.launch", "window.focus", "desktop.isolate_active_window", "probe_driver"}:
                vision_required = False
            else:
                uiish = cat_n in {"input", "window_management", "desktop", "app", "ui", "mouse"}
                uiish = uiish or low.startswith(("mouse.", "ui.", "window.", "app.", "desktop."))
                vision_required = True if uiish else False
        return cls(
            name=name,
            description=data.get("description", ""),
            args_schema=data.get("args", {}) or data.get("args_schema", {}),
            risk_level=data.get("risk_level") or data.get("risk") or "medium",
            category=category if isinstance(category, str) and category.strip() else None,
            vision_required=bool(vision_required),
            notes=data.get("notes"),
            tags=_derive_action_tags(name=name, category=category, raw_tags=tags),
        )


def _derive_action_tags(*, name: str, category: Optional[str], raw_tags: List[str]) -> List[str]:
    tags = list(raw_tags or [])
    cat = str(category or "").strip().lower()
    low = str(name or "").strip().lower()
    if cat in {"input"}:
        tags.append("input")
    if cat in {"app", "window_management", "desktop", "ui"}:
        tags.append("ui_basic")
    if cat in {"web", "browser", "web_nav"}:
        tags.append("web_nav")
    if low.startswith("keyboard."):
        tags.append("input")
    if low.startswith(("app.", "window.", "desktop.")):
        tags.append("ui_basic")
    if "browser" in low:
        tags.append("web_nav")
    cleaned: List[str] = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        t = tag.strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    return cleaned


class ActionCatalog:
    """
    Catálogo de acciones constitucionales cargadas desde config/ajax_actions.json.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.actions: Dict[str, ActionSpec] = {}
        self._load()

    def _load(self) -> None:
        cfg_path = self.root_dir / "config" / "ajax_actions.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No se encuentra {cfg_path}")
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            items = [i for i in data if isinstance(i, dict)]
        elif isinstance(data, dict):
            items = [i for i in data.get("actions", []) if isinstance(i, dict)]
        else:
            raise ValueError("ajax_actions.json debe ser una lista o un objeto con clave 'actions'")

        for item in items:
            spec = ActionSpec.from_dict(item)
            self.actions[spec.name] = spec
        if not self.actions:
            logging.getLogger("ajax.actions_catalog").warning("ActionCatalog vacío tras cargar ajax_actions.json")

    def list_actions(self) -> List[ActionSpec]:
        return list(self.actions.values())

    def get(self, name: str) -> Optional[ActionSpec]:
        return self.actions.get(name)

    def is_allowed(self, name: str) -> bool:
        return name in self.actions

    def requires_vision(self, name: str) -> bool:
        spec = self.actions.get(name)
        if spec is None:
            # Fail-closed: acciones desconocidas se consideran vision-required (no ejecutar sin confirmación explícita).
            return True
        return bool(spec.vision_required)

    def to_brain_payload(self) -> Dict[str, Any]:
        return {
            "actions": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "args_schema": spec.args_schema,
                    "risk_level": spec.risk_level,
                    "category": spec.category,
                    "vision_required": bool(spec.vision_required),
                    "tags": list(spec.tags or []),
                }
                for spec in self.list_actions()
            ]
        }
