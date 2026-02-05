"""
UIA-based inspection utilities.

Propiocepción del SO: obtiene controles de la ventana activa o por título (regex)
sin depender de OCR/visión. Devuelve bounding boxes y metadatos básicos.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import uiautomation as auto  # type: ignore
except Exception as exc:  # pragma: no cover - dependencia opcional
    auto = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

try:
    from agency.windows_driver_client import WindowsDriverClient
except Exception:
    WindowsDriverClient = None  # type: ignore


def _rect_to_list(rect: Any) -> List[int]:
    try:
        return [int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)]
    except Exception:
        return [0, 0, 0, 0]


def inspect_ui_tree(title_regex: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspecciona la ventana activa (o una que coincida con title_regex) y devuelve
    los controles hijos con nombre, tipo y bounding box.
    """
    if auto is None:
        # Fallback: usa el driver Windows si está disponible
        if WindowsDriverClient is None:
            return {"ok": False, "error": f"uiautomation_not_available:{_import_error}"}
        try:
            client = WindowsDriverClient()
            if title_regex:
                try:
                    client.window_focus(title_contains=title_regex)
                except Exception:
                    pass
            res = client.inspect_window()
            if not res.get("ok"):
                return {"ok": False, "error": res.get("error") or "inspect_window_failed"}
            fg = res.get("fg_window") or {}
            children = fg.get("children") or []
            elements: List[Dict[str, Any]] = []
            for child in children:
                rect = child.get("rect") or {}
                elements.append(
                    {
                        "name": child.get("name"),
                        "control_type": child.get("control_type"),
                        "rect": [
                            int(rect.get("x", 0)),
                            int(rect.get("y", 0)),
                            int(rect.get("x", 0) + rect.get("width", 0)),
                            int(rect.get("y", 0) + rect.get("height", 0)),
                        ],
                        "enabled": child.get("enabled"),
                    }
                )
            return {"ok": True, "title": fg.get("title"), "process": fg.get("process"), "elements": elements}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "error": f"uia_fallback_failed:{exc}"}

    try:
        window = auto.WindowControl(searchDepth=1, RegexName=title_regex) if title_regex else auto.GetForegroundControl()
        if not window or not window.Exists(0, 1):
            return {"ok": False, "error": "window_not_found"}

        try:
            window.SetFocus()
        except Exception:
            pass

        elements: List[Dict[str, Any]] = []
        for child in window.GetChildren():
            rect_list = _rect_to_list(getattr(child, "BoundingRectangle", None))
            elements.append(
                {
                    "name": getattr(child, "Name", None),
                    "control_type": getattr(child, "ControlTypeName", None),
                    "rect": rect_list,
                    "enabled": getattr(child, "IsEnabled", None),
                }
            )

        return {"ok": True, "title": getattr(window, "Name", None), "elements": elements}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}
