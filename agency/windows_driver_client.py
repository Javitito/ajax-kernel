"""
Cliente ligero para hablar con el driver Windows (FastAPI en 127.0.0.1:5010).
Pensado como backend del Actuator en entornos win32.
"""
from __future__ import annotations

import json
import os
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List
import re
import platform

import logging

from agency.driver_keys import load_ajax_driver_api_key, ajax_driver_key_paths

log = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("WindowsDriverClient requiere el paquete 'requests'") from exc


DEFAULT_BASE_URL = "http://127.0.0.1:5010"
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_KEY_PATHS = [str(path) for path in ajax_driver_key_paths()]

FORBIDDEN_APPS = [
    "powershell.exe",
    "cmd.exe",
    "windowsterminal.exe",
    "python.exe",
    "python3.exe",
    "code.exe",
    "pycharm64.exe",
]
FORBIDDEN_APPS_NORMALIZED = {app.lower() for app in FORBIDDEN_APPS}


class WindowsDriverError(RuntimeError):
    """Errores de comunicación con el driver Windows."""


class DriverTimeout(WindowsDriverError):
    """Timeout al hablar con el driver Windows."""


class DriverConnectionError(WindowsDriverError):
    """Errores de conexión (HTTP/Socket) con el driver Windows."""


class WindowInspectionError(WindowsDriverError):
    """Errores al inspeccionar la ventana activa."""


class SafetyError(WindowsDriverError):
    """Errores de política de seguridad en el driver Windows."""


def _load_api_key(candidates: Optional[list[str]] = None) -> Optional[str]:
    if candidates is None:
        return load_ajax_driver_api_key()
    for raw in candidates:
        path = Path(os.path.expanduser(raw))
        try:
            data = path.read_text(encoding="utf-8").strip()
            if data:
                return data
        except Exception:
            continue
    env_key = os.environ.get("AJAX_API_KEY")
    return env_key.strip() if env_key else None


class WindowsDriverClient:
    """
    Cliente HTTP sin dependencias especiales (solo requests) para el driver Windows.
    """

    def is_available(self) -> bool:
        """
        Señal rápida de disponibilidad. Version mínima: mejor esfuerzo con health check.
        """
        try:
            resp = self.health()
            return bool((resp or {}).get("ok", True))
        except Exception:
            return False

    def _autodetect_host_base_url(self) -> Optional[str]:
        """
        En WSL, 127.0.0.1 apunta a la VM. Detectamos la IP del host Windows
        leyendo primero el gateway por defecto y, como respaldo, el primer
        nameserver de /etc/resolv.conf.
        """
        try:
            if platform.system().lower() != "linux":
                return None
            # 1) Gateway por defecto (suele ser el host en WSL2)
            import subprocess
            gw = None
            try:
                out = subprocess.check_output(["sh", "-c", "ip route show default | awk '{print $3}'"], timeout=1)
                gw = out.decode().strip()
            except Exception:
                gw = None
            if gw and gw not in {"127.0.0.1", "0.0.0.0"}:
                return f"http://{gw}:5010"
        except Exception:
            pass
        try:
            resolv = Path("/etc/resolv.conf")
            if not resolv.exists():
                return None
            for line in resolv.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "nameserver":
                    host = parts[1].strip()
                    if host and host not in {"127.0.0.1", "0.0.0.0", "1.1.1.1", "8.8.8.8"}:
                        return f"http://{host}:5010"
        except Exception:
            return None
        return None

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = None,
        prefer_env: bool = True,
    ) -> None:
        # Resolución de prioridad (explícitos primero, autodetección al final)
        if prefer_env:
            env_url = os.getenv("OS_DRIVER_URL")
            if env_url:
                resolved = env_url
            else:
                # archivo de usuario (~/.leann/os_driver_url)
                cfg_path = Path(os.path.expanduser("~/.leann/os_driver_url"))
                if cfg_path.exists():
                    try:
                        cfg_val = cfg_path.read_text(encoding="utf-8").strip()
                        resolved = cfg_val or base_url
                    except Exception:
                        resolved = base_url
                else:
                    env_host = os.getenv("OS_DRIVER_HOST")
                    env_port = os.getenv("OS_DRIVER_PORT") or "5010"
                    if env_host:
                        resolved = f"http://{env_host}:{env_port}"
                    else:
                        detected = self._autodetect_host_base_url()
                        resolved = detected or base_url or DEFAULT_BASE_URL
        else:
            resolved = base_url or DEFAULT_BASE_URL
        self._base_url = str(resolved).rstrip("/")
        self._session = requests.Session()
        self._api_key = api_key if api_key is not None else _load_api_key()
        try:
            self._timeout_s = float(timeout_s if timeout_s is not None else os.getenv("OS_DRIVER_TIMEOUT", DEFAULT_TIMEOUT_S))
        except Exception:
            self._timeout_s = DEFAULT_TIMEOUT_S

    # -----------------------
    # HTTP helpers
    # -----------------------
    def _headers(self) -> Dict[str, str]:
        if self._api_key:
            return {"X-AJAX-KEY": self._api_key}
        return {}

    def _request_json(
        self,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            resp = self._session.request(
                method=method.upper(),
                url=url,
                headers=self._headers(),
                json=json_body,
                params=params,
                timeout=self._timeout_s,
            )
        except requests.Timeout as exc:  # type: ignore
            raise DriverTimeout(f"timeout {method} {url}: {exc}") from exc
        except requests.RequestException as exc:  # type: ignore
            raise DriverConnectionError(f"request_failed {method} {url}: {exc}") from exc
        except Exception as exc:
            raise WindowsDriverError(f"request_failed {method} {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise WindowsDriverError(f"driver_http_{resp.status_code}: {resp.text}")
        try:
            return resp.json()
        except Exception:
            raise WindowsDriverError(f"driver_response_not_json: {resp.text[:200]!r}")

    # -----------------------
    # Driver wrappers
    # -----------------------
    def health(self) -> Dict[str, Any]:
        return self._request_json("GET", "/health")

    def capabilities(self) -> Dict[str, Any]:
        return self._request_json("GET", "/capabilities")

    def displays(self) -> Dict[str, Any]:
        return self._request_json("GET", "/displays")

    def get_active_window(self) -> Dict[str, Any]:
        """
        Devuelve la ventana en foreground según el driver (endpoint /health).
        """
        data = self.health()
        if not data.get("ok", True):
            raise WindowsDriverError(data.get("error") or "driver_health_failed")
        fg_window = data.get("fg_window") or {}
        if not isinstance(fg_window, dict):
            raise WindowsDriverError("driver_health_missing_fg_window")
        return fg_window

    def screenshot(self, display_id: Optional[int] = None) -> Path:
        params = None
        if display_id is not None:
            params = {"display_id": int(display_id)}
        data = self._request_json("GET", "/screenshot", params=params)
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "screenshot_failed")
        path_str = data.get("path") or ""
        if not path_str:
            raise WindowsDriverError("driver_returned_no_path")
        return Path(_normalize_driver_path(path_str))

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> None:
        """
        Nota: el backend Windows usa /mouse/click; aquí hacemos un click en el punto,
        alineado con ClickBody de skills/os_mouse.py.
        """
        payload = {"x": int(x), "y": int(y), "button": "left"}
        data = self._request_json("POST", "/mouse/click", payload)
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "mouse_click_failed")

    def mouse_click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left") -> None:
        payload: Dict[str, Any] = {"button": button}
        if x is not None:
            payload["x"] = int(x)
        if y is not None:
            payload["y"] = int(y)
        data = self._request_json("POST", "/mouse/click", payload)
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "mouse_click_failed")

    def app_launch(self, process: Optional[str] = None, path: Optional[str] = None, args: Optional[list[str]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if process:
            payload["process"] = process
        if path:
            payload["path"] = path
        if args:
            payload["args"] = args
        data = self._request_json("POST", "/app/launch", payload)
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "app_launch_failed")
        return data

    def keyboard_type(self, text: str, submit: bool = False) -> Dict[str, Any]:
        """
        Envía texto al driver; devuelve el payload de respuesta.
        """
        data = self._request_json("POST", "/keyboard/type", {"text": text, "submit": bool(submit)})
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "keyboard_type_failed")
        return data

    def keyboard_paste(self, text: str, submit: bool = False) -> Dict[str, Any]:
        data = self._request_json("POST", "/keyboard/paste", {"text": text, "submit": bool(submit)})
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "keyboard_paste_failed")
        return data

    def window_focus(self, process: Optional[str] = None, title_contains: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"match": {"process": process, "title_contains": title_contains}}
        if process:
            payload["process"] = process
        if title_contains:
            payload["title_contains"] = title_contains
        data = self._request_json("POST", "/window/focus", payload)
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "window_focus_failed")
        return data

    def window_move(
        self,
        *,
        process: Optional[str] = None,
        title_contains: Optional[str] = None,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"match": {"process": process, "title_contains": title_contains}}
        if process:
            payload["process"] = process
        if title_contains:
            payload["title_contains"] = title_contains
        payload.update({"x": int(x), "y": int(y), "width": int(width), "height": int(height)})
        data = self._request_json("POST", "/window/move", payload)
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "window_move_failed")
        return data

    def keyboard_hotkey(self, *keys: str, ignore_safety: bool = False) -> Dict[str, Any]:
        if not ignore_safety:
            fg_window = self.get_active_window()
            active_process = str((fg_window or {}).get("process") or "").lower()

            if active_process and active_process in FORBIDDEN_APPS_NORMALIZED:
                raise SafetyError(f"hotkey_blocked_fg_process:{active_process}")

            normalized_keys = [str(k).lower() for k in keys]
            if "alt" in normalized_keys and "f4" in normalized_keys and active_process != "notepad.exe":
                raise SafetyError("hotkey_blocked_alt_f4_outside_notepad")

        data = self._request_json("POST", "/keyboard/hotkey", {"keys": list(keys)})
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "keyboard_hotkey_failed")
        return data

    def inspect_window(self) -> Dict[str, Any]:
        data = self._request_json("GET", "/window/inspect")
        if not data.get("ok"):
            raise WindowInspectionError(data.get("error") or "inspect_window_failed")
        return data

    def find_control(self, *, name: str | None = None, control_type: str | None = None) -> Dict[str, Any] | None:
        """
        Usa inspect_window() y devuelve el primer control que cumpla con name/control_type
        (contains, case-insensitive) en el arbol de hijos de la ventana activa.
        """
        data = self.inspect_window()
        fg_window = data.get("fg_window") or {}
        queue = list((fg_window.get("children") or []))

        def _matches(node: Dict[str, Any]) -> bool:
            if name:
                if name.lower() not in str(node.get("name") or "").lower():
                    return False
            if control_type:
                if control_type.lower() not in str(node.get("control_type") or "").lower():
                    return False
            return True

        while queue:
            node = queue.pop(0)
            if isinstance(node, dict) and _matches(node):
                return node
            children = node.get("children") if isinstance(node, dict) else None
            if isinstance(children, list):
                queue.extend(children)
        return None

    def find_text(self, text: str, app: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if app:
            payload["app"] = app
        data = self._request_json("POST", "/vision/find_text", payload)
        if not data.get("ok"):
            raise WindowsDriverError(data.get("error") or "find_text_failed")
        return data

    def run_sequence(
        self,
        ops: List[Dict[str, Any]],
        evidence: Optional[Dict[str, Any]] = None,
        mode: str = "probe",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"ops": ops, "mode": mode}
        if evidence:
            payload["evidence"] = evidence
        data = self._request_json("POST", "/sequence", payload)
        if not isinstance(data, dict):
            raise WindowsDriverError("sequence_invalid_response")
        return data

    def run_save_notepad_v2_path(self, filepath: str | None) -> Dict[str, Any]:
        """
        Guardado simple: Ctrl+S y escribe la ruta objetivo, con Enter al final.
        """
        if not filepath:
            return {"ok": False, "error": "filepath_required"}
        expanded = os.path.expandvars(os.path.expanduser(filepath))
        fs_path = _windows_to_wsl_path(expanded)
        target = Path(fs_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            try:
                self.window_focus(process="notepad.exe")
                time.sleep(0.2)
            except Exception:
                pass
            self.keyboard_hotkey("ctrl", "s")
            time.sleep(0.3)
            driver_path = _wsl_to_windows_path(str(target))
            res = self.keyboard_paste(text=str(driver_path), submit=True)
            return {"ok": True, "path": str(target), "driver_path": str(driver_path), "type": res}
        except (WindowsDriverError, SafetyError) as exc:
            return {"ok": False, "error": str(exc)}

    def append_diary_entry(self, text: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Mueve el cursor al final, inserta nueva línea y pega el texto vía portapapeles.
        """
        try:
            line = text or self._default_diary_line()
            if file_path:
                expanded = Path(os.path.expandvars(os.path.expanduser(file_path)))
                try:
                    expanded.parent.mkdir(parents=True, exist_ok=True)
                    with expanded.open("a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except Exception as exc:
                    # Log en respuesta pero no abortar UI
                    fs_err = str(exc)
                else:
                    fs_err = None
            else:
                fs_err = None
            try:
                self.window_focus(process="notepad.exe")
                time.sleep(0.2)
            except Exception:
                pass
            self.keyboard_hotkey("ctrl", "end", ignore_safety=True)
            time.sleep(0.1)
            self.keyboard_hotkey("enter", ignore_safety=True)
            time.sleep(0.1)
            res = self.keyboard_paste(line, submit=False)
            result: Dict[str, Any] = {"ok": True, "line": line, "paste": res}
            if file_path:
                result["file_path"] = str(Path(file_path))
            if fs_err:
                result["fs_error"] = fs_err
            return result
        except (WindowsDriverError, SafetyError) as exc:
            return {"ok": False, "error": str(exc)}

    def _default_diary_line(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts}] Sesión de entrenamiento completada con éxito"

    def run_close_notepad(self) -> Dict[str, Any]:
        """
        Cierre defensivo: intenta menú Archivo->Salir con ratón, luego botón de cierre, luego Alt+F4/Enter y, si falla, taskkill.
        """
        try:
            try:
                self.window_focus(process="notepad.exe")
                time.sleep(0.3)
            except Exception:
                pass

            last_fg: Dict[str, Any] | None = None

            # 0) Intento con menú Archivo/Salir vía ratón (si UIA devuelve controles)
            try:
                if self._try_close_via_menu():
                    return {"ok": True, "method": "menu_click", "fg": self._fg_info()}
            except Exception:
                pass

            # 1) Intento con click en botón de cierre (top-right de la ventana)
            try:
                fg = self.get_active_window()
                rect = fg.get("rect") if isinstance(fg, dict) else None
                if rect:
                    x = int(rect.get("x", 0) + rect.get("width", 0) - 10)
                    y = int(rect.get("y", 0) + 10)
                    self.mouse_click(x=x, y=y)
                    time.sleep(0.5)
                    last_fg = self._fg_info()
                    if not self._fg_is_notepad(last_fg):
                        return {"ok": True, "method": "mouse_click", "fg": last_fg}
            except Exception:
                pass

            # 2) Hotkeys de cierre
            for attempt in range(3):
                self.keyboard_hotkey("alt", "f4", ignore_safety=True)
                time.sleep(0.4)
                try:
                    self.keyboard_hotkey("enter", ignore_safety=True)
                except Exception:
                    pass
                time.sleep(0.4)
                last_fg = self._fg_info()
                if not self._fg_is_notepad(last_fg):
                    return {"ok": True, "attempts": attempt + 1, "fg": last_fg}
            # Fallback: intentamos cerrar por la fuerza si sigue en foreground
            try:
                subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"], check=True, capture_output=True)
                time.sleep(0.5)
                last_fg = self._fg_info()
                if not self._fg_is_notepad(last_fg):
                    return {"ok": True, "attempts": 3, "fg": last_fg, "method": "taskkill"}
            except Exception as exc:
                return {"ok": False, "error": f"notepad_still_foreground:{exc}", "fg": last_fg}
            return {"ok": False, "error": "notepad_still_foreground", "fg": last_fg}
        except (WindowsDriverError, SafetyError) as exc:
            return {"ok": False, "error": str(exc)}

    def _fg_info(self) -> Dict[str, Any] | None:
        try:
            data = self.health()
            return data.get("fg_window") if isinstance(data, dict) else None
        except Exception:
            return None

    def _fg_is_notepad(self, fg: Dict[str, Any] | None) -> bool:
        if not fg:
            return True  # sin info, preferimos no asumir cierre
        proc = str((fg.get("process") or "")).lower()
        title = str((fg.get("title") or "")).lower()
        if "notepad" in proc:
            return True
        if "notepad" in title or "bloc de notas" in title:
            return True
        return False

    def _click_rect(self, rect: Dict[str, Any] | None) -> bool:
        if not rect or any(k not in rect for k in ("x", "y", "width", "height")):
            return False
        try:
            x = int(rect["x"] + rect["width"] / 2)
            y = int(rect["y"] + rect["height"] / 2)
            self.mouse_click(x=x, y=y)
            return True
        except Exception:
            return False

    def _try_close_via_menu(self) -> bool:
        menu_names = ["Archivo", "File"]
        exit_names = ["Salir", "Exit", "Close", "Cerrar"]
        for mn in menu_names:
            ctrl = self.find_control(name=mn, control_type="MenuItem")
            if not ctrl:
                continue
            if not self._click_rect(ctrl.get("rect") if isinstance(ctrl, dict) else None):
                continue
            time.sleep(0.3)
            for en in exit_names:
                exit_ctrl = self.find_control(name=en, control_type="MenuItem")
                if not exit_ctrl:
                    continue
                if self._click_rect(exit_ctrl.get("rect") if isinstance(exit_ctrl, dict) else None):
                    time.sleep(0.5)
                    if not self._fg_is_notepad(self._fg_info()):
                        return True
        return False

    def tag_screen_grid(self, rows: int = 4, cols: int = 4) -> Dict[str, Any]:
        """
        Captura la pantalla, dibuja una rejilla rows x cols con IDs en cada celda
        y guarda tanto la imagen como el JSON de marcas en artifacts/vision/.
        """
        if rows <= 0 or cols <= 0:
            raise WindowsDriverError("tag_screen_grid_invalid_dimensions")

        try:  # Import perezoso para no añadir dependencia obligatoria si no se usa
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except Exception as exc:  # pragma: no cover - dependencia opcional
            raise WindowsDriverError(f"pillow_not_available:{exc}") from exc

        snap_path = Path(_normalize_driver_path(str(self.screenshot())))
        try:
            with Image.open(snap_path) as img:
                image = img.convert("RGB")
        except Exception as exc:
            raise WindowsDriverError(f"cannot_open_screenshot:{snap_path}: {exc}") from exc

        # Si hay ventana foreground y rect válido, recorta la captura a esa ventana
        offset_x = 0
        offset_y = 0
        try:
            fg = self.get_active_window()
            rect = fg.get("rect") if isinstance(fg, dict) else None
            if rect:
                x0 = max(0, int(rect.get("x", 0)))
                y0 = max(0, int(rect.get("y", 0)))
                x1 = int(x0 + rect.get("width", 0))
                y1 = int(y0 + rect.get("height", 0))
                x1 = min(x1, image.width)
                y1 = min(y1, image.height)
                if x1 - x0 > 10 and y1 - y0 > 10:
                    image = image.crop((x0, y0, x1, y1))
                    offset_x, offset_y = x0, y0
        except Exception:
            pass

        width, height = image.size
        cell_w = width / cols
        cell_h = height / rows
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        marks: List[Dict[str, Any]] = []
        for row_idx in range(rows):
            for col_idx in range(cols):
                x0 = int(round(col_idx * cell_w))
                y0 = int(round(row_idx * cell_h))
                x1 = int(round((col_idx + 1) * cell_w))
                y1 = int(round((row_idx + 1) * cell_h))
                rect_w = max(1, x1 - x0)
                rect_h = max(1, y1 - y0)

                cell_id = f"{_col_label(col_idx)}{row_idx + 1}"
                screen_bbox = [x0 + offset_x, y0 + offset_y, x1 + offset_x, y1 + offset_y]
                marks.append(
                    {
                        "id": cell_id,
                        "rect": {"x": x0, "y": y0, "width": rect_w, "height": rect_h},
                        "screen_bbox": screen_bbox,
                    }
                )

                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                text_w, text_h = _measure_text(draw, cell_id, font)
                text_x = int(x0 + (rect_w - text_w) / 2)
                text_y = int(y0 + (rect_h - text_h) / 2)
                padding = 2
                draw.rectangle(
                    [text_x - padding, text_y - padding, text_x + text_w + padding, text_y + text_h + padding],
                    fill="white",
                )
                draw.text((text_x, text_y), cell_id, fill="red", font=font)

        ts = int(time.time())
        out_dir = Path.cwd() / "artifacts" / "vision"
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"tagged_{ts}.png"
        json_path = out_dir / f"tagged_{ts}.json"

        image.save(png_path)
        result = {
            "image_path": str(png_path),
            "marks": marks,
            "screen_offset": {"x": offset_x, "y": offset_y},
        }
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def _normalize_driver_path(path_str: str) -> str:
    """
    Convierte rutas C:\\Users\\... a /mnt/c/... cuando se ejecuta en WSL y el
    original no existe. Si el original existe, lo mantiene.
    """
    if not path_str:
        return path_str
    if os.path.exists(path_str):
        return path_str

    p = path_str.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", p)
    if match:
        candidate = f"/mnt/{match.group(1).lower()}/{match.group(2)}"
        if os.path.exists(candidate):
            return candidate
    return path_str


def _windows_to_wsl_path(path_str: str) -> str:
    if not path_str:
        return path_str
    normalized = path_str.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    return path_str


def _wsl_to_windows_path(path_str: str) -> str:
    if not path_str:
        return path_str
    if path_str.startswith("/mnt/"):
        parts = path_str.split("/")
        if len(parts) >= 4:
            drive = parts[2]
            rest = "\\".join(parts[3:])
            return f"{drive.upper()}:\\{rest}"
    return path_str


def _col_label(idx: int) -> str:
    """
    Convierte un índice de columna en etiqueta estilo Excel: 0->A, 1->B, ..., 25->Z, 26->AA, etc.
    """
    if idx < 0:
        return "X"
    letters = []
    n = idx
    while n >= 0:
        n, rem = divmod(n, 26)
        letters.append(chr(ord("A") + rem))
        n -= 1
    return "".join(reversed(letters))


def _measure_text(draw: Any, text: str, font: Any) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:  # pragma: no cover - compatibilidad con versiones antiguas de Pillow
        return draw.textsize(text, font=font)
