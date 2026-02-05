import re


def windows_to_wsl_path(path_str: str) -> str:
    """
    Convierte rutas tipo C:\\Users\\Javi\\AJAX_HOME\\... en /mnt/c/Users/Javi/AJAX_HOME/...
    Si la ruta ya es POSIX o relativa, la devuelve tal cual.
    """
    if not path_str:
        return path_str

    p = str(path_str).replace("\\", "/")

    if p.startswith("/"):
        return p

    match = re.match(r"^([A-Za-z]):/(.*)$", p)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2)
        return f"/mnt/{drive}/{rest}"

    return p
