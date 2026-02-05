from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


EXPECTED_PORT_USERS = {5010: "Javi", 5012: "AJAX"}
DEFAULT_PORTS = (5010, 5012)
CORE_TASK_NAME_PATTERNS = (
    "DriverWatchdog",
    "Run_Driver",
    "Guardian_",
    "Lab_Bootstrap",
    "Start_AJAX",
    "Ensure_LMStudio",
    "Lab_Worker",
)


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _powershell_path() -> str:
    unix_path = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if unix_path.exists():
        return str(unix_path)
    if os.name == "nt":
        return r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    return "powershell.exe"


def _schtasks_path() -> str:
    # When running under WSL, the native `schtasks` binary is not present; use schtasks.exe.
    unix_path = Path("/mnt/c/Windows/System32/schtasks.exe")
    if unix_path.exists():
        return str(unix_path)
    if os.name == "nt":
        return r"C:\Windows\System32\schtasks.exe"
    return "schtasks.exe"


def _to_windows_path(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            val = (proc.stdout or "").strip()
            if val:
                return val
    except Exception:
        pass
    return str(path)


def _run(cmd: List[str], *, cwd: Optional[Path] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    # WSL calling Windows `.exe` (schtasks/powershell) often emits text in an OEM codepage (e.g., cp850),
    # which can raise UnicodeDecodeError under the UTF-8 locale. Decode defensively.
    cmd0 = (cmd[0] or "") if cmd else ""
    is_windows_exe = (os.name != "nt") and cmd0.lower().endswith(".exe")
    run_kwargs: Dict[str, Any] = {}
    if is_windows_exe:
        run_kwargs["encoding"] = "cp850"
        run_kwargs["errors"] = "replace"
    else:
        run_kwargs["errors"] = "replace"
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        **run_kwargs,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def _load_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _parse_qwinsta_sessions(text: str, users: Optional[List[str]] = None) -> Dict[str, Optional[int]]:
    sessions: Dict[str, Optional[int]] = {}
    for user in (users or ["Javi", "AJAX"]):
        sessions[user] = None
        for line in text.splitlines():
            if user.lower() not in line.lower():
                continue
            nums = re.findall(r"\d+", line)
            if not nums:
                continue
            try:
                sessions[user] = int(nums[0])
            except Exception:
                sessions[user] = None
            break
    return sessions


def _default_services_config() -> Dict[str, Any]:
    return {
        "schema": "ajax.services_doctor.v1",
        "mode": "multi_user",
        "single_user": "Javi",
        "expected_port_users": {5010: "Javi", 5012: "AJAX"},
        "ports": list(DEFAULT_PORTS),
    }


def _parse_expected_users_env(value: str) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    if not value:
        return parsed
    for chunk in value.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part and ":" not in part:
            continue
        sep = "=" if "=" in part else ":"
        key, user = part.split(sep, 1)
        try:
            port = int(key.strip())
        except Exception:
            continue
        user = user.strip()
        if user:
            parsed[port] = user
    return parsed


def _normalize_expected_users(raw: Any) -> Dict[int, str]:
    expected: Dict[int, str] = {}
    if isinstance(raw, dict):
        for key, val in raw.items():
            try:
                port = int(key)
            except Exception:
                continue
            user = str(val or "").strip()
            if user:
                expected[port] = user
    return expected


def _load_services_doctor_config(root_dir: Path) -> Dict[str, Any]:
    cfg = _default_services_config()
    config_path = root_dir / "config" / "services_doctor.yaml"
    if config_path.exists():
        data: Optional[Dict[str, Any]] = None
        if yaml:
            try:
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                data = loaded if isinstance(loaded, dict) else None
            except Exception:
                data = None
        if data is None:
            try:
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                data = loaded if isinstance(loaded, dict) else None
            except Exception:
                data = None
        if data:
            cfg.update({k: v for k, v in data.items() if v is not None})
    env_mode = (os.getenv("AJAX_SERVICES_MODE") or "").strip().lower()
    if env_mode:
        cfg["mode"] = env_mode
    env_single = (os.getenv("AJAX_SERVICES_SINGLE_USER") or "").strip()
    if env_single:
        cfg["single_user"] = env_single
    env_expected = _parse_expected_users_env(os.getenv("AJAX_SERVICES_EXPECTED_USERS") or "")
    if env_expected:
        cfg["expected_port_users"] = env_expected
    return cfg


def _resolve_expected_users(cfg: Dict[str, Any]) -> Dict[int, str]:
    mode = str(cfg.get("mode") or "multi_user").strip().lower()
    expected = _normalize_expected_users(cfg.get("expected_port_users"))
    if mode in {"single_user", "single-user", "single"}:
        single_user = str(cfg.get("single_user") or "Javi").strip() or "Javi"
        ports = cfg.get("ports") if isinstance(cfg.get("ports"), list) else list(DEFAULT_PORTS)
        expected = {int(p): single_user for p in ports if isinstance(p, int) or str(p).isdigit()}
    if not expected:
        expected = {int(k): str(v) for k, v in EXPECTED_PORT_USERS.items()}
    return expected


def _hresult_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            if raw.lower().startswith("0x"):
                return int(raw, 16)
            return int(raw)
        except Exception:
            return None
    return None


def _normalize_header_key(key: str) -> str:
    raw = key.strip().lower()
    if not raw:
        return ""
    norm = unicodedata.normalize("NFKD", raw)
    stripped = "".join(ch for ch in norm if ord(ch) < 128)
    return stripped


def _normalize_text(value: str) -> str:
    raw = value.strip().lower()
    if not raw:
        return ""
    norm = unicodedata.normalize("NFKD", raw)
    stripped = "".join(ch for ch in norm if ord(ch) < 128)
    return re.sub(r"\s+", " ", stripped).strip()


def _row_pick(row: Dict[str, str], *candidates: str) -> str:
    norm_map = {_normalize_header_key(k): v for k, v in row.items()}
    for cand in candidates:
        key = _normalize_header_key(cand)
        val = norm_map.get(key)
        if val:
            return val
    return ""


def _guess_task_name(row: Dict[str, str]) -> str:
    best = ""
    for val in row.values():
        if not val:
            continue
        candidate = val.strip()
        if not candidate.startswith("\\") or "\\" not in candidate:
            continue
        if "\\ajax\\" in candidate.lower():
            return candidate
        if not best:
            best = candidate
    return best


def _format_hresult(value: Any) -> Optional[str]:
    num = _hresult_int(value)
    if num is None:
        return None
    return f"0x{num:08X}"


def _load_tasks_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _parse_schtasks_csv(text: str) -> Tuple[List[Dict[str, str]], str]:
    text = text.lstrip("\ufeff")
    rows: List[Dict[str, str]] = []
    if not text.strip():
        return rows, "empty"
    parse_quality = "ok"
    reader = csv.reader(io.StringIO(text))
    header: Optional[List[str]] = None
    for row in reader:
        if not row:
            continue
        if header is None:
            header = [cell.strip().lstrip("\ufeff") for cell in row]
            norm_header = {_normalize_header_key(h) for h in header if h}
            if not any(k in norm_header for k in {"taskname", "nombre de tarea", "nombre de la tarea"}):
                parse_quality = "degraded"
            continue
        if header is None:
            continue
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
            parse_quality = "degraded"
        if len(row) > len(header):
            row = row[: len(header)]
            parse_quality = "degraded"
        record = {header[idx]: (row[idx] or "").strip() for idx in range(len(header))}
        rows.append(record)
    if not rows and parse_quality == "ok":
        parse_quality = "degraded"
    return rows, parse_quality


def _task_path_from_name(name: str) -> str:
    if not name.startswith("\\"):
        return ""
    parts = [p for p in name.split("\\") if p]
    if len(parts) <= 1:
        return "\\"
    return "\\" + "\\".join(parts[:-1]) + "\\"


def _task_leaf_name(name: str) -> str:
    if not name:
        return ""
    if name.startswith("\\"):
        parts = [p for p in name.split("\\") if p]
        return parts[-1] if parts else ""
    return name


def _task_full_name(task_name: str, task_path: str) -> str:
    if task_name.startswith("\\"):
        return task_name
    if not task_path:
        return task_name
    if not task_path.endswith("\\"):
        task_path = task_path + "\\"
    return f"{task_path}{task_name}"


def _dotnet_date_to_iso(value: str) -> Optional[str]:
    if not value:
        return None
    match = re.match(r"^/Date\((-?\d+)([+-]\d{4})?\)/$", value.strip())
    if not match:
        return None
    try:
        ms = int(match.group(1))
    except Exception:
        return None
    try:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except Exception:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _append_note(note: Optional[str], extra: Optional[str]) -> Optional[str]:
    if not extra:
        return note
    if not note:
        return extra
    return f"{note}; {extra}"


def _task_run_fields(task: Dict[str, Any]) -> Dict[str, Optional[str]]:
    last_run_val = task.get("last_run_time")
    last_run_out = None if last_run_val in ("", None) else str(last_run_val)
    if last_run_out:
        iso = _dotnet_date_to_iso(last_run_out)
        if iso:
            last_run_out = iso
    last_result_val = task.get("last_task_result")
    last_result_out = None if last_result_val in ("", None) else str(last_result_val)
    last_result_hex = None
    if last_result_out is not None:
        last_result_hex = _format_hresult(last_result_out)
    note = task.get("note")
    if last_run_out is None and last_result_out is None:
        note = _append_note(note, "never_ran_or_info_unavailable")

    state_norm = _normalize_text(str(task.get("state") or ""))
    disabled = state_norm in {"disabled", "deshabilitado", "deshabilitada"}
    result_num = _hresult_int(last_result_out)
    if disabled and result_num not in (None, 0) and last_run_out is not None:
        note = _append_note(note, "historical_disabled_result")

    return {
        "last_run_time": last_run_out,
        "last_task_result": last_result_out,
        "last_result_hex": last_result_hex,
        "note": note,
    }


def _is_core_task_name(name: str) -> bool:
    if not name:
        return False
    lower = name.lower()
    for pattern in CORE_TASK_NAME_PATTERNS:
        if pattern.lower() in lower:
            return True
    return False


def _parse_powershell_tasks_json(text: str) -> List[Dict[str, Any]]:
    raw = text.strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return []

    def _escape_backslashes(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.replace("\\", "\\\\")

    for item in items:
        if not isinstance(item, dict):
            continue
        if "TaskName" in item:
            item["TaskName"] = _escape_backslashes(item.get("TaskName"))
        if "TaskPath" in item:
            item["TaskPath"] = _escape_backslashes(item.get("TaskPath"))
    return items


def _runlevel_to_highest(value: str) -> Optional[bool]:
    norm = _normalize_text(value)
    if not norm:
        return None
    if "highest" in norm or "mas alto" in norm or "masalto" in norm:
        return True
    if "least" in norm or "menos" in norm:
        return False
    return None


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _is_missing_value(value: Optional[str]) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    norm = _normalize_text(text)
    if norm in {"n/a", "na", "no disponible", "no se ha ejecutado", "not yet", "never", "unknown"}:
        return True
    return False


def _is_missing_run_time(value: Optional[str]) -> bool:
    if _is_missing_value(value):
        return True
    norm = _normalize_text(str(value))
    if "1601" in norm:
        return True
    return False


def _parse_schtasks_list(text: str) -> Dict[str, Optional[str]]:
    last_run = None
    last_result = None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key_norm = _normalize_text(key)
        val = val.strip()
        is_last_run = False
        if key_norm in {"last run time", "ultimo tiempo de ejecucion", "ultima hora de ejecucion"}:
            is_last_run = True
        elif ("last run" in key_norm) or (
            ("ultimo" in key_norm or "ultima" in key_norm)
            and ("tiempo" in key_norm or "hora" in key_norm)
            and "ejec" in key_norm
        ):
            is_last_run = True
        if is_last_run:
            last_run = val
        is_last_result = False
        if key_norm in {"last result", "ultimo resultado"}:
            is_last_result = True
        elif ("last result" in key_norm) or (
            ("ultimo" in key_norm or "ultima" in key_norm) and "resultado" in key_norm
        ):
            is_last_result = True
        if is_last_result:
            last_result = val
    return {"last_run_raw": last_run, "last_result_raw": last_result}


def _collect_task_run_info_ps(task_name: str, task_path: str) -> Dict[str, Any]:
    ps_path = _powershell_path()
    leaf = _task_leaf_name(task_name)
    script = (
        "$ErrorActionPreference='Stop';"
        f"$info=Get-ScheduledTaskInfo -TaskPath {_ps_quote(task_path)} -TaskName {_ps_quote(leaf)};"
        "$obj=[pscustomobject]@{LastRunTime=$info.LastRunTime; LastTaskResult=$info.LastTaskResult};"
        "$obj | ConvertTo-Json -Depth 4"
    )
    proc = _run([ps_path, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script])
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return {"ok": False, "source": "ps", "stderr": stderr.strip()}
    try:
        payload = json.loads(stdout)
    except Exception:
        return {"ok": False, "source": "ps", "stderr": "parse_failed"}
    last_run_raw = None if payload.get("LastRunTime") is None else str(payload.get("LastRunTime"))
    last_result_raw = None if payload.get("LastTaskResult") is None else str(payload.get("LastTaskResult"))
    last_run = None if _is_missing_run_time(last_run_raw) else last_run_raw
    last_result = None if _is_missing_value(last_result_raw) else last_result_raw
    return {
        "ok": True,
        "source": "ps",
        "last_run_time": last_run,
        "last_task_result": last_result,
        "last_run_raw": last_run_raw,
        "last_result_raw": last_result_raw,
    }


def _collect_task_run_info_schtasks(task_name: str) -> Dict[str, Any]:
    schtasks = _schtasks_path()
    proc = _run([schtasks, "/Query", "/TN", task_name, "/V", "/FO", "LIST"])
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return {"ok": False, "source": "schtasks", "stderr": stderr.strip()}
    info = _parse_schtasks_list(stdout)
    last_run_raw = info.get("last_run_raw")
    last_result_raw = info.get("last_result_raw")
    last_run = None if _is_missing_run_time(last_run_raw) else last_run_raw
    last_result = None if _is_missing_value(last_result_raw) else last_result_raw
    return {
        "ok": True,
        "source": "schtasks",
        "last_run_time": last_run,
        "last_task_result": last_result,
        "last_run_raw": last_run_raw,
        "last_result_raw": last_result_raw,
    }


def _collect_powershell_tasks(task_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    warnings: List[Dict[str, Any]] = []
    ps_path = _powershell_path()
    ps_script = (
        "$ErrorActionPreference='Stop';"
        f"$path='{task_path}';"
        "$tasks=@();"
        "try { $list=@(Get-ScheduledTask -TaskPath $path -ErrorAction SilentlyContinue) } catch { $list=@() };"
        "foreach ($t in $list) {"
        "$info=$null; try { $info=Get-ScheduledTaskInfo -TaskName $t.TaskName -TaskPath $t.TaskPath } catch {};"
        "$trigs=@(); foreach($tr in @($t.Triggers)) {"
        "$interval=$null; try { $interval=$tr.Repetition.Interval } catch {};"
        "$trigs += [pscustomobject]@{"
        "Type=$tr.CimClass.CimClassName;"
        "Enabled=$tr.Enabled;"
        "Delay=$tr.Delay;"
        "StartBoundary=$tr.StartBoundary;"
        "EndBoundary=$tr.EndBoundary;"
        "UserId=$tr.UserId;"
        "RepetitionInterval=$interval;"
        "}"
        "};"
        "$acts=@(); foreach($a in @($t.Actions)) {"
        "$acts += [pscustomobject]@{Execute=$a.Execute; Arguments=$a.Arguments; WorkingDirectory=$a.WorkingDirectory}"
        "};"
        "$tasks += [pscustomobject]@{"
        "TaskName=$t.TaskName;"
        "TaskPath=$t.TaskPath;"
        "State=if ($info) { $info.State } else { $null };"
        "LastRunTime=if ($info) { $info.LastRunTime } else { $null };"
        "LastTaskResult=if ($info) { $info.LastTaskResult } else { $null };"
        "PrincipalUser=$t.Principal.UserId;"
        "LogonType=$t.Principal.LogonType;"
        "RunLevel=$t.Principal.RunLevel;"
        "Triggers=$trigs;"
        "Actions=$acts;"
        "}"
        "};"
        "$tasks | ConvertTo-Json -Depth 6"
    )
    try:
        proc = _run([ps_path, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script])
    except FileNotFoundError as exc:
        warnings.append({"code": "powershell_missing", "detail": str(exc)})
        return [], warnings

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        warnings.append(
            {
                "code": "powershell_query_failed",
                "exit_code": proc.returncode,
                "stderr": stderr.strip(),
            }
        )
        return [], warnings
    tasks = _parse_powershell_tasks_json(stdout)
    if not tasks and stdout.strip():
        warnings.append({"code": "powershell_parse_failed", "detail": "No JSON tasks parsed from PowerShell output."})
    return tasks, warnings


def _merge_powershell_tasks(
    tasks_norm: List[Dict[str, Any]], ps_tasks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], bool]:
    if not ps_tasks:
        return tasks_norm, False

    def _ps_obj_to_list(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        return []

    def _ps_actions_to_norm(value: Any) -> List[Dict[str, str]]:
        actions: List[Dict[str, str]] = []
        for item in _ps_obj_to_list(value):
            execute = str(item.get("Execute") or item.get("execute") or "").strip()
            arguments = str(item.get("Arguments") or item.get("arguments") or "").strip()
            working = str(item.get("WorkingDirectory") or item.get("working_directory") or "").strip()
            if execute or arguments or working:
                actions.append(
                    {
                        "execute": execute,
                        "arguments": arguments,
                        "working_directory": working,
                    }
                )
        return actions

    def _ps_triggers_to_norm(value: Any) -> List[Dict[str, Any]]:
        triggers: List[Dict[str, Any]] = []
        for item in _ps_obj_to_list(value):
            trig: Dict[str, Any] = {}
            try:
                cim = item.get("CimClass")
                if isinstance(cim, dict):
                    trig_type = cim.get("CimClassName") or cim.get("cimclassname") or ""
                    if trig_type:
                        trig["type"] = str(trig_type)
            except Exception:
                pass
            for key in ("Enabled", "Delay", "StartBoundary", "EndBoundary", "UserId"):
                try:
                    val = item.get(key)
                except Exception:
                    val = None
                if val is None:
                    continue
                sval = str(val).strip()
                if sval:
                    trig[key.lower()] = sval
            try:
                rep = item.get("Repetition")
                if isinstance(rep, dict):
                    interval = rep.get("Interval") or rep.get("interval")
                    if interval:
                        trig["repetition_interval"] = str(interval)
            except Exception:
                pass
            if trig:
                triggers.append(trig)
        return triggers

    def _normalize_task_name(name: str) -> str:
        if "\\\\" in name:
            name = name.replace("\\\\", "\\")
        if name.startswith("\\\\"):
            name = "\\" + name.lstrip("\\")
        return name

    ps_map: Dict[str, Dict[str, Any]] = {}
    for row in ps_tasks:
        raw_name = str(row.get("TaskName") or "")
        raw_path = str(row.get("TaskPath") or "")
        # PowerShell typically returns leaf TaskName + TaskPath; rebuild full name.
        name = raw_name
        if raw_path and raw_name and not raw_name.startswith("\\"):
            name = f"{raw_path}{raw_name}"
        name = _normalize_task_name(name)
        if not name:
            continue
        if not name.lower().startswith("\\ajax\\"):
            continue
        ps_map[name] = row

    if not ps_map:
        return tasks_norm, False

    if not tasks_norm:
        merged: List[Dict[str, Any]] = []
        for name, row in ps_map.items():
            run_level = str(row.get("RunLevel") or "")
            ps_actions = _ps_actions_to_norm(row.get("Actions"))
            ps_triggers = _ps_triggers_to_norm(row.get("Triggers"))
            merged.append(
                {
                    "task_name": name,
                    "task_path": _normalize_task_name(str(row.get("TaskPath") or "")) or _task_path_from_name(name),
                    "state": row.get("State") or "",
                    "principal_user": row.get("PrincipalUser") or "",
                    "principal_logon_type": row.get("LogonType") or "",
                    "principal_run_level": run_level,
                    "highest_privileges": _runlevel_to_highest(run_level),
                    "last_task_result": row.get("LastTaskResult") or "",
                    "last_run_time": str(row.get("LastRunTime") or ""),
                    "actions": ps_actions,
                    "triggers": ps_triggers,
                }
            )
        return merged, True

    for task in tasks_norm:
        name = str(task.get("task_name") or "")
        if not name:
            continue
        row = ps_map.get(name)
        if not row:
            continue
        if not task.get("principal_user") and row.get("PrincipalUser"):
            task["principal_user"] = row.get("PrincipalUser")
        if not task.get("principal_logon_type") and row.get("LogonType"):
            task["principal_logon_type"] = row.get("LogonType")
        if not task.get("state") and row.get("State"):
            task["state"] = row.get("State")
        if not task.get("last_task_result") and row.get("LastTaskResult") is not None:
            task["last_task_result"] = str(row.get("LastTaskResult"))
        if not task.get("last_run_time") and row.get("LastRunTime"):
            task["last_run_time"] = str(row.get("LastRunTime"))
        run_level = str(row.get("RunLevel") or "")
        if run_level:
            task["principal_run_level"] = run_level
            task["highest_privileges"] = _runlevel_to_highest(run_level)
        existing_actions = task.get("actions") or []
        has_exec = False
        has_args = False
        if isinstance(existing_actions, list):
            for action in existing_actions:
                if not isinstance(action, dict):
                    continue
                if str(action.get("execute") or "").strip():
                    has_exec = True
                if str(action.get("arguments") or "").strip():
                    has_args = True
        if (not has_exec and not has_args) or not isinstance(existing_actions, list) or len(existing_actions) == 0:
            ps_actions = _ps_actions_to_norm(row.get("Actions"))
            if ps_actions:
                # Preserve working_directory if PowerShell doesn't provide it.
                try:
                    if (
                        existing_actions
                        and isinstance(existing_actions, list)
                        and isinstance(existing_actions[0], dict)
                        and isinstance(ps_actions[0], dict)
                    ):
                        wd_existing = str(existing_actions[0].get("working_directory") or "").strip()
                        wd_ps = str(ps_actions[0].get("working_directory") or "").strip()
                        if wd_existing and not wd_ps:
                            ps_actions[0]["working_directory"] = wd_existing
                except Exception:
                    pass
                task["actions"] = ps_actions
        triggers = task.get("triggers")
        has_triggers = isinstance(triggers, list) and len(triggers) > 0
        if not has_triggers:
            ps_triggers = _ps_triggers_to_norm(row.get("Triggers"))
            if ps_triggers:
                task["triggers"] = ps_triggers
    return tasks_norm, True


def _normalize_task_from_row(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    name = _row_pick(row, "TaskName", "Task Name", "Nombre de tarea", "Nombre de la tarea")
    if not name:
        name = _guess_task_name(row)
    if not name:
        return None
    if name.startswith("\\\\"):
        name = "\\" + name.lstrip("\\")
    if not name.lower().startswith("\\ajax\\"):
        return None
    task_path = _task_path_from_name(name)
    state = _row_pick(row, "Status", "Scheduled Task State", "State", "Estado")
    principal = _row_pick(row, "Run As User", "Run as User", "Principal", "Ejecutar como usuario")
    logon_mode = _row_pick(row, "Logon Mode", "LogonMode", "Modo de inicio de sesion")
    run_level = _row_pick(row, "Run Level", "RunLevel", "Nivel de ejecucion", "Nivel de ejecución")
    last_result = _row_pick(row, "Last Result", "Last Task Result", "LastTaskResult", "Ultimo resultado")
    last_run_time = _row_pick(row, "Last Run Time", "Ultima hora de ejecucion")
    action = _row_pick(row, "Task To Run", "Task to Run", "Actions", "Tarea que ejecutar")
    start_in = _row_pick(row, "Start In", "Start in", "Start In (optional)", "Iniciar en")
    schedule_type = _row_pick(row, "Schedule Type", "Schedule", "Tipo de programacion")
    next_run = _row_pick(row, "Next Run Time", "Next Run", "Proxima hora de ejecucion")
    start_date = _row_pick(row, "Start Date", "Fecha de inicio")
    start_time = _row_pick(row, "Start Time", "Hora de inicio")
    end_date = _row_pick(row, "End Date", "Fecha de finalizacion")
    triggers: Optional[List[Dict[str, Any]]] = []
    trigger_entry = {
        "schedule_type": schedule_type,
        "next_run_time": next_run,
        "start_date": start_date,
        "start_time": start_time,
        "end_date": end_date,
    }
    if any(v for v in trigger_entry.values()):
        triggers.append(trigger_entry)
    else:
        triggers = None

    highest = _runlevel_to_highest(run_level)

    return {
        "task_name": name,
        "task_path": task_path,
        "state": state,
        "principal_user": principal,
        "principal_logon_type": logon_mode,
        "principal_run_level": run_level,
        "highest_privileges": highest,
        "last_task_result": last_result,
        "last_run_time": last_run_time,
        "triggers": triggers,
        "actions": [
            {
                "execute": action,
                "arguments": "",
                "working_directory": start_in,
            }
        ]
        if action or start_in
        else [],
    }


def _build_tasks_report_from_schtasks(
    stdout: str,
    stderr: str,
    exit_code: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    warnings: List[Dict[str, Any]] = []
    rows: List[Dict[str, str]] = []
    parse_quality = "empty"
    if stdout.strip():
        rows, parse_quality = _parse_schtasks_csv(stdout)

    if exit_code != 0:
        reason = "schtasks_failed"
        if "cannot find the file specified" in stdout.lower() or "cannot find the file specified" in stderr.lower():
            reason = "task_folder_not_found"
        warnings.append(
            {
                "code": "schtasks_query_failed",
                "skip_reason": reason,
                "exit_code": exit_code,
                "stderr": stderr.strip(),
                "stdout_excerpt": stdout.strip()[:300],
            }
        )
    elif stdout.strip() and not rows:
        warnings.append(
            {
                "code": "schtasks_parse_failed",
                "skip_reason": "csv_parse_failed",
                "exit_code": exit_code,
                "stderr": stderr.strip(),
                "stdout_excerpt": stdout.strip()[:300],
            }
        )
    elif not stdout.strip() and exit_code == 0:
        warnings.append(
            {
                "code": "schtasks_empty_output",
                "skip_reason": "no_stdout_from_query",
                "exit_code": exit_code,
            }
        )

    if parse_quality == "degraded":
        warnings.append(
            {
                "code": "degraded_parse",
                "skip_reason": "headers_or_rows_inconsistent",
                "exit_code": exit_code,
            }
        )

    tasks_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task = _normalize_task_from_row(row)
        if not task:
            continue
        name = str(task.get("task_name") or "")
        if not name:
            continue
        merged = tasks_map.get(name)
        if not merged:
            merged_triggers = task.get("triggers") if task.get("triggers") is not None else None
            merged = {
                "task_name": name,
                "task_path": task.get("task_path"),
                "state": task.get("state"),
                "principal_user": task.get("principal_user"),
                "principal_logon_type": task.get("principal_logon_type"),
                "principal_run_level": task.get("principal_run_level"),
                "highest_privileges": task.get("highest_privileges"),
                "last_task_result": task.get("last_task_result"),
                "last_run_time": task.get("last_run_time"),
                "actions": [],
                "triggers": merged_triggers,
            }
            tasks_map[name] = merged

        if not merged.get("state") and task.get("state"):
            merged["state"] = task.get("state")
        if not merged.get("principal_user") and task.get("principal_user"):
            merged["principal_user"] = task.get("principal_user")
        if not merged.get("principal_logon_type") and task.get("principal_logon_type"):
            merged["principal_logon_type"] = task.get("principal_logon_type")
        if not merged.get("principal_run_level") and task.get("principal_run_level"):
            merged["principal_run_level"] = task.get("principal_run_level")
        if not merged.get("last_task_result") and task.get("last_task_result"):
            merged["last_task_result"] = task.get("last_task_result")
        if not merged.get("last_run_time") and task.get("last_run_time"):
            merged["last_run_time"] = task.get("last_run_time")
        if task.get("highest_privileges") and not merged.get("highest_privileges"):
            merged["highest_privileges"] = True

        for action in task.get("actions") or []:
            if action and action not in merged["actions"]:
                merged["actions"].append(action)
        if isinstance(task.get("triggers"), list):
            if merged.get("triggers") is None:
                merged["triggers"] = []
            for trig in task.get("triggers") or []:
                if trig and trig not in (merged.get("triggers") or []):
                    merged["triggers"].append(trig)

    tasks_norm = list(tasks_map.values())
    for task in tasks_norm:
        task["last_task_result_hex"] = _format_hresult(task.get("last_task_result"))
        task_warnings = _detect_task_warnings(task)
        if task_warnings:
            warnings.append({"task_name": task.get("task_name"), "warnings": task_warnings})

    if rows and not tasks_norm:
        warnings.append(
            {
                "code": "no_ajax_tasks_found",
                "skip_reason": "no_taskname_matches_ajax_prefix",
                "exit_code": exit_code,
            }
        )

    return tasks_norm, warnings, parse_quality


def _detect_task_warnings(task: Dict[str, Any]) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    task_name = str(task.get("task_name") or "")
    principal = str(task.get("principal_user") or "")
    logon_type = str(task.get("principal_logon_type") or "")
    run_level = str(task.get("principal_run_level") or "")
    highest = task.get("highest_privileges")
    triggers_raw = task.get("triggers")
    triggers = triggers_raw if isinstance(triggers_raw, list) else []
    actions = task.get("actions") or []
    logon_norm = _normalize_text(logon_type)
    run_level_norm = _normalize_text(run_level)

    def _win_to_wsl_path(value: str) -> Optional[Path]:
        raw = (value or "").strip().strip('"').strip()
        if not raw:
            return None
        if raw.startswith("/mnt/"):
            return Path(raw)
        m = re.match(r"^([A-Za-z]):\\\\(.+)$", raw)
        if not m:
            return None
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\\\", "/").replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")

    def _extract_action_targets(exe: str, args: str) -> List[str]:
        targets: List[str] = []
        exe_norm = _normalize_text(exe)
        args_raw = args or ""
        if "wscript.exe" in exe_norm or "cscript.exe" in exe_norm:
            for m in re.finditer(r"\"([^\"]+\\.(?:vbs|js))\"", args_raw, flags=re.IGNORECASE):
                targets.append(m.group(1))
        m = re.search(r"(?i)-file\\s+\"([^\"]+)\"", args_raw)
        if m:
            targets.append(m.group(1))
        else:
            m = re.search(r"(?i)-file\\s+([^\\s]+)", args_raw)
            if m:
                targets.append(m.group(1))
        return targets

    hresult = _hresult_int(task.get("last_task_result"))
    if hresult == 0x80070057:
        warnings.append(
            {
                "code": "invalid_parameter_0x80070057",
                "detail": "Revisar action/arguments/working_directory; Task Scheduler reporta parametro incorrecto.",
            }
        )
    if hresult == 0x8007052E:
        warnings.append(
            {
                "code": "logon_failure_0x8007052E",
                "detail": "Fallo de logon; revisar usuario/credenciales del task.",
            }
        )

    if not principal:
        warnings.append({"code": "principal_missing", "detail": "Task sin principal definido."})

    if logon_norm and "interactive" not in logon_norm and "interactivo" not in logon_norm:
        warnings.append(
            {
                "code": "logon_type_noninteractive",
                "detail": f"LogonType={logon_type}; revisar 'Run only when user is logged on'.",
            }
        )

    if run_level_norm and highest is False:
        warnings.append(
            {
                "code": "run_level_not_highest",
                "detail": f"RunLevel={run_level}; considerar 'Run with highest privileges'.",
            }
        )
    if highest is False and ("watchdog" in task_name.lower() or "driver" in task_name.lower()):
        warnings.append(
            {
                "code": "not_highest_privileges",
                "detail": "Watchdog/driver sin HighestPrivileges.",
            }
        )
    if highest is None and ("watchdog" in task_name.lower() or "driver" in task_name.lower()):
        warnings.append(
            {
                "code": "run_level_unknown",
                "detail": "RunLevel no detectado en CSV; usar --deep para confirmar.",
            }
        )

    if isinstance(triggers_raw, list) and len(triggers) == 0:
        warnings.append({"code": "no_triggers", "detail": "Task sin triggers activos."})

    if isinstance(actions, list) and len(actions) == 0:
        warnings.append({"code": "action_missing", "detail": "Task sin actions detectadas."})

    for action in actions if isinstance(actions, list) else []:
        path = str(action.get("execute") or "")
        args = str(action.get("arguments") or "")
        if principal.lower().endswith("\\ajax") and "c:\\users\\javi\\ajax_home" in path.lower():
            warnings.append(
                {
                    "code": "action_path_mismatch",
                    "detail": "Principal AJAX pero action apunta a C:\\Users\\Javi\\AJAX_HOME.",
                }
            )
        if principal.lower().endswith("\\javi") and "c:\\users\\ajax\\ajax_home" in path.lower():
            warnings.append(
                {
                    "code": "action_path_mismatch",
                    "detail": "Principal Javi pero action apunta a C:\\Users\\AJAX\\AJAX_HOME.",
                }
            )
        for target in _extract_action_targets(path, args):
            wsl_path = _win_to_wsl_path(target)
            if wsl_path and not wsl_path.exists():
                warnings.append(
                    {
                        "code": "action_target_missing",
                        "detail": f"Action target missing: {target}",
                    }
                )

    unique: List[Dict[str, str]] = []
    seen = set()
    for item in warnings:
        key = item.get("code") or ""
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def run_tasks_doctor(
    root_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    explain: bool = False,
    deep: bool = False,
) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = root_dir / "artifacts" / "doctor" / _now_ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    schtasks = _schtasks_path()
    cmd = [schtasks, "/Query", "/FO", "CSV", "/V"]
    proc = _run(cmd, cwd=root_dir)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    tasks_norm, warnings, parse_quality = _build_tasks_report_from_schtasks(stdout, stderr, proc.returncode)
    global_warnings = [w for w in warnings if "task_name" not in w]

    ps_tasks, ps_warnings = _collect_powershell_tasks("\\AJAX\\")
    tasks_norm, ps_enriched = _merge_powershell_tasks(tasks_norm, ps_tasks)

    scope_tasks = [t for t in tasks_norm if _is_core_task_name(str(t.get("task_name") or ""))]
    tasks_target = scope_tasks if scope_tasks else tasks_norm
    tasks_scope = "core" if scope_tasks else "all"

    for task in tasks_target:
        name = str(task.get("task_name") or "")
        task_path = str(task.get("task_path") or "\\AJAX\\")
        info = _collect_task_run_info_ps(name, task_path)
        if not info.get("ok") or (info.get("last_run_time") is None and info.get("last_task_result") is None):
            full_name = _task_full_name(name, task_path)
            info = _collect_task_run_info_schtasks(full_name)
        task["last_run_time"] = info.get("last_run_time") if info.get("last_run_time") is not None else task.get("last_run_time")
        task["last_task_result"] = info.get("last_task_result") if info.get("last_task_result") is not None else task.get("last_task_result")
        task["last_run_source"] = info.get("source")
        task["last_run_raw"] = info.get("last_run_raw")
        task["last_result_raw"] = info.get("last_result_raw")

    deep_warnings: List[Dict[str, Any]] = []
    deep_effective = deep
    if not deep and tasks_target:
        for task in tasks_target:
            triggers = task.get("triggers") or []
            if task.get("highest_privileges") is None or len(triggers) == 0:
                deep_effective = True
                break
    deep_note = ""
    if deep_effective and not deep:
        deep_note = "Se uso XML de schtasks para completar RunLevel/triggers faltantes."
    if deep_effective:
        for task in tasks_target:
            triggers = task.get("triggers") or []
            if triggers and task.get("highest_privileges") is not None:
                continue
            name = str(task.get("task_name") or "")
            if not name:
                continue
            xml_res = _run([schtasks, "/Query", "/TN", name, "/XML"])
            if xml_res.returncode != 0 or not (xml_res.stdout or "").strip():
                deep_warnings.append(
                    {
                        "code": "deep_query_failed",
                        "task_name": name,
                        "exit_code": xml_res.returncode,
                        "stderr": (xml_res.stderr or "").strip(),
                    }
                )
                continue
            try:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(xml_res.stdout)
                triggers = []
                for trig in root.findall(".//{*}Triggers/*"):
                    tag = trig.tag
                    if "}" in tag:
                        tag = tag.split("}", 1)[1]
                    triggers.append({"type": tag})
                if triggers:
                    task["triggers"] = triggers
                run_level_xml = ""
                try:
                    run_level_xml = root.findtext(".//{*}Principals//{*}RunLevel") or ""
                except Exception:
                    run_level_xml = ""
                if run_level_xml:
                    task["principal_run_level"] = run_level_xml
                    run_level_norm = _normalize_text(run_level_xml)
                    if "highest" in run_level_norm or "mas alto" in run_level_norm or "masalto" in run_level_norm:
                        task["highest_privileges"] = True
                    else:
                        task["highest_privileges"] = False
            except Exception as exc:
                deep_warnings.append(
                    {
                        "code": "deep_parse_failed",
                        "task_name": name,
                        "detail": str(exc),
                    }
                )
        if tasks_target:
            task_warnings: List[Dict[str, Any]] = []
            for task in tasks_target:
                tw = _detect_task_warnings(task)
                if tw:
                    task_warnings.append({"task_name": task.get("task_name"), "warnings": tw})
            warnings = global_warnings + ps_warnings + deep_warnings + task_warnings
        else:
            warnings = global_warnings + ps_warnings + deep_warnings
    else:
        if tasks_target:
            task_warnings: List[Dict[str, Any]] = []
            for task in tasks_target:
                tw = _detect_task_warnings(task)
                if tw:
                    task_warnings.append({"task_name": task.get("task_name"), "warnings": tw})
            warnings = global_warnings + ps_warnings + task_warnings
        else:
            warnings = global_warnings + ps_warnings

    tasks_out: List[Dict[str, Any]] = []
    triggers_count = 0
    for task in tasks_target:
        action_execute = ""
        action_arguments = ""
        action_workdir = ""
        if task.get("actions"):
            try:
                first = task["actions"][0] if isinstance(task["actions"], list) and task["actions"] else {}
            except Exception:
                first = {}
            if isinstance(first, dict):
                action_execute = str(first.get("execute") or "")
                action_arguments = str(first.get("arguments") or "")
                action_workdir = str(first.get("working_directory") or "")
        action = (action_execute + " " + action_arguments).strip()
        triggers_count += len(task.get("triggers") or [])
        run_fields = _task_run_fields(task)
        last_run_out = run_fields.get("last_run_time")
        last_result_out = run_fields.get("last_task_result")
        last_result_hex = run_fields.get("last_result_hex")
        note = run_fields.get("note")
        tasks_out.append(
            {
                "TaskName": task.get("task_name"),
                "TaskPath": task.get("task_path"),
                "State": task.get("state"),
                "RunAsUser": task.get("principal_user"),
                "LastRunTime": last_run_out,
                "LastResult": last_result_out,
                "LastResultHex": last_result_hex,
                "LastRunSource": task.get("last_run_source"),
                "LastRunRaw": task.get("last_run_raw"),
                "LastResultRaw": task.get("last_result_raw"),
                "Action": action,
                "ActionExecute": action_execute,
                "ActionArguments": action_arguments,
                "ActionWorkingDirectory": action_workdir,
                "HighestPrivileges": task.get("highest_privileges"),
                "TriggersCount": len(task.get("triggers") or []),
                "Note": note,
            }
        )

    tasks_total = len(ps_tasks) if ps_enriched else len(tasks_norm)
    report = {
        "schema": "ajax.doctor.tasks.v0",
        "timestamp_utc": _utc_now(),
        "command": "schtasks /Query /FO CSV /V",
        "parse_quality": parse_quality,
        "powershell_enriched": ps_enriched,
        "deep_used": deep_effective,
        "deep_note": deep_note,
        "tasks_found": len(tasks_out),
        "tasks_total": tasks_total,
        "tasks_scope": tasks_scope,
        "triggers_count": triggers_count,
        "tasks": tasks_out,
        "warnings": warnings,
    }
    _write_text(out_dir / "tasks_explain.json", json.dumps(report, indent=2, ensure_ascii=False))

    summary_lines = [
        "AJAX Doctor tasks",
        f"Tasks in scope: {len(tasks_out)} ({tasks_scope})",
        f"Tasks total in \\\\AJAX\\\\: {tasks_total}",
        f"Triggers found: {triggers_count}",
        f"Warnings: {len(warnings)}",
        f"Evidence: {out_dir}",
    ]
    summary = "\n".join(summary_lines)
    return {"ok": proc.returncode == 0, "out_dir": str(out_dir), "report": report, "summary": summary}


def _load_ports_map(path: Path) -> Dict[int, Dict[str, Any]]:
    raw = _read_text(path)
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    mapping: Dict[int, Dict[str, Any]] = {}
    if isinstance(data, list):
        for row in data:
            try:
                port = int(row.get("LocalPort"))
            except Exception:
                continue
            mapping[port] = dict(row)
    elif isinstance(data, dict):
        for key, row in data.items():
            try:
                port = int(key)
            except Exception:
                continue
            if isinstance(row, dict):
                mapping[port] = dict(row)
    return mapping


def _load_tasks_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _row_get(row: Dict[str, str], *names: str) -> str:
    lookup = {k.strip().lower(): v for k, v in row.items()}
    for name in names:
        hit = lookup.get(name.lower())
        if hit:
            return hit
    return ""


def _select_task_for_port(tasks: List[Dict[str, str]], port: int) -> Optional[Dict[str, str]]:
    best: Optional[Dict[str, str]] = None
    best_score = -1
    token = str(port)
    for row in tasks:
        name = _row_get(row, "TaskName", "Task Name")
        action = _row_get(row, "Task To Run", "Actions", "Task to Run")
        score = 0
        if token in name:
            score += 5
        if token in action:
            score += 5
        if "watchdog" in name.lower():
            score += 2
        if "ajax_driver_watchdog" in action.lower():
            score += 3
        if name.startswith("\\AJAX\\"):
            score += 1
        if score > best_score:
            best_score = score
            best = row
    return best


def _explain_mismatch(
    port: int,
    expected_user: str,
    expected_session: Optional[int],
    actual: Optional[Dict[str, Any]],
    tasks: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, str]]]:
    causes: List[Dict[str, Any]] = []
    task = _select_task_for_port(tasks, port)

    if actual is None:
        causes.append(
            {
                "code": "port_not_listening",
                "detail": "No listener found for port",
            }
        )
        return causes, task

    actual_session = actual.get("SessionId")
    actual_user = actual.get("User") or ""
    cmdline = actual.get("CommandLine") or ""

    if expected_session is None:
        causes.append(
            {
                "code": "expected_session_missing",
                "detail": f"Expected session for {expected_user} not found in qwinsta",
            }
        )
        return causes, task

    if actual_session == expected_session:
        return causes, task

    causes.append(
        {
            "code": "session_swap_detected",
            "detail": f"Port {port} session {actual_session} != expected {expected_session}",
        }
    )

    if actual_user and expected_user.lower() not in actual_user.lower():
        causes.append(
            {
                "code": "process_owner_mismatch",
                "detail": f"Process owner {actual_user} != expected {expected_user}",
            }
        )

    if cmdline:
        if expected_user.lower() == "ajax" and "C:\\Users\\Javi\\AJAX_HOME".lower() in cmdline.lower():
            causes.append(
                {
                    "code": "action_points_to_javi_repo",
                    "detail": "CommandLine references Javi repo root",
                }
            )
        if expected_user.lower() == "javi" and "C:\\Users\\AJAX\\AJAX_HOME".lower() in cmdline.lower():
            causes.append(
                {
                    "code": "action_points_to_ajax_repo",
                    "detail": "CommandLine references AJAX repo root",
                }
            )

    if task is None:
        causes.append(
            {
                "code": "task_not_found",
                "detail": "No scheduled task found for this port",
            }
        )
        return causes, task

    run_as = _row_get(task, "Run As User", "Run as User", "Run As User ")
    logon_mode = _row_get(task, "Logon Mode", "LogonMode", "Logon Mode ")
    run_level = _row_get(task, "Run Level", "RunLevel")
    task_name = _row_get(task, "TaskName", "Task Name")
    task_action = _row_get(task, "Task To Run", "Actions", "Task to Run")

    if run_as and expected_user.lower() not in run_as.lower():
        causes.append(
            {
                "code": "task_runas_mismatch",
                "detail": f"Task {task_name} runs as {run_as}",
            }
        )

    if logon_mode and "interactive" not in logon_mode.lower():
        causes.append(
            {
                "code": "task_logon_noninteractive",
                "detail": f"Task {task_name} logon mode is {logon_mode}",
            }
        )
    if run_level and "highest" not in run_level.lower():
        causes.append(
            {
                "code": "task_runlevel_not_highest",
                "detail": f"Task {task_name} run level is {run_level}",
            }
        )

    if task_action:
        if expected_user.lower() == "ajax" and "C:\\Users\\Javi\\AJAX_HOME".lower() in task_action.lower():
            causes.append(
                {
                    "code": "task_action_wrong_root",
                    "detail": "Task action references Javi repo root",
                }
            )
        if expected_user.lower() == "javi" and "C:\\Users\\AJAX\\AJAX_HOME".lower() in task_action.lower():
            causes.append(
                {
                    "code": "task_action_wrong_root",
                    "detail": "Task action references AJAX repo root",
                }
            )

    if not causes:
        causes.append(
            {
                "code": "possible_acl_or_uac_block",
                "detail": "No direct mismatch found; check ACL/UAC and task logon type",
            }
        )

    return causes, task


def run_services_doctor(
    root_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    explain: bool = False,
) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = root_dir / "artifacts" / "doctor" / _now_ts()
    out_dir.mkdir(parents=True, exist_ok=True)

    ps_script = root_dir / "tools" / "ajax_doctor.ps1"
    ps_path = _powershell_path()
    ps_cmd = [
        ps_path,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        _to_windows_path(ps_script),
        "-OutDir",
        _to_windows_path(out_dir),
    ]
    proc = _run(ps_cmd, cwd=root_dir)
    _write_text(out_dir / "doctor_stdout.txt", proc.stdout or "")
    _write_text(out_dir / "doctor_stderr.txt", proc.stderr or "")

    qwinsta_path = out_dir / "qwinsta.txt"
    ports_path = out_dir / "ports_map.json"
    tasks_csv_path = out_dir / "tasks_ajax_schtasks.csv"

    qwinsta_text = _read_text(qwinsta_path) if qwinsta_path.exists() else ""
    cfg = _load_services_doctor_config(root_dir)
    expected_users = _resolve_expected_users(cfg)
    expected_ports = sorted(expected_users.keys()) if expected_users else list(DEFAULT_PORTS)
    session_users = sorted({user for user in expected_users.values() if user})
    sessions = _parse_qwinsta_sessions(qwinsta_text, users=session_users)
    ports_map = _load_ports_map(ports_path) if ports_path.exists() else {}
    tasks = _load_tasks_csv(tasks_csv_path)

    health: Dict[int, bool] = {}
    for port in expected_ports:
        hp = out_dir / f"health_{port}.json"
        if not hp.exists():
            health[port] = False
            continue
        try:
            payload = _load_json(hp)
            health[port] = bool(payload.get("ok"))
        except Exception:
            health[port] = False

    ports_in_expected: Dict[str, bool] = {}
    for port, expected_user in expected_users.items():
        actual_sid = None
        expected_sid = sessions.get(expected_user)
        if port in ports_map and expected_sid is not None:
            try:
                actual_sid = int(ports_map[port].get("SessionId"))
            except Exception:
                actual_sid = None
        ports_in_expected[str(port)] = actual_sid == int(expected_sid) if actual_sid is not None else False

    invariants = {
        "expected_users": {str(k): v for k, v in expected_users.items()},
        "ports_in_expected": ports_in_expected,
        "health_ok": all(bool(health.get(port)) for port in expected_ports),
    }

    probable: Dict[int, List[Dict[str, Any]]] = {}
    task_refs: Dict[int, Optional[Dict[str, str]]] = {}
    for port, expected_user in expected_users.items():
        causes, task = _explain_mismatch(
            port,
            expected_user,
            sessions.get(expected_user),
            ports_map.get(port),
            tasks,
        )
        probable[port] = causes
        task_refs[port] = task

    explain_payload = {
        "schema": "ajax.doctor.services.v0",
        "timestamp_utc": _utc_now(),
        "out_dir": str(out_dir),
        "mode": str(cfg.get("mode") or "multi_user"),
        "expected_users": {str(k): v for k, v in expected_users.items()},
        "sessions": sessions,
        "ports": ports_map,
        "health": health,
        "invariants": invariants,
        "probable_causes": probable,
        "task_candidates": {
            str(port): {
                "task_name": _row_get(task_refs[port], "TaskName", "Task Name") if task_refs[port] else "",
                "run_as": _row_get(task_refs[port], "Run As User", "Run as User") if task_refs[port] else "",
                "logon_mode": _row_get(task_refs[port], "Logon Mode", "LogonMode") if task_refs[port] else "",
                "run_level": _row_get(task_refs[port], "Run Level", "RunLevel") if task_refs[port] else "",
                "action": _row_get(task_refs[port], "Task To Run", "Actions", "Task to Run") if task_refs[port] else "",
            }
            for port in expected_users.keys()
        },
    }

    if explain:
        _write_text(out_dir / "services_explain.json", json.dumps(explain_payload, indent=2, ensure_ascii=False))

    return {
        "ok": bool(all(ports_in_expected.values()) and invariants["health_ok"]),
        "out_dir": str(out_dir),
        "mode": str(cfg.get("mode") or "multi_user"),
        "expected_users": {str(k): v for k, v in expected_users.items()},
        "sessions": sessions,
        "ports": ports_map,
        "health": health,
        "invariants": invariants,
        "explain": explain_payload,
    }


def _stop_process(pid: int, out_dir: Path) -> Dict[str, Any]:
    cmd = [
        _powershell_path(),
        "-NoProfile",
        "-Command",
        f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue",
    ]
    proc = _run(cmd)
    return {
        "pid": pid,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _run_schtasks(task_name: str) -> Dict[str, Any]:
    schtasks = _schtasks_path()
    proc = _run([schtasks, "/Run", "/TN", task_name])
    return {
        "task_name": task_name,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _start_port_via_tasks(tasks: List[Dict[str, str]], port: int) -> Dict[str, Any]:
    task = _select_task_for_port(tasks, port)
    if task is None:
        return {"ok": False, "error": "task_not_found"}
    task_name = _row_get(task, "TaskName", "Task Name")
    if not task_name:
        return {"ok": False, "error": "task_name_missing"}
    res = _run_schtasks(task_name)
    res["ok"] = res["returncode"] == 0
    return res


def _start_port_fallback(root_dir: Path, port: int) -> Dict[str, Any]:
    if port == 5012:
        return {"ok": False, "error": "no_safe_fallback_for_5012"}
    script = root_dir / "Start-AjaxDriver.ps1"
    if not script.exists():
        return {"ok": False, "error": "start_script_missing"}
    cmd = [
        _powershell_path(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        _to_windows_path(script),
        "-Port",
        str(port),
    ]
    proc = _run(cmd, cwd=root_dir)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _build_case_record(
    *,
    case_id: str,
    baseline: Dict[str, Any],
    final: Dict[str, Any],
    steps: List[Dict[str, Any]],
    evidence: Dict[str, str],
    outcome: str,
    surprise: str,
) -> Dict[str, Any]:
    return {
        "schema": "ajax.case.v0.1",
        "case_id": case_id,
        "timestamp_utc": _utc_now(),
        "expected": {
            "contract": {"5010": "Javi", "5012": "AJAX"},
            "health": "200",
        },
        "actual": {
            "baseline": baseline,
            "final": final,
        },
        "steps": steps,
        "outcome": outcome,
        "surprise": surprise,
        "evidence": evidence,
    }


def _ingest_case_to_leann(root_dir: Path, case_path: Path, out_dir: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(root_dir / "scripts" / "rag_inbox_scan.py"),
        "--src",
        str(case_path),
    ]
    proc = _run(cmd, cwd=root_dir, timeout=900)
    payload = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    _write_text(out_dir / "case_ingest.json", json.dumps(payload, indent=2, ensure_ascii=False))
    payload["ok"] = proc.returncode == 0
    return payload


def _suggest_from_causes(probable: Dict[int, List[Dict[str, Any]]]) -> List[str]:
    suggestions: List[str] = []
    for entries in probable.values():
        for item in entries:
            code = str(item.get("code") or "")
            if code == "task_runas_mismatch":
                suggestions.append("Actualizar el usuario de la tarea a Javi/AJAX según el contrato.")
            elif code == "task_logon_noninteractive":
                suggestions.append("Configurar 'Run only when user is logged on' (InteractiveToken) para evitar sesión incorrecta.")
            elif code == "task_runlevel_not_highest":
                suggestions.append("Configurar la tarea con 'Run with highest privileges' para evitar bloqueo UAC.")
            elif code in ("task_action_wrong_root", "action_points_to_javi_repo", "action_points_to_ajax_repo"):
                suggestions.append("Corregir la ruta del repo en la acción de la tarea al árbol correcto.")
            elif code == "task_not_found":
                suggestions.append("Registrar la tarea watchdog correspondiente (\\AJAX\\AJAX_DriverWatchdog_5012 / 5010).")
            elif code == "port_not_listening":
                suggestions.append("Revisar logs de watchdog en C:\\Users\\Public\\ajax\\watchdog_<port>_task.log.")
            elif code == "possible_acl_or_uac_block":
                suggestions.append("Revisar ACL del repo y permisos UAC para el usuario de la tarea.")
    deduped: List[str] = []
    for item in suggestions:
        if item not in deduped:
            deduped.append(item)
    return deduped


def fix_ports_sessions(
    root_dir: Path,
    *,
    ports: Optional[List[int]] = None,
    verify_only: bool = False,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    raw_ports = ports or list(DEFAULT_PORTS)
    ports = []
    for item in raw_ports:
        try:
            ports.append(int(item))
        except Exception:
            continue
    if not ports:
        ports = list(DEFAULT_PORTS)
    tx_dir = root_dir / "artifacts" / "doctor" / f"fix_ports_{_now_ts()}"
    tx_dir.mkdir(parents=True, exist_ok=True)

    steps: List[Dict[str, Any]] = []
    baseline = run_services_doctor(root_dir, out_dir=tx_dir / "baseline", explain=True)
    steps.append({"ts_utc": _utc_now(), "step": "doctor_baseline", "out_dir": baseline["out_dir"]})

    actions: List[Dict[str, Any]] = []
    if not verify_only:
        ports_map = baseline.get("ports") or {}
        tasks = _load_tasks_csv(Path(baseline["out_dir"]) / "tasks_ajax_schtasks.csv")
        for port in ports:
            expected_user = EXPECTED_PORT_USERS.get(port)
            if expected_user is None:
                continue
            expected_session = baseline.get("sessions", {}).get(expected_user)
            actual = ports_map.get(port)
            actual_session = None
            if isinstance(actual, dict):
                try:
                    actual_session = int(actual.get("SessionId"))
                except Exception:
                    actual_session = None
            if actual is None or expected_session is None or actual_session == expected_session:
                continue
            pid = actual.get("OwningProcess") if isinstance(actual, dict) else None
            cmdline = actual.get("CommandLine") if isinstance(actual, dict) else ""
            if pid and cmdline and ("os_driver.py" in cmdline) and (f"--port {port}" in cmdline):
                actions.append({"action": "stop_wrong_listener", "port": port, "detail": _stop_process(int(pid), tx_dir)})
            else:
                actions.append(
                    {
                        "action": "skip_stop",
                        "port": port,
                        "reason": "pid_not_safe_or_unknown",
                        "pid": pid,
                        "cmdline": cmdline,
                    }
                )
            start_res = _start_port_via_tasks(tasks, port)
            if not start_res.get("ok"):
                fallback = _start_port_fallback(root_dir, port)
                start_res = {"via": "fallback", **start_res, "fallback": fallback}
            else:
                start_res = {"via": "task", **start_res}
            actions.append({"action": "start_port", "port": port, "detail": start_res})

    steps.append({"ts_utc": _utc_now(), "step": "fix_actions", "actions": actions})
    time.sleep(max(2, min(timeout_s, 8)))

    post = run_services_doctor(root_dir, out_dir=tx_dir / "post", explain=True)
    steps.append({"ts_utc": _utc_now(), "step": "doctor_post", "out_dir": post["out_dir"]})

    ok = bool(post.get("ok"))
    resync = None
    if not ok:
        resync = run_services_doctor(root_dir, out_dir=tx_dir / "resync", explain=True)
        steps.append({"ts_utc": _utc_now(), "step": "doctor_resync", "out_dir": resync["out_dir"]})

    outcome = "PASS" if ok else "FAIL"
    surprise = "swap_detected" if not baseline.get("ok") else "none"
    explain_payload = post.get("explain") or {}
    if resync and isinstance(resync, dict):
        explain_payload = resync.get("explain") or explain_payload
    suggestions = _suggest_from_causes(explain_payload.get("probable_causes", {}) or {})

    case_dir = root_dir / "artifacts" / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_path = case_dir / f"INC-OSDRIVER-SESSION-SWAP_case_v0.1_{_now_ts()}.json"
    case_payload = _build_case_record(
        case_id="INC-OSDRIVER-SESSION-SWAP",
        baseline=baseline,
        final=post,
        steps=steps,
        evidence={
            "baseline_receipt": str(baseline["out_dir"]),
            "post_receipt": str(post["out_dir"]),
            "tx_dir": str(tx_dir),
        },
        outcome=outcome,
        surprise=surprise,
    )
    _write_text(case_path, json.dumps(case_payload, indent=2, ensure_ascii=False))
    ingest = _ingest_case_to_leann(root_dir, case_path, tx_dir)

    summary = {
        "ok": ok,
        "baseline": baseline,
        "post": post,
        "resync": resync,
        "actions": actions,
        "case_path": str(case_path),
        "ingest": ingest,
        "suggestions": suggestions,
        "tx_dir": str(tx_dir),
    }
    _write_text(tx_dir / "fix_ports_sessions.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
