from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageChops
import pytesseract

from agency.windows_driver_client import WindowsDriverClient
from agency.explore_policy import load_explore_policy, read_human_signal, compute_human_active


_YOUTUBE_TIME_RE = re.compile(
    r"(?P<cur>(?:\d{1,2}:)?\d{1,2}:\d{2})\s*/\s*(?P<tot>(?:\d{1,2}:)?\d{1,2}:\d{2})"
)


def _time_to_seconds(tok: str) -> Optional[int]:
    tok = str(tok or "").strip()
    if not tok:
        return None
    parts = tok.split(":")
    try:
        nums = [int(p) for p in parts]
    except Exception:
        return None
    if len(nums) == 2:
        m, s = nums
        return m * 60 + s
    if len(nums) == 3:
        h, m, s = nums
        return h * 3600 + m * 60 + s
    return None


def _ocr_text(image_path: Path) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img) or ""


def _extract_youtube_current_time_seconds(image_path: Path) -> Optional[int]:
    text = _ocr_text(image_path)
    m = _YOUTUBE_TIME_RE.search(text.replace("\n", " "))
    if not m:
        return None
    return _time_to_seconds(m.group("cur"))


def _find_word_bbox(image_path: Path, word: str) -> Optional[Tuple[int, int, int, int]]:
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    target = str(word or "").strip().lower()
    if not target:
        return None
    for i in range(len(data.get("text", []) or [])):
        txt = str((data.get("text") or [""])[i] or "").strip().lower()
        if not txt:
            continue
        if target != txt:
            continue
        try:
            x = int((data.get("left") or [0])[i])
            y = int((data.get("top") or [0])[i])
            w = int((data.get("width") or [0])[i])
            h = int((data.get("height") or [0])[i])
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        return (x, y, x + w, y + h)
    return None


def _roi_motion_score(img_a: Path, img_b: Path, roi: Tuple[int, int, int, int]) -> float:
    a = Image.open(img_a).convert("RGB")
    b = Image.open(img_b).convert("RGB")
    a_roi = a.crop(roi)
    b_roi = b.crop(roi)
    diff = ImageChops.difference(a_roi, b_roi)
    # Score: average absolute difference per channel (0..255)
    hist = diff.histogram()
    if not hist or len(hist) < 256 * 3:
        return 0.0
    total = 0
    count = 0
    for channel in range(3):
        h = hist[channel * 256 : (channel + 1) * 256]
        for value, freq in enumerate(h):
            total += value * freq
            count += freq
    return float(total) / float(count or 1)


@dataclass
class MediaVerifyResult:
    ok: bool
    report_path: Path
    artifacts_dir: Path
    artifacts: Dict[str, Any]


def run_media_verify(
    *,
    root_dir: Path,
    mission_id: str,
    browser_process: str = "brave.exe",
    expected_artist: str = "Extremoduro",
    expected_title_hint: str = "Standby",
    time_window_s: float = 4.0,
    min_time_delta_s: float = 2.0,
    audio_motion_threshold: float = 2.0,
) -> MediaVerifyResult:
    """
    Verificación operacional para 'media playback' basada en observables (screenshots+OCR+UIA).
    No ejecuta la misión; solo observa y produce artefactos/report.
    """
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe_mission = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(mission_id or "mission").strip())[:64]
    artifacts_dir = root_dir / "artifacts" / "media_verify" / safe_mission
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / f"report_{safe_mission}.json"

    driver = WindowsDriverClient()

    # Deference: si hay humano activo, pausar verificaciones (no robar foco).
    policy = load_explore_policy(root_dir)
    signal = read_human_signal(root_dir, policy=policy)
    human_active, human_reason = compute_human_active(signal, threshold_s=2.0, unknown_as_human=True)
    if human_active:
        payload = {
            "schema_version": "0.1",
            "mission_id": mission_id,
            "created_utc": ts,
            "ok": False,
            "status": "WAITING_FOR_USER",
            "cause": "deference_human_active",
            "human_active_reason": human_reason,
            "human_signal": signal,
            "artifacts": {},
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return MediaVerifyResult(ok=False, report_path=report_path, artifacts_dir=artifacts_dir, artifacts=payload.get("artifacts", {}))

    # Focus browser and capture player screenshots.
    try:
        driver.window_focus(process=browser_process)
        time.sleep(0.2)
    except Exception:
        pass

    shot_player_t0 = Path(str(driver.screenshot()))
    player_path = artifacts_dir / "screenshot_player.png"
    player_path.write_bytes(shot_player_t0.read_bytes())

    # Try to infer title/yt from OCR + foreground title.
    fg = {}
    try:
        fg = driver.get_active_window()
    except Exception:
        fg = {}
    fg_title = str((fg or {}).get("title") or "")
    player_ocr = _ocr_text(player_path)

    # Wait and capture second frame for currentTime delta.
    time.sleep(max(0.5, float(time_window_s)))
    shot_player_t1 = Path(str(driver.screenshot()))
    player_path_2 = artifacts_dir / "screenshot_player_t1.png"
    player_path_2.write_bytes(shot_player_t1.read_bytes())

    t0 = _extract_youtube_current_time_seconds(player_path)
    t1 = _extract_youtube_current_time_seconds(player_path_2)
    delta = None
    if t0 is not None and t1 is not None:
        delta = max(0, int(t1 - t0))

    # Volume mixer: open sndvol and measure motion in Brave meter.
    try:
        driver.app_launch(process="sndvol.exe")
        time.sleep(0.8)
    except Exception:
        pass

    mixer_shot_a = Path(str(driver.screenshot()))
    mixer_path_a = artifacts_dir / "screenshot_volume_mixer_t0.png"
    mixer_path_a.write_bytes(mixer_shot_a.read_bytes())
    time.sleep(0.7)
    mixer_shot_b = Path(str(driver.screenshot()))
    mixer_path_b = artifacts_dir / "screenshot_volume_mixer.png"
    mixer_path_b.write_bytes(mixer_shot_b.read_bytes())

    brave_bbox = _find_word_bbox(mixer_path_b, "Brave")
    roi = None
    motion_score = 0.0
    if brave_bbox:
        x1, y1, x2, y2 = brave_bbox
        img = Image.open(mixer_path_b)
        w, h = img.size
        # Meter suele estar a la derecha del nombre de la app.
        roi = (
            min(w - 1, max(0, x2 + 20)),
            max(0, y1 - 25),
            w,
            min(h, y2 + 25),
        )
        motion_score = _roi_motion_score(mixer_path_a, mixer_path_b, roi)

    audio_active = bool(motion_score >= float(audio_motion_threshold))

    # Checks
    title_ok = expected_artist.lower() in (fg_title.lower() + " " + player_ocr.lower())
    standby_ok = ("stand by" in (fg_title.lower() + " " + player_ocr.lower())) or ("standby" in (fg_title.lower() + " " + player_ocr.lower()))
    youtube_ok = "youtube.com" in player_ocr.lower() or "youtube" in fg_title.lower()
    time_ok = (delta is not None) and (float(delta) >= float(min_time_delta_s))

    ok = bool(youtube_ok and title_ok and standby_ok and time_ok and audio_active)
    cause = None
    if not youtube_ok:
        cause = "not_on_youtube"
    elif not (title_ok and standby_ok):
        cause = "title_mismatch"
    elif not time_ok:
        cause = "playback_time_not_advancing"
    elif not audio_active:
        cause = "no_audio_meter_activity"

    player_state = {
        "url": None,
        "title": fg_title,
        "t0": t0,
        "t1": t1,
        "delta": delta,
        "paused": None,
        "timestamp_utc": ts,
    }
    player_state_path = artifacts_dir / "player_state.json"
    player_state_path.write_text(json.dumps(player_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema_version": "0.1",
        "mission_id": mission_id,
        "created_utc": ts,
        "rail": "prod",
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
        "cause": cause,
        "checks": {
            "youtube_tab_active": youtube_ok,
            "title_contains_extremoduro": title_ok,
            "title_contains_standby": standby_ok,
            "currentTime_delta_ok": time_ok,
            "audio_meter_activity": audio_active,
        },
        "measurements": {
            "currentTime_t0": t0,
            "currentTime_t1": t1,
            "currentTime_delta_s": delta,
            "audio_motion_score": motion_score,
            "audio_roi": roi,
        },
        "receipts_paths": {
            "screenshot_player_png": str(player_path),
            "screenshot_volume_mixer_png": str(mixer_path_b),
            "player_state_json": str(player_state_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return MediaVerifyResult(ok=ok, report_path=report_path, artifacts_dir=artifacts_dir, artifacts=report.get("receipts_paths", {}))

