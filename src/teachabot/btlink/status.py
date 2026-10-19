"""Small JSON status files shared by the app and the link manager.

    data/runtime/joycon_app_status.json   written by the app (JoyConInputProvider):
        per side, is it delivering reports and when was a button last pressed
    data/runtime/joycon_link_status.json  written by the link manager:
        per side, state / message / seconds until the Joy-Con's own sleep

Measured 2026-10-19~20 (docs/JOYCON_LINK_MANAGER_PLAN.md section 9): a Joy-Con
sleeps 15 min after its last button press; only button presses reset that
timer. Pure stdlib so both processes and the tests can import it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME_DIR = ROOT / "data" / "runtime"
APP_STATUS_PATH = RUNTIME_DIR / "joycon_app_status.json"
LINK_STATUS_PATH = RUNTIME_DIR / "joycon_link_status.json"

SLEEP_AFTER_S = 15 * 60      # Joy-Con sleeps this long after its last button press
WARN_BEFORE_S = 3 * 60       # start warning this long before
FRESH_S = 3.0                # a status file older than this means its writer is gone


def write_json(path: Path, data: dict, attempts: int = 5) -> bool:
    """Atomic write (readers never see half a file). Returns False if it could not.

    On Windows the replace fails with PermissionError while the other process
    happens to have the file open for reading (seen 2026-10-20: it killed the
    manager), so retry briefly and then give up for this round.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.01 * (i + 1))
    return False


def read_fresh(path: Path, now_wall: float | None = None, max_age_s: float = FRESH_S) -> dict | None:
    """The file's content if its writer updated it recently, else None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    now_wall = time.time() if now_wall is None else now_wall
    if now_wall - float(data.get("updated_wall", 0)) > max_age_s:
        return None
    return data


def sleep_remaining_s(last_input_wall: float, now_wall: float) -> float:
    return SLEEP_AFTER_S - (now_wall - last_input_wall)


def fmt_mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def warning_line(link: dict | None) -> str | None:
    """One HUD line from the manager's status: sleep countdown or what to do now."""
    if not link:
        return None
    parts = []
    for side in ("L", "R"):
        st = link.get("sides", {}).get(side)
        if not st:
            continue
        if st.get("state") in ("repairing", "unpaired"):
            parts.append(f"{side}: {st.get('message', '')}")
        elif st.get("state") == "ready":
            remaining = st.get("remaining_s")
            if remaining is None and st.get("warn"):
                parts.append(f"{side} 잠들기까지 남은 시간 모름 — 아무 버튼이나 한 번 눌러 주세요")
            elif remaining is not None and remaining <= WARN_BEFORE_S:
                parts.append(f"{side} {fmt_mmss(remaining)} 뒤 잠듦 — 아무 버튼이나 한 번 눌러 주세요")
    return "  ·  ".join(parts) or None
