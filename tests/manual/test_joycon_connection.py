"""Phase 1 manual test: read both Joy-Cons live, log every report, check pass criteria.

Run from the project root:
    .venv\\Scripts\\python.exe tests\\manual\\test_joycon_connection.py --duration 90

While it runs (the screen tells you the same):
  1. Keep both Joy-Cons still on the desk for the first ~5 s (gravity check).
  2. Move BOTH sticks at the same time in circles.
  3. Press every button once (ZL, ZR, L, R, arrows, ABXY, -, +, stick clicks, SL/SR).
  4. Shake / rotate each Joy-Con.
  5. Disconnect one Joy-Con (hold its SYNC button briefly, or switch Bluetooth off
     and on), then wait for it to reconnect.

Every HID report is written to data/logs/joycon_<timestamp>.jsonl, and a summary
with pass/fail per criterion goes to the matching .summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.input.joycon_device import JoyConDevice, enumerate_joycons  # noqa: E402

RECONNECT_INTERVAL_S = 2.0
FIRST_REPORT_TIMEOUT_S = 3.0
STICK_DEFLECT_RAW = 400      # of 0..4095, ~20% of half range
STICK_FULL_RAW = 1400        # approx. center-to-edge travel, only for the on-screen %
STILL_WINDOW_S = 5.0
GYRO_MOTION_DPS = 100.0
MIN_RUN_S = 60.0

BUTTON_LABELS = {
    "L": {"up": "↑", "down": "↓", "left": "←", "right": "→", "l": "L", "zl": "ZL",
          "minus": "-", "capture": "캡처", "sl": "SL", "sr": "SR", "stick": "스틱누르기"},
    "R": {"a": "A", "b": "B", "x": "X", "y": "Y", "r": "R", "zr": "ZR",
          "plus": "+", "home": "HOME", "sl": "SL", "sr": "SR", "stick": "스틱누르기"},
}


class Session:
    def __init__(self, log_path: Path):
        self.devices = {side: JoyConDevice(side) for side in ("L", "R")}
        self.log_path = log_path
        self.log = log_path.open("w", encoding="utf-8")
        self.events: list[dict] = []
        self.samples: dict[str, list] = {"L": [], "R": []}
        self.t0 = time.perf_counter()
        self._last_attempt = {"L": -1e9, "R": -1e9}
        self._connected_at: dict[str, float | None] = {"L": None, "R": None}
        self._was_connected = {"L": False, "R": False}
        self._stop = threading.Event()
        self._thread_errors: list[str] = []
        threading.excepthook = self._thread_excepthook
        # live checklist state, updated in flush()
        self.track = {side: {"first_t": None, "center": None, "prev": {}, "pressed": set(),
                             "still_mags": [], "dev_pct_max": 0.0, "last_deflect_t": -1e9,
                             "gyro_peak": 0.0} for side in ("L", "R")}
        self.simultaneous = False
        self.all_done_at: float | None = None

    def _thread_excepthook(self, args):
        # pyjoycon's reader thread dies with OSError when the device goes away.
        msg = f"{args.thread.name}: {args.exc_type.__name__}: {args.exc_value}"
        self._thread_errors.append(msg)
        self.event("thread_error", None, detail=msg)

    def event(self, kind: str, side: str | None, **extra) -> None:
        rec = {"type": "event", "event": kind, "side": side,
               "t": round(time.perf_counter(), 6), "wall": round(time.time(), 6), **extra}
        self.events.append(rec)
        self.log.write(json.dumps(rec) + "\n")

    # -- connection manager (background thread) ---------------------------

    def manage_connections(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            for side, dev in self.devices.items():
                if dev.connected:
                    if not self._was_connected[side]:
                        self._was_connected[side] = True
                        self.event("first_report", side)
                    continue
                if self._was_connected[side]:
                    self._was_connected[side] = False
                    self._connected_at[side] = None
                    dev.mark_disconnected()
                    self.event("disconnected", side)
                started = self._connected_at[side]
                if started is not None and dev.awaiting_first_report():
                    if now - started < FIRST_REPORT_TIMEOUT_S:
                        continue
                    dev.mark_disconnected()
                    self._connected_at[side] = None
                    self.event("no_reports_after_open", side)
                if now - self._last_attempt[side] < RECONNECT_INTERVAL_S:
                    continue
                self._last_attempt[side] = now
                if dev.connect():
                    self._connected_at[side] = time.perf_counter()
                    self.event("opened", side)
                elif dev.last_error and not dev.last_error.startswith("not found"):
                    self.event("open_failed", side, detail=dev.last_error)
            self._stop.wait(0.1)

    # -- main loop --------------------------------------------------------

    def flush(self) -> None:
        for side, dev in self.devices.items():
            for s in dev.drain():
                self.samples[side].append(s)
                self._track(s)
                self.log.write(json.dumps(s.to_record()) + "\n")
        self.log.flush()

    def _track(self, s) -> None:
        tr = self.track[s.side]
        if tr["first_t"] is None:
            tr["first_t"], tr["center"] = s.t, s.stick_raw
        if s.t - tr["first_t"] < STILL_WINDOW_S:
            tr["still_mags"].extend(math.sqrt(sum(v * v for v in sub)) for sub in s.accel_g)
        for name, v in s.buttons.items():
            if v and not tr["prev"].get(name, 0):
                tr["pressed"].add(name)
        tr["prev"] = s.buttons
        cx, cy = tr["center"]
        dev = max(abs(s.stick_raw[0] - cx), abs(s.stick_raw[1] - cy))
        tr["dev_pct_max"] = max(tr["dev_pct_max"], 100 * dev / STICK_FULL_RAW)
        if dev > STICK_DEFLECT_RAW:
            tr["last_deflect_t"] = s.t
            other = self.track["R" if s.side == "L" else "L"]
            if abs(s.t - other["last_deflect_t"]) <= 0.1:
                self.simultaneous = True
        tr["gyro_peak"] = max(tr["gyro_peak"], max(abs(v) for sub in s.gyro_dps for v in sub))

    def checklist(self) -> list[tuple[bool, str, list[str]]]:
        """(done, title, detail lines) for each on-screen step."""
        now = time.perf_counter()
        tl, tr_ = self.track["L"], self.track["R"]

        def still_status(t):
            if t["first_t"] is None:
                return False, "연결 대기"
            left = STILL_WINDOW_S - (now - t["first_t"])
            if left > 0:
                return False, f"측정 중 {left:.0f}초"
            return True, f"{statistics.median(t['still_mags']):.2f}G"
        ok_l, txt_l = still_status(tl)
        ok_r, txt_r = still_status(tr_)
        steps = [(ok_l and ok_r, "두 조이콘을 책상에 5초간 가만히 두기 (중력 측정)",
                  [f"L {txt_l}   R {txt_r}"])]

        def live_pct(side):
            s = self.devices[side].latest()
            c = self.track[side]["center"]
            if not s or not c:
                return 0
            return 100 * max(abs(s.stick_raw[0] - c[0]), abs(s.stick_raw[1] - c[1])) / STICK_FULL_RAW
        steps.append((self.simultaneous, "양손으로 두 스틱을 '동시에' 끝까지 빙글빙글 돌리기",
                      [f"지금 L {live_pct('L'):3.0f}%  R {live_pct('R'):3.0f}%   "
                       f"(최대 L {tl['dev_pct_max']:3.0f}%  R {tr_['dev_pct_max']:3.0f}%)"]))

        missing = {side: [BUTTON_LABELS[side][b] for b in BUTTON_LABELS[side]
                          if b not in self.track[side]["pressed"]] for side in ("L", "R")}
        steps.append((not missing["L"] and not missing["R"], "버튼을 전부 한 번씩 누르기", [
            "L 남은 버튼: " + (" ".join(missing["L"]) or "없음"),
            "R 남은 버튼: " + (" ".join(missing["R"]) or "없음"),
            "(SL/SR = 조이콘 레일 쪽 작은 버튼, 스틱누르기 = 스틱을 꾹 누르기)",
        ]))

        steps.append((tl["gyro_peak"] > GYRO_MOTION_DPS and tr_["gyro_peak"] > GYRO_MOTION_DPS,
                      "두 조이콘을 각각 손목으로 세게 흔들기",
                      [f"L {tl['gyro_peak']:4.0f} dps   R {tr_['gyro_peak']:4.0f} dps   (둘 다 {GYRO_MOTION_DPS:.0f} 이상)"]))

        def reconnected(side):
            evs = [e["event"] for e in self.events if e["side"] == side]
            return "disconnected" in evs and evs[::-1].index("first_report") < evs[::-1].index("disconnected") \
                if "first_report" in evs and "disconnected" in evs else False
        recon = reconnected("L") or reconnected("R")
        disc = any(e["event"] == "disconnected" for e in self.events)
        steps.append((recon, "한쪽 조이콘의 SYNC 버튼을 짧게 눌러 연결 끊기 → 몇 초 뒤 아무 버튼 눌러 재연결",
                      ["재연결 완료" if recon else ("끊김 확인, 아무 버튼을 눌러 다시 연결하세요" if disc else "대기 중")]))
        return steps

    def render(self, elapsed: float) -> str:
        lines = [f"Teach-a-Bot Lab - Phase 1 Joy-Con test   (경과 {elapsed:5.1f} s, Ctrl+C로 중단)",
                 f"log: {self.log_path}", "",
                 "=== 할 일: 위에서부터 차례로 하세요. 끝나면 [O]로 바뀝니다 ==="]
        steps = self.checklist()
        current = next((i for i, (done, _, _) in enumerate(steps) if not done), None)
        for i, (done, title, details) in enumerate(steps):
            mark = "[O]" if done else ("[>]" if i == current else "[ ]")
            lines.append(f"{mark} {i + 1}. {title}")
            lines += [f"       {d}" for d in details]
        if current is None:
            left = max(0.0, MIN_RUN_S - elapsed, (self.all_done_at or 0) + 3 - time.perf_counter())
            lines.append(f">>> 모두 완료! 60초 연속 동작 확인 후 자동 종료합니다 ({left:.0f}초). 그대로 두세요.")
        lines.append("")
        for side, title in (("L", "LEFT JOY-CON"), ("R", "RIGHT JOY-CON")):
            dev = self.devices[side]
            s = dev.latest()
            recent = [x for x in self.samples[side] if x.t > time.perf_counter() - 1.0]
            lines.append(title)
            lines.append(f"Connected : {dev.connected}   reports/s: {len(recent):3d}"
                         + (f"   battery: {s.battery_level}" if s else ""))
            if s and dev.connected:
                pressed = [k for k, v in s.buttons.items() if v] or ["-"]
                ax, ay, az = s.accel_g[0]
                gx, gy, gz = s.gyro_dps[0]
                lines += [
                    f"Stick     : ({s.stick_raw[0]:4d}, {s.stick_raw[1]:4d})  raw 12-bit",
                    f"Buttons   : {' '.join(pressed)}",
                    f"Accel [G] : ({ax:+6.3f}, {ay:+6.3f}, {az:+6.3f})  |a|={math.sqrt(ax*ax+ay*ay+az*az):.3f}",
                    f"Gyro [dps]: ({gx:+8.2f}, {gy:+8.2f}, {gz:+8.2f})",
                ]
            else:
                err = dev.last_error or "waiting..."
                lines += [f"Status    : {err}", "", "", ""]
            lines.append("")
        lines.append("Recent events:")
        for e in self.events[-5:]:
            lines.append(f"  {e['t'] - self.t0:7.2f}s  {e['event']:<22} {e['side'] or ''} {e.get('detail', '')}")
        return "\n".join(lines)

    def run(self, duration: float, display_hz: float, show: bool) -> None:
        manager = threading.Thread(target=self.manage_connections, name="joycon-manager", daemon=True)
        manager.start()
        end = self.t0 + duration
        try:
            while time.perf_counter() < end:
                self.flush()
                now = time.perf_counter()
                if self.all_done_at is None and all(done for done, _, _ in self.checklist()):
                    self.all_done_at = now
                    self.event("checklist_complete", None)
                if self.all_done_at is not None and now - self.t0 >= MIN_RUN_S and now - self.all_done_at >= 3:
                    break
                if show:
                    sys.stdout.write("\x1b[H\x1b[J" + self.render(now - self.t0) + "\n")
                    sys.stdout.flush()
                time.sleep(1.0 / display_hz)
        except KeyboardInterrupt:
            self.event("stopped_by_user", None)
        finally:
            self._stop.set()
            manager.join(2.0)
            self.flush()
            self.event("end", None)
            self.log.close()


# -- summary / pass criteria ---------------------------------------------------

def side_summary(samples: list, events: list[dict], side: str) -> dict:
    if not samples:
        return {"reports": 0}
    ts = [s.t for s in samples]
    span = ts[-1] - ts[0]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    timer_deltas = defaultdict(int)
    for a, b in zip(samples, samples[1:]):
        timer_deltas[(b.timer - a.timer) % 256] += 1

    # gravity check on the first STILL_WINDOW_S seconds after first report
    still = [s for s in samples if s.t - ts[0] < STILL_WINDOW_S]
    mags = [math.sqrt(sum(v * v for v in sub)) for s in still for sub in s.accel_g]
    gyro_peak = max(abs(v) for s in samples for sub in s.gyro_dps for v in sub)
    accel_peak = max(math.sqrt(sum(v * v for v in sub)) for s in samples for sub in s.accel_g)

    xs = [s.stick_raw[0] for s in samples]
    ys = [s.stick_raw[1] for s in samples]

    presses: dict[str, int] = defaultdict(int)
    for a, b in zip(samples, samples[1:]):
        for name, v in b.buttons.items():
            if v and not a.buttons.get(name, 0):
                presses[name] += 1
    releases_ok = all(v == 0 for v in samples[-1].buttons.values())

    return {
        "reports": len(samples),
        "span_s": round(span, 2),
        "report_rate_hz": round((len(samples) - 1) / span, 1) if span > 0 else None,
        "imu_rate_hz": round(3 * (len(samples) - 1) / span, 1) if span > 0 else None,
        "gap_ms_median": round(1000 * statistics.median(gaps), 2) if gaps else None,
        "gap_ms_p99": round(1000 * sorted(gaps)[int(0.99 * (len(gaps) - 1))], 2) if gaps else None,
        "gap_ms_max": round(1000 * max(gaps), 1) if gaps else None,
        "timer_delta_histogram": dict(sorted(timer_deltas.items(), key=lambda kv: -kv[1])[:5]),
        "still_accel_norm_g_median": round(statistics.median(mags), 4) if mags else None,
        "accel_norm_g_peak": round(accel_peak, 3),
        "gyro_dps_peak": round(gyro_peak, 1),
        "stick_x_range": [min(xs), max(xs)],
        "stick_y_range": [min(ys), max(ys)],
        "stick_center_first": list(samples[0].stick_raw),
        "button_presses": dict(sorted(presses.items())),
        "all_buttons_released_at_end": releases_ok,
        "disconnects": sum(1 for e in events if e["event"] == "disconnected" and e["side"] == side),
        "reconnects_with_data": max(0, sum(1 for e in events if e["event"] == "first_report" and e["side"] == side) - 1),
    }


def simultaneous_sticks(samples: dict[str, list]) -> bool:
    def deflected(side):
        ss = samples[side]
        if not ss:
            return []
        cx, cy = ss[0].stick_raw
        return [s.t for s in ss if abs(s.stick_raw[0] - cx) > STICK_DEFLECT_RAW
                or abs(s.stick_raw[1] - cy) > STICK_DEFLECT_RAW]
    lt, rt = deflected("L"), deflected("R")
    j = 0
    for t in lt:
        while j < len(rt) and rt[j] < t - 0.1:
            j += 1
        if j < len(rt) and abs(rt[j] - t) <= 0.1:
            return True
    return False


EXPECTED_BUTTONS = {
    "L": {"up", "down", "left", "right", "l", "zl", "minus", "stick", "capture", "sl", "sr"},
    "R": {"a", "b", "x", "y", "r", "zr", "plus", "stick", "home", "sl", "sr"},
}


def evaluate(session: Session) -> dict:
    per_side = {side: side_summary(session.samples[side], session.events, side) for side in ("L", "R")}
    present = enumerate_joycons()
    checks = {}
    checks["L_and_R_identified"] = all(per_side[s]["reports"] > 0 for s in ("L", "R"))
    checks["simultaneous_sticks"] = simultaneous_sticks(session.samples)
    for side in ("L", "R"):
        ps = per_side[side]
        if not ps["reports"]:
            continue
        missing = sorted(EXPECTED_BUTTONS[side] - set(ps["button_presses"]))
        checks[f"{side}_buttons_all_pressed"] = not missing
        if missing:
            checks[f"{side}_buttons_missing"] = missing
        g = ps["still_accel_norm_g_median"]
        checks[f"{side}_gravity_1g_at_rest"] = g is not None and 0.9 <= g <= 1.1
        checks[f"{side}_imu_responds_to_motion"] = ps["gyro_dps_peak"] > 100
    total_span = time.perf_counter() - session.t0
    checks["ran_at_least_60s"] = total_span >= 60.0
    checks["no_fatal_error"] = True  # we got here
    checks["disconnect_survived"] = any(ps.get("disconnects", 0) > 0 for ps in per_side.values())
    checks["reconnected_after_disconnect"] = any(ps.get("reconnects_with_data", 0) > 0 for ps in per_side.values())
    checks["log_saved"] = session.log_path.exists() and session.log_path.stat().st_size > 0
    return {
        "log": str(session.log_path),
        "run_seconds": round(total_span, 1),
        "hid_devices_at_end": [{"product_id": hex(d["product_id"]), "product_string": d["product_string"]} for d in present],
        "per_side": per_side,
        "thread_errors": session._thread_errors,
        "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=300.0,
                    help="max seconds to run (default 300); ends earlier once the checklist is done and 60 s passed")
    ap.add_argument("--display-hz", type=float, default=10.0)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "data" / "logs")
    args = ap.parse_args()

    os.system("")  # enable ANSI escape handling in the Windows console
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"joycon_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    session = Session(log_path)
    session.event("start", None, duration=args.duration,
                  hid_devices=[hex(d["product_id"]) for d in enumerate_joycons()])
    session.run(args.duration, args.display_hz, show=not args.no_display)

    summary = evaluate(session)
    summary_path = log_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
