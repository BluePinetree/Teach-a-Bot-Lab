"""Phase 0 (link manager) diagnosis: log every Joy-Con link change with timestamps.

Run from the project root, in a window you can see:
    .venv\\Scripts\\python.exe scripts\\joycon_link_log.py --note usb-suspend-on
    .venv\\Scripts\\python.exe scripts\\joycon_link_log.py --hold-hid --note idle-app

Per side it tracks three layers, and writes a line whenever one changes:
  bt   Windows Bluetooth link (WinRT ConnectionStatus, event + 0.5 s poll)
  hid  Joy-Con listed in HID enumeration
  rx   (--hold-hid only) HID reports actually arriving. This opens the device
       like the app does, so do NOT run it next to the app.

Keys while running: l / r = "I just pressed a button on L / R" (wake attempt),
s = "I just pressed SYNC", m = free marker, q = quit.
With --hold-hid, button presses and motion are also detected from the reports,
so the log shows how long each Joy-Con was untouched before it dropped.

Output: data/logs/link_<timestamp>.jsonl and .summary.json (wake latency per attempt).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import msvcrt
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.btlink.winbt import BtLink, load_macs  # noqa: E402
from teachabot.input.joycon_device import PRODUCT_IDS, JoyConDevice, enumerate_joycons  # noqa: E402

TICK_S = 0.5
HEARTBEAT_S = 60.0
HID_RETRY_S = 2.0
MOTION_DPS = 30.0          # gyro above this = someone is handling the Joy-Con
WAKE_WINDOW_S = 120.0      # a wake attempt counts as failed if no link within this
SIDES = ("L", "R")


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.f = path.open("w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.events: list[dict] = []

    def now(self) -> float:
        return round(time.perf_counter() - self.t0, 3)

    def write(self, kind: str, echo: bool = True, **fields) -> None:
        rec = {"t": self.now(), "wall": datetime.now().isoformat(timespec="milliseconds"),
               "type": kind, **fields}
        self.events.append(rec)
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()
        if echo:
            detail = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"\r\x1b[K[{rec['wall'][11:19]} +{rec['t']:8.1f}s] {kind:<10} {detail}")


def summarize(events: list[dict]) -> dict:
    """Wake attempts (key markers) -> seconds until bt / hid / rx came up."""
    out: dict = {"disconnects": [], "wake_attempts": []}
    for i, e in enumerate(events):
        if e["type"] == "change" and e["layer"] == "bt" and e["value"] is False:
            out["disconnects"].append({"side": e["side"], "t": e["t"], "wall": e["wall"],
                                       "idle_s": e.get("idle_s")})
        if e["type"] == "mark" and e.get("side") in SIDES:
            side, t = e["side"], e["t"]
            attempt = {"side": side, "t": t, "wall": e["wall"], "what": e["what"]}
            for layer in ("bt", "hid", "rx"):
                up = next((x["t"] for x in events[i + 1:]
                           if x["type"] == "change" and x["side"] == side and x["layer"] == layer
                           and x["value"] is True and x["t"] - t <= WAKE_WINDOW_S), None)
                attempt[f"{layer}_after_s"] = None if up is None else round(up - t, 2)
            out["wake_attempts"].append(attempt)
    return out


async def run(args) -> None:
    macs = load_macs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(log_dir / f"link_{stamp}.jsonl")
    log.write("start", note=args.note, hold_hid=args.hold_hid, macs=macs,
              keepalive=args.keepalive, keepalive_s=args.keepalive_s,
              keepalive_kind=args.keepalive_kind)

    loop = asyncio.get_running_loop()
    bt_events: asyncio.Queue = asyncio.Queue()

    def on_change(side: str, connected: bool) -> None:  # WinRT worker thread
        loop.call_soon_threadsafe(bt_events.put_nowait, (side, connected, log.now()))

    links = {s: BtLink(s, macs[s]) for s in SIDES}
    for s in SIDES:
        st = await links[s].open(on_change)
        log.write("bt_device", side=s, found=st.found, paired=st.paired, name=st.name)

    devices = {s: JoyConDevice(s) for s in SIDES} if args.hold_hid else {}
    last_hid_try = {s: 0.0 for s in SIDES}
    last_keepalive = {s: 0.0 for s in SIDES}
    keepalive_fail = {s: 0 for s in SIDES}
    last_activity = {s: log.now() for s in SIDES}
    rx_count = {s: 0 for s in SIDES}
    state = {s: {"bt": None, "hid": None, "rx": None} for s in SIDES}
    last_heartbeat = 0.0
    deadline = None if args.minutes is None else time.perf_counter() + args.minutes * 60

    def set_layer(side: str, layer: str, value: bool, t: float | None = None) -> None:
        prev = state[side][layer]
        if prev == value:
            return
        state[side][layer] = value
        fields = {"side": side, "layer": layer, "value": value}
        if prev is None:
            log.write("initial", **fields)
            return
        if not value:
            fields["idle_s"] = round(log.now() - last_activity[side], 1)
        if t is not None:
            fields["event_t"] = t
        log.write("change", **fields)

    print("조이콘 연결 기록 중. l/r = 그쪽 버튼 눌렀음, s = SYNC 눌렀음, m = 메모, q = 종료")
    try:
        while deadline is None or time.perf_counter() < deadline:
            while not bt_events.empty():
                side, connected, t = bt_events.get_nowait()
                set_layer(side, "bt", connected, t)

            present = {d["product_id"] for d in enumerate_joycons()}
            for s in SIDES:
                set_layer(s, "bt", links[s].state().connected)
                set_layer(s, "hid", PRODUCT_IDS[s] in present)

                if args.hold_hid:
                    dev = devices[s]
                    for sample in dev.drain():
                        rx_count[s] += 1
                        gyro = max(math.sqrt(sum(v * v for v in g)) for g in sample.gyro_dps)
                        if any(sample.buttons.values()) or gyro > MOTION_DPS:
                            last_activity[s] = log.now()
                    up = dev.connected
                    if not up and state[s]["rx"]:
                        dev.mark_disconnected()
                    set_layer(s, "rx", up)
                    now = time.perf_counter()
                    if up and s in args.keepalive and now - last_keepalive[s] >= args.keepalive_s:
                        last_keepalive[s] = now
                        if not dev.send_keepalive(args.keepalive_kind):
                            keepalive_fail[s] += 1
                    if (not up and not dev.awaiting_first_report() and PRODUCT_IDS[s] in present
                            and now - last_hid_try[s] > HID_RETRY_S):
                        last_hid_try[s] = now
                        ok = await asyncio.to_thread(dev.connect)
                        log.write("hid_open", side=s, ok=ok, error=dev.last_error)

            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "q":
                    return
                if key in ("l", "r"):
                    last_activity[key.upper()] = log.now()
                    log.write("mark", side=key.upper(), what="button")
                elif key == "s":
                    log.write("mark", side=None, what="sync")
                elif key == "m":
                    log.write("mark", side=None, what="note")

            if log.now() - last_heartbeat >= HEARTBEAT_S:
                last_heartbeat = log.now()
                log.write("heartbeat", echo=False, state=state, rx_count=dict(rx_count),
                          keepalive_fail=dict(keepalive_fail),
                          idle_s={s: round(log.now() - last_activity[s], 1) for s in SIDES})

            line = "  ".join(
                f"{s}: bt={'O' if state[s]['bt'] else '-'} hid={'O' if state[s]['hid'] else '-'}"
                + (f" rx={'O' if state[s]['rx'] else '-'} idle={log.now() - last_activity[s]:5.0f}s"
                   if args.hold_hid else "")
                for s in SIDES)
            print(f"\r\x1b[K+{log.now():7.0f}s  {line}", end="", flush=True)
            await asyncio.sleep(TICK_S)
    finally:
        log.write("stop", rx_count=dict(rx_count))
        for link in links.values():
            link.close()
        summary = {"log": log.path.name, "note": args.note, "hold_hid": args.hold_hid,
                   "duration_s": log.now(), **summarize(log.events)}
        summary_path = log.path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {log.path}\n요약: {summary_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--note", default="", help="experiment label stored in the log")
    p.add_argument("--hold-hid", action="store_true", help="open HID and stream like the app")
    p.add_argument("--minutes", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--keepalive", nargs="*", default=[], choices=SIDES,
                   help="(with --hold-hid) send a silent rumble report to these sides")
    p.add_argument("--keepalive-s", type=float, default=1.0, help="keep-alive interval")
    p.add_argument("--keepalive-kind", choices=["rumble", "lamp"], default="rumble")
    args = p.parse_args()
    if args.keepalive and not args.hold_hid:
        p.error("--keepalive needs --hold-hid")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
