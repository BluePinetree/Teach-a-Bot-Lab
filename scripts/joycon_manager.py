"""Joy-Con link manager window: keeps L/R paired, connected and awake.

    .venv\\Scripts\\python.exe scripts\\joycon_manager.py            # small always-on-top window
    .venv\\Scripts\\python.exe scripts\\joycon_manager.py --cli      # text only (one line per second)
    .venv\\Scripts\\python.exe scripts\\joycon_manager.py --no-auto-repair

Start it before the app and leave it running. Per Joy-Con it shows what to do
next: press SYNC (pair / re-pair), press any button (asleep, or 15-minute sleep
coming up), or nothing (ready). A stuck Joy-Con is re-paired automatically
(only the addresses in config/joycon_link.json). Does not open the Joy-Cons'
HID while the app is running, so it can stay open next to the app.
Log: data/logs/link_manager_<timestamp>.log
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.btlink.manager import SIDES, LinkManager  # noqa: E402

COLORS = {"ready": "#2e9e4f", "warn": "#d9a400", "waking": "#3b7dd8", "asleep": "#8a8f98",
          "stuck": "#d9622b", "repairing": "#d9622b", "unpaired": "#c0392b", "unknown": "#8a8f98"}
TITLES = {"ready": "준비됨", "waking": "연결 중", "asleep": "잠듦", "stuck": "고착",
          "repairing": "재페어링", "unpaired": "페어링 필요", "unknown": "확인 중"}


class Logger:
    def __init__(self):
        log_dir = ROOT / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"link_manager_{datetime.now():%Y%m%d_%H%M%S}.log"
        self.f = self.path.open("a", encoding="utf-8")
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        self.f.write(line + "\n")
        self.f.flush()
        self.lines = (self.lines + [line])[-4:]
        print(line, flush=True)


class Runner:
    """Runs the asyncio LinkManager in a background thread."""

    def __init__(self, manager: LinkManager, log: Logger):
        self.manager = manager
        self.log = log
        self.error: str | None = None
        self.loop = asyncio.new_event_loop()
        self.stop_event: asyncio.Event | None = None
        self.thread = threading.Thread(target=self._run, name="link-manager", daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.stop_event = asyncio.Event()
        try:
            self.loop.run_until_complete(self.manager.run(self.stop_event))
        except Exception as e:      # show it in the window instead of vanishing
            self.error = f"{type(e).__name__}: {e}"
            self.log("관리자 오류로 멈춤: " + self.error)
            self.log(traceback.format_exc())

    def start(self) -> None:
        self.thread.start()

    def repair(self, side: str) -> None:
        self.loop.call_soon_threadsafe(self.manager.request_repair, side)

    def stop(self) -> None:
        if self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(3.0)


def run_cli(runner: Runner) -> None:
    try:
        while runner.thread.is_alive():
            snap = runner.manager.snapshot()
            line = "   ".join(f"{s}: [{TITLES.get(v['state'], v['state'])}] {v['message']}"
                              for s, v in snap["sides"].items())
            app = "앱 실행 중" if snap["app_alive"] else "앱 꺼짐"
            print(f"\r\x1b[K{datetime.now():%H:%M:%S} ({app})  {line}", end="", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    print()


def run_window(runner: Runner, log: Logger) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("조이콘 연결 관리자")
    root.attributes("-topmost", True)
    root.configure(bg="#1b1f27")
    root.resizable(False, False)
    widgets = {}
    for col, side in enumerate(SIDES):
        frame = tk.Frame(root, bg="#252a35", padx=12, pady=10)
        frame.grid(row=0, column=col, padx=8, pady=8, sticky="nsew")
        head = tk.Label(frame, text=f"{'왼쪽' if side == 'L' else '오른쪽'} 조이콘 ({side})", fg="white",
                        bg="#252a35", font=("Malgun Gothic", 12, "bold"))
        head.pack(anchor="w")
        state = tk.Label(frame, text="", fg="white", bg="#8a8f98", font=("Malgun Gothic", 18, "bold"),
                         width=12, pady=4)
        state.pack(anchor="w", pady=(6, 4))
        msg = tk.Label(frame, text="", fg="#e8eaf0", bg="#252a35", font=("Malgun Gothic", 11),
                       wraplength=250, justify="left", width=30, height=3, anchor="nw")
        msg.pack(anchor="w")
        btn = tk.Button(frame, text="다시 페어링", command=lambda s=side: runner.repair(s),
                        font=("Malgun Gothic", 9))
        btn.pack(anchor="e", pady=(4, 0))
        widgets[side] = (state, msg)
    footer = tk.Label(root, text="", fg="#aab0bc", bg="#1b1f27", font=("Malgun Gothic", 9), justify="left",
                      anchor="w")
    footer.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=(0, 8))

    def refresh():
        snap = runner.manager.snapshot()
        for side, (state, msg) in widgets.items():
            v = snap["sides"][side]
            key = "warn" if v["state"] == "ready" and v["warn"] else v["state"]
            state.configure(text=TITLES.get(v["state"], v["state"]), bg=COLORS.get(key, "#8a8f98"))
            msg.configure(text=v["message"])
        app = "앱 실행 중 (버튼 입력 반영)" if snap["app_alive"] else "앱 꺼짐"
        auto = "자동 재페어링 켬" if snap["auto_repair"] else "자동 재페어링 끔"
        footer.configure(text=f"{app} · {auto}\n" + "\n".join(log.lines[-3:]))
        if runner.thread.is_alive():
            root.after(250, refresh)
            return
        for state, msg in widgets.values():      # stopped: say so instead of showing stale states
            state.configure(text="멈춤", bg="#c0392b")
            msg.configure(text="관리자가 멈췄습니다. 창을 닫고 다시 실행하세요.\n" + (runner.error or ""))

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    refresh()
    root.mainloop()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cli", action="store_true", help="no window, print status once a second")
    p.add_argument("--no-auto-repair", action="store_true", help="re-pair only when the button is clicked")
    args = p.parse_args()
    log = Logger()
    manager = LinkManager(auto_repair=not args.no_auto_repair, log=log)
    log(f"시작 · MAC {manager.macs or '(아직 없음: 처음 페어링)'} · 자동 재페어링 {'켬' if manager.auto_repair else '끔'}")
    runner = Runner(manager, log)
    runner.start()
    try:
        if args.cli:
            run_cli(runner)
        else:
            run_window(runner, log)
    finally:
        runner.stop()
        log("종료")


if __name__ == "__main__":
    main()
