"""Feasibility recording for "move the Joy-Con = move the gripper" (plan A: IMU strokes + ZUPT).

The screen tells the user, trial by trial, which way and how far to move ONE
Joy-Con (known distances along a ruler / A4 sheet), with still periods before
and after. Every raw HID report (all 3 IMU sub-samples) plus the trial timing
and labels is written to data/logs/strokes_<surface>_<hand>_<time>.jsonl.
Algorithms are evaluated offline afterwards (scripts/analyze_strokes.py), so
the same recording can be re-used for every variant.

    .venv\\Scripts\\python.exe tests\\manual\\record_motion_strokes.py --surface desk
    .venv\\Scripts\\python.exe tests\\manual\\record_motion_strokes.py --surface air
    .venv\\Scripts\\python.exe tests\\manual\\record_motion_strokes.py --surface vertical   # up/down (D18)

Vertical trials (scripts/analyze_vertical.py) are up then down, at normal and slow
speed, plus "tilt only" trials that must NOT change the height.

Trial order keeps the Joy-Con near its start: forward then back, right then left.
Press B (right Joy-Con) or the down arrow (left Joy-Con) during a trial to mark
it as botched; it stays in the file but is flagged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from teachabot.input.joycon_device import JoyConDevice  # noqa: E402

DIR_KO = {"forward": "앞으로", "back": "뒤로(몸 쪽으로)", "right": "오른쪽으로", "left": "왼쪽으로",
          "up": "위로", "down": "아래로", "up_slow": "천천히 위로", "down_slow": "천천히 아래로",
          "tilt": "높이는 그대로 두고 앞으로 기울였다가 되돌리기"}
BLOCK = [("forward", 10), ("back", 10), ("right", 10), ("left", 10),
         ("forward", 20), ("back", 20), ("right", 20), ("left", 20)]
BLOCK_V = [("up", 10), ("down", 10), ("up", 20), ("down", 20),
           ("up_slow", 10), ("down_slow", 10), ("tilt", 0)]
PHASES = (("ready", 2.0), ("move", 2.5), ("hold", 1.5))
BAD_BUTTON = {"R": "b", "L": "down"}
SURFACE_KO = {"desk": "책상 위에서 미끄러뜨리기 (앞면이 위로 눕혀서)",
              "air": "공중에서 평소 쥐는 자세로, 기울이지 말고 손 전체를 옮기기",
              "vertical": "공중에서 평소 쥐는 자세로, 기울이지 말고 손 전체를 위/아래로 옮기기"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", choices=["desk", "air", "vertical"], required=True)
    ap.add_argument("--hand", choices=["R", "L"], default="R")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    os.system("")

    block = BLOCK_V if args.surface == "vertical" else BLOCK
    trials = [(d, cm) for _ in range(args.reps) for d, cm in block]
    dev = JoyConDevice(args.hand)
    log_path = ROOT / "data" / "logs" / f"strokes_{args.surface}_{args.hand}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    write = lambda rec: log.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write({"type": "header", "surface": args.surface, "hand": args.hand, "trials": len(trials),
           "phases": dict(PHASES), "created": datetime.now().isoformat(timespec="seconds")})

    def show(lines):
        sys.stdout.write("\x1b[H\x1b[J" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    # connect
    while not dev.connected:
        if not dev.awaiting_first_report():
            dev.connect()
        show(["Teach-a-Bot Lab - 이동 실측 녹화", "",
              f"Joy-Con ({args.hand}) 연결 중... {dev.last_error or ''}",
              "조이콘 버튼을 하나 눌러 깨워 주세요."])
        time.sleep(0.5)

    def flush():
        for s in dev.drain():
            write(s.to_record())

    # wait until the user is ready (A on the right Joy-Con / right arrow on the left one)
    start_btn = {"R": "a", "L": "right"}[args.hand]
    while True:
        flush()
        s = dev.latest()
        show(["Teach-a-Bot Lab - 이동 실측 녹화", "",
              f"방식: {SURFACE_KO[args.surface]}",
              f"손: {'오른쪽' if args.hand == 'R' else '왼쪽'} 조이콘   시행 {len(trials)}회 (약 {len(trials) * 6 // 60 + 1}분)",
              "준비물: 자(또는 A4 용지: 짧은 변 21cm ≈ 20cm, 그 절반 ≈ 10cm)",
              ("자를 세워 두고 위/아래로 10cm·20cm 갈 높이를 눈으로 확인해 두세요." if args.surface == "vertical"
               else "시작 위치를 정하고, 앞/뒤/좌/우로 10cm·20cm 갈 지점을 눈으로 확인해 두세요."), "",
              f">>> 준비되면 {'A' if args.hand == 'R' else '→'} 버튼을 누르세요"])
        if s and s.buttons.get(start_btn):
            break
        time.sleep(0.05)

    # intro: 6 s still to get a baseline (also skips the ~1 s IMU start-up transient)
    t_intro = time.perf_counter()
    while time.perf_counter() - t_intro < 6.0:
        flush()
        left = 6.0 - (time.perf_counter() - t_intro)
        show(["Teach-a-Bot Lab - 이동 실측 녹화", "",
              f"방식: {SURFACE_KO[args.surface]}",
              f"손: {'오른쪽' if args.hand == 'R' else '왼쪽'} 조이콘   시행 {len(trials)}회 (약 {len(trials) * 6 // 60 + 1}분)",
              "준비물: 자(또는 A4 용지: 짧은 변 21cm ≈ 20cm, 그 절반 ≈ 10cm)", "",
              f">>> 시작 위치에서 조이콘을 가만히 두세요 ({left:3.1f}초)"])
        time.sleep(0.05)
    write({"type": "baseline", "t_start": t_intro + 1.0, "t_end": time.perf_counter()})

    bad_btn = BAD_BUTTON[args.hand]
    bad = set()
    for i, (direction, cm) in enumerate(trials):
        for phase, dur in PHASES:
            t0 = time.perf_counter()
            write({"type": "phase", "trial": i, "direction": direction, "distance_cm": cm, "phase": phase, "t": t0})
            prev_bad = False
            while (now := time.perf_counter()) - t0 < dur:
                flush()
                s = dev.latest()
                pressed = bool(s and s.buttons.get(bad_btn))
                if pressed and not prev_bad and i not in bad:
                    bad.add(i)
                    write({"type": "bad", "trial": i, "t": now})
                prev_bad = pressed
                left = dur - (now - t0)
                bar = "#" * int(30 * (now - t0) / dur)
                if phase == "ready":
                    head = f">>> 준비: 가만히 ({left:3.1f}초)   다음: {DIR_KO[direction]}" + (f" {cm}cm" if cm else "")
                elif phase == "move":
                    head = (f">>> 지금! {DIR_KO[direction]}" if direction == "tilt"
                            else f">>> 지금! {DIR_KO[direction]} {cm}cm 옮기고 멈추세요")
                else:
                    head = ">>> 멈춘 채로 가만히"
                show([f"Teach-a-Bot Lab - 이동 실측 녹화  [{args.surface}]   시행 {i + 1}/{len(trials)}", "",
                      head, f"[{bar:<30}]", "",
                      f"방식: {SURFACE_KO[args.surface]}",
                      f"잘못했으면 이 시행 중에 {'B' if args.hand == 'R' else '↓'} 버튼 → 제외 표시"
                      + ("   (이 시행: 제외됨)" if i in bad else ""),
                      f"연결: {dev.connected}   제외 표시: {len(bad)}회"])
                time.sleep(0.02)
    flush()
    write({"type": "end", "t": time.perf_counter(), "bad_trials": sorted(bad)})
    log.close()
    dev.mark_disconnected()
    show(["녹화 완료!", f"파일: {log_path}", f"시행 {len(trials)}회, 제외 표시 {len(bad)}회"])
    print(json.dumps({"log": str(log_path), "trials": len(trials), "bad": sorted(bad)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
