"""Teach-a-Bot Lab app (Phase 3: teleoperate two grippers in the 3D tabletop).

    .venv\\Scripts\\python.exe -m teachabot.app                    # Joy-Con, stick control
    .venv\\Scripts\\python.exe -m teachabot.app --control tilt     # move by tilting the Joy-Cons
    .venv\\Scripts\\python.exe -m teachabot.app --control point    # move by rotating the Joy-Cons
    .venv\\Scripts\\python.exe -m teachabot.app --input keyboard
    .venv\\Scripts\\python.exe -m teachabot.app --check            # Phase 3 guided check + summary

Joy-Con: + (right) = next control mode, - (left) = re-zero tilt for a new student.
Keys: 1/2/3 = stick/tilt/point, C = run motion calibration, R = reset scene, Esc = quit.
The simulation advances in fixed 1/60 s steps, independent of the render rate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from ursina import Text, Ursina, application, color, window

from teachabot.input.base import InputProvider
from teachabot.input.motion import CAL_STEPS, MODES, MotionCalibrator, MotionControlProvider
from teachabot.simulation.environment import HANDS, HandAction, World, actions_from_input
from teachabot.ui.scene import TabletopView

ROOT = Path(__file__).resolve().parents[2]
SIM_DT = 1 / 60
KOREAN_FONT = Path("C:/Windows/Fonts/malgun.ttf")
MOTION_CAL_PATH = ROOT / "config" / "motion_calibration.json"
HAND_KO = {"left": "왼손", "right": "오른손"}
MODE_KO = {"stick": "스틱", "tilt": "기울이기", "point": "가리키기"}


def make_provider(kind: str, replay: Path | None, control: str) -> InputProvider:
    if kind == "joycon":
        from teachabot.input.joycon import JoyConInputProvider
        return MotionControlProvider(JoyConInputProvider(), MOTION_CAL_PATH, mode="stick")
    if kind == "keyboard":
        from teachabot.input.keyboard import KeyboardInputProvider
        return KeyboardInputProvider()
    from teachabot.input.replay import ReplayInputProvider
    return ReplayInputProvider(replay, realtime=True)


class Phase3Check:
    """On-screen checklist for verifying teleoperation by hand."""

    def __init__(self):
        self.done: dict[str, float] = {}
        self.both_moving_frames = 0
        self.empty_close_seen = False
        self.modes_used: dict[str, int] = {}
        self.t0 = time.perf_counter()

    STEPS = [
        ("left_grasp", "왼손(파랑 집게)을 파란 큐브 위로 옮기고 ZL로 집기"),
        ("left_place", "파란 큐브를 왼쪽 파란 상자 안에 놓기 (ZL 떼기)"),
        ("right_grasp", "오른손(주황 집게)으로 주황 큐브를 ZR로 집기"),
        ("right_place", "주황 큐브를 오른쪽 주황 상자 안에 놓기 (ZR 떼기)"),
        ("both_move", "양손을 동시에 움직여 같이 이동하기 (2초)"),
        ("empty_close", "큐브가 없는 곳에서 집게를 닫았다가 큐브 위로 가기 → 안 집혀야 정상"),
    ]

    def update(self, world: World, frame) -> None:
        now = time.perf_counter() - self.t0
        mode = frame.meta.get("control", "stick")
        self.modes_used[mode] = self.modes_used.get(mode, 0) + 1
        for e in world.events:
            if e.get("_seen"):
                continue
            e["_seen"] = True
            e["control"] = mode
            if e["event"] == "grasp":
                self.done.setdefault(f"{e['hand']}_grasp", now)
            if e["event"] == "release":
                box = world.cube_in_target(e["cube"])
                want = {"blue": "left_box", "orange": "right_box"}[e["cube"]]
                if box == want:
                    self.done.setdefault(f"{e['hand']}_place", now)
        if all(abs(frame.hand(h).stick[0]) + abs(frame.hand(h).stick[1]) > 0.3 for h in HANDS):
            self.both_moving_frames += 1
            if self.both_moving_frames >= 120:
                self.done.setdefault("both_move", now)
        for h in HANDS:
            g = world.grippers[h]
            if g.closed and g.holding is None and world.nearest_free_cube(g.x, g.z) is None:
                self.empty_close_seen = True
            if self.empty_close_seen and g.closed and g.holding is None:
                for cube in world.cubes.values():
                    if cube.held_by is None and abs(cube.x - g.x) < 0.02 and abs(cube.z - g.z) < 0.02:
                        self.done.setdefault("empty_close", now)

    def text(self) -> str:
        cur = next((k for k, _ in self.STEPS if k not in self.done), None)
        lines = ["[ Phase 3 확인 ]  (아무 조작 모드로 해도 됩니다)"]
        for i, (k, label) in enumerate(self.STEPS):
            mark = "[O]" if k in self.done else ("[>]" if k == cur else "[ ]")
            lines.append(f"{mark} {i + 1}. {label}")
        if cur is None:
            lines.append("모두 완료! 3초 뒤 결과를 저장하고 닫습니다.")
        return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["joycon", "keyboard", "replay"], default="joycon")
    ap.add_argument("--control", choices=MODES, default="stick", help="Joy-Con control mode at start")
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--check", action="store_true", help="show the Phase 3 checklist and write a summary on exit")
    ap.add_argument("--screenshot", type=Path, help="(dev) save a screenshot after 2 s and quit")
    args = ap.parse_args(argv)
    if args.input == "replay" and not args.replay:
        ap.error("--replay PATH is required with --input replay")

    app = Ursina(title="Teach-a-Bot Lab", size=(1600, 900), borderless=False, vsync=True,
                 development_mode=False)
    application.asset_folder = ROOT / "assets"
    window.color = color.rgb(0.93, 0.94, 0.96)
    if KOREAN_FONT.exists():
        # Ursina finds fonts by name in application.fonts_folder (no absolute paths).
        # Use the system font in place; it must not be copied into the repo (licence).
        application.fonts_folder = KOREAN_FONT.parent
        Text.default_font = KOREAN_FONT.name
    window.fps_counter.enabled = True
    if args.screenshot or args.check:
        window.always_on_top = True   # otherwise it can open behind other windows

    world = World.default_scene()
    view = TabletopView(world)
    provider = make_provider(args.input, args.replay, args.control)
    provider.connect()
    motion = provider if isinstance(provider, MotionControlProvider) else None
    check = Phase3Check() if args.check else None

    hud = Text(text="", position=window.top_left + (0.02, -0.02), scale=0.9, color=color.black)
    check_text = Text(text="", position=window.bottom_left + (0.02, 0.24), scale=0.85,
                      color=color.black) if check else None
    cal_text = Text(text="", origin=(0, 0), position=(0, 0.25), scale=1.4, color=color.rgb(0.1, 0.1, 0.35),
                    background=False)
    state = {"acc": 0.0, "last": None, "calibrator": None, "pending_mode": None, "prev_buttons": {}}
    frame_dt: list[float] = []

    def set_mode(mode: str):
        if not motion:
            return
        if mode != "stick" and not motion.ready(mode):
            start_calibration(then=mode)
            return
        motion.set_mode(mode)

    def start_calibration(then: str | None = None):
        if not motion:
            return
        steps = [s for s, _ in CAL_STEPS]
        state["calibrator"] = MotionCalibrator(motion.cal, steps=steps)
        state["pending_mode"] = then or (motion.mode if motion.mode != "stick" else "tilt")
        motion.set_mode("stick")

    def pressed(raw, hand, button) -> bool:
        now = raw.hand(hand).buttons.get(button, False)
        key = (hand, button)
        before = state["prev_buttons"].get(key, False)
        state["prev_buttons"][key] = now
        return now and not before

    def update():
        now = time.perf_counter()
        if state["last"] is not None:
            frame_dt.append(now - state["last"])
            state["acc"] += now - state["last"]
        state["last"] = now
        frame = provider.poll()
        raw = motion.last_raw if motion else frame

        cal = state["calibrator"]
        if cal:
            cal.update(raw)
            if cal.done:
                motion.save_calibration()
                state["calibrator"] = None
                motion.set_mode(state["pending_mode"])
                cal_text.text = ""
            else:
                label = dict(CAL_STEPS)[cal.step]
                prog = cal.progress()
                cal_text.text = (f"[ 조이콘 움직임 보정 {cal.i + 1}/{len(cal.steps)} ]\n{label}\n"
                                 f"왼손: {prog['left']}    오른손: {prog['right']}")
            actions = {h: HandAction() for h in HANDS}          # world frozen while calibrating
        else:
            if motion:
                if pressed(raw, "right", "plus"):
                    set_mode(MODES[(MODES.index(motion.mode) + 1) % len(MODES)])
                if pressed(raw, "left", "minus") and motion.mode == "tilt":
                    for h in HANDS:
                        if raw.hand(h).accel is not None:
                            motion.mapper.recenter(h, raw.hand(h).accel)
            actions = actions_from_input(frame)

        steps = 0
        while state["acc"] >= SIM_DT and steps < 6:      # fixed-step simulation
            world.step(actions, SIM_DT)
            state["acc"] -= SIM_DT
            steps += 1
        if steps == 6:
            state["acc"] = 0.0
        view.sync(world)
        if check and not cal:
            check.update(world, frame)
            check_text.text = check.text()
            if all(k in check.done for k, _ in check.STEPS):
                state.setdefault("done_at", now)
                if now - state["done_at"] > 3.0:
                    finish()
        hud.text = hud_text(world, frame, args.input, motion)
        if args.screenshot and world.time > 2.0 and not state.get("shot"):
            # Grab what is actually on screen (Panda's back-buffer capture came back black here).
            from PIL import ImageGrab
            props = application.base.win.getProperties()
            x, y = props.getOrigin()
            w, h = props.getXSize(), props.getYSize()
            ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(args.screenshot)
            state["shot"] = True
            print(f"screenshot saved={state['shot']} -> {args.screenshot}")
            finish()

    def input(key):
        if key == "r":
            world.__init__(**World.default_scene().__dict__)
        if key in ("1", "2", "3"):
            set_mode(MODES[int(key) - 1])
        if key == "c":
            start_calibration()
        if key == "escape":
            finish()

    def finish():
        if state.get("finished"):
            return
        state["finished"] = True
        provider.disconnect()
        if check:
            write_summary(check, world, frame_dt, args.input)
        application.quit()

    if motion and args.control != "stick":
        set_mode(args.control)

    # Ursina picks up update/input from the main module; register explicitly instead
    import __main__
    __main__.update, __main__.input = update, input
    app.run()
    return 0


def hud_text(world: World, frame, source: str, motion: MotionControlProvider | None) -> str:
    mode = MODE_KO[motion.mode] if motion else "-"
    lines = [f"입력: {source}   조작 모드: {mode}"]
    for h in HANDS:
        hand, g = frame.hand(h), world.grippers[h]
        conn = "연결" if hand.connected else "끊김 (조이콘 버튼을 누르세요)"
        hold = f"  잡은 것: {g.holding}" if g.holding else ""
        lines.append(f"{HAND_KO[h]}: {conn}  집게 {'닫힘' if g.closed else '열림'}{hold}")
    if motion:
        lines.append("조이콘 + = 조작 모드 바꾸기,  - = 기울이기 영점 다시 맞추기,  ZL/ZR = 집기")
    lines.append("키보드: 1 스틱 / 2 기울이기 / 3 가리키기,  C = 움직임 보정,  R = 처음부터,  Esc = 종료")
    return "\n".join(lines)


def write_summary(check: Phase3Check, world: World, frame_dt: list[float], source: str) -> None:
    ts = sorted(frame_dt[60:]) or [0.0]
    q = lambda p: ts[min(len(ts) - 1, int(p * (len(ts) - 1)))]
    summary = {
        "source": source,
        "seconds": round(time.perf_counter() - check.t0, 1),
        "steps_done_at_s": {k: round(v, 1) for k, v in check.done.items()},
        "all_steps_done": all(k in check.done for k, _ in check.STEPS),
        "control_mode_frames": check.modes_used,
        "fps_mean": round(len(ts) / sum(ts), 1) if sum(ts) else None,
        "frame_ms_median": round(1000 * statistics.median(ts), 2),
        "frame_ms_p99": round(1000 * q(0.99), 2),
        "frames_over_33ms": sum(1 for d in ts if d > 0.033),
        "world_events": [{k: v for k, v in e.items() if k != "_seen"} for e in world.events],
    }
    out = ROOT / "data" / "logs" / f"phase3_check_{source}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
