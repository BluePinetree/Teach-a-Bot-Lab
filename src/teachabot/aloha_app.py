"""Teach-a-Bot Lab on ALOHA 2 (MuJoCo) — S1 "정리하기" teleoperation.

    .venv\\Scripts\\python.exe -m teachabot.aloha_app [--input keyboard] [--check] [--screenshot PATH]
    .venv\\Scripts\\python.exe -m teachabot.aloha_app --record s01                  # record demonstrations
    .venv\\Scripts\\python.exe -m teachabot.aloha_app --policy P.npz [--record t01]  # AI drives, people take over

Window: MuJoCo renders offscreen, the Korean HUD is drawn with PIL (system Malgun Gothic,
used in place) and shown with OpenCV. Keys: R = new episode, Esc = quit, T = retrain (AI mode).
Joy-Con: capture (L) / home (R) = new episode, - / + = re-zero that hand's wrist, X (R) = retrain.

Takeover (AI mode): moving a hand's stick or pressing its grip/shoulder hands that arm to the
person for the rest of the episode; only the person's part is saved for learning
(DAgger-inspired correction data, not a full DAgger implementation).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from teachabot.aloha.env import CONTROL_DT, AlohaSortEnv, ArmCommand
from teachabot.aloha.gym_env import action_to_commands, commands_to_action
from teachabot.aloha.scene import CUBE_HALF, HANDS
from teachabot.aloha.teleop import MOVE_SOURCES, TeleopMapper, TeleopSettings
from teachabot.input.base import InputFrame, InputProvider
from teachabot.input.motion import MotionCalibration
from teachabot.input.replay import InputRecorder
from teachabot.learning.bc import NumpyPolicy
from teachabot.learning.demos import DemoWriter
from teachabot.learning.live import LiveTrainer

ROOT = Path(__file__).resolve().parents[2]
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/malgunbd.ttf")
MOTION_CAL_PATH = ROOT / "config" / "motion_calibration.json"
LOG_DIR = ROOT / "data" / "logs"
W, H = 1280, 720
HAND_KO = {"left": "왼손", "right": "오른손"}
CUBE_KO = {"left": "파란", "right": "주황"}
HAND_RGB = {"left": (70, 130, 245), "right": (255, 150, 40)}
WINDOW = "Teach-a-Bot Lab - ALOHA"
PRACTICE_SEED_BASE = 3_000_000   # AI-mode layouts: unseen in training, separate from the test layouts
TAKEOVER_STICK = 0.35            # stick deflection that hands an arm to the person
VIDEO_FPS = 30


def make_provider(kind: str, replay: Path | None = None) -> InputProvider:
    if kind == "replay":                       # recorded input (tests)
        from teachabot.input.replay import ReplayInputProvider
        return ReplayInputProvider(replay, realtime=True)
    if kind == "joycon":
        from teachabot.input.joycon import JoyConInputProvider
        return JoyConInputProvider()
    from teachabot.input.keyboard import KeyboardInputProvider
    return KeyboardInputProvider()


@dataclass
class S1Check:
    """Noon checkpoint (2026-10-20): can a person pick and place with the Joy-Cons?"""
    STEPS = [
        ("left_lift", "왼손으로 파란 큐브 들어 올리기"),
        ("left_place", "파란 큐브를 파란 상자에 넣기"),
        ("right_lift", "오른손으로 주황 큐브 들어 올리기"),
        ("right_place", "주황 큐브를 주황 상자에 넣기"),
        ("wrist_tilt", "조이콘 앞으로 기울이기 → 집게 기울기"),
        ("wrist_twist", "조이콘 옆으로 기울이기 → 집게 회전"),
        ("episode", "한 판 완료: 두 큐브 모두 상자에"),
    ]
    t0: float = field(default_factory=time.perf_counter)
    done: dict = field(default_factory=dict)

    def mark(self, key: str) -> None:
        if key not in self.done:
            self.done[key] = round(time.perf_counter() - self.t0, 1)

    def update(self, env: AlohaSortEnv, info) -> None:
        for h in HANDS:
            held = env.targets[h].grip and env.cube_pos(h)[2] > CUBE_HALF + 0.04
            if held:
                self.mark(f"{h}_lift")
            if info.in_tray[h] and f"{h}_lift" in self.done:
                self.mark(f"{h}_place")
            if abs(env.targets[h].pitch) > np.deg2rad(20):
                self.mark("wrist_tilt")
            if abs(env.targets[h].twist) > np.deg2rad(30):
                self.mark("wrist_twist")
        if info.success:
            self.mark("episode")

    @property
    def complete(self) -> bool:
        return all(k in self.done for k, _ in self.STEPS)


class Hud:
    def __init__(self):
        self.f_title = ImageFont.truetype(str(FONT_BOLD_PATH if FONT_BOLD_PATH.exists() else FONT_PATH), 30)
        self.f_big = ImageFont.truetype(str(FONT_BOLD_PATH if FONT_BOLD_PATH.exists() else FONT_PATH), 64)
        self.f = ImageFont.truetype(str(FONT_PATH), 22)
        self.f_small = ImageFont.truetype(str(FONT_PATH), 18)
        self.f_tiny = ImageFont.truetype(str(FONT_PATH), 15)

    @staticmethod
    def _box(d: ImageDraw.ImageDraw, xy, alpha=150):
        d.rounded_rectangle(xy, radius=10, fill=(15, 20, 30, alpha))

    def draw(self, rgb: np.ndarray, env: AlohaSortEnv, info, frame: InputFrame | None, mapper: TeleopMapper,
             source: str, episode: int, result: str | None, check: S1Check | None, fps: float,
             panel: tuple[str, list[tuple[str, tuple]]] | None = None, mode_line: str | None = None,
             controllers: dict | None = None, history: list[dict] | None = None,
             guide_text: str | None = None) -> np.ndarray:
        img = Image.fromarray(rgb).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        self._box(d, (16, 14, 640, 98))
        d.text((32, 20), "S1 정리하기", font=self.f_title, fill=(255, 255, 255))
        d.text((32, 62), "파란 큐브 → 파란 상자 (왼손),  주황 큐브 → 주황 상자 (오른손)", font=self.f_small,
               fill=(220, 225, 235))
        if mode_line:
            mw = d.textlength(mode_line, font=self.f) + 28
            self._box(d, (W - 16 - mw, 106, W - 16, 142))
            d.text((W - 2 - mw, 110), mode_line, font=self.f, fill=(255, 225, 120))
        self._box(d, (W - 330, 14, W - 16, 98))
        d.text((W - 314, 20), f"{info.time:5.1f} 초", font=self.f_title, fill=(255, 255, 255))
        d.text((W - 314, 62), f"{episode}번째 판 · 입력 {source} · {fps:4.0f}fps", font=self.f_small,
               fill=(200, 205, 215))

        y0 = H - 168
        for i, h in enumerate(HANDS):
            x0 = 16 if h == "left" else W - 336
            self._box(d, (x0, y0, x0 + 320, y0 + 92))
            hand = frame.hand(h) if frame else None
            conn = hand is not None and hand.connected
            tg = env.targets[h]
            d.text((x0 + 14, y0 + 6), f"{HAND_KO[h]} ({CUBE_KO[h]})", font=self.f, fill=HAND_RGB[h])
            if controllers:
                who = controllers[h]
                tag, col = ("AI", (120, 200, 255)) if who == "ai" else ("사람이 넘겨받음", (255, 190, 90))
                tw = d.textlength(tag, font=self.f_small) + 16
                d.rounded_rectangle((x0 + 304 - tw, y0 - 30, x0 + 320, y0 - 4), radius=8, fill=(15, 20, 30, 200))
                d.text((x0 + 312 - tw, y0 - 28), tag, font=self.f_small, fill=col)
            ai_hand = bool(controllers) and controllers[h] == "ai"
            ok = conn or ai_hand                   # an AI-driven arm does not need its Joy-Con
            status = "연결 안 됨" if not ok else ("집는 중" if tg.grip else "열림")
            if ok and info.in_tray[h]:
                status += " · 상자 안!"
            d.text((x0 + 150, y0 + 8), status, font=self.f_small, fill=(255, 255, 255) if ok else (255, 120, 120))
            if ai_hand or mapper.wrist_ready(h):
                wrist = f"손목  기울기 {np.rad2deg(tg.pitch):+4.0f}°   회전 {np.rad2deg(tg.twist):+4.0f}°"
            else:
                wrist = "손목  고정 (기울기 보정 없음)"
            d.text((x0 + 14, y0 + 38), wrist, font=self.f_small, fill=(220, 225, 235))
            d.text((x0 + 14, y0 + 62), f"높이 {tg.pos[2] * 100:4.1f} cm", font=self.f_small, fill=(220, 225, 235))

        self._box(d, (16, H - 62, W - 16, H - 16))
        guide = ("스틱: 이동  ·  L/R 누르고 있기: 내리기  ·  ZL/ZR 누르고 있기: 집기  ·  조이콘 기울이기: 손목"
                 "  ·  -/+: 손목 영점  ·  캡처/홈: 새 판")
        if source == "keyboard":
            guide = "WASD/방향키: 이동 · E/오른Shift: 내리기 · Q/오른Ctrl: 집기 · Space: 시작 · R: 새 판 · Esc: 종료"
        if guide_text:
            guide = guide_text
        elif controllers:
            guide = ("AI가 멈추면 그 손의 스틱이나 버튼을 움직여 넘겨받기  ·  A: 시작  ·  캡처/홈: 새 판"
                     "  ·  X(오른쪽) 또는 T: 모은 데이터로 다시 학습")
        d.text((32, H - 52), guide, font=self.f_small, fill=(235, 238, 245))

        if check is not None:
            x0, yy = 16, 112           # left edge: over the frame post, clear of the trays
            self._box(d, (x0, yy, x0 + 345, yy + 36 + 24 * len(check.STEPS)))
            d.text((x0 + 12, yy + 6), "실기 점검 (자동 체크)", font=self.f_small, fill=(255, 255, 255))
            for j, (k, label) in enumerate(check.STEPS):
                ok = k in check.done
                d.text((x0 + 12, yy + 34 + 24 * j), ("■ " if ok else "□ ") + label, font=self.f_tiny,
                       fill=(120, 230, 140) if ok else (230, 232, 238))

        if result:
            color = (120, 230, 140) if result.startswith("성공") else (255, 130, 110)
            tw = d.textlength(result, font=self.f_big)
            self._box(d, ((W - tw) / 2 - 30, 250, (W + tw) / 2 + 30, 350), alpha=190)
            d.text(((W - tw) / 2, 262), result, font=self.f_big, fill=color)
            sub = "캡처/홈 버튼(또는 R)으로 새 판"
            sw = d.textlength(sub, font=self.f)
            d.text(((W - sw) / 2, 362), sub, font=self.f, fill=(255, 255, 255))
        if history:
            self._draw_history(d, history)
        if panel:
            title, lines = panel
            pw = max([d.textlength(title, font=self.f_title)] + [d.textlength(t, font=self.f) for t, _ in lines]) + 60
            ph = 70 + 36 * len(lines)
            x0, y0 = (W - pw) / 2, (H - ph) / 2 - 30
            self._box(d, (x0, y0, x0 + pw, y0 + ph), alpha=205)
            d.text((x0 + 30, y0 + 16), title, font=self.f_title, fill=(255, 255, 255))
            for j, (t, col) in enumerate(lines):
                d.text((x0 + 30, y0 + 64 + 36 * j), t, font=self.f, fill=col)
        return np.asarray(Image.alpha_composite(img, over).convert("RGB"))


    def _draw_history(self, d: ImageDraw.ImageDraw, history: list[dict]) -> None:
        """Success rate per training round (measured on held-out layouts)."""
        rows = history[-5:]
        x0, y0, bw = 16, 112, 330
        self._box(d, (x0, y0, x0 + bw, y0 + 40 + 34 * len(rows)))
        d.text((x0 + 12, y0 + 6), "AI 실력 (처음 보는 배치에서 시험)", font=self.f_small, fill=(255, 255, 255))
        for j, r in enumerate(rows):
            yy = y0 + 38 + 34 * j
            rate = float(r["success_rate"])
            d.text((x0 + 12, yy), r["label"], font=self.f_tiny, fill=(225, 228, 236))
            bx = x0 + 128
            d.rectangle((bx, yy + 2, bx + 120, yy + 20), fill=(60, 66, 80, 220))
            d.rectangle((bx, yy + 2, bx + int(120 * rate), yy + 20), fill=(120, 230, 140, 255))
            d.text((bx + 128, yy), f"{rate * 100:3.0f}%  큐브 {r['cubes_mean']:.2f}", font=self.f_tiny,
                   fill=(225, 228, 236))


def start_panel(state: str, connected: dict, source: str, ai: bool = False):
    if state not in ("waiting", "paused"):
        return None
    if ai:
        return "AI 차례", [("A 버튼(또는 스페이스바)을 누르면 AI가 혼자 해 봅니다", (240, 242, 248)),
                           ("AI가 멈추면 그 손의 스틱이나 버튼을 움직여 넘겨받아 마무리해 주세요", (255, 210, 120)),
                           ("넘겨받은 부분만 AI의 새 공부 자료로 저장됩니다", (200, 205, 215))]
    ok, bad, white = (120, 230, 140), (255, 130, 110), (240, 242, 248)
    lines = []
    if source != "keyboard":
        for h in HANDS:
            lines.append((f"● {HAND_KO[h]} 조이콘: 연결됨", ok) if connected[h] else
                         (f"○ {HAND_KO[h]} 조이콘: 연결 안 됨 (아무 버튼이나 눌러 깨워 주세요)", bad))
    go = "스페이스바" if source == "keyboard" else "오른쪽 조이콘 A 버튼"
    if all(connected.values()):
        if state == "waiting":
            lines += [(f"두 조이콘을 편하게 쥐고 {go}을 누르면 시작", white),
                      ("(누를 때 쥐고 있던 자세가 손목의 기준 자세가 됩니다)", (200, 205, 215))]
        else:
            lines.append((f"다시 연결됐어요. {go}을 누르면 계속", white))
    else:
        lines.append(("두 조이콘이 모두 연결되면 시작할 수 있어요", white))
    return ("시작 준비" if state == "waiting" else "일시정지: 조이콘 연결 끊김"), lines


def train_panel(trainer: LiveTrainer):
    white, dim = (240, 242, 248), (200, 205, 215)
    el = time.perf_counter() - trainer.t_started
    if trainer.state == "training":
        return "AI가 다시 공부하는 중…", [(trainer.message, white), (f"{el:4.0f}초", dim)]
    if trainer.state == "evaluating":
        return "AI 시험 보는 중…", [(trainer.message, white), (f"{el:4.0f}초", dim)]
    if trainer.state == "error":
        return "다시 학습 실패", [(trainer.message[:60], (255, 130, 110)), ("이전 AI로 계속합니다", dim)]
    r = trainer.result or {}
    prev = trainer.history[-2] if len(trainer.history) >= 2 else None
    lines = [(f"새 AI ({r.get('label', '')}): 성공 {r.get('success_rate', 0) * 100:.0f}% · "
              f"판당 큐브 {r.get('cubes_mean', 0):.2f}개", (120, 230, 140))]
    if prev:
        lines.append((f"이전 AI ({prev['label']}): 성공 {prev['success_rate'] * 100:.0f}% · "
                      f"판당 큐브 {prev['cubes_mean']:.2f}개", dim))
    lines.append(("A 버튼을 누르면 새 AI가 해 봅니다", white))
    return f"다시 배운 AI의 시험 결과 ({r.get('episodes', '?')}판, {el:.0f}초 걸림)", lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=("joycon", "keyboard", "replay"), default="joycon")
    ap.add_argument("--replay", type=Path, help="input recording for --input replay")
    ap.add_argument("--time-limit", type=float, default=40.0)
    ap.add_argument("--move", choices=MOVE_SOURCES, default="stick", help="movement source (D9)")
    ap.add_argument("--wrist", choices=("imu", "off"), default="imu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="noon checkpoint checklist")
    ap.add_argument("--screenshot", type=Path, help="render a few seconds with no input, save, exit")
    ap.add_argument("--max-seconds", type=float, help="quit after this long (smoke tests)")
    ap.add_argument("--record", metavar="SESSION", help="save each episode as a demonstration (anonymous code, e.g. s01)")
    ap.add_argument("--policy", type=Path, help="let a trained policy (.npz) drive instead of the Joy-Cons")
    ap.add_argument("--autostart", action="store_true", help="start the first episode without a button (smoke tests)")
    ap.add_argument("--video", type=Path, help="also write what the window shows to this .mp4 (30 fps)")
    ap.add_argument("--eval-episodes", type=int, default=40, help="layouts used to test a retrained AI")
    args = ap.parse_args()

    env = AlohaSortEnv(time_limit=args.time_limit)
    mapper = TeleopMapper(TeleopSettings(move_source=args.move, wrist_source=args.wrist),
                          MotionCalibration.load(MOTION_CAL_PATH))
    renderer = mujoco.Renderer(env.model, H, W)
    hud = Hud()
    check = S1Check() if args.check else None
    seed, episode = args.seed, 1
    if args.policy and args.seed == 0:
        seed = PRACTICE_SEED_BASE
    env.reset(seed)
    info = env.info()

    if args.screenshot:
        for _ in range(int(1.0 / CONTROL_DT)):
            _, info = env.step({h: ArmCommand() for h in HANDS})
        renderer.update_scene(env.data, camera="demo_cam")
        out = hud.draw(renderer.render(), env, info, None, mapper, "joycon", episode, None, check, 0,
                       panel=start_panel("waiting", {"left": True, "right": False}, "joycon"))
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(args.screenshot)
        print("saved", args.screenshot)
        return

    policy = NumpyPolicy.load(args.policy) if args.policy else None
    provider = make_provider(args.input, args.replay)
    provider.connect()
    # AI mode always saves takeover episodes (session "live" unless --record names one)
    writer = DemoWriter(args.record or "live", provider.source) if (args.record or policy) else None
    controllers = {h: "ai" if policy else "human" for h in HANDS}
    takeover_t: dict = {}
    trainer = LiveTrainer(episodes=args.eval_episodes) if policy else None
    if trainer:
        trainer.add_baseline(args.policy)
    policy_path = args.policy
    retrains: list[dict] = []
    video = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (W, H))
    next_video_t = 0.0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    recorder = InputRecorder(LOG_DIR / f"aloha_s1_input_{stamp}.jsonl", provider.source,
                             meta={"app": "aloha_app", "move": args.move, "wrist": args.wrist})
    episodes: list[dict] = []
    frame_ms: list[float] = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)
    prev_buttons: dict = {}
    result: str | None = None
    state = "waiting"          # waiting -> running <-> paused -> finished
    last = time.perf_counter()
    fps = 0.0

    def new_episode():
        nonlocal seed, episode, result, info, acc, state
        if writer:
            writer.start()          # an unfinished episode is discarded
        for h in HANDS:
            controllers[h] = "ai" if policy else "human"
        takeover_t.clear()
        acc = 0.0
        state = "waiting"
        seed += 1
        episode += 1
        result = None
        env.reset(seed)
        info = env.info()

    def mode_line() -> str | None:
        if policy is not None:
            n, t = policy.meta.get("n_demos"), policy.meta.get("n_takeovers", 0)
            learned = f"시범 {n}개" + (f" + 개입 {t}번" if t else "")
            line = f"AI가 조종 중 ({learned}으로 배움)" if n is not None else "AI가 조종 중"
            if writer and writer.saved:
                line += f" · 새 개입 {writer.saved}번 저장"
            return line
        if writer:
            return f"시범 기록 중 · 저장 {writer.saved}개 (성공 {writer.saved_success}개)"
        return None

    key_start = bool(args.autostart)
    key_train = False
    t_run = time.perf_counter()
    t_prev, acc = t_run, 0.0
    try:
        while True:
            t_start = time.perf_counter()
            if args.max_seconds and t_start - t_run > args.max_seconds:
                break
            frame = provider.poll()
            recorder.record(frame)
            pressed = {}
            for h in HANDS:
                hb = frame.hand(h).buttons
                pressed[h] = {b: v and not prev_buttons.get((h, b), False) for b, v in hb.items()}
                for b, v in hb.items():
                    prev_buttons[(h, b)] = v
            if pressed["left"].get("minus"):
                mapper.recenter("left", frame.left.accel)
            if pressed["right"].get("plus"):
                mapper.recenter("right", frame.right.accel)
            if pressed["left"].get("capture") or pressed["right"].get("home"):
                new_episode()
                continue
            if trainer and (pressed["right"].get("x") or key_train) and state in ("waiting", "finished"):
                trainer.start()
            key_train = False
            if trainer and trainer.state == "done" and trainer.result and \
                    trainer.result["policy"] != str(policy_path):
                policy_path = Path(trainer.result["policy"])
                policy = NumpyPolicy.load(policy_path)          # the next episode uses the new AI
                retrains.append(trainer.result)
                print("retrained", trainer.result, flush=True)

            # Start gate: nothing moves until both hands are connected and A (Space) is pressed.
            connected = {h: frame.hand(h).connected for h in HANDS}
            start = pressed["right"].get("a", False) or key_start
            key_start = False
            ready = all(connected.values()) or policy is not None
            if trainer and trainer.busy():
                start = False                      # no new episode while the AI is studying
            if state == "waiting" and ready and start:
                if trainer and trainer.state in ("done", "error"):
                    trainer.state = "idle"             # the result panel has been seen
                for h in HANDS:                    # the grip held at start becomes the wrist neutral
                    mapper.recenter(h, frame.hand(h).accel)
                if writer:
                    writer.start()
                state, acc = "running", 0.0
            elif state == "running" and not ready:
                state = "paused"
            elif state == "paused" and ready and start:
                state, acc = "running", 0.0

            # Advance physics by real elapsed time (rendering runs ~35-45 fps; control is 50 Hz).
            acc = min(acc + (t_start - t_prev), 3 * CONTROL_DT) if state == "running" else 0.0
            t_prev = t_start
            n_steps = 0
            while result is None and acc >= CONTROL_DT:
                acc -= CONTROL_DT
                n_steps += 1
                obs = env.observation()
                human_cmds = mapper.commands(frame, CONTROL_DT)
                if policy is not None:
                    for h in HANDS:                # takeover: the person grabs this arm for the rest of the episode
                        hand = frame.hand(h)
                        if controllers[h] == "ai" and hand.connected and (
                                float(np.hypot(*hand.stick)) > TAKEOVER_STICK
                                or hand.buttons["grip"] or hand.buttons["shoulder"]):
                            controllers[h] = "human"
                            takeover_t[h] = round(env.t, 2)
                            mapper.recenter(h, hand.accel)
                            print(f"takeover {h} at {env.t:.2f}s", flush=True)
                    ai_cmds = action_to_commands(policy(obs))
                    cmds = {h: human_cmds[h] if controllers[h] == "human" else ai_cmds[h] for h in HANDS}
                else:
                    cmds = human_cmds
                if writer:
                    writer.add(obs, commands_to_action(cmds), tuple(controllers[h] == "human" for h in HANDS))
                _, info = env.step(cmds)
                if check:
                    check.update(env, info)
                if info.done:
                    break
            if result is None and info.done:
                if info.success:
                    result = f"성공!  {info.time:.1f}초"
                else:
                    fell = any(env.cube_pos(h)[2] < -0.05 for h in HANDS)
                    result = "실패: 큐브가 떨어졌어요" if fell else "실패: 시간 초과"
                state = "finished"
                if writer:
                    saved = writer.finish(seed, info.success, info.time, in_tray=info.in_tray,
                                          kind="takeover" if policy is not None else "demo",
                                          extra={"takeover_t": dict(takeover_t),
                                                 "policy": str(policy_path) if policy is not None else None})
                    print("demo saved", saved.name if saved else None, flush=True)
                episodes.append({"episode": episode, "seed": seed, "success": info.success,
                                 "time": round(info.time, 2), "in_tray": {k: bool(v) for k, v in info.in_tray.items()},
                                 "takeover": dict(takeover_t) if policy is not None else None,
                                 "policy": policy_path.name if policy is not None else None})
                print("episode", episodes[-1], flush=True)

            renderer.update_scene(env.data, camera="demo_cam")
            panel = start_panel(state, connected, provider.source, ai=policy is not None)
            if trainer and trainer.state != "idle" and state in ("waiting", "finished"):
                panel = train_panel(trainer)
            out = hud.draw(renderer.render(), env, info, frame, mapper, provider.source, episode,
                           None if (panel and trainer and trainer.state != "idle") else result, check, fps,
                           panel=panel, mode_line=mode_line(),
                           controllers=dict(controllers) if policy is not None else None,
                           history=trainer.history if trainer else None)
            if video is not None:
                now_v = time.perf_counter() - t_run
                while next_video_t <= now_v:
                    video.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                    next_video_t += 1.0 / VIDEO_FPS
            cv2.imshow(WINDOW, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("r"), ord("R")):
                new_episode()
            if key == 32:
                key_start = True
            if key in (ord("t"), ord("T")):
                key_train = True
            spent = time.perf_counter() - t_start
            if spent < CONTROL_DT:
                time.sleep(CONTROL_DT - spent)
            now = time.perf_counter()
            frame_ms.append((now - t_start) * 1000)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
    finally:
        recorder.close()
        provider.disconnect()
        if video is not None:
            video.release()
        cv2.destroyAllWindows()
        summary = {"created": stamp, "input": provider.source, "move": args.move, "wrist": args.wrist,
                   "episodes": episodes, "frames": len(frame_ms),
                   "wall_s": round(time.perf_counter() - t_run, 2), "sim_s_current_episode": round(env.t, 2),
                   "frame_ms_p50": float(np.percentile(frame_ms, 50)) if frame_ms else None,
                   "frame_ms_p99": float(np.percentile(frame_ms, 99)) if frame_ms else None,
                   "check": {k: check.done.get(k) for k, _ in S1Check.STEPS} if check else None,
                   "check_complete": check.complete if check else None,
                   "input_log": recorder.path.name, "record_session": args.record,
                   "demos_saved": writer.saved if writer else None,
                   "policy": str(args.policy) if args.policy else None,
                   "retrains": retrains, "video": str(args.video) if args.video else None}
        path = LOG_DIR / f"aloha_s1_{'check' if check else 'run'}_{stamp}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("summary", path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
