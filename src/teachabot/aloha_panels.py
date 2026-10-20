"""Panel layout for the ALOHA app (docs/UI_REVIEW.md, chosen 2026-10-20).

1920x1080 canvas: a top bar with the steps, the 3D view (1280x720) in the middle with nothing
drawn over it during play, one Joy-Con panel per hand (drawing with callouts; the pressed
button and its callout light up), one message area under the view, a status panel for the
presenter and a record panel. The learning sheet (network, success graph, progress, a live AI
trial) opens over the 3D view.

Speed: everything that does not change is drawn once into a cached base image keyed by the
layout (input source, height mode, ...); per frame only the changing parts are drawn
(measured on the review: redrawing all 1080p overlays 33 ms vs cached 1.6 ms).

The drawings are simple shapes of our own, not Nintendo artwork. Every number on the learning
sheet comes from the running trainer, the policy file or config/visit_reference.json.
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from teachabot.aloha.env import CONTROL_DT, Z_CARRY, Z_MIN
from teachabot.aloha.scene import HANDS

FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/malgunbd.ttf")
CW, CH = 1920, 1080
SIM_X, SIM_Y, SIM_W, SIM_H = 320, 72, 1280, 720
BG, PANEL, LINE = (16, 20, 28), (28, 34, 46), (60, 70, 88)
TXT, DIM, YEL = (235, 238, 245), (150, 160, 178), (255, 214, 102)
GOOD, BAD = (130, 220, 150), (255, 130, 110)
HAND_RGB = {"left": (70, 130, 245), "right": (255, 150, 40)}
HAND_KO = {"left": "왼손", "right": "오른손"}
TRIAL_SEED_BASE = 1_000_500        # test-layout family (never trained on), shown on the learning sheet
IN_GROUPS = [(3, "집게 목표 위치"), (3, "집게 실제 위치"), (2, "손가락 · 집기"),
             (3, "큐브 위치"), (2, "큐브 방향"), (3, "큐브까지 거리"), (2, "상자까지 거리")]   # bc_hand.hand_obs
OUT_LABELS = ["좌우 이동", "앞뒤 이동", "높이", "기울기", "회전", "집기"]          # vx vy vz pitch twist grip


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
    return ImageFont.truetype(str(path), size)


def fit(d: ImageDraw.ImageDraw, text: str, font, width: float) -> str:
    """Cut with … so the text never runs past `width` pixels."""
    if d.textlength(text, font=font) <= width:
        return text
    while len(text) > 1 and d.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text + "…"


def wrap(d: ImageDraw.ImageDraw, text: str, font, width: float, max_lines: int = 3) -> list[str]:
    """Greedy word wrap (spaces; long words are cut), at most `max_lines` lines."""
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if d.textlength(trial, font=font) <= width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit(d, lines[-1] + " …", font, width)
    return [fit(d, ln, font, width) for ln in lines]


class NpDraw:
    """The few ImageDraw calls the panels use, drawn straight onto an RGB numpy canvas.

    Converting the 1920x1080 frame to PIL and back cost ~7.5 of 12.4 ms per frame (profiled);
    text masks are rendered once per (text, font, anchor) and blended, shapes use OpenCV."""

    _cache: dict = {}

    def __init__(self, canvas: np.ndarray):
        self.a = canvas

    @staticmethod
    def textlength(text: str, font) -> float:
        return font.getlength(text)

    def _mask(self, text: str, font, anchor: str):
        key = (text, id(font), anchor)
        hit = self._cache.get(key)
        if hit is None:
            l, t, r, b = font.getbbox(text, anchor=anchor)
            im = Image.new("L", (max(1, r - l), max(1, b - t)), 0)
            ImageDraw.Draw(im).text((-l, -t), text, font=font, fill=255, anchor=anchor)
            hit = (np.asarray(im, np.float32)[..., None] / 255.0, l, t)
            if len(self._cache) > 4000:                 # timers make new strings every frame
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def text(self, xy, text: str, font=None, fill=(255, 255, 255), anchor: str = "la") -> None:
        if not text:
            return
        if "\n" in text:
            for k, line in enumerate(text.split("\n")):
                self.text((xy[0], xy[1] + k * font.size * 1.3), line, font=font, fill=fill, anchor=anchor)
            return
        m, l, t = self._mask(text, font, anchor)
        x, y = int(round(xy[0] + l)), int(round(xy[1] + t))
        h, w = m.shape[:2]
        H, W = self.a.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        mm = m[y0 - y:y1 - y, x0 - x:x1 - x]
        reg = self.a[y0:y1, x0:x1]
        col = np.asarray(fill[:3], np.float32)
        reg[:] = (reg * (1.0 - mm) + col * mm).astype(np.uint8)

    @staticmethod
    def _pts(xy):
        if len(xy) == 4 and not isinstance(xy[0], (tuple, list)):
            return [(xy[0], xy[1]), (xy[2], xy[3])]
        return [tuple(p) for p in xy]

    def line(self, xy, fill=(255, 255, 255), width: int = 1) -> None:
        pts = self._pts(xy)
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            cv2.line(self.a, (int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by))), tuple(fill[:3]),
                     max(1, int(width)), cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width: int = 1) -> None:
        x0, y0, x1, y1 = (int(round(v)) for v in xy)
        if fill is not None:
            cv2.rectangle(self.a, (x0, y0), (x1, y1), tuple(fill[:3]), -1)
        if outline is not None:
            cv2.rectangle(self.a, (x0, y0), (x1, y1), tuple(outline[:3]), max(1, int(width)))

    def rounded_rectangle(self, xy, radius: int = 0, fill=None, outline=None, width: int = 1) -> None:
        x0, y0, x1, y1 = (int(round(v)) for v in xy)
        r = int(max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2)))
        if fill is not None:
            c = tuple(fill[:3])
            cv2.rectangle(self.a, (x0 + r, y0), (x1 - r, y1), c, -1)
            cv2.rectangle(self.a, (x0, y0 + r), (x1, y1 - r), c, -1)
            for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
                cv2.circle(self.a, (cx, cy), r, c, -1, cv2.LINE_AA)
        if outline is not None:
            c, w = tuple(outline[:3]), max(1, int(width))
            cv2.line(self.a, (x0 + r, y0), (x1 - r, y0), c, w, cv2.LINE_AA)
            cv2.line(self.a, (x0 + r, y1), (x1 - r, y1), c, w, cv2.LINE_AA)
            cv2.line(self.a, (x0, y0 + r), (x0, y1 - r), c, w, cv2.LINE_AA)
            cv2.line(self.a, (x1, y0 + r), (x1, y1 - r), c, w, cv2.LINE_AA)
            for (cx, cy), ang in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0),
                                  ((x0 + r, y1 - r), 90)):
                cv2.ellipse(self.a, (cx, cy), (r, r), ang, 0, 90, c, w, cv2.LINE_AA)

    def ellipse(self, xy, fill=None, outline=None, width: int = 1) -> None:
        x0, y0, x1, y1 = xy
        c = (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2)))
        ax = (max(1, int(round((x1 - x0) / 2))), max(1, int(round((y1 - y0) / 2))))
        if fill is not None:
            cv2.ellipse(self.a, c, ax, 0, 0, 360, tuple(fill[:3]), -1, cv2.LINE_AA)
        if outline is not None:
            cv2.ellipse(self.a, c, ax, 0, 0, 360, tuple(outline[:3]), max(1, int(width)), cv2.LINE_AA)

    def paste(self, rgb: np.ndarray, xy) -> None:
        x, y = int(xy[0]), int(xy[1])
        h, w = rgb.shape[:2]
        self.a[y:y + h, x:x + w] = rgb


def ref_rate(value: str) -> float | None:
    m = re.search(r"성공 (\d+(?:\.\d+)?)%", value)
    return float(m.group(1)) / 100 if m else None


class TrialView:
    """A real closed-loop run of a policy on unseen layouts, shown small on the learning sheet."""

    def __init__(self, w: int = 560, h: int = 315):
        from teachabot.aloha.env import AlohaSortEnv
        self.env = AlohaSortEnv(time_limit=40.0)
        self.renderer = mujoco.Renderer(self.env.model, h, w)
        self.path: Path | None = None
        self.policy = None
        self.k = 0
        self.t_last = None
        self.acc = 0.0
        self.done_at = None
        self.info = None
        self.results: list[bool] = []

    def show(self, path: Path | None) -> tuple[np.ndarray | None, str]:
        from teachabot.aloha.gym_env import action_to_commands
        from teachabot.learning.bc import NumpyPolicy
        if path is None:
            self.t_last = None
            return None, ""
        if path != self.path:
            self.path, self.policy, self.k, self.results = path, NumpyPolicy.load(path), 0, []
            self._reset()
        now = time.perf_counter()
        dt = 0.0 if self.t_last is None else now - self.t_last
        self.t_last = now
        if self.done_at is not None:
            if now - self.done_at > 1.5:
                self.k += 1
                self._reset()
        else:
            self.acc = min(self.acc + dt, 3 * CONTROL_DT)
            while self.acc >= CONTROL_DT:
                self.acc -= CONTROL_DT
                _, self.info = self.env.step(action_to_commands(self.policy(self.env.observation())))
                if self.info.done:
                    self.done_at = now
                    self.results.append(bool(self.info.success))
                    break
        self.renderer.update_scene(self.env.data, camera="demo_cam")
        info = self.info
        if info is not None and info.done:
            status = "성공!" if info.success else f"실패 (상자 안 {sum(info.in_tray.values())}개)"
        else:
            status = f"{self.env.t:4.1f}초"
        return self.renderer.render(), status

    def _reset(self) -> None:
        self.env.reset(TRIAL_SEED_BASE + self.k)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.info, self.done_at, self.acc = None, None, 0.0


class PanelUI:
    def __init__(self, task: str, text: dict):
        self.task, self.text = task, text
        self.f = {k: _font(*v) for k, v in {
            "title": (30, True), "h": (24, True), "chip": (21, True), "big": (34, True), "huge": (60, True),
            "msg": (25, False), "body": (20, False), "bodyb": (20, True), "small": (17, False),
            "smallb": (18, True), "tiny": (15, False)}.items()}
        self._base_key = None
        self._base: np.ndarray | None = None
        self._nn_key = None
        self._nn_img: Image.Image | None = None
        self.trial: TrialView | None = None
        self.sim_clear = True          # after draw(): nothing was drawn over the 3D view
        self._visit = False

    # ---------------------------------------------------------------- static layer
    def _labels(self, h: str, mapper, source: str, trainer_on: bool, visit: bool = False) -> list[tuple[str, str, str]]:
        """(part, key name, action) callouts for one Joy-Con, in drawing order."""
        mode = mapper.height_mode(h)
        height = {"button": "누르고 있기 = 내리기", "tilt": "누른 채 꺾기 = 높이",
                  "motion": "누른 채 위아래 = 높이"}[mode]
        if h == "left":
            return [("grip", "ZL", "누르고 있기 = 집기"), ("shoulder", "L", height),
                    ("stick", "스틱", "밀기 = 이동"), ("capture", "캡처", "새 판")]
        rows = [("grip", "ZR", "누르고 있기 = 집기"), ("shoulder", "R", height)]
        if trainer_on:
            rows.append(("x", "X", "다시 학습"))
        rows.append(("a", "A", "시작 · 확인"))
        if visit:
            rows.append(("b", "B (1초 누르기)", "다음 단계"))
        rows += [("y", "Y", "학습 화면 열기/닫기"),
                 ("stick", "스틱", "밀기 = 이동"), ("home", "홈", "새 판")]
        return rows

    @staticmethod
    def _parts(h: str, x0: int) -> dict:
        """Part geometry of our simple Joy-Con drawing (top view, far end up)."""
        cx = x0 + (80 if h == "left" else 240)
        y0 = 214
        p = {"cx": cx, "body": (cx - 48, y0, cx + 48, y0 + 330),
             "grip": (cx - 30, y0 - 42, cx + 30, y0 - 24), "shoulder": (cx - 42, y0 - 20, cx + 42, y0 + 4)}
        if h == "left":
            p["stick"] = (cx, y0 + 95)
            for name, (dx, dy) in {"up": (0, -26), "down": (0, 26), "left": (-26, 0), "right": (26, 0)}.items():
                p[name] = (cx + dx, y0 + 215 + dy)
            p["minus"] = (cx + 26, y0 + 34)
            p["capture"] = (cx + 20, y0 + 292)
        else:
            for name, (dx, dy) in {"x": (0, -26), "b": (0, 26), "y": (-26, 0), "a": (26, 0)}.items():
                p[name] = (cx + dx, y0 + 95 + dy)
            p["stick"] = (cx, y0 + 215)
            p["plus"] = (cx - 26, y0 + 34)
            p["home"] = (cx - 20, y0 + 292)
        return p

    def _anchor(self, p: dict, part: str) -> tuple[float, float]:
        v = p[part]
        if len(v) == 4:
            return (v[0] + v[2]) / 2, (v[1] + v[3]) / 2
        return v

    def _draw_joycon(self, d, h: str, x0: int, pressed: set, stick_on: bool, labels, highlight: bool) -> None:
        p = self._parts(h, x0)
        col = HAND_RGB[h]
        on = lambda part: highlight and (part in pressed or (part == "stick" and stick_on))
        if not highlight:
            d.rounded_rectangle(p["body"], radius=46, fill=tuple(int(c * 0.5) for c in col), outline=col, width=3)
        for part in ("grip", "shoulder"):
            if not highlight or on(part):
                d.rounded_rectangle(p[part], radius=8, fill=YEL if on(part) else (70, 78, 96), outline=col, width=2)
        cx, cy = p["stick"]
        if not highlight or on("stick"):
            d.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), fill=YEL if on("stick") else (40, 44, 56),
                      outline=(200, 205, 215), width=3)
            d.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=(120, 100, 40) if on("stick") else (90, 96, 110))
        for part in (("up", "down", "left", "right") if h == "left" else ("x", "b", "y", "a")):
            if not highlight or on(part):
                bx, by = p[part]
                d.ellipse((bx - 12, by - 12, bx + 12, by + 12), fill=YEL if on(part) else (55, 60, 74),
                          outline=(200, 205, 215), width=2)
                if h == "right" and not highlight:
                    d.text((bx, by), part.upper(), font=self.f["tiny"], fill=(200, 205, 215), anchor="mm")
        for part in (("minus", "capture") if h == "left" else ("plus", "home")):
            if not highlight or on(part):
                bx, by = p[part]
                d.ellipse((bx - 9, by - 9, bx + 9, by + 9), fill=YEL if on(part) else (55, 60, 74),
                          outline=(200, 205, 215), width=2)
        # callouts: key name (bold) over the action, left Joy-Con labels on the right side
        n = len(labels)
        ys = np.linspace(150, 520, n) if n > 4 else [150, 206, 300, 486][:n]
        for (part, key, act), ly in zip(labels, ys):
            lit = on(part)
            if highlight and not lit:
                continue
            ax, ay = self._anchor(p, part)
            colr = YEL if lit else (170, 178, 196)
            if h == "left":
                tx = x0 + 150
                d.line((ax + (30 if part == "stick" else 6), ay, tx - 8, ly + 12), fill=colr, width=2)
                d.text((tx, ly), key, font=self.f["smallb"], fill=YEL if lit else TXT)
                d.text((tx, ly + 24), fit(d, act, self.f["tiny"], x0 + 312 - tx), font=self.f["tiny"],
                       fill=YEL if lit else DIM)
            else:
                tx = x0 + 178
                d.line((ax - (30 if part == "stick" else 6), ay, tx + 8, ly + 12), fill=colr, width=2)
                d.text((tx, ly), key, font=self.f["smallb"], fill=YEL if lit else TXT, anchor="ra")
                d.text((tx, ly + 24), fit(d, act, self.f["tiny"], tx - x0 - 14), font=self.f["tiny"],
                       fill=YEL if lit else DIM, anchor="ra")
            d.ellipse((ax - 3, ay - 3, ax + 3, ay + 3), fill=colr)

    KEYBOARD_ROWS = {"left": [("shoulder", "E", "누르고 있기 = 내리기"), ("grip", "Q", "누르고 있기 = 집기"),
                              ("stick", "W A S D", "이동")],
                     "right": [("shoulder", "오른 Shift", "내리기"), ("grip", "오른 Ctrl", "집기"),
                               ("stick", "방향키", "이동"), ("a", "Space", "시작"), ("x", "T", "다시 학습"),
                               ("y", "L", "학습 화면"), ("home", "R", "새 판"), ("", "N", "다음 단계"),
                               ("", "F", "전체 화면"), ("", "Esc", "끝내기")]}

    def _base_layer(self, key, mapper, source: str, trainer_on: bool) -> np.ndarray:
        img = Image.new("RGB", (CW, CH), BG)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, CW, 64), fill=(22, 27, 38))
        d.text((24, 12), fit(d, self.text["title"], self.f["title"], 420), font=self.f["title"], fill=TXT)
        for h in HANDS:
            x0 = 0 if h == "left" else CW - 320
            d.rounded_rectangle((x0 + 10, 76, x0 + 310, 788), radius=16, fill=PANEL, outline=LINE, width=2)
            title = ("왼손" if h == "left" else "오른손") + (" 조이콘" if source != "keyboard" else " 키보드")
            d.text((x0 + 28, 88), title, font=self.f["h"], fill=HAND_RGB[h])
            if source == "keyboard":
                for j, (_, k, act) in enumerate(self.KEYBOARD_ROWS[h]):
                    d.text((x0 + 28, 150 + 42 * j), k, font=self.f["smallb"], fill=TXT)
                    d.text((x0 + 150, 152 + 42 * j), fit(d, act, self.f["small"], 150), font=self.f["small"], fill=DIM)
            else:
                self._draw_joycon(d, h, x0, set(), False, self._labels(h, mapper, source, trainer_on, key[-1]), False)
            d.line((x0 + 26, 568, x0 + 294, 568), fill=LINE, width=1)
            third = "꺾기" if mapper.height_mode(h) == "tilt" and not mapper.wrist_ready(h) else "손목"
            for j, lab in enumerate(("집게", "높이", third)):
                d.text((x0 + 28, 610 + 40 * j), lab, font=self.f["bodyb"], fill=DIM)
        for box in ((10, 800, 310, 1066), (330, 800, 1590, 1066), (1610, 800, 1910, 1066)):
            d.rounded_rectangle(box, radius=16, fill=PANEL, outline=LINE, width=2)
        d.text((28, 812), "진행", font=self.f["h"], fill=TXT)
        d.rectangle((SIM_X - 2, SIM_Y - 2, SIM_X + SIM_W + 1, SIM_Y + SIM_H + 1), outline=LINE, width=2)
        return np.asarray(img).copy()

    # ---------------------------------------------------------------- per frame
    def draw(self, rgb: np.ndarray, env, info, frame, mapper, source: str, episode: int, result: str | None,
             check, fps: float, panel=None, mode_line: str | None = None, controllers: dict | None = None,
             history: list[dict] | None = None, guide_text: str | None = None, notice: str | None = None,
             steps=None, alert_hand: str | None = None, *, trainer=None, policy=None, policy_path=None,
             learn_open: bool = False, reference: list[dict] | None = None, episodes: list[dict] | None = None,
             flow=None, state: str = "running", time_limit: float | None = None, result_sheet=None) -> np.ndarray:
        key = (source, tuple(mapper.height_mode(h) for h in HANDS), tuple(mapper.wrist_ready(h) for h in HANDS),
               trainer is not None, flow is not None)
        self._visit = flow is not None
        if key != self._base_key:
            self._base_key, self._base = key, self._base_layer(key, mapper, source, trainer is not None)
        canvas = self._base.copy()
        sim = np.asarray(rgb)
        sheet = learn_open or result_sheet is not None
        if sheet:
            sim = (sim.astype(np.uint16) * 90 // 255).astype(np.uint8)        # dim the 3D view under a sheet
        canvas[SIM_Y:SIM_Y + SIM_H, SIM_X:SIM_X + SIM_W] = sim
        d = img = NpDraw(canvas)                 # per-frame drawing straight on the canvas

        self._top(d, info, steps, mode_line, controllers, time_limit, flow)
        for h in HANDS:
            self._hand(d, h, env, info, frame, mapper, source, controllers, alert_hand, trainer is not None)
        self._status(d, mode_line, source, fps, episode, notice, frame)
        self._message(d, env, info, result, panel, state, flow, controllers, guide_text, trainer)
        self._record(d, history, check, flow, episodes)
        self.sim_clear = not sheet and not result
        if result and not sheet:
            self._result_banner(d, result)
        if result_sheet is not None:
            self._result_sheet(img, d, *result_sheet)
        elif learn_open:
            self._learning(img, d, trainer, policy, policy_path, history, reference, state)
        return canvas

    def _top(self, d, info, steps, mode_line, controllers, time_limit, flow) -> None:
        if steps:
            chips = steps[0]
        else:
            name = "AI 조종" if controllers else ("시범 기록" if mode_line and "시범 기록" in mode_line else "조종")
            chips = [(name, "current")]
        widths = [d.textlength(t, font=self.f["chip"]) + 36 for t, _ in chips]
        x = max(470, (CW - sum(widths) - 26 * (len(chips) - 1)) / 2)
        for (t, st), w in zip(chips, widths):
            cur = st == "current"
            d.rounded_rectangle((x, 14, x + w, 50), radius=18, fill=YEL if cur else (40, 47, 62),
                                outline=None if cur else LINE)
            d.text((x + 18, 18), t, font=self.f["chip"], fill=(20, 20, 20) if cur else (GOOD if st == "done" else DIM))
            x += w
            if (t, st) != chips[-1]:
                d.text((x + 6, 18), "›", font=self.f["chip"], fill=DIM)
                x += 26
        d.text((CW - 250, 10), f"{info.time:5.1f} 초", font=self.f["big"], fill=TXT)
        if time_limit and time_limit < 1e8:
            d.text((CW - 84, 24), f"/ {time_limit:.0f}", font=self.f["small"], fill=DIM)

    def _hand(self, d, h, env, info, frame, mapper, source, controllers, alert_hand, trainer_on) -> None:
        x0 = 0 if h == "left" else CW - 320
        hand = frame.hand(h) if frame else None
        conn = bool(hand and hand.connected)
        ai = bool(controllers) and controllers[h] == "ai"
        if alert_hand == h and int(time.perf_counter() * 3) % 2 == 0:
            d.rounded_rectangle((x0 + 6, 72, x0 + 314, 792), radius=18, outline=YEL, width=6)
        if source == "keyboard":
            conn_txt, conn_col = "키보드 입력", DIM
        elif conn:
            conn_txt, conn_col = "● 연결됨", GOOD
        else:
            conn_txt, conn_col = "○ 연결 안 됨 · 아무 버튼이나 눌러 깨우기", BAD
        d.text((x0 + 28, 122), fit(d, conn_txt, self.f["tiny"], 280), font=self.f["tiny"], fill=conn_col)
        if controllers:
            tag, col = ("AI가 조종", (120, 200, 255)) if ai else ("사람이 조종", (255, 190, 90))
            tw = d.textlength(tag, font=self.f["smallb"])
            d.text((x0 + 294 - tw, 92), tag, font=self.f["smallb"], fill=col)
        if conn and source != "keyboard":
            pressed = {b for b, v in hand.buttons.items() if v}
            stick_on = float(np.hypot(*hand.stick)) > 0.2
            if pressed or stick_on:
                self._draw_joycon(d, h, x0, pressed, stick_on,
                                  self._labels(h, mapper, source, trainer_on, self._visit), True)
        tg = env.targets[h]
        grip = "집는 중" if tg.grip else "열림"
        if info.in_tray[h]:
            grip += " · " + self.text["done"]
        d.text((x0 + 100, 610), grip, font=self.f["bodyb"], fill=GOOD if info.in_tray[h] else TXT)
        frac = max(0.0, min(1.0, (tg.pos[2] - Z_MIN) / (Z_CARRY - Z_MIN)))
        d.rounded_rectangle((x0 + 100, 656, x0 + 230, 674), radius=8, fill=(50, 58, 74))
        if frac > 0.04:
            d.rounded_rectangle((x0 + 100, 656, x0 + 100 + int(130 * frac), 674), radius=8, fill=HAND_RGB[h])
        d.text((x0 + 238, 650), f"{tg.pos[2] * 100:4.1f}cm", font=self.f["small"], fill=DIM)
        mode = mapper.height_mode(h)
        if ai or mapper.wrist_ready(h):
            wrist = f"기울기 {np.rad2deg(tg.pitch):+3.0f}° 회전 {np.rad2deg(tg.twist):+3.0f}°"
            col = TXT
        elif mode == "tilt" and conn and hand.buttons.get("shoulder"):
            b = mapper.last_bend[h]
            wrist, col = f"{'↓' if b > 0 else '↑'}{abs(b):2.0f}° {'내려가는 중' if b > 2.5 else '올라가는 중' if b < -2.5 else '멈춤'}", YEL
        elif mode == "tilt":
            wrist, col = ("L" if h == "left" else "R") + " 누르고 꺾기", DIM
        else:
            wrist, col = "고정", DIM
        d.text((x0 + 100, 690), fit(d, wrist, self.f["body"], 204), font=self.f["body"], fill=col)
        if mapper.wrist_ready(h) and source != "keyboard":
            d.text((x0 + 28, 736), fit(d, "기울이면 손목이 따라가요 · " + ("-" if h == "left" else "+") + ": 영점",
                                       self.f["tiny"], 280), font=self.f["tiny"], fill=DIM)
        if ai:
            d.text((x0 + 28, 760), fit(d, "스틱·버튼을 움직이면 넘겨받기", self.f["tiny"], 280), font=self.f["tiny"],
                   fill=(120, 200, 255))

    def _status(self, d, mode_line, source, fps, episode, notice, frame) -> None:
        y = 852
        for ln in wrap(d, mode_line or "자유 조종", self.f["small"], 266, 3):
            d.text((28, y), ln, font=self.f["small"], fill=YEL)
            y += 24
        d.text((28, y + 4), f"{episode}번째 판 · {source} · {fps:3.0f}fps", font=self.f["tiny"], fill=DIM)
        y += 32
        if notice:
            for ln in wrap(d, notice, self.f["tiny"], 266, 4):
                d.text((28, y), ln, font=self.f["tiny"], fill=(255, 190, 90))
                y += 20

    def _message(self, d, env, info, result, panel, state, flow, controllers, guide_text, trainer) -> None:
        x, y, w = 360, 818, 1200
        title, lines, sub, tcol = None, [], None, YEL
        if panel:
            title, lines = panel[0], list(panel[1])
        elif result:
            title, tcol = result, (GOOD if result.startswith("성공") else BAD)
            lines = [("캡처/홈 버튼(또는 R)을 누르면 새 판", TXT)]
            if flow:
                lines.append((flow.hint(), DIM))
        else:
            title = flow.hint() if flow else self.text.get("goal", self.text["sub"])
            coach = self._coach(env)
            if coach:
                lines.append((coach, TXT))
            if not flow:
                lines.append((self.text["sub"], DIM))
            if controllers:
                lines.append(("AI가 멈추거나 틀리면 그 손의 스틱이나 버튼을 움직여 넘겨받아 고쳐 주세요", DIM))
            if guide_text:
                lines.append((guide_text, DIM))
        right_w = 400 if hasattr(env, "tray_rot") else 0
        d.text((x, y), fit(d, title, self.f["big"], w - right_w), font=self.f["big"], fill=tcol)
        yy = y + 56
        for t, col in lines[:4]:
            font = self.f["msg"] if yy < y + 100 else self.f["body"]
            d.text((x, yy), fit(d, t, font, w - right_w), font=font, fill=col)
            yy += 40 if font is self.f["msg"] else 32
        if hasattr(env, "tray_rot"):                     # S2: how level the tray is
            tilt = float(np.degrees(np.arccos(np.clip(env.tray_rot()[2, 2], -1.0, 1.0))))
            col = GOOD if tilt < 5 else YEL if tilt < 10 else BAD
            gx = 1180
            d.text((gx, 970), f"트레이 기울기 {tilt:4.1f}°", font=self.f["bodyb"], fill=col)
            d.rounded_rectangle((gx, 1004, gx + 380, 1024), radius=10, fill=(50, 58, 74))
            d.rounded_rectangle((gx, 1004, gx + max(20, int(380 * min(tilt, 20) / 20)), 1024), radius=10, fill=col)

    @staticmethod
    def _coach(env) -> str | None:
        """S2: say which hand is low while the tray is up (from the simulation, not a script)."""
        if not hasattr(env, "handle_pos") or env.tray_lifted() < 0.02:
            return None
        dz = env.handle_pos("left")[2] - env.handle_pos("right")[2]
        if abs(dz) < 0.012:
            return "좋아요, 수평이에요 — 그대로 노란 칸 위로 옮기세요"
        low = "왼손" if dz < 0 else "오른손"
        return f"{low}이 조금 낮아요 — 트레이가 기울면 공이 굴러갑니다"

    def _record(self, d, history, check, flow, episodes) -> None:
        x0, y = 1628, 812
        if history:
            d.text((x0, y), "AI 실력 (처음 보는 배치)", font=self.f["h"], fill=TXT)
            for j, r in enumerate(history[-5:]):
                yy = y + 46 + 40 * j
                rate = float(r["success_rate"])
                d.text((x0, yy), fit(d, r["label"], self.f["tiny"], 150), font=self.f["tiny"], fill=TXT)
                d.rounded_rectangle((x0 + 154, yy + 2, x0 + 224, yy + 18), radius=6, fill=(50, 58, 74))
                if rate > 0:
                    d.rounded_rectangle((x0 + 154, yy + 2, x0 + 154 + max(8, int(70 * rate)), yy + 18), radius=6, fill=GOOD)
                d.text((x0 + 230, yy - 2), f"{rate * 100:3.0f}%", font=self.f["smallb"], fill=GOOD)
            return
        if check is not None:
            d.text((x0, y), "실기 점검", font=self.f["h"], fill=TXT)
            for j, (k, label) in enumerate(check.STEPS):
                ok = k in check.done
                d.text((x0, y + 40 + 29 * j), fit(d, ("■ " if ok else "□ ") + label, self.f["tiny"], 268),
                       font=self.f["tiny"], fill=GOOD if ok else TXT)
            return
        if flow and flow.step == "practice":
            from teachabot.visit import PRACTICE_ITEMS
            d.text((x0, y), "연습 체크", font=self.f["h"], fill=TXT)
            for j, (k, label) in enumerate(PRACTICE_ITEMS):
                ok = k in flow.practice_done
                d.text((x0, y + 46 + 40 * j), ("■ " if ok else "□ ") + label, font=self.f["body"], fill=GOOD if ok else TXT)
            return
        d.text((x0, y), "지난 판", font=self.f["h"], fill=TXT)
        rows = (episodes or [])[-5:]
        if not rows:
            d.text((x0, y + 46), "아직 없음", font=self.f["body"], fill=DIM)
        for j, e in enumerate(rows):
            ok = e.get("success")
            t = f"{e['episode']}판  " + (f"성공 {e['time']:.1f}초" if ok else f"실패 {e['time']:.0f}초")
            d.text((x0, y + 46 + 38 * j), ("● " if ok else "○ ") + t, font=self.f["body"], fill=GOOD if ok else BAD)

    def _result_banner(self, d, result: str) -> None:
        col = GOOD if result.startswith("성공") else BAD
        tw = d.textlength(result, font=self.f["huge"])
        x = SIM_X + (SIM_W - tw) / 2
        d.rounded_rectangle((x - 40, SIM_Y + 40, x + tw + 40, SIM_Y + 150), radius=20, fill=(16, 20, 28),
                            outline=col, width=3)
        d.text((x, SIM_Y + 52), result, font=self.f["huge"], fill=col)

    # ---------------------------------------------------------------- sheets over the 3D view
    def _sheet(self, d, title: str, hint: str) -> None:
        d.rounded_rectangle((SIM_X + 16, SIM_Y + 16, SIM_X + SIM_W - 16, SIM_Y + SIM_H - 16), radius=20,
                            fill=(24, 30, 42), outline=YEL, width=3)
        d.text((SIM_X + 46, SIM_Y + 32), title, font=self.f["title"], fill=TXT)
        d.text((SIM_X + 60 + d.textlength(title, font=self.f["title"]), SIM_Y + 42), hint, font=self.f["small"], fill=DIM)

    def _box(self, d, xy, title: str | None) -> None:
        d.rounded_rectangle(xy, radius=14, fill=PANEL, outline=LINE, width=2)
        if title:
            d.text((xy[0] + 18, xy[1] + 12), title, font=self.f["h"], fill=TXT)

    def _learning(self, img, d, trainer, policy, policy_path, history, reference, state) -> None:
        busy = bool(trainer and trainer.busy())
        hint = "학습은 닫아도 계속됩니다 · Y(또는 L): 닫기" if busy else "Y(또는 L): 닫기"
        if trainer and trainer.state == "done" and state in ("waiting", "finished"):
            hint = "A: 새 AI로 해 보기 · Y(또는 L): 닫기"
        self._sheet(d, "학습 화면", hint)
        L, T = SIM_X + 36, SIM_Y + 80
        # network
        self._box(d, (L, T, L + 600, T + 356), "AI의 뇌 (신경망)")
        weights, wlabel, nn_key = None, "아직 배운 AI가 없어요 (구조만)", None
        if trainer and trainer.live_weights is not None and busy:
            weights, nn_key = trainer.live_weights, ("live", id(trainer.live_weights))
            wlabel = f"배우는 중: 연결 세기가 바뀌고 있어요 ({trainer.progress['it']}/{trainer.progress['iters']}번)"
        elif policy is not None and hasattr(policy, "weights"):
            weights, nn_key = [np.asarray(w) for w, _ in policy.weights], ("policy", id(policy))
            wlabel = "지금 AI의 연결 세기 (파랑 +, 빨강 −)"
        if nn_key != self._nn_key or self._nn_img is None:
            self._nn_key, self._nn_img = nn_key, self._nn_image(weights)
        img.paste(self._nn_img, (L + 10, T + 48))            # numpy (made once per weight snapshot)
        d.text((L + 18, T + 326), fit(d, wlabel, self.f["tiny"], 570), font=self.f["tiny"], fill=DIM)
        # success graph
        self._graph(d, (L + 620, T, L + 1192, T + 356), history or [], reference or [])
        # progress
        P = T + 372
        self._box(d, (L, P, L + 600, P + 236), "학습 진행")
        self._progress(d, (L, P), trainer)
        # live trial
        self._box(d, (L + 620, P, L + 1192, P + 236), None)
        show = None
        if trainer and trainer.state == "evaluating" and trainer.pending_policy:
            show = trainer.pending_policy
        elif policy_path is not None and policy is not None:
            show = Path(policy_path)
        d.text((L + 638, P + 12), "AI 시험 화면", font=self.f["h"], fill=TXT)
        if show is None:
            for j, t in enumerate(("학습이 끝나면", "새 AI가 여기서", "처음 보는 배치를", "직접 조종해요")):
                d.text((L + 638, P + 56 + 30 * j), t, font=self.f["body"], fill=DIM)
            if self.trial:
                self.trial.show(None)
            return
        if self.trial is None:
            self.trial = TrialView(372, 210)
        frame_rgb, status = self.trial.show(show)
        img.paste(frame_rgb, (L + 808, P + 14))
        d.text((L + 638, P + 56), "처음 보는 배치", font=self.f["small"], fill=DIM)
        d.text((L + 638, P + 82), f"{self.trial.k + 1}번째 판", font=self.f["bodyb"], fill=TXT)
        d.text((L + 638, P + 116), status, font=self.f["bodyb"], fill=GOOD if status.startswith("성공") else
               BAD if status.startswith("실패") else TXT)
        ok = sum(self.trial.results)
        if self.trial.results:
            d.text((L + 638, P + 150), f"여기서 {ok}/{len(self.trial.results)}판 성공", font=self.f["small"], fill=DIM)
        d.text((L + 638, P + 190), "실제 AI가 조종 중", font=self.f["small"], fill=YEL)

    def _nn_image(self, weights) -> Image.Image:
        w, h = 580, 270
        im = Image.new("RGB", (w, h), PANEL)
        d = ImageDraw.Draw(im)
        sizes = [18, 256, 256, 6]
        shown = [18, 14, 14, 6]
        rng = np.random.default_rng(0)
        picks = [np.sort(rng.choice(n, k, replace=False)) if n > k else np.arange(n) for n, k in zip(sizes, shown)]
        xs = [128, 262, 382, 470]
        ys = [np.linspace(10, h - 26, len(p)) for p in picks]
        if weights is not None:
            for li, Wm in enumerate(weights[:3]):
                sub = np.asarray(Wm)[np.ix_(picks[li], picks[li + 1])]
                s = np.abs(sub).max() + 1e-9
                order = np.argsort(np.abs(sub), axis=None)
                for flat in order[len(order) // 3:]:                    # the stronger two thirds
                    i, j = np.unravel_index(flat, sub.shape)
                    v = sub[i, j] / s
                    col = (90, 160, 255) if v > 0 else (255, 120, 90)
                    c = tuple(int(PANEL[k] + (col[k] - PANEL[k]) * min(1.0, abs(v) * 1.2)) for k in range(3))
                    d.line((xs[li], ys[li][i], xs[li + 1], ys[li + 1][j]), fill=c, width=1)
        else:
            for li in range(3):
                for yi in ys[li]:
                    for yj in ys[li + 1]:
                        d.line((xs[li], yi, xs[li + 1], yj), fill=(44, 52, 66), width=1)
        for li, (x, yy) in enumerate(zip(xs, ys)):
            for y in yy:
                d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(60, 68, 86), outline=(200, 205, 215), width=1)
        f = ImageFont.truetype(str(FONT_PATH), 13)
        k = 0
        for n, label in IN_GROUPS:                                   # input groups (bc_hand.hand_obs order)
            ymid = float(np.mean(ys[0][k:k + n]))
            d.text((116, ymid), label, font=f, fill=TXT, anchor="rm")
            k += n
        for y, label in zip(ys[3], OUT_LABELS):
            d.text((482, y), label, font=f, fill=TXT, anchor="lm")
        for x, t in zip(xs, ("입력 18", "256개", "256개", "출력 6")):
            d.text((x, h - 8), t, font=f, fill=DIM, anchor="mm")
        return np.asarray(im).copy()

    def _graph(self, d, box, history, reference) -> None:
        x0, y0, x1, y1 = box
        self._box(d, box, "AI 실력 (처음 보는 배치 성공률)")
        gx0, gy0, gx1, gy1 = x0 + 56, y0 + 60, x1 - 24, y1 - 58
        rates = [float(r["success_rate"]) for r in history]
        refs = [(r["label"], ref_rate(r["value"])) for r in reference]
        refs = [(lab, v) for lab, v in refs if v is not None]
        top = max([0.4] + [v + 0.08 for v in rates] + [v + 0.08 for _, v in refs])
        top = math.ceil(top * 10) / 10
        py = lambda v: gy1 - v / top * (gy1 - gy0)
        for k in range(0, int(round(top * 10)) + 1, 1 if top <= 0.5 else 2):
            v = k / 10
            d.line((gx0, py(v), gx1, py(v)), fill=(44, 52, 66))
            d.text((gx0 - 8, py(v)), f"{v * 100:.0f}%", font=self.f["tiny"], fill=DIM, anchor="rm")
        shown_refs = [r for r in refs if "강화학습" in r[0]][:1] + [r for r in refs if "강화학습" not in r[0]][:1]
        for lab, v in shown_refs:
            col = BAD if "강화학습" in lab else (170, 178, 196)
            for xx in range(int(gx0), int(gx1), 14):
                d.line((xx, py(v), min(xx + 8, gx1), py(v)), fill=col, width=2)
            short = re.sub(r"\s*\(.*?\)", "", lab)
            d.text((gx0 + 6, py(v) - 4), fit(d, f"{short}: {v * 100:.0f}%", self.f["tiny"], (gx1 - gx0) * 0.6),
                   font=self.f["tiny"], fill=col, anchor="lb")
        if not history:
            d.text(((gx0 + gx1) / 2, (gy0 + gy1) / 2), "아직 시험 전", font=self.f["bodyb"], fill=DIM, anchor="mm")
            return
        n = len(history)
        px = lambda i: gx0 + 90 + (gx1 - gx0 - 180) * (i / max(1, n - 1) if n > 1 else 0.5)
        for i in range(n - 1):
            d.line((px(i), py(rates[i]), px(i + 1), py(rates[i + 1])), fill=GOOD, width=4)
        for i, (r, v) in enumerate(zip(history, rates)):
            d.ellipse((px(i) - 7, py(v) - 7, px(i) + 7, py(v) + 7), fill=GOOD)
            d.text((px(i), py(v) - 12), f"{v * 100:.0f}%", font=self.f["smallb"], fill=GOOD, anchor="mb")
            label = re.sub(r"^\S+조 ", "", r["label"])
            d.text((px(i), gy1 + 8 + 20 * (i % 2)), fit(d, label, self.f["tiny"], 170), font=self.f["tiny"],
                   fill=TXT, anchor="ma")

    def _progress(self, d, origin, trainer) -> None:
        L, P = origin
        if trainer is None:
            d.text((L + 18, P + 60), "이 모드에는 학습이 없어요", font=self.f["body"], fill=DIM)
            return
        pr = trainer.progress
        st = trainer.state
        train_frac = 1.0 if st in ("evaluating", "done") else (pr["it"] / max(1, pr["iters"]) if st == "training" else 0.0)
        eval_frac = 1.0 if st == "done" else (pr["eval_done"] / max(1, pr["eval_total"]) if st == "evaluating" else 0.0)
        rows = [("① 배우기", train_frac, f"{train_frac * 100:3.0f}%"),
                ("② 시험", eval_frac, f"{pr['eval_done']}/{pr['eval_total']}판" if st in ("evaluating", "done") else "대기")]
        for j, (lab, frac, txt) in enumerate(rows):
            y = P + 58 + 46 * j
            d.text((L + 18, y), lab, font=self.f["bodyb"], fill=TXT)
            d.rounded_rectangle((L + 130, y + 4, L + 470, y + 26), radius=11, fill=(50, 58, 74))
            if frac > 0.02:
                d.rounded_rectangle((L + 130, y + 4, L + 130 + int(340 * frac), y + 26), radius=11, fill=YEL)
            d.text((L + 480, y + 2), txt, font=self.f["small"], fill=DIM)
        losses = pr["loss"]
        if len(losses) >= 2:                       # error going down while it learns
            gx0, gy0, gx1, gy1 = L + 18, P + 160, L + 300, P + 222
            lv = np.log(np.asarray(losses) + 1e-6)
            lo, hi = float(lv.min()), float(lv.max()) + 1e-9
            pts = [(gx0 + (gx1 - gx0) * i / (len(lv) - 1), gy1 - (v - lo) / (hi - lo) * (gy1 - gy0))
                   for i, v in enumerate(lv)]
            d.line(pts, fill=(120, 200, 255), width=2)
            d.text((L + 316, P + 166), "틀린 정도(오차)", font=self.f["small"], fill=DIM)
            first = float(np.mean(losses[:3]))
            last = float(np.mean(losses[-3:]))
            d.text((L + 316, P + 192), f"{first:.2f} → {last:.2f}", font=self.f["bodyb"], fill=(120, 200, 255))
        else:
            msg = {"idle": "X 버튼: 지금까지 모은 시범으로 학습", "error": trainer.message}.get(st, trainer.message)
            for j, ln in enumerate(wrap(d, msg or "", self.f["body"], 560, 2)):
                d.text((L + 18, P + 160 + 30 * j), ln, font=self.f["body"], fill=DIM if st != "error" else BAD)

    def _result_sheet(self, img, d, group: str, history: list[dict], reference: list[dict]) -> None:
        self._sheet(d, f"{group} 결과", "진행자: 캡처/홈으로 닫기")
        L, T = SIM_X + 36, SIM_Y + 80
        self._graph(d, (L, T, L + 1192, T + 420), history, reference)
        y = T + 440
        d.text((L, y), "참고 (미리 잰 값)", font=self.f["h"], fill=TXT)
        for j, r in enumerate(reference[:4]):
            d.text((L, y + 40 + 32 * j), fit(d, f"· {r['label']}: {r['value']}", self.f["body"], 1190),
                   font=self.f["body"], fill=DIM)
