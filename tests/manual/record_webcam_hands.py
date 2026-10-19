"""Webcam hand-position feasibility recording (plan B: webcam = where the hand is).

A window shows the camera, the detected hands and one instruction at a time.
Hold both Joy-Cons as in the demo. An A4 sheet lies LANDSCAPE in front of you;
touch the bottom tip of the Joy-Con to the requested point on the sheet.

  Each point: move there, press ZR/ZL (or Space) when ready, then hold 1.5 s while it measures.
  1. calibration : right hand to the 4 corners        -> image->table homography
  2. test right  : right hand to centre + 4 edge midpoints (held out from calibration)
  3. test left   : left hand to the same 5 points
  4. still       : both hands still for 5 s           -> jitter, both-hands detection
  5. free        : move both hands freely for 15 s    -> detection while moving, L/R matching with Joy-Con IMU

Only hand landmarks (and Joy-Con reports) are saved, never video frames:
data/logs/webcam_<condition>_<time>.jsonl

    .venv\\Scripts\\python.exe tests\\manual\\record_webcam_hands.py --condition front
    .venv\\Scripts\\python.exe tests\\manual\\record_webcam_hands.py --condition side
Start with A (right Joy-Con) or Space. Esc aborts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from teachabot.input.joycon_device import JoyConDevice  # noqa: E402

FONT = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 30)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 22)
# A4 landscape, origin at sheet centre, x = right, z = forward (away from you), cm
HX, HZ = 14.85, 10.5
POINTS = {"앞-왼쪽 모서리": (-HX, HZ), "앞-오른쪽 모서리": (HX, HZ), "뒤-오른쪽 모서리": (HX, -HZ),
          "뒤-왼쪽 모서리": (-HX, -HZ), "가운데": (0, 0), "앞 가장자리 가운데": (0, HZ),
          "뒤 가장자리 가운데": (0, -HZ), "왼쪽 가장자리 가운데": (-HX, 0), "오른쪽 가장자리 가운데": (HX, 0)}
CORNERS = list(POINTS)[:4]
TESTS = list(POINTS)[4:]
HOLD_S, SETTLE_S = 1.5, 0.5   # after the confirm press: settle, then measure
PALM = (0, 5, 9, 13, 17)


def steps():
    out = [("cal", "right", p) for p in CORNERS]
    out += [("test", "right", p) for p in TESTS]
    out += [("test", "left", p) for p in TESTS]
    out += [("still", "both", None), ("free", "both", None)]
    return out


def put_text(frame, lines, org=(30, 30), big=True, color=(255, 255, 255)):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    y = org[1]
    for i, line in enumerate(lines):
        f = FONT if (big and i == 0) else FONT_S
        d.text((org[0] + 2, y + 2), line, font=f, fill=(0, 0, 0))
        d.text(org[0:1] + (y,), line, font=f, fill=color)
        y += (42 if (big and i == 0) else 30)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_sheet(frame, target, hand):
    h, w = frame.shape[:2]
    sx, sy, s = w - 330, h - 250, 9.0          # schematic: 1 cm = 9 px
    cv2.rectangle(frame, (sx, sy), (sx + int(2 * HX * s), sy + int(2 * HZ * s)), (255, 255, 255), 2)
    for name, (x, z) in POINTS.items():
        px, py = int(sx + (x + HX) * s), int(sy + (HZ - z) * s)
        on = name == target
        cv2.circle(frame, (px, py), 11 if on else 5, (0, 200, 255) if on else (200, 200, 200), -1)
    cv2.putText(frame, "FAR (away from you)", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "YOU", (sx + int(HX * s) - 15, sy + int(2 * HZ * s) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, help="e.g. front, side, tv")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-joycon", action="store_true")
    ap.add_argument("--smoke", type=float, default=0, help="(dev) quit after N seconds without starting")
    ap.add_argument("--distance-sim", action="store_true",
                    help="also run detection on downscaled frames (as if the camera were 2-4x farther), "
                         "full frame and cropped to the work area, and log all of them")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        # MediaPipe cannot open files under a non-ASCII path (this project lives in "잡동사니"),
        # so hand it the model bytes instead of a path.
        base_options=mpt.BaseOptions(model_asset_buffer=(ROOT / "models" / "hand_landmarker.task").read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    model_bytes = (ROOT / "models" / "hand_landmarker.task").read_bytes()
    # distance simulation: scale s shrinks the image like a camera 1/s times farther;
    # "roi" additionally crops the work area (digital zoom) before detection
    VARIANTS = ([(s, m) for s in (0.5, 0.33, 0.25) for m in ("full", "roi")] + [(1.0, "roi")]) \
        if args.distance_sim else []
    var_lm = {f"s{s:.2f}_{m}": vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO, num_hands=2)) for s, m in VARIANTS}
    roi = {"box": None}   # (x0, y0, x1, y1) in full-res pixels, fixed when the run starts
    devs = {} if args.no_joycon else {"L": JoyConDevice("L"), "R": JoyConDevice("R")}

    def run_variants(frame, ts_ms):
        out = {}
        for (s, m) in VARIANTS:
            name = f"s{s:.2f}_{m}"
            if m == "roi" and roi["box"] is None:
                continue
            x0, y0, x1, y1 = roi["box"] if m == "roi" else (0, 0, W, H)
            crop = frame[y0:y1, x0:x1]
            small = cv2.resize(crop, (max(16, int(crop.shape[1] * s)), max(16, int(crop.shape[0] * s))),
                               interpolation=cv2.INTER_AREA)
            r = var_lm[name].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), ts_ms)
            hs = []
            cw, ch = x1 - x0, y1 - y0
            for lmk in r.hand_landmarks:
                palm = np.mean([(lmk[k].x, lmk[k].y) for k in PALM], axis=0)
                size = np.hypot((lmk[0].x - lmk[9].x) * small.shape[1], (lmk[0].y - lmk[9].y) * small.shape[0])
                hs.append({"palm_px": [float(x0 + palm[0] * cw), float(y0 + palm[1] * ch)],
                           "size_px": round(float(size), 1)})     # wrist -> middle knuckle, in the image the model saw
            out[name] = hs
        return out

    def keep_connected():   # background: (re)connect any Joy-Con that is not delivering reports
        import threading
        def loop():
            while True:
                for d in devs.values():
                    if not d.connected and not d.awaiting_first_report():
                        d.mark_disconnected()
                        d.connect()
                time.sleep(2.0)
        threading.Thread(target=loop, daemon=True).start()
    if devs:
        keep_connected()

    log_path = ROOT / "data" / "logs" / f"webcam_{args.condition}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log = log_path.open("w", encoding="utf-8")
    write = lambda r: log.write(json.dumps(r, ensure_ascii=False) + "\n")
    write({"type": "header", "condition": args.condition, "width": W, "height": H, "points_cm": POINTS,
           "protocol": "user-paced", "settle_s": SETTLE_S, "hold_s": HOLD_S, "created": datetime.now().isoformat(timespec="seconds")})

    win = "Teach-a-Bot Lab - webcam hands"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    t0 = time.perf_counter()
    plan = steps()
    state = {"i": -1, "t_step": None, "sub": "wait"}
    stats = {"frames": 0, "both": 0}
    prev_btn = {"v": False}

    def confirm_pressed() -> bool:
        down = False
        for k, d in devs.items():
            s = d.latest()
            if s and (s.buttons.get("zr") or s.buttons.get("zl") or s.buttons.get("a")):
                down = True
        edge = down and not prev_btn["v"]
        prev_btn["v"] = down
        return edge

    def flush_joycons():
        for d in devs.values():
            for s in d.drain():
                rec = s.to_record()
                write({"type": "joycon", "side": rec["side"], "t": rec["t"], "accel_g": rec["accel_g"],
                       "gyro_dps": rec["gyro_dps"], "buttons": rec["buttons"]})

    aborted = False
    while True:
        ok, frame = cap.read()
        now = time.perf_counter()
        if not ok:
            continue
        ts_ms = int((now - t0) * 1000)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts_ms)
        hands = []
        for lmk, hd in zip(res.hand_landmarks, res.handedness):
            pts = [(p.x, p.y, p.z) for p in lmk]
            palm = np.mean([pts[k][:2] for k in PALM], axis=0)
            hands.append({"label": hd[0].category_name, "score": round(hd[0].score, 3),
                          "palm_px": [float(palm[0] * W), float(palm[1] * H)],
                          "size_px": round(float(np.hypot((pts[0][0] - pts[9][0]) * W, (pts[0][1] - pts[9][1]) * H)), 1),
                          "lm": [[round(c, 5) for c in p] for p in pts]})
        variants = run_variants(frame, ts_ms) if VARIANTS else None
        # user-paced: each step waits for a confirm press (ZR/ZL/A or Space), settles, then measures
        confirm = confirm_pressed() or state.pop("space", False)
        step = plan[state["i"]] if 0 <= state["i"] < len(plan) else None
        phase = None
        if step:
            kind = step[0]
            hold_s = HOLD_S if kind in ("cal", "test") else (5.0 if kind == "still" else 15.0)
            if state["sub"] == "wait":
                phase = "move"
                if confirm:
                    state["sub"], state["t_step"] = "settle", now
            elif state["sub"] == "settle":
                phase = "move"
                if now - state["t_step"] >= SETTLE_S:
                    state["sub"], state["t_step"] = "hold", now
            else:
                phase = "hold"
                if now - state["t_step"] >= hold_s:
                    state["i"] += 1
                    state["sub"], state["t_step"] = "wait", now
                    step = plan[state["i"]] if state["i"] < len(plan) else None
        rec = {"type": "frame", "t": now, "step": state["i"], "phase": phase, "hands": hands}
        if variants is not None:
            rec["variants"] = variants
        write(rec)
        flush_joycons()
        stats["frames"] += 1
        stats["both"] += len(hands) == 2

        # -- display --
        view = frame.copy()
        for hnd in hands:
            for p in hnd["lm"]:
                cv2.circle(view, (int(p[0] * W), int(p[1] * H)), 4, (0, 255, 0), -1)
            cv2.circle(view, tuple(int(v) for v in hnd["palm_px"]), 12, (0, 128, 255), 3)
        view = cv2.resize(view, (1280, 720))
        labels = " / ".join(f"{h['label']} {h['score']:.2f}" for h in hands) or "-"
        det = f"손 검출: {len(hands)}개   ({labels})"
        joy = "  조이콘: " + " ".join(f"{k}{'O' if d.connected else 'X'}" for k, d in devs.items()) if devs else ""
        if state["i"] < 0:
            lines = ["[구도 맞추기] 웹캠을 아래로 숙여 주세요",
                     "1) 복사용지 한 장을 '가로로' 몸 앞 책상에 놓기",
                     "2) 양손에 조이콘을 쥐고 종이 위에 올리기 → 종이 전체와 두 손이 화면 안에 다 보이게 카메라 각도 조절",
                     "3) 두 손에 초록 점이 찍히면(손 검출 2개) 준비 끝 → A(오른쪽 조이콘) 또는 Space로 시작",
                     "※ 시작하면 조이콘 아래 끝을 종이의 모서리·변 가운데에 차례로 대게 됩니다 (오른쪽 아래 그림)",
                     det + joy]
            draw_sheet(view, None, "both")
            if confirm:
                state["i"], state["t_step"], state["sub"] = 0, now, "wait"
                if VARIANTS:   # work-area crop: half the width, 60% of the height, centred on the hands
                    c = np.mean([h["palm_px"] for h in hands], axis=0) if hands else np.array([W / 2, H / 2])
                    bw, bh = int(W * 0.5), int(H * 0.6)
                    x0 = int(np.clip(c[0] - bw / 2, 0, W - bw))
                    y0 = int(np.clip(c[1] - bh / 2, 0, H - bh))
                    roi["box"] = (x0, y0, x0 + bw, y0 + bh)
                    write({"type": "roi", "box": roi["box"]})
        elif step is None:
            break
        else:
            kind, hand, target = step
            who = {"right": "오른손", "left": "왼손", "both": "양손"}[hand]
            if state["sub"] == "wait":
                if kind in ("cal", "test"):
                    head = f"{who} 조이콘 끝을 [{target}]에 대고 → 준비되면 ZR/ZL 누르기"
                elif kind == "still":
                    head = "양손을 종이 위에 두고 → 준비되면 ZR/ZL 누르기 (그다음 5초 가만히)"
                else:
                    head = "마지막: ZR/ZL 누르면 15초 동안 양손을 자유롭게 움직이기"
            elif state["sub"] == "settle":
                head = "좋아요, 그대로…"
            else:
                left = hold_s - (now - state["t_step"])
                head = (f"● 측정 중 — 가만히 ({left:3.1f}초)" if kind != "free"
                        else f"● 자유롭게 움직이세요: 앞뒤·좌우·위아래 ({left:3.1f}초)")
            lines = [head, f"단계 {state['i'] + 1}/{len(plan)}   (시간 제한 없음, 버튼을 누를 때 측정)   조건: {args.condition}",
                     det + joy]
            draw_sheet(view, target, hand)
        view = put_text(view, lines)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            aborted = True
            break
        if key == 32:
            state["space"] = True
        if args.smoke and now - t0 > args.smoke:
            aborted = True
            break

    write({"type": "end", "t": time.perf_counter(), "aborted": aborted, **stats})
    log.close()
    cap.release()
    cv2.destroyAllWindows()
    for d in devs.values():
        d.mark_disconnected()
    print(json.dumps({"log": str(log_path), "aborted": aborted, **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
