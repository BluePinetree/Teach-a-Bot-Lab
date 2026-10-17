"""Phase 2 manual test: standardized input from a live provider, recorded, then replayed.

Run from the project root (in a window the user can see):
    .venv\\Scripts\\python.exe tests\\manual\\test_input_providers.py            # Joy-Con
    .venv\\Scripts\\python.exe tests\\manual\\test_input_providers.py --source keyboard

The screen shows a checklist. When every item is [O] (or time runs out) it:
  1. saves the polled InputFrames to data/logs/input_<source>_<time>.jsonl,
  2. reloads that file through ReplayInputProvider and checks the reference
     consumer produces exactly the same hand trajectory as live,
  3. writes a .summary.json with pass/fail per check.
Stick directions are asked one at a time, so a flipped axis convention shows up
as "반대 방향 입력" instead of passing silently.
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

from teachabot.input.base import HANDS  # noqa: E402
from teachabot.input.joycon import JoyConInputProvider  # noqa: E402
from teachabot.input.keyboard import KeyboardInputProvider  # noqa: E402
from teachabot.input.reference_consumer import ReferenceConsumer, run_consumer  # noqa: E402
from teachabot.input.replay import InputRecorder, ReplayInputProvider  # noqa: E402

POLL_HZ = 60.0
DISPLAY_HZ = 10.0
FULL = 0.9        # stick value counted as "pushed to the end"
CROSS_MAX = 0.4   # the other axis must stay below this for a clean direction
HAND_KO = {"left": "왼쪽", "right": "오른쪽"}
DIRS = {"위": (1, +1), "아래": (1, -1), "왼쪽": (0, -1), "오른쪽": (0, +1)}  # name -> (axis, sign)
DIR_ORDER = ("위", "오른쪽", "아래", "왼쪽")
DIR_STEP = {"left": 1, "right": 2}  # checklist index of each hand's direction step
KEY_HINT = {"left": {"위": "W", "아래": "S", "왼쪽": "A", "오른쪽": "D", "grip": "Q"},
            "right": {"위": "↑", "아래": "↓", "왼쪽": "←", "오른쪽": "→", "grip": "오른쪽 Ctrl"}}


class Checklist:
    def __init__(self, source: str, provider):
        self.source, self.provider = source, provider
        self.dir_idx = {h: 0 for h in HANDS}          # index into DIR_ORDER of the direction asked now
        self.wrong_dir = {h: 0 for h in HANDS}        # times the opposite of the asked direction was pushed
        self._wrong_latched = {h: False for h in HANDS}
        self.deflected = {h: False for h in HANDS}
        self.release_zero = {h: False for h in HANDS}
        self.grip_seen = {h: False for h in HANDS}
        self.disconnect_ok = {h: False for h in HANDS}   # saw connected=False with neutral output
        self.reconnected = {h: False for h in HANDS}

    def update(self, f) -> None:
        cur = self.current_index()
        for h in HANDS:
            hand = f.hand(h)
            x, y = hand.stick
            if cur == DIR_STEP[h] and self.dir_idx[h] < len(DIR_ORDER):
                axis, sign = DIRS[DIR_ORDER[self.dir_idx[h]]]
                v, other = hand.stick[axis], hand.stick[1 - axis]
                if sign * v >= FULL and abs(other) <= CROSS_MAX:
                    self.dir_idx[h] += 1
                elif -sign * v >= FULL and abs(other) <= CROSS_MAX:
                    if not self._wrong_latched[h]:
                        self.wrong_dir[h] += 1
                        self._wrong_latched[h] = True
                elif (x, y) == (0.0, 0.0):
                    self._wrong_latched[h] = False
            if abs(x) > 0.5 or abs(y) > 0.5:
                self.deflected[h] = True
            if self.deflected[h] and hand.connected and (x, y) == (0.0, 0.0):
                self.release_zero[h] = True
            if hand.grip:
                self.grip_seen[h] = True
            if not hand.connected and hand.stick == (0.0, 0.0) and not any(hand.buttons.values()) \
                    and cur == len(self.steps()) - 1:
                self.disconnect_ok[h] = True
            if self.disconnect_ok[h] and hand.connected:
                self.reconnected[h] = True

    def steps(self) -> list[tuple[bool, str, list[str]]]:
        kb = self.source == "keyboard"
        out = [(all(getattr(self.provider, "_state", {}).get(h).calibrator.calibrated for h in HANDS)
                if not kb else True,
                "두 스틱에서 손을 떼고 1초 기다리기 (중립 보정)", [self._center_text()])]
        for h in HANDS:
            i = self.dir_idx[h]
            done = i >= len(DIR_ORDER)
            if done:
                detail = "완료"
            else:
                d = DIR_ORDER[i]
                key = f" (키 {KEY_HINT[h][d]})" if kb else ""
                detail = f">>> 지금: {HAND_KO[h]} 스틱을 [{d}] 방향{key} 끝까지 밀었다 놓기   ({i}/{len(DIR_ORDER)})"
            lines = [detail]
            if self.wrong_dir[h]:
                lines.append(f"!! 반대 방향 입력 {self.wrong_dir[h]}회 감지 — 방향 규칙이 뒤집혔을 수 있음")
            out.append((done, f"{HAND_KO[h]} 스틱: 화면이 말하는 방향으로 하나씩 밀기 (위→오른쪽→아래→왼쪽)", lines))
        out.append((all(self.release_zero.values()), "스틱을 놓으면 정확히 0으로 돌아오는지 (데드존)",
                    [f"왼쪽 {'O' if self.release_zero['left'] else '-'}   오른쪽 {'O' if self.release_zero['right'] else '-'}"]))
        g = "ZL / ZR" if not kb else "Q / 오른쪽 Ctrl"
        out.append((all(self.grip_seen.values()), f"집게 버튼 누르기: {g}",
                    [f"왼쪽 {'O' if self.grip_seen['left'] else '-'}   오른쪽 {'O' if self.grip_seen['right'] else '-'}"]))
        if kb:
            return out
        dis = any(self.disconnect_ok.values())
        rec = any(self.reconnected.values())
        out.append((rec, "한쪽 조이콘 SYNC 버튼을 짧게 눌러 끊기 → 아무 버튼 눌러 재연결",
                    ["재연결 확인" if rec else ("끊김 시 입력 0 확인, 아무 버튼을 눌러 재연결하세요" if dis else "대기 중")]))
        return out

    def _center_text(self) -> str:
        st = getattr(self.provider, "_state", None)
        if not st:
            return "키보드는 보정 불필요"
        return "   ".join(f"{HAND_KO[h]} 중립 {st[h].calibrator.center or '측정 중'}" for h in HANDS)

    def current_index(self) -> int | None:
        return next((i for i, (done, _, _) in enumerate(self.steps()) if not done), None)


def render(f, cl: Checklist, consumer: ReferenceConsumer, elapsed: float, rec_path: Path) -> str:
    lines = [f"Teach-a-Bot Lab - Phase 2 입력 표준화 테스트 [{f.source}]   경과 {elapsed:5.1f}s  (Ctrl+C 중단)",
             f"기록: {rec_path}", "", "=== 할 일: 위에서부터 차례로. 끝나면 [O] ==="]
    cur = cl.current_index()
    for i, (done, title, details) in enumerate(cl.steps()):
        lines.append(f"{'[O]' if done else '[>]' if i == cur else '[ ]'} {i + 1}. {title}")
        lines += [f"       {d}" for d in details]
    if cur is None:
        lines.append(">>> 모두 완료! 3초 뒤 기록을 저장하고 재생 검증을 합니다.")
    lines += ["", "=== 표준 입력 프레임 (게임이 받는 값) ==="]
    for h in HANDS:
        hand = f.hand(h)
        pressed = [b for b, v in hand.buttons.items() if v] or ["-"]
        lines.append(f"{HAND_KO[h]:>3}  연결 {str(hand.connected):5}  stick ({hand.stick[0]:+.2f}, {hand.stick[1]:+.2f})"
                     f"  grip {str(hand.grip):5}  buttons {' '.join(pressed)}")
        if hand.accel is not None:
            a, g = hand.accel, hand.gyro
            lines.append(f"      accel[G] ({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f})  gyro[dps] ({g[0]:+7.1f}, {g[1]:+7.1f}, {g[2]:+7.1f})")
    snap = consumer.snapshot()
    lines += ["", "참고: 이 입력으로 움직인 가상 손 위치 (x, z, 집게)  "
              + "   ".join(f"{HAND_KO[h]} ({snap[h][0]:+.2f}, {snap[h][1]:+.2f}, {'닫힘' if snap[h][2] else '열림'})" for h in HANDS)]
    if f.meta.get("imu_frame_verified") is False:
        lines.append("주의: R IMU 축 매핑 미측정 (accel/gyro는 아직 센서 좌표계)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["joycon", "keyboard"], default="joycon")
    ap.add_argument("--duration", type=float, default=300.0)
    args = ap.parse_args()
    os.system("")

    provider = JoyConInputProvider() if args.source == "joycon" else KeyboardInputProvider()
    log_dir = ROOT / "data" / "logs"
    rec_path = log_dir / f"input_{args.source}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    recorder = InputRecorder(rec_path, source=args.source)
    consumer = ReferenceConsumer()
    cl = Checklist(args.source, provider)
    live_frames, invalid = [], []
    provider.connect()
    t0 = time.perf_counter()
    done_at = None
    next_draw = 0.0
    try:
        while time.perf_counter() - t0 < args.duration:
            f = provider.poll()
            try:
                recorder.record(f)  # validates
                live_frames.append(f)
            except Exception as e:  # keep going, but count it
                invalid.append(f"seq {f.seq}: {e}")
            consumer.step(f)
            cl.update(f)
            now = time.perf_counter()
            if cl.current_index() is None:
                done_at = done_at or now
                if now - done_at >= 3.0:
                    break
            if now >= next_draw:
                sys.stdout.write("\x1b[H\x1b[J" + render(f, cl, consumer, now - t0, rec_path) + "\n")
                sys.stdout.flush()
                next_draw = now + 1.0 / DISPLAY_HZ
            time.sleep(max(0.0, 1.0 / POLL_HZ - (time.perf_counter() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        provider.disconnect()
        recorder.close()

    # replay verification
    replay = ReplayInputProvider(rec_path)
    replay.connect()
    replayed = [replay.poll() for _ in range(recorder.count)]
    same = run_consumer(replayed) == run_consumer(live_frames)

    steps = cl.steps()
    checks = {f"step{i + 1}_{title[:20]}": done for i, (done, title, _) in enumerate(steps)}
    checks["stick_direction_convention_ok"] = all(cl.dir_idx[h] >= len(DIR_ORDER) for h in HANDS) \
        and not any(cl.wrong_dir.values())
    checks["all_frames_valid"] = not invalid
    checks["replay_identical_to_live"] = same and len(replayed) == len(live_frames) > 0
    summary = {
        "source": args.source, "recording": str(rec_path), "frames": recorder.count,
        "seconds": round(time.perf_counter() - t0, 1),
        "poll_rate_hz": round(recorder.count / max(1e-9, live_frames[-1].timestamp - live_frames[0].timestamp), 1)
        if len(live_frames) > 1 else None,
        "invalid_frames": invalid[:20],
        "calibration": provider.calibration_status() if args.source == "joycon" else None,
        "stick_wrong_direction_count": cl.wrong_dir,
        "connection_events": [{k: v for k, v in e.items() if k != "t"} | {"t": round(e["t"] - t0, 2)}
                              for e in getattr(provider, "events", [])],
        "checks": checks,
    }
    out = rec_path.with_suffix(".summary.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nsummary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
