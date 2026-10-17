"""Motion control: drive the grippers by moving the Joy-Con instead of the stick.

The IMU is not a position sensor (double-integrating acceleration drifts within
seconds), so motion is turned into a *virtual stick* and the game is unchanged:

- tilt  : tilting the Joy-Con acts like pushing a joystick. Uses the gravity
          direction from the accelerometer, low-pass filtered so hand shake and
          the acceleration of moving the hand are suppressed. No drift.
- point : rotating the Joy-Con moves the gripper like a mouse (gyro rate ->
          gripper speed). Stops when the hand stops; small rates are dead-zoned.

Every axis is learned per hand from a short calibration (hold still, tilt
forward, tilt right, point right), all in that Joy-Con's own sensor frame. No
assumption is made about how L and R chips are oriented or about the sign
convention between accel and gyro, which is why the unverified R->L axis map
(DECISIONS D4) does not matter here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .base import HANDS, HandInput, InputFrame, InputProvider
from .processing import radial_deadzone

MODES = ("stick", "tilt", "point")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


# -- calibration -----------------------------------------------------------------

@dataclass
class HandMotionCalibration:
    """Per-hand axes in the Joy-Con's own sensor frame (accel in G, gyro in dps)."""

    neutral_g: list[float] | None = None       # unit gravity direction while held comfortably
    forward_dir: list[float] | None = None     # unit change of gravity when tilting the top forward
    right_dir: list[float] | None = None       # ... when tilting right (orthogonalised to forward)
    forward_tilt: float = 0.4                  # |change| at the calibration tilt (≈ sin of angle)
    right_tilt: float = 0.4
    pitch_axis: list[float] | None = None      # gyro axis (unit) for "tilt forward" rotation, sign = forward
    yaw_axis: list[float] | None = None        # gyro axis for "point right", sign = right

    @property
    def tilt_ready(self) -> bool:
        return None not in (self.neutral_g, self.forward_dir, self.right_dir)

    @property
    def point_ready(self) -> bool:
        return self.tilt_ready and None not in (self.pitch_axis, self.yaw_axis)


@dataclass
class MotionCalibration:
    hands: dict[str, HandMotionCalibration] = field(
        default_factory=lambda: {h: HandMotionCalibration() for h in HANDS})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({h: c.__dict__ for h, c in self.hands.items()}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MotionCalibration:
        if not path.exists():
            return cls()
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(hands={h: HandMotionCalibration(**d[h]) for h in HANDS})


CAL_STEPS = (
    ("neutral", "두 조이콘을 편하게 쥐고 가만히 (1.5초)"),
    ("forward", "두 조이콘 윗부분을 앞으로 기울였다가 돌아오기"),
    ("right", "두 조이콘을 오른쪽으로 기울였다가 돌아오기"),
    ("point", "두 조이콘으로 오른쪽을 가리키듯 돌렸다가 돌아오기"),
)


class MotionCalibrator:
    """Guided calibration fed with InputFrames; both hands are calibrated together.

    neutral : 1.5 s with gravity stable (std < 0.02 G) and gyro quiet -> neutral_g
    forward / right : wait until gravity moves > min_tilt away from neutral, keep
              the largest deviation, finish when the hand comes back near neutral.
              The gyro is integrated over the same motion to get the rotation axis.
    point   : integrate gyro until the hand has turned > min_turn_deg and come back
              to rest; the integrated vector's direction is the yaw axis.
    """

    def __init__(self, cal: MotionCalibration | None = None, steps=("neutral", "forward", "right", "point"),
                 min_tilt: float = 0.25, min_turn_deg: float = 25.0):
        self.cal = cal or MotionCalibration()
        self.steps = list(steps)
        self.i = 0
        self.min_tilt = min_tilt
        self.min_turn_deg = min_turn_deg
        self._reset_step()

    @property
    def done(self) -> bool:
        return self.i >= len(self.steps)

    @property
    def step(self) -> str | None:
        return None if self.done else self.steps[self.i]

    def progress(self) -> dict[str, str]:
        return {h: self._status[h] for h in HANDS}

    def _reset_step(self):
        self._buf = {h: [] for h in HANDS}
        self._peak = {h: (0.0, None) for h in HANDS}
        self._peak_rot = {h: np.zeros(3) for h in HANDS}
        self._gyro_int = {h: np.zeros(3) for h in HANDS}
        self._turned = {h: False for h in HANDS}
        self._hand_done = {h: False for h in HANDS}
        self._status = {h: "대기" for h in HANDS}
        self._last_t = None

    def update(self, frame: InputFrame) -> None:
        if self.done:
            return
        dt = 0.0 if self._last_t is None else min(max(frame.timestamp - self._last_t, 0.0), 0.1)
        self._last_t = frame.timestamp
        for h in HANDS:
            hand = frame.hand(h)
            if self._hand_done[h] or not hand.connected or hand.accel is None:
                continue
            a, w = np.asarray(hand.accel, float), np.asarray(hand.gyro, float)
            getattr(self, f"_{self.step}")(h, a, w, dt)
        if all(self._hand_done.values()):
            self.i += 1
            self._reset_step()

    def _neutral(self, h, a, w, dt):
        buf = self._buf[h]
        buf.append(a)
        quiet = np.linalg.norm(w) < 15.0 and abs(np.linalg.norm(a) - 1.0) < 0.15
        if not quiet:
            buf.clear()
        need = 90  # 1.5 s at 60 Hz
        self._status[h] = f"측정 중 {min(100, 100 * len(buf) // need)}%"
        if len(buf) >= need and np.asarray(buf[-need:]).std(axis=0).max() < 0.02:
            self.cal.hands[h].neutral_g = _unit(np.mean(buf[-need:], axis=0)).tolist()
            self._hand_done[h], self._status[h] = True, "완료"

    def _tilt(self, h, a, w, dt, which: str):
        g0 = np.asarray(self.cal.hands[h].neutral_g)
        d = _unit(a) - g0
        mag = float(np.linalg.norm(d))
        self._gyro_int[h] += w * dt
        best, _ = self._peak[h]
        if mag > best:
            # rotation so far = the tilt motion only (the way back would cancel it out)
            self._peak[h] = (mag, d)
            self._peak_rot[h] = self._gyro_int[h].copy()
        if self._peak[h][0] < self.min_tilt:
            self._status[h] = f"더 기울이세요 ({mag:.2f}/{self.min_tilt})"
            if mag < 0.05:
                self._gyro_int[h] = np.zeros(3)   # restart integration while still at rest
            return
        if mag > 0.1:
            self._status[h] = "좋아요, 이제 처음 자세로"
            return
        c = self.cal.hands[h]
        peak_mag, peak_d = self._peak[h]
        if which == "forward":
            c.forward_dir, c.forward_tilt = _unit(peak_d).tolist(), peak_mag
            c.pitch_axis = _unit(self._peak_rot[h]).tolist()
        else:
            f = np.asarray(c.forward_dir)
            r = peak_d - np.dot(peak_d, f) * f            # keep it orthogonal to forward
            c.right_dir, c.right_tilt = _unit(r).tolist(), float(np.linalg.norm(r))
        self._hand_done[h], self._status[h] = True, "완료"

    def _forward(self, h, a, w, dt):
        self._tilt(h, a, w, dt, "forward")

    def _right(self, h, a, w, dt):
        self._tilt(h, a, w, dt, "right")

    def _point(self, h, a, w, dt):
        self._gyro_int[h] += w * dt
        turned = float(np.linalg.norm(self._gyro_int[h]))
        if turned >= self.min_turn_deg:
            self._turned[h] = True
            self._peak[h] = (turned, self._gyro_int[h].copy()) if turned > self._peak[h][0] else self._peak[h]
        if not self._turned[h]:
            self._status[h] = f"더 돌리세요 ({turned:.0f}/{self.min_turn_deg:.0f}°)"
            if np.linalg.norm(w) < 5.0 and turned < 5.0:
                self._gyro_int[h] = np.zeros(3)
            return
        if np.linalg.norm(w) > 10.0:
            self._status[h] = "좋아요, 이제 처음 자세로"
            return
        c = self.cal.hands[h]
        c.yaw_axis = _unit(self._peak[h][1]).tolist()
        self._hand_done[h], self._status[h] = True, "완료"


# -- runtime mapping ---------------------------------------------------------------

@dataclass
class HandMotionState:
    g_lp: np.ndarray | None = None


class MotionMapper:
    """Turns a hand's IMU sample into a virtual stick for the chosen mode."""

    def __init__(self, cal: MotionCalibration, lowpass_hz: float = 3.0, tilt_deadzone: float = 0.15,
                 tilt_full_fraction: float = 0.8, point_full_dps: float = 90.0, point_deadzone_dps: float = 8.0,
                 accel_gate_g: float = 0.35):
        self.cal = cal
        self.lowpass_hz = lowpass_hz
        self.tilt_deadzone = tilt_deadzone
        self.tilt_full_fraction = tilt_full_fraction
        self.point_full_dps = point_full_dps
        self.point_deadzone_dps = point_deadzone_dps
        self.accel_gate_g = accel_gate_g
        self.state = {h: HandMotionState() for h in HANDS}

    def recenter(self, hand: str, accel) -> None:
        """Quick re-zero for a new student: current gravity direction becomes neutral."""
        self.cal.hands[hand].neutral_g = _unit(np.asarray(accel, float)).tolist()
        self.state[hand].g_lp = None

    def gravity(self, hand: str, accel, dt: float) -> np.ndarray:
        """First-order low-pass of the accelerometer. Samples far from 1 G (the hand
        is being accelerated) get a smaller weight, so a quick jerk does not register
        as tilt."""
        a = np.asarray(accel, float)
        st = self.state[hand]
        if st.g_lp is None:
            st.g_lp = a.copy()
            return _unit(st.g_lp)
        alpha = 1.0 - math.exp(-2 * math.pi * self.lowpass_hz * dt)
        if abs(np.linalg.norm(a) - 1.0) > self.accel_gate_g:
            alpha *= 0.1
        st.g_lp += alpha * (a - st.g_lp)
        return _unit(st.g_lp)

    def stick(self, mode: str, hand: str, h: HandInput, dt: float) -> tuple[float, float]:
        c = self.cal.hands[hand]
        if not h.connected or h.accel is None:
            return 0.0, 0.0
        g = self.gravity(hand, h.accel, dt)
        if mode == "tilt" and c.tilt_ready:
            d = g - np.asarray(c.neutral_g)
            x = float(np.dot(d, c.right_dir)) / (c.right_tilt * self.tilt_full_fraction)
            y = float(np.dot(d, c.forward_dir)) / (c.forward_tilt * self.tilt_full_fraction)
            x, y = max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))
            return radial_deadzone(x, y, self.tilt_deadzone)
        if mode == "point" and c.point_ready:
            w = np.asarray(h.gyro, float)
            yaw, pitch = float(np.dot(w, c.yaw_axis)), float(np.dot(w, c.pitch_axis))
            out = []
            for rate in (yaw, pitch):
                mag = abs(rate) - self.point_deadzone_dps
                out.append(0.0 if mag <= 0 else math.copysign(min(1.0, mag / self.point_full_dps), rate))
            return out[0], out[1]
        return h.stick


class MotionControlProvider(InputProvider):
    """Wraps a Joy-Con provider; in tilt/point mode replaces each hand's stick with
    the IMU-derived virtual stick. Buttons (grip etc.) pass through unchanged."""

    def __init__(self, base: InputProvider, calibration_path: Path, mode: str = "stick"):
        self.base = base
        self.source = base.source
        self.calibration_path = calibration_path
        self.cal = MotionCalibration.load(calibration_path)
        self.mapper = MotionMapper(self.cal)
        self.mode = mode
        self._last_t: float | None = None
        self.last_raw: InputFrame | None = None

    def connect(self) -> None:
        self.base.connect()

    def disconnect(self) -> None:
        self.base.disconnect()

    def ready(self, mode: str | None = None) -> bool:
        mode = mode or self.mode
        if mode == "stick":
            return True
        attr = "tilt_ready" if mode == "tilt" else "point_ready"
        return all(getattr(self.cal.hands[h], attr) for h in HANDS)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        for st in self.mapper.state.values():
            st.g_lp = None          # restart the low-pass from the current reading

    def save_calibration(self) -> None:
        self.cal.save(self.calibration_path)

    def poll(self) -> InputFrame:
        f = self.base.poll()
        self.last_raw = f
        dt = 0.0 if self._last_t is None else min(max(f.timestamp - self._last_t, 0.0), 0.1)
        self._last_t = f.timestamp
        hands = {}
        for h in HANDS:
            hand = f.hand(h)
            stick = self.mapper.stick(self.mode, h, hand, dt) if self.mode != "stick" else hand.stick
            if self.mode != "stick" and not self.ready():
                stick = (0.0, 0.0)
            hands[h] = replace(hand, stick=(float(stick[0]), float(stick[1]))) if hand.connected else hand
        return replace(f, left=hands["left"], right=hands["right"], meta={**f.meta, "control": self.mode})
