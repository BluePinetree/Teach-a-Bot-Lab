"""InputFrame -> ArmCommand for the ALOHA arms (decision D9).

Default mapping per hand:
  stick                      -> gripper moves (x right, y away)
  hold L / R (shoulder)      -> gripper goes down; release -> back up to carry height
  ZL / ZR (grip)             -> close the fingers while held
  Joy-Con tilt (IMU gravity) -> wrist: top forward = gripper tilts forward,
                                top to the right = gripper turns clockwise (seen from above)
The movement source is a setting so the shelved motion modes (tilt/point virtual stick,
later webcam) can be switched back in without touching the environment or learning code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from teachabot.input.base import InputFrame
from teachabot.input.motion import MotionCalibration, MotionMapper

from .env import MAX_VZ, PITCH_LIMIT, TWIST_LIMIT, ArmCommand
from .scene import HANDS

MOVE_SOURCES = ("stick", "tilt", "point")   # webcam / stroke: shelved (D9)
WRIST_SOURCES = ("imu", "off")
HEIGHT_SOURCES = ("button", "tilt", "motion")   # tilt: hold L/R + bend the wrist; motion: shelved (D18)


@dataclass
class TeleopSettings:
    move_source: str = "stick"
    wrist_source: str = "imu"
    wrist_gain: float = 1.0          # gripper angle per Joy-Con angle
    wrist_deadzone_deg: float = 4.0
    height_source: str = "button"    # button: hold L/R to lower, release to rise; tilt / motion: see HEIGHT_SOURCES
    height_gain: float = 1.0         # gripper speed per Joy-Con speed (motion)
    # tilt: while L/R is held, bending the wrist from where it was at the press moves the gripper;
    # front of the Joy-Con down = down. Speed grows from 0 at the deadzone to tilt_max_vz at tilt_full_deg.
    tilt_deadzone_deg: float = 2.5   # 4 / 20 deg felt too dull in the first try (2026-10-20)
    tilt_full_deg: float = 12.0
    tilt_max_vz: float = 0.5         # fraction of MAX_VZ (0.15 m/s): the 13 cm range takes ~1 s at full bend


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class WristFromTilt:
    """Joy-Con tilt angles from the low-passed gravity direction, in the frame learned by
    the per-hand calibration (neutral, forward, right). Gravity cannot see rotation about
    the vertical, so only the two tilt angles are used."""

    def __init__(self, cal: MotionCalibration, mapper: MotionMapper):
        self.cal, self.mapper = cal, mapper

    def ready(self, hand: str) -> bool:
        return self.cal.hands[hand].tilt_ready

    def angles(self, hand: str, accel, dt: float) -> tuple[float, float]:
        """(forward, right) tilt in radians relative to the neutral grip."""
        c = self.cal.hands[hand]
        g = self.mapper.gravity(hand, accel, dt)
        n = _unit(c.neutral_g)
        f = _unit(np.asarray(c.forward_dir) - np.dot(c.forward_dir, n) * n)
        r = np.asarray(c.right_dir) - np.dot(c.right_dir, n) * n
        r = _unit(r - np.dot(r, f) * f)
        a = float(np.dot(g, n))
        return math.atan2(float(np.dot(g, f)), a), math.atan2(float(np.dot(g, r)), a)

    def recenter(self, hand: str, accel) -> None:
        self.mapper.recenter(hand, accel)


class TeleopMapper:
    def __init__(self, settings: TeleopSettings, cal: MotionCalibration | None = None):
        if settings.move_source not in MOVE_SOURCES:
            raise ValueError(f"move_source must be one of {MOVE_SOURCES}")
        if settings.height_source not in HEIGHT_SOURCES:
            raise ValueError(f"height_source must be one of {HEIGHT_SOURCES}")
        if settings.wrist_source not in WRIST_SOURCES:
            raise ValueError(f"wrist_source must be one of {WRIST_SOURCES}")
        self.settings = settings
        self.cal = cal or MotionCalibration()
        self.mapper = MotionMapper(self.cal)
        self.wrist = WristFromTilt(self.cal, self.mapper)
        self.last_angles = {h: (0.0, 0.0) for h in HANDS}
        self.tilt_ref: dict[str, float | None] = {h: None for h in HANDS}   # forward angle at the L/R press
        self.last_bend = {h: 0.0 for h in HANDS}                           # degrees, for the HUD

    def wrist_ready(self, hand: str) -> bool:
        return self.settings.wrist_source == "imu" and self.wrist.ready(hand)

    def height_mode(self, hand: str) -> str:
        """The height source in effect: tilt needs the hand's calibration, else buttons."""
        s = self.settings.height_source
        return "button" if s == "tilt" and not self.wrist.ready(hand) else s

    def _tilt_vz(self, hand: str, held: bool, fwd: float) -> float:
        if not held:
            self.tilt_ref[hand], self.last_bend[hand] = None, 0.0
            return 0.0
        if self.tilt_ref[hand] is None:
            self.tilt_ref[hand] = fwd
        bend = math.degrees(fwd - self.tilt_ref[hand])
        self.last_bend[hand] = bend
        s = self.settings
        frac = (abs(bend) - s.tilt_deadzone_deg) / (s.tilt_full_deg - s.tilt_deadzone_deg)
        return -math.copysign(max(0.0, min(1.0, frac)) * s.tilt_max_vz, bend)   # front down = down

    def _shape(self, angle: float, limit: float) -> float:
        dz = math.radians(self.settings.wrist_deadzone_deg)
        mag = max(0.0, abs(angle) - dz) * self.settings.wrist_gain
        return math.copysign(min(mag, limit), angle)

    def commands(self, frame: InputFrame, dt: float) -> dict[str, ArmCommand]:
        out = {}
        for h in HANDS:
            hand = frame.hand(h)
            if not hand.connected:
                out[h] = ArmCommand(vz=1.0)       # safe: open, rise, stop
                continue
            if self.settings.move_source == "stick":
                move = hand.stick
            else:
                move = self.mapper.stick(self.settings.move_source, h, hand, dt)
            pitch = twist = 0.0
            mode = self.height_mode(h)
            fwd = None
            if (self.wrist_ready(h) or mode == "tilt") and hand.accel is not None:
                fwd, right = self.wrist.angles(h, hand.accel, dt)       # once per frame: it low-passes
                self.last_angles[h] = (fwd, right)
                if self.wrist_ready(h):
                    pitch, twist = self._shape(fwd, PITCH_LIMIT), self._shape(right, TWIST_LIMIT)
            if mode == "tilt":
                vz = self._tilt_vz(h, hand.buttons["shoulder"] and fwd is not None, fwd or 0.0)
            elif mode == "motion":
                v = float(frame.meta.get("vertical_speed", {}).get(h, 0.0))
                vz = max(-1.0, min(1.0, self.settings.height_gain * v / MAX_VZ))   # holds height when still
            else:
                vz = -1.0 if hand.buttons["shoulder"] else 1.0
            out[h] = ArmCommand(move=(float(move[0]), float(move[1])),
                                vz=vz,
                                pitch=pitch, twist=twist, grip=hand.grip)
        return out

    def recenter(self, hand: str, accel) -> None:
        if accel is not None:
            self.wrist.recenter(hand, accel)
