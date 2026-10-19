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

from .env import PITCH_LIMIT, TWIST_LIMIT, ArmCommand
from .scene import HANDS

MOVE_SOURCES = ("stick", "tilt", "point")   # webcam / stroke: shelved (D9)
WRIST_SOURCES = ("imu", "off")


@dataclass
class TeleopSettings:
    move_source: str = "stick"
    wrist_source: str = "imu"
    wrist_gain: float = 1.0          # gripper angle per Joy-Con angle
    wrist_deadzone_deg: float = 4.0


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
        if settings.wrist_source not in WRIST_SOURCES:
            raise ValueError(f"wrist_source must be one of {WRIST_SOURCES}")
        self.settings = settings
        self.cal = cal or MotionCalibration()
        self.mapper = MotionMapper(self.cal)
        self.wrist = WristFromTilt(self.cal, self.mapper)
        self.last_angles = {h: (0.0, 0.0) for h in HANDS}

    def wrist_ready(self, hand: str) -> bool:
        return self.settings.wrist_source == "imu" and self.wrist.ready(hand)

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
            if self.wrist_ready(h) and hand.accel is not None:
                fwd, right = self.wrist.angles(h, hand.accel, dt)
                self.last_angles[h] = (fwd, right)
                pitch, twist = self._shape(fwd, PITCH_LIMIT), self._shape(right, TWIST_LIMIT)
            out[h] = ArmCommand(move=(float(move[0]), float(move[1])),
                                vz=-1.0 if hand.buttons["shoulder"] else 1.0,
                                pitch=pitch, twist=twist, grip=hand.grip)
        return out

    def recenter(self, hand: str, accel) -> None:
        if accel is not None:
            self.wrist.recenter(hand, accel)
