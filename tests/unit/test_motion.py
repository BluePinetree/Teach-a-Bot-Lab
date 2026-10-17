"""Motion control on synthetic IMU data from physically simulated hand rotations.

World frame matches the game: x = right, y = up, z = forward (away from player).
Each Joy-Con gets an arbitrary sensor frame M (hand -> sensor). The right one is
given a *reflection* so the tests prove nothing depends on L/R chip orientation
or on the accel/gyro handedness convention.
"""
import math

import numpy as np
import pytest

from teachabot.input.base import BUTTONS, HandInput, InputFrame
from teachabot.input.motion import MotionCalibration, MotionCalibrator, MotionMapper

DT = 1 / 60
UP = np.array([0.0, 1.0, 0.0])


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


M_LEFT = rot(_ := np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1, 2, 3]), 70)
M_RIGHT = rot(np.array([0.0, 1.0, 0.0]), 180) @ np.diag([1.0, 1.0, -1.0])   # reflection
MS = {"left": M_LEFT, "right": M_RIGHT}

# basic motions in the world frame (hand orientation R, world-from-hand)
FORWARD = (np.array([1.0, 0, 0]), +1)    # top toward +z = rotate about +x
RIGHT_TILT = (np.array([0, 0, 1.0]), -1)  # top toward +x = rotate about -z
YAW_RIGHT = (np.array([0, 1.0, 0]), +1)   # pointing direction z -> +x


class Sim:
    def __init__(self, seed=0):
        self.R = {h: np.eye(3) for h in MS}
        self.t = 0.0
        self.rng = np.random.default_rng(seed)

    def sample(self, hand, w_world):
        M, R = MS[hand], self.R[hand]
        accel = M @ R.T @ UP + self.rng.normal(0, 0.005, 3)
        w_body = R.T @ w_world
        gyro = np.linalg.det(M) * (M @ w_body) + self.rng.normal(0, 0.3, 3)
        return HandInput(True, (0.0, 0.0), {b: False for b in BUTTONS[hand]},
                         accel=tuple(accel), gyro=tuple(gyro))

    def frames(self, seconds, motion=None, deg=0.0):
        """Rotate both hands by `deg` about the motion axis over `seconds` (or hold still)."""
        n = max(1, round(seconds / DT))
        axis, sign = motion if motion else (np.array([1.0, 0, 0]), 0)
        rate = sign * deg / seconds if motion else 0.0         # deg/s
        out = []
        for _ in range(n):
            w_world = axis * rate
            step = rot(axis, rate * DT)
            for h in MS:
                self.R[h] = step @ self.R[h]
            self.t += DT
            out.append(InputFrame(self.t, len(out), "sim", self.sample("left", w_world),
                                  self.sample("right", w_world)))
        return out


def calibrated(seed=0):
    sim, cal = Sim(seed), MotionCalibrator()
    script = [(2.0, None, 0), (0.5, FORWARD, 30), (0.5, FORWARD, -30), (0.5, None, 0),
              (0.5, RIGHT_TILT, 30), (0.5, RIGHT_TILT, -30), (0.5, None, 0),
              (0.6, YAW_RIGHT, 45), (0.6, YAW_RIGHT, -45), (0.5, None, 0)]
    for seconds, motion, deg in script:
        if motion and deg < 0:
            motion = (motion[0], -motion[1])
        for f in sim.frames(seconds, motion, abs(deg)):
            cal.update(f)
    return sim, cal


def test_calibration_completes_for_both_hands():
    _, cal = calibrated()
    assert cal.done, (cal.step, cal.progress())
    for h in ("left", "right"):
        c = cal.cal.hands[h]
        assert c.tilt_ready and c.point_ready
        assert abs(np.dot(c.forward_dir, c.right_dir)) < 0.1


def settle(mapper, sim, mode, seconds=1.5, motion=None, deg=0.0):
    out = None
    for f in sim.frames(seconds, motion, deg):
        out = {h: mapper.stick(mode, h, f.hand(h), DT) for h in ("left", "right")}
    return out


@pytest.mark.parametrize("motion,expect", [(FORWARD, (0, +1)), ((FORWARD[0], -1), (0, -1)),
                                           (RIGHT_TILT, (+1, 0)), ((RIGHT_TILT[0], +1), (-1, 0))])
def test_tilt_mode_directions(motion, expect):
    sim, cal = calibrated()
    mapper = MotionMapper(cal.cal)
    settle(mapper, sim, "tilt")
    for f in sim.frames(0.5, motion, 20):
        pass
    sticks = settle(mapper, sim, "tilt", seconds=1.0)
    for h, (x, y) in sticks.items():
        ex, ey = expect
        if ex:
            assert x * ex > 0.5 and abs(y) < 0.3, (h, x, y)
        if ey:
            assert y * ey > 0.5 and abs(x) < 0.3, (h, x, y)


def test_tilt_neutral_is_zero_and_shake_is_filtered():
    sim, cal = calibrated()
    mapper = MotionMapper(cal.cal)
    sticks = settle(mapper, sim, "tilt")
    assert all(s == (0.0, 0.0) for s in sticks.values())
    # a sharp 1-frame jerk of 2 G sideways must not register as a full tilt
    f = sim.frames(DT)[0]
    jerk = {h: HandInput(True, (0.0, 0.0), f.hand(h).buttons,
                         accel=tuple(np.asarray(f.hand(h).accel) + MS[h] @ np.array([2.0, 0, 0])),
                         gyro=f.hand(h).gyro) for h in ("left", "right")}
    for h in jerk:
        x, y = mapper.stick("tilt", h, jerk[h], DT)
        assert math.hypot(x, y) < 0.3


@pytest.mark.parametrize("motion,expect", [(YAW_RIGHT, (+1, 0)), ((YAW_RIGHT[0], -1), (-1, 0)),
                                           (FORWARD, (0, +1))])
def test_point_mode_follows_rotation_and_stops(motion, expect):
    sim, cal = calibrated()
    mapper = MotionMapper(cal.cal)
    moving = settle(mapper, sim, "point", seconds=0.5, motion=motion, deg=40)   # 80 deg/s
    for h, (x, y) in moving.items():
        ex, ey = expect
        if ex:
            assert x * ex > 0.5 and abs(y) < 0.3, (h, x, y)
        if ey:
            assert y * ey > 0.5 and abs(x) < 0.3, (h, x, y)
    still = settle(mapper, sim, "point", seconds=0.3)
    assert all(s == (0.0, 0.0) for s in still.values())


def test_uncalibrated_falls_back_to_stick():
    mapper = MotionMapper(MotionCalibration())
    h = HandInput(True, (0.3, -0.2), {b: False for b in BUTTONS["left"]}, accel=(0, 0, 1), gyro=(0, 0, 0))
    assert mapper.stick("tilt", "left", h, DT) == (0.3, -0.2)


def test_calibration_roundtrip(tmp_path):
    _, cal = calibrated()
    p = tmp_path / "cal.json"
    cal.cal.save(p)
    assert MotionCalibration.load(p).hands["right"].yaw_axis == pytest.approx(cal.cal.hands["right"].yaw_axis)
