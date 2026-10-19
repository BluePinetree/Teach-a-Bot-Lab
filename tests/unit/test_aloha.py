import math

import numpy as np
import pytest

from teachabot.aloha.env import (CONTROL_DT, MAX_SPEED, WORKSPACE, Z_CARRY, Z_MIN, AlohaSortEnv, ArmCommand,
                                 target_rotation)
from teachabot.aloha.scene import CUBE_HALF, HANDS, S1_LAYOUT, TRAY_WALL_H
from teachabot.aloha.teleop import TeleopMapper, TeleopSettings
from teachabot.input.base import HandInput, InputFrame
from teachabot.input.motion import HandMotionCalibration, MotionCalibration

from aloha_scripted import run_scripted


@pytest.fixture(scope="module")
def env():
    return AlohaSortEnv()


def test_reset_is_reproducible_and_inside_spawn(env):
    env.reset(7)
    a = {h: env.cube_pos(h) for h in HANDS}
    env.reset(7)
    for h in HANDS:
        np.testing.assert_allclose(env.cube_pos(h), a[h])
    env.reset(8)
    assert any(not np.allclose(env.cube_pos(h), a[h]) for h in HANDS)
    for seed in range(20):
        env.reset(seed)
        for h in HANDS:
            (xr, yr) = S1_LAYOUT.cube_spawn[h]
            p = env.cube_pos(h)
            assert xr[0] <= p[0] <= xr[1] and yr[0] <= p[1] <= yr[1]


def test_home_pose_points_down_at_target(env):
    env.reset(0)
    for h in HANDS:
        np.testing.assert_allclose(env.site_pos(h), env.targets[h].pos, atol=2e-3)
        R = env.data.site(f"{h}/gripper").xmat.reshape(3, 3)
        np.testing.assert_allclose(R, target_rotation(h, 0, 0), atol=1e-2)


def test_stick_moves_gripper_and_workspace_clamps(env):
    env.reset(0)
    x0 = env.targets["right"].pos[0]
    for _ in range(25):                      # 0.5 s full right
        env.step({"right": ArmCommand(move=(1, 0))})
    assert env.targets["right"].pos[0] == pytest.approx(x0 + MAX_SPEED * 0.5, abs=1e-6)
    for _ in range(50):
        env.step({"right": ArmCommand(move=(1, 0))})
    assert env.targets["right"].pos[0] == pytest.approx(WORKSPACE["right"][0][1])
    for _ in range(25):
        env.step({"right": ArmCommand()})
    assert np.linalg.norm(env.site_pos("right") - env.targets["right"].pos) < 0.02


def test_height_stays_in_range(env):
    env.reset(0)
    for _ in range(100):
        env.step({"left": ArmCommand(vz=-1)})
    assert env.targets["left"].pos[2] == pytest.approx(Z_MIN)
    for _ in range(100):
        env.step({"left": ArmCommand(vz=1)})
    assert env.targets["left"].pos[2] == pytest.approx(Z_CARRY)


def test_wrist_tilt_and_twist_follow(env):
    env.reset(0)
    cmd = ArmCommand(pitch=math.radians(30), twist=math.radians(45))
    for _ in range(75):
        env.step({"left": cmd, "right": cmd})
    for h in HANDS:
        R = env.data.site(f"{h}/gripper").xmat.reshape(3, 3)
        Rt = target_rotation(h, math.radians(30), math.radians(45))
        angle = math.degrees(math.acos(np.clip((np.trace(Rt.T @ R) - 1) / 2, -1, 1)))
        assert angle < 5.0


def test_tray_detection_and_fall(env):
    env.reset(0)
    for h in HANDS:
        assert not env.in_tray(h)
    for h in HANDS:
        a = env._cube_qadr[h]
        env.data.qpos[a:a + 3] = (*S1_LAYOUT.tray_pos[h], CUBE_HALF + 0.005)
    for _ in range(25):
        _, info = env.step({})
    assert info.success and all(info.in_tray.values())
    env.reset(0)
    a = env._cube_qadr["left"]
    env.data.qpos[a:a + 3] = (-0.7, 0.0, 0.3)   # off the table
    for _ in range(50):
        _, info = env.step({})
    assert info.failed and not info.success


def test_physics_can_solve_s1(env):
    """Capability check with a scripted controller (test-only, never used as AI output)."""
    results = [run_scripted(env, s)[0] for s in range(20)]
    assert sum(r.success for r in results) >= 18


def test_command_vector_round_trip():
    c = ArmCommand(move=(0.5, -0.25), vz=-1, pitch=0.3, twist=-0.2, grip=True)
    assert ArmCommand.from_vector(c.to_vector()) == c


# -- teleop mapping ----------------------------------------------------------------------

def _frame(left: HandInput, right: HandInput) -> InputFrame:
    return InputFrame(timestamp=0.0, seq=0, source="test", left=left, right=right)


def _hand(name, **kw) -> HandInput:
    h = HandInput.neutral(name)
    buttons = dict(h.buttons)
    buttons.update(kw.pop("buttons", {}))
    return HandInput(connected=True, stick=kw.pop("stick", (0.0, 0.0)), buttons=buttons,
                     accel=kw.pop("accel", (0.0, 0.0, 1.0)), gyro=(0.0, 0.0, 0.0), **kw)


def _cal() -> MotionCalibration:
    c = HandMotionCalibration(neutral_g=[0, 0, 1], forward_dir=[1, 0, 0], right_dir=[0, 1, 0])
    return MotionCalibration(hands={"left": c, "right": HandMotionCalibration(**c.__dict__)})


def test_teleop_buttons_and_disconnect():
    m = TeleopMapper(TeleopSettings(wrist_source="off"), _cal())
    f = _frame(_hand("left", stick=(0.5, 0.0), buttons={"shoulder": True, "grip": True}),
               HandInput.neutral("right"))
    c = m.commands(f, CONTROL_DT)
    assert c["left"].move == (0.5, 0.0) and c["left"].vz == -1.0 and c["left"].grip
    assert c["right"] == ArmCommand(vz=1.0)        # disconnected: open, rise, stop


def test_teleop_wrist_angles_from_gravity():
    m = TeleopMapper(TeleopSettings(wrist_deadzone_deg=0.0), _cal())
    a = math.radians(20)
    g_fwd = (math.sin(a), 0.0, math.cos(a))        # tilted 20° toward the calibrated forward
    g_right = (0.0, math.sin(a), math.cos(a))
    for _ in range(3):                             # first sample initialises the low-pass
        c = m.commands(_frame(_hand("left", accel=g_fwd), _hand("right", accel=g_right)), CONTROL_DT)
    assert math.degrees(c["left"].pitch) == pytest.approx(20, abs=0.5) and abs(c["left"].twist) < 1e-6
    assert math.degrees(c["right"].twist) == pytest.approx(20, abs=0.5) and abs(c["right"].pitch) < 1e-6


def test_teleop_wrist_deadzone():
    m = TeleopMapper(TeleopSettings(wrist_deadzone_deg=4.0), _cal())
    a = math.radians(3)
    c = m.commands(_frame(_hand("left", accel=(math.sin(a), 0, math.cos(a))), _hand("right")), CONTROL_DT)
    assert c["left"].pitch == 0.0
