import numpy as np
import pytest

from teachabot.aloha.env import AlohaSortEnv, ArmCommand
from teachabot.aloha.scene import S2_GOAL
from teachabot.aloha.scripted import ScriptedTrayCarrier
from teachabot.aloha.tray_env import S2_OBS_DIM, AlohaTrayEnv
from teachabot.learning.demos import DemoWriter, load_demos


@pytest.fixture(scope="module")
def env():
    return AlohaTrayEnv()


def test_s2_reset_is_reproducible_and_starts_unsolved(env):
    a = env.reset(5)
    b = env.reset(5)
    np.testing.assert_allclose(a, b)
    assert a.shape == (S2_OBS_DIM,)
    info = env.info()
    assert env.ball_on_tray() and not info.success and not info.failed and not env.in_goal()


def _carry(env, seeds, **kw):
    ok = 0
    for s in seeds:
        env.reset(s)
        c = ScriptedTrayCarrier(**kw)
        while True:
            _, info = env.step(c.commands(env))
            if info.done or env.t > 20:
                break
        ok += info.success
    return ok


def test_s2_smooth_carry_keeps_the_ball_and_a_jerky_one_spills_it(env):
    """The task property option B is built on (test-only scripted controller, both arms lift together)."""
    assert _carry(env, range(10), smooth=True) >= 9
    assert _carry(env, range(10), smooth=False) <= 2


def test_s2_ball_off_tray_fails_with_reason(env):
    env.reset(0)
    c = env._ball_qadr
    env.data.qpos[c:c + 3] = (0.35, -0.2, 0.03)          # on the table, beside the tray
    for _ in range(5):
        _, info = env.step({})
    assert info.failed and info.reason == "ball_off"


def test_s2_timeout_reason():
    e = AlohaTrayEnv(time_limit=0.1)
    e.reset(0)
    for _ in range(10):
        _, info = e.step({h: ArmCommand() for h in ("left", "right")})
    assert info.failed and info.reason == "timeout"


def test_s2_goal_is_reachable_by_both_hands(env):
    env.reset(0)
    for h, sign in (("left", -1), ("right", 1)):
        x = S2_GOAL[0] + sign * 0.13
        assert (x < -0.10) if h == "left" else (x > 0.10)    # inside each hand's S1 workspace


def test_s1_layouts_unchanged_by_task_hooks():
    e = AlohaSortEnv()
    e.reset(0)
    np.testing.assert_allclose([*e.cube_pos("left"), *e.cube_pos("right")],
                               [-0.177195, -0.106043, 0.021, 0.132149, 0.002654, 0.021], atol=1e-6)
    e.reset(1_000_000)
    np.testing.assert_allclose([*e.cube_pos("left"), *e.cube_pos("right")],
                               [-0.206612, -0.118294, 0.021, 0.219393, -0.016628, 0.021], atol=1e-6)


def test_s2_demos_never_mix_into_s1_training(tmp_path):
    for task in ("s1", "s2"):
        w = DemoWriter("x", "test", tmp_path, task=task)
        w.start()
        w.add(np.zeros(10), np.zeros(12))
        w.finish(0, True, 0.1)
    assert [d.meta["task"] for d in load_demos(tmp_path)] == ["s1"]
    assert [d.meta["task"] for d in load_demos(tmp_path, task="s2")] == ["s2"]
