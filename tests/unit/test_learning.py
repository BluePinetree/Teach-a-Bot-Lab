from pathlib import Path

import numpy as np
import pytest

from teachabot.aloha.env import OBS_DIM, AlohaSortEnv, ArmCommand
from teachabot.aloha.gym_env import AlohaSortGymEnv, action_to_commands, commands_to_action, dense_hand_reward
from teachabot.aloha.scene import CUBE_HALF, HANDS, S1_LAYOUT
from teachabot.learning.bc import BCConfig, NumpyPolicy, train_bc
from teachabot.learning.demos import DemoWriter, load_demos, trim_idle_start
from teachabot.learning.evaluate import evaluate_policy

from aloha_scripted_demos import record_scripted


def test_action_mapping_round_trip():
    cmds = {"left": ArmCommand(move=(0.5, -1.0), vz=-1.0, pitch=0.2, twist=-0.4, grip=True),
            "right": ArmCommand(move=(0.0, 0.3), vz=1.0, pitch=0.0, twist=0.0, grip=False)}
    back = action_to_commands(commands_to_action(cmds))
    for h in HANDS:
        assert back[h].grip == cmds[h].grip
        np.testing.assert_allclose([*back[h].move, back[h].vz, back[h].pitch, back[h].twist],
                                   [*cmds[h].move, cmds[h].vz, cmds[h].pitch, cmds[h].twist], atol=1e-9)


def _put_cube_in_tray(env: AlohaSortEnv, h: str):
    a = env._cube_qadr[h]
    env.data.qpos[a:a + 3] = (*S1_LAYOUT.tray_pos[h], CUBE_HALF + 0.005)


def test_sparse_reward_credits_each_cube_once():
    g = AlohaSortGymEnv("sparse")
    g.reset(seed=0)
    _put_cube_in_tray(g.sim, "left")
    rewards = [g.step(np.zeros(12, np.float32))[1] for _ in range(5)]
    assert rewards[0] == 1.0 and sum(rewards) == 1.0


def test_dense_reward_is_highest_in_tray():
    g = AlohaSortGymEnv("dense")
    g.reset(seed=0)
    r0 = dense_hand_reward(g.sim, "left")
    _put_cube_in_tray(g.sim, "left")
    g.step(np.zeros(12, np.float32))
    assert dense_hand_reward(g.sim, "left") == 3.5 > r0


def test_demo_writer_round_trip_and_filters(tmp_path):
    w = DemoWriter("s99", "test", tmp_path)
    for success in (True, False):
        w.start()
        for t in range(5):
            w.add(np.full(OBS_DIM, t, np.float32), np.zeros(12, np.float32))
        w.finish(7, success, 0.1)
    assert w.saved == 2 and w.saved_success == 1
    ok = load_demos(tmp_path)
    assert len(ok) == 1 and ok[0].obs.shape == (5, OBS_DIM) and ok[0].meta["session"] == "s99"
    assert len(load_demos(tmp_path, only_success=False)) == 2
    assert load_demos(tmp_path, sessions=["other"]) == []


def test_trim_idle_start(tmp_path):
    w = DemoWriter("s1", "test", tmp_path)
    w.start()
    idle = commands_to_action({h: ArmCommand(vz=1.0) for h in HANDS})     # released buttons, stick centred
    act = commands_to_action({"left": ArmCommand(move=(1.0, 0.0), vz=1.0), "right": ArmCommand(vz=1.0)})
    for t in range(10):
        w.add(np.full(OBS_DIM, t, np.float32), idle if t < 4 else act)
    w.finish(0, True, 0.2)
    obs, a = trim_idle_start(load_demos(tmp_path)[0])
    assert len(obs) == 6 and obs[0, 0] == 4


@pytest.fixture(scope="module")
def scripted_demos(tmp_path_factory):
    out = tmp_path_factory.mktemp("demos")
    record_scripted(AlohaSortEnv(), range(200, 203), out)
    return load_demos(out)


def test_bc_trains_and_numpy_policy_round_trips(scripted_demos, tmp_path):
    pol = train_bc(scripted_demos, BCConfig(iters=200))
    obs = scripted_demos[0].obs[0]
    a = pol(obs)
    assert a.shape == (12,) and set(np.unique(a[[5, 11]])) <= {-1.0, 1.0} and np.all(np.abs(a) <= 1.0)
    pol.save(tmp_path / "p.npz")
    np.testing.assert_allclose(NumpyPolicy.load(tmp_path / "p.npz")(obs), a)
    assert pol.meta["n_demos"] == len(scripted_demos)


def test_evaluate_policy_runs_in_parallel(scripted_demos, tmp_path):
    train_bc(scripted_demos, BCConfig(iters=100)).save(tmp_path / "p.npz")
    res = evaluate_policy(tmp_path / "p.npz", episodes=2, workers=2, time_limit=2.0)
    assert len(res.episodes) == 2 and {e["layout_seed"] for e in res.episodes} == {1_000_000, 1_000_001}
    assert 0.0 <= res.success_rate <= 1.0


# -- BC v2 (per hand, mirrored, time-compressed) ------------------------------------------

from teachabot.learning.bc_hand import (HAND_OBS_DIM, HandBCConfig, HandPolicy, hand_act, hand_dataset,  # noqa: E402
                                        hand_obs, train_hand_bc, unmirror_act)


def test_hand_mirroring_is_consistent():
    env = AlohaSortEnv()
    obs = env.reset(3)
    l, r = hand_obs(obs, "left"), hand_obs(obs, "right")
    assert l.shape == r.shape == (HAND_OBS_DIM,)
    np.testing.assert_allclose(l[:6], r[:6], atol=1e-6)       # mirrored home poses coincide
    assert r[8] == pytest.approx(-env.cube_pos("right")[0])   # cube x is mirrored
    a = np.linspace(-1, 1, 12)
    np.testing.assert_allclose(unmirror_act(hand_act(a, "right"), "right"), a[6:])


def test_waiting_frames_are_cut_per_hand(tmp_path):
    env = AlohaSortEnv()
    obs = env.reset(0)
    w = DemoWriter("s1", "test", tmp_path)
    w.start()
    go = commands_to_action({"left": ArmCommand(move=(1.0, 0.0)), "right": ArmCommand(vz=1.0)})
    wait = commands_to_action({h: ArmCommand(vz=1.0) for h in HANDS})
    for t in range(30):                 # 10 active, 10 waiting, 10 active (left); right only waits
        w.add(obs, wait if 10 <= t < 20 else go)
    w.finish(0, True, 0.6)
    X, Y, counts = hand_dataset(load_demos(tmp_path), HandBCConfig())
    assert counts["kept"] == 20          # left: 20 active frames kept, right: all waiting -> cut
    assert counts["dropped_idle"] == 10 + 30


def test_hand_bc_trains_and_loads_as_hand_policy(scripted_demos, tmp_path):
    from teachabot.learning.bc import NumpyPolicy
    pol = train_hand_bc(scripted_demos, HandBCConfig(iters=100))
    pol.save(tmp_path / "h.npz")
    loaded = NumpyPolicy.load(tmp_path / "h.npz")
    assert isinstance(loaded, HandPolicy)
    obs = scripted_demos[0].obs[0]
    np.testing.assert_allclose(loaded(obs), pol(obs))
    assert loaded(obs).shape == (12,)


# -- takeover data and in-session retraining ---------------------------------------------

def test_takeover_uses_only_the_persons_part_of_a_fixed_hand(tmp_path):
    env = AlohaSortEnv()
    obs = env.reset(0)
    w = DemoWriter("t1", "test", tmp_path)
    w.start()
    go = commands_to_action({"left": ArmCommand(move=(1.0, 0.0)), "right": ArmCommand(move=(-1.0, 0.0))})
    for t in range(30):                 # AI drives both for 10 steps, then the person takes the right hand
        w.add(obs, go, (False, t >= 10))
    path = w.finish(0, False, 0.6, in_tray={"left": False, "right": True}, kind="takeover")
    assert path.name.endswith("_take.npz")
    demos = load_demos(tmp_path, kinds=("demo", "takeover"))
    assert len(demos) == 1 and demos[0].kind == "takeover"
    assert load_demos(tmp_path) == []                      # default: full demonstrations only
    X, Y, counts = hand_dataset(demos, HandBCConfig())
    assert counts["kept"] == 20                            # right hand, person's 20 steps; left (AI, not fixed) unused


def test_takeover_without_person_is_not_saved(tmp_path):
    w = DemoWriter("t1", "test", tmp_path)
    w.start()
    w.add(np.zeros(OBS_DIM), np.zeros(12), (False, False))
    assert w.finish(0, True, 0.1, kind="takeover") is None and w.saved == 0


def test_live_trainer_retrains_and_evaluates(scripted_demos, tmp_path):
    import shutil
    from teachabot.learning import live
    for d in scripted_demos:
        shutil.copy(d.path, tmp_path / d.path.name)
    live.POLICY_DIR = tmp_path / "policies"
    t = live.LiveTrainer(tmp_path, episodes=2, workers=2, cfg=HandBCConfig(iters=100))
    assert t.start() and not t.start()                     # one retrain at a time
    t._thread.join(timeout=300)
    assert t.state == "done", t.message
    assert t.result["episodes"] == 2 and t.history[-1] is t.result
    assert (tmp_path / "policies" / Path(t.result["policy"]).name).exists()
    # learning-screen hooks: training progress, a copy of the weights, the test count, the new file
    assert t.progress["it"] == t.progress["iters"] == 100 and len(t.progress["loss"]) >= 2
    assert t.progress["eval_done"] == t.progress["eval_total"] == 2
    assert [w.shape for w in t.live_weights][0][0] == HAND_OBS_DIM
    assert t.pending_policy == Path(t.result["policy"])


def test_compare_view_runs_policies_side_by_side_on_the_same_layout(scripted_demos, tmp_path, monkeypatch):
    import time as _t
    from teachabot import compare_view as cvm
    monkeypatch.setattr(cvm, "COMPARE_TIME_LIMIT", 0.4)
    monkeypatch.setattr(cvm, "HOLD_S", 0.0)
    pol = train_hand_bc(scripted_demos, HandBCConfig(iters=100))
    pol.save(tmp_path / "a.npz")
    pol.save(tmp_path / "b.npz")
    view = cvm.CompareView()
    view.set_runners([{"key": "a", "label": "A", "color": (1, 2, 3), "path": tmp_path / "a.npz", "measured": 0.1},
                      {"key": "b", "label": "B", "color": (1, 2, 3), "path": tmp_path / "b.npz"},
                      {"key": "c", "label": "C", "color": (1, 2, 3), "path": tmp_path / "missing.npz"}])
    rows = view.update()
    assert rows[2][2] == "정책 파일 없음" and rows[2][1] is None                # shows its error ...
    assert rows[0][1].shape == (cvm.PANE_H, cvm.PANE_W, 3)
    for _ in range(200):                                                 # ... and does not hold the others up
        view.update()
        _t.sleep(0.01)
        if view.runners[0].shown:
            break
    assert view.runners[0].shown == view.runners[1].shown >= 1 and view.runners[2].shown == 0
    view.set_runners(view_specs := [{"key": k, "label": k, "color": (1, 2, 3), "path": tmp_path / f"{k}.npz"}
                                    for k in ("a", "b")])
    t0 = _t.perf_counter()
    while view.k < 2 and _t.perf_counter() - t0 < 30:
        rows = view.update()
        _t.sleep(0.01)
    a, b = view.runners
    assert view.k >= 2 and a.shown >= 2 and b.shown == a.shown          # same layouts, finished together
    assert a.results == b.results                                      # same policy, same layout -> same outcome
    assert rows[0][1].shape == (cvm.PANE_H, cvm.PANE_W, 3)
    view.set_runners(view_specs)                                       # unchanged specs keep the tally
    assert a.shown >= 2
