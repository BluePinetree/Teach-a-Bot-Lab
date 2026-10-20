import json

from teachabot.visit import STEP_LABEL, VisitFlow, load_reference


def test_visit_flow_goes_demo_train_ai_retrain_result():
    f = VisitFlow("1조", target_demos=2, practice=False)
    assert f.step == "demo" and not f.can_train() and f.next() is None    # nothing to learn yet
    f.on_episode_saved("demo", False)
    assert not f.can_train()                                             # failed demos are not learned
    f.on_episode_saved("demo", True)
    assert f.can_train() and "1/2" in f.hint()
    assert f.next() == "train" and f.step == "train"
    f.on_trained(True)
    assert f.step == "ai" and f.ai_mode and not f.can_train()            # no correction yet
    f.on_episode_saved("takeover", False)
    assert f.can_train() and f.start_training() and f.step == "retrain"
    f.on_trained(True)
    assert f.step == "ai" and f.takeovers == 0 and f.trainings == 2
    assert f.next() == "result" and f.step == "result"
    s = f.summary()
    assert s["demos_ok"] == 1 and s["demos_tried"] == 2 and s["takeovers_total"] == 1
    assert [x["step"] for x in s["steps"]] == ["demo", "train", "ai", "retrain", "ai", "result"]


def test_failed_training_returns_to_previous_step():
    f = VisitFlow("2조", practice=False)
    f.on_episode_saved("demo", True)
    f.start_training()
    f.on_trained(False)
    assert f.step == "demo" and f.trainings == 0


def test_banner_marks_current_step_and_sessions_include_base():
    f = VisitFlow("3조", base_sessions=("s01",), practice=False)
    assert f.sessions() == ["s01", "3조"]
    states = dict(f.banner())
    assert states[STEP_LABEL["demo"]] == "current" and states[STEP_LABEL["result"]] == "todo"
    assert STEP_LABEL["practice"] not in states


def test_reference_lines_need_a_source(tmp_path):
    p = tmp_path / "ref.json"
    p.write_text(json.dumps({"lines": [{"label": "a", "value": "1%", "source": "x.csv"},
                                       {"label": "b", "value": "2%"}]}), encoding="utf-8")
    assert [r["label"] for r in load_reference(p)] == ["a"]
    assert load_reference(tmp_path / "missing.json") == []


def test_live_trainer_learns_only_listed_sessions(tmp_path):
    import numpy as np
    from teachabot.aloha.env import OBS_DIM
    from teachabot.learning.demos import DemoWriter
    from teachabot.learning.live import LiveTrainer
    for session in ("1조", "2조"):
        w = DemoWriter(session, "test", tmp_path)
        w.start()
        w.add(np.zeros(OBS_DIM), np.zeros(12))
        w.finish(0, True, 0.1)
    t = LiveTrainer(tmp_path, sessions=["1조"])
    assert t.data_counts() == (1, 0)
    assert LiveTrainer(tmp_path).data_counts() == (2, 0)


def test_practice_comes_first_is_not_recorded_and_resets_per_student():
    from teachabot.visit import PRACTICE_ITEMS
    f = VisitFlow("4조")
    assert f.step == "practice" and not f.recording and dict(f.banner())[STEP_LABEL["practice"]] == "current"
    for k, _ in PRACTICE_ITEMS[:-1]:
        f.mark_practice(k)
    assert not f.practice_complete and "□" in f.hint()
    f.mark_practice(PRACTICE_ITEMS[-1][0])
    assert f.practice_complete and "준비 완료" in f.hint()
    f.reset_practice()                                   # next student
    assert not f.practice_done and f.practice_rounds == 1
    assert f.next() == "demo" and f.step == "demo" and f.recording
    f.mark_practice("move")                              # ignored outside practice
    assert not f.practice_done


def test_base_session_can_be_limited_to_its_first_demos(tmp_path):
    import numpy as np
    from teachabot.aloha.env import OBS_DIM
    from teachabot.learning.demos import DemoWriter
    from teachabot.learning.live import LiveTrainer, load_session_demos
    w = DemoWriter("s01", "test", tmp_path)
    for i in range(4):
        w.start()
        w.add(np.full(OBS_DIM, i), np.zeros(12))
        w.finish(i, True, 0.1)
    w.start()
    w.add(np.zeros(OBS_DIM), np.zeros(12), (False, True))
    w.finish(9, False, 0.1, in_tray={"left": False, "right": True}, kind="takeover")
    g = DemoWriter("1조", "test", tmp_path)
    g.start()
    g.add(np.zeros(OBS_DIM), np.zeros(12))
    g.finish(0, True, 0.1)
    picked = load_session_demos(tmp_path, ["s01:2", "1조"])
    assert [(d.meta["session"], d.meta["layout_seed"]) for d in picked] == [("s01", 0), ("s01", 1), ("1조", 0)]
    assert LiveTrainer(tmp_path, sessions=["s01:2", "1조"]).data_counts() == (3, 0)
    assert LiveTrainer(tmp_path, sessions=["s01"]).data_counts() == (4, 1)


def test_visit_demo_layouts_are_per_group_and_away_from_other_ranges():
    from teachabot.aloha_app import PRACTICE_SEED_BASE, VISIT_DEMO_SEED_BASE, visit_demo_seed
    a, b = visit_demo_seed("1조"), visit_demo_seed("2조")
    assert a != b and a == visit_demo_seed("1조")
    for s in (a, b):
        assert VISIT_DEMO_SEED_BASE <= s < VISIT_DEMO_SEED_BASE + 1_000_000      # not s01 (0-10), test (1e6),
        assert s > PRACTICE_SEED_BASE + 1_000                                   # tuning (2e6), AI rounds (3e6)
