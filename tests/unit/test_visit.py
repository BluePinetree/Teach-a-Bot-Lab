import json

from teachabot.visit import STEP_LABEL, VisitFlow, load_reference


def test_visit_flow_goes_demo_train_ai_retrain_result():
    f = VisitFlow("1조", target_demos=2)
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
    f = VisitFlow("2조")
    f.on_episode_saved("demo", True)
    f.start_training()
    f.on_trained(False)
    assert f.step == "demo" and f.trainings == 0


def test_banner_marks_current_step_and_sessions_include_base():
    f = VisitFlow("3조", base_sessions=("s01",))
    assert f.sessions() == ["s01", "3조"]
    states = dict(f.banner())
    assert states[STEP_LABEL["demo"]] == "current" and states[STEP_LABEL["result"]] == "todo"


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
