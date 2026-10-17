"""Phase 2 completion: the same consumer gives the same result whether driven
live (here: scripted keyboard) or by replaying the recording of that session."""
from teachabot.input.keyboard import VK, KeyboardInputProvider
from teachabot.input.reference_consumer import run_consumer
from teachabot.input.replay import InputRecorder, ReplayInputProvider, load_recording


def scripted_keyboard_frames(n=200):
    held_by_step = lambda i: {"W", "Q"} if i < 50 else {"D", "UP"} if i < 120 else {"S", "RIGHT", "RCTRL"}
    step = {"i": 0}
    p = KeyboardInputProvider(key_reader=lambda vk: vk in {VK[k] for k in held_by_step(step["i"])},
                              clock=lambda: step["i"] / 60.0)
    p.connect()
    frames = []
    for i in range(n):
        step["i"] = i
        frames.append(p.poll())
    return frames


def test_record_then_replay_is_identical(tmp_path):
    live = scripted_keyboard_frames()
    path = tmp_path / "rec.jsonl"
    with InputRecorder(path, source="keyboard") as rec:
        for f in live:
            rec.record(f)
    header, loaded = load_recording(path)
    assert header["source"] == "keyboard" and len(loaded) == len(live)

    replay = ReplayInputProvider(path)
    replay.connect()
    replayed = [replay.poll() for _ in live]
    assert all(f.source == "replay" for f in replayed)
    assert run_consumer(replayed) == run_consumer(live)

    end = replay.poll()  # past the end: safe neutral stop
    assert replay.finished and not end.left.connected and end.left.stick == (0.0, 0.0)


def test_consumer_actually_moved():
    poses = run_consumer(scripted_keyboard_frames())
    assert poses[-1]["left"][0] > 0.5 and poses[-1]["right"][2] is True
