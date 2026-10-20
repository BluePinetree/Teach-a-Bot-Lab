import types

import numpy as np
import pytest
from PIL import Image, ImageDraw

from teachabot.aloha.env import AlohaSortEnv, ArmCommand
from teachabot.aloha.scene import HANDS
from teachabot.aloha.teleop import TeleopMapper, TeleopSettings
from teachabot.aloha_app import TASK_TEXT, start_panel
from teachabot.aloha_panels import CH, CW, PanelUI, fit, ref_rate, wrap
from teachabot.input.base import HandInput, InputFrame


def _frame(held=()):
    hands = {}
    for h in HANDS:
        b = dict(HandInput.neutral(h).buttons)
        for k in held:
            b[k] = True
        hands[h] = HandInput(connected=True, stick=(0.0, 0.0), buttons=b, accel=(0.0, 0.0, 1.0), gyro=(0.0, 0.0, 0.0))
    return InputFrame(0.0, 0, "joycon", hands["left"], hands["right"])


@pytest.fixture(scope="module")
def scene():
    env = AlohaSortEnv()
    env.reset(0)
    _, info = env.step({h: ArmCommand() for h in HANDS})
    return env, info, np.zeros((720, 1280, 3), np.uint8)


def test_text_helpers_never_exceed_the_width():
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    from teachabot.aloha_panels import _font
    f = _font(20)
    long = "L 잠들기까지 남은 시간 모름 — 아무 버튼이나 한 번 눌러 주세요 · R 잠들기까지 남은 시간 모름"
    assert d.textlength(fit(d, long, f, 200), font=f) <= 200
    lines = wrap(d, long, f, 260, max_lines=3)
    assert len(lines) <= 3 and all(d.textlength(ln, font=f) <= 260 for ln in lines)
    assert ref_rate("성공 22% · 판당 큐브 0.88개") == pytest.approx(0.22) and ref_rate("없음") is None


def test_panels_draw_play_start_result_and_learning(scene):
    env, info, rgb = scene
    ui = PanelUI("s1", TASK_TEXT["s1"])
    mapper = TeleopMapper(TeleopSettings())
    out = ui.draw(rgb, env, info, _frame(("grip", "shoulder")), mapper, "joycon", 1, None, None, 30.0,
                  panel=start_panel("waiting", {"left": True, "right": True}, "joycon"), time_limit=40.0)
    assert out.shape == (CH, CW, 3)
    out = ui.draw(rgb, env, info, _frame(), mapper, "keyboard", 2, "성공!  12.0초", None, 30.0,
                  episodes=[{"episode": 1, "success": True, "time": 12.0}])
    assert out.shape == (CH, CW, 3)
    trainer = types.SimpleNamespace(state="training", message="학습 중", t_started=0.0, live_weights=None,
                                    pending_policy=None, busy=lambda: True,
                                    progress={"it": 50, "iters": 100, "loss": [0.9, 0.5, 0.3], "eval_done": 0,
                                              "eval_total": 40})
    hist = [{"label": "1조 시범 3 (+기본 5)", "success_rate": 0.2, "cubes_mean": 0.9}]
    ref = [{"label": "강화학습 306시간 연습 (시범 없이)", "value": "성공 0% · 판당 큐브 0.03개"}]
    out = ui.draw(rgb, env, info, _frame(), mapper, "joycon", 3, None, None, 30.0, history=hist, trainer=trainer,
                  learn_open=True, reference=ref, state="finished")
    assert out.shape == (CH, CW, 3)
    sheet = ui.draw(rgb, env, info, _frame(), mapper, "joycon", 3, None, None, 30.0, result_sheet=("1조", hist, ref))
    assert sheet.shape == (CH, CW, 3)


def test_pressed_button_lights_its_joycon_part(scene):
    env, info, rgb = scene
    ui = PanelUI("s1", TASK_TEXT["s1"])
    mapper = TeleopMapper(TeleopSettings())
    idle = ui.draw(rgb, env, info, _frame(), mapper, "joycon", 1, None, None, 30.0)
    held = ui.draw(rgb, env, info, _frame(("grip",)), mapper, "joycon", 1, None, None, 30.0)
    cx, top = 80 + 20, 214 - 33                                 # left ZL part (off the callout dot)
    assert tuple(held[top, cx]) == (255, 214, 102) and tuple(idle[top, cx]) != (255, 214, 102)   # yellow
