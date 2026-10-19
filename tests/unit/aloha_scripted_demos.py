"""Record scripted pick-and-place episodes as demos (pipeline tests only, never shown as student data)."""
import numpy as np
from aloha_scripted import run_scripted
from teachabot.aloha.gym_env import commands_to_action
from teachabot.aloha.scene import HANDS
from teachabot.aloha.env import ArmCommand
from teachabot.learning.demos import DemoWriter


def record_scripted(env, seeds, out_dir):
    w = DemoWriter("scripted", "scripted_test", out_dir)
    orig = env.step
    def step(cmds):
        full = {h: cmds.get(h, ArmCommand()) for h in HANDS}
        w.add(env.observation(), commands_to_action(full))
        return orig(full)
    env.step = step
    try:
        for s in seeds:
            w.start()
            info, _ = run_scripted(env, s)
            w.finish(s, info.success, info.time)
    finally:
        env.step = orig
    return w
