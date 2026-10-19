"""Scripted pick-and-place used only by tests to check that the physics/IK can do S1.
Never used as a demo or as AI output."""
import numpy as np
from teachabot.aloha.env import ArmCommand, MAX_SPEED, CONTROL_DT, Z_MIN, Z_CARRY

def goto_cmd(env, h, xy, vz, grip):
    d = np.asarray(xy) - env.targets[h].pos[:2]
    dist = np.linalg.norm(d)
    v = d / max(dist, 1e-9) * min(1.0, dist / (MAX_SPEED * CONTROL_DT))
    return ArmCommand(move=(v[0], v[1]), vz=vz, grip=grip)

def run_scripted(env, seed, hands=("left", "right"), max_t=30.0):
    env.reset(seed)
    phase = {h: 0 for h in hands}; timer = {h: 0.0 for h in hands}
    while env.t < max_t:
        cmds = {}
        for h in hands:
            tg, cube = env.targets[h], env.cube_pos(h)
            tray = env.layout.tray_pos[h]
            p = phase[h]
            if p == 0:   # over the cube
                cmds[h] = goto_cmd(env, h, cube[:2], +1, False)
                if np.linalg.norm(tg.pos[:2] - cube[:2]) < 0.003: phase[h] = 1
            elif p == 1: # descend
                cmds[h] = goto_cmd(env, h, cube[:2], -1, False)
                if tg.pos[2] <= Z_MIN + 1e-6: phase[h] = 2; timer[h] = env.t
            elif p == 2: # close
                cmds[h] = ArmCommand(vz=-1, grip=True)
                if env.t - timer[h] > 0.5: phase[h] = 3
            elif p == 3: # lift and carry
                cmds[h] = goto_cmd(env, h, tray, +1, True)
                if np.linalg.norm(tg.pos[:2] - np.asarray(tray)) < 0.003 and tg.pos[2] >= Z_CARRY - 1e-6: phase[h] = 4; timer[h] = env.t
            else:        # release
                cmds[h] = ArmCommand(vz=0, grip=False)
        obs, info = env.step(cmds)
        if info.done: return info, phase
    return env.info(), phase
