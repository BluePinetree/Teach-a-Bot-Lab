"""Scripted pick-and-place controller — TEST ONLY.

Used by automated tests and by the app's --test-autopilot smoke run to exercise the whole
visit flow without people. Its episodes are never shown or used as student or AI results
(the autopilot session code starts with "_" and its files are deleted after the run).
"""
from __future__ import annotations

import numpy as np

from .env import CONTROL_DT, MAX_SPEED, Z_CARRY, Z_MIN, AlohaSortEnv, ArmCommand
from .scene import HANDS


class ScriptedSorter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.phase = {h: 0 for h in HANDS}
        self.timer = {h: 0.0 for h in HANDS}

    @staticmethod
    def _goto(env: AlohaSortEnv, h: str, xy, vz: float, grip: bool) -> ArmCommand:
        d = np.asarray(xy) - env.targets[h].pos[:2]
        dist = float(np.linalg.norm(d))
        v = d / max(dist, 1e-9) * min(1.0, dist / (MAX_SPEED * CONTROL_DT))
        return ArmCommand(move=(float(v[0]), float(v[1])), vz=vz, grip=grip)

    def commands(self, env: AlohaSortEnv) -> dict[str, ArmCommand]:
        out = {}
        for h in HANDS:
            tg, cube, tray = env.targets[h], env.cube_pos(h), env.layout.tray_pos[h]
            p = self.phase[h]
            if p == 0:
                out[h] = self._goto(env, h, cube[:2], +1, False)
                if np.linalg.norm(tg.pos[:2] - cube[:2]) < 0.003:
                    self.phase[h] = 1
            elif p == 1:
                out[h] = self._goto(env, h, cube[:2], -1, False)
                if tg.pos[2] <= Z_MIN + 1e-6:
                    self.phase[h], self.timer[h] = 2, env.t
            elif p == 2:
                out[h] = ArmCommand(vz=-1, grip=True)
                if env.t - self.timer[h] > 0.5:
                    self.phase[h] = 3
            elif p == 3:
                out[h] = self._goto(env, h, tray, +1, True)
                if np.linalg.norm(tg.pos[:2] - np.asarray(tray)) < 0.003 and tg.pos[2] >= Z_CARRY - 1e-6:
                    self.phase[h] = 4
            else:
                out[h] = ArmCommand(vz=0, grip=False)
        return out


class ScriptedTrayCarrier:
    """S2 reference: both arms grip the handles, lift together, carry, lower, release.
    `lag` delays the right arm's lift (seconds); `smooth` ramps the carry speed up over the
    first 5 cm and down over the last 8 cm instead of starting and stopping at full speed."""

    def __init__(self, lag: float = 0.0, smooth: bool = True):
        self.lag, self.smooth = lag, smooth
        self.reset()

    def reset(self) -> None:
        self.phase, self.t_phase, self.offset, self.carry_start = 0, 0.0, None, None

    def commands(self, env) -> dict:
        from .scene import S2_GOAL
        ph = self.phase
        out = {}
        if ph == 0:            # above the handles
            done = True
            for h in HANDS:
                c = ScriptedSorter._goto(env, h, env.handle_pos(h)[:2], +1, False)
                out[h] = c
                done &= bool(np.linalg.norm(env.targets[h].pos[:2] - env.handle_pos(h)[:2]) < 0.003)
            if done:
                self.phase = 1
        elif ph == 1:          # down
            out = {h: ScriptedSorter._goto(env, h, env.handle_pos(h)[:2], -1, False) for h in HANDS}
            if all(env.targets[h].pos[2] <= Z_MIN + 1e-6 for h in HANDS):
                self.phase, self.t_phase = 2, env.t
        elif ph == 2:          # grip
            out = {h: ArmCommand(vz=-1, grip=True) for h in HANDS}
            if env.t - self.t_phase > 0.6:
                self.phase, self.t_phase = 3, env.t
                tray = env.tray_pos()[:2]
                self.offset = {h: env.targets[h].pos[:2] - tray for h in HANDS}
        elif ph == 3:          # lift (right arm optionally late)
            out = {"left": ArmCommand(vz=+1, grip=True),
                   "right": ArmCommand(vz=+1 if env.t - self.t_phase >= self.lag else -1, grip=True)}
            if all(env.targets[h].pos[2] >= Z_CARRY - 1e-6 for h in HANDS):
                self.phase = 4
        elif ph == 4:          # carry: identical move for both arms keeps the tray rigid between them
            goal = np.asarray(S2_GOAL) + self.offset["left"]
            pos = env.targets["left"].pos[:2]
            if self.carry_start is None:
                self.carry_start = pos.copy()
            d = goal - pos
            dist = float(np.linalg.norm(d))
            if self.smooth:
                k = min(1.0, dist / 0.08, 0.15 + float(np.linalg.norm(pos - self.carry_start)) / 0.05)
            else:
                k = min(1.0, dist / (MAX_SPEED * CONTROL_DT))
            v = d / max(dist, 1e-9) * k
            out = {h: ArmCommand(move=(float(v[0]), float(v[1])), vz=+1, grip=True) for h in HANDS}
            if dist < 0.003:
                self.phase = 5
        elif ph == 5:          # lower
            out = {h: ArmCommand(vz=-1, grip=True) for h in HANDS}
            if all(env.targets[h].pos[2] <= Z_MIN + 0.01 for h in HANDS):
                self.phase, self.t_phase = 6, env.t
        else:                  # release
            out = {h: ArmCommand(vz=-1 if env.t - self.t_phase < 0.4 else +1, grip=False) for h in HANDS}
        return out
