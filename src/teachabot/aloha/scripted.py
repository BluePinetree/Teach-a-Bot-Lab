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
