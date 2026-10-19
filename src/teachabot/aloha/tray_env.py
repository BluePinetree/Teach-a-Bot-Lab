"""S2 "양팔 트레이 들기": carry one tray with both arms to the goal mat without spilling the cube.

Each arm grips the handle at its end of the tray (fingers close across the 2.5 cm handle).
Same command interface as S1 (stick move, hold L/R to lower, ZL/ZR grip, tilt -> wrist).
Success: the tray rests (lowered, upright) with its centre inside the goal area and the cube
still on it. Fail: the cube leaves the tray, the tray falls off the table, or 40 s pass.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .env import AlohaSortEnv, StepInfo
from .scene import (CUBE_HALF, HANDS, S2_GOAL, S2_GOAL_HALF, S2_PLATE_HALF, S2_TRAY_SPAWN, build_model)

S2_OBS_DIM = 39


class AlohaTrayEnv(AlohaSortEnv):
    task = "s2"
    obs_dim = S2_OBS_DIM

    def _build_model(self) -> mujoco.MjModel:
        return build_model(self.layout, task="s2")

    def _init_objects(self) -> None:
        m = self.model
        self._tray_qadr = m.joint("s2_tray_joint").qposadr[0]
        self._tcube_qadr = m.joint("s2_cube_joint").qposadr[0]
        self._tray_body = m.body("s2_tray").id

    def _place_objects(self) -> None:
        d = self.data
        x, y = self.rng.uniform(*S2_TRAY_SPAWN[0]), self.rng.uniform(*S2_TRAY_SPAWN[1])
        pz = S2_PLATE_HALF[2]
        a = self._tray_qadr
        d.qpos[a:a + 3] = (x, y, pz + 0.001)
        d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        cx, cy = self.rng.uniform(-0.04, 0.04), self.rng.uniform(-0.02, 0.02)
        yaw = self.rng.uniform(-0.5, 0.5)
        c = self._tcube_qadr
        d.qpos[c:c + 3] = (x + cx, y + cy, 2 * pz + CUBE_HALF + 0.002)
        d.qpos[c + 3:c + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))

    # -- state -----------------------------------------------------------------------------
    def tray_pos(self) -> np.ndarray:
        return self.data.xpos[self._tray_body].copy()

    def tray_rot(self) -> np.ndarray:
        return self.data.xmat[self._tray_body].reshape(3, 3).copy()

    def handle_pos(self, h: str) -> np.ndarray:
        return self.data.site(f"s2_handle_{h}").xpos.copy()

    def cube_pos(self, h: str) -> np.ndarray:
        """Generic "this hand's object" (used by the app's HUD, checks and practice): the handle."""
        return self.handle_pos(h)

    def tray_cube_pos(self) -> np.ndarray:
        c = self._tcube_qadr
        return self.data.qpos[c:c + 3].copy()

    def cube_on_tray(self) -> bool:
        rel = self.tray_rot().T @ (self.tray_cube_pos() - self.tray_pos())
        px, py, pz = S2_PLATE_HALF
        return abs(rel[0]) < px and abs(rel[1]) < py and pz < rel[2] < pz + 4 * CUBE_HALF

    def in_goal(self) -> bool:
        t = self.tray_pos()
        return abs(t[0] - S2_GOAL[0]) < S2_GOAL_HALF[0] and abs(t[1] - S2_GOAL[1]) < S2_GOAL_HALF[1]

    def tray_lifted(self) -> float:
        """Height of the tray above where it rests on the table (m)."""
        return float(self.tray_pos()[2] - S2_PLATE_HALF[2])

    def in_tray(self, h: str) -> bool:
        return self.info().success

    def info(self) -> StepInfo:
        on = self.cube_on_tray()
        upright = self.tray_rot()[2, 2] > 0.95
        placed = self.tray_lifted() < 0.01 and upright
        success = bool(on and placed and self.in_goal())
        fell = self.tray_pos()[2] < -0.05
        reason = None if success else ("cube_off" if not on else "tray_fell" if fell else
                                       "timeout" if self.t >= self.time_limit else None)
        return StepInfo(time=self.t, success=success, failed=(not success) and reason is not None,
                        in_tray={h: success for h in HANDS}, reason=reason)

    def observation(self) -> np.ndarray:
        parts = []
        for h in HANDS:
            tg = self.targets[h]
            parts += [tg.pos, self.site_pos(h), [tg.pitch, tg.twist, self.finger_opening(h), float(tg.grip)]]
        R, t = self.tray_rot(), self.tray_pos()
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        parts += [t, R[:, 2], [np.sin(yaw), np.cos(yaw)]]
        parts += [R.T @ (self.tray_cube_pos() - t), np.asarray(S2_GOAL) - t[:2]]
        for h in HANDS:
            parts.append(self.handle_pos(h) - self.targets[h].pos)
        obs = np.concatenate([np.asarray(p, float).ravel() for p in parts])
        assert obs.shape == (self.obs_dim,)
        return obs
