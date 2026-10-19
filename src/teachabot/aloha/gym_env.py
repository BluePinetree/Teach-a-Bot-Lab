"""Gymnasium view of the S1 task for learning (RL baseline and evaluation).

Same observation (30-d) and the same command interface people use; actions are the
`ArmCommand` fields of both hands normalised to [-1, 1]:
    per hand: vx, vy, vz, pitch / PITCH_LIMIT, twist / TWIST_LIMIT, grip (> 0 = closed)

Rewards (the RL baseline runs both, see DECISIONS D11):
  sparse: +1 the first time each cube is inside its own tray (max 2 per episode)
  dense : hand-designed shaping per hand (reach -> grasp -> lift -> carry -> in tray)
          + 20 on success; the "careful reward design" variant
Episode: success (both cubes in) or a cube off the table ends it; 40 s of simulated
time truncates it.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .env import OBS_DIM, PITCH_LIMIT, TWIST_LIMIT, AlohaSortEnv, ArmCommand
from .scene import CUBE_HALF, HANDS

ACT_DIM = 12
EVAL_SEED_BASE = 1_000_000          # held-out layouts for evaluation; training seeds are below this
REWARDS = ("sparse", "dense")


def action_to_commands(a) -> dict[str, ArmCommand]:
    a = np.clip(np.asarray(a, float), -1.0, 1.0)
    out = {}
    for i, h in enumerate(HANDS):
        v = a[6 * i:6 * i + 6]
        out[h] = ArmCommand(move=(float(v[0]), float(v[1])), vz=float(v[2]), pitch=float(v[3]) * PITCH_LIMIT,
                            twist=float(v[4]) * TWIST_LIMIT, grip=bool(v[5] > 0.0))
    return out


def commands_to_action(cmds: dict[str, ArmCommand]) -> np.ndarray:
    parts = []
    for h in HANDS:
        c = cmds[h]
        parts.append([c.move[0], c.move[1], c.vz, c.pitch / PITCH_LIMIT, c.twist / TWIST_LIMIT,
                      1.0 if c.grip else -1.0])
    return np.clip(np.asarray(parts, float).ravel(), -1.0, 1.0)


def dense_hand_reward(env: AlohaSortEnv, h: str) -> float:
    if env.in_tray(h):
        return 3.5
    site, cube = env.site_pos(h), env.cube_pos(h)
    d_reach = float(np.linalg.norm(site - cube))
    r = 0.5 * (1.0 - np.tanh(10.0 * d_reach))
    grasped = env.targets[h].grip and env.finger_opening(h) > 0.01 and d_reach < 0.04
    if grasped:
        r += 0.5
        if cube[2] > CUBE_HALF + 0.02:
            tray = np.asarray(env.layout.tray_pos[h])
            r += 1.0 + (1.0 - np.tanh(5.0 * float(np.linalg.norm(cube[:2] - tray))))
    return float(r)


class AlohaSortGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward: str = "sparse", action_repeat: int = 5, time_limit: float = 40.0):
        if reward not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}")
        self.reward_kind = reward
        self.action_repeat = action_repeat
        self.sim = AlohaSortEnv(time_limit=time_limit)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self._credited: set[str] = set()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        layout_seed = (options or {}).get("layout_seed")
        if layout_seed is None:
            layout_seed = int(self.np_random.integers(0, EVAL_SEED_BASE))
        obs = self.sim.reset(layout_seed)
        self._credited = set()
        return obs.astype(np.float32), {"layout_seed": layout_seed}

    def step(self, action):
        cmds = action_to_commands(action)
        for _ in range(self.action_repeat):
            obs, info = self.sim.step(cmds)
            if info.done:
                break
        reward = 0.0
        if self.reward_kind == "sparse":
            for h in HANDS:
                if info.in_tray[h] and h not in self._credited:
                    self._credited.add(h)
                    reward += 1.0
        else:
            reward = 0.1 * sum(dense_hand_reward(self.sim, h) for h in HANDS) + (20.0 if info.success else 0.0)
        fell = info.failed and self.sim.t < self.sim.time_limit
        terminated = bool(info.success or fell)
        truncated = bool(not terminated and info.failed)
        return obs.astype(np.float32), float(reward), terminated, truncated, {
            "success": bool(info.success), "cubes_in_tray": int(sum(info.in_tray.values())),
            "sim_time": self.sim.t}
