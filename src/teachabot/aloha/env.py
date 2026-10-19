"""S1 environment on the ALOHA 2 model: physics in MuJoCo, arm control by inverse kinematics.

Both people and the AI drive the arms through the same `ArmCommand` per hand:
  move  (x, y) in [-1, 1]  horizontal velocity of the gripper target, x = right, y = away
  vz    in [-1, 1]         vertical velocity; the target stays between Z_MIN and Z_CARRY
  pitch rad                gripper tilt toward +y (forward) from pointing straight down
  twist rad                gripper rotation about its own approach axis
  grip  bool               close / open the fingers
The environment integrates these into an end-effector target, solves IK with mink and
sends joint position targets to the Menagerie position actuators. Grasping is plain
contact physics (no attachment trick).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np

from .scene import (ARM_JOINTS, CUBE_HALF, CUBE_OF, HANDS, S1_LAYOUT, TRAY_INNER, TRAY_WALL_H,
                    TaskLayout, build_model)

CONTROL_DT = 0.02                 # 50 Hz control
MAX_SPEED = 0.30                  # m/s, horizontal
MAX_VZ = 0.30                     # m/s
Z_MIN, Z_CARRY = 0.032, 0.16      # gripper-site height range (table top = 0)
PITCH_LIMIT = np.deg2rad(45)
TWIST_LIMIT = np.deg2rad(90)
GRIP_OPEN, GRIP_CLOSED = 0.037, 0.002
# S1 keeps the hands apart: the two gripper bases collide when the sites are < ~0.2 m apart.
WORKSPACE = {"left": ((-0.40, -0.10), (-0.24, 0.32)), "right": ((0.10, 0.40), (-0.24, 0.32))}
HOME_XY = {"left": (-0.22, -0.06), "right": (0.22, -0.06)}
NEUTRAL_Q = (0.0, -0.96, 1.16, 0.0, -0.3, 0.0)
OBS_DIM = 40
OBS_VERSION = 2                   # v2 adds relative vectors (cube - target, tray - cube) per hand
ACT_DIM_PER_HAND = 6              # vx, vy, vz, pitch, twist, grip


@dataclass
class ArmCommand:
    move: tuple[float, float] = (0.0, 0.0)
    vz: float = 0.0
    pitch: float = 0.0
    twist: float = 0.0
    grip: bool = False

    def to_vector(self) -> np.ndarray:
        return np.array([self.move[0], self.move[1], self.vz, self.pitch, self.twist, float(self.grip)])

    @classmethod
    def from_vector(cls, v) -> "ArmCommand":
        v = np.asarray(v, float)
        return cls(move=(float(v[0]), float(v[1])), vz=float(v[2]), pitch=float(v[3]),
                   twist=float(v[4]), grip=bool(v[5] > 0.5))


@dataclass
class HandTarget:
    pos: np.ndarray
    pitch: float = 0.0
    twist: float = 0.0
    grip: bool = False


@dataclass
class StepInfo:
    time: float
    success: bool
    failed: bool
    in_tray: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.success or self.failed


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# Gripper-site frame pointing straight down. Site axes: x = approach direction,
# y = finger opening direction. In the neutral pose the left site's y is world +y and
# the right site's is world -y; keep that so the wrists do not have to spin.
R_DOWN = {
    "left": np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).astype(float),
    "right": np.column_stack([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]).astype(float),
}


def target_rotation(hand: str, pitch: float, twist: float) -> np.ndarray:
    """Tilt toward +y about world x, then twist about the gripper's own approach axis."""
    return _rot_x(pitch) @ R_DOWN[hand] @ _rot_x(twist)


class AlohaSortEnv:
    def __init__(self, layout: TaskLayout = S1_LAYOUT, time_limit: float = 40.0, ik_iters: int = 2):
        self.layout = layout
        self.time_limit = time_limit
        self.ik_iters = ik_iters
        self.model = build_model(layout)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(CONTROL_DT / self.model.opt.timestep))

        # Robot-only model for IK (no cubes, so the IK state has no free joints).
        self.ik_model = build_model(layout, with_task_objects=False)
        self.config = mink.Configuration(self.ik_model)
        self.tasks = {h: mink.FrameTask(frame_name=f"{h}/gripper", frame_type="site", position_cost=1.0,
                                        orientation_cost=0.5, lm_damping=1.0) for h in HANDS}
        self.posture = mink.PostureTask(self.ik_model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(self.ik_model),
                       mink.VelocityLimit(self.ik_model, {f"{h}/{j}": np.pi for h in HANDS for j in ARM_JOINTS})]

        m, im = self.model, self.ik_model
        self._arm_qadr = {h: np.array([m.joint(f"{h}/{j}").qposadr[0] for j in ARM_JOINTS]) for h in HANDS}
        self._ik_qadr = {h: np.array([im.joint(f"{h}/{j}").qposadr[0] for j in ARM_JOINTS]) for h in HANDS}
        self._arm_act = {h: np.array([m.actuator(f"{h}/{j}").id for j in ARM_JOINTS]) for h in HANDS}
        self._grip_act = {h: m.actuator(f"{h}/gripper").id for h in HANDS}
        self._finger_qadr = {h: m.joint(f"{h}/left_finger").qposadr[0] for h in HANDS}
        self._cube_qadr = {h: m.joint(f"{CUBE_OF[h]}_joint").qposadr[0] for h in HANDS}
        self._arm_bodies = [m.body(f"{h}/base_link").id for h in HANDS]
        self._marker = {h: m.body(f"{h}/marker").mocapid[0] for h in HANDS}
        self.targets: dict[str, HandTarget] = {}
        self.rng = np.random.default_rng(0)
        self.seed: int | None = None
        self.t = 0.0

    # -- episode ---------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.t = 0.0
        for h in HANDS:
            d.qpos[self._arm_qadr[h]] = NEUTRAL_Q
            (xr, yr) = self.layout.cube_spawn[h]
            x, y = self.rng.uniform(*xr), self.rng.uniform(*yr)
            yaw = self.rng.uniform(-self.layout.cube_yaw_range, self.layout.cube_yaw_range)
            a = self._cube_qadr[h]
            d.qpos[a:a + 3] = (x, y, CUBE_HALF + 0.001)
            d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
            self.targets[h] = HandTarget(pos=np.array([*HOME_XY[h], Z_CARRY]))
        # Start the arms already in the pointing-down home pose (converged IK, no transient).
        q = self.config.q.copy()
        for h in HANDS:
            q[self._ik_qadr[h]] = NEUTRAL_Q
        self.config.update(q)
        self.posture.set_target_from_configuration(self.config)
        self._solve_ik(iters=200)
        for h in HANDS:
            d.qpos[self._arm_qadr[h]] = self.config.q[self._ik_qadr[h]]
            d.qpos[self._finger_qadr[h]] = GRIP_OPEN
            d.qpos[m.joint(f"{h}/right_finger").qposadr[0]] = GRIP_OPEN
        self._apply_ctrl()
        mujoco.mj_forward(m, d)
        return self.observation()

    def step(self, cmds: dict[str, ArmCommand]) -> tuple[np.ndarray, StepInfo]:
        for h in HANDS:
            self._integrate(h, cmds.get(h, ArmCommand()))
        self._solve_ik(self.ik_iters)
        self._apply_ctrl()
        for _ in range(self.n_substeps):
            self._compensate_gravity()
            mujoco.mj_step(self.model, self.data)
        self.t += CONTROL_DT
        return self.observation(), self.info()

    # -- control ---------------------------------------------------------------------------
    def _integrate(self, h: str, c: ArmCommand) -> None:
        tg = self.targets[h]
        mx, my = np.clip(c.move, -1.0, 1.0)
        norm = np.hypot(mx, my)
        if norm > 1.0:
            mx, my = mx / norm, my / norm
        (xr, yr) = WORKSPACE[h]
        tg.pos[0] = np.clip(tg.pos[0] + mx * MAX_SPEED * CONTROL_DT, *xr)
        tg.pos[1] = np.clip(tg.pos[1] + my * MAX_SPEED * CONTROL_DT, *yr)
        tg.pos[2] = np.clip(tg.pos[2] + np.clip(c.vz, -1, 1) * MAX_VZ * CONTROL_DT, Z_MIN, Z_CARRY)
        tg.pitch = float(np.clip(c.pitch, -PITCH_LIMIT, PITCH_LIMIT))
        tg.twist = float(np.clip(c.twist, -TWIST_LIMIT, TWIST_LIMIT))
        tg.grip = bool(c.grip)

    def _solve_ik(self, iters: int) -> None:
        for h in HANDS:
            tg = self.targets[h]
            T = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(target_rotation(h, tg.pitch, tg.twist)), tg.pos)
            self.tasks[h].set_target(T)
        tasks = [*self.tasks.values(), self.posture]
        for _ in range(iters):
            vel = mink.solve_ik(self.config, tasks, CONTROL_DT, "daqp", limits=self.limits, damping=1e-5)
            self.config.integrate_inplace(vel, CONTROL_DT)

    def _apply_ctrl(self) -> None:
        d = self.data
        for h in HANDS:
            d.mocap_pos[self._marker[h]] = (*self.targets[h].pos[:2], 0.0015)
            d.ctrl[self._arm_act[h]] = self.config.q[self._ik_qadr[h]]
            d.ctrl[self._grip_act[h]] = GRIP_CLOSED if self.targets[h].grip else GRIP_OPEN

    def _compensate_gravity(self) -> None:
        m, d = self.model, self.data
        d.qfrc_applied[:] = 0.0
        jac = np.empty((3, m.nv))
        for b in self._arm_bodies:
            mujoco.mj_jacSubtreeCom(m, d, jac, b)
            d.qfrc_applied[:] -= m.opt.gravity * m.body_subtreemass[b] @ jac

    # -- state -----------------------------------------------------------------------------
    def site_pos(self, h: str) -> np.ndarray:
        return self.data.site(f"{h}/gripper").xpos.copy()

    def cube_pos(self, h: str) -> np.ndarray:
        a = self._cube_qadr[h]
        return self.data.qpos[a:a + 3].copy()

    def cube_yaw(self, h: str) -> float:
        a = self._cube_qadr[h]
        w, x, y, z = self.data.qpos[a + 3:a + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def finger_opening(self, h: str) -> float:
        return float(self.data.qpos[self._finger_qadr[h]])

    def in_tray(self, h: str) -> bool:
        p = self.cube_pos(h)
        tx, ty = self.layout.tray_pos[h]
        inner = TRAY_INNER / 2
        return abs(p[0] - tx) < inner and abs(p[1] - ty) < inner and p[2] < TRAY_WALL_H

    def info(self) -> StepInfo:
        in_tray = {h: self.in_tray(h) for h in HANDS}
        fell = any(self.cube_pos(h)[2] < -0.05 for h in HANDS)
        success = all(in_tray.values())
        return StepInfo(time=self.t, success=success, failed=(not success) and (fell or self.t >= self.time_limit),
                        in_tray=in_tray)

    def observation(self) -> np.ndarray:
        parts = []
        for h in HANDS:
            tg = self.targets[h]
            parts += [tg.pos, self.site_pos(h), [tg.pitch, tg.twist, self.finger_opening(h), float(tg.grip)]]
        for h in HANDS:
            yaw = self.cube_yaw(h)
            parts += [self.cube_pos(h), [np.sin(4 * yaw), np.cos(4 * yaw)]]  # cube looks the same every 90°
        for h in HANDS:   # relative vectors: easy for a small network to act on with little data
            cube = self.cube_pos(h)
            parts += [cube - self.targets[h].pos, np.asarray(self.layout.tray_pos[h]) - cube[:2]]
        obs = np.concatenate([np.asarray(p, float).ravel() for p in parts])
        assert obs.shape == (OBS_DIM,)
        return obs
