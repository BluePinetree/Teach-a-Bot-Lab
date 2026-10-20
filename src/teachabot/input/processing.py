"""Pure signal-processing steps applied to raw controller data. No hardware access here."""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def radial_deadzone(x: float, y: float, deadzone: float) -> tuple[float, float]:
    """Zero inside a circle of radius `deadzone`, rescale the rest so the edge still reaches 1.

    Radial (not per-axis) so diagonal motion keeps its direction.
    """
    r = math.hypot(x, y)
    if r <= deadzone:
        return 0.0, 0.0
    scale = min(1.0, (r - deadzone) / (1.0 - deadzone)) / r
    return x * scale, y * scale


class StickCalibrator:
    """Neutral-point calibration and normalization of a raw 12-bit stick to [-1, 1].

    pyjoycon ignores the factory stick calibration, and Phase 1 measured neutral
    points far from 2048 (L ~ (1940, 2348), R ~ (2122, 1877)). The neutral point
    is the median of the first `settle_samples` reports, accepted only if they
    stay within `max_spread` (i.e. nobody was touching the stick); otherwise
    collection restarts. Travel per direction starts at `initial_half_range`
    (below the measured ~1040-1440) and grows to the largest deflection seen,
    capped at `max_half_range` so a glitch cannot shrink the output for good.
    """

    def __init__(self, initial_half_range: float = 1000.0, max_half_range: float = 1700.0,
                 settle_samples: int = 20, max_spread: int = 60):
        self.initial_half_range = initial_half_range
        self.max_half_range = max_half_range
        self.settle_samples = settle_samples
        self.max_spread = max_spread
        self.center: tuple[float, float] | None = None
        self._pending: list[tuple[int, int]] = []
        # travel for -x, +x, -y, +y
        self.half_range = [initial_half_range] * 4

    @property
    def calibrated(self) -> bool:
        return self.center is not None

    def feed(self, raw: tuple[int, int]) -> None:
        if self.center is not None:
            return
        self._pending.append(raw)
        if len(self._pending) < self.settle_samples:
            return
        xs = [p[0] for p in self._pending]
        ys = [p[1] for p in self._pending]
        if max(xs) - min(xs) <= self.max_spread and max(ys) - min(ys) <= self.max_spread:
            self.center = (float(np.median(xs)), float(np.median(ys)))
        self._pending.clear()

    def normalize(self, raw: tuple[int, int]) -> tuple[float, float]:
        if self.center is None:
            return 0.0, 0.0
        out = []
        for axis, value in enumerate(raw):
            d = value - self.center[axis]
            idx = axis * 2 + (1 if d > 0 else 0)
            if abs(d) > self.half_range[idx]:
                self.half_range[idx] = min(abs(d), self.max_half_range)
            out.append(max(-1.0, min(1.0, d / self.half_range[idx])))
        return out[0], out[1]


class Debouncer:
    """A new button state is accepted only after it has held for `hold_s`.

    With Joy-Con reports every ~15 ms and hold_s = 0.012 this means two
    consecutive reports must agree (adds ~15 ms latency, drops 1-report glitches).
    """

    def __init__(self, hold_s: float = 0.012):
        self.hold_s = hold_s
        self.state = False
        self._candidate: bool | None = None
        self._since = 0.0

    def update(self, raw: bool, t: float) -> bool:
        if raw == self.state:
            self._candidate = None
            return self.state
        if self._candidate != raw:
            self._candidate, self._since = raw, t
        if t - self._since >= self.hold_s:
            self.state, self._candidate = raw, None
        return self.state

    def reset(self) -> None:
        self.state, self._candidate = False, None


class GyroBiasEstimator:
    """Re-estimates gyro zero-rate offset whenever the controller is held still.

    Still = over the last `window` sub-samples, per-axis gyro std < `max_std_dps`
    and |mean| < `max_bias_dps`. The bias is then that mean.
    """

    def __init__(self, window: int = 60, max_std_dps: float = 1.5, max_bias_dps: float = 5.0):
        self.window = window
        self.max_std_dps = max_std_dps
        self.max_bias_dps = max_bias_dps
        self.bias = (0.0, 0.0, 0.0)
        self.estimated = False
        self._buf: deque = deque(maxlen=window)

    def update(self, gyro: tuple[float, float, float]) -> None:
        self._buf.append(gyro)
        if len(self._buf) < self.window:
            return
        a = np.asarray(self._buf)
        mean, std = a.mean(axis=0), a.std(axis=0)
        if np.all(std < self.max_std_dps) and np.all(np.abs(mean) < self.max_bias_dps):
            self.bias = tuple(float(v) for v in mean)
            self.estimated = True

    def apply(self, gyro):
        return tuple(g - b for g, b in zip(gyro, self.bias))


@dataclass(frozen=True)
class AxisMap:
    """Per-axis sign flip that maps one Joy-Con's sensor frame onto the unified frame.

    Both IMU chips are right-handed, so a valid map flips an even number of axes
    (det = +1); the same map applies to accel and gyro.
    """

    signs: tuple[int, int, int] = (1, 1, 1)
    verified: bool = False
    detail: dict | None = None

    def apply(self, v):
        return tuple(s * c for s, c in zip(self.signs, v))

    def to_json(self) -> dict:
        return {"signs": list(self.signs), "verified": self.verified, "detail": self.detail}

    @classmethod
    def from_json(cls, d: dict) -> AxisMap:
        return cls(signs=tuple(int(s) for s in d["signs"]), verified=bool(d["verified"]),
                   detail=d.get("detail"))


def load_axis_map(path: Path) -> AxisMap:
    if path.exists():
        return AxisMap.from_json(json.loads(path.read_text(encoding="utf-8")))
    return AxisMap()


def save_axis_map(path: Path, axis_map: AxisMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(axis_map.to_json(), indent=2), encoding="utf-8")


def _kabsch(R: np.ndarray, L: np.ndarray, proper: bool = True) -> np.ndarray:
    """Orthogonal M minimizing sum |M r_i - l_i|^2 (rotation if proper, else may reflect)."""
    U, _, Vt = np.linalg.svd(R.T @ L)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) if proper else 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def estimate_axis_map(left_accel, right_accel, max_rms_g: float = 0.08,
                      min_tilt_std_g: float = 0.1, max_offdiag: float = 0.3) -> AxisMap:
    """Find signs s so that s * right_accel tracks left_accel, from a period where
    both Joy-Cons were held rigidly together in the same orientation and tilted.

    Fits the best 3-D rotation between the two gravity-vector sequences
    (Kabsch / orthogonal Procrustes), so correlated axes (tilting mostly in one
    plane couples x and z) do not confuse it, then requires that rotation to be
    a pure per-axis sign flip. Both IMU chips are right-handed, so only proper
    rotations (det = +1, even number of flips) are allowed; tilting in a single
    plane is enough because the third axis follows from right-handedness.

    verified=False (reason in detail) when: too little tilt, fit residual too
    large (the two were not moving together), the rotation is not a sign flip,
    or the data fits a reflection (odd flips: impossible for these sensors).
    """
    L = np.asarray(left_accel, dtype=float)
    R = np.asarray(right_accel, dtype=float)
    detail: dict = {"samples": int(len(L))}
    if len(L) < 30 or L.shape != R.shape:
        detail["reason"] = "not enough paired samples"
        return AxisMap(detail=detail)
    stds = L.std(axis=0)
    detail["left_axis_std_g"] = [round(float(s), 3) for s in stds]
    if stds.max() < min_tilt_std_g:
        detail["reason"] = "tilt in more directions / further (too little motion)"
        return AxisMap(detail=detail)
    M = _kabsch(R, L)
    rms = float(np.sqrt(np.mean(np.sum((R @ M.T - L) ** 2, axis=1))))
    detail["rotation"] = [[round(float(v), 3) for v in row] for row in M]
    detail["rms_g"] = round(rms, 4)
    if rms > max_rms_g:
        Mr = _kabsch(R, L, proper=False)
        rms_r = float(np.sqrt(np.mean(np.sum((R @ Mr.T - L) ** 2, axis=1))))
        if np.linalg.det(Mr) < 0 and rms_r <= max_rms_g:
            detail["reason"] = "data fits a reflection (odd number of flips), impossible for right-handed sensors"
        else:
            detail["reason"] = f"fit residual {rms:.3f} G > {max_rms_g} G (were they held together?)"
        return AxisMap(detail=detail)
    off = np.abs(M - np.diag(np.diag(M))).max()
    if off > max_offdiag or np.any(np.abs(np.diag(M)) < 1 - max_offdiag):
        detail["reason"] = "rotation is not a per-axis sign flip (held in different orientations?)"
        return AxisMap(detail=detail)
    signs = tuple(int(np.sign(M[i, i])) for i in range(3))
    return AxisMap(signs=signs, verified=True, detail=detail)


class VerticalMotionEstimator:
    """How fast one Joy-Con is moving up or down right now (m/s), from its 200 Hz IMU.

    Causal zero-velocity updates: while the hand is still (vertical specific force near its
    resting value and little rotation for `rest_s`), the speed is 0 and the resting value and
    gravity direction are re-learned; while it moves, the resting value is frozen and the vertical
    acceleration is integrated once. Integrating twice to a position drifts (59.5 cm in 10 s,
    D8); one integration that restarts at every stop does not accumulate, and the person closes
    the loop by watching the gripper.

    Why vertical is the easy axis: a few degrees of unintended tilt leak gravity into the
    horizontal axes as G sin(tilt) (0.5 m/s^2 at 3 deg, the plan-A problem) but into the vertical
    one only as G (1 - cos(tilt)) (0.013 m/s^2 at 3 deg). The resting value absorbs the sensor's
    scale error (the right Joy-Con reads 1.09 G at rest).
    """

    G = 9.81

    def __init__(self, dt: float = 0.005, rest_s: float = 0.2, rest_force: float = 0.2,
                 rest_gyro_dps: float = 8.0, gravity_tau: float = 0.5, deadband: float = 0.02,
                 max_move_s: float = 2.0, refractory_s: float = 0.4, warmup_s: float = 1.0,
                 max_speed: float = 2.0):
        """Rest thresholds from the stroke logs' still windows: vertical force noise sigma
        ~0.07 m/s^2 (0.2 = ~3 sigma), gyro |w| p95 2-3 dps (8 dps margin). The IMU's first ~1 s after connecting is a
        transient (it read 14.9 m/s in a replay), so nothing is output until `warmup_s` has passed
        and the hand has been still once; a speed above `max_speed` (no hand stroke is that fast)
        means the reference is wrong and the estimator settles again."""
        self.dt, self.deadband = dt, deadband
        self.rest_n = max(1, int(round(rest_s / dt)))
        self.rest_force, self.rest_gyro = rest_force, rest_gyro_dps
        self._k = dt / gravity_tau
        self.max_move_n = int(round(max_move_s / dt))      # a person's stroke is shorter: longer = bad reference
        self.refractory_n = int(round(refractory_s / dt))  # just after a stroke, ignore the opposite direction
        self.warmup_n, self.max_speed = int(round(warmup_s / dt)), max_speed
        self._k_settle = min(1.0, dt / 0.05)
        self.reset()

    def reset(self) -> None:
        self.g_hat: np.ndarray | None = None
        self.base = 0.0
        self.velocity = 0.0
        self.still_count = 0
        self.direction = 0           # sign of the current stroke; a stroke does not reverse
        self.moving_count = 0
        self.last_direction = 0
        self.since_stroke = 10 ** 9
        self.settled = False         # False: warming up / re-learning the reference, output 0
        self.settle_count = 0

    @property
    def moving(self) -> bool:
        return self.still_count < self.rest_n

    def update(self, accel_g, gyro_dps) -> float:
        a = np.asarray(accel_g, float) * self.G
        n = float(np.linalg.norm(a))
        if self.g_hat is None:
            if n < 1e-6:
                return 0.0
            self.g_hat, self.base = a / n, n
            return 0.0
        w_dps = np.asarray(gyro_dps, float)
        if not self.settled:                      # follow the raw signal closely until it is quiet
            g = (1 - self._k_settle) * self.g_hat + self._k_settle * a / max(n, 1e-6)
            self.g_hat = g / np.linalg.norm(g)
            dev = float(a @ self.g_hat) - self.base
            self.base += self._k_settle * dev
            quiet = abs(dev) < self.rest_force and float(np.linalg.norm(w_dps)) < self.rest_gyro
            self.still_count = self.still_count + 1 if quiet else 0
            self.settle_count += 1
            if self.settle_count >= self.warmup_n and self.still_count >= self.rest_n:
                self.settled, self.moving_count = True, 0
            return 0.0
        g = self.g_hat - np.cross(np.radians(w_dps), self.g_hat) * self.dt       # follow the rotation
        self.g_hat = g / np.linalg.norm(g)
        dev = float(a @ self.g_hat) - self.base
        quiet = abs(dev) < self.rest_force and float(np.linalg.norm(w_dps)) < self.rest_gyro
        self.still_count = self.still_count + 1 if quiet else 0
        self.moving_count = self.moving_count + 1 if self.moving else 0
        self.since_stroke += 1
        if self.moving_count > self.max_move_n:   # "moving" for too long: the reference is wrong, re-learn
            self.g_hat, self.base = a / max(n, 1e-6), n
            self.velocity, self.direction, self.moving_count = 0.0, 0, 0
            self.still_count = self.rest_n
            return 0.0
        if not self.moving:                       # still: speed 0, re-learn gravity and resting value
            if self.direction:
                self.last_direction, self.since_stroke = self.direction, 0
            self.velocity = 0.0
            self.direction = 0
            g = (1 - self._k) * self.g_hat + self._k * a / max(n, 1e-6)
            self.g_hat = g / np.linalg.norm(g)
            self.base += self._k * (float(a @ self.g_hat) - self.base)
        else:
            self.velocity += dev * self.dt
            if self.direction == 0 and abs(self.velocity) > self.deadband:
                d = 1 if self.velocity > 0 else -1
                if d == -self.last_direction and self.since_stroke < self.refractory_n:
                    self.velocity = 0.0           # braking tail of the previous stroke, not a new one
                else:
                    self.direction = d
            if self.direction and self.velocity * self.direction < 0:
                self.velocity = 0.0               # the braking at the end of a stroke is not a new stroke
            if abs(self.velocity) > self.max_speed:   # impossible for a hand: bad reference, settle again
                self.velocity, self.direction = 0.0, 0
                self.settled, self.settle_count, self.still_count = False, self.warmup_n, 0
        return self.output

    @property
    def output(self) -> float:
        v = self.velocity
        return 0.0 if abs(v) <= self.deadband else v - math.copysign(self.deadband, v)
