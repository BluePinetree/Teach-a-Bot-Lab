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
