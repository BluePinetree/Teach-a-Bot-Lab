"""Minimal stand-in for the game's control logic, used to check that every
InputProvider drives the same interface identically (Phase 2 completion test).

Each hand moves on the X/Z plane at `speed` units/s scaled by its stick, inside
[-bounds, bounds]; grip state is tracked. dt comes from frame timestamps,
clamped to `max_dt` so a stall (e.g. reconnect) cannot teleport a hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import HANDS, InputFrame


@dataclass
class HandPose:
    x: float = 0.0
    z: float = 0.0
    grip: bool = False


@dataclass
class ReferenceConsumer:
    speed: float = 1.0
    bounds: float = 1.0
    max_dt: float = 0.1
    poses: dict = field(default_factory=lambda: {h: HandPose() for h in HANDS})
    _last_t: float | None = None

    def step(self, frame: InputFrame) -> dict:
        dt = 0.0 if self._last_t is None else min(max(frame.timestamp - self._last_t, 0.0), self.max_dt)
        self._last_t = frame.timestamp
        for h in HANDS:
            hand, pose = frame.hand(h), self.poses[h]
            pose.x = max(-self.bounds, min(self.bounds, pose.x + hand.stick[0] * self.speed * dt))
            pose.z = max(-self.bounds, min(self.bounds, pose.z + hand.stick[1] * self.speed * dt))
            pose.grip = hand.grip
        return self.snapshot()

    def snapshot(self) -> dict:
        return {h: (round(p.x, 9), round(p.z, 9), p.grip) for h, p in self.poses.items()}


def run_consumer(frames) -> list[dict]:
    c = ReferenceConsumer()
    return [c.step(f) for f in frames]
