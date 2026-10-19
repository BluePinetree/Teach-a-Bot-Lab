"""Standard input frame shared by every input source, and the InputProvider interface.

Game and learning code consume only InputFrame. Whether the frame came from
Joy-Cons, the keyboard or a recording must not matter to them.

Conventions (schema version 1):
- timestamp: seconds, host monotonic clock (time.perf_counter) when the frame was polled.
- stick: (x, y) in [-1, 1] after neutral calibration and radial dead zone.
  +x = stick pushed right, +y = stick pushed up (away from the player).
- buttons: fixed key set per hand (LEFT_BUTTONS / RIGHT_BUTTONS), debounced booleans.
  "grip" is ZL (left) / ZR (right) and closes that hand's gripper.
- accel [G] / gyro [dps]: most recent IMU sub-sample in the *unified* frame
  (= Left Joy-Con sensor frame; Right is mapped into it by a measured axis map),
  gyro with stationary bias removed. accel_sensor / gyro_sensor keep the
  pre-unification values (device frame, factory calibration only).
  All four are None for sources without an IMU (keyboard).
- A disconnected hand reports connected=False, stick (0, 0), all buttons False.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

SCHEMA_NAME = "teachabot.input"
SCHEMA_VERSION = 1

LEFT_BUTTONS = ("grip", "shoulder", "stick_click", "minus", "capture",
                "up", "down", "left", "right", "sl", "sr")
RIGHT_BUTTONS = ("grip", "shoulder", "stick_click", "plus", "home",
                 "a", "b", "x", "y", "sl", "sr")
BUTTONS = {"left": LEFT_BUTTONS, "right": RIGHT_BUTTONS}
HANDS = ("left", "right")

Vec3 = tuple[float, float, float]


class InputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class HandInput:
    connected: bool
    stick: tuple[float, float]
    buttons: dict[str, bool]
    accel: Vec3 | None = None
    gyro: Vec3 | None = None
    accel_sensor: Vec3 | None = None
    gyro_sensor: Vec3 | None = None

    @property
    def grip(self) -> bool:
        return self.buttons["grip"]

    @classmethod
    def neutral(cls, hand: str, connected: bool = False) -> HandInput:
        return cls(connected=connected, stick=(0.0, 0.0),
                   buttons={b: False for b in BUTTONS[hand]})

    def validate(self, hand: str) -> None:
        if set(self.buttons) != set(BUTTONS[hand]):
            raise InputValidationError(
                f"{hand} buttons must be exactly {sorted(BUTTONS[hand])}, got {sorted(self.buttons)}")
        if not all(isinstance(v, bool) for v in self.buttons.values()):
            raise InputValidationError(f"{hand} button values must be bool")
        if len(self.stick) != 2 or not all(_finite(v) and -1.0 <= v <= 1.0 for v in self.stick):
            raise InputValidationError(f"{hand} stick must be 2 finite values in [-1, 1], got {self.stick}")
        for name in ("accel", "gyro", "accel_sensor", "gyro_sensor"):
            v = getattr(self, name)
            if v is not None and (len(v) != 3 or not all(_finite(c) for c in v)):
                raise InputValidationError(f"{hand}.{name} must be None or 3 finite values, got {v}")
        if not self.connected and (self.stick != (0.0, 0.0) or any(self.buttons.values())):
            raise InputValidationError(f"{hand} is disconnected but reports stick/buttons")

    def to_dict(self) -> dict:
        return {
            "connected": self.connected,
            "stick": [self.stick[0], self.stick[1]],
            "grip": self.grip,
            "buttons": dict(self.buttons),
            "accel": _list_or_none(self.accel),
            "gyro": _list_or_none(self.gyro),
            "accel_sensor": _list_or_none(self.accel_sensor),
            "gyro_sensor": _list_or_none(self.gyro_sensor),
        }

    @classmethod
    def from_dict(cls, d: dict) -> HandInput:
        return cls(
            connected=bool(d["connected"]),
            stick=(float(d["stick"][0]), float(d["stick"][1])),
            buttons={k: bool(v) for k, v in d["buttons"].items()},
            accel=_tuple_or_none(d.get("accel")),
            gyro=_tuple_or_none(d.get("gyro")),
            accel_sensor=_tuple_or_none(d.get("accel_sensor")),
            gyro_sensor=_tuple_or_none(d.get("gyro_sensor")),
        )


@dataclass(frozen=True)
class InputFrame:
    timestamp: float
    seq: int
    source: str
    left: HandInput
    right: HandInput
    meta: dict = field(default_factory=dict)  # source-specific extras (e.g. imu_frame_verified)

    def hand(self, name: str) -> HandInput:
        return self.left if name == "left" else self.right

    def validate(self) -> InputFrame:
        if not _finite(self.timestamp):
            raise InputValidationError(f"timestamp must be finite, got {self.timestamp}")
        if not isinstance(self.seq, int) or self.seq < 0:
            raise InputValidationError(f"seq must be a non-negative int, got {self.seq}")
        self.left.validate("left")
        self.right.validate("right")
        return self

    def with_source(self, source: str) -> InputFrame:
        return replace(self, source=source)

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "timestamp": self.timestamp, "seq": self.seq,
                "source": self.source, "left": self.left.to_dict(), "right": self.right.to_dict(),
                "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: dict) -> InputFrame:
        if d.get("schema_version") != SCHEMA_VERSION:
            raise InputValidationError(f"unsupported schema_version {d.get('schema_version')!r}")
        return cls(timestamp=float(d["timestamp"]), seq=int(d["seq"]), source=str(d["source"]),
                   left=HandInput.from_dict(d["left"]), right=HandInput.from_dict(d["right"]),
                   meta=dict(d.get("meta", {}))).validate()


class InputProvider(ABC):
    """Source of InputFrames. poll() never blocks; it returns the latest state."""

    source: str = "?"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def poll(self) -> InputFrame: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    def status_line(self) -> str | None:
        """Optional one-line notice for the game HUD (e.g. a controller about to sleep)."""
        return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _list_or_none(v):
    return None if v is None else [float(c) for c in v]


def _tuple_or_none(v):
    return None if v is None else tuple(float(c) for c in v)
