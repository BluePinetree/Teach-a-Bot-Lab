"""Tabletop world: two grippers moving on a fixed-height plane, cubes, target boxes.

Pure Python, no rendering, so the same world runs inside the Ursina app, in
tests, and later headless for AI evaluation.

Coordinates (match Ursina): x = right, y = up, z = away from the player.
The table top is y = 0. Units are metres.

This is an educational abstraction, NOT robot physics (요구사항 Phase 3):
- grippers move on the X/Z plane at a fixed working height;
- grasp rule: when grip goes from open to closed while the gripper centre is
  within `grasp_radius` (horizontal) of a free cube, the cube attaches and is
  lifted with the gripper; opening the grip drops it on the table below;
- closing on empty space and then moving over a cube does not pick it up;
- no collisions between grippers or cubes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

HANDS = ("left", "right")


@dataclass(frozen=True)
class WorldConfig:
    table_half_x: float = 0.6      # table spans x in [-0.6, 0.6]
    table_half_z: float = 0.4      # z in [-0.4, 0.4]
    hand_speed: float = 0.5        # m/s at full stick
    work_height: float = 0.14      # gripper body height above the table
    carry_height: float = 0.12     # held cube centre height (between the fingers)
    cube_size: float = 0.06
    grasp_radius: float = 0.05
    center_overlap: float = 0.15   # each hand may cross the centre line by this much
    max_dt: float = 0.1            # clamp so a stall cannot teleport a hand


@dataclass
class Gripper:
    hand: str
    x: float
    z: float
    closed: bool = False
    holding: str | None = None     # cube id


@dataclass
class Cube:
    id: str
    color: str                     # "blue" | "orange"
    x: float
    z: float
    held_by: str | None = None
    y: float = 0.03                # centre height: resting on the table, or carry height when held


@dataclass
class TargetBox:
    id: str
    color: str
    x: float
    z: float
    half_x: float = 0.09
    half_z: float = 0.09

    def contains(self, x: float, z: float) -> bool:
        return abs(x - self.x) <= self.half_x and abs(z - self.z) <= self.half_z


@dataclass(frozen=True)
class HandAction:
    """What one gripper is asked to do this step: velocity in [-1, 1] and grip."""
    vx: float = 0.0
    vz: float = 0.0
    grip: bool = False


@dataclass
class World:
    config: WorldConfig = field(default_factory=WorldConfig)
    grippers: dict[str, Gripper] = field(default_factory=dict)
    cubes: dict[str, Cube] = field(default_factory=dict)
    targets: dict[str, TargetBox] = field(default_factory=dict)
    time: float = 0.0
    events: list[dict] = field(default_factory=list)   # grasp / release log, for HUD and tests

    @classmethod
    def default_scene(cls, config: WorldConfig | None = None) -> World:
        """Phase 3 layout: blue cube + box on the left, orange on the right."""
        c = config or WorldConfig()
        w = cls(config=c)
        w.grippers = {"left": Gripper("left", -0.3, -0.25), "right": Gripper("right", 0.3, -0.25)}
        w.cubes = {"blue": Cube("blue", "blue", -0.2, 0.05), "orange": Cube("orange", "orange", 0.2, 0.05)}
        w.targets = {"left_box": TargetBox("left_box", "blue", -0.45, 0.25),
                     "right_box": TargetBox("right_box", "orange", 0.45, 0.25)}
        for cube in w.cubes.values():
            cube.y = c.cube_size / 2
        return w

    # -- stepping --------------------------------------------------------------

    def step(self, actions: dict[str, HandAction], dt: float) -> None:
        dt = min(max(dt, 0.0), self.config.max_dt)
        self.time += dt
        for hand in HANDS:
            self._step_hand(self.grippers[hand], actions.get(hand, HandAction()), dt)

    def _step_hand(self, g: Gripper, a: HandAction, dt: float) -> None:
        c = self.config
        vx, vz = _clip_unit(a.vx, a.vz)
        g.x += vx * c.hand_speed * dt
        g.z += vz * c.hand_speed * dt
        lo_x, hi_x = self.hand_x_limits(g.hand)
        g.x = min(max(g.x, lo_x), hi_x)
        g.z = min(max(g.z, -c.table_half_z), c.table_half_z)

        was_closed, g.closed = g.closed, bool(a.grip)
        if g.closed and not was_closed and g.holding is None:
            cube = self.nearest_free_cube(g.x, g.z)
            if cube is not None:
                g.holding, cube.held_by = cube.id, g.hand
                self.events.append({"t": self.time, "event": "grasp", "hand": g.hand, "cube": cube.id})
        elif not g.closed and was_closed and g.holding is not None:
            cube = self.cubes[g.holding]
            cube.held_by, g.holding = None, None
            cube.y = c.cube_size / 2
            self.events.append({"t": self.time, "event": "release", "hand": g.hand, "cube": cube.id,
                                "x": cube.x, "z": cube.z})

        if g.holding is not None:
            cube = self.cubes[g.holding]
            cube.x, cube.z, cube.y = g.x, g.z, c.carry_height

    # -- queries ---------------------------------------------------------------

    def hand_x_limits(self, hand: str) -> tuple[float, float]:
        c = self.config
        if hand == "left":
            return -c.table_half_x, c.center_overlap
        return -c.center_overlap, c.table_half_x

    def nearest_free_cube(self, x: float, z: float) -> Cube | None:
        best, best_d = None, self.config.grasp_radius
        for cube in self.cubes.values():
            if cube.held_by is not None:
                continue
            d = math.hypot(cube.x - x, cube.z - z)
            if d <= best_d:
                best, best_d = cube, d
        return best

    def cube_in_target(self, cube_id: str) -> str | None:
        cube = self.cubes[cube_id]
        if cube.held_by is not None:
            return None
        for t in self.targets.values():
            if t.contains(cube.x, cube.z):
                return t.id
        return None


def _clip_unit(x: float, z: float) -> tuple[float, float]:
    r = math.hypot(x, z)
    return (x / r, z / r) if r > 1.0 else (x, z)


def actions_from_input(frame) -> dict[str, HandAction]:
    """Teleoperation mapping (요구사항 Phase 3 조작표): stick X/Y -> gripper X/Z, ZL/ZR -> grip.

    A disconnected hand already reports a neutral stick and no buttons, so it stops.
    """
    out = {}
    for hand in HANDS:
        h = frame.hand(hand)
        out[hand] = HandAction(vx=h.stick[0], vz=h.stick[1], grip=h.grip)
    return out
