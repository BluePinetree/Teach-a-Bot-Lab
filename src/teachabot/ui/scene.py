"""Ursina entities mirroring a simulation.World. Built from primitives only (no asset files)."""
from __future__ import annotations

from ursina import AmbientLight, DirectionalLight, Entity, Vec3, camera, color
from ursina.shaders import lit_with_shadows_shader, unlit_shader

from teachabot.simulation.environment import World

HAND_COLOR = {"left": color.rgb(0.25, 0.55, 1.0), "right": color.rgb(1.0, 0.55, 0.15)}
CUBE_COLOR = {"blue": color.rgb(0.15, 0.4, 0.95), "orange": color.rgb(1.0, 0.5, 0.05)}
TABLE_COLOR = color.rgb(0.72, 0.62, 0.48)


class GripperView:
    """Body block + two fingers that slide together when closed, and a floor marker."""

    OPEN_GAP, CLOSED_GAP = 0.085, 0.064

    def __init__(self, hand: str, world: World):
        c = world.config
        col = HAND_COLOR[hand]
        self.root = Entity()
        Entity(parent=self.root, model="cube", color=color.rgb(0.35, 0.35, 0.38),
               scale=(0.025, 0.08, 0.025), y=0.07)                     # short wrist stub
        Entity(parent=self.root, model="cube", color=col, scale=(0.12, 0.03, 0.05), y=0.03)
        self.fingers = [Entity(parent=self.root, model="cube", color=col.tint(-0.2),
                               scale=(0.012, 0.07, 0.04), y=-0.02) for _ in range(2)]
        # floor marker = the gripper's grasp area, plus a thin guide line down to it,
        # so students can see where above the table the gripper is
        self.marker = Entity(model="circle", color=col, alpha=0.45, shader=unlit_shader,
                             rotation_x=90, scale=c.grasp_radius * 2, y=0.002)
        guide_h = c.work_height - 0.055
        self.guide = Entity(model="cube", color=col, alpha=0.5, shader=unlit_shader,
                            scale=(0.004, guide_h, 0.004))
        self.guide_h = guide_h
        self.hand = hand
        self.height = c.work_height

    def sync(self, g) -> None:
        self.root.position = Vec3(g.x, self.height, g.z)
        half = (self.CLOSED_GAP if g.closed else self.OPEN_GAP) / 2
        self.fingers[0].x, self.fingers[1].x = -half, half
        self.marker.position = Vec3(g.x, 0.002, g.z)
        self.guide.position = Vec3(g.x, self.guide_h / 2, g.z)


class TabletopView:
    def __init__(self, world: World):
        c = world.config
        # Lit shading so cube faces read as 3-D; flat markers below use unlit_shader.
        Entity.default_shader = lit_with_shadows_shader
        Entity(model="cube", color=TABLE_COLOR, scale=(c.table_half_x * 2 + 0.1, 0.04, c.table_half_z * 2 + 0.1),
               y=-0.02)
        for sx in (-1, 1):
            for sz in (-1, 1):
                Entity(model="cube", color=TABLE_COLOR.tint(-0.25), scale=(0.04, 0.7, 0.04),
                       position=(sx * c.table_half_x, -0.39, sz * c.table_half_z))
        Entity(model="cube", color=color.rgb(0.55, 0.57, 0.6), scale=(6, 0.01, 6), y=-0.75)   # floor
        self.targets = {}
        for t in world.targets.values():
            col = CUBE_COLOR[t.color]
            Entity(model="cube", color=col.tint(0.55), scale=(t.half_x * 2, 0.004, t.half_z * 2),
                   position=(t.x, 0.002, t.z))
            wall = 0.012
            for dx, dz, sx, sz in ((-t.half_x, 0, wall, t.half_z * 2), (t.half_x, 0, wall, t.half_z * 2),
                                   (0, -t.half_z, t.half_x * 2, wall), (0, t.half_z, t.half_x * 2, wall)):
                Entity(model="cube", color=col, scale=(sx, 0.03, sz), position=(t.x + dx, 0.015, t.z + dz))
        self.cubes = {cid: Entity(model="cube", color=CUBE_COLOR[cube.color], scale=c.cube_size)
                      for cid, cube in world.cubes.items()}
        self.cube_shadows = {cid: Entity(model="circle", color=color.black, alpha=0.25, rotation_x=90, shader=unlit_shader,
                                         scale=c.cube_size * 1.2, y=0.003) for cid in world.cubes}
        self.grippers = {h: GripperView(h, world) for h in world.grippers}

        sun = DirectionalLight(shadows=False, color=color.rgba(0.75, 0.75, 0.72, 1))
        sun.look_at(Vec3(0.4, -1, 0.6))
        AmbientLight(color=color.rgba(0.42, 0.42, 0.46, 1))
        camera.position = (0, 1.3, -1.45)
        camera.rotation_x = 42
        camera.fov = 42

    def sync(self, world: World) -> None:
        for cid, cube in world.cubes.items():
            e = self.cubes[cid]
            e.position = Vec3(cube.x, cube.y, cube.z)
            self.cube_shadows[cid].position = Vec3(cube.x, 0.003, cube.z)
            self.cube_shadows[cid].enabled = cube.held_by is not None
        for h, g in world.grippers.items():
            self.grippers[h].sync(g)
