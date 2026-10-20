"""Sharp window output on high-DPI screens (2026-10-20: "해상도가 왜 이리 낮지?").

The app's PC runs a 3840x2160 screen at 150 %. A Python process that is not DPI aware gets its
window bitmap stretched 1.5x by Windows, and OpenCV's own WINDOW_NORMAL stretch is crude, so the
1920x1080 canvas looked soft and jagged. Here the process declares itself DPI aware (this
process only, no system setting), the window is sized in physical pixels, the canvas is scaled
by us (area / cubic filter) to the window's real size, and while nothing is drawn over the 3D
view the view itself is rendered again at that size and pasted in, so the robot stays crisp.
"""
from __future__ import annotations

import sys

import cv2
import mujoco
import numpy as np


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)       # per-monitor, this process only
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def screen_size() -> tuple[int, int]:
    if sys.platform == "win32":
        import ctypes
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    return 1920, 1080


class Display:
    def __init__(self, title: str, canvas: tuple[int, int], model: mujoco.MjModel | None = None,
                 sim_rect: tuple[int, int, int, int] | None = None, camera: str = "demo_cam",
                 fill: float = 0.85):
        enable_dpi_awareness()
        self.title, self.cw, self.ch = title, canvas[0], canvas[1]
        self.model, self.sim_rect, self.camera = model, sim_rect, camera
        sw, sh = screen_size()
        s = min(fill * sw / self.cw, fill * sh / self.ch)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, int(self.cw * s), int(self.ch * s))
        self._hi: mujoco.Renderer | None = None
        self._hi_size = None
        self._layout = None

    def _fit(self) -> tuple[float, int, int, int, int]:
        """Uniform scale and offsets that fit the canvas in the window's image area."""
        try:
            _, _, w, h = cv2.getWindowImageRect(self.title)
        except cv2.error:
            w = h = -1
        if w <= 0 or h <= 0:
            w, h = self.cw, self.ch
        s = min(w / self.cw, h / self.ch)
        return s, w, h, int((w - self.cw * s) / 2), int((h - self.ch * s) / 2)

    def sim_render(self, data: mujoco.MjData, fallback: mujoco.Renderer) -> tuple[np.ndarray, np.ndarray | None]:
        """(canvas-size view, window-size view or None). The window-size view is rendered only
        when the window is larger than the canvas; the canvas copy is then downscaled from it."""
        s, *_ = self._fit()
        if self.model is None or self.sim_rect is None or s <= 1.05:
            fallback.update_scene(data, camera=self.camera)
            return fallback.render(), None
        _, _, rw, rh = self.sim_rect
        g = self.model.vis.global_
        q = round(s * 4) / 4                                   # renderer rebuilt only per quarter step
        size = (min(int(rw * q), g.offwidth), min(int(rh * q), g.offheight))
        if size != self._hi_size:
            if self._hi is not None:
                self._hi.close()
            self._hi, self._hi_size = mujoco.Renderer(self.model, size[1], size[0]), size
        self._hi.update_scene(data, camera=self.camera)
        hi = self._hi.render()
        return cv2.resize(hi, (rw, rh), interpolation=cv2.INTER_AREA), hi

    def show(self, canvas_rgb: np.ndarray, sim_hi: np.ndarray | None = None) -> None:
        s, w, h, ox, oy = self._fit()
        if (w, h) == (self.cw, self.ch):
            img = canvas_rgb
        else:
            tw, th = int(round(self.cw * s)), int(round(self.ch * s))
            inner = cv2.resize(canvas_rgb, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            if sim_hi is not None and self.sim_rect is not None:
                x, y, rw, rh = self.sim_rect
                X, Y = int(round(x * s)), int(round(y * s))
                RW, RH = int(round((x + rw) * s)) - X, int(round((y + rh) * s)) - Y
                if sim_hi.shape[1] != RW or sim_hi.shape[0] != RH:
                    sim_hi = cv2.resize(sim_hi, (RW, RH), interpolation=cv2.INTER_AREA)
                inner[Y:Y + RH, X:X + RW] = sim_hi
            if (tw, th) == (w, h):
                img = inner
            else:
                img = np.zeros((h, w, 3), np.uint8)
                img[oy:oy + th, ox:ox + tw] = inner
        cv2.imshow(self.title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
