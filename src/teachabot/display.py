"""Sharp, non-blocking window output on high-DPI screens.

2026-10-20: "해상도가 왜 이리 낮지?" — the PC runs 3840x2160 at 150 %. A process that is not DPI
aware gets its window bitmap stretched 1.5x by Windows, on top of OpenCV's crude WINDOW_NORMAL
stretch. So the process declares itself DPI aware (this process only, no system setting), the
window is sized in physical pixels, the canvas is scaled by us to the window's real size, and
while nothing covers the 3D view that view is rendered again at window size and pasted in.

Then "해상도를 높이니까 프레임이 많이 떨어졌다" (rehearsal3: 60 ms per frame vs 37 ms headless):
scaling, colour conversion and showing a ~6 MP image added ~20 ms to every frame of the
simulation loop. The window now lives in its own thread — it is created, scaled, shown and
pumped (waitKey) there, OpenCV releases the GIL while it works, and the simulation loop only
hands over its newest frame. Keys come back through a queue.
"""
from __future__ import annotations

import queue
import sys
import threading

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


def fit(canvas: tuple[int, int], window: tuple[int, int]) -> tuple[float, int, int]:
    """Uniform scale and offsets that fit the canvas inside the window."""
    cw, ch = canvas
    w, h = window
    s = min(w / cw, h / ch)
    return s, int((w - cw * s) / 2), int((h - ch * s) / 2)


def compose(canvas_bgr: np.ndarray, window: tuple[int, int], sim_hi_bgr: np.ndarray | None,
            sim_rect: tuple[int, int, int, int] | None) -> np.ndarray:
    """The image the window shows: canvas scaled to the window (letterboxed), with the
    window-size 3D view pasted over the canvas' 3D area when given."""
    ch, cw = canvas_bgr.shape[:2]
    w, h = window
    if (w, h) == (cw, ch) or w <= 0 or h <= 0:
        return canvas_bgr
    s, ox, oy = fit((cw, ch), (w, h))
    tw, th = int(round(cw * s)), int(round(ch * s))
    inner = cv2.resize(canvas_bgr, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if sim_hi_bgr is not None and sim_rect is not None:
        x, y, rw, rh = sim_rect
        X, Y = int(round(x * s)), int(round(y * s))
        RW, RH = int(round((x + rw) * s)) - X, int(round((y + rh) * s)) - Y
        if sim_hi_bgr.shape[1] != RW or sim_hi_bgr.shape[0] != RH:
            sim_hi_bgr = cv2.resize(sim_hi_bgr, (RW, RH), interpolation=cv2.INTER_AREA)
        inner[Y:Y + RH, X:X + RW] = sim_hi_bgr
    if (tw, th) == (w, h):
        return inner
    out = np.zeros((h, w, 3), np.uint8)
    out[oy:oy + th, ox:ox + tw] = inner
    return out


class Display:
    def __init__(self, title: str, canvas: tuple[int, int], model: mujoco.MjModel | None = None,
                 sim_rect: tuple[int, int, int, int] | None = None, camera: str = "demo_cam",
                 fill: float = 0.85, sharp_3d: bool = True):
        enable_dpi_awareness()
        self.title, self.cw, self.ch = title, canvas[0], canvas[1]
        self.model, self.sim_rect, self.camera = model, sim_rect, camera
        self.sharp_3d = sharp_3d          # False: scale the 3D view with the canvas (saves ~4.5 ms per frame)
        sw, sh = screen_size()
        s = min(fill * sw / self.cw, fill * sh / self.ch)
        self._initial = (int(self.cw * s), int(self.ch * s))
        self.window = self._initial                     # updated by the window thread
        self.closed = False
        self._keys: queue.Queue = queue.Queue()
        self._latest = None
        self._new = threading.Event()
        self._stop = threading.Event()
        self._fullscreen_req: bool | None = None
        self.show_ms = 0.0
        self._hi: mujoco.Renderer | None = None
        self._hi_size = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="window", daemon=True)
        self._thread.start()
        self._ready.wait(5.0)

    # -- called from the simulation loop -------------------------------------------------
    def sim_render(self, data: mujoco.MjData, fallback: mujoco.Renderer) -> tuple[np.ndarray, np.ndarray | None]:
        """(canvas-size 3D view, window-size 3D view or None). Rendering stays in this thread
        (the GL context lives here); the window-size view only when the window is larger."""
        s, _, _ = fit((self.cw, self.ch), self.window)
        if self.model is None or self.sim_rect is None or s <= 1.05 or not self.sharp_3d:
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
        """Hand over the newest frame (never blocks; an unshown older frame is replaced)."""
        self._latest = (canvas_rgb, sim_hi)
        self._new.set()

    def poll_key(self) -> int:
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return 255

    def toggle_fullscreen(self) -> None:
        self._fullscreen_req = not bool(self._fullscreen_req)

    def close(self) -> None:
        self._stop.set()
        self._new.set()
        self._thread.join(timeout=3.0)

    # -- window thread -------------------------------------------------------------------
    def _run(self) -> None:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, *self._initial)
        self._ready.set()
        fullscreen = False
        try:
            while not self._stop.is_set():
                if self._new.wait(0.01):
                    self._new.clear()
                    item = self._latest
                    if item is not None:
                        import time
                        t0 = time.perf_counter()
                        canvas, hi = item
                        try:
                            _, _, w, h = cv2.getWindowImageRect(self.title)
                        except cv2.error:
                            w, h = self.cw, self.ch
                        if w > 0 and h > 0:
                            self.window = (w, h)
                        img = compose(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), self.window,
                                      None if hi is None else cv2.cvtColor(hi, cv2.COLOR_RGB2BGR), self.sim_rect)
                        cv2.imshow(self.title, img)
                        self.show_ms = 0.9 * self.show_ms + 0.1 * (time.perf_counter() - t0) * 1000
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._keys.put(key)
                if self._fullscreen_req is not None and self._fullscreen_req != fullscreen:
                    fullscreen = self._fullscreen_req
                    cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    self.closed = True
                    break
        finally:
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass
