"""Video recording off the frame loop.

Encoding a 1920x1080 mp4v frame inside the loop (several copies per loop to keep 30 fps) dragged
the visit window to 78 ms per frame on 2026-10-20 (31 ms without video). Frames now go to a
background thread through a small queue; if the encoder falls behind, frames are dropped and
counted (the video then runs slightly short, the window stays smooth).
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class VideoSink:
    def __init__(self, path: Path, fps: int):
        self.path, self.fps = Path(path), fps
        self.frames = 0
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=3)
        self._writer: cv2.VideoWriter | None = None
        self._thread = threading.Thread(target=self._run, name="video", daemon=True)
        self._thread.start()

    def write(self, rgb: np.ndarray, copies: int = 1) -> None:
        """Queue `copies` repeats of this frame (the caller keeps the 30 fps clock).
        `rgb` must not be changed afterwards (the app makes a new image every frame)."""
        if copies <= 0:
            return
        try:
            self._q.put_nowait((rgb, copies))
        except queue.Full:
            self.dropped += copies

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            rgb, copies = item
            if self._writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                               (rgb.shape[1], rgb.shape[0]))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for _ in range(copies):
                self._writer.write(bgr)
            self.frames += copies

    def close(self) -> dict:
        self._q.put(None)
        self._thread.join(timeout=30)
        if self._writer is not None:
            self._writer.release()
        return {"frames": self.frames, "dropped": self.dropped}
