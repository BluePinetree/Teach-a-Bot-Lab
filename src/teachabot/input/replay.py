"""Recording InputFrames to JSONL and replaying them through the InputProvider interface.

File format (schema teachabot.input v1):
  line 1 : {"type": "header", "schema": "teachabot.input", "schema_version": 1,
            "source": <original source>, "created": <ISO time>, "meta": {...}}
  line 2+: {"type": "frame", ...InputFrame.to_dict()}

Replaying input is NOT a learning result. It only re-drives the game with the
exact inputs a person gave, for verification and as a demo-day fallback.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from .base import (HANDS, SCHEMA_NAME, SCHEMA_VERSION, HandInput, InputFrame, InputProvider,
                   InputValidationError)


class InputRecorder:
    def __init__(self, path: Path, source: str, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._f.write(json.dumps({"type": "header", "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
                                  "source": source, "created": datetime.now().isoformat(timespec="seconds"),
                                  "meta": meta or {}}) + "\n")
        self.count = 0

    def record(self, frame: InputFrame) -> None:
        self._f.write(json.dumps({"type": "frame", **frame.validate().to_dict()}) + "\n")
        self.count += 1

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_recording(path: Path) -> tuple[dict, list[InputFrame]]:
    with Path(path).open(encoding="utf-8") as f:
        header = json.loads(f.readline())
        if header.get("type") != "header" or header.get("schema") != SCHEMA_NAME:
            raise InputValidationError(f"{path}: not a {SCHEMA_NAME} recording")
        if header.get("schema_version") != SCHEMA_VERSION:
            raise InputValidationError(f"{path}: unsupported schema_version {header.get('schema_version')}")
        frames = []
        for n, line in enumerate(f, start=2):
            rec = json.loads(line)
            if rec.pop("type", None) != "frame":
                raise InputValidationError(f"{path}:{n}: expected a frame record")
            frames.append(InputFrame.from_dict(rec))
    return header, frames


class ReplayInputProvider(InputProvider):
    """Plays back recorded frames.

    realtime=False (default): each poll() returns the next frame — deterministic,
    used for tests. realtime=True: returns the frame matching wall time since
    connect(). After the last frame both hands report disconnected/neutral so a
    consumer stops safely; `finished` becomes True.
    Frames keep their original timestamps and seq; only `source` becomes "replay".
    """

    source = "replay"

    def __init__(self, recording: Path | list[InputFrame], realtime: bool = False):
        if isinstance(recording, list):
            self.header, self.frames = {"source": "memory"}, recording
        else:
            self.header, self.frames = load_recording(recording)
        if not self.frames:
            raise InputValidationError("recording has no frames")
        self.realtime = realtime
        self.finished = False
        self._i = 0
        self._t_start = 0.0

    def connect(self) -> None:
        self._i, self.finished = 0, False
        self._t_start = time.perf_counter()

    def disconnect(self) -> None:
        self.finished = True

    def poll(self) -> InputFrame:
        if self.realtime and not self.finished:
            elapsed = time.perf_counter() - self._t_start
            t0 = self.frames[0].timestamp
            while self._i + 1 < len(self.frames) and self.frames[self._i + 1].timestamp - t0 <= elapsed:
                self._i += 1
            if elapsed > self.frames[-1].timestamp - t0:
                self.finished = True
            return self.frames[self._i].with_source(self.source)
        if self.finished or self._i >= len(self.frames):
            self.finished = True
            last = self.frames[-1]
            return InputFrame(timestamp=last.timestamp, seq=last.seq + 1, source=self.source,
                              left=HandInput.neutral("left"), right=HandInput.neutral("right"),
                              meta={"replay_finished": True})
        frame = self.frames[self._i]
        self._i += 1
        return frame.with_source(self.source)
