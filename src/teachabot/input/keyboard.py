"""KeyboardInputProvider: fallback input with the same InputFrame interface.

Default key reader is Win32 GetAsyncKeyState (works without a window, reads
keys globally). A 3D engine can pass its own `key_reader(vk_code) -> bool`.

Layout:
  left hand : W/A/S/D move, Q grip (ZL), E shoulder (L), Tab minus
  right hand: arrow keys move, Right Ctrl grip (ZR), Right Shift shoulder (R),
              Enter plus, Space a
"""
from __future__ import annotations

import math
import time
from typing import Callable

from .base import BUTTONS, HandInput, InputFrame, InputProvider

VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44, "Q": 0x51, "E": 0x45, "TAB": 0x09,
      "LEFT": 0x25, "UP": 0x26, "RIGHT": 0x27, "DOWN": 0x28,
      "RCTRL": 0xA3, "RSHIFT": 0xA1, "ENTER": 0x0D, "SPACE": 0x20}

LAYOUT = {
    "left": {"up": "W", "down": "S", "left": "A", "right": "D",
             "buttons": {"grip": "Q", "shoulder": "E", "minus": "TAB"}},
    "right": {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT",
              "buttons": {"grip": "RCTRL", "shoulder": "RSHIFT", "plus": "ENTER", "a": "SPACE"}},
}


def win32_key_reader(vk: int) -> bool:
    import ctypes
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


class KeyboardInputProvider(InputProvider):
    source = "keyboard"

    def __init__(self, key_reader: Callable[[int], bool] | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        self._key = key_reader or win32_key_reader
        self._clock = clock
        self._seq = 0
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _hand(self, hand: str) -> HandInput:
        if not self._connected:
            return HandInput.neutral(hand)
        lay = LAYOUT[hand]
        pressed = lambda name: self._key(VK[name])
        x = float(pressed(lay["right"])) - float(pressed(lay["left"]))
        y = float(pressed(lay["up"])) - float(pressed(lay["down"]))
        r = math.hypot(x, y)
        if r > 1.0:
            x, y = x / r, y / r
        buttons = {b: False for b in BUTTONS[hand]}
        for b, key in lay["buttons"].items():
            buttons[b] = pressed(key)
        return HandInput(connected=True, stick=(x, y), buttons=buttons)

    def poll(self) -> InputFrame:
        frame = InputFrame(timestamp=self._clock(), seq=self._seq, source=self.source,
                           left=self._hand("left"), right=self._hand("right"))
        self._seq += 1
        return frame
