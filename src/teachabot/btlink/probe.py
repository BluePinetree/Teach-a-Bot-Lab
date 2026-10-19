"""One-shot HID health check, used only when the app is not running.

A stuck Joy-Con (docs/JOYCON_LINK_MANAGER_PLAN.md section 9) is listed in HID
enumeration but output writes fail (hidapi write -> -1) and nothing comes back.
A healthy one answers subcommand 0x02 (device info) with an 0x21 reply.
Opens the device briefly, so never call it while the app holds the Joy-Con.
"""
from __future__ import annotations

import time

import hid

from teachabot.input.joycon_device import JOYCON_VENDOR_ID, PRODUCT_IDS

_DEVICE_INFO = bytes([0x01, 0x00, 0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40, 0x02]) + bytes(38)


def probe_hid(side: str, timeout_s: float = 1.0) -> bool:
    devices = hid.enumerate(JOYCON_VENDOR_ID, PRODUCT_IDS[side])
    if not devices:
        return False
    dev = hid.device()
    try:
        dev.open_path(devices[0]["path"])
        dev.set_nonblocking(1)
        if dev.write(_DEVICE_INFO) < 0:
            return False
        end = time.perf_counter() + timeout_s
        while time.perf_counter() < end:
            report = dev.read(64)
            if report and report[0] == 0x21:
                return True
            if not report:
                time.sleep(0.005)
        return False
    except (OSError, ValueError):
        return False
    finally:
        try:
            dev.close()
        except Exception:
            pass
