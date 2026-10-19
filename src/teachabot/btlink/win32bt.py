"""Win32 Bluetooth calls that WinRT does not offer (bthprops.cpl via ctypes).

toggle_hid_service(): disable then re-enable the HID service (0x1124) for one
paired device. Windows tears down and rebuilds that device's HID devnode without
forgetting the pairing (recovery ladder step L2 in docs/JOYCON_LINK_MANAGER_PLAN.md).

NOT USED: measured 2026-10-19 on a stuck Joy-Con R, it made things worse (link
dropped within ~1 s, never showed up in HID enumeration). Kept as the record of
that experiment; re-pairing (winbt.unpair/pair) is the recovery path.
"""
from __future__ import annotations

import ctypes
import time
import uuid
from ctypes import wintypes

from teachabot.btlink.winbt import mac_to_int

_bt = ctypes.WinDLL("bthprops.cpl")
_k32 = ctypes.WinDLL("kernel32")
_bt.BluetoothFindFirstRadio.restype = ctypes.c_void_p
_bt.BluetoothFindFirstRadio.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.HANDLE)]
_bt.BluetoothFindNextRadio.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.HANDLE)]
_bt.BluetoothFindRadioClose.argtypes = [ctypes.c_void_p]
_bt.BluetoothGetDeviceInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
_bt.BluetoothSetServiceState.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                         wintypes.DWORD]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]

HID_SERVICE = uuid.UUID("00001124-0000-1000-8000-00805f9b34fb")
BLUETOOTH_SERVICE_DISABLE, BLUETOOTH_SERVICE_ENABLE = 0, 1


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_uuid(cls, u: uuid.UUID) -> "GUID":
        g = cls()
        ctypes.memmove(ctypes.byref(g), u.bytes_le, 16)
        return g


class BLUETOOTH_DEVICE_INFO(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("Address", ctypes.c_ulonglong),
                ("ulClassofDevice", wintypes.ULONG), ("fConnected", wintypes.BOOL),
                ("fRemembered", wintypes.BOOL), ("fAuthenticated", wintypes.BOOL),
                ("stLastSeen", wintypes.WORD * 8), ("stLastUsed", wintypes.WORD * 8),
                ("szName", wintypes.WCHAR * 248)]


class BLUETOOTH_FIND_RADIO_PARAMS(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD)]


def _radios() -> list[wintypes.HANDLE]:
    params = BLUETOOTH_FIND_RADIO_PARAMS(ctypes.sizeof(BLUETOOTH_FIND_RADIO_PARAMS))
    radio = wintypes.HANDLE()
    find = _bt.BluetoothFindFirstRadio(ctypes.byref(params), ctypes.byref(radio))
    if not find:
        return []
    out = [radio]
    while True:
        nxt = wintypes.HANDLE()
        if not _bt.BluetoothFindNextRadio(find, ctypes.byref(nxt)):
            break
        out.append(nxt)
    _bt.BluetoothFindRadioClose(find)
    return out


def toggle_hid_service(mac: str, off_s: float = 1.0) -> str:
    """Returns "ok" or a short error description. Tries every local radio."""
    errors = []
    guid = GUID.from_uuid(HID_SERVICE)
    for radio in _radios():
        try:
            info = BLUETOOTH_DEVICE_INFO()
            info.dwSize = ctypes.sizeof(BLUETOOTH_DEVICE_INFO)
            info.Address = mac_to_int(mac)
            rc = _bt.BluetoothGetDeviceInfo(radio, ctypes.byref(info))
            if rc != 0:
                errors.append(f"GetDeviceInfo={rc}")
                continue
            rc = _bt.BluetoothSetServiceState(radio, ctypes.byref(info), ctypes.byref(guid),
                                              BLUETOOTH_SERVICE_DISABLE)
            if rc != 0:
                errors.append(f"disable={rc}")
                continue
            time.sleep(off_s)
            rc = _bt.BluetoothSetServiceState(radio, ctypes.byref(info), ctypes.byref(guid),
                                              BLUETOOTH_SERVICE_ENABLE)
            return "ok" if rc == 0 else f"enable={rc} (HID service left DISABLED)"
        finally:
            _k32.CloseHandle(radio)
    return "; ".join(errors) or "no Bluetooth radio found"
