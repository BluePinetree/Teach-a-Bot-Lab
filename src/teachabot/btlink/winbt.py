"""Thin WinRT wrapper for classic Bluetooth Joy-Con link state.

Read pairing / connection state and get notified on changes (Phase 0), and
unpair / discover / pair Joy-Cons in SYNC mode (Phase 1, recovery step L3).
See docs/JOYCON_LINK_MANAGER_PLAN.md.

WinRT calls are async; callers run them on their own asyncio loop.
ConnectionStatusChanged fires on a WinRT worker thread, so the callback must be
thread-safe.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from winrt.windows.devices.bluetooth import BluetoothConnectionStatus, BluetoothDevice
from winrt.windows.devices.enumeration import (
    DeviceInformation,
    DeviceInformationKind,
    DevicePairingKinds,
    DevicePairingResultStatus,
    DeviceUnpairingResultStatus,
)

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "joycon_link.json"
JOYCON_NAMES = {"L": "Joy-Con (L)", "R": "Joy-Con (R)"}

# Unpaired classic-Bluetooth devices in range (association endpoints).
_CLASSIC_BT_PROTOCOL = "{e0cbf06c-cd8b-4647-bb8a-263b43f0f974}"
_UNPAIRED_CLASSIC_AQS = (f'System.Devices.Aep.ProtocolId:="{_CLASSIC_BT_PROTOCOL}" AND '
                         "System.Devices.Aep.IsPaired:=System.StructuredQueryType.Boolean#False")
_PAIRING_KINDS = (DevicePairingKinds.CONFIRM_ONLY | DevicePairingKinds.DISPLAY_PIN
                  | DevicePairingKinds.CONFIRM_PIN_MATCH | DevicePairingKinds.PROVIDE_PIN)


def mac_to_int(mac: str) -> int:
    return int(mac.replace(":", "").replace("-", ""), 16)


def load_macs(path: Path = CONFIG_PATH) -> dict[str, str]:
    """{"L": "70:2C:...", "R": ...} from config/joycon_link.json ({} if not set up yet)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("joycons", {})


def save_mac(side: str, mac: str, path: Path = CONFIG_PATH) -> None:
    """Remember the address of a newly paired Joy-Con (only that one is accepted later)."""
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg.setdefault("joycons", {})[side] = mac
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class LinkState:
    found: bool       # Windows knows this address (paired at some point)
    paired: bool
    connected: bool   # baseband link up (not the same as HID reports flowing)
    name: str | None


class BtLink:
    """One Joy-Con's Bluetooth device object, kept alive so its events keep firing."""

    def __init__(self, side: str, mac: str):
        self.side = side
        self.mac = mac
        self._device: BluetoothDevice | None = None
        self._token = None

    async def open(self, on_change: Callable[[str, bool], None] | None = None) -> LinkState:
        self._device = await BluetoothDevice.from_bluetooth_address_async(mac_to_int(self.mac))
        if self._device is not None and on_change is not None:
            side = self.side

            def _handler(dev, _args):
                on_change(side, dev.connection_status == BluetoothConnectionStatus.CONNECTED)

            self._token = self._device.add_connection_status_changed(_handler)
        return self.state()

    def state(self) -> LinkState:
        dev = self._device
        if dev is None:
            return LinkState(found=False, paired=False, connected=False, name=None)
        return LinkState(
            found=True,
            paired=bool(dev.device_information.pairing.is_paired),
            connected=dev.connection_status == BluetoothConnectionStatus.CONNECTED,
            name=dev.name,
        )

    async def refresh_paired(self) -> bool:
        """Current pairing state (the device object's own copy is a snapshot)."""
        if self._device is None:
            return False
        info = await DeviceInformation.create_from_id_async(self._device.device_id)
        return bool(info.pairing.is_paired)

    def close(self) -> None:
        if self._device is not None:
            if self._token is not None:
                self._device.remove_connection_status_changed(self._token)
            self._device.close()
        self._device, self._token = None, None


def normalize_mac(mac: str) -> str:
    return mac.replace("-", ":").upper()


async def unpair(mac: str) -> str:
    """Forget the pairing. Returns "ok", "not_paired" or the WinRT status name."""
    dev = await BluetoothDevice.from_bluetooth_address_async(mac_to_int(mac))
    if dev is None:
        return "not_paired"
    try:
        pairing = dev.device_information.pairing
        if not pairing.is_paired:
            return "not_paired"
        result = await pairing.unpair_async()
        if result.status in (DeviceUnpairingResultStatus.UNPAIRED,
                             DeviceUnpairingResultStatus.ALREADY_UNPAIRED):
            return "ok"
        return result.status.name
    finally:
        dev.close()


@dataclass(frozen=True)
class Pairable:
    side: str
    mac: str
    info: DeviceInformation


async def find_pairable(sides: set[str], timeout_s: float,
                        macs: dict[str, str] | None = None,
                        on_seen: Callable[[str, str], None] | None = None) -> list[Pairable]:
    """Joy-Cons in SYNC mode (unpaired, in range) for the given sides.

    A side with a known MAC in `macs` is matched by address only (an unpaired
    device's name is often empty or "Bluetooth xx:xx:..." at first); a side
    without one is matched by the name "Joy-Con (L)" / "Joy-Con (R)", which may
    only arrive in a later Updated event. Returns once every side is found.
    `on_seen(name, mac)` (worker thread) reports every device for diagnosis.
    """
    loop = asyncio.get_running_loop()
    found: dict[str, Pairable] = {}
    done = asyncio.Event()
    by_mac = {normalize_mac(macs[s]): s for s in sides if macs and s in macs}
    by_name = {JOYCON_NAMES[s]: s for s in sides if s not in by_mac.values()}
    infos: dict[str, DeviceInformation] = {}

    def _consider(info):  # WinRT worker thread
        # Property values arrive boxed; the id ends with the device address instead:
        # "Bluetooth#Bluetooth<radio mac>-<device mac>"
        mac = normalize_mac(info.id.rsplit("-", 1)[-1])
        if on_seen is not None:
            on_seen(info.name, mac)
        side = by_mac.get(mac) or by_name.get(info.name)
        if side is None:
            return

        def _store():
            found.setdefault(side, Pairable(side, mac, info))
            if set(found) == sides:
                done.set()
        loop.call_soon_threadsafe(_store)

    def _on_added(_watcher, info):
        infos[info.id] = info
        _consider(info)

    def _on_updated(_watcher, update):
        info = infos.get(update.id)
        if info is not None:
            info.update(update)
            _consider(info)

    watcher = DeviceInformation.create_watcher_with_kind_aqs_filter_and_additional_properties(
        _UNPAIRED_CLASSIC_AQS, [], DeviceInformationKind.ASSOCIATION_ENDPOINT)
    # Windows raises Added only when Updated and Removed have handlers too.
    tokens = (watcher.add_added(_on_added), watcher.add_updated(_on_updated),
              watcher.add_removed(lambda _w, _u: None))
    watcher.start()
    try:
        await asyncio.wait_for(done.wait(), timeout_s)
    except asyncio.TimeoutError:
        pass
    finally:
        watcher.stop()
        watcher.remove_added(tokens[0])
        watcher.remove_updated(tokens[1])
        watcher.remove_removed(tokens[2])
    return list(found.values())


async def pair(target: Pairable) -> str:
    """Pair a Joy-Con found by find_pairable(). Returns "ok" or the WinRT status name."""
    custom = target.info.pairing.custom

    def _on_request(_custom, args):  # accept whatever the Joy-Con asks for
        if args.pairing_kind == DevicePairingKinds.PROVIDE_PIN:
            args.accept_with_pin("0000")
        else:
            args.accept()

    token = custom.add_pairing_requested(_on_request)
    try:
        result = await custom.pair_async(_PAIRING_KINDS)
    finally:
        custom.remove_pairing_requested(token)
    if result.status in (DevicePairingResultStatus.PAIRED, DevicePairingResultStatus.ALREADY_PAIRED):
        return "ok"
    return result.status.name
