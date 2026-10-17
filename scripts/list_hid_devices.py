"""List Nintendo (VID 057E) HID devices Windows exposes. Use after Bluetooth pairing."""
import hid

NAMES = {0x2006: "Joy-Con (L)", 0x2007: "Joy-Con (R)", 0x2009: "Pro Controller"}

devices = [d for d in hid.enumerate() if d["vendor_id"] == 0x057E]
print(f"{len(devices)} Nintendo HID device(s)")
for d in devices:
    pid = d["product_id"]
    print(f"  057E:{pid:04X}  {NAMES.get(pid, '?'):<14} product_string={d['product_string']!r} "
          f"serial={d['serial_number']!r}")
