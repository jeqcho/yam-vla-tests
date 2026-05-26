"""List connected RealSense cameras with serial numbers — used to fill in
yam_client.py's --top/--left/--right-cam-serial flags.

    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/list_cams.py
"""
import pyrealsense2 as rs

devices = list(rs.context().query_devices())
if not devices:
    print("No RealSense devices detected. Plug them in and try again.")
    raise SystemExit(1)

print(f"Found {len(devices)} device(s):\n")
for i, d in enumerate(devices):
    name = d.get_info(rs.camera_info.name)
    serial = d.get_info(rs.camera_info.serial_number)
    fw = d.get_info(rs.camera_info.firmware_version)
    usb = d.get_info(rs.camera_info.usb_type_descriptor)
    print(f"  [{i}] {name}")
    print(f"      serial={serial}  fw={fw}  usb={usb}")

print("\nFlag template:")
serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
if len(serials) >= 3:
    print(f"  --top-cam-serial {serials[0]} --left-cam-serial {serials[1]} --right-cam-serial {serials[2]}")
    print("\n(Verify which physical mount each serial is — the order above is whatever USB returned.)")
