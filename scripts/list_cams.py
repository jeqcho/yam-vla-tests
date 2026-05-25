"""Enumerate every RealSense camera attached to the host.

Run with the i2rt venv:
    /home/andon/yam-tests/i2rt/.venv/bin/python scripts/list_cams.py

Output: serial / name / firmware for every detected device.
"""
import pyrealsense2 as rs


def main() -> int:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("No RealSense devices found.")
        print("If you have cameras plugged in, install the udev rules:")
        print("  https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules")
        return 1
    print(f"Found {len(devices)} RealSense device(s):\n")
    for i, d in enumerate(devices):
        name = d.get_info(rs.camera_info.name)
        serial = d.get_info(rs.camera_info.serial_number)
        fw = d.get_info(rs.camera_info.firmware_version)
        usb = d.get_info(rs.camera_info.usb_type_descriptor)
        print(f"  [{i}] {name:30s}  serial={serial}  fw={fw}  usb={usb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
