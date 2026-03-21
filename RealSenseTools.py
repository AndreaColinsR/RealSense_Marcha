import pyrealsense2 as rs

def list_realsense_devices():
    # Create a context object, which is the entry point to the device list
    ctx = rs.context()

    # Query all currently connected devices
    devices = ctx.query_devices()
    SN = []
    if len(devices) == 0:
        print("No RealSense devices connected.")
    else:
        print(f"Found {len(devices)} connected RealSense devices:")
        for i, device in enumerate(devices):
            # Print device information
            print(f"\n--- Device {i+1} ---")
            print(f"  Name: {device.get_info(rs.camera_info.name)}")
            print(f"  Serial Number: {device.get_info(rs.camera_info.serial_number)}")
            print(f"  Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
            print(f"  Product ID: 0x{device.get_info(rs.camera_info.product_id)}")
            SN.append(device.get_info(rs.camera_info.serial_number))
    return SN