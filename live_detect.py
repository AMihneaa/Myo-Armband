import sys
import json
import asyncio
import qasync

from PySide6.QtWidgets import QApplication

from config import BLE_ADDR, GYRO_CALIBRATION_SAMPLES
from acquisition.client import MyoStreamClient
from acquisition.calibrator import GyroBias
from myo.types import ClassifierMode, EMGMode, IMUMode
from ui.bridge import SignalBridge
from ui.live_window import LiveDetectWindow
from network.ws_client import websocket_session

sys.stdout.reconfigure(line_buffering=True)


async def main(app):
    print("[main] started")
    app_close_event= asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    bridge= SignalBridge()
    send_queue= asyncio.Queue(maxsize=400)

    print("[main] connecting to myo")

    client= await MyoStreamClient.with_device(
        BLE_ADDR,
        aggregate_all=False,
        aggregate_emg=False,
    )

    def emg_callback(sample):
        bridge.emg_ready.emit([sample.channels])
        payload= json.dumps({
            "type": "emg",
            "samples": [list(sample.channels)],
            "n_channels": 8,
        })
        try:
            send_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def imu_callback(sample):
        bridge.imu_ready.emit((sample.gyroscope, sample.accelerometer))
        payload= json.dumps({
            "type": "imu",
            "accelerometer": list(sample.accelerometer),
            "gyroscope": list(sample.gyroscope),
        })
        try:
            send_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


    def calibration_done_callback(bias: GyroBias):
        print(f"[calibration] complete — bias x={bias.x:.4f} y={bias.y:.4f} z={bias.z:.4f}")

    client.set_emg_callback(emg_callback)
    client.set_imu_callback(imu_callback)
    client.set_calibration_callback(calibration_done_callback)

    window= LiveDetectWindow(bridge)
    window.show()

    try:
        await client.setup(
            classifier_mode=ClassifierMode.DISABLED,
            emg_mode=EMGMode.SEND_RAW,
            imu_mode=IMUMode.SEND_ALL,
        )

        print(f"[calibration] hold arm still for ~{GYRO_CALIBRATION_SAMPLES // 50}s ({GYRO_CALIBRATION_SAMPLES} samples at 50 Hz)...")

        await client.start()

        print("[main] starting websocket session")
        asyncio.ensure_future(websocket_session(bridge, send_queue))
        print("[main] websocket session scheduled")

        await app_close_event.wait()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await send_queue.put(None)
        await client.stop()
        await client.disconnect()


if __name__ == "__main__":
    app= QApplication(sys.argv)
    qasync.run(main(app))