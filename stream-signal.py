import sys
import asyncio
import logging
from collections import deque

import pyqtgraph as pg
import qasync

from myo import MyoClient
from myo.types import (
    ClassifierEvent,
    AggregatedData,
    EMGDataSingle,
    FVData,
    IMUData,
    EMGData,
    MotionEvent,
    ClassifierMode,
    EMGMode,
    IMUMode,
)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow

addr = "E4:96:A9:A7:5C:74"
logger = logging.getLogger(__name__)


class CircularBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def append(self, value):
        self.buffer.append(value)

    def get(self):
        return list(self.buffer)


class TestClient(MyoClient):
    def __init__(
        self,
        aggregate_all=False,
        aggregate_emg=False,
        emgCircularBuffer_size: int = 500,
        imuCircularBuffer_size: int = 150,
    ):
        super().__init__(aggregate_all=aggregate_all, aggregate_emg=aggregate_emg)
        self._emgCircularBuffer = CircularBuffer(emgCircularBuffer_size)
        self._imuCircularBuffer = CircularBuffer(imuCircularBuffer_size)

    async def on_classifier_event(self, ce: ClassifierEvent):
        pass

    async def on_aggregated_data(self, ad: AggregatedData):
        pass

    async def on_emg_data_aggregated(self, eds: EMGDataSingle):
        self._emgCircularBuffer.append(tuple(eds))

    async def on_emg_data(self, data: EMGData):
        self._emgCircularBuffer.append(tuple(data.sample1))
        self._emgCircularBuffer.append(tuple(data.sample2))

    async def on_motion_event(self, me: MotionEvent):
        pass

    async def on_fv_data(self, fvd: FVData):
        pass

    async def on_imu_data(self, imu: IMUData):
        self._imuCircularBuffer.append(
            (
                tuple(imu.orientation),
                tuple(imu.accelerometer),
                tuple(imu.gyroscope),
            )
        )

    @property
    def emgCircularBuffer(self) -> CircularBuffer:
        return self._emgCircularBuffer

    @property
    def imuCircularBuffer(self) -> CircularBuffer:
        return self._imuCircularBuffer


class MainWindow(QMainWindow):
    def __init__(self, client: TestClient, refresh_ms: int = 30):
        super().__init__()

        self.client = client
        self.refresh_ms = refresh_ms

        self.setWindowTitle("Myo Live Monitor")
        self.resize(1400, 1100)

        self._create_plot_widgets()
        self._create_curve_handles()
        self._create_timer()
        self._connect_timer()
        self._start_timer()

    def _create_plot_widgets(self):
        self.graphics = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graphics)

        self.emg_plots = []

        # 8 separate EMG plots
        for ch in range(8):
            plot = self.graphics.addPlot(title=f"Raw EMG - Channel {ch + 1}")
            plot.showGrid(x=True, y=True)
            plot.setLabel("left", f"EMG {ch + 1}")
            if ch == 7:
                plot.setLabel("bottom", "Samples")
            self.emg_plots.append(plot)
            self.graphics.nextRow()

        # Accelerometer plot
        self.acc_plot = self.graphics.addPlot(title="Accelerometer")
        self.acc_plot.showGrid(x=True, y=True)
        self.acc_plot.setLabel("left", "g")
        self.graphics.nextRow()

        # Gyroscope plot
        self.gyro_plot = self.graphics.addPlot(title="Gyroscope")
        self.gyro_plot.showGrid(x=True, y=True)
        self.gyro_plot.setLabel("left", "deg/s")
        self.gyro_plot.setLabel("bottom", "Samples")

    def _create_curve_handles(self):
        self.emg_curves = []
        for ch in range(8):
            curve = self.emg_plots[ch].plot()
            self.emg_curves.append(curve)

        self.acc_curves = []
        for _ in range(3):
            curve = self.acc_plot.plot()
            self.acc_curves.append(curve)

        self.gyro_curves = []
        for _ in range(3):
            curve = self.gyro_plot.plot()
            self.gyro_curves.append(curve)

    def _create_timer(self):
        self.timer = QTimer(self)

    def _connect_timer(self):
        self.timer.timeout.connect(self.update_plots)

    def _start_timer(self):
        self.timer.start(self.refresh_ms)

    def update_plots(self):
        self._update_emg_plot()
        self._update_imu_plots()

    def _update_emg_plot(self):
        emg_samples = self.client.emgCircularBuffer.get()

        if not emg_samples:
            return

        for ch in range(8):
            y = [sample[ch] for sample in emg_samples]
            self.emg_curves[ch].setData(y)

    def _update_imu_plots(self):
        imu_samples = self.client.imuCircularBuffer.get()

        if not imu_samples:
            return

        acc_samples = [sample[1] for sample in imu_samples]
        gyro_samples = [sample[2] for sample in imu_samples]

        for axis in range(3):
            acc_y = [sample[axis] for sample in acc_samples]
            gyro_y = [sample[axis] for sample in gyro_samples]

            self.acc_curves[axis].setData(acc_y)
            self.gyro_curves[axis].setData(gyro_y)


async def main(app, addr):
    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    client = await TestClient.with_device(
        addr,
        aggregate_all=False,
        aggregate_emg=False,
    )

    window = MainWindow(client, refresh_ms=50)
    window.show()

    try:
        info = await client.get_services()
        logger.info(info)

        await client.setup(
            classifier_mode=ClassifierMode.DISABLED,
            emg_mode=EMGMode.SEND_RAW,
            imu_mode=IMUMode.SEND_ALL,
        )

        await client.start()
        await app_close_event.wait()

    except Exception as e:
        logger.exception(f"Error while connecting/streaming: {e}")

    finally:
        await client.stop()
        await client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)
    qasync.run(main(app, addr))