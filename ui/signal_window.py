import pyqtgraph as pg

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QScrollArea,
)

from acquisition.client import MyoStreamClient


EMG_COLORS= ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
             '#911eb4', '#42d4f4', '#f032e6', '#bfef45']
IMU_COLORS= ['r', 'g', 'b']
IMU_LABELS= ['X', 'Y', 'Z']

PLOT_HEIGHT= 200
PLOT_WIDTH= 1360


class SignalWindow(QMainWindow):
    def __init__(
        self,
        client: MyoStreamClient,
        refresh_ms: int= 33,
    ) -> None:
        super().__init__()

        self._client= client

        self.setWindowTitle("Myo Recorder — Signals")
        self.resize(1400, 900)

        self._build_ui()
        self._build_curves()
        self._start_timer(refresh_ms)

    def _build_ui(self) -> None:
        scroll_area= QScrollArea()
        scroll_area.setWidgetResizable(True)

        container= QWidget()
        layout= QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._emg_plots= []
        for ch in range(8):
            plot= pg.PlotWidget(title=f"EMG ch{ch + 1}")
            plot.setFixedHeight(PLOT_HEIGHT)
            plot.setMinimumWidth(PLOT_WIDTH)
            plot.showGrid(x=True, y=True)
            plot.setLabel("left", f"ch{ch + 1}")
            if ch == 7:
                plot.setLabel("bottom", "samples")
            self._emg_plots.append(plot)
            layout.addWidget(plot)

        self._acc_plot= pg.PlotWidget(title="Accelerometer")
        self._acc_plot.setFixedHeight(PLOT_HEIGHT)
        self._acc_plot.setMinimumWidth(PLOT_WIDTH)
        self._acc_plot.showGrid(x=True, y=True)
        self._acc_plot.setLabel("left", "g")
        self._acc_plot.addLegend()
        layout.addWidget(self._acc_plot)

        self._gyro_plot= pg.PlotWidget(title="Gyroscope")
        self._gyro_plot.setFixedHeight(PLOT_HEIGHT)
        self._gyro_plot.setMinimumWidth(PLOT_WIDTH)
        self._gyro_plot.showGrid(x=True, y=True)
        self._gyro_plot.setLabel("left", "deg/s")
        self._gyro_plot.setLabel("bottom", "samples")
        self._gyro_plot.addLegend()
        layout.addWidget(self._gyro_plot)

        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)

    def _build_curves(self) -> None:
        self._emg_curves= []
        for ch in range(8):
            curve= self._emg_plots[ch].plot(
                pen=pg.mkPen(color=EMG_COLORS[ch], width=1)
            )
            self._emg_curves.append(curve)

        self._acc_curves= []
        self._gyro_curves= []
        for i in range(3):
            self._acc_curves.append(
                self._acc_plot.plot(
                    pen=pg.mkPen(color=IMU_COLORS[i], width=1.5),
                    name=IMU_LABELS[i],
                )
            )
            self._gyro_curves.append(
                self._gyro_plot.plot(
                    pen=pg.mkPen(color=IMU_COLORS[i], width=1.5),
                    name=IMU_LABELS[i],
                )
            )

    def _start_timer(self, refresh_ms: int) -> None:
        self._timer= QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(refresh_ms)

    def _update(self) -> None:
        self._update_emg()
        self._update_imu()

    def _update_emg(self) -> None:
        samples= self._client.emg_buffer.snapshot()
        if not samples:
            return
        for ch in range(8):
            y= [s.channels[ch] for s in samples]
            self._emg_curves[ch].setData(y)

    def _update_imu(self) -> None:
        samples= self._client.imu_buffer.snapshot()
        if not samples:
            return
        for i in range(3):
            self._acc_curves[i].setData([s.accelerometer[i] for s in samples])
            self._gyro_curves[i].setData([s.gyroscope[i] for s in samples])