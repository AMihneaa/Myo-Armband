import pyqtgraph as pg

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QFont

from ..acquisition.client import MyoStreamClient
from ..recording.session import SessionRecorder


EMG_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
              '#911eb4', '#42d4f4', '#f032e6', '#bfef45']
IMU_COLORS = ['r', 'g', 'b']
IMU_LABELS = ['X', 'Y', 'Z']


class MainWindow(QMainWindow):

    def __init__(
        self,
        client: MyoStreamClient,
        session_recorder: SessionRecorder = None,
        refresh_ms: int = 50,
    ) -> None:
        super().__init__()

        self._client = client
        self._session_recorder = session_recorder
        self._refresh_ms = refresh_ms

        self.setWindowTitle("Myo Recorder")
        self.resize(1400, 1200)

        self._build_ui()
        self._build_curves()
        self._start_timer()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Cue label
        self._cue_label = QLabel("")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self._cue_label.setFont(font)
        self._cue_label.setStyleSheet("color: #00ff99; background: #111; padding: 6px;")
        layout.addWidget(self._cue_label)

        # Plot area
        self._graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graphics)

        self.setCentralWidget(root)

        # EMG plots
        self._emg_plots = []
        for ch in range(8):
            plot = self._graphics.addPlot(title=f"EMG ch{ch + 1}")
            plot.showGrid(x=True, y=True)
            plot.setLabel("left", f"ch{ch + 1}")
            if ch == 7:
                plot.setLabel("bottom", "samples")
            self._emg_plots.append(plot)
            self._graphics.nextRow()

        # Accelerometer
        self._acc_plot = self._graphics.addPlot(title="Accelerometer")
        self._acc_plot.showGrid(x=True, y=True)
        self._acc_plot.setLabel("left", "g")
        self._acc_plot.addLegend()
        self._graphics.nextRow()

        # Gyroscope
        self._gyro_plot = self._graphics.addPlot(title="Gyroscope")
        self._gyro_plot.showGrid(x=True, y=True)
        self._gyro_plot.setLabel("left", "deg/s")
        self._gyro_plot.setLabel("bottom", "samples")
        self._gyro_plot.addLegend()

    def _build_curves(self) -> None:
        self._emg_curves = []
        for ch in range(8):
            curve = self._emg_plots[ch].plot(
                pen=pg.mkPen(color=EMG_COLORS[ch], width=1)
            )
            self._emg_curves.append(curve)

        self._acc_curves = []
        self._gyro_curves = []
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

    def _start_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(self._refresh_ms)

    def _update(self) -> None:
        self._update_cue()
        self._update_emg()
        self._update_imu()

    def _update_cue(self) -> None:
        if self._session_recorder is None:
            return
        self._cue_label.setText(self._session_recorder.cue)

    def _update_emg(self) -> None:
        samples = self._client.emg_buffer.snapshot()
        if not samples:
            return
        for ch in range(8):
            y = [s.channels[ch] for s in samples]
            self._emg_curves[ch].setData(y)

    def _update_imu(self) -> None:
        samples = self._client.imu_buffer.snapshot()
        if not samples:
            return
        for i in range(3):
            self._acc_curves[i].setData([s.accelerometer[i] for s in samples])
            self._gyro_curves[i].setData([s.gyroscope[i] for s in samples])