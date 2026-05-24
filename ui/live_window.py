from collections import deque

from PySide6.QtWidgets import (
    QMainWindow, QWidget,
    QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QPainterPath

from ui.bridge import SignalBridge

HISTORY= 200
EMG_COLORS= [
    "#00FF9C", "#00D4FF", "#FF6B35", "#FFD700",
    "#FF3CAC", "#7B2FFF", "#00FF85", "#FF8C00",
]
GYRO_COLORS=  ["#FF4444", "#44FF44", "#4444FF"]
ACCEL_COLORS= ["#FF8844", "#44FF88", "#8844FF"]

GESTURE_NAMES= {
    0: "Relax",
    1: "Hand Open",
    2: "Hand Close",
    3: "Wrist Extension",
    4: "Wrist Flexion",
}


class OscilloscopeWidget(QWidget):
    def __init__(self, n_traces, colors, label, y_range=(-128, 128), parent=None):
        super().__init__(parent)
        self._n= n_traces
        self._colors= colors
        self._label= label
        self._y_min, self._y_max= y_range
        self._buffers= [deque([0.0] * HISTORY, maxlen=HISTORY) for _ in range(n_traces)]
        self.setMinimumHeight(120)

    def push(self, values):
        for i, v in enumerate(values[:self._n]):
            self._buffers[i].append(float(v))
        self.update()

    def paintEvent(self, event):
        p= QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w= self.width()
        h= self.height()

        p.fillRect(0, 0, w, h, QColor("#0A0A0F"))

        grid_pen= QPen(QColor("#1A1A2E"))
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        for i in range(1, 4):
            y= int(h * i / 4)
            p.drawLine(0, y, w, y)
        for i in range(1, 8):
            x= int(w * i / 8)
            p.drawLine(x, 0, x, h)

        zero_pen= QPen(QColor("#2A2A4A"))
        zero_pen.setWidth(1)
        p.setPen(zero_pen)
        p.drawLine(0, int(h * 0.5), w, int(h * 0.5))

        y_range= self._y_max - self._y_min if self._y_max != self._y_min else 1

        for i, buf in enumerate(self._buffers):
            color= QColor(self._colors[i % len(self._colors)])
            pen= QPen(color)
            pen.setWidth(1)
            p.setPen(pen)

            path= QPainterPath()
            pts= list(buf)
            for j, val in enumerate(pts):
                x= int(w * j / (HISTORY - 1))
                norm= (val - self._y_min) / y_range
                y= int(h * (1.0 - norm))
                y= max(0, min(h - 1, y))
                if j == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.drawPath(path)

        label_pen= QPen(QColor("#444466"))
        p.setPen(label_pen)
        p.setFont(QFont("Courier New", 8))
        p.drawText(4, 14, self._label)
        p.end()


class PredictionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gesture= None
        self._confidence= 0.0
        self.setMinimumWidth(220)

    def update_prediction(self, gesture, confidence):
        self._gesture= gesture
        self._confidence= confidence
        self.update()

    def paintEvent(self, event):
        p= QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w= self.width()
        h= self.height()

        p.fillRect(0, 0, w, h, QColor("#0A0A0F"))

        border_pen= QPen(QColor("#1A1A3A"))
        border_pen.setWidth(1)
        p.setPen(border_pen)
        p.drawRect(0, 0, w - 1, h - 1)

        if self._gesture is None:
            p.setPen(QColor("#333355"))
            p.setFont(QFont("Courier New", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "WAITING\nFOR SIGNAL")
            p.end()
            return

        gesture_int= int(self._gesture) if str(self._gesture).isdigit() else -1
        name= GESTURE_NAMES.get(gesture_int, f"Gesture {self._gesture}")

        p.setPen(QColor("#00FF9C"))
        p.setFont(QFont("Courier New", 36, QFont.Bold))
        p.drawText(0, 10, w, int(h * 0.45), Qt.AlignCenter, str(self._gesture))

        p.setPen(QColor("#AAAACC"))
        p.setFont(QFont("Courier New", 10))
        p.drawText(0, int(h * 0.48), w, 24, Qt.AlignCenter, name)

        p.setPen(QColor("#555577"))
        p.setFont(QFont("Courier New", 8))
        p.drawText(8, int(h * 0.68), f"CONF  {self._confidence:.1%}")

        bar_x= 8
        bar_y= int(h * 0.76)
        bar_w= w - 16
        bar_h= 8
        p.fillRect(bar_x, bar_y, bar_w, bar_h, QColor("#1A1A2E"))

        fill_w= int(bar_w * self._confidence)
        if fill_w > 0:
            grad= QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
            grad.setColorAt(0.0, QColor("#00FF9C"))
            grad.setColorAt(1.0, QColor("#00D4FF"))
            p.fillRect(bar_x, bar_y, fill_w, bar_h, grad)

        p.end()


class LiveDetectWindow(QMainWindow):
    def __init__(self, bridge: SignalBridge):
        super().__init__()
        self.setWindowTitle("MYO — Live Gesture Detection")
        self.setStyleSheet("background-color: #0A0A0F; color: #CCCCEE;")
        self.resize(1100, 700)

        central= QWidget()
        self.setCentralWidget(central)
        root= QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header= QLabel("● MYO EMG LIVE DETECTION")
        header.setFont(QFont("Courier New", 10, QFont.Bold))
        header.setStyleSheet("color: #00FF9C; padding: 2px 0px;")
        root.addWidget(header)

        div= QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color: #1A1A3A;")
        root.addWidget(div)

        emg_grid= QGridLayout()
        emg_grid.setSpacing(4)
        self._emg_plots= []
        for i in range(8):
            plot= OscilloscopeWidget(
                n_traces=1,
                colors=[EMG_COLORS[i]],
                label=f"EMG CH{i}",
                y_range=(-128, 128),
            )
            emg_grid.addWidget(plot, i // 4, i % 4)
            self._emg_plots.append(plot)
        root.addLayout(emg_grid, stretch=3)

        bottom= QHBoxLayout()
        bottom.setSpacing(6)

        self._gyro_plot= OscilloscopeWidget(
            n_traces=3,
            colors=GYRO_COLORS,
            label="GYROSCOPE  X/Y/Z",
            y_range=(-2000, 2000),
        )
        bottom.addWidget(self._gyro_plot, stretch=2)

        self._accel_plot= OscilloscopeWidget(
            n_traces=3,
            colors=ACCEL_COLORS,
            label="ACCELEROMETER  X/Y/Z",
            y_range=(-4, 4),
        )
        bottom.addWidget(self._accel_plot, stretch=2)

        self._pred_widget= PredictionWidget()
        bottom.addWidget(self._pred_widget, stretch=1)

        root.addLayout(bottom, stretch=2)

        bridge.emg_ready.connect(self._on_emg)
        bridge.imu_ready.connect(self._on_imu)
        bridge.prediction_ready.connect(self._on_prediction)

    def _on_emg(self, samples):
        for s in samples:
            for i, v in enumerate(s[:8]):
                self._emg_plots[i].push([v])

    def _on_imu(self, imu):
        gyro, accel= imu
        self._gyro_plot.push(gyro)
        self._accel_plot.push(accel)

    def _on_prediction(self, gesture: str, confidence: float):
        self._pred_widget.update_prediction(gesture, confidence)