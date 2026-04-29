import sys
import asyncio
import numpy as np
import qasync

from collections import deque
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QPainterPath

from config import BLE_ADDR
from acquisition.client import MyoStreamClient
from inference.live_pipeline import LiveInferencePipeline

from myo.types import ClassifierMode, EMGMode, IMUMode

MODEL_PATH= "models/xgb_gpu_search_200hz_g1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17_trainSession1_valSession2_testSession3_20260326_160454_trainval_final.pkl"

HISTORY= 200
EMG_COLORS= [
    "#00FF9C", "#00D4FF", "#FF6B35", "#FFD700",
    "#FF3CAC", "#7B2FFF", "#00FF85", "#FF8C00",
]
GYRO_COLORS=  ["#FF4444", "#44FF44", "#4444FF"]
ACCEL_COLORS= ["#FF8844", "#44FF88", "#8844FF"]

GESTURE_NAMES= {
    1:  "Mana Relaxata",
    2:  "Pumn inchis",
    5:  "Palma Deschisa",
    7:  "Degetul Mare Sus",
    9:  "Semnul OK",
}
 

class SignalBridge(QObject):
    emg_ready=        Signal(object)
    imu_ready=        Signal(object)
    prediction_ready= Signal(int, float)

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
        zero_y= int(h * 0.5)
        p.drawLine(0, zero_y, w, zero_y)

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
        font= QFont("Courier New", 8)
        p.setFont(font)
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
            font= QFont("Courier New", 11)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignCenter, "WAITING\nFOR SIGNAL")
            p.end()
            return

        name= GESTURE_NAMES.get(self._gesture, f"Gesture {self._gesture}")

        p.setPen(QColor("#00FF9C"))
        num_font= QFont("Courier New", 36, QFont.Bold)
        p.setFont(num_font)
        p.drawText(0, 10, w, int(h * 0.45), Qt.AlignCenter, str(self._gesture))

        p.setPen(QColor("#AAAACC"))
        name_font= QFont("Courier New", 10)
        p.setFont(name_font)
        p.drawText(0, int(h * 0.48), w, 24, Qt.AlignCenter, name)

        p.setPen(QColor("#555577"))
        label_font= QFont("Courier New", 8)
        p.setFont(label_font)
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
                n_traces= 1,
                colors= [EMG_COLORS[i]],
                label= f"EMG CH{i}",
                y_range= (-128, 128),
            )
            emg_grid.addWidget(plot, i // 4, i % 4)
            self._emg_plots.append(plot)
        root.addLayout(emg_grid, stretch=3)

        bottom= QHBoxLayout()
        bottom.setSpacing(6)

        self._gyro_plot= OscilloscopeWidget(
            n_traces= 3,
            colors= GYRO_COLORS,
            label= "GYROSCOPE  X/Y/Z",
            y_range= (-2000, 2000),
        )
        bottom.addWidget(self._gyro_plot, stretch=2)

        self._accel_plot= OscilloscopeWidget(
            n_traces= 3,
            colors= ACCEL_COLORS,
            label= "ACCELEROMETER  X/Y/Z",
            y_range= (-4, 4),
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

    def _on_prediction(self, gesture, confidence):
        self._pred_widget.update_prediction(gesture, confidence)


class DetectionPipeline(LiveInferencePipeline):
    def __init__(self, bridge: SignalBridge, **kwargs):
        super().__init__(**kwargs)
        self._bridge= bridge

    def _on_prediction(self, gesture, confidence):
        self._bridge.prediction_ready.emit(gesture, confidence)


async def main(app):
    app_close_event= asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    bridge= SignalBridge()

    pipeline= DetectionPipeline(
        bridge= bridge,
        payload_path= MODEL_PATH,
        fs= 200.0,
        bp_low= 10.0,
        bp_high= 95.0,
        env_cutoff= 10.0,
        notch_freq= 50.0,
        notch_q= 30.0,
        apply_notch= True,
        n_channels= 8,
        win= 50,
        hop= 25,
        bp_bands= [(10.0, 25.0), (25.0, 40.0), (40.0, 65.0), (65.0, 95.0)],
        ievd_L= 25,
        ievd_k= 10,
        smoother_n= 5,
    )

    client= await MyoStreamClient.with_device(
        BLE_ADDR,
        aggregate_all= False,
        aggregate_emg= False,
    )

    def emg_callback(sample):
        chunk= np.array([sample.channels], dtype=np.float32)
        pipeline.on_emg(chunk)
        bridge.emg_ready.emit([sample.channels])

    client.set_emg_callback(emg_callback)

    window= LiveDetectWindow(bridge)
    window.show()

    try:
        await client.setup(
            classifier_mode= ClassifierMode.DISABLED,
            emg_mode= EMGMode.SEND_RAW,
            imu_mode= IMUMode.SEND_ALL,
        )
        await client.start()
        await app_close_event.wait()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.stop()
        await client.disconnect()


if __name__ == "__main__":
    app= QApplication(sys.argv)
    qasync.run(main(app))