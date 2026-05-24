from PySide6.QtCore import Signal, QObject


class SignalBridge(QObject):
    emg_ready= Signal(object)
    imu_ready= Signal(object)
    prediction_ready= Signal(str, float)