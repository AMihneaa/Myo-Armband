import numpy as np

from collections import deque
from enum import Enum

from myo import EMGSample, IMUSample


class TrialState(Enum):
    IDLE= 'idle'
    RECORDING= 'recording'
    COMPLETE= 'complete'


class TrialRecorder:
    def __init__(
            self,
            n_emg_samples: int,
            n_imu_samples: int,
            n_channels: int,
    ) -> None:
        self._n_emg_samples= n_emg_samples
        self._n_imu_samples= n_imu_samples
        self._n_channels= n_channels

        self._emg_buf= deque(maxlen=n_emg_samples)
        self._imu_buf= deque(maxlen=n_imu_samples)

        self._state= TrialState.IDLE

    def start(self) -> None:
        self._emg_buf.clear()
        self._imu_buf.clear()
        self._state= TrialState.RECORDING

    def on_emg_sample(self, sample: EMGSample) -> None:
        if self._state != TrialState.RECORDING:
            return

        self._emg_buf.append(sample)

        if len(self._emg_buf) == self._n_emg_samples:
            self._state= TrialState.COMPLETE

    def on_imu_sample(self, sample: IMUSample) -> None:
        if self._state != TrialState.RECORDING:
            return

        self._imu_buf.append(sample)

    def is_complete(self) -> bool:
        return self._state == TrialState.COMPLETE

    def get_emg_array(self) -> np.ndarray:
        if not self.is_complete():
            raise RuntimeError(f"Trial not complete. State: {self._state}")

        return np.array([s.channels for s in self._emg_buf], dtype=np.float32)

    def get_imu_array(self) -> np.ndarray:
        if not self.is_complete():
            raise RuntimeError(f"Trial not complete. State: {self._state}")

        return np.array(
            [(*s.accelerometer, *s.gyroscope) for s in self._imu_buf],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self._emg_buf.clear()
        self._imu_buf.clear()
        self._state= TrialState.IDLE

    @property
    def state(self) -> TrialState:
        return self._state