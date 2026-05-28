import time
from typing import Optional

from config import (
    TRIAL_SAMPLES,
    IMU_TRIAL_SAMPLES,
    GYRO_CALIBRATION_SAMPLES,
)

from .buffer import (
    CircularBuffer,
    EMGSample,
    IMUSample,
)
from .calibrator import GyroscopeCalibrator

from myo import MyoClient
from myo.types import (
    IMUData,
    EMGData,
    ClassifierEvent,
    AggregatedData,
    MotionEvent,
    FVData,
    EMGDataSingle,
)


class MyoStreamClient(MyoClient):

    def __init__(
            self,
            aggregate_all: bool= False,
            aggregate_emg: bool= False,
            emg_buffer_capacity: int= TRIAL_SAMPLES,
            imu_buffer_capacity: int= IMU_TRIAL_SAMPLES,
    ):
        super().__init__(aggregate_all=aggregate_all, aggregate_emg=aggregate_emg)

        self._emg_seq: int= 0
        self._emg_callback: Optional[callable]= None
        self._emg_buffer: CircularBuffer= CircularBuffer(capacity=emg_buffer_capacity)

        self._imu_buffer: CircularBuffer= CircularBuffer(capacity=imu_buffer_capacity)
        self._imu_callback: Optional[callable]= None
        self._calibration_callback: Optional[callable]= None

        self._gyro_calibrator= GyroscopeCalibrator(n_samples=GYRO_CALIBRATION_SAMPLES)

    @property
    def gyro_calibrator(self) -> GyroscopeCalibrator:
        return self._gyro_calibrator

    async def on_classifier_event(self, ce: ClassifierEvent):
        pass

    async def on_aggregated_data(self, ad: AggregatedData):
        pass

    async def on_emg_data_aggregated(self, eds):
        return await super().on_emg_data_aggregated(eds)

    async def on_emg_data(self, data: EMGData):
        t= time.monotonic()

        sample1= EMGSample(
            timestamp=t,
            seq=self._emg_seq,
            channels=tuple(data.sample1),
        )
        self._emg_buffer.append(sample1)

        sample2= EMGSample(
            timestamp=t,
            seq=self._emg_seq,
            channels=tuple(data.sample2),
        )
        self._emg_buffer.append(sample2)

        if self._emg_callback is not None:
            self._emg_callback(sample1)
            self._emg_callback(sample2)

        self._emg_seq += 1

    async def on_imu_data(self, imu: IMUData):
        t= time.monotonic()

        raw_gyro= tuple(imu.gyroscope)

        if not self._gyro_calibrator.is_calibrated:
            self._gyro_calibrator.feed(raw_gyro)

            if self._gyro_calibrator.is_calibrated:
                if self._calibration_callback is not None:
                    self._calibration_callback(self._gyro_calibrator.bias)
            return

        corrected_gyro= self._gyro_calibrator.correct(raw_gyro)

        sample= IMUSample(
            timestamp=t,
            orientation=(
                imu.orientation.w,
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
            ),
            accelerometer=tuple(imu.accelerometer),
            gyroscope=corrected_gyro,
        )
        self._imu_buffer.append(sample)

        if self._imu_callback is not None:
            self._imu_callback(sample)

    async def on_motion_event(self, me: MotionEvent):
        pass

    async def on_fv_data(self, fvd: FVData):
        pass

    def set_emg_callback(self, fn: callable) -> None:
        self._emg_callback= fn

    def set_imu_callback(self, fn: callable) -> None:
        self._imu_callback= fn

    def set_calibration_callback(self, fn: callable) -> None:
        self._calibration_callback= fn

    @property
    def emg_buffer(self) -> CircularBuffer:
        return self._emg_buffer

    @property
    def imu_buffer(self) -> CircularBuffer:
        return self._imu_buffer