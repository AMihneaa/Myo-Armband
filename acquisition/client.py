import time

from typing import Optional

from config import (
    TRIAL_SAMPLES,
    IMU_TRIAL_SAMPLES,
)

from .buffer import (
    CircularBuffer,
    EMGSample,
    IMUSample,
)

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
            aggregate_all= False,
            aggregate_emg= False,
            emg_buffer_capacity: int= TRIAL_SAMPLES,
            imu_buffer_capacity: int= IMU_TRIAL_SAMPLES,
    ):
        super().__init__(aggregate_all= aggregate_all, aggregate_emg= aggregate_emg)
        
        self._emg_seq: int= 0
        self._emg_callback: Optional[callable]= None
        self._emg_buffer: CircularBuffer= CircularBuffer(capacity= emg_buffer_capacity)
        
        self._imu_buffer: CircularBuffer= CircularBuffer(capacity= imu_buffer_capacity)

    async def on_classifier_event(self, ce: ClassifierEvent):
        pass

    async def on_aggregated_data(self, ad: AggregatedData):
        pass

    async def on_emg_data_aggregated(self, eds):
        return await super().on_emg_data_aggregated(eds)

    async def on_emg_data(self, data: EMGData):
        t= time.monotonic()

        sample1= EMGSample(
                    timestamp= t,
                    seq= self._emg_seq,
                    channels= tuple(data.sample1)
                )
        self._emg_buffer.append(
            sample1
        )

        sample2= EMGSample(
                    timestamp= t,
                    seq= self._emg_seq,
                    channels= tuple(data.sample2)
            )
        self._emg_buffer.append(
            sample2
        )

        if self._emg_callback is not None:
            self._emg_callback(sample1)
            self._emg_callback(sample2)

        self._emg_seq+= 1

    async def on_imu_data(self, imu: IMUData):
        t= time.monotonic()

        self._imu_buffer.append(
            IMUSample(
                timestamp= t,
                orientation= (imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z),
                accelerometer= tuple(imu.accelerometer),
                gyroscope= tuple(imu.gyroscope)
            )
        )

    async def on_motion_event(self, me: MotionEvent):
        pass

    async def on_fv_data(self, fvd: FVData):
        pass

    def set_emg_callback(self, fn: callable) -> None:
        self._emg_callback= fn

    @property
    def emg_buffer(self) -> CircularBuffer:
        return self._emg_buffer
    
    @property
    def imu_buffer(self) -> CircularBuffer:
        return self._imu_buffer