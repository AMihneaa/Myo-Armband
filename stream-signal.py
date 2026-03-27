import asyncio
import logging

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

addr = "E4:96:A9:A7:5C:74"
logger = logging.getLogger(__name__)

class TestClient(MyoClient):
    async def on_classifier_event(self, ce: ClassifierEvent):
        pass

    async def on_aggregated_data(self, ad: AggregatedData):
        '''
        0-7:  FVData
        8-11: oRIENTATION qUATERNION
        12-14: Accelerometru
        15-17: Gyro
        '''
        print("AGG:", ad)

    async def on_emg_data_aggregated(self, eds: EMGDataSingle):
        print("EMG single:", eds)

    async def on_emg_data(self, data: EMGData):
        print("EMG:", data)

    async def on_motion_event(self, me: MotionEvent):
        print("MOTION:", me)

    async def on_fv_data(self, fvd: FVData):
        print("FV:", fvd)

    async def on_imu_data(self, imu: IMUData):
        print("IMU:", imu)

async def main(address):
    client = await TestClient.with_device(address, aggregate_all=False, aggregate_emg= True)

    try:
        info = await client.get_services()
        logger.info(info)

        await client.setup(
            classifier_mode=ClassifierMode.DISABLED,
            emg_mode=EMGMode.SEND_RAW,
            imu_mode=IMUMode.SEND_ALL,
        )

        await client.start()
        await asyncio.sleep(8)
        await client.stop()
        await client.disconnect()

    except Exception as e:
        logger.exception("Error while connecting/streaming: %s", e)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(addr))