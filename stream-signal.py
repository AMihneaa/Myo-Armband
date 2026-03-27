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
    MotionEvent
)

addr = "E4:96:A9:A7:5C:74"

logger= logging.getLogger(__name__)

class TestClient(MyoClient):
    async def on_classifier_event(self, ce: ClassifierEvent):
        return await super().on_classifier_event(ce)
    
    async def on_aggregated_data(self, ad: AggregatedData):
        return await super().on_aggregated_data(ad)

    async def on_emg_data_aggregated(self, eds: EMGDataSingle):
        return await super().on_emg_data_aggregated(eds)
    
    async def on_emg_data(self, data: EMGData):
        return await super().on_data(data)
    
    async def on_motion_event(self, me: MotionEvent):
        return await super().on_motion_event(me)
    
    async def on_fv_data(self, fvd: FVData):
        return await super().on_fv_data(fvd)
    
    async def on_imu_data(self, imu: IMUData):
        return await super().on_imu_data(imu)

async def main(address):
    client= await TestClient.with_device(addr, aggregate_all= True)

    try:
        info= await client.get_services()
        logger.info(info)


    except Exception as e:
        print(f'Error: {e}')


if __name__ == "__main__":
    logging.basicConfig(level= logging.INFO)
    logger.info(f' -- Log --')

    asyncio.run(main(addr))