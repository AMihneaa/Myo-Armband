import sys
import qasync
import asyncio
import argparse

from PySide6.QtWidgets import QApplication

from config import (
    BLE_ADDR,
    TRIAL_SAMPLES,
    N_CHANNELS
)
from acquisition.client import MyoStreamClient
from recording.trial import TrialRecorder
from storage.writer import TrialWriter
from recording.session import SessionRecorder
from ui.monitor import MainWindow

from myo.types import (
    ClassifierMode,
    EMGMode,
    IMUMode,
)

async def main(app, args):
    app_close_event = asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    client = await MyoStreamClient.with_device(
        BLE_ADDR,
        aggregate_all=False,
        aggregate_emg=False
    )

    trial= TrialRecorder(
        n_samples= TRIAL_SAMPLES,
        n_channels= N_CHANNELS
    )

    writer= TrialWriter(
        out_dir=args.out_dir,
        participant_id=args.p_id,
        session_id=args.s_id,
    )

    session= SessionRecorder(
        client= client,
        trial_recorder= trial,
        writer= writer,
        gesture_ids= args.gesture_ids,
        n_trials= args.trials,
    )

    window = MainWindow(
        client= client,
        session_recorder= session,
        refresh_ms= 50,
    )
    window.show()

    try:
        await client.setup(
            classifier_mode= ClassifierMode.DISABLED,
            emg_mode= EMGMode.SEND_RAW,
            imu_mode= IMUMode.SEND_ALL,
        )
        await client.start()

        asyncio.ensure_future(session.run())
        
        await app_close_event.wait()
    except Exception as e:
        print(f'Error: {e}')
    finally:
        await client.stop()
        await client.disconnect()




if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument('--out_dir', type= str, default='out_dir')
    parser.add_argument('--p_id', type=int, default=1)
    parser.add_argument('--s_id', type=int, default=1)
    parser.add_argument('--gesture_ids', type=int, nargs='+', default=[-1])
    parser.add_argument('--trials', type=int, default=1)
    args= parser.parse_args()


    app = QApplication(sys.argv)
    qasync.run(main(app, args))