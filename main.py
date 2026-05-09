import sys
import qasync
import asyncio
import argparse

from PySide6.QtWidgets import QApplication

from config import (
    BLE_ADDR,
    TRIAL_SAMPLES,
    IMU_TRIAL_SAMPLES,
    N_CHANNELS,
    N_TRIALS,
    GESTURE_LABELS,
)
from acquisition.client import MyoStreamClient
from recording.trial import TrialRecorder
from storage.writer import TrialWriter
from recording.session import SessionRecorder
from ui.cue_window import CueWindow
from ui.signal_window import SignalWindow

from myo.types import ClassifierMode, EMGMode, IMUMode


async def main(app: QApplication, args: argparse.Namespace) -> None:
    app_close_event= asyncio.Event()
    app.aboutToQuit.connect(app_close_event.set)

    client= await MyoStreamClient.with_device(
        BLE_ADDR,
        aggregate_all=False,
        aggregate_emg=False,
    )

    trial= TrialRecorder(
        n_emg_samples=TRIAL_SAMPLES,
        n_imu_samples=IMU_TRIAL_SAMPLES,
        n_channels=N_CHANNELS,
    )

    writer= TrialWriter(
        subject_id=args.p_id,
        session_id=args.s_id,
    )

    session= SessionRecorder(
        client=client,
        trial_recorder=trial,
        writer=writer,
        gesture_ids=args.gesture_ids,
        n_trials=args.trials,
        start_trial=args.start_trial,
    )

    cue_window= CueWindow(session_recorder=session)
    signal_window= SignalWindow(client=client)

    cue_window.show()
    signal_window.show()

    try:
        await client.setup(
            classifier_mode=ClassifierMode.DISABLED,
            emg_mode=EMGMode.SEND_RAW,
            imu_mode=IMUMode.SEND_ALL,
        )
        await client.start()
        asyncio.ensure_future(session.run())
        await app_close_event.wait()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        writer.close()
        await client.stop()
        await client.disconnect()


if __name__ == "__main__":
    valid_ids= list(GESTURE_LABELS.keys())

    parser= argparse.ArgumentParser()
    parser.add_argument("--p_id", type=int, required=True)
    parser.add_argument("--s_id", type=int, required=True)
    parser.add_argument(
        "--gesture_ids",
        type=int,
        nargs="+",
        default=valid_ids,
        choices=valid_ids,
    )
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--start_trial", type=int, default=1)
    args= parser.parse_args()

    app= QApplication(sys.argv)
    qasync.run(main(app, args))

    