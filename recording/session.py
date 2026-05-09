import asyncio

from dataclasses import dataclass

from config import (
    PREPARE_DURATION_S,
    REST_BETWEEN_TRIALS_S,
    REST_BETWEEN_GESTURES_S,
    FAMILIARIZATION_REPS,
    FAMILIARIZATION_DURATION_S,
    FAMILIARIZATION_REST_S,
    GESTURE_LABELS,
)

from acquisition.client import MyoStreamClient
from recording.trial import TrialRecorder
from storage.writer import TrialWriter


@dataclass
class SessionCue:
    color: str
    text: str
    timer: float= 0.0


class SessionRecorder:
    def __init__(
        self,
        client: MyoStreamClient,
        trial_recorder: TrialRecorder,
        writer: TrialWriter,
        gesture_ids: list[int],
        n_trials: int,
        start_trial: int= 1,
    ) -> None:
        self._client= client
        self._trial_recorder= trial_recorder
        self._writer= writer
        self._gesture_ids= gesture_ids
        self._n_trials= n_trials

        self.cue= SessionCue(color="gray", text="IDLE")

        self._client.set_emg_callback(self._trial_recorder.on_emg_sample)
        self._client.set_imu_callback(self._trial_recorder.on_imu_sample)

        self._start_trial= start_trial


    async def _rest(self, duration: float, text: str) -> None:
        elapsed= 0.0
        step= 0.1

        while elapsed < duration:
            remaining= duration - elapsed
            self.cue= SessionCue(color="orange", text=text, timer=remaining)
            await asyncio.sleep(step)
            elapsed += step

    async def _familiarization(self, gesture_id: int) -> None:
        gesture_name= GESTURE_LABELS[gesture_id]

        for rep in range(1, FAMILIARIZATION_REPS + 1):
            self.cue= SessionCue(
                color="white",
                text=f"Familiarization | {gesture_name} | Rep {rep}/{FAMILIARIZATION_REPS}",
            )
            await asyncio.sleep(PREPARE_DURATION_S)

            self.cue= SessionCue(color="green", text=f"GO | {gesture_name}")
            await asyncio.sleep(FAMILIARIZATION_DURATION_S)

            if rep < FAMILIARIZATION_REPS:
                await self._rest(FAMILIARIZATION_REST_S, f"Rest | Next rep {rep + 1}/{FAMILIARIZATION_REPS}")

    async def run(self) -> None:
        for gesture_idx, gesture_id in enumerate(self._gesture_ids):
            gesture_name= GESTURE_LABELS[gesture_id]

            await self._familiarization(gesture_id)

            for trial_num in range(self._start_trial, self._start_trial + self._n_trials):
                self.cue= SessionCue(
                    color="white",
                    text=f"{gesture_name} | Trial {trial_num}/{self._start_trial + self._n_trials - 1} | Prepare",
                )
                await asyncio.sleep(PREPARE_DURATION_S)

                self.cue= SessionCue(color="green", text=f"GO | {gesture_name} | Trial {trial_num}")
                self._trial_recorder.start()

                while not self._trial_recorder.is_complete():
                    await asyncio.sleep(0.01)

                emg= self._trial_recorder.get_emg_array()
                imu= self._trial_recorder.get_imu_array()
                self._writer.save(emg, imu, gesture_id, trial_num)

                self._trial_recorder.reset()

                if trial_num < self._start_trial + self._n_trials - 1:
                    await self._rest(REST_BETWEEN_TRIALS_S, f"Rest | {gesture_name} | Next trial {trial_num + 1}")

            if gesture_idx < len(self._gesture_ids) - 1:
                next_gesture= GESTURE_LABELS[self._gesture_ids[gesture_idx + 1]]
                await self._rest(REST_BETWEEN_GESTURES_S, f"Rest | Next gesture: {next_gesture}")

        self._writer.close()
        self.cue= SessionCue(color="gray", text="Session complete")