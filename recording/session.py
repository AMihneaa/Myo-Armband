import asyncio

from config import REST_DURATION_S, PREPARE_DURATION_S

from acquisition.client import MyoStreamClient
from recording.trial import TrialRecorder
from storage.writer import TrialWriter

class SessionRecorder:
    def __init__(
        self, 
        client: MyoStreamClient,
        trial_recorder: TrialRecorder,
        writer: TrialWriter,
        gesture_ids: list[int],
        n_trials: int,
    ) -> None:
        self._client= client
        self._trial_recorder= trial_recorder
        self._writer= writer
        self._gesture_ids= gesture_ids
        self._n_trials= n_trials

        self.cue: str = "IDLE"

        self._client.set_emg_callback(self._trial_recorder.on_emg_sample)

    async def run(self) -> None:
        for gesture_id in self._gesture_ids:
            for trial_num in range(1, self._n_trials + 1):

                self.cue= f'Gesture {gesture_id} | Trial {trial_num} | Get ready...'
                await asyncio.sleep(PREPARE_DURATION_S)

                self.cue= f'Recording...'
                self._trial_recorder.start()

                while not self._trial_recorder.is_complete():
                    await asyncio.sleep(0.01)

                array= self._trial_recorder.get_array()
                self._writer.save(array, gesture_id, trial_num)
                self.cue= f'Saved. Resting...'

                self._trial_recorder.reset()
                await asyncio.sleep(REST_DURATION_S)

        self.cue= f'Session complete.'

