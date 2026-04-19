import numpy as np

from collections import deque
from enum import Enum

from ..acquisition.buffer import(
    EMGSample,
)

class TrialState(Enum):
    IDLE= 'idle'
    RECORDING= 'recording'
    COMPLETE= 'complete'

class TrialRecorder:
    def __init__(
            self,
            n_samples: int,
            n_channels: int
        ) ->  None:
        
        self._n_samples= n_samples
        self._n_channels= n_channels
        self._deq= deque(maxlen= n_samples)

        self._state= TrialState.IDLE

    def start(self) -> None:
        self._deq.clear()
        self._state= TrialState.RECORDING



    def on_emg_sample(self, sample: EMGSample) -> None:
        if self._state != TrialState.RECORDING:
            return
        
        self._deq.append(sample)
        if len(self._deq) == self._n_samples:
            self._state= TrialState.COMPLETE

    def is_complete(self) -> bool:
        return self._state == TrialState.COMPLETE

    def get_array(self) -> np.ndarray:
        if not self.is_complete():
            raise RuntimeError(f'Recording is not complete. Current state {self._state}')
        
        return np.array([s.channels for s in self._deq], dtype= np.float32)

    def reset(self) -> None:
        self._deq.clear()
        self._state= TrialState.IDLE

    @property
    def state(self) -> TrialState:
        return self._state
