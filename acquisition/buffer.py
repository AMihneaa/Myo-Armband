from dataclasses import dataclass
from collections import deque
from typing import Optional

@dataclass(slots= True)
class EMGSample:
    timestamp: float
    seq: int
    channels: tuple

@dataclass(slots= True)
class IMUSample:
    timestamp: float
    orientation: tuple
    accelerometer: tuple
    gyroscope: tuple

class CircularBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f'Capacity must be > 0, {capacity}')
        self._buffer: deque= deque(maxlen= capacity)
        self.capacity= capacity

    def append(self, value) -> None:
        self._buffer.append(value)

    def snapshot(self) -> list:
        return list(self._buffer)
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.capacity
    
