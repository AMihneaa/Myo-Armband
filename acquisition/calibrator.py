from dataclasses import dataclass


@dataclass(slots=True)
class GyroBias:
    x: float
    y: float
    z: float


class GyroscopeCalibrator:

    def __init__(self, n_samples: int) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")

        self._n_samples= n_samples
        self._samples: list[tuple]= []
        self._bias: GyroBias | None= None

    @property
    def is_calibrating(self) -> bool:
        return self._bias is None and len(self._samples) < self._n_samples

    @property
    def is_calibrated(self) -> bool:
        return self._bias is not None

    @property
    def bias(self) -> GyroBias | None:
        return self._bias

    @property
    def progress(self) -> float:
        if self.is_calibrated:
            return 1.0
        return len(self._samples) / self._n_samples

    def feed(self, gyroscope: tuple) -> None:
        if self.is_calibrated:
            return
        self._samples.append(gyroscope)
        if len(self._samples) == self._n_samples:
            self._compute_bias()

    def correct(self, gyroscope: tuple) -> tuple:
        if not self.is_calibrated:
            raise RuntimeError("correct() called before calibration is complete")
        x, y, z= gyroscope
        return (
            x - self._bias.x,
            y - self._bias.y,
            z - self._bias.z,
        )

    def _compute_bias(self) -> None:
        n= len(self._samples)
        self._bias= GyroBias(
            x=sum(s[0] for s in self._samples) / n,
            y=sum(s[1] for s in self._samples) / n,
            z=sum(s[2] for s in self._samples) / n,
        )
        self._samples.clear()
